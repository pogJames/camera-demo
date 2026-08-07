#!/usr/bin/env python3
"""Smart-camera -> NPU object detection -> LAN browser stream. See CLAUDE.md."""

import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import urllib3

import config
import preprocess
import postprocess
import controller
import spatial
import modbus
import camera
import interpreter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._detections = []
        self._det_seq = 0
        self._state = None
        self._state_seq = 0
        self._state_cond = threading.Condition(self._lock)
        self._proofs = {}

    def put_frame(self, frame):
        frame.flags.writeable = False  # consumers share this buffer; see CLAUDE.md
        with self._lock:
            self._frame = frame
            self._seq += 1
            return self._seq

    def get_frame(self, since=None):
        with self._lock:
            if self._frame is None or self._seq == since:
                return None, self._seq
            return self._frame, self._seq

    def set_detections(self, dets):
        with self._lock:
            self._detections = dets
            self._det_seq += 1

    def get_detections(self):
        with self._lock:
            return list(self._detections)

    def get_detections_seq(self):
        with self._lock:
            return list(self._detections), self._det_seq

    def set_proof(self, idx, name):
        with self._lock:
            self._proofs[idx] = name

    def get_proof(self, idx):
        with self._lock:
            return self._proofs.get(idx)

    def clear_proofs(self):
        with self._lock:
            self._proofs.clear()

    def set_state(self, state):
        with self._lock:
            if state == self._state:
                return
            self._state = state
            self._state_seq += 1
            self._state_cond.notify_all()

    def get_state(self):
        with self._lock:
            return self._state

    def wait_state(self, last_seq, timeout):
        with self._state_cond:
            if self._state_seq == last_seq:
                self._state_cond.wait(timeout)
            return self._state, self._state_seq


store = FrameStore()
_stop = threading.Event()
_controller = None
_reset_flush = threading.Event()


# Capture
def _iter_mjpeg(resp, chunk=8192):
    buf = bytearray()
    for data in resp.iter_content(chunk_size=chunk):
        if _stop.is_set():
            return
        if not data:
            continue
        buf.extend(data)
        while True:
            soi = buf.find(b"\xff\xd8")
            if soi < 0:
                if len(buf) > chunk:
                    del buf[:-2]
                break
            eoi = buf.find(b"\xff\xd9", soi + 2)
            if eoi < 0:
                if soi > 0:
                    del buf[:soi]
                break
            jpg = bytes(buf[soi:eoi + 2])
            del buf[:eoi + 2]
            yield jpg


def capture_loop():
    auth = HTTPBasicAuth(config.CAM_USER, config.CAM_PASS) if config.CAM_USER else None
    backoff = 1.0
    while not _stop.is_set():
        try:
            print(f"[capture] connecting to {config.STREAM_URL}")
            with requests.get(
                config.STREAM_URL, stream=True, auth=auth,
                verify=config.VERIFY_TLS, timeout=(10, 30),
            ) as resp:
                resp.raise_for_status()
                backoff = 1.0
                print("[capture] connected")
                for jpg in _iter_mjpeg(resp):
                    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        store.put_frame(frame)
        except Exception as e:
            if _stop.is_set():
                break
            print(f"[capture] stream error: {e!r}; reconnecting in {backoff:.0f}s")
            _stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)
    print("[capture] stopped")


# Inference
class Detector:
    def __init__(self):
        self.npu = interpreter.get_npu(config.MODEL_PATH, config.USE_NPU)
        self.inp = self.npu.get_input_details()[0]
        self.outs = self.npu.get_output_details()
        self.input_size = int(self.inp["shape"][1])
        self.labels = postprocess.load_labels(config.LABELS_PATH)

        self.levels = {}
        for o in self.outs:
            if len(o["shape"]) != 4:
                continue
            grid, ch = int(o["shape"][1]), int(o["shape"][-1])
            kind = "cls" if ch == len(self.labels) else "box"
            self.levels.setdefault(grid, {})[kind] = o
        bad = [g for g, p in self.levels.items() if set(p) != {"box", "cls"}]
        if not self.levels or bad:
            raise RuntimeError(
                f"Wrong model; got shapes {[list(o['shape']) for o in self.outs]}"
            )

    def infer(self, frame_bgr):
        canvas, meta = preprocess.letterbox(frame_bgr, self.input_size)
        x = preprocess.to_input_tensor(canvas, self.inp)
        self.npu.set_tensor(self.inp["index"], x)
        self.npu.invoke()
        levels = {g: (self._dequant(p["box"]), self._dequant(p["cls"]))
                  for g, p in self.levels.items()}
        dets = postprocess.postprocess_yolo(
            levels, meta, score_thres=config.SCORE_THRES,
            iou_thres=config.NMS_IOU, max_dets=config.MAX_DETS,
        )
        return spatial.mark(dets, self.labels, config.DEMO_STEPS, config.EXCLUSIVE_IOU)

    def _dequant(self, detail):
        raw = self.npu.get_tensor(detail["index"])[0].astype(np.float32)
        scale, zero_point = detail["quantization"]
        return (raw - zero_point) * scale if scale else raw


def inference_loop(detector):
    last_seq = -1
    counter = 0
    while not _stop.is_set():
        frame, seq = store.get_frame(last_seq)
        if frame is None:
            _stop.wait(0.005)
            continue
        last_seq = seq
        counter += 1
        if counter % config.INFER_EVERY != 0:
            continue
        t0 = time.time()
        try:
            dets = detector.infer(frame)
        except Exception as e:
            print(f"[infer] error: {e!r}")
            continue
        store.set_detections(dets)
        dt = (time.time() - t0) * 1000
        if counter % (config.INFER_EVERY * 30) == 0:
            print(f"[infer] {len(dets)} dets, {dt:.1f}ms")
    print("[infer] stopped")


# Sequence control
def control_loop(ctrl, lamps):
    last_seq = -1
    last_current = 0
    while not _stop.is_set():
        dets, seq = store.get_detections_seq()
        flush = _reset_flush.is_set()
        if seq == last_seq and not flush:
            _stop.wait(0.005)
            continue
        last_seq = seq
        if flush:
            _reset_flush.clear()
            state, regs = ctrl.snapshot()
        else:
            state, regs = ctrl.update(spatial.present(dets), spatial.visible(dets))
        store.set_state(state)
        lamps.apply(regs)

        cur = state["current"]
        if cur > last_current:
            for i in range(last_current, cur):
                _trigger_recording(i)
        elif cur < last_current:
            store.clear_proofs()
        last_current = cur
    print("[control] stopped")


def _trigger_recording(idx):
    if not config.RECORD_ENABLE:
        return
    try:
        name = camera.trigger(f"step{idx + 1}")
    except Exception as e:
        print(f"[camera] trigger step{idx + 1} failed: {e!r}")
        return
    if name:
        store.set_proof(idx, name)
        print(f"[camera] step{idx + 1} recording: {name}")


# Annotation
_COLORS = {}


def _color(cid):
    if cid not in _COLORS:
        rng = np.random.RandomState(cid * 7 + 1)
        _COLORS[cid] = tuple(int(c) for c in rng.randint(60, 256, size=3))
    return _COLORS[cid]


def annotate(frame, dets, scale=1.0):
    for d in dets:
        x1, y1, x2, y2 = (int(v * scale) for v in d["box"])
        col = _color(d["class_id"])
        inside = d.get("inside", True)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        where = f" in {inside}" if isinstance(inside, str) else ("" if inside else " outside")
        text = f"{d['label']}{where} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + tw + 2, y1 + th + 6), col, -1)
        cv2.putText(frame, text, (x1 + 1, y1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


# HTTP server (FastAPI)
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="NPU camera detection")

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


def _mjpeg_generator():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
    last_seq = -1
    while not _stop.is_set():
        frame, seq = store.get_frame(last_seq)
        if frame is None:
            time.sleep(0.005)
            continue
        last_seq = seq
        scale = config.PREVIEW_SCALE
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            frame = frame.copy()
        annotate(frame, store.get_detections(), scale)
        ok, jpg = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + jpg.tobytes() + b"\r\n")


@app.get("/stream")
def stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/detections")
def detections():
    out = []
    for d in store.get_detections():
        out.append({"label": d["label"], "class_id": d["class_id"],
                    "inside": d["inside"], "score": round(d["score"], 4),
                    "box": list(d["box"])})
    return JSONResponse(out)


@app.get("/state")
def state():
    return JSONResponse(store.get_state() or {})


@app.get("/log/{i}")
def log(i: int):
    name = store.get_proof(i)
    if not name:
        return Response(status_code=404)
    try:
        r = camera.open_video(name)
        r.raise_for_status()
    except Exception as e:
        print(f"[camera] fetch {name} failed: {e!r}")
        return Response(status_code=502)

    def body():
        try:
            for chunk in r.iter_content(chunk_size=65536):
                yield chunk
        finally:
            r.close()
    return StreamingResponse(
        body(), media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{name}"'})


@app.get("/events")
def events():
    def gen():
        last = -1
        while not _stop.is_set():
            s, seq = store.wait_state(last, timeout=15.0)
            if seq != last and s is not None:
                last = seq
                yield f"data: {json.dumps(s, separators=(',', ':'))}\n\n"
            else:
                yield ": ping\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/reset")
def reset():
    if _controller is not None:
        _controller.reset()
        st, _ = _controller.snapshot()
        store.set_state(st)
        _reset_flush.set()
    return JSONResponse({"ok": True})


# Entrypoint
def main():
    global _controller
    print("[config]\n" + config.summary())
    detector = Detector()

    _controller = controller.DemoController(
        config.DEMO_STEPS, config.MODBUS_REGISTERS, config.IDLE_STATE,
        config.CONFIRM_FRAMES, config.REGRESS_FRAMES)
    lamps = modbus.LampBank(
        config.MODBUS_ENABLE, config.MODBUS_PORT, config.MODBUS_SLAVE,
        config.MODBUS_REFRESH_SECS)
    st0, _ = _controller.snapshot()
    store.set_state(st0)

    if config.RECORD_ENABLE:
        camera.enable_recording()

    threads = [
        threading.Thread(target=capture_loop, name="capture", daemon=True),
        threading.Thread(target=inference_loop, args=(detector,),
                         name="infer", daemon=True),
        threading.Thread(target=control_loop, args=(_controller, lamps),
                         name="control", daemon=True),
    ]
    for t in threads:
        t.start()

    import uvicorn
    print(f"[http] serving on 0.0.0.0:{config.HTTP_PORT} "
          f"(open http://<host>:{config.HTTP_PORT}/ )")
    try:
        uvicorn.run(app, host="0.0.0.0", port=config.HTTP_PORT, log_level="warning",
                    timeout_graceful_shutdown=1)
    finally:
        _shutdown(threads, lamps)


def _shutdown(threads, lamps):
    _stop.set()
    for t in threads:
        t.join(timeout=2.0)
    try:
        _controller.reset()
        _, off = _controller.snapshot()
        lamps.apply(off)
    except Exception as e:
        print(f"[shutdown] lamp-off failed: {e!r}")
    finally:
        lamps.close()
    print("[shutdown] workers stopped, lamps off, serial closed")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
