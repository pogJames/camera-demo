#!/usr/bin/env python3
"""Smart-camera -> NPU object detection -> LAN browser stream. See CLAUDE.md."""

import json
import os
import sys
import threading
import time
import io

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import urllib3

import config
import preprocess
import postprocess
import controller
import modbus
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

    def put_frame(self, frame):
        with self._lock:
            self._frame = frame
            self._seq += 1
            return self._seq

    def get_frame(self):
        with self._lock:
            if self._frame is None:
                return None, self._seq
            return self._frame.copy(), self._seq

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

    def set_state(self, state):
        with self._lock:
            self._state = state

    def get_state(self):
        with self._lock:
            return self._state


store = FrameStore()
_stop = threading.Event()
_controller = None
_reset_flush = threading.Event()   # /reset asks the control thread to push lamps off now


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
    decode_flag = {1: cv2.IMREAD_COLOR,
                   2: cv2.IMREAD_REDUCED_COLOR_2,
                   4: cv2.IMREAD_REDUCED_COLOR_4,
                   8: cv2.IMREAD_REDUCED_COLOR_8}.get(config.CAPTURE_REDUCE,
                                                      cv2.IMREAD_COLOR)
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
                    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), decode_flag)
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

        boxes = [o for o in self.outs if len(o["shape"]) == 3 and o["shape"][-1] == 4]
        count = [o for o in self.outs if len(o["shape"]) == 1]
        pair = [o for o in self.outs if len(o["shape"]) == 2]
        if not (len(boxes) == 1 and len(count) == 1 and len(pair) == 2):
            raise RuntimeError(
                f"Wrong model; got shapes "
                f"{[list(o['shape']) for o in self.outs]}"
            )
        pair.sort(key=lambda o: o["index"])
        self.box_idx = boxes[0]["index"]
        self.class_idx = pair[0]["index"]
        self.score_idx = pair[1]["index"]
        self.count_idx = count[0]["index"]

    def infer(self, frame_bgr):
        canvas, meta = preprocess.letterbox(frame_bgr, self.input_size)
        x = preprocess.to_input_tensor(canvas, self.inp)
        self.npu.set_tensor(self.inp["index"], x)
        self.npu.invoke()
        return postprocess.postprocess_builtin(
            self.npu.get_tensor(self.box_idx)[0],
            self.npu.get_tensor(self.class_idx)[0],
            self.npu.get_tensor(self.score_idx)[0],
            self.npu.get_tensor(self.count_idx)[0],
            meta, score_thres=config.SCORE_THRES, max_dets=config.MAX_DETS,
        )


def inference_loop(detector):
    last_seq = -1
    counter = 0
    while not _stop.is_set():
        frame, seq = store.get_frame()
        if frame is None or seq == last_seq:
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


# Sequence control (decoupled from inference; owns the Modbus lamps)
def control_loop(ctrl, lamps):
    last_seq = -1
    while not _stop.is_set():
        dets, seq = store.get_detections_seq()
        flush = _reset_flush.is_set()
        if seq == last_seq and not flush:
            _stop.wait(0.005)
            continue
        last_seq = seq
        if flush:
            # ctrl was already reset by the HTTP handler; just push the off state,
            # so the lamps clear even if the detection stream has stalled.
            _reset_flush.clear()
            state, regs = ctrl.snapshot()
        else:
            labels = {_labels[d["class_id"]] for d in dets
                      if d["class_id"] < len(_labels)}
            state, regs = ctrl.update(labels)
        store.set_state(state)
        lamps.apply(regs)
    print("[control] stopped")


# Annotation
_COLORS = {}


def _color(cid):
    if cid not in _COLORS:
        rng = np.random.RandomState(cid * 7 + 1)
        _COLORS[cid] = tuple(int(c) for c in rng.randint(60, 256, size=3))
    return _COLORS[cid]


def annotate(frame, dets, labels):
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        col = _color(d["class_id"])
        name = labels[d["class_id"]] if d["class_id"] < len(labels) else str(d["class_id"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        text = f"{name} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), col, -1)
        cv2.putText(frame, text, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


# HTTP server (FastAPI)
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="NPU camera detection")
_labels = []

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


def _mjpeg_generator():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
    last_seq = -1
    while not _stop.is_set():
        frame, seq = store.get_frame()
        if frame is None or seq == last_seq:
            time.sleep(0.005)
            continue
        last_seq = seq
        annotate(frame, store.get_detections(), _labels)
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
        name = _labels[d["class_id"]] if d["class_id"] < len(_labels) else str(d["class_id"])
        out.append({"label": name, "class_id": d["class_id"],
                    "score": round(d["score"], 4), "box": list(d["box"])})
    return JSONResponse(out)


@app.get("/state")
def state():
    return JSONResponse(store.get_state() or {})


@app.get("/events")
def events():
    def gen():
        last = None
        while not _stop.is_set():
            s = store.get_state()
            if s is not None:
                msg = json.dumps(s, separators=(",", ":"))
                if msg != last:
                    last = msg
                    yield f"data: {msg}\n\n"
            time.sleep(0.1)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/reset")
def reset():
    if _controller is not None:
        _controller.reset()
        st, _ = _controller.snapshot()
        store.set_state(st)
        _reset_flush.set()   # control thread turns the lamps off on its next tick
    return JSONResponse({"ok": True})


# Entrypoint
def main():
    global _labels, _controller
    print("[config]\n" + config.summary())
    detector = Detector()
    _labels = detector.labels

    _controller = controller.DemoController(
        config.DEMO_STEPS, config.CONFIRM_FRAMES, config.AUTO_RESET_SECS,
        config.STEP_REGS, config.FAULT_REG, config.DONE_REG)
    lamps = modbus.LampBank(
        config.MODBUS_ENABLE, config.MODBUS_PORT, config.MODBUS_BAUD,
        config.MODBUS_PARITY, config.MODBUS_STOPBITS, config.MODBUS_BYTESIZE,
        config.MODBUS_SLAVE, config.MODBUS_REFRESH_SECS)
    st0, _ = _controller.snapshot()
    store.set_state(st0)

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
