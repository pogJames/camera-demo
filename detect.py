#!/usr/bin/env python3
"""Smart-camera -> NPU object detection -> LAN browser stream.

Pulls an MJPEG stream, runs SSD-MobileNetV2 detection (CPU on the dev box,
Ethos-U NPU on the board), draws boxes, and serves the annotated video as an
MJPEG endpoint any browser can open with a plain <img>.

Threads:
  - capture thread : reads the MJPEG source, keeps only the LATEST frame.
  - inference thread: every INFER_EVERY-th frame, runs the model; detections are
                      cached and reused for in-between frames (rate decoupling).
  - HTTP (uvicorn)  : serves / , /stream , /detections.

Everything platform-specific is an env var (see config.py / .env.example).
"""

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
from interp import make_interpreter

# Camera link is a self-signed isolated device; silence the verify=False warning.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FrameStore:
    """Latest-frame-wins slot shared between capture, inference, and HTTP.

    Never queues a backlog: capture overwrites; consumers read the newest. This
    keeps end-to-end latency bounded even if inference or the network stalls.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None          # latest full-res BGR frame
        self._seq = 0               # increments on every new frame
        self._detections = []       # most recent detections (list of dicts)
        self._det_seq = -1          # frame seq the detections were computed on

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

    def set_detections(self, dets, seq):
        with self._lock:
            self._detections = dets
            self._det_seq = seq

    def get_detections(self):
        with self._lock:
            return list(self._detections)


store = FrameStore()
_stop = threading.Event()


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
def _iter_mjpeg(resp, chunk=8192):
    """Yield complete JPEG byte blobs from a multipart/x-mixed-replace stream.

    Scans the raw byte stream for SOI (0xFFD8) .. EOI (0xFFD9) markers rather
    than trusting the boundary string — robust across camera quirks.
    """
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
                # No start marker yet; keep only a tail to bound memory.
                if len(buf) > chunk:
                    del buf[:-2]
                break
            eoi = buf.find(b"\xff\xd9", soi + 2)
            if eoi < 0:
                if soi > 0:
                    del buf[:soi]   # drop junk before the start marker
                break
            jpg = bytes(buf[soi:eoi + 2])
            del buf[:eoi + 2]
            yield jpg


def capture_loop():
    """Connect to STREAM_URL and push decoded frames into the store, forever.

    Reconnects with exponential backoff on any drop/error.
    """
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


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
class Detector:
    """Wraps the interpreter for the built-in-postprocess (4-output) model.

    The model's TFLite_Detection_PostProcess op emits pre-decoded, NMS'd boxes,
    so inference is just: letterbox -> invoke -> threshold + class-offset + rescale.
    """

    def __init__(self):
        self.interp = make_interpreter(config.MODEL_PATH, config.USE_NPU)
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()
        self.input_size = int(self.inp["shape"][1])
        self.labels = postprocess.load_labels(config.LABELS_PATH)

        # Standard TFLite_Detection_PostProcess outputs: boxes[1,N,4],
        # classes[1,N], scores[1,N], count[1]. Identify by shape; use the tensor
        # index order to break the classes-vs-scores tie (classes precede scores).
        boxes = [o for o in self.outs if len(o["shape"]) == 3 and o["shape"][-1] == 4]
        count = [o for o in self.outs if len(o["shape"]) == 1]
        pair = [o for o in self.outs if len(o["shape"]) == 2]
        if not (len(boxes) == 1 and len(count) == 1 and len(pair) == 2):
            raise RuntimeError(
                f"Expected a 4-output built-in-postprocess model; got shapes "
                f"{[list(o['shape']) for o in self.outs]}"
            )
        pair.sort(key=lambda o: o["index"])   # lower index = classes, higher = scores
        self.box_idx = boxes[0]["index"]
        self.class_idx = pair[0]["index"]
        self.score_idx = pair[1]["index"]
        self.count_idx = count[0]["index"]

    def infer(self, frame_bgr):
        canvas, meta = preprocess.letterbox(frame_bgr, self.input_size)
        x = preprocess.to_input_tensor(canvas, self.inp)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return postprocess.postprocess_builtin(
            self.interp.get_tensor(self.box_idx)[0],
            self.interp.get_tensor(self.class_idx)[0],
            self.interp.get_tensor(self.score_idx)[0],
            self.interp.get_tensor(self.count_idx)[0],
            meta, score_thres=config.SCORE_THRES, max_dets=config.MAX_DETS,
        )


def inference_loop(detector):
    """Run detection on every INFER_EVERY-th new frame; cache the result."""
    last_seq = -1
    counter = 0
    ema_ms = None
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
        store.set_detections(dets, seq)
        dt = (time.time() - t0) * 1000
        ema_ms = dt if ema_ms is None else 0.9 * ema_ms + 0.1 * dt
        if counter % (config.INFER_EVERY * 30) == 0:
            print(f"[infer] {len(dets)} dets, {dt:.1f}ms (ema {ema_ms:.1f}ms)")
    print("[infer] stopped")


# --------------------------------------------------------------------------- #
# Annotation
# --------------------------------------------------------------------------- #
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
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), col, -1)
        cv2.putText(frame, text, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- #
# HTTP server (FastAPI)
# --------------------------------------------------------------------------- #
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

app = FastAPI(title="NPU camera detection")
_labels = []   # populated in main()

_INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>NPU camera detection</title>
<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}
img{max-width:100%;height:auto}h1{font-size:1rem;padding:.5rem;margin:0}</style></head>
<body><h1>NPU object detection &mdash; live</h1>
<img src="/stream" alt="stream"></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _INDEX_HTML


def _mjpeg_generator():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
    last_seq = -1
    while not _stop.is_set():
        frame, seq = store.get_frame()
        # Only annotate+encode when a NEW frame has arrived. Otherwise we'd
        # re-encode the same frame ~100x/s (the loop rate) against a 15 fps
        # source — pure wasted A55 JPEG-encode work. Wait for the next frame.
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
        # No sleep needed here: the seq-gate above already paces us to the
        # source frame rate (one encode per new frame).


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


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main():
    global _labels
    print("[config]\n" + config.summary())
    detector = Detector()
    _labels = detector.labels

    threading.Thread(target=capture_loop, name="capture", daemon=True).start()
    threading.Thread(target=inference_loop, args=(detector,),
                     name="infer", daemon=True).start()

    import uvicorn
    print(f"[http] serving on 0.0.0.0:{config.HTTP_PORT} "
          f"(open http://<host>:{config.HTTP_PORT}/ )")
    try:
        uvicorn.run(app, host="0.0.0.0", port=config.HTTP_PORT, log_level="warning")
    finally:
        _stop.set()


if __name__ == "__main__":
    main()
