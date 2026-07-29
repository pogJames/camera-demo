#!/usr/bin/env python3
"""Standalone dataset-capture app — gesture training images, labeled as you go.

Separate from detect.py on purpose: no model, no inference, just the live camera
feed with capture buttons. Stand in front of the camera holding a pose, click a
label (1-4), and a crop of the ROI lands in data/<label>/. Cropping to the SAME
box here that the classifier will see at inference is the whole point — the
training data matches demo conditions exactly.

Reuses config.py for the camera connection only (STREAM_URL / creds / TLS); the
capture specifics below are hardcoded — edit them here, they're the only knobs.

Run:  python app.py   then open  http://<host>:8001/
"""

import os
import threading
import time
from uuid import uuid4

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import urllib3

import config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --------------------------------------------------------------------------- #
# Capture knobs — the only things to edit
# --------------------------------------------------------------------------- #
LABELS   = ["1", "2", "3", "4"]     # one button + one folder each
DATA_DIR = "data"                    # images go to DATA_DIR/<label>/
# Crop region in pixels of the captured frame, as (x, y, w, h). None -> a
# centered square sized to half the shorter side (fine until you pin an exact
# box for your camera placement).
ROI = None
BURST = 5                            # frames saved per button press
JPEG_QUALITY = 95                    # high — this is training data, not preview
HTTP_PORT = 8001                     # distinct from detect.py's 8000


# --------------------------------------------------------------------------- #
# Latest-frame store (same drop-stale approach as detect.py)
# --------------------------------------------------------------------------- #
class FrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0

    def put(self, frame):
        with self._lock:
            self._frame = frame
            self._seq += 1

    def get(self):
        with self._lock:
            if self._frame is None:
                return None, self._seq
            return self._frame.copy(), self._seq


store = FrameStore()
_stop = threading.Event()
_last_saved = {}   # label -> path of most recent save, for undo


# --------------------------------------------------------------------------- #
# Capture thread — pull MJPEG, keep newest frame
# --------------------------------------------------------------------------- #
def _iter_mjpeg(resp, chunk=8192):
    """Yield complete JPEG blobs by scanning for SOI..EOI markers."""
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
            yield bytes(buf[soi:eoi + 2])
            del buf[:eoi + 2]


def capture_loop():
    auth = HTTPBasicAuth(config.CAM_USER, config.CAM_PASS) if config.CAM_USER else None
    backoff = 1.0
    while not _stop.is_set():
        try:
            print(f"[capture] connecting to {config.STREAM_URL}")
            with requests.get(config.STREAM_URL, stream=True, auth=auth,
                              verify=config.VERIFY_TLS, timeout=(10, 30)) as resp:
                resp.raise_for_status()
                backoff = 1.0
                print("[capture] connected")
                for jpg in _iter_mjpeg(resp):
                    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        store.put(frame)
        except Exception as e:
            if _stop.is_set():
                break
            print(f"[capture] stream error: {e!r}; reconnecting in {backoff:.0f}s")
            _stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)
    print("[capture] stopped")


# --------------------------------------------------------------------------- #
# ROI + saving
# --------------------------------------------------------------------------- #
def roi_box(shape):
    """(x1, y1, x2, y2) of the crop ROI for a frame of this shape, clamped."""
    h, w = shape[:2]
    if ROI:
        x, y, rw, rh = ROI
    else:
        side = int(min(h, w) * 0.5)
        x = (w - side) // 2
        y = (h - side) // 2
        rw = rh = side
    x1 = max(0, min(x, w - 1))
    y1 = max(0, min(y, h - 1))
    x2 = max(x1 + 1, min(x + rw, w))
    y2 = max(y1 + 1, min(y + rh, h))
    return x1, y1, x2, y2


def save_crop(frame, label):
    x1, y1, x2, y2 = roi_box(frame.shape)
    d = os.path.join(DATA_DIR, label)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{uuid4().hex}.jpg")
    cv2.imwrite(path, frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return path


def counts():
    out = {}
    for label in LABELS:
        d = os.path.join(DATA_DIR, label)
        try:
            out[label] = sum(1 for f in os.listdir(d) if f.endswith(".jpg"))
        except FileNotFoundError:
            out[label] = 0
    return out


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

app = FastAPI(title="dataset capture")


def _page():
    btns = "\n".join(
        f'<button onclick="cap(\'{l}\')">{l} <small id="c-{l}">0</small></button>'
        for l in LABELS
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>dataset capture</title>
<style>
 body{{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}}
 h1{{font-size:1rem;padding:.4rem;margin:0}}
 img{{max-width:100%;height:auto}}
 #bar{{display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap;padding:.6rem}}
 button{{font-size:1.4rem;padding:.6rem 1.2rem;border:0;border-radius:.4rem;
        background:#2a6;color:#fff;cursor:pointer}}
 button:active{{background:#1a4}}
 button small{{font-size:.8rem;opacity:.8}}
 #undo{{background:#a44}} #msg{{padding:.3rem;min-height:1.2rem;font-size:.9rem}}
</style></head><body>
<h1>dataset capture &mdash; burst {BURST}/press, ROI = yellow box</h1>
<img src="/stream" alt="stream">
<div id="bar">{btns}
 <button id="undo" onclick="undo()">undo</button></div>
<div id="msg"></div>
<script>
 let lastLabel = null;
 async function refresh() {{
   const c = await (await fetch('/stats')).json();
   for (const k in c) document.getElementById('c-'+k).textContent = c[k];
 }}
 async function cap(label) {{
   lastLabel = label;
   const r = await (await fetch('/capture?label='+label, {{method:'POST'}})).json();
   document.getElementById('msg').textContent = '+'+r.saved+' to '+label;
   refresh();
 }}
 async function undo() {{
   if (!lastLabel) return;
   const r = await (await fetch('/undo?label='+lastLabel, {{method:'POST'}})).json();
   document.getElementById('msg').textContent = r.removed ? 'removed 1 from '+lastLabel : 'nothing to undo';
   refresh();
 }}
 document.addEventListener('keydown', e => {{
   if ({LABELS}.includes(e.key)) cap(e.key);
   else if (e.key === 'Backspace') {{ e.preventDefault(); undo(); }}
 }});
 refresh();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _page()


def _mjpeg():
    params = [cv2.IMWRITE_JPEG_QUALITY, 80]
    last_seq = -1
    while not _stop.is_set():
        frame, seq = store.get()
        if frame is None or seq == last_seq:
            time.sleep(0.005)
            continue
        last_seq = seq
        x1, y1, x2, y2 = roi_box(frame.shape)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        ok, jpg = cv2.imencode(".jpg", frame, params)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


@app.get("/stream")
def stream():
    return StreamingResponse(_mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/capture")
def capture(label: str, n: int = BURST):
    if label not in LABELS:
        return JSONResponse({"error": f"unknown label {label!r}"}, status_code=400)
    # Save n DISTINCT frames — wait for a new seq between saves so a burst isn't
    # n identical copies of one frame. Bounded by a deadline so a stalled stream
    # can't hang the request.
    saved = []
    last_seq = -1
    deadline = time.time() + 5.0
    while len(saved) < n and time.time() < deadline:
        frame, seq = store.get()
        if frame is None or seq == last_seq:
            time.sleep(0.01)
            continue
        last_seq = seq
        saved.append(save_crop(frame, label))
    if saved:
        _last_saved[label] = saved[-1]
    return JSONResponse({"label": label, "saved": len(saved), "counts": counts()})


@app.post("/undo")
def undo(label: str):
    path = _last_saved.pop(label, None)
    removed = False
    if path and os.path.exists(path):
        os.remove(path)
        removed = True
    return JSONResponse({"label": label, "removed": removed, "counts": counts()})


@app.get("/stats")
def stats():
    return JSONResponse(counts())


def main():
    print(f"[capture] labels={LABELS} dir={DATA_DIR}/ burst={BURST} "
          f"roi={'auto-center' if ROI is None else ROI}")
    threading.Thread(target=capture_loop, name="capture", daemon=True).start()
    import uvicorn
    print(f"[http] open http://<host>:{HTTP_PORT}/")
    try:
        uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="warning")
    finally:
        _stop.set()


if __name__ == "__main__":
    main()
