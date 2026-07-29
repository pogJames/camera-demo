#!/usr/bin/env python3
"""Tiny MJPEG server for dev-box testing without the real camera.

Loops a local video file or a folder of JPEGs and serves it as
multipart/x-mixed-replace at  http://localhost:8080/mock.mjpeg  so detect.py
runs completely unchanged (just point STREAM_URL at it).

Usage:
  python3 mock_server.py path/to/video.mp4
  python3 mock_server.py path/to/frames_dir/          # folder of *.jpg
  python3 mock_server.py path/to/single.jpg           # single still, looped

Env: MOCK_PORT (default 8080), MOCK_FPS (default 15).
"""

import glob
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PORT = int(os.getenv("MOCK_PORT", "8080"))
FPS = float(os.getenv("MOCK_FPS", "15"))
SOURCE = sys.argv[1] if len(sys.argv) > 1 else None


def _frames():
    """Infinite generator of JPEG bytes from the configured source."""
    if SOURCE and os.path.isdir(SOURCE):
        files = sorted(glob.glob(os.path.join(SOURCE, "*.jpg")) +
                       glob.glob(os.path.join(SOURCE, "*.jpeg")) +
                       glob.glob(os.path.join(SOURCE, "*.png")))
        if not files:
            raise SystemExit(f"no images in {SOURCE}")
        while True:
            for f in files:
                img = cv2.imread(f)
                if img is not None:
                    ok, jpg = cv2.imencode(".jpg", img)
                    if ok:
                        yield jpg.tobytes()

    elif SOURCE and SOURCE.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(SOURCE)
        if img is None:
            raise SystemExit(f"cannot read {SOURCE}")
        ok, jpg = cv2.imencode(".jpg", img)
        blob = jpg.tobytes()
        while True:
            yield blob

    elif SOURCE:  # treat as a video file
        while True:
            cap = cv2.VideoCapture(SOURCE)
            if not cap.isOpened():
                raise SystemExit(f"cannot open video {SOURCE}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                enc, jpg = cv2.imencode(".jpg", frame)
                if enc:
                    yield jpg.tobytes()
            cap.release()
    else:
        raise SystemExit(__doc__)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def do_GET(self):
        if self.path.rstrip("/") not in ("/mock.mjpeg", "/mock", ""):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        delay = 1.0 / FPS
        try:
            for jpg in _frames():
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected


def main():
    if not SOURCE:
        raise SystemExit(__doc__)
    print(f"[mock] serving {SOURCE} at http://localhost:{PORT}/mock.mjpeg "
          f"@ {FPS} fps")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
