"""Code 128 reader for the closed-box crop (zxing-cpp). See CLAUDE.md."""

import cv2
import numpy as np

try:
    import zxingcpp
except ImportError:
    zxingcpp = None
    print("[barcode] zxingcpp not installed; scanning disabled")


ROI_PAD = 12  # quiet zone around a gradient ROI, which ends at the last bar


def read(frame_bgr, box=None, crop=1.0, scale=3, max_rois=4):
    if zxingcpp is None:
        return None
    gray = _crop(frame_bgr, box, crop)
    if gray is None or gray.size == 0:
        return None
    hit = _decode(gray)
    if hit:
        return hit
    for roi in _rois(gray, max_rois):
        hit = _read_roi(gray, roi, scale)
        if hit:
            return hit
    return None


def _crop(frame_bgr, box, crop):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (0, 0, w, h) if box is None else [int(v) for v in box]
    if crop < 1.0:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half_w, half_h = (x2 - x1) * crop / 2, (y2 - y1) * crop / 2
        x1, x2 = cx - half_w, cx + half_w
        y1, y2 = cy - half_h, cy + half_h
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)


def _decode(gray):
    for r in zxingcpp.read_barcodes(gray, try_rotate=True, try_downscale=True):
        if r.valid and r.text:
            return r.text, r.format.name
    return None


def _rois(gray, max_rois):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
    grad = cv2.convertScaleAbs(cv2.subtract(np.abs(gx), np.abs(gy)))
    grad = cv2.blur(grad, (9, 9))
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=4)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:max_rois * 3]:
        x, y, w, h = cv2.boundingRect(c)
        if w < 30 or h < 6 or not 1.5 <= w / h <= 30:
            continue
        out.append((x, y, w, h))
        if len(out) >= max_rois:
            break
    return out


def _read_roi(gray, roi, scale):
    ih, iw = gray.shape
    x, y, w, h = roi
    x0, y0 = max(0, x - ROI_PAD), max(0, y - ROI_PAD)
    x1, y1 = min(iw, x + w + ROI_PAD), min(ih, y + h + ROI_PAD)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return _decode(up)
