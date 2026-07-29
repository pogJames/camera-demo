"""Postprocessing for the built-in-postprocess SSD-MobileNetV2 model. See CLAUDE.md."""

import numpy as np


def load_labels(path):
    with open(path) as f:
        return [ln.strip() for ln in f]


def postprocess_builtin(boxes, classes, scores, count, letterbox_meta,
                        score_thres=0.5, max_dets=50, label_offset=1):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    classes = np.asarray(classes).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n = min(int(count), len(scores))

    detections = []
    for k in range(n):
        if scores[k] < score_thres:
            continue
        ymin, xmin, ymax, xmax = boxes[k]
        x1, y1, x2, y2 = _unletterbox(xmin, ymin, xmax, ymax, letterbox_meta)
        detections.append({
            "class_id": int(classes[k]) + label_offset,
            "score": float(scores[k]),
            "box": (x1, y1, x2, y2),
        })
        if len(detections) >= max_dets:
            break
    return detections


def _unletterbox(xmin, ymin, xmax, ymax, meta):
    S = meta["input_size"]
    scale = meta["scale"]
    pad_x, pad_y = meta["pad_x"], meta["pad_y"]
    ow, oh = meta["orig_w"], meta["orig_h"]

    px1 = (xmin * S - pad_x) / scale
    py1 = (ymin * S - pad_y) / scale
    px2 = (xmax * S - pad_x) / scale
    py2 = (ymax * S - pad_y) / scale

    x1 = int(round(min(max(px1, 0), ow - 1)))
    y1 = int(round(min(max(py1, 0), oh - 1)))
    x2 = int(round(min(max(px2, 0), ow - 1)))
    y2 = int(round(min(max(py2, 0), oh - 1)))
    return x1, y1, x2, y2
