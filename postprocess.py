"""Postprocessing for the raw YOLO detection head (DFL boxes + NMS). See CLAUDE.md."""

import cv2
import numpy as np


def load_labels(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def postprocess_yolo(levels, letterbox_meta, score_thres=0.25, iou_thres=0.45,
                     max_dets=50, label_offset=0):
    S = letterbox_meta["input_size"]
    boxes, scores, class_ids = [], [], []

    for grid, (box_logits, cls_logits) in levels.items():
        stride = S / grid
        cls = cls_logits.reshape(-1, cls_logits.shape[-1])
        conf = _sigmoid(cls)
        best = conf.max(axis=1)
        keep = best >= score_thres
        if not keep.any():
            continue

        reg = box_logits.reshape(-1, 4, box_logits.shape[-1] // 4)[keep]
        ltrb = (_softmax(reg) * np.arange(reg.shape[-1], dtype=np.float32)).sum(axis=-1)

        gy, gx = np.divmod(np.nonzero(keep)[0], grid)
        cx, cy = gx + 0.5, gy + 0.5
        boxes.append(np.stack([
            (cx - ltrb[:, 0]) * stride,
            (cy - ltrb[:, 1]) * stride,
            (cx + ltrb[:, 2]) * stride,
            (cy + ltrb[:, 3]) * stride,
        ], axis=1))
        scores.append(best[keep])
        class_ids.append(conf[keep].argmax(axis=1))

    if not boxes:
        return []

    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    class_ids = np.concatenate(class_ids)

    wh = np.stack([boxes[:, 0], boxes[:, 1],
                   boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
    idx = cv2.dnn.NMSBoxes(wh.tolist(), scores.tolist(), score_thres, iou_thres)
    if len(idx) == 0:
        return []
    idx = np.asarray(idx).reshape(-1)
    idx = idx[np.argsort(-scores[idx])][:max_dets]

    detections = []
    for k in idx:
        x1, y1, x2, y2 = _unletterbox(boxes[k, 0] / S, boxes[k, 1] / S,
                                      boxes[k, 2] / S, boxes[k, 3] / S,
                                      letterbox_meta)
        detections.append({
            "class_id": int(class_ids[k]) + label_offset,
            "score": float(scores[k]),
            "box": (x1, y1, x2, y2),
        })
    return detections


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


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
