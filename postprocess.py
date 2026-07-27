"""Postprocessing for the built-in-postprocess SSD-MobileNetV2 model.

The model runs TFLite_Detection_PostProcess *inside* the graph, so its 4 outputs
are ALREADY decoded + NMS'd (incl. cross-class NMS):

    boxes    [1, N, 4]   (ymin, xmin, ymax, xmax) normalized
    classes  [1, N]      class ids (0-indexed, background dropped)
    scores   [1, N]      confidences in [0,1], sorted descending
    count    [1]         number of valid detections

So there is no anchor decode, softmax, or manual NMS to do here — this module
just thresholds, offsets the class ids to index the labels file, and maps the
boxes back onto the original full-res frame.
"""

import numpy as np


def load_labels(path):
    """Load coco_labels_list.txt -> list of names, indexable directly by the
    model's class index (line 0 == '???' background)."""
    with open(path) as f:
        return [ln.strip() for ln in f]


def postprocess_builtin(boxes, classes, scores, count, letterbox_meta,
                        score_thres=0.5, max_dets=50, label_offset=1):
    """Read the outputs of a model with a built-in TFLite_Detection_PostProcess op.

    boxes:   [N,4] normalized (ymin, xmin, ymax, xmax) in the letterboxed input.
    classes: [N]   float class ids, 0-indexed over the *real* classes.
    scores:  [N]   confidences in [0,1], sorted descending by the op.
    count:   int   number of valid detections (rest of the arrays are padding).
    label_offset: added to each class id so it indexes directly into the labels
                  file. The op drops the background class, so +1 aligns ids with
                  a labels file whose line 0 is background.

    Returns list of dicts: {class_id, score, box:(x1,y1,x2,y2) in full-res px}.
    """
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
    """Normalized [0,1] coords in the letterboxed input -> full-res pixel box.

    meta carries: input_size (S), scale (orig->resized), pad_x, pad_y, orig_w, orig_h.
    The model input is SxS; normalized coords multiply back to S px, then we
    subtract padding and divide by the resize scale.
    """
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
