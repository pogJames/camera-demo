"""Geometry gate: an item counts only when it sits inside its container. See CLAUDE.md."""


def mark(dets, labels, containers, exclusive_iou):
    kept = _resolve(dets, exclusive_iou)
    for d in kept:
        d["label"] = labels[d["class_id"]] if d["class_id"] < len(labels) else str(d["class_id"])

    by_label = {}
    for d in kept:
        by_label.setdefault(d["label"], []).append(d)

    for d in kept:
        container = containers.get(d["label"])
        if container is None:
            d["inside"] = True
            continue
        d["inside"] = container if any(contained(d["box"], c["box"]) for c in by_label.get(container, ())) else False
    return kept


def present(dets):
    return {d["label"] for d in dets if d.get("inside")}


def visible(dets):
    return {d["label"] for d in dets}


def contained(item, container):
    return (_area(item) > 0
            and item[0] >= container[0] and item[1] >= container[1]
            and item[2] <= container[2] and item[3] <= container[3])


def _resolve(dets, iou_thres):
    kept = []
    for d in sorted(dets, key=lambda x: -x["score"]):
        if any(k["class_id"] != d["class_id"] and _iou(k["box"], d["box"]) >= iou_thres for k in kept):
            continue
        kept.append(d)
    return kept


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a, b):
    return (max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1])))


def _iou(a, b):
    union = _area(a) + _area(b) - _intersection(a, b)
    return _intersection(a, b) / union if union > 0 else 0.0
