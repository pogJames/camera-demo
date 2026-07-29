"""Frame preprocessing: letterbox resize + quantization. See CLAUDE.md."""

import cv2
import numpy as np


def letterbox(frame_bgr, input_size):
    oh, ow = frame_bgr.shape[:2]
    scale = min(input_size / ow, input_size / oh)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_x = (input_size - nw) // 2
    pad_y = (input_size - nh) // 2
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    meta = {
        "input_size": input_size,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "orig_w": ow,
        "orig_h": oh,
    }
    return canvas_rgb, meta


def to_input_tensor(canvas_rgb, input_detail, mean=127.5, std=127.5):
    dtype = np.dtype(input_detail["dtype"])
    norm = (canvas_rgb.astype(np.float32) - mean) / std

    scale, zero_point = input_detail["quantization"]
    if not scale:
        x = np.clip(np.round(canvas_rgb), *_dtype_range(dtype)).astype(dtype)
    else:
        q = np.round(norm / scale + zero_point)
        x = np.clip(q, *_dtype_range(dtype)).astype(dtype)
    return np.expand_dims(x, axis=0)


def _dtype_range(dtype):
    info = np.iinfo(dtype)
    return info.min, info.max
