"""Single source of truth for all runtime configuration.

Every setting is hardcoded here — edit this one file to change behavior; no env
vars, no .env. Other modules just `import config` and read `config.X`. The
values below are the dev-box defaults (mock stream, CPU); flip USE_NPU / point
MODEL_PATH + STREAM_URL at the real camera when deploying to the board.
"""

STREAM_URL  = "http://192.168.1.170/streaming/stream3/video.mjpeg"
# The INT8 built-in-postprocess model (NPU-ready, 4 outputs): boxes are decoded
# + NMS'd inside the graph. On the board, point this at the _vela.tflite build.
MODEL_PATH  = "tflite_model/ssd_mobilenet_v2_coco_quant_postprocess_vela.tflite"
LABELS_PATH = "tflite_model/coco_labels_list.txt"

USE_NPU     = False        # True on the board
INFER_EVERY = 3            # run inference every Nth frame

SCORE_THRES = 0.5
MAX_DETS    = 50

CAM_USER    = ""
CAM_PASS    = ""
VERIFY_TLS  = False        # camera uses a self-signed cert

HTTP_PORT   = 8000
JPEG_QUALITY = 80

# Decode incoming camera JPEGs at reduced scale (libjpeg DCT scaling — much
# cheaper than full decode). 1 = full res, 2 = 1/2 each axis (720p -> 640x360),
# 4 = 1/4. The model only needs 300x300 and a LAN <img> looks fine at 360p, so
# on a 720p stream CAPTURE_REDUCE=2 roughly quarters the decode+encode CPU cost.
CAPTURE_REDUCE = 1


def summary():
    """Human-readable dump for startup logging (password redacted)."""
    return (
        f"STREAM_URL={STREAM_URL}\n"
        f"MODEL_PATH={MODEL_PATH}\n"
        f"LABELS_PATH={LABELS_PATH}\n"
        f"USE_NPU={USE_NPU}  INFER_EVERY={INFER_EVERY}  "
        f"SCORE_THRES={SCORE_THRES}  MAX_DETS={MAX_DETS}\n"
        f"CAM_USER={CAM_USER!r}  CAM_PASS={'***' if CAM_PASS else ''!r}  "
        f"VERIFY_TLS={VERIFY_TLS}\n"
        f"HTTP_PORT={HTTP_PORT}  JPEG_QUALITY={JPEG_QUALITY}  "
        f"CAPTURE_REDUCE={CAPTURE_REDUCE}"
    )
