"""Single source of truth for all runtime configuration. See CLAUDE.md."""

STREAM_URL  = "http://192.168.1.170/streaming/stream3/video.mjpeg"
MODEL_PATH  = "tflite_model/ssd_mobilenet_v2_coco_quant_postprocess_vela.tflite"
LABELS_PATH = "tflite_model/coco_labels_list.txt"

USE_NPU     = False
INFER_EVERY = 3

SCORE_THRES = 0.5
MAX_DETS    = 50

CAM_USER    = ""
CAM_PASS    = ""
VERIFY_TLS  = False

HTTP_PORT   = 8000
JPEG_QUALITY = 80

CAPTURE_REDUCE = 1


def summary():
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
