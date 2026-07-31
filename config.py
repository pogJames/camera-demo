"""Single source of truth for all runtime configuration. See CLAUDE.md."""

STREAM_URL  = "http://192.168.1.170/streaming/stream3/video.mjpeg"
MODEL_PATH  = "tflite_model/ssd_mobilenet_v2_coco_quant_postprocess_vela.tflite"
LABELS_PATH = "tflite_model/coco_labels_list.txt"

USE_NPU     = True
INFER_EVERY = 3
SCORE_THRES = 0.5
MAX_DETS    = 50

CAM_USER    = "admin"
CAM_PASS    = "pixoel"
VERIFY_TLS  = False

HTTP_PORT   = 8000
JPEG_QUALITY = 80
CAPTURE_REDUCE = 1

DEMO_STEPS   = ["bottle", "phone", "scissors"]
CONFIRM_FRAMES = 3
AUTO_RESET_SECS = 20

MODBUS_ENABLE   = True
MODBUS_PORT     = "/dev/ttyUSB0"
MODBUS_SLAVE    = 1
STEP_REGS       = [0x000D, 0x000E, 0x000F]
FAULT_REG       = 0x0010
MODBUS_REFRESH_SECS = 2.0

RECORD_ENABLE    = True
RECORD_STREAM    = "Stream1"
RECORD_PRE_SECS  = 2
RECORD_POST_SECS = 1


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
        f"CAPTURE_REDUCE={CAPTURE_REDUCE}\n"
        f"DEMO_STEPS={DEMO_STEPS}  CONFIRM_FRAMES={CONFIRM_FRAMES}  "
        f"AUTO_RESET_SECS={AUTO_RESET_SECS}\n"
        f"MODBUS_ENABLE={MODBUS_ENABLE}  MODBUS_PORT={MODBUS_PORT}  "
        f"MODBUS_SLAVE={MODBUS_SLAVE}\n"
        f"STEP_REGS={[hex(r) for r in STEP_REGS]}  "
        f"FAULT_REG={hex(FAULT_REG) if FAULT_REG is not None else None}\n"
        f"RECORD_ENABLE={RECORD_ENABLE}  RECORD_STREAM={RECORD_STREAM}  "
        f"RECORD_PRE_SECS={RECORD_PRE_SECS}  RECORD_POST_SECS={RECORD_POST_SECS}"
    )
