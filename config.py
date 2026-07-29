"""Single source of truth for all runtime configuration. See CLAUDE.md."""

STREAM_URL  = "http://192.168.1.170/streaming/stream3/video.mjpeg"
MODEL_PATH  = "tflite_model/ssd_mobilenet_v2_coco_quant_postprocess_vela.tflite"
LABELS_PATH = "tflite_model/coco_labels_list.txt"

USE_NPU     = True
INFER_EVERY = 3

SCORE_THRES = 0.5
MAX_DETS    = 50

CAM_USER    = ""
CAM_PASS    = ""
VERIFY_TLS  = False

HTTP_PORT   = 8000
JPEG_QUALITY = 80

CAPTURE_REDUCE = 1

# Guided-sequence demo: labels must match lines in LABELS_PATH exactly.
DEMO_STEPS   = ["bottle", "cell phone", "scissors"]
CONFIRM_FRAMES = 2
AUTO_RESET_SECS = 5

# Modbus indicator lamps (RTU/serial, holding registers written 1/0). Off = dev
# box, LampBank just logs. Register addresses per the gateway's Li light map.
MODBUS_ENABLE   = True
MODBUS_PORT     = "/dev/ttyUSB0"
MODBUS_BAUD     = 9600
MODBUS_PARITY   = "N"
MODBUS_STOPBITS = 1
MODBUS_BYTESIZE = 8
MODBUS_SLAVE    = 1
STEP_REGS       = [0x000D, 0x000E, 0x000F]   # Li0001..Li0003, index-aligned with DEMO_STEPS
FAULT_REG       = 0x0010                       # Li0004
DONE_REG        = None
MODBUS_REFRESH_SECS = 2.0


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
        f"MODBUS_BAUD={MODBUS_BAUD}  MODBUS_SLAVE={MODBUS_SLAVE}\n"
        f"STEP_REGS={[hex(r) for r in STEP_REGS]}  "
        f"FAULT_REG={hex(FAULT_REG) if FAULT_REG is not None else None}  "
        f"DONE_REG={hex(DONE_REG) if DONE_REG is not None else None}"
    )
