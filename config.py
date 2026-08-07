"""Single source of truth for all runtime configuration. See CLAUDE.md."""

STREAM_URL  = "http://192.168.1.170/streaming/stream3/video.mjpeg"
MODEL_PATH  = "tflite_model/box_detector_y8n_int8_320_vela.tflite"
LABELS_PATH = "tflite_model/box_detector.txt"

USE_NPU     = True
INFER_EVERY = 3
SCORE_THRES = 0.25
NMS_IOU     = 0.45
MAX_DETS    = 10

CAM_USER    = "admin"
CAM_PASS    = "pixoel"
VERIFY_TLS  = False

HTTP_PORT   = 8000
JPEG_QUALITY = 80
PREVIEW_SCALE = 1
CONFIRM_FRAMES = 2
REGRESS_FRAMES = 5

DEMO_STEPS = [
    {"title": "Scan barcode",  "label": "closed_box", "container": None,       "state": "1000", "kind": "scan"},
    {"title": "Open box",      "label": "open_box",   "container": None,       "state": "1000"},
    {"title": "Matrix in box", "label": "matrix",     "container": "open_box", "state": "1100"},
    {"title": "Foam in box",   "label": "foam",       "container": "open_box", "state": "1110"},
    {"title": "Card in box",   "label": "card",       "container": "open_box", "state": "1111"},
    {"title": "Close box",     "label": "closed_box", "container": None,       "state": "0000"},
]
IDLE_STATE = "0000"
EXCLUSIVE_IOU   = 0.6

BARCODE_ENABLE        = True
BARCODE_INTERVAL_SECS = 0.5
BARCODE_ROI_SCALE     = 0.6
BARCODE_UPSCALE       = 1

SPECS = [
    {
    "C642660001": {
        "sku": "Matrix-800",
        "name": "NXP i.MX93 Industrial Linux Computer",
        "features": "NPU, Quad 3Mbps RS-485, PQC TPM 2.0",
        "origin": "Made in Taiwan"
        }
    }
]

MODBUS_ENABLE   = True
MODBUS_PORT     = "/dev/ttyUSB0"
MODBUS_REGISTERS = [0x000D, 0x000E, 0x000F, 0x0010]
MODBUS_SLAVE    = 1
MODBUS_REFRESH_SECS = 2.0

RECORD_ENABLE    = True
RECORD_STREAM    = "Stream1"
RECORD_PRE_SECS  = 1
RECORD_POST_SECS = 2


def summary():
    steps = " > ".join(
        f"{s['label']}{'*' if s.get('container') else ''}:{s['state']}"
        for s in DEMO_STEPS
    )
    return (
        f"src   {STREAM_URL}\n"
        f"model {MODEL_PATH}  npu={USE_NPU} every={INFER_EVERY} thres={SCORE_THRES}\n"
        f"steps {IDLE_STATE} > {steps}   (* = must be inside container)\n"
        f"out   regs={[hex(r) for r in MODBUS_REGISTERS]} "
        f"modbus={MODBUS_ENABLE}@{MODBUS_PORT} "
        f"barcode={BARCODE_ENABLE} record={RECORD_ENABLE}"
    )
