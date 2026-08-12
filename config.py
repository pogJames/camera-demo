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
PREVIEW_SCALE = 0.5
CONFIRM_FRAMES = 2
REGRESS_FRAMES = 5

IDLE_STATE = "1111"         # lamps while waiting for a barcode
SCAN_STATE = "0000"         # lamps once a product is loaded, before step 1
SCAN_LABEL = "closed_box"   # the class whose crop is searched for a barcode
EXCLUSIVE_IOU   = 0.6

BARCODE_ENABLE        = True
BARCODE_INTERVAL_SECS = 0.5
BARCODE_CROP          = 0.3
BARCODE_SCALE         = 3

SPECS = {
    "C642660001": {
        "sku": "Matrix-800",
        "name": "Linux-Ready 64-bit Multi-Core Secure Edge AI Industrial Gateway",
        "features": [
            "NXP i.MX93 Dual-Core Arm Cortex-A55 + M33 Real-Time MCU",
            "NPU with End-to-End Tooling for Edge AI Inference",
            "PQC Ready Hardware TPM",
            "2 x Gigabit Ethernet with TSN on LAN 1",
            "4 x 3Mbps Baud Rate RS-485",
            "2 x Digital Input with 24V Opto-isolation",
            "1 x Relay-switched Digital Output",
            "1 x M.2 B-Key Expansion Slot for Wireless or Cellular Connectivity",
            "Standalone Watchdog backed with Super Capacitor and Battery",
            "USB 2.0 Type-C and Micro-SD Slot for Backup",
            "Advanced Security with Arm TrustZone and NXP EdgeLock Secure Enclave",
        ],
        "origin": "Made in Taiwan",
        "steps": [
            {"title": "Open box",      "label": "open_box",   "container": None,       "state": "1000"},
            {"title": "Matrix in box", "label": "matrix",     "container": "open_box", "state": "1100"},
            {"title": "Foam in box",   "label": "foam",       "container": "open_box", "state": "1110"},
            {"title": "Card in box",   "label": "card",       "container": "open_box", "state": "1111"},
            {"title": "Close box",     "label": "closed_box", "container": None,       "state": "0000"},
        ],
    },
}


def containers():
    """label -> container class, merged across every product's steps."""
    out = {}
    for spec in SPECS.values():
        for s in spec.get("steps", ()):
            if s.get("container"):
                out.setdefault(s["label"], s["container"])
    return out

MODBUS_ENABLE   = True
MODBUS_PORT     = "/dev/ttyUSB0"
MODBUS_REGISTERS = [0x000D, 0x000E, 0x000F, 0x0010]
MODBUS_SLAVE    = 1
MODBUS_REFRESH_SECS = 2.0

RUNS_ENABLE = True
RUNS_DB     = "data/runs.db"

RECORD_ENABLE    = True
RECORD_STREAM    = "Stream1"
RECORD_PRE_SECS  = 1
RECORD_POST_SECS = 2


def summary():
    products = "\n".join(
        f"      {code} {spec['sku']}: {IDLE_STATE} > " + " > ".join(
            f"{s['label']}{'*' if s.get('container') else ''}:{s['state']}"
            for s in spec.get("steps", ()))
        for code, spec in SPECS.items()
    )
    return (
        f"src   {STREAM_URL}\n"
        f"model {MODEL_PATH}  npu={USE_NPU} every={INFER_EVERY} thres={SCORE_THRES}\n"
        f"steps scan {SCAN_LABEL}, then per product   (* = must be inside container)\n"
        f"{products}\n"
        f"out   regs={[hex(r) for r in MODBUS_REGISTERS]} "
        f"modbus={MODBUS_ENABLE}@{MODBUS_PORT} "
        f"barcode={BARCODE_ENABLE} record={RECORD_ENABLE}"
    )
