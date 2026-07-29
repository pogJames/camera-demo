# Smart-camera → NPU object detection → LAN browser stream

Pulls an MJPEG stream, runs **SSD-MobileNetV2** COCO detection (CPU on a dev box,
**Arm Ethos-U65 NPU** on the NXP i.MX93 board), draws boxes, and serves the
annotated video to any LAN browser via a plain `<img>` MJPEG endpoint.

Pure Python: `requests` (capture) + `opencv` (decode/draw) + a tflite interpreter
+ `FastAPI`/`uvicorn` (serve). Design rationale lives in **`CLAUDE.md`**.

## Model

One model: INT8 built-in-postprocess `ssd_mobilenet_v2_coco_quant_postprocess.tflite`.

Its **4 output tensors** mean a built-in `TFLite_Detection_PostProcess` op — boxes
are decoded + NMS'd (incl. cross-class) inside the graph:

| tensor | shape | dtype | meaning |
|---|---|---|---|
| input | `[1,300,300,3]` | uint8 `(scale 1/128, zp 128)` | feed raw `[0,255]` px |
| out 0 | `[1,N,4]` | float32 | boxes `(ymin,xmin,ymax,xmax)` normalized |
| out 1 | `[1,N]` | float32 | class ids (0-indexed, background dropped) |
| out 2 | `[1,N]` | float32 | scores, sorted desc |
| out 3 | `[1]` | float32 | detection count |

`postprocess_builtin()` thresholds, unletterboxes, and applies a **`+1` class
offset** so ids index directly into `coco_labels_list.txt` (line 0 = `???`
background). Being INT8, it **Vela-compiles for the Ethos-U** (the postprocess op
falls back to the A55 CPU — expected, cheap).

## Files

| file | role |
|---|---|
| `detect.py` | main app: capture + inference + control threads + FastAPI server |
| `preprocess.py` | letterbox resize + normalization + dtype/quant conversion |
| `postprocess.py` | built-in-postprocess reader (threshold + class-offset + unletterbox) |
| `interpreter.py` | interpreter factory, optional Ethos-U delegate loader |
| `controller.py` | guided-sequence state machine (advance / fault / reset), pure logic |
| `modbus.py` | Modbus-RTU indicator lamps (serial holding registers), called by the control thread |
| `web/` | browser UI: `index.html` + `style.css` + `app.js` (vanilla JS + SSE) |
| `config.py` | all config, hardcoded (single source of truth — edit this file) |
| `dev_files/fake_server.py` | dev-box MJPEG server (loops a video / JPEGs / a still) |
| `dev_files/capture.py` | standalone dataset-capture app for training images |

## Dev box (WSL / x86 Linux, Python 3.12)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`tflite-runtime` has no py3.12 wheel, so `requirements.txt` installs
**`ai-edge-litert`**; `interpreter.py` auto-selects whichever is present
(`tflite_runtime` → `ai_edge_litert`).

Run the mock camera + the app in two terminals:

```bash
# terminal 1 — serve any local media as MJPEG at :8080
python3 dev_files/fake_server.py path/to/clip.mp4   # or a folder of *.jpg, or one still

# terminal 2 — config.py defaults already point at the mock stream + CPU
python3 detect.py
```

Open **http://localhost:8000/**. Endpoints:

- `GET /` — demo page: live video (`<img src="/stream">`) + sequence side panel
- `GET /stream` — `multipart/x-mixed-replace` annotated MJPEG
- `GET /detections` — JSON `{label, class_id, score, box}`
- `GET /state` — current demo state `{steps, current, complete, fault}`
- `GET /events` — Server-Sent-Events stream of demo state (drives the panel)
- `GET /log/{i}` — proof JPEG snapshotted when step `i` completed (404 if none)
- `POST /reset` — reset the sequence to step 1 (also clears a fault)

## Guided-sequence demo

Present the objects in `config.DEMO_STEPS` (default `bottle → phone →
scissors`) to the camera in order. Each confirmed object advances the side panel
and lights its Modbus lamp; showing the wrong object freezes the sequence and
lights the fault lamp until it's removed. Each completed step keeps a **proof
snapshot** (the annotated frame that confirmed it), reachable via a "view" link
on the step (`/log/{i}`). The run auto-resets `AUTO_RESET_SECS` after completion,
or on the Reset button (which also clears the proofs). Detections must hold for
`CONFIRM_FRAMES` inference cycles to count (debounce). With `MODBUS_ENABLE=False`
(dev-box default) the lamp writes are just logged, so the whole demo runs with no
gateway attached — no `pymodbus` needed until the board. Run the pure-logic
tests with `python -m pytest tests/`.

## Board (NXP i.MX93, Ethos-U65)

The vendor BSP already provides `tflite_runtime` + the Ethos-U delegate at
`/usr/local/lib/libethosu_delegate.so`. Do **not** pip-install a tflite
interpreter there; only the pure-Python deps are needed. The Modbus lamps also
need `pymodbus` (pure-Python): `pip install pymodbus` (pulls `pyserial`).

```bash
sudo mkdir -p /opt/npu
sudo cp detect.py preprocess.py postprocess.py interpreter.py \
        controller.py modbus.py config.py /opt/npu/
sudo cp -r web tflite_model /opt/npu/        # web/ UI + the *_vela.tflite model
```

Edit `/opt/npu/config.py` for the board:

```python
STREAM_URL  = "https://192.168.10.1/streaming/stream3/video.mjpeg"
MODEL_PATH  = "/opt/npu/tflite_model/ssd_mobilenet_v2_coco_quant_postprocess_vela.tflite"
USE_NPU     = True
CAM_USER    = "<user>"
CAM_PASS    = "<pass>"
# Modbus-RTU indicator lamps:
MODBUS_ENABLE = True
MODBUS_PORT   = "/dev/ttyUSB0"   # 9600 8N1 hardcoded; set MODBUS_SLAVE to match the gateway
```

Vela-compile the INT8 model (`vela ...`) to get the `_vela.tflite`. On load,
`interpreter.py` logs input/output tensor details and the delegate init — confirm the
backbone landed on the NPU. Then open `http://<board-ip>:8000/` from the LAN.

### Tuning (on the board only)

`INFER_EVERY` (default 3) controls how often the model runs; in-between frames
reuse the last detections. Tune it — plus the camera's `stream3` resolution/fps —
on the board; dev-CPU timings don't transfer. `[infer]` log lines report
per-inference latency.

## Notes / gotchas

- **Never** load a `_vela.tflite` with `USE_NPU=False` — the unresolved `ethos-u`
  op fails to prepare. The dev box runs the plain INT8 model (every op on CPU).
- `VERIFY_TLS=False` is acceptable only on the isolated camera link.
- Capture is **latest-frame-wins**: no backlog, so latency stays bounded if
  inference or the network stalls.
- No HTTP auth on the server — isolated LAN only.
- `DEMO_STEPS` strings must match lines in `coco_labels_list.txt` exactly (e.g.
  `phone`, not `phone`) — they're what the model emits.
- Lamps are Modbus **holding registers** written `1`/`0` (`write_register`), not
  coils — `STEP_REGS` / `FAULT_REG` are register addresses (see the gateway's Li
  light map).
- Only the control thread writes the Modbus bus. `/reset` resets the controller
  and signals the control thread (via `_reset_flush`) to push the lamps off on
  its next tick — so reset clears the lamps even if the detection stream stalls.
