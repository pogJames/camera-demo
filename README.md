# Smart-camera → NPU object detection → LAN browser stream

Pulls an MJPEG stream, runs **YOLO box/parts detection** (CPU on a dev box,
**Arm Ethos-U65 NPU** on the NXP i.MX93 board), draws boxes, and serves the
annotated video to any LAN browser via a plain `<img>` MJPEG endpoint.

On top of that it runs a **guided packing demo**: scan the box's barcode, then
place matrix → foam → card inside and close it, with a browser side panel and
four Modbus lamps tracking progress.

Pure Python: `requests` (capture) + `opencv` (decode/draw) + a tflite interpreter
+ `zxing-cpp` (barcode) + `FastAPI`/`uvicorn` (serve). Design rationale lives in
**`CLAUDE.md`**.

## Model

`box_detector_y8n_int8_320.tflite` — a YOLOv8n detector, 5 classes
(`box_detector.txt`: card, closed_box, foam, matrix, open_box).

It's a **raw detection head**: 6 outputs, nothing decoded in-graph, 2 tensors per
feature level:

| tensor | shape | dtype | meaning |
|---|---|---|---|
| input | `[1,320,320,3]` | int8 `(scale 1/255, zp -128)` | `[0,1]`-normalized px |
| box ×3 | `[1,G,G,64]` | int8 | DFL logits — 4 sides × 16 bins |
| cls ×3 | `[1,G,G,5]` | int8 | raw class logits (**not** sigmoid'd) |

`G` ∈ {40, 20, 10} → strides 8/16/32. `postprocess_yolo()` does the whole decode
in numpy: sigmoid + threshold, then DFL softmax-expectation on survivors only,
anchor decode, `cv2.dnn.NMSBoxes`, unletterbox. Thresholding first keeps the
normal case at **~0.08 ms** (vs the invoke itself).

NMS is **class-aware** (boxes offset by class id before NMS). This matters for
this model: items sit *inside* the box, so a class-agnostic pass deletes whichever
of `matrix`/`open_box` scores lower.

`box_detector.txt` has **no background line**, so `label_offset` is `0`. Being
INT8 it **Vela-compiles for the Ethos-U**; the decode is pure numpy on the A55.

## Files

| file | role |
|---|---|
| `detect.py` | main app: capture + inference + control threads + FastAPI server |
| `preprocess.py` | letterbox resize + normalization + dtype/quant conversion |
| `postprocess.py` | YOLO head decode (sigmoid + threshold, DFL, anchors, class-aware NMS, unletterbox) |
| `spatial.py` | containment gate — an item counts only when it's inside its container |
| `barcode.py` | Code 128 reader (zxing-cpp) for the closed-box crop |
| `interpreter.py` | interpreter factory, optional Ethos-U delegate loader |
| `controller.py` | sequence state machine (advance / error / regress / scan gate), pure logic |
| `modbus.py` | Modbus-RTU indicator lamps (serial holding registers), called by the control thread |
| `camera.py` | uEye REST client: trigger event recording + fetch clip for the `/log` proxy |
| `web/` | browser UI: `index.html` + `style.css` + `app.js` (vanilla JS + SSE) |
| `config.py` | all config, hardcoded (single source of truth — edit this file) |
| `tests/fake_server.py` | dev-box MJPEG server (loops a video / JPEGs / a still) |
| `tests/capture.py` | standalone dataset-capture app for training images |
| `tests/tune_barcode.py` | sweeps barcode crop/scale against the live camera, prints what works |

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
python3 tests/fake_server.py path/to/clip.mp4   # or a folder of *.jpg, or one still

# terminal 2 — point MODEL_PATH at the non-vela build and set USE_NPU=False
python3 detect.py
```

Open **http://localhost:8000/**. Endpoints:

- `GET /` — demo page: live video (`<img src="/stream">`) + sequence side panel
- `GET /stream` — `multipart/x-mixed-replace` annotated MJPEG
- `GET /detections` — JSON `{label, class_id, inside, score, box}`
- `GET /state` — demo state `{steps, current, complete, error, misplaced, scan, lamps}`
- `GET /events` — Server-Sent-Events stream of demo state (drives the panel)
- `GET /log/{i}` — proxies the camera clip recorded when step `i` completed (404 if none)
- `POST /reset` — reset the sequence to step 1 (also clears the scan and clips)

## Guided packing demo

Steps come from `config.DEMO_STEPS`, one dict each:

```python
{"title": "Matrix in box", "label": "matrix", "container": "open_box", "state": "1100"}
```

- **`label`** — the model class that satisfies the step.
- **`container`** — the item must be **fully inside** a detection of this class,
  or it doesn't count. Holding the matrix beside the box does nothing.
- **`state`** — the lamp pattern once the step is done, one bit per
  `MODBUS_REGISTERS`. The last step's `"0000"` is how the lamps clear at the end.
- **`kind: "scan"`** — also requires a decoded barcode. Step 0 will not complete
  until the label reads, so the run can't start on an unidentified box.

Behaviour worth knowing:

- **Wrong item** → red banner and the sequence freezes, until it's removed. An
  item that's merely *visible* never errors; only one placed **in the box** does.
- **Item removed** → the sequence steps back and dims that lamp. Only the last
  completed step is re-checked, and only while the box is in view — otherwise
  occlusion would undo correct work.
- **Item visible but not inside** → amber "not inside the box" hint, so a stalled
  sequence explains itself.
- **Barcode** → `config.SPECS` maps the code to part detail, printed in the
  sidebar; unknown codes still run, flagged "not in catalog".
- Detections must hold for `CONFIRM_FRAMES` cycles to count, and be missing for
  `REGRESS_FRAMES` to un-count.
- Each completed step triggers a **camera clip** (uEye event recording), linked
  from the step (`/log/{i}`, proxied). It finalizes ~`RECORD_POST_SECS` after the
  trigger and is the raw `RECORD_STREAM` (no boxes).
- Reset is the only way back to step 0 — no auto-reset.

With `MODBUS_ENABLE=False` the lamp writes are just logged, so the demo runs with
no gateway attached. Run the pure-logic tests with `python -m pytest tests/`.

## Board (NXP i.MX93, Ethos-U65)

The vendor BSP already provides `tflite_runtime` + the Ethos-U delegate at
`/usr/local/lib/libethosu_delegate.so`. Do **not** pip-install a tflite
interpreter there. Also needed: `pymodbus` (lamps) and `zxing-cpp` (barcode) —
both have prebuilt aarch64 wheels.

```bash
sudo mkdir -p /opt/npu
sudo cp detect.py preprocess.py postprocess.py interpreter.py spatial.py \
        barcode.py controller.py modbus.py camera.py config.py /opt/npu/
sudo cp -r web tflite_model /opt/npu/        # web/ UI + the *_vela.tflite model
```

Copy **all** of them together — a stale `barcode.py` against a new `config.py`
fails with a `TypeError` on every scan attempt.

Edit `/opt/npu/config.py` for the board:

```python
STREAM_URL  = "https://192.168.10.1/streaming/stream3/video.mjpeg"   # must be 1080p
MODEL_PATH  = "/opt/npu/tflite_model/box_detector_y8n_int8_320_vela.tflite"
USE_NPU     = True
CAM_USER    = "<user>"
CAM_PASS    = "<pass>"
MODBUS_ENABLE = True
MODBUS_PORT   = "/dev/ttyUSB0"   # 9600 8N1 hardcoded; set MODBUS_SLAVE to match the gateway
```

Vela-compile the INT8 model (`vela ...`) to get the `_vela.tflite`. On load,
`interpreter.py` logs input/output tensor details and the delegate init — confirm
the backbone landed on the NPU. Then open `http://<board-ip>:8000/` from the LAN.

### Tuning (on the board only)

- **`INFER_EVERY`** (default 3) — how often the model runs; in-between frames
  reuse the last detections. `[infer]` log lines report per-inference latency.
- **Barcode** — run `python tests/tune_barcode.py` with the box in front of the
  camera. It sweeps `BARCODE_CROP` × `BARCODE_SCALE` over captured frames and
  prints read rate and worst-case time per combination; copy the winning row into
  `config.py`. Worst case matters more than median: it's what occupies the
  control thread while waiting. `[barcode] slow read:` warns above 200 ms.
- **`PREVIEW_SCALE`** — shrinks the browser preview before encoding. Encode cost
  is per connected viewer, so this is the lever if the stream lags with several
  tabs open.

## Notes / gotchas

- **The stream must be 1080p.** At 720p the barcode is ~65×15 px at demo
  distance and no amount of upscaling decodes it.
- **Never** load a `_vela.tflite` with `USE_NPU=False` — the unresolved `ethos-u`
  op fails to prepare. The dev box runs the plain INT8 model (every op on CPU).
- `VERIFY_TLS=False` is acceptable only on the isolated camera link.
- Capture is **latest-frame-wins**, and frames from the store are **read-only and
  shared, not copied** — anything that draws must resize or copy first.
  `put_frame` marks them non-writeable so a violation raises immediately.
- No HTTP auth on the server — isolated LAN only.
- `DEMO_STEPS` labels must match lines in `box_detector.txt` exactly. Classes not
  listed are ignored by the state machine entirely.
- Lamps are Modbus **holding registers** written `1`/`0` (`write_register`), not
  coils — `MODBUS_REGISTERS` are register addresses (see the gateway's Li light
  map). There is no fault lamp; wrong items are a UI error only.
- Only the control thread writes the Modbus bus. `/reset` resets the controller
  and signals the control thread (via `_reset_flush`) to push the lamps off on
  its next tick — so reset clears the lamps even if the detection stream stalls.
