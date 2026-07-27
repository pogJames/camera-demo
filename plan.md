# Project Plan: Smart-camera → NPU object detection → LAN browser stream

## Goal
Pull an MJPEG video stream from an IDS uEye camera over the network, run
SSD-MobileNetV2 object detection on the NXP i.MX93 Ethos-U65 NPU, draw
bounding boxes on the frames, and serve the annotated video to any browser on
the LAN via a plain `<img>` MJPEG endpoint.

Detection target: generic COCO (80 classes), starting model SSD-MobileNetV2.

## Hardware / platform context (do not re-derive this)
- Board: NXP i.MX93 (2x Cortex-A55 + Arm Ethos-U65 NPU, 0.5 TOPS, INT8-only).
- NPU is confirmed WORKING: delegate + /dev/ethosu0 + Vela verified via
  benchmark_model and a prior custom live-inference project.
- Ethos-U delegate: `/usr/lib/libethosu_delegate.so`, loaded explicitly.
- NPU runs INT8 only. Two model artifacts exist and MUST be kept separate:
    * plain INT8 `.tflite`         -> runs on CPU (dev box / WSL)
    * vela-compiled `_vela.tflite` -> runs on NPU (the board)
  Same source model; select by env var, never hardcode.
- Camera: IDS uEye, MJPEG over HTTP(S), e.g.
  `https://192.168.10.1/streaming/stream3/video.mjpeg`, HTTP Basic auth,
  self-signed cert. The camera's OpenAPI file is control-plane only; video is
  this MJPEG URL.

## Design decision (already made — do not switch to GStreamer/NNStreamer)
Pure Python: requests (capture) + OpenCV (decode/draw) + tflite_runtime
(inference) + FastAPI (serve). Chosen for debuggability and easy LAN serving.
GStreamer/NNStreamer is the fallback ONLY if the A55 cores bottleneck on JPEG
decode/encode — out of scope for now.

## STEP 0 — Inspect the model BEFORE writing postprocessing (blocking step)
A model file is already in the project folder. First thing: load it and print
tensor details. Do NOT write the box-decode logic until this is known.

```python
import tflite_runtime.interpreter as tflite  # fall back to tensorflow.lite
i = tflite.Interpreter(model_path=MODEL_PATH); i.allocate_tensors()
print("INPUT:",  i.get_input_details())
print("OUTPUT:", i.get_output_details())
```

Branch on the output tensors — this determines the whole postprocess path:
- **4 output tensors** (boxes, classes, scores, count)  -> model has built-in
  `TFLite_Detection_PostProcess`. Boxes are already decoded + NMS'd. Just read
  them. (Note: that op runs on CPU, fine for dev; on NPU it shows as CPU
  fallback in the Vela report — acceptable.)
- **2 raw tensors** (e.g. raw box regressions + class scores) -> "no_postprocess"
  model. Needs anchor decode against `box_priors.txt` + manual NMS. NPU-friendly
  form. Expect a `box_priors.txt` + `coco_labels_list.txt` alongside the model.

Report which branch applies before continuing.

## Architecture
```
requests (MJPEG, stream=True, basic auth, verify=False)
  -> parse multipart boundary -> cv2.imdecode  (capture thread)
  -> letterbox/resize to model input + INT8 quantize
  -> tflite Interpreter [+ ethos-u delegate on board]  (invoke)
  -> dequantize -> decode boxes -> NMS
  -> draw boxes/labels on latest full-res frame  (annotate)
  -> FastAPI multipart/x-mixed-replace  ->  browser <img>
```
Decouple rates: capture/display at stream fps; run inference every Nth frame;
reuse the most recent detections for in-between frames. N is tunable and MUST be
tuned ON THE BOARD, not on the dev box (dev CPU timings don't transfer).

## Config — all platform differences behind env vars (single source of truth)
```python
STREAM_URL = os.getenv("STREAM_URL", "http://localhost:8080/mock.mjpeg")
MODEL_PATH = os.getenv("MODEL_PATH", "models/ssd_mobilenet_v2_coco.tflite")
LABELS_PATH= os.getenv("LABELS_PATH","models/coco_labels_list.txt")
PRIORS_PATH= os.getenv("PRIORS_PATH","models/box_priors.txt")  # only if no_postprocess
USE_NPU    = os.getenv("USE_NPU", "0") == "1"     # "1" on board
INFER_EVERY= int(os.getenv("INFER_EVERY", "3"))   # inference frame stride
SCORE_THRES= float(os.getenv("SCORE_THRES", "0.5"))
CAM_USER   = os.getenv("CAM_USER", "")
CAM_PASS   = os.getenv("CAM_PASS", "")
HTTP_PORT  = int(os.getenv("HTTP_PORT", "8000"))
```
Dev box: defaults + plain INT8 model, USE_NPU=0, mock/recorded stream.
Board: MODEL_PATH -> _vela.tflite, USE_NPU=1, real camera URL, creds set.

## Files to create
- `detect.py` — main app (capture thread + inference loop + FastAPI server).
- `postprocess.py` — box decode + NMS (branch per STEP 0). NumPy only.
- `interp.py` — interpreter factory with optional delegate loader.
- `mock_server.py` — tiny HTTP MJPEG server that loops a local video/JPEGs,
  for dev-box testing without the real camera.
- `requirements.txt`
- `npu-python.service` — systemd unit (env vars via EnvironmentFile).
- `.env.example` — documents all env vars above.
- `README.md` — run instructions for both dev box and board.

## Implementation detail per component

### interp.py — delegate factory
```python
def make_interpreter(model_path, use_npu):
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    if use_npu:
        d = tflite.load_delegate("/usr/lib/libethosu_delegate.so")
        return tflite.Interpreter(model_path=model_path, experimental_delegates=[d])
    return tflite.Interpreter(model_path=model_path)
```
On load, log input/output details and (board) the delegate init lines so we can
confirm "N nodes delegated".

### Capture (thread)
- `requests.get(STREAM_URL, stream=True, auth=HTTPBasicAuth(...), verify=False)`.
- Parse `multipart/x-mixed-replace`: split on boundary, read each JPEG part,
  `cv2.imdecode(np.frombuffer(...), cv2.IMREAD_COLOR)`.
- Keep only the latest frame in a lock-guarded slot (drop stale frames; do not
  queue up backlog). Reconnect with backoff on stream drop.

### Preprocess
- Letterbox to model input size (from STEP 0 input details), keep aspect ratio,
  record scale+pad to map boxes back to full-res.
- Convert to model dtype. If INT8 input, quantize with the input tensor's
  (scale, zero_point) from get_input_details.
- Match SSD-MobileNet normalization if the model expects float input:
  `(pixel - 127.5) / 127.5`. Confirm against STEP 0 dtype (uint8 vs float32).

### Inference loop
- Only every INFER_EVERY-th frame. set_tensor -> invoke -> get_tensor.
- Dequantize outputs using output (scale, zero_point) if INT8.

### postprocess.py — TWO code paths, pick per STEP 0
- **Built-in postprocess (4 outputs):** read boxes[ymin,xmin,ymax,xmax normalized],
  classes, scores; filter by SCORE_THRES; unletterbox to full-res coords.
- **no_postprocess (raw + priors):** load `box_priors.txt`; decode each anchor:
  SSD center-size decode (with the standard 0.1/0.2 variance scales — verify
  against the model's training config), apply sigmoid/softmax to class scores as
  appropriate, threshold, then `cv2.dnn.NMSBoxes` (or NumPy NMS) per class,
  unletterbox. Keep decode vectorized in NumPy.

### Annotate
- `cv2.rectangle` + `cv2.putText` (label + score) on the full-res latest frame,
  using the most recent detections. Map class id -> name via LABELS_PATH.

### Serve (FastAPI + uvicorn)
- `GET /` : minimal HTML page with `<img src="/stream">`.
- `GET /stream` : `StreamingResponse` yielding
  `--frame\r\nContent-Type: image/jpeg\r\n\r\n<bytes>` repeatedly
  (`multipart/x-mixed-replace; boundary=frame`), re-encoding the annotated
  frame with `cv2.imencode(".jpg", ...)`.
- `GET /detections` : JSON of current detections (label, score, box) for a
  future sidebar / debugging.
- Bind `0.0.0.0:HTTP_PORT`. No auth (isolated LAN); note this in README.

### mock_server.py
- Serve a looping local `.mp4`/folder of JPEGs as MJPEG at
  `http://localhost:8080/mock.mjpeg` so detect.py runs unchanged on the dev box.

### npu-python.service (board)
```
[Unit]
Description=NPU camera object detection
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/npu
EnvironmentFile=/opt/npu/.env
ExecStart=/usr/bin/python3 /opt/npu/detect.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

## Build order (do in this sequence)
1. STEP 0: inspect model, report output-tensor branch. STOP and confirm branch.
2. Scaffolding: requirements.txt, interp.py, config block, .env.example.
3. mock_server.py + capture path; prove frames decode on the dev box (save a
   frame to disk to verify).
4. Preprocess + inference on dev box (USE_NPU=0, plain INT8 model). Print raw
   output shapes; confirm they match STEP 0.
5. postprocess.py for the correct branch; draw boxes; save an annotated frame;
   eyeball correctness.
6. FastAPI server; view `/stream` in a browser against the mock stream.
7. Board bring-up: set env to _vela.tflite + USE_NPU=1 + real camera URL/creds.
   Confirm delegate log shows nodes delegated to NPU. View stream over LAN.
8. Tune INFER_EVERY + camera stream3 resolution/fps ON THE BOARD for smoothness.
9. Install systemd unit; `systemctl enable --now npu-python.service`; verify it
   survives reboot; check `journalctl -u npu-python.service -f`.

## Constraints / gotchas to respect
- Do NOT load a `_vela.tflite` without the delegate (dev box) — it will fail on
  the unresolved `ethos-u` custom op. Dev box uses the plain INT8 model.
- Keep the dev model INT8 (not FP32) so dev numerics mirror the board.
- verify=False is acceptable only on the isolated camera link; note in README.
- Never queue frame backlog; always process the latest frame to avoid latency
  creep.
- Do not tune performance/frame-stride on the dev box; only on the board.
- Vela compilation of the model is done separately (compile_models.sh or the
  vela CLI); this app just consumes whichever MODEL_PATH it's given.

## Open item to confirm at STEP 0
Whether the provided model is built-in-postprocess (4 outputs) or no_postprocess
(raw + box_priors.txt). Everything in postprocess.py forks on this. Confirm
before implementing that file.