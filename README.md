# Smart-camera → NPU object detection → LAN browser stream

Pulls an MJPEG video stream, runs **SSD-MobileNetV2** COCO object detection
(CPU on a dev box, **Arm Ethos-U65 NPU** on the NXP i.MX93 board), draws bounding
boxes, and serves the annotated video to any browser on the LAN via a plain
`<img>` MJPEG endpoint.

Pure Python: `requests` (capture) + `opencv` (decode/draw) + a tflite interpreter
(inference) + `FastAPI`/`uvicorn` (serve).

## Model: `ssd_mobilenet_v2_coco_quant_postprocess.tflite` (INT8, NPU-ready)

The app supports one model: the INT8 built-in-postprocess SSD-MobileNetV2.

**4 output tensors** = built-in `TFLite_Detection_PostProcess` — boxes are
already decoded + NMS'd (incl. cross-class NMS) inside the graph.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| input `normalized_input_image_tensor` | `[1,300,300,3]` | **uint8** `(scale 1/128, zp 128)` | feed raw `[0,255]` px |
| output 0 | `[1,20,4]` | float32 | boxes `(ymin,xmin,ymax,xmax)` normalized |
| output 1 | `[1,20]` | float32 | class ids (0-indexed, background dropped) |
| output 2 | `[1,20]` | float32 | scores, sorted desc |
| output 3 | `[1]` | float32 | detection count |

`postprocess_builtin()` just thresholds, unletterboxes, and applies a **`+1`
class offset** so ids index directly into `coco_labels_list.txt` (line 0 =
`???` background). This is INT8, so it **Vela-compiles for the Ethos-U NPU**
(the detection-postprocess op falls back to the A55 CPU — expected, cheap).
`preprocess.to_input_tensor` normalizes with the SSD `mean/std=127.5` and then
re-quantizes with the model's own quant params — the round-trip reproduces the
raw uint8 pixels this model wants.

## Files

| file | role |
|---|---|
| `detect.py` | main app: capture thread + inference thread + FastAPI server |
| `preprocess.py` | letterbox resize + normalization + dtype/quant conversion |
| `postprocess.py` | built-in-postprocess reader (threshold + class-offset + unletterbox) |
| `interp.py` | interpreter factory, optional Ethos-U delegate loader |
| `config.py` | all config, hardcoded (single source of truth — edit this file) |
| `mock_server.py` | dev-box MJPEG server (loops a video / JPEGs / a still) |
| `npu-python.service` | systemd unit for the board |

## Dev box (WSL / x86 Linux, Python 3.12)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`tflite-runtime` has no Python-3.12 wheel, so `requirements.txt` installs
**`ai-edge-litert`** instead; `interp.py` auto-selects whichever interpreter is
present (`tflite_runtime` → `ai_edge_litert` → `tensorflow.lite`).

Run the mock camera + the app in two terminals:

```bash
# terminal 1 — serve any local media as MJPEG at :8080
python3 mock_server.py path/to/clip.mp4        # or a folder of *.jpg, or one still

# terminal 2 — config.py defaults already point at the mock stream + CPU
python3 detect.py
```

Open **http://localhost:8000/** in a browser. Endpoints:

- `GET /` — HTML page with `<img src="/stream">`
- `GET /stream` — `multipart/x-mixed-replace` annotated MJPEG
- `GET /detections` — JSON of current detections `{label, class_id, score, box}`

## Board (NXP i.MX93, Ethos-U65)

The vendor BSP already provides `tflite_runtime` + the Ethos-U delegate at
`/usr/lib/libethosu_delegate.so`. Do **not** pip-install a tflite interpreter
there; only the pure-Python deps (`numpy`, `opencv`, `requests`, `fastapi`,
`uvicorn`) are needed.

```bash
sudo mkdir -p /opt/npu
sudo cp detect.py preprocess.py postprocess.py interp.py config.py /opt/npu/
sudo cp -r tflite_model /opt/npu/            # include the *_vela.tflite here
```

Edit `/opt/npu/config.py` for the board:

```python
STREAM_URL  = "https://192.168.10.1/streaming/stream3/video.mjpeg"
# Vela-compile the INT8 quant_postprocess model, then point here:
MODEL_PATH  = "/opt/npu/tflite_model/ssd_mobilenet_v2_coco_quant_postprocess_vela.tflite"
USE_NPU     = True
CAM_USER    = "<user>"
CAM_PASS    = "<pass>"
VERIFY_TLS  = False
```

The INT8 `ssd_mobilenet_v2_coco_quant_postprocess.tflite` is the one to run
through Vela (`vela ...`) to get the `_vela.tflite`. On load, `interp.py` logs
the delegated node count — confirm the backbone landed on the NPU (the
`TFLite_Detection_PostProcess` op stays on CPU, as expected).

Install and run under systemd:

```bash
sudo cp npu-python.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now npu-python.service
journalctl -u npu-python.service -f
```

On startup, `interp.py` logs the input/output tensor details and the delegate
init lines — confirm the log shows nodes delegated to the NPU. Then open
`http://<board-ip>:8000/` from any machine on the LAN.

### Tuning (on the board only)

`INFER_EVERY` in `config.py` (default 3) controls how often the model runs;
frames in between reuse the last detections. Tune it — plus the camera's
`stream3` resolution/fps — **on the board**, not the dev box: dev-CPU timings
don't transfer. `[infer]` log lines report per-inference latency.

## Notes / gotchas

- **Never** load a `_vela.tflite` without `USE_NPU=True` — the unresolved
  `ethos-u` custom op fails to prepare. The dev box runs the plain (non-vela)
  INT8 model with `USE_NPU=False` (every op on CPU).
- `VERIFY_TLS=False` (verify=False) is acceptable only on the isolated camera link.
- The capture path is **latest-frame-wins**: it never queues a backlog, so
  end-to-end latency stays bounded if inference or the network stalls.
- No HTTP auth on the server — intended for an isolated LAN only.
