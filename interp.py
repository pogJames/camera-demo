"""Interpreter factory with optional Ethos-U delegate loader.

One place that hides every runtime difference between the dev box and the board:
- dev box (WSL/Linux CPU): plain float/INT8 .tflite, no delegate.
- board (i.MX93): _vela.tflite + /usr/lib/libethosu_delegate.so on the NPU.

The tflite Python API is imported by preference:
  1. tflite_runtime            (standard on the NXP board image)
  2. ai_edge_litert            (the modern successor; what pip installs on py3.12)
"""

import numpy as np


def _import_tflite():
    """Return a module exposing Interpreter + load_delegate, whichever exists."""
    try:
        import tflite_runtime.interpreter as tflite
        return tflite, "tflite_runtime"
    except ImportError:
        pass
    try:
        from ai_edge_litert import interpreter as tflite
        return tflite, "ai_edge_litert"
    except ImportError:
        pass


ETHOSU_DELEGATE = "/usr/local/lib/libethosu_delegate.so"


def make_interpreter(model_path, use_npu, log=print):
    tflite, backend = _import_tflite()
    log(f"[interp] tflite backend: {backend}")
    log(f"[interp] model: {model_path}  use_npu={use_npu}")

    if use_npu:
        log(f"[interp] loading Ethos-U delegate: {ETHOSU_DELEGATE}")
        delegate = tflite.load_delegate(ETHOSU_DELEGATE)
        interp = tflite.Interpreter(
            model_path=model_path, experimental_delegates=[delegate]
        )
    else:
        interp = tflite.Interpreter(model_path=model_path)

    interp.allocate_tensors()
    _log_io(interp, log)
    return interp


def _log_io(interp, log):
    """Log input/output tensor details so we can confirm the STEP-0 branch and,
    on the board, eyeball that the delegate accepted the model."""
    for d in interp.get_input_details():
        log(f"[interp] INPUT  {d['name']!r} shape={[int(s) for s in d['shape']]} "
            f"dtype={np.dtype(d['dtype']).name} quant={d['quantization']}")
    for d in interp.get_output_details():
        log(f"[interp] OUTPUT {d['name']!r} shape={[int(s) for s in d['shape']]} "
            f"dtype={np.dtype(d['dtype']).name} quant={d['quantization']}")
