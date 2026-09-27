"""Keras -> TFLite (FP32 default, static INT8 ablation), TFLM op check, host parity ([S13.4]). ADR 0010.

Device parity/latency/arena must still be measured on the ESP32-S3 (P10, [S13.4] measure_on_device);
here the host TFLite interpreter is the reference for "model file == Keras model".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tensorflow as tf

TFLM_OPS_FILE = Path(__file__).with_name("tflm_ops.txt")


def tflm_supported() -> set[str]:
    lines = TFLM_OPS_FILE.read_text(encoding="utf-8").splitlines()
    return {ln.strip().lower() for ln in lines if ln.strip() and not ln.startswith("#")}


def _norm(op: str) -> str:
    return op.replace("_", "").lower()


def convert(model: Any, int8_rep: npt.NDArray[np.float32] | None = None) -> bytes:
    """FP32 flatbuffer; with int8_rep, full-integer static PTQ with that representative set."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8_rep is not None:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep() -> Any:
            for i in range(min(len(int8_rep), 500)):
                yield [int8_rep[i : i + 1]]

        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.float32  # quantize/dequantize at the boundary
        conv.inference_output_type = tf.float32
    out: bytes = conv.convert()
    return out


def ops_used(flatbuffer: bytes) -> list[str]:
    interp = tf.lite.Interpreter(model_content=flatbuffer)
    return sorted({d["op_name"] for d in interp._get_ops_details()})


def op_check(flatbuffer: bytes) -> dict[str, Any]:
    used = ops_used(flatbuffer)
    sup = tflm_supported()
    missing = [o for o in used if _norm(o) not in sup]
    return {"ops": used, "unsupported_in_tflm": missing, "ok": not missing}


def run(flatbuffer: bytes, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Host TFLite interpreter, one sample at a time (as on the device)."""
    interp = tf.lite.Interpreter(model_content=flatbuffer)
    interp.allocate_tensors()
    i_det, o_det = interp.get_input_details()[0], interp.get_output_details()[0]
    out = []
    for k in range(len(x)):
        interp.set_tensor(i_det["index"], x[k : k + 1].astype(np.float32))
        interp.invoke()
        out.append(interp.get_tensor(o_det["index"])[0].copy())
    return np.asarray(out, dtype=np.float32)


def arena_estimate(flatbuffer: bytes) -> int:
    """Upper bound of activation memory: sum of non-constant tensor bytes (TFLM plans reuse -> less)."""
    interp = tf.lite.Interpreter(model_content=flatbuffer)
    interp.allocate_tensors()
    total = 0
    for t in interp.get_tensor_details():
        if t.get("sparsity_parameters") or not len(t["shape"]):
            continue
        try:
            interp.get_tensor(t["index"])
            is_const = t["name"].endswith(("kernel", "bias")) or "ReadVariableOp" in t["name"]
        except ValueError:
            is_const = False
        if not is_const:
            total += int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
    return total
