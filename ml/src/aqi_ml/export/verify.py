"""Student artifact verification before registration/OTA (cloud [C09] verification, [S18]).

verify(model_dir) re-checks, from the files on disk:
- schema_hash of the artifact == active feature_schema.yaml hash (I05; firmware would reject it);
- size budgets: RF <= 400 KB; CNN <= 300 KB and TFLM ops only ([C09]);
- parity on 1000 validation vectors: RF blob through the host-compiled C evaluator (same code as the
  firmware), CNN FP32 .tflite through the TFLite interpreter vs the Keras model; max |diff| < 1e-4;
- sha256 + size for the OTA manifest ([S16.4]).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from aqi_ml.features.dataset import ROOT
from aqi_shared.schema_loader import load as load_schema

RF_BUDGET, CNN_BUDGET, ARENA_BUDGET = 400_000, 300_000, 128_000
PARITY_TOL = 1e-4


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rf_blob_schema_hash(blob: bytes) -> str:
    """rf_blob header: b"AQRF" + <HHHH (version, n_out, n_feat, 0) + schema hash (72 B, NUL-padded)."""
    return blob[12:84].split(b"\0", 1)[0].decode("ascii")


def verify(model_dir: Path, dataset: str = "feat-v1", n: int = 1000) -> dict[str, Any]:
    meta = json.loads((model_dir / "meta.json").read_text())
    arch, role = meta["arch"], meta["role"]
    schema_hash = load_schema().hash
    root = ROOT / "data" / "datasets" / dataset
    rng = np.random.default_rng(0)
    rep: dict[str, Any] = {"model_id": meta["model_id"], "arch": arch, "role": role}
    if arch == "rf":
        import joblib

        from aqi_ml.export import rf_blob
        from aqi_ml.features.dataset import load

        art = model_dir / "model.bin"
        blob = art.read_bytes()
        rep["schema_hash"] = rf_blob_schema_hash(blob)
        va = load(root, role, split="val", holdout=False, seq=False, fraction=0.05, seed=0)
        x = va["X_rf"][rng.choice(len(va["X_rf"]), size=min(n, len(va["X_rf"])), replace=False)]
        forests = joblib.load(model_dir / "forests.joblib")["forests"]  # sklearn = Python reference
        y_ref = np.column_stack([f.predict(x) for f in forests])
        rep["parity_max_abs"] = rf_blob.host_parity(blob, x, y_ref)
        rep["size_ok"] = len(blob) <= RF_BUDGET
        rep["ops_ok"] = True
    else:
        import keras

        from aqi_ml.export import tflite as tl
        from aqi_ml.models.edge.gdc_cnn import train as ct
        from aqi_ml.models.edge.gdc_cnn.model import build

        art = model_dir / "model_fp32.tflite"
        fb = art.read_bytes()
        rep["schema_hash"] = meta["schema_hash"]  # TFLite carries no header; meta travels with it
        va = ct.data(root, role, "val", 0.05, 0)
        x = va["X"][rng.choice(len(va["X"]), size=min(n, len(va["X"])), replace=False)]
        m, _ = build(x.shape[1], x.shape[2], int(meta["D"]))
        m.load_weights(model_dir / "best.weights.h5")
        ref = np.asarray(keras.ops.convert_to_numpy(m(x, training=False)))
        rep["parity_max_abs"] = float(np.max(np.abs(tl.run(fb, x) - ref)))
        ops = tl.op_check(fb)
        rep["ops_ok"], rep["ops"] = bool(ops["ok"]), ops.get("ops")
        rep["size_ok"] = len(fb) <= CNN_BUDGET
        rep["arena_upper_bound"] = tl.arena_estimate(fb)
        rep["arena_note"] = (
            "upper bound (sum of activations); measure arena_used_bytes() on device [S13.4]"
        )
    rep["schema_ok"] = rep["schema_hash"] == schema_hash
    rep["parity_ok"] = rep["parity_max_abs"] < PARITY_TOL
    rep["sha256"], rep["size"], rep["artifact"] = sha256_file(art), art.stat().st_size, art.name
    rep["ok"] = all(rep[k] for k in ("schema_ok", "parity_ok", "size_ok", "ops_ok"))
    (model_dir / "verify.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
