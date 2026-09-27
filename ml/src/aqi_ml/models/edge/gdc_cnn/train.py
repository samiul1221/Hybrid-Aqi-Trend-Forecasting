"""GDC-CNN students A/B ([S13.4]) — no_KD now; KD losses plug in via `loss_fn` (P8).

    uv run python -m aqi_ml.models.edge.gdc_cnn.train --role A      # resumable; rerun to continue

Stages (cached in data/models/<dataset>/cnn-<role>/): D search on a subsample (short runs) ->
final training with early stopping (checkpoint + state every epoch, so a stopped run resumes) ->
[S21] evaluation -> TFLite FP32 export + TFLM op check + host parity (+ static INT8 ablation, E7).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from aqi_ml.eval.metrics import aqi_metrics, point_metrics, rmse
from aqi_ml.export import tflite as tl
from aqi_ml.features.dataset import ROOT, load
from aqi_ml.models.edge.gdc_cnn.model import build, export_model
from aqi_ml.models.edge.gdc_cnn.preprocess import prepare
from aqi_shared.schema_loader import load as load_schema

F32 = npt.NDArray[np.float32]
D_SEARCH = (16, 32, 64)  # [S13.4] D_search
CFG: dict[str, Any] = {"lr": 1e-3, "batch": 128, "max_epochs": 100, "patience": 10, "huber_delta": 1.0,  # [S13.4] pretrain
       "search_epochs": 6, "search_fraction": 0.15, "train_fraction": 0.5, "seed": 0,
       "budget_s": 520}  # fmt: skip
SIZE_BUDGET, ARENA_BUDGET = 300_000, 128_000  # cloud [C09] verification budgets


def data(root: Path, role: str, split: str, fraction: float, seed: int, labels: bool = False,
         holdout: bool | None = False) -> dict[str, Any]:  # fmt: skip
    schema = load_schema()
    d = load(
        root,
        role,
        split=split,
        holdout=holdout,
        seq=True,
        labels=labels,
        fraction=fraction,
        seed=seed,
    )
    d["X"], d["scale"] = prepare(d.pop("X_seq"), schema, role, d.get("X_extra"))
    d.pop("M_seq", None)
    return d


def make_model(d: dict[str, Any], d_model: int, lr: float, delta: float) -> keras.Model:
    m, _ = build(d["X"].shape[1], d["X"].shape[2], d_model)
    m.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.Huber(delta=delta))
    return m


def predict(m: keras.Model, d: dict[str, Any]) -> F32:
    """De-normalised log-change forecasts ([S13.4] step 7)."""
    out: F32 = np.asarray(
        m.predict(d["X"], batch_size=1024, verbose=0) * d["scale"], dtype=np.float32
    )
    return out


def fit_resumable(
    m: keras.Model, tr: dict[str, Any], va: dict[str, Any], out: Path, cfg: dict[str, Any]
) -> bool:
    """Epoch loop with early stopping; state + best weights saved every epoch. True when finished."""
    state_f, best_f = out / "state.json", out / "best.weights.h5"
    st: dict[str, Any] = (
        json.loads(state_f.read_text())
        if state_f.exists()
        else {"epoch": 0, "best": 1e9, "best_epoch": -1, "history": []}
    )
    if (out / "last.weights.h5").exists():
        m.load_weights(out / "last.weights.h5")
    y_tr = tr["y"] / tr["scale"]
    y_va = va["y"] / va["scale"]
    t0 = time.time()
    while st["epoch"] < cfg["max_epochs"] and st["epoch"] - st["best_epoch"] <= cfg["patience"]:
        h = m.fit(tr["X"], y_tr, batch_size=cfg["batch"], epochs=1, verbose=0, shuffle=True)
        val = float(m.evaluate(va["X"], y_va, batch_size=1024, verbose=0))
        st["history"].append(
            {"epoch": st["epoch"], "loss": float(h.history["loss"][0]), "val_loss": val}
        )
        if val < st["best"]:
            st["best"], st["best_epoch"] = val, st["epoch"]
            m.save_weights(best_f)
        st["epoch"] += 1
        m.save_weights(out / "last.weights.h5")
        state_f.write_text(json.dumps(st, indent=1))
        if time.time() - t0 > cfg["budget_s"]:
            return False
    m.load_weights(best_f)
    return True


def evaluate(pred: F32, d: dict[str, Any], seed: int, aqi_n: int = 5000) -> dict[str, Any]:
    H = load_schema().horizons
    sel = np.random.default_rng(seed).choice(len(pred), size=min(aqi_n, len(pred)), replace=False)
    return {"n": len(pred), "point": point_metrics(d["y"].astype(float), pred.astype(float), d["pm_base"], H),
            "aqi": aqi_metrics(d["pm_hist"][sel], d["pm_hist_ok"][sel], d["y"][sel].astype(float),
                               pred[sel].astype(float), H)}  # fmt: skip


def train(role: str, version: str, cfg: dict[str, Any] = CFG) -> tuple[Path, bool]:
    keras.utils.set_random_seed(cfg["seed"])
    root = ROOT / "data" / "datasets" / version
    out = ROOT / "data" / "models" / version / f"cnn-{role}"
    out.mkdir(parents=True, exist_ok=True)
    va = data(root, role, "val", 1.0, cfg["seed"])

    search_f = out / "search.json"
    if not search_f.exists():
        tr_s = data(root, role, "train", cfg["search_fraction"], cfg["seed"])
        res = []
        for d_model in D_SEARCH:
            m = make_model(tr_s, d_model, cfg["lr"], cfg["huber_delta"])
            m.fit(
                tr_s["X"],
                tr_s["y"] / tr_s["scale"],
                batch_size=cfg["batch"],
                epochs=cfg["search_epochs"],
                verbose=0,
            )
            r = float(np.mean([rmse(c) for c in (predict(m, va) - va["y"]).T]))
            res.append({"D": d_model, "val_rmse_log": r, "params": m.count_params()})
        search_f.write_text(json.dumps(sorted(res, key=lambda r: r["val_rmse_log"]), indent=1))
        return out, False  # continue in the next run (keeps each run inside the time budget)
    best_d = json.loads(search_f.read_text())[0]["D"]

    tr = data(root, role, "train", cfg["train_fraction"], cfg["seed"])
    m = make_model(tr, best_d, cfg["lr"], cfg["huber_delta"])
    if not (out / "done.json").exists():
        if not fit_resumable(m, tr, va, out, cfg):
            return out, False
        (out / "done.json").write_text(json.dumps({"D": best_d}))
    else:
        m.load_weights(out / "best.weights.h5")
    del tr

    results = {}
    for name, split, holdout in (
        ("val", "val", False),
        ("test", "test", False),
        ("test_holdout", "test", True),
    ):
        f = out / f"metrics_{name}.json"
        if not f.exists():
            d = data(
                root,
                role,
                split,
                1.0 if split == "val" else 0.25,
                cfg["seed"],
                labels=True,
                holdout=holdout,
            )
            pred = predict(m, d)
            f.write_text(json.dumps({"cnn": evaluate(pred, d, cfg["seed"]),
                                     "persistence": evaluate(np.zeros_like(pred), d, cfg["seed"])}, indent=1))  # fmt: skip
        results[name] = json.loads(f.read_text())

    # export: FP32 (default) + static INT8 ablation (E7); op check, host parity, size/arena
    e = export_model(m, va["X"].shape[1], va["X"].shape[2], best_d)
    fp32 = tl.convert(e)
    int8 = tl.convert(e, int8_rep=va["X"][:500])
    (out / "model_fp32.tflite").write_bytes(fp32)
    (out / "model_int8.tflite").write_bytes(int8)
    sel = np.random.default_rng(cfg["seed"]).choice(
        len(va["X"]), size=min(1000, len(va["X"])), replace=False
    )
    ref = m.predict(va["X"][sel], verbose=0)
    lite = tl.run(fp32, va["X"][sel])
    lite8 = tl.run(int8, va["X"][sel])
    parity = float(np.max(np.abs(lite - ref)))
    if parity >= 1e-4:
        raise RuntimeError(f"TFLite FP32 parity {parity:.2e} >= 1e-4")
    ops = tl.op_check(fp32)
    if not ops["ok"]:
        raise RuntimeError(f"ops not supported by TFLM: {ops['unsupported_in_tflm']}")
    y_va = va["y"][sel]
    meta = {
        "model_id": f"cnn-{role}-{time.strftime('%Y-%m-%d')}", "arch": "cnn", "role": role, "kd_variant": "no_KD",
        "schema_hash": load_schema().hash, "dataset_version": version, "D": best_d, "params": m.count_params(),
        "input": {"patches": va["X"].shape[1], "patch_dim": va["X"].shape[2]}, "precision": "fp32",
        "size_bytes": len(fp32), "size_budget": SIZE_BUDGET, "arena_upper_bound": tl.arena_estimate(fp32),
        "arena_budget": ARENA_BUDGET, "tflm_ops": ops, "host_parity_fp32": parity,
        "int8_ablation": {"size_bytes": len(int8), "max_abs_vs_fp32": float(np.max(np.abs(lite8 - ref))),
                          "val_rmse_log_fp32": rmse((lite * va["scale"][sel] - y_va).ravel()),
                          "val_rmse_log_int8": rmse((lite8 * va["scale"][sel] - y_va).ravel())},
        "config": cfg, "metrics": results, "history": json.loads((out / "state.json").read_text())["history"],
        "note": "scores on synthetic data (sim-v1) - not results per [S11.3]; device latency/arena: measure on ESP32 (P10)",
    }  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out, True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aqi_ml.models.edge.gdc_cnn.train")
    ap.add_argument("--role", choices=["A", "B"], required=True)
    ap.add_argument("--version", default="feat-v1")
    a = ap.parse_args(argv)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    out, done = train(a.role, a.version)
    print(("done " if done else "partial (rerun to continue) ") + str(out))


if __name__ == "__main__":
    main()
