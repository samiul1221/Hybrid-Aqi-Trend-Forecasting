"""FROZEN v1 baseline (ADR 0016): precondition + RF out_KD (alpha grid) with the LightGBM teacher.

    uv run python -m aqi_ml.distill.run_kd --role A --arch rf     # resumable per (variant, seed)

The LightGBM teacher sees clean neighbour CPCB PM ("clean-neighbour teacher", ADR 0019), so these numbers
are not comparable to the v2 graph teachers. CNN out_KD with it was removed; results: docs/research_notes/e2_kd.md.
v2 student KD (S0-S4) lives in distill/variants.py (CR v2 [R08]).

Protocol (Shmuel et al. adapted to time series, [S13.1]/[S13.2]): the no-KD configuration is kept fixed;
each variant is retrained for several seeds; test windows are split into 5 contiguous time blocks; the
per-run RMSE (seed x block) of KD vs no-KD is compared with a paired t-test per horizon. Scores are on
synthetic data (not results, [S11.3]). full_KD needs a teacher with feat_seq/attn (graph teacher,
phase 2); with the interim LightGBM teacher it reduces to out_KD ([S12.2]).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from aqi_ml.distill import kd
from aqi_ml.distill import teacher_store as ts
from aqi_ml.eval.metrics import paired_t, pm_from_y, rmse
from aqi_ml.features.dataset import ROOT, load
from aqi_shared.schema_loader import load as load_schema

F32 = npt.NDArray[np.float32]
BLOCKS = 5


def teacher_for(d: dict[str, Any], role: str, version: str, outputs: list[str]) -> F32:
    """y_teacher aligned to the windows; Model B re-expressed relative to the last valid PM ([S14])."""
    idx = d["index"].copy()
    store = ts.read(
        version, node_ids=sorted(idx.node_id.unique()), columns=["base_ts", *ts.y_columns(outputs)]
    )
    y_t = np.array(
        ts.join(idx, store, outputs), dtype=np.float32
    )  # writable copy (gating edits it)
    if role == "B":
        # teacher forecasts from base t; label PM at t and at the last valid hour t - k
        pm_t = np.stack([_label_at(d, 0)[:, 0], _label_at(d, 0)[:, 1]], axis=1)
        y_t = kd.model_b_teacher(y_t, np.log1p(pm_t), np.log1p(d["pm_base"]).astype(np.float32))
    return y_t


def _label_at(d: dict[str, Any], lag: int) -> npt.NDArray[np.float64]:
    out: npt.NDArray[np.float64] = d["pm_hist"][:, -1 - lag, :]
    return out


def block_rmse(pred: F32, d: dict[str, Any], outputs: list[str]) -> dict[str, list[float]]:
    """RMSE (µg/m³) per output for each of 5 contiguous test time blocks."""
    order = np.argsort(pd.to_datetime(d["index"].base_ts).to_numpy())
    blocks = np.array_split(order, BLOCKS)
    out: dict[str, list[float]] = {}
    H = len(outputs) // 2
    for j, name in enumerate(outputs):
        base = d["pm_base"][:, j // H]
        out[name] = [rmse(pm_from_y(base[b], pred[b, j].astype(float)) - pm_from_y(base[b], d["y"][b, j].astype(float)))
                     for b in blocks]  # fmt: skip
    return out


def run_rf(role: str, dataset: str, teacher: str, seeds: int, fraction: float) -> Path:
    from aqi_ml.models.edge.rf.train import fit_forests

    schema = load_schema()
    outputs = list(schema.output_order)
    root = ROOT / "data" / "datasets" / dataset
    base_dir = ROOT / "data" / "models" / dataset / f"rf-{role}"
    cfg = json.loads((base_dir / "meta.json").read_text())["config"]
    out = ROOT / "data" / "models" / dataset / f"kd-rf-{role}"
    out.mkdir(parents=True, exist_ok=True)
    tr = load(
        root,
        role,
        split="train",
        holdout=False,
        seq=False,
        labels=role == "B",
        fraction=fraction,
        seed=0,
    )
    y_t_tr = teacher_for(tr, role, teacher, outputs)
    y_t_tr[:, ~precondition_gate(dataset, role, "rf", outputs)] = np.nan  # [S12.1] gate
    te = load(root, role, split="test", holdout=False, seq=False, labels=True)
    for alpha in kd.ALPHAS:
        for seed in range(seeds):
            f = out / f"alpha{alpha}_seed{seed}.json"
            if f.exists():
                continue
            y_fit = kd.rf_targets(tr["y"], y_t_tr, alpha)
            forests = fit_forests(tr["X_rf"], y_fit.astype(float), cfg, 1000 + seed)
            pred = np.column_stack([m.predict(te["X_rf"]) for m in forests]).astype(np.float32)
            f.write_text(json.dumps(block_rmse(pred, te, outputs)))
    return _summarise(out, [f"alpha{a}" for a in kd.ALPHAS], "alpha1.0", seeds, outputs,
                      {"teacher": teacher, "config": cfg, "train_fraction": fraction,
                       "teacher_coverage_train": float(np.isfinite(y_t_tr).all(axis=1).mean())})  # fmt: skip


def _summarise(
    out: Path, variants: list[str], ref: str, seeds: int, outputs: list[str], extra: dict[str, Any]
) -> Path:
    runs = {
        v: [json.loads((out / f"{v}_seed{s}.json").read_text()) for s in range(seeds)]
        for v in variants
    }
    summary: dict[str, Any] = {
        "protocol": f"{seeds} seeds x {BLOCKS} test time blocks, paired t-test vs {ref}",
        **extra,
    }
    for v in variants:
        per = {}
        for name in outputs:
            a = np.array([r[name] for r in runs[v]]).ravel()
            b = np.array([r[name] for r in runs[ref]]).ravel()
            t, p = paired_t(a, b) if v != ref else (float("nan"), float("nan"))
            per[name] = {"rmse_mean": float(a.mean()), "delta_vs_ref_pct": float(100 * (a.mean() - b.mean()) / b.mean()),
                         "t": t, "p": p}  # fmt: skip
        summary[v] = per
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return out


def precondition_gate(
    dataset: str, role: str, arch: str, outputs: list[str]
) -> npt.NDArray[np.bool_]:
    """Per-output KD switch from the [S12.1] precondition (True = teacher beats the no-KD student)."""
    rep = precondition_report(dataset, role)[arch]["per_output"]
    return np.array([rep[o]["pass"] for o in outputs])


def precondition_report(dataset: str, role: str) -> dict[str, Any]:
    """[S12.1]/E3: teacher vs no-KD student RMSE (log space) on validation, per output."""
    outputs = list(load_schema().output_order)
    t_meta = json.loads(
        (ROOT / "data" / "models" / dataset / "teacher-lgbm" / "meta.json").read_text()
    )
    rep: dict[str, Any] = {}
    for arch, key in (("rf", "rf"), ("cnn", "cnn")):
        f = ROOT / "data" / "models" / dataset / f"{arch}-{role}" / "meta.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())["metrics"]["val"][key]["point"]
        rep[arch] = kd.precondition({n: t_meta["rmse_log"]["val"][n] for n in outputs},
                                    {n: s[n]["rmse_log"] for n in outputs})  # fmt: skip
    return rep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aqi_ml.distill.run_kd")
    ap.add_argument("--role", choices=["A", "B"], default="A")
    ap.add_argument("--arch", choices=["rf", "precondition"], required=True)
    ap.add_argument("--dataset", default="feat-v1")
    ap.add_argument("--teacher", default="lgbm-interim-feat-v1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fraction", type=float, default=0.5)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.arch == "precondition":
        print(json.dumps(precondition_report(a.dataset, a.role), indent=1))
    else:
        print(run_rf(a.role, a.dataset, a.teacher, a.seeds, a.fraction), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
