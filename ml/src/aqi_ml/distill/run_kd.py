"""P8 distillation experiments (E2/E3): precondition, RF out_KD (alpha grid), CNN out_KD (lambda grid).

    uv run python -m aqi_ml.distill.run_kd --role A --arch rf     # resumable per (variant, seed)
    uv run python -m aqi_ml.distill.run_kd --role A --arch cnn

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


def run_cnn(role: str, dataset: str, teacher: str, seeds: int) -> Path:
    import keras

    from aqi_ml.models.edge.gdc_cnn import train as ct
    from aqi_ml.models.edge.gdc_cnn.model import build

    schema = load_schema()
    outputs = list(schema.output_order)
    root = ROOT / "data" / "datasets" / dataset
    d_model = json.loads(
        (ROOT / "data" / "models" / dataset / f"cnn-{role}" / "search.json").read_text()
    )[0]["D"]
    out = ROOT / "data" / "models" / dataset / f"kd-cnn-{role}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ct.CFG)
    tr = ct.data(root, role, "train", cfg["train_fraction"], cfg["seed"], labels=role == "B")
    va = ct.data(root, role, "val", 1.0, cfg["seed"], labels=role == "B")
    te = ct.data(root, role, "test", 0.25, cfg["seed"], labels=True)
    y_t_tr, y_t_va = (
        teacher_for(tr, role, teacher, outputs),
        teacher_for(va, role, teacher, outputs),
    )
    gate = precondition_gate(
        dataset, role, "cnn", outputs
    )  # [S12.1]: KD only where the teacher is better
    y_t_tr[:, ~gate] = np.nan  # -> falls back to the true target for those outputs
    y_t_va[:, ~gate] = np.nan
    variants = [("lambda1.0", 1.0)] + [(f"lambda{lam}", lam) for lam in kd.LAMBDAS]
    n = len(outputs)
    for name, lam in variants:
        for seed in range(seeds):
            f = out / f"{name}_seed{seed}.json"
            if f.exists():
                continue
            keras.utils.set_random_seed(100 + seed)
            m, _ = build(tr["X"].shape[1], tr["X"].shape[2], d_model)
            m.compile(
                optimizer=keras.optimizers.Adam(cfg["lr"]),
                loss=lambda yt, yp, lam=lam: kd.out_kd_loss(yt, yp, lam, n),
            )
            ytr = _pack(
                tr, y_t_tr
            )  # [normalised y | normalised teacher]; early stopping uses val RMSE
            best, best_w, wait = 1e9, None, 0
            for _ in range(40):
                m.fit(tr["X"], ytr, batch_size=cfg["batch"], epochs=1, verbose=0)
                v = rmse(
                    (m.predict(va["X"], batch_size=1024, verbose=0) * va["scale"] - va["y"]).ravel()
                )
                if v < best:
                    best, best_w, wait = v, m.get_weights(), 0
                else:
                    wait += 1
                    if wait > 5:
                        break
            m.set_weights(best_w)
            pred = ct.predict(m, te)
            f.write_text(json.dumps(block_rmse(pred, te, outputs)))
    return _summarise(out, [v for v, _ in variants], "lambda1.0", seeds, outputs,
                      {"teacher": teacher, "D": d_model, "kd_outputs": [o for o, g in zip(outputs, gate, strict=True) if g]})  # fmt: skip


def _pack(d: dict[str, Any], y_t: F32) -> F32:
    """out_KD target pack: true and teacher targets, both normalised like the student output."""
    y_teacher = np.where(np.isfinite(y_t), y_t, d["y"])
    return np.concatenate([d["y"] / d["scale"], y_teacher / d["scale"]], axis=1).astype(np.float32)


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
    ap.add_argument("--arch", choices=["rf", "cnn", "precondition"], required=True)
    ap.add_argument("--dataset", default="feat-v1")
    ap.add_argument("--teacher", default="lgbm-interim-feat-v1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fraction", type=float, default=0.5)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.arch == "precondition":
        print(json.dumps(precondition_report(a.dataset, a.role), indent=1))
    elif a.arch == "rf":
        print(run_rf(a.role, a.dataset, a.teacher, a.seeds, a.fraction), f"{time.time() - t0:.0f}s")
    else:
        print(run_cnn(a.role, a.dataset, a.teacher, min(a.seeds, 3)), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
