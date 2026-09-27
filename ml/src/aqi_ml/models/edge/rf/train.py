"""RandomForest students A/B, no_KD ([S13.2]); KD variants reuse fit_forests with blended targets (P8).

    uv run python -m aqi_ml.models.edge.rf.train --role A     # or --role B

Flow: grid search on a train subsample (multi-output forest as a fast proxy) -> keep configs whose
estimated blob fits the flash budget -> best validation RMSE -> final 6 single-output forests on all
training windows -> real blob size check -> host C parity on 1000 test vectors -> [S21] evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestRegressor

from aqi_ml.eval.metrics import aqi_metrics, point_metrics, rmse
from aqi_ml.export import rf_blob
from aqi_ml.features.dataset import ROOT, load
from aqi_shared.schema_loader import load as load_schema

F = npt.NDArray[np.float64]
GRID: dict[str, list[Any]] = {  # [S13.2] hyperparam_search
    "n_estimators": [10, 20, 30],
    "max_depth": [6, 8, 10],
    "min_samples_leaf": [5, 20, 50],
    "max_features": [0.3, 0.5],
}
BUDGET_BYTES = 400_000  # [S13.2] all 6 forests < 400 KB flash
NODE_BYTES = 9


def fit_forests(
    x: npt.NDArray[np.float32], y: F, cfg: dict[str, Any], seed: int
) -> list[RandomForestRegressor]:
    """One RandomForestRegressor per output (simplest export, [S13.2])."""
    return [RandomForestRegressor(**cfg, random_state=seed + j, n_jobs=-1).fit(x, y[:, j])
            for j in range(y.shape[1])]  # fmt: skip


def search(x_tr: npt.NDArray[np.float32], y_tr: F, x_va: npt.NDArray[np.float32], y_va: F,
           seed: int) -> list[dict[str, Any]]:  # fmt: skip
    """Rank the grid by validation RMSE (mean over the 6 log targets) with a size estimate."""
    rows = []
    for vals in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID, vals, strict=True))
        m = RandomForestRegressor(**cfg, random_state=seed, n_jobs=-1).fit(x_tr, y_tr)
        nodes = sum(e.tree_.node_count for e in m.estimators_)
        pred = m.predict(x_va)
        score = float(np.mean([rmse(pred[:, j] - y_va[:, j]) for j in range(y_va.shape[1])]))
        rows.append(
            {**cfg, "val_rmse": score, "est_bytes": nodes * NODE_BYTES * y_tr.shape[1] + 2000}
        )
    return sorted(rows, key=lambda r: r["val_rmse"])


def evaluate(pred: F, d: dict[str, Any], horizons: int, aqi_n: int, seed: int) -> dict[str, Any]:
    """[S21] point metrics + AQI metrics (AQI on a seeded subsample: aqi_trend runs in Python)."""
    out: dict[str, Any] = {
        "n": len(pred),
        "point": point_metrics(d["y"].astype(float), pred, d["pm_base"], horizons),
    }
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(pred), size=min(aqi_n, len(pred)), replace=False)
    out["aqi"] = aqi_metrics(
        d["pm_hist"][sel], d["pm_hist_ok"][sel], d["y"][sel].astype(float), pred[sel], horizons
    )
    return out


def _cached(path: Path, compute: Any) -> Any:
    """Resumable stage: load JSON/joblib result if present, else compute and save it."""
    import joblib

    if path.exists():
        return json.loads(path.read_text("utf-8")) if path.suffix == ".json" else joblib.load(path)
    value = compute()
    if path.suffix == ".json":
        path.write_text(json.dumps(value, indent=1), encoding="utf-8")
    else:
        joblib.dump(value, path)
    return value


def train(
    role: str, version: str, fraction_search: float = 0.25, aqi_n: int = 5000, seed: int = 0
) -> Path:
    """Staged and resumable: each stage's result is cached in data/models/<version>/rf-<role>/."""
    t0 = time.time()
    schema = load_schema()
    root = ROOT / "data" / "datasets" / version
    H = schema.horizons
    out = ROOT / "data" / "models" / version / f"rf-{role}"
    out.mkdir(parents=True, exist_ok=True)
    tr = load(root, role, split="train", holdout=False, seq=False)

    def _search() -> list[dict[str, Any]]:
        va = load(root, role, split="val", holdout=False, seq=False)
        sub = np.random.default_rng(seed).random(len(tr["y"])) < fraction_search
        return search(
            tr["X_rf"][sub], tr["y"][sub].astype(float), va["X_rf"], va["y"].astype(float), seed
        )

    ranking = _cached(out / "search.json", _search)

    def _fit() -> dict[str, Any]:
        for cand in ranking:
            if cand["est_bytes"] > BUDGET_BYTES * 1.1:
                continue
            cfg = {k: cand[k] for k in GRID}
            forests = fit_forests(tr["X_rf"], tr["y"].astype(float), cfg, seed)
            blob = rf_blob.write_blob(forests, tr["X_rf"].shape[1], schema.hash)
            if len(blob) <= BUDGET_BYTES:
                return {"cand": cand, "forests": forests, "blob": blob}
        raise RuntimeError("no configuration fits the flash budget")

    fitted = _cached(out / "forests.joblib", _fit)
    forests, blob, chosen = fitted["forests"], fitted["blob"], fitted["cand"]
    (out / "model.bin").write_bytes(blob)

    results: dict[str, Any] = {}
    for name, split, holdout in (
        ("val", "val", False),
        ("test", "test", False),
        ("test_holdout", "test", True),
    ):

        def _eval(split: str = split, holdout: bool = holdout) -> dict[str, Any]:
            d = load(root, role, split=split, holdout=holdout, seq=False, labels=True)
            pred = np.column_stack([f.predict(d["X_rf"]) for f in forests])
            return {"rf": evaluate(pred, d, H, aqi_n, seed),
                    "persistence": evaluate(np.zeros_like(pred), d, H, aqi_n, seed),
                    "diurnal_persistence": evaluate(d["y_diurnal"], d, H, aqi_n, seed)}  # fmt: skip

        results[name] = _cached(out / f"metrics_{name}.json", _eval)

    def _parity() -> dict[str, float]:
        d = load(root, role, split="test", holdout=False, seq=False)
        sel = np.random.default_rng(seed).choice(len(d["y"]), size=1000, replace=False)
        x = d["X_rf"][sel]
        y = np.column_stack([f.predict(x) for f in forests])
        return {"max_abs": rf_blob.host_parity(blob, x, y)}

    parity = _cached(out / "parity.json", _parity)["max_abs"]
    if parity >= 1e-4:
        raise RuntimeError(f"device parity {parity:.2e} >= 1e-4 ([S13.2] parity_test)")

    meta = {
        "model_id": f"rf-{role}-{time.strftime('%Y-%m-%d')}", "arch": "rf", "role": role,
        "kd_variant": "no_KD", "alpha": 1.0, "schema_hash": schema.hash, "dataset_version": version,
        "config": {k: chosen[k] for k in GRID}, "features": list(schema.rf_names_a if role == "A" else schema.rf_names_b),
        "outputs": list(schema.output_order), "n_train": len(tr["y"]), "size_bytes": len(blob),
        "size_budget_bytes": BUDGET_BYTES, "parity_max_abs": parity, "metrics": results, "seed": seed,
        "train_seconds_last_run": round(time.time() - t0, 1),
        "note": "scores on synthetic data (sim-v1) - not results per [S11.3]",
    }  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aqi_ml.models.edge.rf.train")
    ap.add_argument("--role", choices=["A", "B"], required=True)
    ap.add_argument("--version", default="feat-v1")
    ap.add_argument("--fraction-search", type=float, default=0.25)
    a = ap.parse_args(argv)
    print(f"wrote {train(a.role, a.version, a.fraction_search)}")


if __name__ == "__main__":
    main()
