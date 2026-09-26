"""Training dataset: simulated (or real) node records -> per-node hourly features + window tables.

    uv run python -m aqi_ml.features.dataset          # data/datasets/<version>/

Layout (ADR 0008): node_id=<id>/hourly.parquet (filled channels, masks, labels), windows_a.parquet,
windows_b.parquet, plus manifest.json. X_seq [24 x 23] is not stored per window (it would be ~9 GB);
`load` slices it from hourly.parquet by row index, which is exact and cheap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aqi_ml.features.builder import build_node, windows_a
from aqi_ml.models.edge.model_b.dropout import build_model_b
from aqi_shared.schema_loader import FeatureSchema
from aqi_shared.schema_loader import load as load_schema

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "ml" / "configs" / "features.yaml"


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg


def _node_seed(seed: int, node_id: str) -> int:
    return int(hashlib.sha256(f"{seed}/{node_id}".encode()).hexdigest()[:16], 16)


def build(cfg: dict[str, Any] | None = None, max_nodes: int | None = None) -> Path:
    cfg = cfg or load_config()
    schema = load_schema()
    src = ROOT / "data" / "synthetic" / cfg["source"]
    out = ROOT / "data" / "datasets" / str(cfg["version"])
    sim_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    parts = sorted((src / "records").glob("station_id=*/unit=*/part-0.parquet"))
    if max_nodes:
        parts = parts[:max_nodes]
    counts = []
    for part in parts:
        rec = pd.read_parquet(part)
        node = str(rec.node_id.iloc[0])
        nf = build_node(
            rec, schema, (rec.truth_pm25.to_numpy(float), rec.truth_pm10.to_numpy(float)), cfg
        )
        wa = windows_a(nf, schema)
        wb = build_model_b(nf, schema, np.random.default_rng(_node_seed(cfg["seed"], node)))
        d = out / f"node_id={node}"
        d.mkdir(parents=True, exist_ok=True)
        nf.hourly.to_parquet(d / "hourly.parquet", index=False)
        wa.to_parquet(d / "windows_a.parquet", index=False)
        wb.to_parquet(d / "windows_b.parquet", index=False)
        counts.append({"node_id": node, "hours": len(rec), "windows_a": len(wa), "windows_b": len(wb),
                       "windows_b_outage": int((wb.source == "outage").sum())})  # fmt: skip
        print(f"{node}: {len(wa)} A / {len(wb)} B windows", flush=True)
    manifest = {"version": cfg["version"], "config": cfg, "schema_hash": schema.hash,
                "source": cfg["source"], "sim_manifest": sim_manifest, "nodes": counts}  # fmt: skip
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, default=str), encoding="utf-8"
    )
    return out


def load(root: Path, model: str, split: str | None = None, holdout: bool | None = None,
         schema: FeatureSchema | None = None) -> dict[str, Any]:  # fmt: skip
    """Materialise one model's training arrays.

    Returns X_seq [N x 24 x C] (C = 23 for A, 17 for B), M_seq masks, X_rf [N x 50|32], y [N x 6],
    X_extra [N x 5] (B only) and the window index frame. `split` in {train, val, test}; `holdout`
    filters spatial hold-out stations (True = only hold-out, False = exclude them).
    """
    schema = schema or load_schema()
    L = schema.lookback_hours
    chans = [c.name for c in (schema.channels if model == "A" else schema.model_b_channels)]
    rf_names = schema.rf_names_a if model == "A" else schema.rf_names_b
    xs, ms, frames = [], [], []
    for d in sorted(root.glob("node_id=*")):
        w = pd.read_parquet(d / f"windows_{model.lower()}.parquet")
        if split is not None:
            w = w[w.split == split]
        if holdout is not None:
            w = w[w.spatial_holdout == holdout]
        if w.empty:
            continue
        h = pd.read_parquet(d / "hourly.parquet", columns=chans + [f"m_{c}" for c in chans])
        v, m = h[chans].to_numpy(np.float32), h[[f"m_{c}" for c in chans]].to_numpy(np.uint8)
        idx = w.row.to_numpy()[:, None] + np.arange(-L + 1, 1)[None, :]
        xs.append(v[idx])
        ms.append(m[idx])
        frames.append(w.assign(node_id=d.name.split("=", 1)[1]))
    if not frames:
        raise ValueError("no windows matched")
    win = pd.concat(frames, ignore_index=True)
    out: dict[str, Any] = {
        "X_seq": np.concatenate(xs),
        "M_seq": np.concatenate(ms),
        "X_rf": win[[f"rf_{n}" for n in rf_names]].to_numpy(np.float32),
        "y": win[[f"y_{n}" for n in schema.output_order]].to_numpy(np.float32),
        "index": win[["node_id", "base_ts", "split"] + (["k", "source"] if model == "B" else [])],
    }
    if model == "B":
        out["X_extra"] = win[[f"xb_{n}" for n in schema.model_b_extra]].to_numpy(np.float32)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aqi_ml.features.dataset")
    ap.add_argument("--max-nodes", type=int, default=None)
    a = ap.parse_args(argv)
    print(f"wrote {build(max_nodes=a.max_nodes)}")


if __name__ == "__main__":
    main()
