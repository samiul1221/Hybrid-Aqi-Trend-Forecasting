"""sim_params.yaml loading and per-unit parameter draws (domain randomisation, [S10.2]/[S10.3])."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
SIM_PARAMS = ROOT / "ml" / "configs" / "sim_params.yaml"
SENSORS = ("sps30", "hdc1080", "bmp280", "wind", "rain", "bh1750", "mq7", "mq135", "sgp41")


def load(path: Path = SIM_PARAMS) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    cfg["_hash"] = (
        "sha256:"
        + hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    )
    return cfg


def is_dist(v: Any) -> bool:
    return isinstance(v, dict) and len(v) == 1 and next(iter(v)) in ("u", "ln", "const")


def draw(v: Any, rng: np.random.Generator) -> Any:
    """One draw from a {u: [lo, hi]} / {ln: [median, sigma]} / {const: x} spec; scalars pass through."""
    if not is_dist(v):
        return v
    ((kind, arg),) = v.items()
    if kind == "u":
        return float(rng.uniform(arg[0], arg[1]))
    if kind == "ln":
        return float(arg[0] * np.exp(rng.normal(0.0, arg[1])))
    return arg


def draw_unit(cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, dict[str, Any]]:
    """All sensor parameters for one virtual unit, drawn once for its lifetime ([S10.2])."""
    return {s: {k: draw(v, rng) for k, v in cfg[s].items()} for s in SENSORS}


def unit_rng(seed: int, station_id: str, unit: int) -> np.random.Generator:
    """Deterministic per-unit generator (stable across runs and station order)."""
    h = int(hashlib.sha256(f"{seed}/{station_id}/{unit}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(h)
