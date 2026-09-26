"""feature_schema.yaml loader, validation, RF feature expansion and schema hash [S08], I05.

Schema hash = "sha256:" + sha256 of canonical JSON (sorted keys, no whitespace, UTF-8) of the
parsed YAML. It depends only on content, not on formatting, comments or line endings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "schema" / "feature_schema.yaml"

OPS = frozenset(
    {"identity", "log1p", "clip", "offset", "scale", "log_ratio", "base_diff_scale",
     "cyc_sin", "cyc_cos", "measured_mask"}
)  # fmt: skip
RF_KINDS = ("lags", "roll_mean", "roll_std", "diff", "rain_hours_last")
_RF_PREFIX = {"lags": "lag", "roll_mean": "rmean", "roll_std": "rstd", "diff": "diff",
              "rain_hours_last": "rainh"}  # fmt: skip


@dataclass(frozen=True)
class Channel:
    i: int
    name: str
    src: str
    op: str
    op_args: dict[str, Any]
    instnorm: bool
    model_b: bool


@dataclass(frozen=True)
class RfFeature:
    name: str
    src: str
    kind: str  # one of RF_KINDS
    k: int
    pm: bool  # PM-derived: dropped for Model B


@dataclass(frozen=True)
class FeatureSchema:
    version: int
    lookback_hours: int
    horizons: int
    output_order: tuple[str, ...]
    channels: tuple[Channel, ...]
    rf_features: tuple[RfFeature, ...]  # Model A order
    model_b_extra: tuple[str, ...]
    unit_baselines: dict[str, dict[str, Any]]
    hash: str
    raw: dict[str, Any]

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)

    @property
    def model_b_channels(self) -> tuple[Channel, ...]:
        return tuple(c for c in self.channels if c.model_b)

    @property
    def rf_names_a(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.rf_features)

    @property
    def rf_names_b(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.rf_features if not f.pm) + self.model_b_extra


def canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _parse_channel(d: dict[str, Any]) -> Channel:
    (op, args), *extra = d["op"].items()
    if extra or op not in OPS:
        raise ValueError(f"channel {d['name']}: op must be one of {sorted(OPS)}")
    return Channel(d["i"], d["name"], d["src"], op, dict(args or {}), d["instnorm"], d["model_b"])


def _expand_rf(groups: list[dict[str, Any]], known_src: set[str]) -> tuple[RfFeature, ...]:
    out: list[RfFeature] = []
    for g in groups:
        if g["src"] not in known_src:
            raise ValueError(f"rf_features: unknown src {g['src']!r}")
        for kind in RF_KINDS:
            for k in g.get(kind, []):
                name = f"{g['src']}_{_RF_PREFIX[kind]}{k}"
                out.append(RfFeature(name, g["src"], kind, int(k), bool(g.get("pm", False))))
    return tuple(out)


def parse(raw: dict[str, Any]) -> FeatureSchema:
    channels = tuple(_parse_channel(c) for c in raw["channels"])
    if [c.i for c in channels] != list(range(len(channels))):
        raise ValueError("channel indices must be 0..N-1 in order")
    known = {c.name for c in channels} | set(raw["rf_aux_sources"])
    rf = _expand_rf(raw["rf_features"], known)
    extra = tuple(e["name"] for e in raw["model_b_extra"])
    names = [c.name for c in channels] + [f.name for f in rf] + list(extra)
    if len(names) != len(set(names)):
        raise ValueError("duplicate feature names")
    horizons = int(raw["horizons"])
    if len(raw["output_order"]) != 2 * horizons:
        raise ValueError("output_order must list pm25 and pm10 for every horizon")
    return FeatureSchema(
        version=int(raw["version"]),
        lookback_hours=int(raw["lookback_hours"]),
        horizons=horizons,
        output_order=tuple(raw["output_order"]),
        channels=channels,
        rf_features=rf,
        model_b_extra=extra,
        unit_baselines=dict(raw["unit_baselines"]),
        hash=canonical_hash(raw),
        raw=raw,
    )


@cache
def load(path: Path = DEFAULT_PATH) -> FeatureSchema:
    """Load and validate the feature schema (cached per path)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return parse(raw)
