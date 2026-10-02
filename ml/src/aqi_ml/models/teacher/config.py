"""Teacher experiment config: one YAML per variant (ml/configs/teacher/T0..T3.yaml), no hidden defaults ([R03]).

Variant switches are plain booleans ([R07.6c]):
    T0 = temporal only, T1 = + graph (distance prior), T2 = + wind prior, T3 = + ADR physics block.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TeacherConfig:
    name: str
    # variant
    use_graph: bool
    use_wind: bool
    use_adr: bool
    quantiles: bool  # extra q10/q90 heads + pinball loss ([R07.6b])
    # sizes
    d_model: int
    heads: int
    temporal_layers: int
    dropout: float
    # graph / physics ([R07.3]-[R07.6])
    d_min_km: float
    calm_ms: float
    sigma_km: float
    gat_eps: float
    beta_init: float
    adr_steps: int
    adr_h: float
    # data
    graph_version: str
    units: list[int]
    # training ([R07.6c])
    batch: int
    lr: float
    weight_decay: float
    grad_clip: float
    max_epochs: int
    patience: int
    lr_patience: int
    lr_factor: float
    station_dropout: float
    huber_delta: float
    pinball_weight: float
    seed: int
    max_train_samples: int  # 0 = all; > 0 = smoke runs

    def __post_init__(self) -> None:
        if self.use_wind and not self.use_graph:
            raise ValueError("use_wind needs use_graph")
        if self.use_adr and not (self.use_graph and self.use_wind):
            raise ValueError(
                "use_adr (T3) needs the wind graph: its advection reuses the GAT logits (I6)"
            )
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")


def load(path: Path) -> TeacherConfig:
    """Every field must be in the YAML: a missing or unknown key is an error, never a silent default."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    names = {f.name for f in fields(TeacherConfig)}
    missing, unknown = names - raw.keys(), raw.keys() - names
    if missing or unknown:
        raise ValueError(f"{path.name}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    return TeacherConfig(**raw)
