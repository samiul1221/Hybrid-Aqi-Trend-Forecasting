"""TI-STGNN: Embedding -> Temporal Transformer -> wind-aware GAT -> ADR -> Decoder ([R02], [R07]).

Fixed order (ADR 0013). Variants are booleans: T0 skips the GAT (h_spatial = h_t), T0-T2 skip ADR
(h_adr = h_spatial). Inputs are the edge sensor-proxy features of every station (ADR 0019): the teacher's
advantage is the network (neighbours, wind graph, transport), never cleaner PM.

Batch dict (from train.py, already z-scored where noted):
    X [B, N, L, F] z-scored node features (23 sensor + 7 Open-Meteo), M [B, N, L, F] masks,
    S [B, N, F_s] z-scored static, X_fut [B, N, H, F_fut] z-scored, wind_t [B, N, 2] raw (ws m/s, wd_from deg)
Plus geometry for the current station set: dist, theta [N, N].
Outputs: y [B, N, 12] (+ q10, q90) and the distillation exports feat_seq, attn, h_t, h_spatial, h_adr.
"""

from __future__ import annotations

import torch
from torch import nn

from aqi_ml.models.teacher import graph
from aqi_ml.models.teacher.adr import ADR
from aqi_ml.models.teacher.config import TeacherConfig
from aqi_ml.models.teacher.decoder import Decoder
from aqi_ml.models.teacher.gat import WindGAT
from aqi_ml.models.teacher.temporal import Embedding, Temporal

N_SENSOR = 23  # the student's channels come first in X (graph_windows.window)
MET_SLICE = slice(23, 30)  # Open-Meteo channels at base time feed the ADR reaction term


class TISTGNN(nn.Module):
    def __init__(
        self, cfg: TeacherConfig, f: int = 30, f_s: int = 6, f_fut: int = 9, horizons: int = 6
    ) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.embed = Embedding(f, f_s, d)
        self.temporal = Temporal(d, cfg.heads, cfg.temporal_layers, cfg.dropout)
        self.gat = WindGAT(d, cfg.beta_init, cfg.gat_eps) if cfg.use_graph else None
        self.adr = (
            ADR(d, MET_SLICE.stop - MET_SLICE.start, cfg.adr_steps, cfg.adr_h)
            if cfg.use_adr
            else None
        )
        self.decoder = Decoder(d, cfg.heads, f_fut, horizons, cfg.dropout, cfg.quantiles)

    def prior(
        self, batch: dict[str, torch.Tensor], dist: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """[B, N, N] source -> target prior: wind (T2, T3) or distance only (T1)."""
        b = batch["X"].shape[0]
        if self.cfg.use_wind:
            ws, wd = batch["wind_t"][..., 0], batch["wind_t"][..., 1]  # [B, N]
            return graph.wind_prior(ws, wd, dist, theta, self.cfg.d_min_km, self.cfg.calm_ms)
        return graph.static_prior(dist, self.cfg.d_min_km)[None].expand(b, -1, -1)

    def forward(
        self, batch: dict[str, torch.Tensor], dist: torch.Tensor, theta: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        x, m = batch["X"], batch["M"]
        b, n, _, _ = x.shape
        valid = (m[..., :N_SENSOR].sum(dim=(2, 3)) > 0).float()  # [B, N] node has any sensor data

        e = self.embed(x, m, batch["S"])  # [B, N, L, D]
        feat_seq, attn = self.temporal(e)  # [B, N, L, D], [B, N, heads, L, L]
        h_t = feat_seq[:, :, -1]  # [B, N, D] temporal summary at base_ts (I5)

        h_spatial, logits = h_t, None
        if self.gat is not None:
            h_spatial, logits = self.gat(
                h_t, self.prior(batch, dist, theta), valid
            )  # [B, N, D], [B, N, N]

        h_adr = h_spatial
        if self.adr is not None:
            assert logits is not None
            with torch.no_grad():  # geometry only; no gradient through the eigendecomposition
                v, lam = graph.diffusion_basis(dist, valid, self.cfg.sigma_km)
            met = x[:, :, -1, MET_SLICE]  # [B, N, 7] weather at base time
            h_adr = self.adr(h_spatial, logits, v, lam, feat_seq.mean(dim=2), met, valid)

        out: dict[str, torch.Tensor] = self.decoder(h_adr, h_t, feat_seq, batch["X_fut"])
        assert out["y"].shape == (b, n, 12)
        out.update(
            feat_seq=feat_seq, attn=attn, h_t=h_t, h_spatial=h_spatial, h_adr=h_adr, valid=valid
        )
        return out
