"""Graph geometry and priors ([R07.3], ADR 0013). Dense [N, N] matrices; N ≈ 30 stations.

Index convention used everywhere in the teacher: **w[i, j] = weight of the edge SOURCE i -> TARGET j**
(TransNet). A receiver i therefore reads its incoming prior from column i: w[:, i] (see gat.py, I26).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

EARTH_R_KM = 6371.0


def geometry(
    lat: npt.NDArray[np.float64], lon: npt.NDArray[np.float64]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Great-circle distance d[i, j] (km) and initial bearing theta[i, j] from i to j (deg, 0 = north)."""
    la, lo = np.deg2rad(lat)[:, None], np.deg2rad(lon)[:, None]
    dla, dlo = la.T - la, lo.T - lo
    a = np.sin(dla / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlo / 2) ** 2
    d = 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # [N, N]
    y = np.sin(dlo) * np.cos(la.T)
    x = np.cos(la) * np.sin(la.T) - np.sin(la) * np.cos(la.T) * np.cos(dlo)
    theta = np.rad2deg(np.arctan2(y, x)) % 360.0  # [N, N]
    return torch.tensor(d, dtype=torch.float32), torch.tensor(theta, dtype=torch.float32)


def wind_prior(ws: torch.Tensor, wd_from: torch.Tensor, dist: torch.Tensor, theta: torch.Tensor,
               d_min: float, calm: float) -> torch.Tensor:  # fmt: skip
    """TransNet-style transport prior w[b, i, j] = ReLU(ws_i cos(theta_ij - phi_i)) / max(d_ij, d_min).

    ws, wd_from: [B, N] wind at the SOURCE station (m/s, meteorological FROM direction).
    phi_i = wd_from + 180: the direction the air moves TOWARD. TransNet puts the FROM direction straight into
    the cosine, which points edges upwind (I4, ADR 0013); we correct it so a downwind j gets the weight.
    Calm air (ws < calm) carries nothing: the whole row is 0. Diagonal = 0 (self-loop handled in gat.py).
    """
    phi = (wd_from + 180.0) % 360.0  # [B, N]
    ang = torch.deg2rad(theta[None] - phi[:, :, None])  # [B, N, N]
    w = torch.relu(ws[:, :, None] * torch.cos(ang)) / dist.clamp_min(d_min)[None]  # [B, N, N]
    w = w * (ws >= calm).float()[:, :, None]
    return w * (1.0 - torch.eye(dist.shape[0], device=w.device))[None]


def static_prior(dist: torch.Tensor, d_min: float) -> torch.Tensor:
    """T1 prior without wind: w[i, j] = 1 / max(d_ij, d_min), symmetric, zero diagonal. [N, N]"""
    w = 1.0 / dist.clamp_min(d_min)
    return w * (1.0 - torch.eye(dist.shape[0], device=w.device))


def diffusion_basis(
    dist: torch.Tensor, valid: torch.Tensor, sigma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigen-decomposition L_hat = V diag(lam) V^T of the normalised Laplacian of A = exp(-d^2 / sigma^2).

    valid: [B, N] node mask; invalid nodes are cut out of the graph (no diffusion to or from them) and get
    an all-zero Laplacian row, so the implicit step leaves them unchanged. Static graph (I20): a symmetric
    matrix gives an exact implicit step via eigh, no linear solver (TransNet Eq. 13-14).
    Returns V [B, N, N], lam [B, N] with lam in [0, 2].
    """
    n = dist.shape[0]
    a = torch.exp(-(dist**2) / sigma**2) * (1.0 - torch.eye(n, device=dist.device))  # [N, N]
    a = a[None] * valid[:, :, None] * valid[:, None, :]  # [B, N, N]
    deg = a.sum(-1)  # [B, N]
    has = (deg > 0).float()
    dinv = torch.where(deg > 0, deg.clamp_min(1e-12).rsqrt(), torch.zeros_like(deg))
    lap = torch.diag_embed(has) - dinv[:, :, None] * a * dinv[:, None, :]  # [B, N, N]
    lam, v = torch.linalg.eigh(lap)
    return v, lam.clamp(0.0, 2.0)
