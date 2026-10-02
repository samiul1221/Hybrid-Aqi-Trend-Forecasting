"""Advection-diffusion-reaction (ADR) block, T3 only ([R07.6]; TransNet, Dimri et al. 2026, Eq. 8-16).

Operator splitting, repeated K times with step h (ADR 0013): advection -> diffusion -> reaction.
The state U is the per-station hidden vector [B, N, D]; the "physics" is in the operator structure
(mass-conserving transport, non-negative symmetric diffusion), not in physical units.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def advection_weights(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """P[b, i, j] = share of source j's state that moves to receiver i (I6).

    Softmax over RECEIVERS i of the GAT logits L[i, j], diagonal masked: every valid source sends a total
    weight of exactly 1 (TransNet's per-source normalisation = mass conservation). Invalid nodes neither
    send nor receive. logits, P: [B, N(i), N(j)]; valid: [B, N].
    """
    n = logits.shape[-1]
    eye = torch.eye(n, device=logits.device, dtype=torch.bool)[None]
    v = valid.bool()
    ok = v[:, :, None] & v[:, None, :] & ~eye  # receiver valid, sender valid, no self
    p = torch.softmax(logits.masked_fill(~ok, float("-inf")), dim=1)  # normalise over receivers
    return torch.nan_to_num(p, nan=0.0)  # a source with no valid receiver sends nothing


def advect(u: torch.Tensor, p: torch.Tensor, h: float) -> torch.Tensor:
    """U_i <- U_i + h (sum_j P_ij U_j - out_i U_i), out_i = sum_k P_ki (1 for a valid source).

    TransNet Eq. 1-3: inflow from upwind sources minus everything a source sends away. Total mass
    sum_i U_i is conserved exactly. u: [B, N, D].
    """
    out = p.sum(dim=1)[:, :, None]  # [B, N, 1] total weight each node sends
    return u + h * (p @ u - out * u)


def diffuse(
    u: torch.Tensor, v: torch.Tensor, lam: torch.Tensor, kappa: torch.Tensor, h: float
) -> torch.Tensor:
    """Exact implicit step U <- (I + h kappa_c L_hat)^-1 U per channel c (TransNet Eq. 13-14).

    With L_hat = V diag(lam) V^T: U <- V diag(1 / (1 + h kappa_c lam)) V^T U. kappa in [0, 1] (Eq. 12):
    no anti-diffusion. u: [B, N, D]; v: [B, N, N]; lam: [B, N]; kappa: [D].
    """
    coef = 1.0 / (1.0 + h * kappa[None, None, :] * lam[:, :, None])  # [B, N(eig), D]
    return v @ (coef * (v.transpose(1, 2) @ u))  # [B, N, D]


class Reaction(nn.Module):
    """f = ReLU(R1(U) + tanh(R2(U)) * U + R3(mean_t feat_seq) + R4(met_t)); U <- U + h f (TransNet Eq. 15-16).

    Local change at each station (emission, chemistry, boundary-layer effects) from its own state, its
    history summary and the current weather. No PBL height: null for Delhi (I17).
    """

    def __init__(self, d: int, f_met: int) -> None:
        super().__init__()
        self.r1 = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.r2 = nn.Linear(d, d)
        self.r3 = nn.Linear(d, d)
        self.r4 = nn.Linear(f_met, d)

    def forward(self, u: torch.Tensor, hist: torch.Tensor, met: torch.Tensor) -> torch.Tensor:
        return F.relu(self.r1(u) + torch.tanh(self.r2(u)) * u + self.r3(hist) + self.r4(met))


class ADR(nn.Module):
    """K steps of advection -> diffusion -> reaction on U = h_spatial.

    u: [B, N, D]; logits: [B, N, N] from the GAT; v, lam: diffusion basis; hist: [B, N, D];
    met: [B, N, F_met]; valid: [B, N]. Returns h_adr [B, N, D].
    """

    def __init__(self, d: int, f_met: int, steps: int, h: float) -> None:
        super().__init__()
        self.kappa_raw = nn.Parameter(torch.full((d,), 0.5))
        self.reaction = Reaction(d, f_met)
        self.steps, self.h = steps, h

    def kappa(self) -> torch.Tensor:
        return F.hardtanh(self.kappa_raw, 0.0, 1.0)  # [D] in [0, 1]

    def forward(self, u: torch.Tensor, logits: torch.Tensor, v: torch.Tensor, lam: torch.Tensor,
                hist: torch.Tensor, met: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:  # fmt: skip
        p = advection_weights(logits, valid)
        for _ in range(self.steps):
            u = advect(u, p, self.h)
            u = diffuse(u, v, lam, self.kappa(), self.h)
            u = u + self.h * self.reaction(u, hist, met)
        return u
