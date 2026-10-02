"""One wind-biased graph attention layer ([R07.5]; WSDST-GAT edge-biased attention, Wu et al. 2026).

Receiver i attends to senders j. The physics prior enters as an additive bias beta * log(w + eps), so a pair
the wind does not connect gets a large negative bias but is not hard-forbidden (the data can still use it).
alpha is a learned weighting nudged by physics, NOT a transport coefficient (spec §10).

I26: graph.py stores w[i, j] as SOURCE i -> TARGET j (TransNet). For receiver i and sender j the relevant
edge is j -> i, i.e. w[j, i]: we bias with the TRANSPOSE. Using w[i, j] would favour downwind senders.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WindGAT(nn.Module):
    """h_spatial = h_t + W_o sum_j alpha_ij W h_j.

    h_t: [B, N, D]; prior: [B, N, N] (source -> target) or None (no bias); valid: [B, N].
    Returns h_spatial [B, N, D] and logits L [B, N, N] (receiver x sender; reused by the ADR advection, I6).
    """

    def __init__(self, d: int, beta_init: float, eps: float) -> None:
        super().__init__()
        self.w = nn.Linear(d, d, bias=False)
        self.a_recv, self.a_send = nn.Linear(d, 1, bias=False), nn.Linear(d, 1, bias=False)
        self.out = nn.Linear(d, d)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.eps = eps

    def logits(self, h_t: torch.Tensor, prior: torch.Tensor | None) -> torch.Tensor:
        wh = self.w(h_t)  # [B, N, D]
        # a^T [W h_i || W h_j] split into a receiver and a sender part -> dense [B, N(i), N(j)]
        score = F.leaky_relu(self.a_recv(wh) + self.a_send(wh).transpose(1, 2), 0.2)  # [B, N, N]
        if prior is None:
            return score
        incoming = prior.transpose(1, 2)  # [B, i, j] = w[j -> i]  (I26)
        n = incoming.shape[-1]
        eye = torch.eye(n, device=incoming.device, dtype=torch.bool)
        self_w = incoming.amax(dim=-1, keepdim=True)  # self-loop weight = strongest incoming edge
        incoming = torch.where(eye[None], self_w.expand_as(incoming), incoming)
        return score + self.beta * torch.log(incoming + self.eps)

    def forward(
        self, h_t: torch.Tensor, prior: torch.Tensor | None, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, d = h_t.shape
        lg = self.logits(h_t, prior)  # [B, N, N]
        eye = torch.eye(n, device=h_t.device, dtype=torch.bool)[None]
        send_ok = (
            valid[:, None, :].bool() | eye
        )  # an invalid sender sends nothing; self always allowed
        alpha = torch.softmax(
            lg.masked_fill(~send_ok, float("-inf")), dim=-1
        )  # [B, N, N], rows sum to 1
        h = h_t + self.out(alpha @ self.w(h_t))  # [B, N, D]
        assert h.shape == (b, n, d)
        return h, lg
