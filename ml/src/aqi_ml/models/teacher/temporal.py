"""Node embedding + per-node temporal Transformer ([R07.4]).

Each station's 24 h history is encoded on its own (shared weights): space comes later (gat.py, adr.py), so
`feat_seq` and `attn` are purely temporal and can be distilled into the single-station student (Na et al.
2026, L_feat / L_AFD). Pre-norm encoder with sinusoidal positions as in WSDST-GAT (Wu et al. 2026).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal(length: int, d: int) -> torch.Tensor:
    """Fixed sinusoidal positions (Vaswani et al. 2017). [length, d]"""
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(length, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class Embedding(nn.Module):
    """e = Linear(2F -> D)([X * M, M]) + Linear(F_s -> D)(S).

    Feeding the mask next to the masked value lets the model tell "0 because missing" from "0 measured".
    X, M: [B, N, L, F]; S: [B, N, F_s] -> e: [B, N, L, D].
    """

    def __init__(self, f: int, f_s: int, d: int) -> None:
        super().__init__()
        self.values = nn.Linear(2 * f, d)
        self.static = nn.Linear(f_s, d)

    def forward(self, x: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        e: torch.Tensor = self.values(torch.cat([x * m, m], dim=-1))  # [B, N, L, D]
        st: torch.Tensor = self.static(s)  # [B, N, D]
        return e + st[:, :, None, :]


class EncoderLayer(nn.Module):
    """Pre-norm Transformer layer that also returns per-head attention [*, heads, L, L] for L_AFD.

    nn.TransformerEncoderLayer cannot return attention weights, hence this small copy.
    """

    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        a, w = self.attn(
            h, h, h, need_weights=True, average_attn_weights=False
        )  # w: [B', heads, L, L]
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, w


class Temporal(nn.Module):
    """Stack of encoder layers over the time axis of every node.

    e: [B, N, L, D] -> feat_seq [B, N, L, D] (after a final LayerNorm), attn [B, N, heads, L, L] (last layer).
    """

    pe: torch.Tensor  # registered buffer [L, D]

    def __init__(self, d: int, heads: int, layers: int, dropout: float, length: int = 24) -> None:
        super().__init__()
        self.register_buffer("pe", sinusoidal(length, d), persistent=False)
        self.layers = nn.ModuleList([EncoderLayer(d, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, length, d = e.shape
        x = (e + self.pe[None, None]).reshape(b * n, length, d)  # [B*N, L, D]: nodes share weights
        w = torch.empty(0)
        for layer in self.layers:
            x, w = layer(x)
        feat_seq = self.norm(x).reshape(b, n, length, d)  # [B, N, L, D]
        attn = w.reshape(b, n, -1, length, length)  # [B, N, heads, L, L]
        return feat_seq, attn
