"""Fusion, horizon queries and output heads ([R07.6b]).

- GLU fusion of the spatial/physics state with the node's own temporal summary (WSDST-GAT Eq. 18).
- One query per horizon k = 1..6: position + future weather at t+k (WDDSTG-Net, Wang et al. 2024: weather
  forecasts guide the decoder; here ERA5 "oracle" weather, I19) + the fused node state.
- Non-autoregressive decoder: self-attention over horizons, cross-attention to the node's feat_seq.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    """(h_adr, h_t, feat_seq, X_fut) -> y [B, N, 2H] pollutant-major (+ q10/q90 [B, N, 2H] if quantiles).

    h_adr, h_t: [B, N, D]; feat_seq: [B, N, L, D]; x_fut: [B, N, H, F_fut].
    """

    def __init__(
        self, d: int, heads: int, f_fut: int, horizons: int, dropout: float, quantiles: bool
    ) -> None:
        super().__init__()
        self.fuse = nn.Linear(2 * d, 2 * d)  # GLU halves -> D
        self.pos = nn.Embedding(horizons, d)
        self.fut = nn.Linear(f_fut, d)
        self.layer = nn.TransformerDecoderLayer(
            d, heads, 4 * d, dropout, batch_first=True, norm_first=True
        )
        self.point = nn.Linear(d, 2)
        self.spread = nn.Linear(d, 4) if quantiles else None
        self.h = horizons

    def forward(self, h_adr: torch.Tensor, h_t: torch.Tensor, feat_seq: torch.Tensor,
                x_fut: torch.Tensor) -> dict[str, torch.Tensor]:  # fmt: skip
        b, n, length, d = feat_seq.shape
        z = F.glu(self.fuse(torch.cat([h_adr, h_t], dim=-1)), dim=-1)  # [B, N, D]
        k = torch.arange(self.h, device=z.device)
        q = self.pos(k)[None, None] + self.fut(x_fut) + z[:, :, None, :]  # [B, N, H, D]
        out = self.layer(
            q.reshape(b * n, self.h, d), feat_seq.reshape(b * n, length, d)
        )  # [B*N, H, D]
        out = out.reshape(b, n, self.h, d)

        def pollutant_major(t: torch.Tensor) -> torch.Tensor:  # [B, N, H, 2] -> [B, N, 2H]
            return t.permute(0, 1, 3, 2).reshape(b, n, 2 * self.h)

        y = self.point(out)  # [B, N, H, 2]
        res = {"y": pollutant_major(y)}
        if self.spread is not None:
            s = F.softplus(self.spread(out))  # [B, N, H, 4] >= 0 -> monotone q10 <= y <= q90
            res["q10"] = pollutant_major(y - s[..., :2])
            res["q90"] = pollutant_major(y + s[..., 2:])
        return res
