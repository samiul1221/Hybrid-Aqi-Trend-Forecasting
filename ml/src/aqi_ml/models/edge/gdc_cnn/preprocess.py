"""CNN preprocessing outside the graph ([S13.4] steps 1, 2, 7) — Python reference for the C twin. ADR 0010.

- instance norm: channels with instnorm=true -> (x - mu) / sqrt(var + eps) over the 24 h window;
- Model B: the 5 [S08.2] extras are appended as constant channels (no instance norm);
- patching: P = 2 consecutive hours -> L = 12 patches of P*C values (hour-major within a patch);
- target scale: y = y_norm * s, s = max(sigma_window(ln_pm), SIGMA_FLOOR) for the pm25/pm10 outputs
  (Model A); Model B has no PM channels, so s = 1.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from aqi_shared.schema_loader import FeatureSchema

F32 = npt.NDArray[np.float32]
PATCH = 2
SIGMA_FLOOR = 0.1  # ln-PM window std floor for de-norm (a flat window would explode y_norm)


def instance_norm(x: F32, norm_mask: npt.NDArray[np.bool_], eps: float) -> tuple[F32, F32]:
    """x [N x T x C] -> (normalised x, sigma [N x C]); channels with norm_mask False pass through."""
    mu = x.mean(axis=1, keepdims=True)
    sd = np.sqrt(x.var(axis=1, keepdims=True) + eps)
    out = np.where(norm_mask[None, None, :], (x - mu) / sd, x).astype(np.float32)
    return out, sd[:, 0, :].astype(np.float32)


def patchify(x: F32, patch: int = PATCH) -> F32:
    """[N x T x C] -> [N x T/patch x patch*C] (hour-major inside a patch)."""
    n, t, c = x.shape
    return x.reshape(n, t // patch, patch * c).astype(np.float32)


def prepare(
    x_seq: F32, schema: FeatureSchema, model: str, x_extra: F32 | None = None
) -> tuple[F32, F32]:
    """Model inputs (patches) and per-output target scale [N x 6] for Model A or B windows."""
    chans = schema.channels if model == "A" else schema.model_b_channels
    norm = np.array([c.instnorm for c in chans])
    eps = float(schema.raw["instnorm_eps"])
    xn, sd = instance_norm(x_seq.astype(np.float32), norm, eps)
    H = schema.horizons
    if model == "A":
        names = [c.name for c in chans]
        s25 = np.maximum(sd[:, names.index("ln_pm25")], SIGMA_FLOOR)
        s10 = np.maximum(sd[:, names.index("ln_pm10")], SIGMA_FLOOR)
        scale = np.concatenate(
            [np.repeat(s25[:, None], H, 1), np.repeat(s10[:, None], H, 1)], axis=1
        )
    else:
        if x_extra is None:
            raise ValueError("Model B needs the [S08.2] extras")
        xn = np.concatenate([xn, np.repeat(x_extra[:, None, :], xn.shape[1], axis=1)], axis=2)
        scale = np.ones((len(xn), 2 * H), dtype=np.float32)
    return patchify(xn.astype(np.float32)), scale.astype(np.float32)
