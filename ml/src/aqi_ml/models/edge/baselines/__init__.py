"""Baselines [S13.0]: persistence (y_hat = 0) and diurnal persistence."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

F = npt.NDArray[np.float64]
DIURNAL_DAYS = 14


def persistence(n: int, outputs: int) -> F:
    """Persistence forecast in log-change space: PM[t+h] = PM[t] -> y_hat = 0."""
    return np.zeros((n, outputs))


def diurnal_persistence(pm25: F, pm10: F, ts: pd.Series, horizons: int) -> F:
    """[T x 2H] mean change for that IST hour over the last 14 days, per base hour t ([S13.0]).

    For base t and horizon h: mean over d = 1..14 of ln(1+PM[t-24d+h]) - ln(1+PM[t-24d]); only
    already-observed hours are used (t-24d+h <= t). Requires a contiguous hourly series; NaN-aware
    (no data for that hour in 14 days -> 0, i.e. plain persistence).
    """
    del ts  # the series is contiguous IST hours; same IST hour = 24 rows back
    n = len(pm25)
    out = np.zeros((n, 2 * horizons))
    for p_i, pm in enumerate((pm25, pm10)):
        lp = pd.Series(np.log1p(pm))
        for h in range(1, horizons + 1):
            change = lp.shift(-h) - lp  # change starting at tau, known once tau + h <= t
            acc = np.zeros(n)
            cnt = np.zeros(n)
            for d in range(1, DIURNAL_DAYS + 1):
                v = change.shift(24 * d).to_numpy()
                ok = np.isfinite(v)
                acc[ok] += v[ok]
                cnt[ok] += 1
            out[:, p_i * horizons + h - 1] = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    return out
