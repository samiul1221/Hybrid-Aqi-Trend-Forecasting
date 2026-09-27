"""Evaluation metrics [S21]: PM and AQI errors per horizon, skill, category/band-crossing scores,
Diebold-Mariano vs persistence and paired t-tests (Shmuel protocol).

Targets are pollutant-major log changes (I04). PM-space errors use the label PM at the base hour:
PM_hat[t+h] = exp(ln(1+PM[t]) + y_hat) - 1 ([S07] step 1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from aqi_shared import aqi

F = npt.NDArray[np.float64]
POLLUTANTS = ("pm25", "pm10")


def pm_from_y(pm_base: F, y: F) -> F:
    """Vectorised [S07] step 1 (same formula as aqi_shared.aqi.pm_from_y)."""
    out: F = np.exp(np.log1p(pm_base) + y) - 1.0
    return out


def rmse(e: F) -> float:
    return float(np.sqrt(np.mean(np.square(e))))


def diebold_mariano(e_model: F, e_ref: F, lag: int = 24) -> tuple[float, float]:
    """DM test on squared-error loss differential with Newey-West variance (`lag` hours).

    Negative statistic = model better than reference. Windows are consecutive hours, so the default
    lag (24) is larger than the classic h-1 to respect the overlap. Returns (statistic, two-sided p).
    """
    d = np.square(e_model) - np.square(e_ref)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    dc = d - d.mean()
    var = float(np.dot(dc, dc) / n)
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        var += 2.0 * w * float(np.dot(dc[k:], dc[:-k]) / n)
    stat = float(d.mean() / np.sqrt(max(var, 1e-300) / n))
    return stat, float(2.0 * stats.norm.sf(abs(stat)))


def paired_t(a: F, b: F) -> tuple[float, float]:
    """Paired t-test on matched per-run scores (e.g. RMSE per seed x split), Shmuel et al. protocol."""
    res = stats.ttest_rel(a, b)
    return float(res.statistic), float(res.pvalue)


def point_metrics(
    y_true: F, y_pred: F, pm_base: npt.NDArray[np.float64], horizons: int
) -> dict[str, Any]:
    """Per pollutant x horizon: RMSE/MAE in log space and µg/m³, skill vs persistence (y_hat = 0), DM."""
    out: dict[str, Any] = {}
    for p_i, pol in enumerate(POLLUTANTS):
        for h in range(1, horizons + 1):
            j = p_i * horizons + h - 1
            base = pm_base[:, p_i]
            actual = pm_from_y(base, y_true[:, j])
            pred = pm_from_y(base, y_pred[:, j])
            e, e0 = pred - actual, base - actual
            r, r0 = rmse(e), rmse(e0)
            dm, p = diebold_mariano(e, e0)
            out[f"{pol}_h{h}"] = {
                "rmse_log": rmse(y_pred[:, j] - y_true[:, j]), "rmse": r, "mae": float(np.mean(np.abs(e))),
                "rmse_persistence": r0, "skill": 1.0 - r / r0 if r0 > 0 else float("nan"),
                "dm_vs_persistence": dm, "dm_p": p,
            }  # fmt: skip
    return out


def aqi_metrics(pm_hist: F, measured: npt.NDArray[np.bool_], y_true: F, y_pred: F,
                horizons: int) -> dict[str, Any]:  # fmt: skip
    """AQI errors via the device roll ([S07]): MAE, category accuracy and band-crossing POD/FAR/CSI.

    pm_hist: [N x 24 x 2] label PM (pm25, pm10) for the input window; measured: [N x 24] bool.
    Uses aqi_shared.aqi.aqi_trend for both truth and prediction (identical AQI code, [S17] C02).
    """
    n = len(y_true)
    a_true = np.full((n, horizons), np.nan)
    a_pred = np.full((n, horizons), np.nan)
    warn_true = np.zeros(n, bool)
    warn_pred = np.zeros(n, bool)
    for i in range(n):
        p25, p10, m = pm_hist[i, :, 0].tolist(), pm_hist[i, :, 1].tolist(), measured[i].tolist()
        rt = aqi.aqi_trend(p25, p10, m, m, y_true[i].tolist())
        rp = aqi.aqi_trend(p25, p10, m, m, y_pred[i].tolist())
        a_true[i] = [np.nan if v is None else v for v in rt.aqi]
        a_pred[i] = [np.nan if v is None else v for v in rp.aqi]
        warn_true[i], warn_pred[i] = rt.band_warning, rp.band_warning
    out: dict[str, Any] = {}
    cat = np.vectorize(aqi.category_index, otypes=[float])
    for h in range(horizons):
        ok = np.isfinite(a_true[:, h]) & np.isfinite(a_pred[:, h])
        out[f"aqi_h{h + 1}"] = {
            "mae": float(np.mean(np.abs(a_pred[ok, h] - a_true[ok, h]))) if ok.any() else float("nan"),
            "category_accuracy": float(np.mean(cat(a_pred[ok, h]) == cat(a_true[ok, h]))) if ok.any() else float("nan"),
            "n": int(ok.sum()),
        }  # fmt: skip
    hits = int(np.sum(warn_true & warn_pred))
    misses = int(np.sum(warn_true & ~warn_pred))
    false = int(np.sum(~warn_true & warn_pred))
    out["band_crossing"] = {
        "pod": hits / (hits + misses) if hits + misses else float("nan"),
        "far": false / (hits + false) if hits + false else float("nan"),
        "csi": hits / (hits + misses + false) if hits + misses + false else float("nan"),
        "events": hits + misses,
    }
    return out
