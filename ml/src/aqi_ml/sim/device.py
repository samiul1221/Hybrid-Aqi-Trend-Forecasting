"""On-device hourly processing applied to simulated readings ([S10.3] post_simulation, [S06], [S14]).

Uses the shared pipeline and CPCB AQI code, so simulated and real records are processed identically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from aqi_shared import aqi
from aqi_shared import pipeline as pl

F = npt.NDArray[np.float64]
B = npt.NDArray[np.bool_]


def lost_streak(measured: B) -> npt.NDArray[np.int64]:
    """Hours since PM was last measured (0 when measured this hour)."""
    m = pd.Series(measured)
    grp = m.cumsum()
    streak = (~m).groupby(grp).cumsum()
    out: npt.NDArray[np.int64] = np.asarray(streak, dtype=np.int64)
    return out


def pm_state(streak: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
    """[S14] runtime policy: ok | filled_le_1h (<=1 h) | fallback_model_b (<=24 h) | unavailable."""
    out: npt.NDArray[np.str_] = np.select(
        [streak == 0, streak <= 1, streak <= 24],
        ["ok", "filled_le_1h", "fallback_model_b"],
        "unavailable",
    )
    return out


def rolling_aqi(pm25: F, pm10: F, m25: B, m10: B) -> dict[str, npt.NDArray[Any]]:
    """24 h CPCB AQI per hour (window ends at that hour), identical to aqi.aqi_pm_24h."""
    w = aqi.WINDOW_HOURS

    def mean_count(v: F, m: B) -> tuple[F, F]:
        x = pd.Series(np.where(m & ~np.isnan(v), v, np.nan))
        return x.rolling(w, min_periods=1).mean().to_numpy(), x.rolling(
            w, min_periods=1
        ).count().to_numpy()

    mean25, n25 = mean_count(pm25, m25)
    mean10, n10 = mean_count(pm10, m10)
    n = np.minimum(n25, n10)
    valid = n >= aqi.MIN_MEASURED_HOURS
    sub = np.vectorize(aqi.sub_index, otypes=[float])
    val = np.full(len(pm25), np.nan)
    if valid.any():
        val[valid] = np.maximum(sub("pm25", mean25[valid]), sub("pm10", mean10[valid]))
    aqi_int = np.where(valid, np.floor(val + 0.5), np.nan)
    # wire names [S16.2]: aqi_pm = integer AQI (NaN = not_ready); aqi_pm_float keeps full precision
    return {"aqi_pm": aqi_int, "aqi_pm_float": val, "aqi_valid_hours": n.astype(np.int64)}


def process(raw: pd.DataFrame, pm_measured: B, cal: dict[str, Any]) -> pd.DataFrame:
    """Raw hourly sensor frame -> device record columns (proc, flags, pm_state, baselines, AQI).

    `raw` columns: pm1, pm25, pm10, t, rh, p, wind_speed, wind_dir, rain_frac, lux, mq7_rs, mq135_rs,
    sraw_voc, sraw_nox (hourly, contiguous IST hours). `pm_measured` = SPS30 delivered a valid hour.
    """
    out = pd.DataFrame(index=raw.index)
    streak = lost_streak(pm_measured)
    out["pm_state"] = pm_state(streak)
    rh = raw.rh.to_numpy(float)
    # filled_le_1h: carry the last measured raw value forward one hour (display + model A, mask = 0)
    filled = streak == 1
    pm_cols = {}
    for c in ("pm1", "pm25", "pm10"):
        v = np.where(pm_measured, raw[c].to_numpy(float), np.nan)
        prev = pd.Series(v).ffill(limit=1).to_numpy()
        pm_cols[c] = np.where(filled, prev, v)
        out[c] = pm_cols[c]
    out["pm25_corr"] = pl.correct_pm(pm_cols["pm25"], rh, cal)
    out["pm10_corr"] = pl.correct_pm(pm_cols["pm10"], rh, cal)
    ratio = pl.coarse_ratio(pm_cols["pm1"], pm_cols["pm25"], pm_cols["pm10"])
    out["coarse_ratio"] = ratio
    dust = pl.dust_flag(ratio, rh, cal) & pm_measured
    out["dust"] = dust
    out["kappa"] = cal["kappa"]
    high_rh = rh > cal["high_humidity_rh"]

    flag = np.where(pm_measured, pl.MEASURED, np.where(filled, pl.FILLED, pl.SENSOR_FAULT))
    flag = flag | np.where(high_rh, pl.HIGH_HUMIDITY, 0) | np.where(dust, pl.DUST, 0)
    out["flag_pm"] = flag.astype(np.uint8)
    for c, col in (("t", "t"), ("rh", "rh"), ("p", "p"), ("wind", "wind_speed"), ("rain", "rain_frac"),
                   ("lux", "lux"), ("mq7", "mq7_rs"), ("mq135", "mq135_rs"), ("voc", "sraw_voc"),
                   ("nox", "sraw_nox")):  # fmt: skip
        ok = ~np.isnan(raw[col].to_numpy(float))
        f = np.where(ok, pl.MEASURED, pl.MISSING)
        if c == "rh":
            f = f | np.where(high_rh, pl.HIGH_HUMIDITY, 0)
        out[f"flag_{c}"] = f.astype(np.uint8)

    for name, col, q in (("Rbase7", "mq7_rs", 0.90), ("Rbase135", "mq135_rs", 0.90),
                         ("base_voc", "sraw_voc", 0.90), ("base_nox", "sraw_nox", 0.10)):  # fmt: skip
        out[name] = pl.rolling_baseline(raw[col].to_numpy(float), q)

    measured = pm_measured & ~np.isnan(out.pm25_corr.to_numpy())
    a = rolling_aqi(out.pm25_corr.to_numpy(), out.pm10_corr.to_numpy(), measured, measured)
    for k, v in a.items():
        out[k] = v
    return out
