"""Put source time series onto IST clock-hour buckets [S09] alignment, I02. ADR 0005.

Buckets are [b, b+3600) with b = aqi_shared.time_utils.bucket_epoch(...) (UTC epoch seconds of an
IST hour start). Three rules:
- intervals (OpenAQ periods; Open-Meteo precipitation/shortwave whose value at T covers (T-1h, T]):
  time-weighted mean over the overlap, plus coverage = covered seconds / 3600;
- instantaneous hourly UTC samples (Open-Meteo T, RH, P, u, v, ...): exact mean of the linearly
  interpolated series over the bucket = 1/8 v(T-1h) + 3/4 v(T) + 1/8 v(T+1h), bucket centred on T;
- wind: average u, v (vector mean, [S06.2]) and derive speed and direction from the means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqi_shared.time_utils import HOUR_S, IST_OFFSET_S

HALF_HOUR_S = HOUR_S // 2


def _bucket(epoch: np.ndarray) -> np.ndarray:
    # Same formula as aqi_shared.time_utils.bucket_epoch, vectorised (floor division).
    return (epoch + IST_OFFSET_S) // HOUR_S * HOUR_S - IST_OFFSET_S


def intervals_to_ist(start: np.ndarray, end: np.ndarray, value: np.ndarray) -> pd.DataFrame:
    """Time-weighted mean of [start, end) intervals (epoch s, each <= 1 h) per IST bucket.

    Returns columns bucket (epoch s), value, coverage (0..1). NaN values and duplicate intervals
    are dropped first; intervals longer than 1 h are rejected.
    """
    df = pd.DataFrame({"s": start, "e": end, "v": value}).dropna().drop_duplicates(["s", "e"])
    if ((df.e - df.s) > HOUR_S).any() or ((df.e - df.s) <= 0).any():
        raise ValueError("intervals must have 0 < duration <= 1 h")
    s = df.s.to_numpy(np.int64)
    e = df.e.to_numpy(np.int64)
    v = df.v.to_numpy(float)
    b0 = _bucket(s)
    b1 = _bucket(e - 1)
    first_end = np.minimum(e, b0 + HOUR_S)
    pieces = pd.DataFrame({"bucket": b0, "w": first_end - s, "v": v})
    split = b1 != b0
    second = pd.DataFrame({"bucket": b1[split], "w": e[split] - b1[split], "v": v[split]})
    allp = pd.concat([pieces, second], ignore_index=True)
    allp["wv"] = allp.w * allp.v
    g = allp.groupby("bucket", sort=True)[["w", "wv"]].sum()
    out = pd.DataFrame(
        {"bucket": g.index.to_numpy(np.int64), "value": g.wv / g.w, "coverage": g.w / HOUR_S}
    ).reset_index(drop=True)
    out["coverage"] = out.coverage.clip(upper=1.0)
    return out


def instantaneous_to_ist(epoch_utc_hour: np.ndarray, value: np.ndarray) -> pd.DataFrame:
    """Hourly UTC point samples -> IST bucket means (weights 1/8, 3/4, 1/8). NaN if a sample is missing."""
    t = np.asarray(epoch_utc_hour, dtype=np.int64)
    if len(t) and (t % HOUR_S != 0).any():
        raise ValueError("samples must be on UTC hours")
    s = pd.Series(np.asarray(value, dtype=float), index=t)
    full = s.reindex(np.arange(t.min(), t.max() + HOUR_S, HOUR_S)) if len(t) else s
    v = full.to_numpy()
    mean = np.full(len(v), np.nan)
    mean[1:-1] = 0.125 * v[:-2] + 0.75 * v[1:-1] + 0.125 * v[2:]
    return pd.DataFrame({"bucket": full.index.to_numpy(np.int64) - HALF_HOUR_S, "value": mean})


def accumulated_to_ist(epoch_utc_hour: np.ndarray, value: np.ndarray) -> pd.DataFrame:
    """Open-Meteo 'preceding hour' values (sum or mean over (T-1h, T]) -> IST bucket values."""
    t = np.asarray(epoch_utc_hour, dtype=np.int64)
    out = intervals_to_ist(t - HOUR_S, t, np.asarray(value, dtype=float))
    out.loc[out.coverage < 1.0, "value"] = np.nan
    return out[["bucket", "value"]]


def wind_uv(speed: np.ndarray, dir_from_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological (speed, direction FROM) -> (u, v) [S06.2]: u = -ws sin th, v = -ws cos th."""
    th = np.deg2rad(dir_from_deg)
    return -speed * np.sin(th), -speed * np.cos(th)


def wind_speed_dir(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) -> (speed, direction FROM in [0, 360)) [S06.2]."""
    return np.hypot(u, v), np.rad2deg(np.arctan2(-u, -v)) % 360.0
