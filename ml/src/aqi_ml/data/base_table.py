"""Base table [S09]: one row per (station, IST hour) with cleaned pollutants and 'true' weather.

No imputation here: gaps stay NaN (the feature builder applies the <=2 h interpolate+mask rule,
and target hours are never imputed).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aqi_ml.data.align import wind_speed_dir
from aqi_shared.time_utils import HOUR_S, IST_OFFSET_S

POLLUTANTS = ["pm25", "pm10", "pm1", "no2_ppb", "no_ppb", "nox_ppb", "o3_ppb", "so2_ppb",
              "nh3_ppb", "co_ppm", "benzene_ugm3"]  # fmt: skip


def _flatline_mask(s: pd.Series, hours: int) -> pd.Series:
    """True for values inside runs of >= `hours` identical consecutive non-NaN hourly values."""
    run_id = (s != s.shift()).cumsum()
    size = s.groupby(run_id).transform("size")
    return s.notna() & (size >= hours)


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """[S09] cleaning on a complete hourly index. Returns cleaned copy and counts of nulled values."""
    df = df.copy()
    stats: dict[str, int] = {}
    present = [c for c in POLLUTANTS if c in df]
    for c in present:
        neg = df[c] < 0
        stats[f"{c}_negative"] = int(neg.sum())
        df.loc[neg, c] = np.nan
    if "pm25" in df:
        big = df.pm25 > cfg["pm25_max"]
        stats["pm25_over_max"] = int(big.sum())
        df.loc[big, "pm25"] = np.nan
    if "pm25" in df and "pm10" in df:
        bad = df.pm25 > cfg["pm25_over_pm10_max"] * df.pm10
        stats["pm25_gt_pm10"] = int(bad.sum())
        df.loc[bad, ["pm25", "pm10"]] = np.nan
    for c in present:
        flat = _flatline_mask(df[c], cfg["flatline_hours"])
        stats[f"{c}_flatline"] = int(flat.sum())
        df.loc[flat, c] = np.nan
    return df, stats


def _source(stn_ok: pd.Series, om_ok: pd.Series) -> pd.Series:
    """'stn' where the station value is used, 'om' where Open-Meteo is used, None if neither."""
    src = pd.Series(None, index=om_ok.index, dtype=object)
    src[om_ok] = "om"
    src[stn_ok] = "stn"
    return src


def _pick(
    stn: pd.Series | None, om: pd.Series, lo: float, hi: float
) -> tuple[pd.Series, pd.Series]:
    """Station value when present and plausible, else Open-Meteo. Returns (value, source)."""
    if stn is None:
        return om, _source(pd.Series(False, index=om.index), om.notna())
    ok = stn.notna() & stn.between(lo, hi)
    return stn.where(ok, om), _source(ok, om.notna())


def dewpoint(t: pd.Series, rh: pd.Series) -> pd.Series:
    """Magnus formula (Alduchov & Eskridge 1996 constants), °C."""
    g = np.log(rh.clip(lower=1) / 100.0) + 17.625 * t / (243.04 + t)
    td: pd.Series = 243.04 * g / (17.625 - g)
    return td


def merge_station(
    station_id: str,
    aq: pd.DataFrame,
    om: pd.DataFrame,
    start_epoch: int,
    end_epoch: int,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Station pollutants (aq) + Open-Meteo (om), both keyed by 'bucket', onto every IST hour."""
    idx = pd.Index(np.arange(start_epoch, end_epoch + 1, HOUR_S, dtype=np.int64), name="bucket")
    a = aq.set_index("bucket").reindex(idx)
    w = om.set_index("bucket").reindex(idx)
    a, stats = clean(a, cfg)
    pl = cfg["plausible"]
    out = pd.DataFrame(index=idx)
    for c in POLLUTANTS:
        out[c] = a[c] if c in a else np.nan
    out["t"], out["t_src"] = _pick(a.get("t_stn"), w.t2m, *pl["t"])
    out["rh"], out["rh_src"] = _pick(a.get("rh_stn"), w.rh2m, *pl["rh"])
    out["p"], out["p_src"] = _pick(a.get("p_stn"), w.sp, *pl["p"])
    stn_wind_ok = (a["ws_stn"].between(*pl["ws"]) & a["u_stn"].notna()) if "ws_stn" in a else None
    if stn_wind_ok is not None:
        out["wind_u"] = a.u_stn.where(stn_wind_ok, w.u10)
        out["wind_v"] = a.v_stn.where(stn_wind_ok, w.v10)
        out["wind_src"] = _source(stn_wind_ok, w.u10.notna())
    else:
        out["wind_u"], out["wind_v"] = w.u10, w.v10
        out["wind_src"] = _source(pd.Series(False, index=idx), w.u10.notna())
    ws, wd = wind_speed_dir(out.wind_u.to_numpy(float), out.wind_v.to_numpy(float))
    out["wind_speed"], out["wind_dir"] = ws, wd
    out["td"] = dewpoint(out.t, out.rh)
    for c in ("precip", "swrad", "blh"):
        out[c] = w[c] if c in w else np.nan
    out = out.reset_index()
    out.insert(0, "station_id", station_id)
    out.insert(1, "ts", pd.to_datetime(out.bucket, unit="s", utc=True))
    local = out.ts.dt.tz_localize(None) + pd.Timedelta(seconds=IST_OFFSET_S)
    out["ist_hour"] = local.dt.hour.astype("int8")
    out["ist_dow"] = local.dt.dayofweek.astype("int8")  # 0 = Monday (ADR 0003)
    out["ist_doy"] = local.dt.dayofyear.astype("int16")
    return out.drop(columns="bucket"), stats
