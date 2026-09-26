"""Open-Meteo historical weather (ERA5 archive), historical-forecast and elevation [S09], [C07].

Requests always use timezone=GMT and wind_speed_unit=ms; IST alignment is done here (align.py),
because Open-Meteo's own +05:30 handling labels the 01:00Z value as 06:00 local (ADR 0005).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from aqi_ml.data.align import accumulated_to_ist, instantaneous_to_ist, wind_speed_dir, wind_uv
from aqi_ml.data.http import CachedClient

# Open-Meteo variable -> base-table column
COLUMNS = {
    "temperature_2m": "t2m",
    "relative_humidity_2m": "rh2m",
    "dew_point_2m": "td2m",
    "surface_pressure": "sp",
    "precipitation": "precip",
    "shortwave_radiation": "swrad",
    "boundary_layer_height": "blh",
}
RECENT_DAYS = 7  # ERA5 lags ~5 days; responses touching the last week are not cached


def _chunks(start: date, end: date, days: int) -> list[tuple[date, date]]:
    out = []
    s = start
    while s <= end:
        e = min(end, s + timedelta(days=days - 1))
        out.append((s, e))
        s = e + timedelta(days=1)
    return out


def fetch_hourly_utc(
    client: CachedClient,
    url: str,
    lat: float,
    lon: float,
    start: date,
    end: date,
    hourly: list[str],
    model: str | None = None,
    chunk_days: int = 366,
) -> pd.DataFrame:
    """Hourly UTC frame: column 'epoch' (s) + one column per Open-Meteo variable."""
    frames = []
    for s, e in _chunks(start, end, chunk_days):
        params: dict[str, Any] = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": s.isoformat(),
            "end_date": e.isoformat(),
            "hourly": ",".join(hourly),
            "timezone": "GMT",
            "wind_speed_unit": "ms",
        }
        if model:
            params["models"] = model
        recent = e >= date.today() - timedelta(days=RECENT_DAYS)
        data = client.get_json(url, params, cache=not recent)
        if data.get("utc_offset_seconds", 0) != 0:
            raise ValueError("expected GMT response")
        h = data["hourly"]
        df = pd.DataFrame({k: h[k] for k in hourly}, dtype=float)
        # resolution-independent epoch seconds (pandas 3 infers second resolution here)
        epoch = (pd.to_datetime(h["time"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")
        df.insert(0, "epoch", epoch.astype("int64"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates("epoch").sort_values("epoch")


def to_ist(df: pd.DataFrame, accumulated: list[str]) -> pd.DataFrame:
    """Hourly UTC Open-Meteo frame -> IST buckets with base-table column names (+ u10, v10, ws10, wd10)."""
    t = df.epoch.to_numpy(np.int64)
    out: pd.DataFrame | None = None

    def put(name: str, frame: pd.DataFrame) -> None:
        nonlocal out
        col = frame.set_index("bucket")["value"].rename(name)
        out = col.to_frame() if out is None else out.join(col, how="outer")

    for var, col in COLUMNS.items():
        if var not in df:
            continue
        v = df[var].to_numpy(float)
        put(col, accumulated_to_ist(t, v) if var in accumulated else instantaneous_to_ist(t, v))
    if "wind_speed_10m" in df and "wind_direction_10m" in df:
        u, v = wind_uv(df.wind_speed_10m.to_numpy(float), df.wind_direction_10m.to_numpy(float))
        put("u10", instantaneous_to_ist(t, u))
        put("v10", instantaneous_to_ist(t, v))
    assert out is not None
    out = out.sort_index()
    if "u10" in out:
        ws, wd = wind_speed_dir(out.u10.to_numpy(), out.v10.to_numpy())
        out["ws10"], out["wd10"] = ws, wd
    out.index.name = "bucket"
    return out.reset_index()


def elevations(
    client: CachedClient, url: str, coords: list[tuple[float, float]]
) -> list[float]:
    """Elevation (m) per (lat, lon) from the Open-Meteo elevation API (<= 100 per call)."""
    out: list[float] = []
    for i in range(0, len(coords), 100):
        part = coords[i : i + 100]
        params = {
            "latitude": ",".join(f"{la:.4f}" for la, _ in part),
            "longitude": ",".join(f"{lo:.4f}" for _, lo in part),
        }
        out += [float(x) for x in client.get_json(url, params)["elevation"]]
    return out
