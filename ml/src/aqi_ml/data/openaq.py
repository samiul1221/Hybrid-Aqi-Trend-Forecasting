"""CPCB hourly station data via the OpenAQ v3 API [S09]. API key from env (X-API-Key).

Raw measurements (not OpenAQ's /hours, which aggregates on its own clock) are fetched per sensor
and aligned to IST buckets with align.intervals_to_ist. Fallback if OpenAQ lacks a station or
period: manual CSV export from the CPCB CCR portal (https://airquality.cpcb.gov.in/ccr), loaded
with the same column names (documented in docs/research_notes/data_card.md).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from aqi_ml.data.align import intervals_to_ist, wind_speed_dir, wind_uv
from aqi_ml.data.http import CachedClient

MOLAR_VOLUME = 24.45  # L/mol at 25 °C, 1 atm; spec [S09] factors are 24.45/MW
MOLAR_MASS = {"co": 28.01, "no2": 46.01, "no": 30.01, "nox": 46.01, "o3": 48.00,
              "so2": 64.07, "nh3": 17.03}  # fmt: skip
# parameter -> (base-table column, canonical unit)
CANONICAL = {
    "pm25": ("pm25", "ug/m3"), "pm10": ("pm10", "ug/m3"), "pm1": ("pm1", "ug/m3"),
    "no2": ("no2_ppb", "ppb"), "no": ("no_ppb", "ppb"), "nox": ("nox_ppb", "ppb"),
    "o3": ("o3_ppb", "ppb"), "so2": ("so2_ppb", "ppb"), "nh3": ("nh3_ppb", "ppb"),
    "co": ("co_ppm", "ppm"), "benzene": ("benzene_ugm3", "ug/m3"),
    "temperature": ("t_stn", "c"), "relativehumidity": ("rh_stn", "%"),
    "pressure": ("p_stn", "hpa"), "wind_speed": ("ws_stn", "m/s"),
    "wind_direction": ("wd_stn", "deg"),
}  # fmt: skip
_UNIT_ALIASES = {"µg/m³": "ug/m3", "μg/m³": "ug/m3", "ug/m³": "ug/m3", "mg/m³": "mg/m3",
                 "°c": "c", "deg c": "c", "degc": "c", "mb": "hpa", "mbar": "hpa",
                 "degrees": "deg", "°": "deg"}  # fmt: skip


def norm_unit(u: str) -> str:
    u = u.strip().lower()
    return _UNIT_ALIASES.get(u, u)


def convert(parameter: str, unit: str, values: np.ndarray) -> np.ndarray:
    """Convert to the canonical unit of `parameter`; raise on anything unexpected."""
    target = CANONICAL[parameter][1]
    u = norm_unit(unit)
    if u == target:
        return values
    mw = MOLAR_MASS.get(parameter)
    factor = {
        ("ug/m3", "ppb"): None if mw is None else MOLAR_VOLUME / mw,
        ("mg/m3", "ppm"): None if mw is None else MOLAR_VOLUME / mw,
        ("ug/m3", "ppm"): None if mw is None else MOLAR_VOLUME / mw / 1000,
        ("ppm", "ppb"): 1000.0,
        ("ppb", "ppm"): 0.001,
        ("mg/m3", "ug/m3"): 1000.0,
        ("kpa", "hpa"): 10.0,
        ("km/h", "m/s"): 1 / 3.6,
    }.get((u, target))
    if factor is None:
        raise ValueError(f"no conversion for {parameter}: {unit!r} -> {target}")
    return values * factor


def _paged(client: CachedClient, url: str, params: dict[str, Any], limit: int,
           cache: bool = True) -> list[dict[str, Any]]:  # fmt: skip
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        data = client.get_json(url, {**params, "limit": limit, "page": page}, cache=cache)
        res = data["results"]
        out += res
        if len(res) < limit:
            return out
        page += 1


def list_locations(client: CachedClient, base_url: str, bbox: list[float], monitor: bool,
                   limit: int = 1000) -> list[dict[str, Any]]:  # fmt: skip
    params: dict[str, Any] = {"bbox": ",".join(f"{x:.4f}" for x in bbox)}
    if monitor:
        params["monitor"] = "true"
    return _paged(client, f"{base_url}/locations", params, limit)


def select_stations(locations: list[dict[str, Any]], provider_regex: str,
                    parameters: list[str]) -> pd.DataFrame:  # fmt: skip
    """One row per CPCB location: station_id, openaq_id, name, lat, lon, provider, sensors."""
    rx = re.compile(provider_regex)
    rows = []
    for loc in locations:
        provider = (loc.get("provider") or {}).get("name") or ""
        if not rx.search(provider) or loc.get("isMobile"):
            continue
        sensors = [
            {"sensor_id": s["id"], "parameter": s["parameter"]["name"],
             "units": s["parameter"]["units"]}
            for s in loc.get("sensors", []) if s["parameter"]["name"] in parameters
        ]  # fmt: skip
        rows.append({
            "station_id": f"oaq{loc['id']}", "openaq_id": loc["id"], "name": loc["name"],
            "lat": loc["coordinates"]["latitude"], "lon": loc["coordinates"]["longitude"],
            "provider": provider, "sensors": sensors,
        })  # fmt: skip
    return pd.DataFrame(rows)


def _epoch(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def fetch_sensor(client: CachedClient, base_url: str, sensor_id: int, start: date, end: date,
                 chunk_days: int = 31, limit: int = 1000) -> pd.DataFrame:  # fmt: skip
    """Raw measurements for one sensor: columns s, e (epoch s), value."""
    rows: list[tuple[int, int, float]] = []
    s = start
    while s <= end:
        e = min(end, s + timedelta(days=chunk_days - 1))
        params = {"datetime_from": f"{s.isoformat()}T00:00:00Z",
                  "datetime_to": f"{(e + timedelta(days=1)).isoformat()}T00:00:00Z"}  # fmt: skip
        recent = e >= date.today() - timedelta(days=7)
        url = f"{base_url}/sensors/{sensor_id}/measurements"
        for r in _paged(client, url, params, limit, cache=not recent):
            p = r["period"]
            if r.get("value") is not None:
                rows.append((_epoch(p["datetimeFrom"]["utc"]), _epoch(p["datetimeTo"]["utc"]),
                             float(r["value"])))  # fmt: skip
        s = e + timedelta(days=1)
    return pd.DataFrame(rows, columns=["s", "e", "value"])


def station_to_ist(raw: dict[str, tuple[str, pd.DataFrame]], min_coverage: float
                   ) -> tuple[pd.DataFrame, dict[str, float]]:  # fmt: skip
    """{parameter: (unit, raw frame)} -> wide IST frame (bucket + canonical columns).

    Returns the frame and, per parameter, the fraction of source periods that were already IST
    hour buckets (1.0 = CPCB-style From-To hours, no re-binning happened).
    """
    cols: list[pd.Series] = []
    ist_aligned: dict[str, float] = {}
    for param, (unit, df) in raw.items():
        if df.empty or param in ("wind_speed", "wind_direction"):
            continue
        s, e = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
        ist_aligned[param] = float(np.mean(((s + 19800) % 3600 == 0) & (e - s == 3600)))
        v = convert(param, unit, df.value.to_numpy(float))
        if param not in ("temperature",):
            v = np.where(v < 0, np.nan, v)  # [S09] drop negatives (not for temperature)
        a = intervals_to_ist(s, e, v)
        a = a[a.coverage >= min_coverage]
        cols.append(a.set_index("bucket")["value"].rename(CANONICAL[param][0]))
    if "wind_speed" in raw and "wind_direction" in raw:
        (u_ws, ws), (_, wd) = raw["wind_speed"], raw["wind_direction"]
        m = ws.merge(wd, on=["s", "e"], suffixes=("_ws", "_wd")).dropna()
        if not m.empty:
            speed = convert("wind_speed", u_ws, m.value_ws.to_numpy(float))
            u, v = wind_uv(speed, m.value_wd.to_numpy(float))
            s, e = m.s.to_numpy(np.int64), m.e.to_numpy(np.int64)
            au, av = intervals_to_ist(s, e, u), intervals_to_ist(s, e, v)
            au, av = au[au.coverage >= min_coverage], av[av.coverage >= min_coverage]
            su, sv = au.set_index("bucket")["value"], av.set_index("bucket")["value"]
            su, sv = su.align(sv, join="inner")
            spd, dirn = wind_speed_dir(su.to_numpy(), sv.to_numpy())
            cols += [su.rename("u_stn"), sv.rename("v_stn"),
                     pd.Series(spd, su.index, name="ws_stn"), pd.Series(dirn, su.index, name="wd_stn")]  # fmt: skip
    if not cols:
        return pd.DataFrame({"bucket": pd.Series(dtype="int64")}), ist_aligned
    wide = pd.concat(cols, axis=1).sort_index()
    wide.index.name = "bucket"
    return wide.reset_index(), ist_aligned
