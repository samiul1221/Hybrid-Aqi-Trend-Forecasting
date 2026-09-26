"""Keyless CPCB/DPCC station data from the OpenAQ open-data archive on AWS S3 [S09]. ADR 0006.

Layout (docs.openaq.org/aws/about): records/csv.gz/locationid={id}/year={Y}/month={MM}/
location-{id}-{YYYYMMDD}.csv.gz with columns location_id, sensors_id, location, datetime, lat, lon,
parameter, units, value. Verified 2026-09-26 on Delhi files: `datetime` is the END of the averaging
period (a day holds 00:15 ... 24:00 IST for 15-min CPCB data), local offset +05:30.
Raw files are mirrored byte-for-byte under data/raw/openaq_archive/ (immutable, I08; resumable).
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import gzip
import io
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from aqi_ml.data.openaq import CANONICAL, convert

BASE = "https://openaq-data-archive.s3.amazonaws.com/"
MAX_PERIOD_S = 3600


def list_keys(client: httpx.Client, location_id: int, year: int, month: int) -> list[str]:
    """All object keys for one location-month (handles S3 pagination)."""
    keys: list[str] = []
    token: str | None = None
    while True:
        params = {
            "list-type": "2",
            "prefix": f"records/csv.gz/locationid={location_id}/year={year}/month={month:02d}/",
        }
        if token:
            params["continuation-token"] = token
        r = client.get(BASE, params=params)
        r.raise_for_status()
        keys += re.findall(r"<Key>([^<]*)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", r.text)
        if not m:
            return keys
        token = m.group(1)


def fetch(client: httpx.Client, key: str, mirror: Path) -> bytes:
    """Object bytes, served from the local mirror when present (files are immutable)."""
    path = mirror / key
    if path.exists():
        return path.read_bytes()
    r = client.get(BASE + key)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(path)
    return r.content


def parse_day(blob: bytes) -> pd.DataFrame:
    """One archive day file -> rows (sensors_id, parameter, units, end, value) with end in epoch s."""
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(blob).decode("utf-8"))))
    if not rows:
        return pd.DataFrame(columns=["sensors_id", "parameter", "units", "end", "value"])
    df = pd.DataFrame(rows)[["sensors_id", "parameter", "units", "datetime", "value"]]
    end = (
        pd.to_datetime(df.pop("datetime"), utc=True) - pd.Timestamp(0, tz="UTC")
    ) // pd.Timedelta("1s")
    df["end"] = end.astype("int64")
    df["value"] = pd.to_numeric(df.value, errors="coerce").astype("float32")
    # compact dtypes: a station has ~600k rows; object strings exhausted memory on a full run
    df["sensors_id"] = pd.to_numeric(df.sensors_id).astype("int32")
    df["parameter"] = df.parameter.astype("category")
    df["units"] = df.units.astype("category")
    return df


def infer_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Add start 's' per row: period = modal spacing of that sensor's timestamps (<= 1 h)."""
    out = []
    for _, g in df.sort_values("end").groupby("sensors_id", observed=True):
        steps = np.diff(np.unique(g.end.to_numpy()))
        step = int(pd.Series(steps).mode().iloc[0]) if len(steps) else MAX_PERIOD_S
        step = min(step, MAX_PERIOD_S)
        out.append(g.assign(s=g.end - step))
    return pd.concat(out, ignore_index=True) if out else df.assign(s=pd.Series(dtype="int64"))


# Plausible MEDIAN of a gas in its canonical unit (co ppm, others ppb) for Delhi-NCR hourly data.
# CPCB feeds mislabel units (verified 2026-09-26: 2025+ 'co ppb' values ~1.5 are mg/m3; 'nox ppb'
# values ~0.05 are ppm). A (unit, year) group whose median is implausible is re-interpreted with the
# first candidate unit that makes it plausible; otherwise it is dropped. ADR 0006.
PLAUSIBLE_MEDIAN = {"co": (0.1, 10.0), "no": (0.5, 500.0), "no2": (2.0, 300.0), "nox": (5.0, 800.0),
                    "o3": (0.5, 150.0), "so2": (0.3, 150.0), "nh3": (1.0, 500.0)}  # fmt: skip
# Re-interpretation order when a label is implausible: CPCB reports CO in mg/m3; for the other
# gases a factor-1000 (ppm labelled ppb) slip is the observed failure (nox 2025: 0.058 'ppb' -> 58 ppb,
# which keeps NOx >= NO2; mg/m3 would give 30 ppb < NO2 = 53 ppb, impossible).
CANDIDATE_UNITS = {"co": ["mg/m³", "ppm", "µg/m³", "ppb"]}
DEFAULT_CANDIDATES = ["ppm", "µg/m³", "ppb", "mg/m³"]


def _repair_unit(param: str, unit: str, values: np.ndarray) -> str | None:
    lo, hi = PLAUSIBLE_MEDIAN[param]
    candidates = CANDIDATE_UNITS.get(param, DEFAULT_CANDIDATES)
    for u in [unit, *[c for c in candidates if c != unit]]:
        try:
            med = float(np.nanmedian(convert(param, u, values)))
        except ValueError:
            continue
        if lo <= med <= hi:
            return u
    return None


def to_canonical(
    df: pd.DataFrame, parameters: list[str], repairs: list[dict[str, Any]] | None = None
) -> dict[str, tuple[str, pd.DataFrame]]:
    """Rows of any units -> {parameter: (canonical unit, frame s, e, value)} for station_to_ist.

    Gas unit labels are checked per (unit, year) group and repaired if implausible; every repair
    or drop is appended to `repairs`.
    """
    df = df[df.parameter.isin(parameters) & df.parameter.isin(list(CANONICAL))]
    year = pd.to_datetime(df.end, unit="s", utc=True).dt.year
    out: dict[str, tuple[str, pd.DataFrame]] = {}
    for param, g in df.groupby("parameter", observed=True):
        param = str(param)
        parts = []
        for (unit, yr), u in g.groupby(["units", year.loc[g.index]], observed=True):
            v = u.value.to_numpy(float)
            used: str | None = str(unit)
            if param in PLAUSIBLE_MEDIAN and len(v):
                used = _repair_unit(param, str(unit), v)
                if used != unit and repairs is not None:
                    repairs.append({"parameter": param, "year": int(str(yr)), "label": str(unit),
                                    "used": used, "n": len(v), "median_raw": float(np.nanmedian(v))})  # fmt: skip
            if used is None:
                continue
            parts.append(pd.DataFrame({"s": u.s.to_numpy(np.int64), "e": u.end.to_numpy(np.int64),
                                       "value": convert(param, used, v)}))  # fmt: skip
        if parts:
            frame = pd.concat(parts, ignore_index=True).sort_values(["s", "e"])
            out[param] = (CANONICAL[param][1], frame)
    return out


def download_location(
    client: httpx.Client,
    location_id: int,
    start: date,
    end: date,
    mirror: Path,
    workers: int = 16,
) -> pd.DataFrame:
    """All rows of one location in [start, end] (whole months), with inferred periods."""
    months = pd.period_range(start, end, freq="M")
    with cf.ThreadPoolExecutor(workers) as ex:
        keys = [
            k
            for ks in ex.map(lambda p: list_keys(client, location_id, p.year, p.month), months)
            for k in ks
        ]
        frames = list(ex.map(lambda k: parse_day(fetch(client, k, mirror)), keys))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["sensors_id", "parameter", "units", "end", "value", "s"])
    df = pd.concat(frames, ignore_index=True)
    for c in ("parameter", "units"):
        df[c] = df[c].astype(str).astype("category")
    lo = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp()) - 86400
    return infer_periods(df[df.end >= lo])


def probe(client: httpx.Client, location_id: int, mirror: Path) -> dict[str, Any] | None:
    """Name and coordinates of a location from its earliest archive file, or None."""
    r = client.get(BASE, params={"list-type": "2", "prefix": f"records/csv.gz/locationid={location_id}/",
                                 "max-keys": "1"})  # fmt: skip
    keys = re.findall(r"<Key>([^<]*)</Key>", r.text)
    if not keys:
        return None
    rows = list(
        csv.DictReader(io.StringIO(gzip.decompress(fetch(client, keys[0], mirror)).decode("utf-8")))
    )
    if not rows:
        return None
    x = rows[0]
    name = re.sub(r"-\d+$", "", x["location"]).strip()
    return {"openaq_id": location_id, "name": name, "lat": float(x["lat"]), "lon": float(x["lon"])}
