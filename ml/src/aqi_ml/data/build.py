"""Data pipeline CLI [S09]. Every step is resumable (HTTP cache + interim files).

    uv run python -m aqi_ml.data.build all            # stations -> openaq -> weather -> base -> card
    uv run python -m aqi_ml.data.build weather --max-stations 2
    uv run python -m aqi_ml.data.build hforecast      # Open-Meteo historical-forecast (teacher, C07)

ZenML (P9) will call the same functions.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from aqi_shared.time_utils import ist_local_to_ts

from aqi_ml.data import open_meteo, openaq
from aqi_ml.data.base_table import POLLUTANTS, merge_station
from aqi_ml.data.http import CachedClient

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "ml" / "configs" / "data.yaml"


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = ROOT / v
    return cfg


def _dates(cfg: dict[str, Any]) -> tuple[date, date]:
    return date.fromisoformat(cfg["period"]["start"]), date.fromisoformat(cfg["period"]["end"])


def _bounds(cfg: dict[str, Any]) -> tuple[int, int]:
    """First/last bucket epoch: IST 00:00 of the start date .. IST 23:00 of the end date."""
    s, e = _dates(cfg)
    first = ist_local_to_ts(datetime(s.year, s.month, s.day, 0))
    last = ist_local_to_ts(datetime(e.year, e.month, e.day, 23))
    return int(first.timestamp()), int(last.timestamp())


def _openaq_client(cfg: dict[str, Any]) -> CachedClient:
    key = os.environ.get(cfg["openaq"]["api_key_env"]) or _dotenv(cfg["openaq"]["api_key_env"])
    if not key:
        raise SystemExit(f"set {cfg['openaq']['api_key_env']} in .env (free key: explore.openaq.org)")
    return CachedClient(cfg["paths"]["cache"] / "openaq", cfg["openaq"]["min_interval_s"],
                        headers={"X-API-Key": key})  # fmt: skip


def _om_client(cfg: dict[str, Any]) -> CachedClient:
    return CachedClient(cfg["paths"]["cache"] / "open_meteo", cfg["open_meteo"]["min_interval_s"])


def _dotenv(name: str) -> str | None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == name and v.strip():
                return v.strip()
    return None


def _stations(cfg: dict[str, Any], max_stations: int | None) -> pd.DataFrame:
    df = pd.read_parquet(cfg["paths"]["interim"] / "stations.parquet")
    return df.head(max_stations) if max_stations else df


def step_stations(cfg: dict[str, Any]) -> None:
    oq = cfg["openaq"]
    client = _openaq_client(cfg)
    locs = openaq.list_locations(client, oq["base_url"], cfg["region"]["bbox"], oq["monitor_only"],
                                 oq["page_limit"])  # fmt: skip
    providers = sorted({(loc.get("provider") or {}).get("name") or "?" for loc in locs})
    st = openaq.select_stations(locs, oq["provider_name_regex"], oq["parameters"])
    if st.empty:
        raise SystemExit(f"no station matched provider regex; providers seen: {providers}")
    om = cfg["open_meteo"]
    st["elevation_m"] = open_meteo.elevations(_om_client(cfg), om["elevation_url"],
                                              list(zip(st["lat"], st["lon"], strict=True)))  # fmt: skip
    st["sensors"] = st["sensors"].map(json.dumps)
    out = cfg["paths"]["interim"]
    out.mkdir(parents=True, exist_ok=True)
    st.to_parquet(out / "stations.parquet", index=False)
    (out / "providers_seen.json").write_text(json.dumps(providers, indent=1), encoding="utf-8")
    print(f"{len(st)} stations (providers seen: {providers})")


def step_openaq(cfg: dict[str, Any], max_stations: int | None) -> None:
    oq = cfg["openaq"]
    client = _openaq_client(cfg)
    start, end = _dates(cfg)
    out = cfg["paths"]["interim"] / "openaq"
    out.mkdir(parents=True, exist_ok=True)
    align_stats: dict[str, dict[str, float]] = {}
    for st in _stations(cfg, max_stations).to_dict("records"):
        raw = {}
        for s in json.loads(st["sensors"]):
            raw[s["parameter"]] = (s["units"], openaq.fetch_sensor(
                client, oq["base_url"], s["sensor_id"], start, end, oq["chunk_days"], oq["page_limit"]))  # fmt: skip
        wide, aligned = openaq.station_to_ist(raw, cfg["cleaning"]["min_bucket_coverage"])
        wide.to_parquet(out / f"{st['station_id']}.parquet", index=False)
        align_stats[str(st["station_id"])] = aligned
        print(f"{st['station_id']}: {len(wide)} IST hours")
    (out / "_alignment.json").write_text(json.dumps(align_stats, indent=1), encoding="utf-8")


def step_weather(cfg: dict[str, Any], max_stations: int | None, historical_forecast: bool) -> None:
    om = cfg["open_meteo"]
    client = _om_client(cfg)
    start, end = _dates(cfg)
    if historical_forecast:
        start = max(start, date(2022, 1, 1))  # archive of stitched short-lead forecasts (ADR 0005)
    url = om["historical_forecast_url"] if historical_forecast else om["archive_url"]
    model = None if historical_forecast else om["archive_model"]
    out = cfg["paths"]["interim"] / ("open_meteo_hf" if historical_forecast else "open_meteo")
    out.mkdir(parents=True, exist_ok=True)
    # +-1 day: the first IST hour starts at 18:30Z the previous day, and alignment needs neighbours
    start, end = start - timedelta(days=1), min(end + timedelta(days=1), date.today())
    for st in _stations(cfg, max_stations).to_dict("records"):
        utc = open_meteo.fetch_hourly_utc(client, url, float(st["lat"]), float(st["lon"]), start, end,
                                          om["hourly"], model, om["chunk_days"])  # fmt: skip
        open_meteo.to_ist(utc, om["accumulated"]).to_parquet(out / f"{st['station_id']}.parquet",
                                                             index=False)  # fmt: skip
        print(f"{st['station_id']}: weather {len(utc)} UTC hours")


def step_base(cfg: dict[str, Any], max_stations: int | None) -> None:
    interim, out = cfg["paths"]["interim"], cfg["paths"]["base_table"]
    s, e = _bounds(cfg)
    stats = {}
    for st in _stations(cfg, max_stations).to_dict("records"):
        aq_path = interim / "openaq" / f"{st['station_id']}.parquet"
        om_path = interim / "open_meteo" / f"{st['station_id']}.parquet"
        if not (aq_path.exists() and om_path.exists()):
            print(f"skip {st['station_id']}: missing interim data")
            continue
        df, st_stats = merge_station(str(st["station_id"]), pd.read_parquet(aq_path),
                                     pd.read_parquet(om_path), s, e, cfg["cleaning"])  # fmt: skip
        part = out / f"station_id={st['station_id']}"
        part.mkdir(parents=True, exist_ok=True)
        df.drop(columns="station_id").to_parquet(part / "part-0.parquet", index=False)
        stats[str(st["station_id"])] = st_stats
    (interim / "_cleaning.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


def step_card(cfg: dict[str, Any]) -> None:
    from aqi_ml.data.datacard import write_card

    write_card(cfg)
    print(f"wrote {cfg['paths']['data_card']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aqi_ml.data.build")
    ap.add_argument("step", choices=["stations", "openaq", "weather", "hforecast", "base", "card", "all"])
    ap.add_argument("--max-stations", type=int, default=None, help="limit for trial runs")
    ap.add_argument("--config", type=Path, default=CONFIG)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.step in ("stations", "all"):
        step_stations(cfg)
    if a.step in ("openaq", "all"):
        step_openaq(cfg, a.max_stations)
    if a.step in ("weather", "all"):
        step_weather(cfg, a.max_stations, historical_forecast=False)
    if a.step == "hforecast":
        step_weather(cfg, a.max_stations, historical_forecast=True)
    if a.step in ("base", "all"):
        step_base(cfg, a.max_stations)
    if a.step in ("card", "all"):
        step_card(cfg)


__all__ = ["POLLUTANTS", "load_config", "main"]

if __name__ == "__main__":
    main()
