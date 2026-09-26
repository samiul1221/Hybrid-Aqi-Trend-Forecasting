"""Data card for the base table: coverage, missingness, cleaning and alignment stats [S09]."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aqi_ml.data.base_table import POLLUTANTS

WEATHER = ["t", "rh", "p", "wind_u", "wind_v", "td", "precip", "swrad", "blh"]
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "pre-monsoon", 4: "pre-monsoon",
           5: "pre-monsoon", 6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
           10: "post-monsoon", 11: "post-monsoon"}  # fmt: skip


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _pct(x: float) -> str:
    return f"{100 * x:.0f}"


def _md(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids the optional 'tabulate' dependency)."""
    head = "| " + " | ".join([str(df.index.name or ""), *map(str, df.columns)]) + " |"
    rows = [
        "| " + " | ".join([str(i), *map(str, r)]) + " |"
        for i, r in zip(df.index, df.to_numpy(), strict=True)
    ]
    return "\n".join([head, "|" + "---|" * (len(df.columns) + 1), *rows])


def summarise(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-station coverage table and PM2.5 seasonal quantiles (pure; used by write_card and tests)."""
    cols = [c for c in POLLUTANTS + WEATHER if c in base]
    cov = base[cols].notna().groupby(base.station_id).mean()
    span = base.dropna(subset=["pm25"]).groupby("station_id").ts.agg(["min", "max"])
    season = base.ts.dt.tz_convert("Asia/Kolkata").dt.month.map(SEASONS)
    q = base.assign(season=season).groupby("season").pm25.quantile(np.array([0.05, 0.5, 0.95])).unstack()
    src = {c: base[c].value_counts(normalize=True) for c in ("t_src", "rh_src", "p_src", "wind_src") if c in base}
    return {"coverage": cov, "span": span, "pm25_season_q": q, "sources": pd.DataFrame(src).fillna(0)}


def write_card(cfg: dict[str, Any]) -> None:
    base = pd.read_parquet(cfg["paths"]["base_table"])
    base["station_id"] = base.station_id.astype(str)
    st = pd.read_parquet(cfg["paths"]["interim"] / "stations.parquet").set_index("station_id")
    interim = cfg["paths"]["interim"]
    cleaning = _load_json(interim / "_cleaning.json")
    aligned = _load_json(interim / "openaq" / "_alignment.json")
    providers = _load_json(interim / "providers_seen.json")
    s = summarise(base)
    cov = s["coverage"]
    lines = [
        "# Data card — base table [S09]",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m aqi_ml.data.build card`. "
        "Regenerate after every data build; numbers below come from the parquet files.",
        "",
        "## Sources",
        f"- CPCB stations via OpenAQ v3 (raw measurements, provider regex "
        f"`{cfg['openaq']['provider_name_regex']}`). Providers seen in the bbox: {', '.join(providers) or 'n/a'}.",
        f"- Weather: Open-Meteo archive, model `{cfg['open_meteo']['archive_model']}`, timezone=GMT, "
        "aligned to IST hours (ADR 0005). Station met used when present and plausible (`*_src`).",
        "- Attribution: OpenAQ (CC BY 4.0 platform terms; data licences per provider), "
        "Open-Meteo (CC BY 4.0), ERA5 (Copernicus C3S).",
        f"- Period: {cfg['period']['start']} .. {cfg['period']['end']}; bbox {cfg['region']['bbox']}.",
        "- CCR fallback: stations/periods missing in OpenAQ can be exported manually from "
        "https://airquality.cpcb.gov.in/ccr (15-min or hourly CSV) and loaded with the same columns.",
        "",
        f"## Stations ({len(cov)})",
        "",
        "| station | name | lat | lon | elev m | PM2.5 from | to | PM2.5 % | PM10 % | NO2 % | CO % | IST-aligned PM2.5 periods % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    cov_rows = {str(k): v for k, v in cov.to_dict("index").items()}
    for sid, r in cov_rows.items():
        m = st.loc[sid]
        first, last = ("", "")
        if sid in s["span"].index:
            first = f"{s['span'].loc[sid, 'min']:%Y-%m-%d}"
            last = f"{s['span'].loc[sid, 'max']:%Y-%m-%d}"
        pcts = [_pct(float(r.get(c, 0.0))) for c in ("pm25", "pm10", "no2_ppb", "co_ppm")]
        ist = _pct(aligned.get(sid, {}).get("pm25", float("nan")))
        lines.append(
            f"| {sid} | {m['name']} | {m['lat']:.4f} | {m['lon']:.4f} | {m['elevation_m']:.0f} | "
            f"{first} | {last} | {' | '.join(pcts)} | {ist} |"
        )
    lines += ["", "## Coverage by variable (mean over stations, % of IST hours in period)", "",
              "| variable | coverage % |", "|---|---|"]  # fmt: skip
    for c, v in cov.mean().items():
        lines.append(f"| {c} | {_pct(v)} |")
    lines += ["", "## Weather source share (fraction of hours)", "", _md(s["sources"].round(3)), ""]
    lines += ["## PM2.5 distribution by season (µg/m³, cleaned, all stations)", "",
              _md(s["pm25_season_q"].round(1)), ""]  # fmt: skip
    tot: dict[str, int] = {}
    for d in cleaning.values():
        for k, v in d.items():
            tot[k] = tot.get(k, 0) + v
    lines += ["## Cleaning (values set to NaN, all stations) [S09]", "", "| rule | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(tot.items()) if v]
    lines += ["", "## Known limitations", "",
              "- `blh` (boundary layer height) is not served by the Open-Meteo archive/historical-forecast for Delhi (verified 2026-09-26): almost always null (see coverage table).",
              "- No imputation in the base table; gaps stay NaN (feature builder applies the <=2 h rule).",
              "- Historical-forecast (teacher covariates) is stitched from the first hours of successive runs, not forecasts as issued at base_ts (ADR 0005).",
              ""]  # fmt: skip
    Path(cfg["paths"]["data_card"]).write_text("\n".join(lines), encoding="utf-8", newline="\n")
