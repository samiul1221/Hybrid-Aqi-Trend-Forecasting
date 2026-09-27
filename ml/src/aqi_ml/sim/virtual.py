"""Virtual nodes: base table (true PM + weather + gases) -> simulated device records [S10].

    uv run python -m aqi_ml.sim.virtual [--max-stations N]

Output (data/synthetic/<version>/): records/station_id=*/unit=*/part-0.parquet (one row per unit-hour,
same columns as a real node record + truth/label columns), units.parquet (all drawn parameters),
manifest.json (versions and hashes). Labels are TRUE CPCB PM ([S10.3] post_simulation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from aqi_ml.data.align import wind_speed_dir, wind_uv
from aqi_ml.sim import device, forward
from aqi_ml.sim import params as sp
from aqi_shared import pipeline as pl
from aqi_shared.schema_loader import load as load_schema

ROOT = sp.ROOT
BASE_TABLE = ROOT / "data" / "processed" / "base_table"
OUT = ROOT / "data" / "synthetic"
GASES = ("co_ppm", "no2_ppb", "o3_ppb")
SRC_MEAS, SRC_INTERP, SRC_STATION_CLIM, SRC_NETWORK_CLIM = 0, 1, 2, 3


def gas_climatology(base: pd.DataFrame) -> dict[str, pd.Series]:
    """Network-wide month x IST-hour median per gas (fallback for stations lacking a gas)."""
    month = base.ts.dt.tz_convert("Asia/Kolkata").dt.month
    return {g: base.groupby([month, base.ist_hour])[g].median() for g in GASES if g in base}


def fill_gas(
    df: pd.DataFrame, gas: str, max_gap: int, network: pd.Series | None
) -> tuple[pd.Series, pd.Series]:
    """Gas truth with gap handling: interpolate <= max_gap h, else station clim, else network clim."""
    s = df[gas] if gas in df else pd.Series(np.nan, index=df.index)
    src = pd.Series(np.where(s.notna(), SRC_MEAS, -1), index=df.index)
    filled = s.interpolate(limit=max_gap, limit_area="inside")
    src[(src == -1) & filled.notna()] = SRC_INTERP
    month = df.ts.dt.tz_convert("Asia/Kolkata").dt.month
    key = pd.MultiIndex.from_arrays([month, df.ist_hour])
    clim = s.groupby([month, df.ist_hour]).median()
    st_clim = pd.Series(clim.reindex(key).to_numpy(), index=df.index)
    use = filled.isna() & st_clim.notna()
    filled[use], src[use] = st_clim[use], SRC_STATION_CLIM
    if network is not None:
        net = pd.Series(network.reindex(key).to_numpy(), index=df.index)
        use = filled.isna() & net.notna()
        filled[use], src[use] = net[use], SRC_NETWORK_CLIM
    return filled, src


def kappa_series(n: int, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Aerosol hygroscopicity kappa(t): AR(1) around the mean, clipped ([Mandariya2024], [Arub2020])."""
    phi = np.exp(-1.0 / cfg["decorrelation_h"])
    eps = rng.normal(0.0, np.sqrt(1 - phi**2), n)
    z = np.empty(n)
    z[0] = rng.normal()
    for i in range(1, n):
        z[i] = phi * z[i - 1] + eps[i]
    lo, hi = cfg["clip"]
    out: npt.NDArray[np.float64] = np.clip(cfg["mean"] + cfg["sd"] * z, lo, hi)
    return out


def station_truth(df: pd.DataFrame, cfg: dict[str, Any], network: dict[str, pd.Series],
                  rng: np.random.Generator) -> pd.DataFrame:  # fmt: skip
    """Atmosphere-side truth shared by all units at a station."""
    tr = cfg["truth"]
    t = pd.DataFrame({"ts": df.ts.to_numpy()}, index=df.index)
    for g in GASES:
        t[f"truth_{g}"], t[f"truth_{g}_src"] = fill_gas(
            df, g, tr["gas_interp_max_gap_h"], network.get(g)
        )
    voc_per_co = sp.draw(tr["voc_per_co"], rng)
    t["truth_voc_ppb"] = (
        sp.draw(tr["voc_ppb_background"], rng) + voc_per_co * t.truth_co_ppm * 1000.0
    )
    t["truth_kappa"] = kappa_series(len(df), tr["kappa_aer"], rng)
    for c in ("t", "rh", "td", "p", "wind_u", "wind_v", "precip", "swrad", "pm25", "pm10"):
        t[f"truth_{c}"] = df[c].to_numpy(float)
    ws, wd = wind_speed_dir(df.wind_u.to_numpy(float), df.wind_v.to_numpy(float))
    t["truth_ws"], t["truth_wd"] = ws, wd
    t.attrs["voc_per_co"] = voc_per_co
    return t


def pm_fault_mask(
    n: int, cfg: dict[str, Any], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """(random_missing, outage) hour masks for one unit [S10.3] pm_common."""
    missing = rng.random(n) < cfg["random_missing_frac"]
    outage = np.zeros(n, dtype=bool)
    n_out = rng.poisson(cfg["outages_per_year"] * n / 8766.0)
    for _ in range(n_out):
        length = round(sp.draw(cfg["outage_hours"], rng))
        start = int(rng.integers(0, max(n - length, 1)))
        outage[start : start + length] = True
    return missing, outage


def simulate_unit(truth: pd.DataFrame, p: dict[str, dict[str, Any]], cfg: dict[str, Any],
                  cal: dict[str, Any], rng: np.random.Generator,
                  t0: pd.Timestamp | None = None) -> pd.DataFrame:  # fmt: skip
    """One virtual unit over one CONTIGUOUS hourly segment: forward models + device processing.

    `t0` is the unit's first hour (drift time base); defaults to the segment start.
    """
    n = len(truth)
    if n and (truth.ts.diff().dropna() != pd.Timedelta(hours=1)).any():
        raise ValueError("truth segment must be contiguous hourly")
    start = truth.ts.iloc[0] if t0 is None else t0
    seconds = truth.ts.sub(start).dt.total_seconds().to_numpy(dtype=np.float64)
    days = cast(npt.NDArray[np.float64], seconds / 86400.0)  # dtype forced above

    def g(c: str) -> npt.NDArray[np.float64]:
        return truth[c].to_numpy(dtype=np.float64)

    raw = pd.DataFrame(index=truth.index)
    # RH inside the SPS30 sensing volume: same dew point, air warmed by internal_dt_c (ADR 0007)
    rh_sps = np.clip(
        forward.rh_from_dewpoint(g("truth_td"), g("truth_t") + p["sps30"]["internal_dt_c"]),
        0.0,
        100.0,
    )
    raw["pm1"], raw["pm25"], raw["pm10"] = forward.sps30(
        g("truth_pm25"), g("truth_pm10"), rh_sps, g("truth_kappa"), p["sps30"],
        cfg["truth"]["dust_fine_fraction"]["const"], rng)  # fmt: skip
    ws_m, wd_m = forward.wind(g("truth_ws"), g("truth_wd"), p["wind"], rng)
    raw["t"], raw["rh"] = forward.hdc1080(g("truth_t"), g("truth_td"), g("truth_swrad"),
                                          p["wind"]["shelter_k"] * g("truth_ws"), p["hdc1080"], rng)  # fmt: skip
    raw["p"] = forward.bmp280(g("truth_p"), p["bmp280"], rng)
    u, v = wind_uv(ws_m, wd_m)
    raw["wind_u"], raw["wind_v"], raw["wind_speed"], raw["wind_dir"] = u, v, ws_m, wd_m
    raw["rain_frac"] = forward.rain(g("truth_precip"), g("truth_rh"), p["rain"], rng)
    raw["lux"] = forward.bh1750(g("truth_swrad"), g("truth_pm25"), p["bh1750"], rng)
    t_s, rh_s = raw.t.to_numpy(float), raw.rh.to_numpy(float)
    raw["mq7_rs"] = forward.mq7(g("truth_co_ppm"), t_s, rh_s, days, p["mq7"], rng)
    raw["mq135_rs"] = forward.mq135(g("truth_co_ppm"), g("truth_voc_ppb"), g("truth_no2_ppb"),
                                    t_s, rh_s, days, p["mq135"], rng)  # fmt: skip
    raw["sraw_voc"], raw["sraw_nox"] = forward.sgp41(g("truth_voc_ppb"), g("truth_no2_ppb"),
                                                     g("truth_o3_ppb"), days, p["sgp41"], rng)  # fmt: skip

    missing, outage = pm_fault_mask(n, p["sps30"] | cfg["sps30"], rng)
    truth_ok = ~np.isnan(raw.pm25.to_numpy()) & ~np.isnan(raw.pm10.to_numpy())
    pm_measured = truth_ok & ~missing & ~outage
    proc = device.process(raw, pm_measured, cal)
    rec = pd.concat([truth[["ts"]], raw.drop(columns=["pm1", "pm25", "pm10"]), proc], axis=1)
    rec["sim_gap"] = ~truth_ok  # CPCB truth missing: not a simulated sensor failure
    rec["sim_outage"] = outage
    for c in truth.columns:
        if c.startswith("truth_"):
            rec[c] = truth[c].to_numpy()
    return rec


def _split(ts: pd.Series, splits: dict[str, list[str]]) -> np.ndarray:
    out = np.full(len(ts), "", dtype=object)
    for name, (a, b) in splits.items():
        lo = pd.Timestamp(a, tz="Asia/Kolkata")
        hi = pd.Timestamp(b, tz="Asia/Kolkata") + pd.Timedelta(days=1)
        out[(ts >= lo) & (ts < hi)] = name
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _hash_dir(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(path.rglob("*.parquet")):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(f.read_bytes())
    return "sha256:" + h.hexdigest()


def _unit_row(sid: str, u: int, truth: pd.DataFrame, p: dict[str, dict[str, Any]],
              holdout: list[str]) -> dict[str, Any]:  # fmt: skip
    return {"node_id": f"v-{sid}-{u}", "station_id": sid, "unit": u,
            "voc_per_co": truth.attrs["voc_per_co"], "holdout": sid in holdout,
            **{f"{s}.{k}": v for s, d in p.items() for k, v in d.items() if isinstance(v, int | float)}}  # fmt: skip


def build(
    max_stations: int | None = None, cfg_path: Path = sp.SIM_PARAMS, force: bool = False
) -> Path:
    """Simulate every eligible station; resumes (skips unit shards that already exist) unless force."""
    cfg = sp.load(cfg_path)
    cal = pl.load_calibration()
    base_cols = ["station_id", "ts", "ist_hour", "pm25", "pm10", *GASES, "t", "rh", "td", "p", "wind_u",
                 "wind_v", "precip", "swrad"]  # fmt: skip
    base = pd.read_parquet(BASE_TABLE, columns=base_cols)  # only what the simulator uses (memory)
    base["station_id"] = base.station_id.astype(str)
    network = gas_climatology(base)
    counts = base.groupby("station_id").pm25.count()
    stations = sorted(counts[counts >= cfg["min_station_pm25_hours"]].index)
    rng0 = np.random.default_rng(cfg["seed"])
    # hold-out candidates: well-covered stations (full-period share is low because of the 2023-24 gap)
    eligible = [s for s in stations if counts[s] >= cfg["holdout_min_pm25_hours"]]
    holdout = sorted(rng0.choice(eligible, size=min(cfg["spatial_holdout_stations"], len(eligible)),
                                 replace=False).tolist())  # fmt: skip
    if max_stations:
        stations = stations[:max_stations]
    out = OUT / str(cfg["version"])
    units_rows = []
    for sid in stations:
        df = base[base.station_id == sid].sort_values("ts").reset_index(drop=True)
        month = df.ts.dt.tz_convert("Asia/Kolkata").dt.to_period("M")
        has_pm = df.pm25.notna().groupby(month).transform("any")
        df = df[has_pm.to_numpy()].reset_index(
            drop=True
        )  # drop months without any truth (archive gap)
        truth = station_truth(df, cfg, network, sp.unit_rng(cfg["seed"], sid, -1))
        segment = (
            truth.ts.diff() != pd.Timedelta(hours=1)
        ).cumsum()  # re-deployment after long gaps
        for u in range(cfg["units_per_station"]):
            rng = sp.unit_rng(cfg["seed"], sid, u)
            p = sp.draw_unit(cfg, rng)
            part = out / "records" / f"station_id={sid}" / f"unit={u}"
            if (part / "part-0.parquet").exists() and not force:
                units_rows.append(_unit_row(sid, u, truth, p, holdout))
                continue
            rec = pd.concat([simulate_unit(tr.reset_index(drop=True), p, cfg, cal, rng, truth.ts.iloc[0])
                             for _, tr in truth.groupby(segment)], ignore_index=True)  # fmt: skip
            rec.insert(0, "node_id", f"v-{sid}-{u}")
            rec["split"] = _split(rec.ts, cfg["splits"])
            rec["spatial_holdout"] = sid in holdout
            rec["calibration_version"] = cal["version"]
            part.mkdir(parents=True, exist_ok=True)
            rec.to_parquet(part / "part-0.parquet", index=False)
            units_rows.append(_unit_row(sid, u, truth, p, holdout))
        print(f"{sid}: {len(df)} hours x {cfg['units_per_station']} units", flush=True)
    pd.DataFrame(units_rows).to_parquet(out / "units.parquet", index=False)
    manifest = {
        "sim_version": cfg["version"], "sim_params_hash": cfg["_hash"], "seed": cfg["seed"],
        "schema_hash": load_schema().hash, "calibration_version": cal["version"],
        "base_table_hash": _hash_dir(BASE_TABLE), "git_sha": _git_sha(),
        "stations": stations, "spatial_holdout": holdout, "splits": cfg["splits"],
    }  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aqi_ml.sim.virtual")
    ap.add_argument("--max-stations", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-simulate existing shards")
    a = ap.parse_args(argv)
    print(f"wrote {build(a.max_stations, force=a.force)}")


if __name__ == "__main__":
    main()
