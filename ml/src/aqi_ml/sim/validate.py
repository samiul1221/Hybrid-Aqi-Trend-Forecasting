"""Simulator data card: checks simulated records against field/datasheet targets ([S10.3] tests).

uv run python -m aqi_ml.sim.validate  ->  docs/research_notes/sim_data_card.md
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from aqi_ml.sim import params as sp
from aqi_shared import pipeline as pl

CARD = sp.ROOT / "docs" / "research_notes" / "sim_data_card.md"
# [IITD2025] Delhi winter (Nov-Feb) hourly SPS30 vs BAM (units without cobweb fault)
TARGET_SLOPE, TARGET_R2 = (1.46, 1.86), (0.85, 0.95)
WINTER = (11, 12, 1, 2)


def winter_fit(g: pd.DataFrame, cal: dict[str, float]) -> tuple[float, float]:
    """Slope and R2 of raw SPS30 PM2.5 (device correction undone) vs BAM truth, winter ok-hours."""
    raw = g.pm25_corr * pl.growth_factor(g.rh.to_numpy(float), cal["kappa"], cal["aw_max"])
    m = raw.notna() & g.truth_pm25.notna()
    x, y = g.truth_pm25[m].to_numpy(), raw[m].to_numpy()
    if len(x) < 200:
        return float("nan"), float("nan")
    return float(np.polyfit(x, y, 1)[0]), float(np.corrcoef(x, y)[0, 1] ** 2)


def _feature_corrs(a: pd.DataFrame) -> dict[str, float]:
    meas = a.truth_co_ppm_src.eq(0) & a.truth_co_ppm.gt(0.05)
    feats = {
        "mq7_n vs ln CO": (np.log(a.mq7_rs / a.Rbase7), np.log(a.truth_co_ppm.clip(lower=0.05)), meas),
        "mq135_n vs ln CO": (np.log(a.mq135_rs / a.Rbase135), np.log(a.truth_co_ppm.clip(lower=0.05)), meas),
        "voc_n vs ln VOC": ((a.sraw_voc - a.base_voc) / 1000, np.log(a.truth_voc_ppb), meas),
        "nox_n vs ln(1+(NO2+3.5 O3)/10)": ((a.sraw_nox - a.base_nox) / 1000,
                                           np.log1p((a.truth_no2_ppb + 3.5 * a.truth_o3_ppb) / 10), a.truth_no2_ppb_src.eq(0)),
        "ln_lux vs SW (day)": (np.log1p(a.lux), a.truth_swrad, a.truth_swrad.gt(1)),
        "t vs truth": (a.t, a.truth_t, a.truth_t.notna()),
        "rh vs truth": (a.rh, a.truth_rh, a.truth_rh.notna()),
    }  # fmt: skip
    out = {}
    for k, (x, y, m) in feats.items():
        d = pd.DataFrame({"x": x[m], "y": y[m]}).replace([np.inf, -np.inf], np.nan).dropna()
        out[k] = float(np.corrcoef(d.x, d.y)[0, 1]) if len(d) > 100 else float("nan")
    return out


def write_card(path: Path = CARD) -> dict[str, object]:
    cfg = sp.load()
    root = sp.ROOT / "data" / "synthetic" / cfg["version"]
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cal = pl.load_calibration()
    cols = ["node_id", "ts", "pm_state", "sim_gap", "sim_outage", "split", "dust", "rh", "t", "lux",
            "pm25_corr", "pm10_corr", "p", "wind_speed", "aqi_pm", "mq7_rs", "mq135_rs", "sraw_voc",
            "sraw_nox", "Rbase7", "Rbase135", "base_voc", "base_nox", "truth_pm25", "truth_t", "truth_rh",
            "truth_swrad", "truth_co_ppm", "truth_co_ppm_src", "truth_voc_ppb", "truth_no2_ppb",
            "truth_no2_ppb_src", "truth_o3_ppb"]  # fmt: skip
    r = pd.read_parquet(root / "records", columns=cols)  # column subset keeps memory bounded
    r["month"] = r.ts.dt.tz_convert("Asia/Kolkata").dt.month
    ok = r[r.pm_state.eq("ok") & r.truth_pm25.notna()]
    fits = {u: winter_fit(g, cal) for u, g in ok[ok.month.isin(WINTER)].groupby("node_id")}
    fit_df = pd.DataFrame(fits, index=["slope", "r2"]).T.dropna()
    in_slope = fit_df.slope.between(*TARGET_SLOPE).mean() if len(fit_df) else float("nan")
    corrs = _feature_corrs(ok)
    season = ok.month.map({12: "winter", 1: "winter", 2: "winter", 3: "pre", 4: "pre", 5: "pre",
                           6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "post", 11: "post"})  # fmt: skip
    lines = [
        "# Simulator data card (virtual nodes) [S10]",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m aqi_ml.sim.validate`. "
        f"Sim `{man['sim_version']}` ({man['sim_params_hash'][:19]}…), seed {man['seed']}, "
        f"calibration `{man['calibration_version']}`, schema `{man['schema_hash'][:19]}…`, "
        f"base table `{man['base_table_hash'][:19]}…`, git `{man['git_sha'][:8]}`.",
        "",
        f"- Stations: {len(man['stations'])}; spatial hold-out: {', '.join(man['spatial_holdout'])}",
        f"- Records: {len(r):,} unit-hours in {r.node_id.nunique()} virtual units; "
        f"split share {r.split.value_counts(normalize=True).round(3).to_dict()}",
        f"- pm_state share: {r.pm_state.value_counts(normalize=True).round(3).to_dict()} "
        f"(CPCB truth gaps = {r.sim_gap.mean():.1%}, simulated SPS30 outages = {r.sim_outage.mean():.2%})",
        "",
        "## PM calibration check vs Delhi field data [IITD2025]",
        "",
        f"Target: winter hourly raw SPS30 vs BAM slope {TARGET_SLOPE}, R² {TARGET_R2}. "
        f"Units inside the slope range: {in_slope:.0%} of {len(fit_df)}.",
        "",
        f"slope median {fit_df.slope.median():.2f} (P10-P90 {fit_df.slope.quantile(0.1):.2f}-{fit_df.slope.quantile(0.9):.2f}); "
        f"R² median {fit_df.r2.median():.2f} (P10-P90 {fit_df.r2.quantile(0.1):.2f}-{fit_df.r2.quantile(0.9):.2f})",
        "",
        "## Signal checks (ok hours, correlation of the model feature with its physical driver)",
        "",
        "| feature vs driver | r |",
        "|---|---|",
        *[f"| {k} | {v:+.2f} |" for k, v in corrs.items()],
        "",
        "## Rates",
        "",
        f"- dust flag (R_dust={cal['R_dust']}) by season: {ok.groupby(season).dust.mean().round(3).to_dict()}",
        f"- corrected/true PM2.5 median by season: {(ok.pm25_corr / ok.truth_pm25).groupby(season).median().round(2).to_dict()}",
        f"- gas truth source share (CO; 0 meas, 1 interp, 2 station clim, 3 network clim): "
        f"{r.truth_co_ppm_src.value_counts(normalize=True).round(3).to_dict()}",
        f"- AQI valid share: {r.aqi_pm.notna().mean():.3f}",
        "",
        "## Raw channel ranges (P1 / P50 / P99)",
        "",
        "| channel | P1 | P50 | P99 |",
        "|---|---|---|---|",
    ]
    for c in (
        "pm25_corr",
        "pm10_corr",
        "t",
        "rh",
        "p",
        "wind_speed",
        "lux",
        "mq7_rs",
        "mq135_rs",
        "sraw_voc",
        "sraw_nox",
    ):
        q = r[c].quantile([0.01, 0.5, 0.99]).to_numpy()
        lines.append(f"| {c} | {q[0]:.4g} | {q[1]:.4g} | {q[2]:.4g} |")
    lines += [
        "",
        "Interpretation and parameter sources: docs/research_notes/simulator_parameters.md, ADR 0007.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {"slope_in_range": in_slope, "fits": fit_df, "corrs": corrs}


if __name__ == "__main__":
    res = write_card()
    print(f"wrote {CARD}; units with slope in field range: {res['slope_in_range']:.0%}")
