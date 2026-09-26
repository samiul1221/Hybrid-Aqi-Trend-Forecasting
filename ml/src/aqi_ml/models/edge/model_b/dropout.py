"""Model B (no-PM fallback) training windows [S14], [S08.2], cloud [C09].

Two sources:
- dropout: from each valid Model A window, k ~ U{1..24}; PM is treated as lost for the last k hours,
  so the last valid PM hour is t-k. Inputs are the model_b channels (never PM), so hiding needs no
  value changes; the [S08.2] extras are computed from hour t-k.
- outage: real loss runs (simulated SPS30 outages, or fallback hours in real logs); k = hours since
  PM was last measured. CPCB truth gaps (sim_gap) are never used as outages.
Targets are relative to the last valid PM (label series): y[p,h] = ln(1+PM[t+h]) - ln(1+PM[t-k]).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from aqi_ml.features.builder import NodeFeatures, window_ok
from aqi_shared.schema_loader import FeatureSchema

F = npt.NDArray[np.float64]
MAX_K = 24


def extras(hourly: pd.DataFrame, last: npt.NDArray[np.int64], k: npt.NDArray[np.int64]) -> F:
    """[S08.2] features for windows whose last valid PM hour is row `last` and loss age is k hours."""
    lp25 = np.log1p(hourly.pm25_corr.to_numpy(float))
    lp10 = np.log1p(hourly.pm10_corr.to_numpy(float))
    aqi = hourly.aqi_pm.to_numpy(float)
    before = np.clip(last - 3, 0, None)
    return np.column_stack([
        lp25[last],
        lp10[last],
        aqi[last] / 500.0,
        np.clip(k, 0, MAX_K) / MAX_K,
        np.where(last >= 3, lp25[last] - lp25[before], np.nan),
    ])  # fmt: skip


def build_model_b(
    nf: NodeFeatures, schema: FeatureSchema, rng: np.random.Generator
) -> pd.DataFrame:
    """Model B window table: row, base_ts, k, source, split, y_*, xb_* ([S08.2] extras), rf_* (32)."""
    hourly = nf.hourly
    L, H = schema.lookback_hours, schema.horizons
    n = len(hourly)
    b_names = [c.name for c in schema.model_b_channels]
    seq_ok = window_ok(np.isfinite(hourly[b_names].to_numpy(float)).all(axis=1), L)
    measured = hourly.pm_measured.to_numpy(bool)
    tp25 = hourly.target_pm25.to_numpy(float)
    tp10 = hourly.target_pm10.to_numpy(float)

    # 1) dropout windows from valid Model A bases
    rows_d = np.flatnonzero(nf.valid_a)
    k_d = rng.integers(1, MAX_K + 1, size=len(rows_d))
    # 2) outage windows: base hours inside a simulated/real PM loss (not a CPCB truth gap)
    streak = pd.Series(~measured).groupby(np.cumsum(measured)).cumsum().to_numpy(np.int64)
    outage_hours = (hourly.sim_outage.to_numpy(bool) if "sim_outage" in hourly
                    else hourly.pm_state.eq("fallback_model_b").to_numpy())  # fmt: skip
    gap = hourly.sim_gap.to_numpy(bool) if "sim_gap" in hourly else np.zeros(n, bool)
    rows_o = np.flatnonzero(outage_hours & ~gap & (streak >= 1) & (streak <= MAX_K))
    k_o = streak[rows_o]

    rows = np.concatenate([rows_d, rows_o])
    k = np.concatenate([k_d, k_o])
    source = np.array(["dropout"] * len(rows_d) + ["outage"] * len(rows_o))
    last = rows - k
    ok = (last >= 3) & seq_ok[rows]
    ok &= measured[np.clip(last, 0, n - 1)] & measured[np.clip(last - 3, 0, n - 1)]
    rows, k, source, last = rows[ok], k[ok], source[ok], last[ok]

    y = np.full((len(rows), 2 * H), np.nan)
    for p_i, tp in enumerate((tp25, tp10)):
        base = np.log1p(tp[last])
        for h in range(1, H + 1):
            fut = np.clip(rows + h, 0, n - 1)
            y[:, p_i * H + h - 1] = np.where(rows + h < n, np.log1p(tp[fut]) - base, np.nan)
    xb = extras(hourly, last, k)
    keep = np.isfinite(y).all(axis=1) & np.isfinite(xb).all(axis=1)

    out = pd.DataFrame({"row": rows[keep], "k": k[keep], "source": source[keep]})
    out["base_ts"] = hourly.ts.to_numpy()[out.row]
    for c in ("split", "spatial_holdout"):
        if c in hourly:
            out[c] = hourly[c].to_numpy()[out.row]
    for j, name in enumerate(schema.output_order):
        out[f"y_{name}"] = y[keep, j].astype(np.float32)
    for j, name in enumerate(schema.model_b_extra):
        out[f"xb_{name}"] = xb[keep, j].astype(np.float32)
    # Model B RF = non-PM RF features at the base hour (no PM inputs) + the extras above
    pm_free = [j for j, f in enumerate(schema.rf_features) if not f.pm]
    rf_b = np.column_stack(
        [
            nf.rf[out.row.to_numpy()][:, pm_free],
            out[[f"xb_{n}" for n in schema.model_b_extra]].to_numpy(),
        ]
    )
    for j, name in enumerate(schema.rf_names_b):
        out[f"rf_{name}"] = rf_b[:, j].astype(np.float32)
    result: pd.DataFrame = out[np.isfinite(rf_b).all(axis=1)].reset_index(drop=True)
    return result
