"""Python reference feature/target builder [S07], [S08], [S11.1] — golden for the firmware C twin.

Input: one node's hourly device records (contiguous IST hours; simulated or real, same columns as a
[S16.2] record: pm1, pm25_corr, pm10_corr, coarse_ratio, t, rh, p, wind_u, wind_v, wind_speed,
rain_frac, lux, mq7_rs, mq135_rs, sraw_voc, sraw_nox, Rbase7, Rbase135, base_voc, base_nox, aqi_pm,
dust, pm_state, flag_* bitmasks). Everything is driven by shared/schema/feature_schema.yaml (I05).
Conventions are in the schema file and ADR 0008.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from aqi_shared import pipeline as pl
from aqi_shared.schema_loader import FeatureSchema
from aqi_shared.time_utils import IST_OFFSET_S

F = npt.NDArray[np.float64]

# schema `src` -> record column (sources that are not plain record columns are handled below)
SRC_COLUMN = {"coarse_ratio": "coarse_ratio"}
# channel -> record flag column whose MEASURED bit is the channel mask
MASK_FLAG = {"ln_pm25": "flag_pm", "ln_pm10": "flag_pm", "ln_pm1": "flag_pm", "coarse_ratio": "flag_pm",
             "t": "flag_t", "rh": "flag_rh", "p": "flag_p", "wind_u": "flag_wind", "wind_v": "flag_wind",
             "rain_frac": "flag_rain", "ln_lux": "flag_lux", "mq7_n": "flag_mq7", "mq135_n": "flag_mq135",
             "voc_n": "flag_voc", "nox_n": "flag_nox"}  # fmt: skip
PM_OK_STATES = ("ok", "filled_le_1h")  # Model A applies ([S14] runtime policy)


def ist_calendar(ts: pd.Series) -> dict[str, npt.NDArray[np.int64]]:
    """IST hour 0-23, dow 0=Mon, doy 1-366 (schema calendar_conventions)."""
    local = ts.dt.tz_convert("UTC").dt.tz_localize(None) + pd.Timedelta(seconds=IST_OFFSET_S)
    return {
        "ist_hour": local.dt.hour.to_numpy(np.int64),
        "ist_dow": local.dt.dayofweek.to_numpy(np.int64),
        "ist_doy": local.dt.dayofyear.to_numpy(np.int64),
    }


def _op_raw(op: str, args: dict[str, Any], x: F, rec: pd.DataFrame) -> Any:
    if op == "identity":
        return x
    if op == "log1p":
        return np.log1p(np.maximum(x, 0.0))
    if op == "clip":
        return np.clip(x, args["lo"], args["hi"])
    if op == "offset":
        return x + args["value"]
    if op == "scale":
        return x * args["value"]
    if op == "log_ratio":
        return np.log(x / rec[args["base"]].to_numpy(float))
    if op == "base_diff_scale":
        return (x - rec[args["base"]].to_numpy(float)) * args["scale"]
    if op == "cyc_sin":
        return np.sin(2 * np.pi * x / args["period"])
    if op == "cyc_cos":
        return np.cos(2 * np.pi * x / args["period"])
    raise ValueError(f"unknown op {op}")


def _apply_op(op: str, args: dict[str, Any], x: F, rec: pd.DataFrame) -> F:
    return np.asarray(_op_raw(op, args, x, rec), dtype=np.float64)


def channel_matrix(rec: pd.DataFrame, schema: FeatureSchema) -> tuple[F, npt.NDArray[np.uint8]]:
    """[T x C] transformed channel values (NaN = no value) and masks (1 = measured) [S08.1]."""
    cal = ist_calendar(rec.ts)
    pm_measured = (rec.flag_pm.to_numpy() & pl.MEASURED) > 0
    n, c = len(rec), len(schema.channels)
    vals = np.full((n, c), np.nan)
    masks = np.zeros((n, c), dtype=np.uint8)
    for ch in schema.channels:
        if ch.op == "measured_mask":
            vals[:, ch.i] = pm_measured.astype(float)
            masks[:, ch.i] = 1
            continue
        if ch.src in cal:
            x = cal[ch.src].astype(float)
            m = np.ones(n, dtype=bool)
        else:
            x = rec[SRC_COLUMN.get(ch.src, ch.src)].to_numpy(float)
            if ch.name in MASK_FLAG:
                m = (rec[MASK_FLAG[ch.name]].to_numpy() & pl.MEASURED) > 0
            else:  # aqi_now: valid AQI on a measured PM hour
                m = ~np.isnan(x) & pm_measured
        v = _apply_op(ch.op, ch.op_args, x, rec)
        vals[:, ch.i] = np.where(np.isfinite(v), v, np.nan)
        masks[:, ch.i] = (m & np.isfinite(v)).astype(np.uint8)
    return vals, masks


def fill_short_gaps(vals: F, max_gap: int) -> F:
    """Linear interpolation of interior NaN runs of length <= max_gap only ([S08.1] missing policy).

    Longer runs stay NaN (windows touching them are invalid for training). Masks are not changed.
    """
    out = vals.copy()
    for j in range(vals.shape[1]):
        s = pd.Series(vals[:, j])
        na = s.isna()
        run = na.groupby((~na).cumsum()).transform(
            "sum"
        )  # length of the NaN run each NaN belongs to
        interp = s.interpolate(limit_area="inside")
        take = na & (run <= max_gap) & interp.notna()
        out[take.to_numpy(), j] = interp[take].to_numpy()
    return out


def fill_for_inference(vals: F) -> F:
    """Inference rule [S08.1]: carry the last value forward (masks stay 0)."""
    out: F = pd.DataFrame(vals).ffill().to_numpy()
    return out


def aux_sources(rec: pd.DataFrame) -> dict[str, F]:
    """RF-only sources (schema rf_aux_sources)."""
    return {
        "dust": rec.dust.fillna(False).to_numpy(dtype=float),
        "wind_speed": rec.wind_speed.to_numpy(float),
        "rain_flag": (rec.rain_frac.to_numpy(float) > 0.1).astype(float),
    }


def rf_matrix(vals: F, aux: dict[str, F], schema: FeatureSchema) -> F:
    """[T x 50] RF features for every base hour t, Model A order [S08.3]; NaN where history is short."""
    idx = {c.name: c.i for c in schema.channels}
    out = np.full((len(vals), len(schema.rf_features)), np.nan)
    for j, f in enumerate(schema.rf_features):
        s = pd.Series(vals[:, idx[f.src]] if f.src in idx else aux[f.src])
        if f.kind == "lags":
            col = s.shift(f.k)
        elif f.kind == "roll_mean":
            col = s.rolling(f.k).mean()
        elif f.kind == "roll_std":
            col = s.rolling(f.k).std(ddof=0)
        elif f.kind == "diff":
            col = s - s.shift(f.k)
        elif f.kind == "rain_hours_last":
            col = s.rolling(f.k).sum()
        else:
            raise ValueError(f.kind)
        out[:, j] = col.to_numpy()
    return out


def log_change_targets(pm25: F, pm10: F, horizons: int) -> F:
    """[T x 2H] targets y[p,h] = ln(1+PM_p[t+h]) - ln(1+PM_p[t]), pollutant-major (I04, [S07])."""
    n = len(pm25)
    y = np.full((n, 2 * horizons), np.nan)
    for p_i, pm in enumerate((pm25, pm10)):
        lp = np.log1p(pm)
        for h in range(1, horizons + 1):
            y[: n - h, p_i * horizons + h - 1] = lp[h:] - lp[: n - h]
    return y


def real_log_targets(rec: pd.DataFrame) -> tuple[F, F]:
    """Label PM for real node logs [S11.1]: own corrected PM, measured hours only (I07: filled,
    missing, fallback and model values are never targets)."""
    measured = (rec.flag_pm.to_numpy() & pl.MEASURED) > 0
    pm25 = np.where(measured, rec.pm25_corr.to_numpy(float), np.nan)
    pm10 = np.where(measured, rec.pm10_corr.to_numpy(float), np.nan)
    return pm25, pm10


def window_ok(finite_rows: npt.NDArray[np.bool_], lookback: int) -> npt.NDArray[np.bool_]:
    """True at t when rows t-lookback+1..t are all True."""
    s = pd.Series(finite_rows.astype(float)).rolling(lookback).min()
    out: npt.NDArray[np.bool_] = (s == 1.0).to_numpy()
    return out


@dataclass
class NodeFeatures:
    hourly: pd.DataFrame  # per hour: filled channel values, masks, labels, states
    rf: F  # [T x 50] Model A RF features per base hour
    y: F  # [T x 6] Model A targets per base hour
    valid_a: npt.NDArray[np.bool_]  # Model A training-window validity per base hour


def build_node(rec: pd.DataFrame, schema: FeatureSchema, target_pm: tuple[F, F],
               cfg: dict[str, Any]) -> NodeFeatures:  # fmt: skip
    """One node's hourly records -> features, targets and Model A window validity.

    target_pm: (pm25, pm10) label series - TRUE CPCB PM for synthetic nodes ([S10.3]); the node's own
    corrected PM with non-measured hours set to NaN for real logs ([S11.1], I07).
    Model A window valid at t ([S11.1], [S08.1], [S14]): full 24 h history with no NaN after the
    <= 2 h fill, PM state ok or filled <= 1 h, >= 20 measured PM hours, finite RF row and targets.
    """
    rec = rec.reset_index(drop=True)
    L, H = schema.lookback_hours, schema.horizons
    vals, masks = channel_matrix(rec, schema)
    filled = fill_short_gaps(vals, cfg["interpolate_max_gap_h"])
    rf = rf_matrix(filled, aux_sources(rec), schema)
    y = log_change_targets(target_pm[0], target_pm[1], H)
    pm_measured = (rec.flag_pm.to_numpy() & pl.MEASURED) > 0
    n_meas = pd.Series(pm_measured.astype(float)).rolling(L).sum().to_numpy()
    seq_ok = window_ok(np.asarray(np.isfinite(filled).all(axis=1), dtype=bool), L)
    valid_a = (seq_ok & rec.pm_state.isin(PM_OK_STATES).to_numpy() & (n_meas >= cfg["min_measured_pm_hours"])
               & np.isfinite(rf).all(axis=1) & np.isfinite(y).all(axis=1))  # fmt: skip

    hourly = pd.DataFrame({"ts": rec.ts})
    for ch in schema.channels:
        hourly[ch.name] = filled[:, ch.i].astype(np.float32)
        hourly[f"m_{ch.name}"] = masks[:, ch.i]
    hourly["pm25_corr"], hourly["pm10_corr"] = rec.pm25_corr.to_numpy(), rec.pm10_corr.to_numpy()
    hourly["aqi_pm"], hourly["pm_state"] = rec.aqi_pm.to_numpy(), rec.pm_state.to_numpy()
    hourly["pm_measured"] = pm_measured
    hourly["target_pm25"], hourly["target_pm10"] = target_pm
    for c in ("split", "spatial_holdout", "sim_outage", "sim_gap"):
        if c in rec:
            hourly[c] = rec[c].to_numpy()
    return NodeFeatures(hourly, rf, y, valid_a)


def windows_a(nf: NodeFeatures, schema: FeatureSchema) -> pd.DataFrame:
    """Model A window table: row (index of base hour t in `hourly`), base_ts, split, y_*, rf_*."""
    rows = np.flatnonzero(nf.valid_a)
    win = pd.DataFrame({"row": rows, "base_ts": nf.hourly.ts.to_numpy()[rows]})
    for c in ("split", "spatial_holdout"):
        if c in nf.hourly:
            win[c] = nf.hourly[c].to_numpy()[rows]
    for j, name in enumerate(schema.output_order):
        win[f"y_{name}"] = nf.y[rows, j].astype(np.float32)
    for j, name in enumerate(schema.rf_names_a):
        win[f"rf_{name}"] = nf.rf[rows, j].astype(np.float32)
    return win
