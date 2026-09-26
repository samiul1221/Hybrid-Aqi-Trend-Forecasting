"""P5 tests: feature builder [S08], targets [S07], validity [S11.1], Model B builder [S14]."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from aqi_ml.features import builder as b
from aqi_ml.features import golden
from aqi_ml.features.dataset import load_config
from aqi_ml.models.edge.model_b.dropout import build_model_b
from aqi_shared import aqi
from aqi_shared.schema_loader import load as load_schema

S = load_schema()
CFG = load_config()
REC = golden.golden_records()
TRUTH = (REC.pm25_corr.to_numpy() * 1.1, REC.pm10_corr.to_numpy() * 1.1)
NF = b.build_node(REC, S, TRUTH, CFG)
CH = {c.name: c.i for c in S.channels}
RF = {n: j for j, n in enumerate(S.rf_names_a)}
GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "shared"
    / "fixtures"
    / "feature_golden"
    / "feature_golden.json"
)


def test_golden_file_matches_builder() -> None:
    assert json.loads(json.dumps(golden.build())) == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_channel_formulas_by_hand() -> None:
    v = NF.hourly
    r = 35
    x = REC.iloc[r]
    assert v.ln_pm25[r] == pytest.approx(math.log1p(x.pm25_corr), rel=1e-6)
    assert v.p[r] == pytest.approx(x.p - 1000, rel=1e-6)
    assert v.mq7_n[r] == pytest.approx(math.log(x.mq7_rs / x.Rbase7), rel=1e-6)
    assert v.voc_n[r] == pytest.approx((x.sraw_voc - x.base_voc) / 1000, rel=1e-6)
    assert v.nox_n[r] == pytest.approx((x.sraw_nox - x.base_nox) / 1000, rel=1e-6)
    assert v.aqi_now[r] == pytest.approx(x.aqi_pm / 500, rel=1e-6)
    assert v.ln_lux[r] == pytest.approx(math.log1p(x.lux), rel=1e-6)
    # ts[0] = 2026-01-01T18:30Z = IST Fri 2026-01-02 00:00; row 35 = IST 11:00, doy 2
    assert v.hod_sin[35] == pytest.approx(math.sin(2 * math.pi * 11 / 24), abs=1e-6)
    assert v.dow_cos[0] == pytest.approx(math.cos(2 * math.pi * 4 / 7), abs=1e-6)  # Friday = 4
    assert v.doy_sin[0] == pytest.approx(math.sin(2 * math.pi * 2 / 365), abs=1e-6)
    assert v.pm_mask[20] == 0 and v.m_ln_pm25[20] == 0 and v.pm_mask[19] == 1  # filled hour
    assert v.ln_pm25[20] == v.ln_pm25[19]  # filled hour carries the last value
    assert v.m_aqi_now[10] == 0 and np.isnan(v.aqi_now[10])  # AQI not ready yet


def test_short_gap_rule() -> None:
    x = np.array([[1.0], [np.nan], [np.nan], [4.0], [np.nan], [np.nan], [np.nan], [8.0], [np.nan]])
    out = b.fill_short_gaps(x, 2)[:, 0]
    assert out[:4].tolist() == [1.0, 2.0, 3.0, 4.0]  # 2 h gap interpolated
    assert np.isnan(out[4:7]).all() and out[7] == 8.0  # 3 h gap stays missing
    assert np.isnan(out[8])  # trailing gap is not extrapolated
    assert b.fill_for_inference(x)[8, 0] == 8.0


def test_rf_features_by_hand() -> None:
    t = 45
    lp = np.log1p(REC.pm25_corr.to_numpy())
    lp[20] = lp[19]
    assert NF.rf[t, RF["ln_pm25_lag3"]] == pytest.approx(lp[t - 3], rel=1e-6)
    assert NF.rf[t, RF["ln_pm25_rmean6"]] == pytest.approx(lp[t - 5 : t + 1].mean(), rel=1e-6)
    assert NF.rf[t, RF["ln_pm25_rstd6"]] == pytest.approx(lp[t - 5 : t + 1].std(ddof=0), rel=1e-6)
    assert NF.rf[t, RF["ln_pm25_diff3"]] == pytest.approx(lp[t] - lp[t - 3], rel=1e-6)
    assert NF.rf[t, RF["ln_pm25_lag23"]] == pytest.approx(lp[t - 23], rel=1e-6)
    assert NF.rf[12, RF["rain_flag_rainh3"]] == 3 and NF.rf[14, RF["rain_flag_rainh3"]] == 1
    assert NF.rf[t, RF["dust_lag0"]] == float(REC.dust[t])
    ws = REC.wind_speed.to_numpy()
    assert NF.rf[t, RF["wind_speed_rmean3"]] == pytest.approx(ws[t - 2 : t + 1].mean(), rel=1e-6)


def test_targets_definition_and_inverse() -> None:
    t = 40
    pm25, pm10 = TRUTH
    for h in (1, 2, 3):
        assert NF.y[t, h - 1] == pytest.approx(
            math.log1p(pm25[t + h]) - math.log1p(pm25[t]), rel=1e-9
        )
        assert NF.y[t, 3 + h - 1] == pytest.approx(
            math.log1p(pm10[t + h]) - math.log1p(pm10[t]), rel=1e-9
        )
        assert aqi.pm_from_y(pm25[t], NF.y[t, h - 1]) == pytest.approx(pm25[t + h])  # [S07] step 1
    assert np.isnan(NF.y[-1]).all()  # no future -> no target


def test_validity_rules() -> None:
    assert np.flatnonzero(NF.valid_a).tolist() == list(range(39, 50))
    # 39: first hour with 24 consecutive ready-AQI hours (AQI ready from 16); 50-52 lux gap (3 h) kills 50..;
    # the 2 h temperature gap (30-31) is interpolated and does not invalidate windows
    rec = REC.copy()
    rec.loc[25:30, "flag_pm"] = (
        0  # 6 PM hours not measured -> < 20 measured in windows covering them
    )
    rec.loc[25:30, "pm_state"] = "fallback_model_b"
    nf = b.build_node(rec, S, TRUTH, CFG)
    assert not nf.valid_a[39:49].any()


def test_no_future_leakage() -> None:
    rec = REC.copy()
    rec.loc[46:, ["pm25_corr", "pm10_corr", "t", "lux", "sraw_nox"]] *= 3.0
    nf = b.build_node(rec, S, TRUTH, CFG)
    cols = [c.name for c in S.channels]
    assert np.allclose(nf.rf[45], NF.rf[45], equal_nan=True)
    assert np.allclose(
        nf.hourly[cols].to_numpy()[:46], NF.hourly[cols].to_numpy()[:46], equal_nan=True
    )


def test_model_b_windows() -> None:
    wb = build_model_b(NF, S, np.random.default_rng(0))
    assert len(wb) and set(wb.source) == {"dropout"} and wb.k.between(1, 24).all()
    pm25, _ = TRUTH
    for r in wb.itertuples():
        last = r.row - r.k
        assert r.y_pm25_h1 == pytest.approx(
            math.log1p(pm25[r.row + 1]) - math.log1p(pm25[last]), rel=1e-5
        )
        assert r.xb_hours_since_pm_lost == pytest.approx(min(r.k, 24) / 24)
        assert r.xb_last_valid_aqi == pytest.approx(REC.aqi_pm[last] / 500)
    assert [c for c in wb.columns if c.startswith("rf_")] == [f"rf_{n}" for n in S.rf_names_b]
    assert not any(n.startswith(("ln_pm", "coarse", "dust", "aqi_now")) for n in S.rf_names_b[:27])


def test_model_b_uses_outages_not_truth_gaps() -> None:
    h = NF.hourly.copy()
    nf = b.NodeFeatures(h.assign(sim_outage=False, sim_gap=False), NF.rf, NF.y, NF.valid_a)
    n = len(h)
    measured = h.pm_measured.to_numpy().copy()
    measured[44:47] = False
    out = nf.hourly.assign(pm_measured=measured, sim_outage=np.isin(np.arange(n), [44, 45, 46]))
    wb = build_model_b(b.NodeFeatures(out, NF.rf, NF.y, NF.valid_a), S, np.random.default_rng(0))
    got = wb[wb.source == "outage"]
    assert got.row.tolist() == [44, 45, 46] and got.k.tolist() == [1, 2, 3]
    gap = out.assign(sim_outage=False, sim_gap=out.sim_outage)
    wb2 = build_model_b(b.NodeFeatures(gap, NF.rf, NF.y, NF.valid_a), S, np.random.default_rng(0))
    assert (wb2.source == "outage").sum() == 0  # CPCB truth gaps are never outages


def test_real_log_targets_use_measured_only() -> None:
    pm25, pm10 = b.real_log_targets(REC)
    assert np.isnan(pm25[20]) and np.isnan(pm10[20])  # filled hour is never a target
    assert pm25[19] == REC.pm25_corr[19]
    nf = b.build_node(REC, S, (pm25, pm10), CFG)
    assert np.isnan(nf.y[17:20, 0:3]).any(axis=1).all() and not nf.valid_a[17:20].any()
