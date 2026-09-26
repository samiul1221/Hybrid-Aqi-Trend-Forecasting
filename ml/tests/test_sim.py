"""Simulator tests [S10.3]: monotonicity, ranges, datasheet consistency, device parity, end-to-end."""

import numpy as np
import pandas as pd
import pytest

from aqi_ml.sim import device, forward, virtual
from aqi_ml.sim import params as sp
from aqi_shared import aqi
from aqi_shared import pipeline as pl

CFG = sp.load()


def unit(seed: int = 1) -> dict:
    return sp.draw_unit(CFG, np.random.default_rng(seed))


def quiet(p: dict) -> dict:
    """Same unit with all noise switched off (for monotonicity checks)."""
    q = {k: dict(v) for k, v in p.items()}
    q["sps30"].update(noise_rel=0.0, noise_abs=0.0, rh95_bias=0.0)
    q["hdc1080"].update(t_noise=0.0, rh_noise=0.0)
    q["bmp280"]["noise"] = 0.0
    q["wind"].update(speed_noise=0.0, dir_noise_deg=0.0)
    q["bh1750"]["noise"] = 0.0
    q["mq7"]["noise"] = q["mq135"]["noise"] = 0.0
    q["sgp41"].update(voc_noise=0.0, nox_noise=0.0)
    return q


R = np.random.default_rng(0)
a = np.asarray


def increasing(x: np.ndarray) -> bool:
    return bool(np.all(np.diff(x) > 0))


@pytest.mark.parametrize("seed", range(20))
def test_sps30_monotone_and_ranges(seed: int) -> None:
    p = quiet(unit(seed))["sps30"]
    pm = np.linspace(5, 600, 50)
    pm1, pm25, pm10 = forward.sps30(pm, pm * 1.6, np.full(50, 60.0), np.full(50, 0.3), p, 0.25, R)
    assert increasing(pm25[pm25 < 1000]) and np.all((pm1 <= pm25) & (pm25 <= pm10))
    rh = np.linspace(20, 97, 50)
    _, wet, _ = forward.sps30(
        np.full(50, 80.0), np.full(50, 130.0), rh, np.full(50, 0.3), p, 0.25, R
    )
    assert increasing(wet)  # hygroscopic growth
    noisy = unit(seed)["sps30"]
    x = forward.sps30(
        np.full(500, 900.0),
        np.full(500, 1500.0),
        np.full(500, 99.0),
        np.full(500, 0.7),
        noisy,
        0.25,
        R,
    )
    assert all(np.all((v >= 0) & (v <= 1000)) for v in x)


def test_sps30_dust_is_mostly_invisible() -> None:
    p = quiet(unit(3))["sps30"]
    smoke = forward.sps30(a([150.0]), a([180.0]), a([40.0]), a([0.3]), p, 0.25, R)
    dust = forward.sps30(a([150.0]), a([650.0]), a([40.0]), a([0.3]), p, 0.25, R)
    assert dust[1][0] < smoke[1][0]  # same PM2.5 truth, dust reads lower [Gautam2025]
    assert dust[2][0] / 650.0 < 0.6  # PM10 strongly under-read in dust
    ratio = pl.coarse_ratio(dust[0], dust[1], dust[2])[0]
    assert ratio > pl.load_calibration()["R_dust"]  # the calibrated dust flag can fire


def test_hdc1080_solar_heating_and_rh() -> None:
    p = quiet(unit())["hdc1080"]
    t0, rh0 = forward.hdc1080(a([25.0]), a([15.0]), a([0.0]), a([1.0]), p, R)
    t1, rh1 = forward.hdc1080(a([25.0]), a([15.0]), a([900.0]), a([1.0]), p, R)
    t2, _ = forward.hdc1080(a([25.0]), a([15.0]), a([900.0]), a([4.0]), p, R)
    assert t1[0] > t2[0] > t0[0] and rh1[0] < rh0[0]  # sun warms, wind cools, warmer -> lower RH
    assert forward.rh_from_dewpoint(a([20.0]), a([20.0]))[0] == pytest.approx(100.0)


def test_pressure_height_and_wind() -> None:
    p = quiet(unit())
    assert forward.bmp280(a([1000.0]), p["bmp280"], R)[0] < 1000.0 + p["bmp280"]["bias"]
    ws, _ = forward.wind(a([0.2, 3.0]), a([90.0, 90.0]), p["wind"], R)
    assert ws[0] == 0.0 and ws[1] == pytest.approx(3.0 * p["wind"]["shelter_k"])


def test_rain_drying_and_lux() -> None:
    p = quiet(unit())
    p["rain"]["false_wet_p"] = 0.0
    frac = forward.rain(a([0.0, 2.0, 0.0, 0.0, 0.0]), a([50.0] * 5), p["rain"], R)
    assert frac[0] == 0 and frac[1] == 1 and frac[2] > 0 and frac[4] == 0
    lux = forward.bh1750(
        a([0.0, 200.0, 800.0, 800.0]), a([50.0, 50.0, 50.0, 400.0]), p["bh1750"], R
    )
    assert lux[0] == 0 and lux[1] < lux[2] and lux[3] < lux[2] and lux.max() <= 121557


@pytest.mark.parametrize("seed", range(20))
def test_mq_sensors_directions_and_datasheet(seed: int) -> None:
    p = quiet(unit(seed))
    co = np.array([0.5, 1.0, 2.0, 5.0])
    z = np.zeros(4)
    rs7 = forward.mq7(co, z + 20, z + 65, z, p["mq7"], R)
    assert np.all(np.diff(rs7) < 0)  # Rs falls with CO
    hi = forward.mq7(a([50.0, 300.0]), a([20.0, 20.0]), a([65.0, 65.0]), a([0.0, 0.0]), p["mq7"], R)
    assert hi[1] / hi[0] <= 0.6  # Winsen MQ-7 v1.3: alpha = Rs(300)/Rs(50) <= 0.6
    warm = forward.mq7(a([1.0, 1.0]), a([10.0, 40.0]), a([65.0, 65.0]), a([0.0, 0.0]), p["mq7"], R)
    assert warm[1] < warm[0]  # datasheet Fig.4: Rs falls with temperature
    r135 = forward.mq135(co, co * 50, z + 20, z + 20, z + 65, z, p["mq135"], R)
    assert np.all(np.diff(r135) < 0)
    ox = forward.mq135(
        a([1.0, 1.0]),
        a([50.0, 50.0]),
        a([10.0, 80.0]),
        a([20.0] * 2),
        a([65.0] * 2),
        a([0.0] * 2),
        p["mq135"],
        R,
    )
    assert ox[1] >= ox[0]  # oxidising NO2 raises Rs


@pytest.mark.parametrize("seed", range(20))
def test_sgp41_directions_and_datasheet(seed: int) -> None:
    p = quiet(unit(seed))["sgp41"]
    z = np.zeros(2)
    voc, _ = forward.sgp41(a([5000.0, 10000.0]), z, z, z, p, R)
    per_doubling = voc[1] - voc[0]
    assert -1030 - 5 <= per_doubling <= -710 + 5  # Table 1 (ethanol, specified range)
    _, nox = forward.sgp41(z + 50, a([150.0, 300.0]), z, z, p, R)
    assert 950 - 5 <= nox[1] - nox[0] <= 2010 + 5  # Table 1 (+950..+2010 per doubling NO2)
    _, nox_o3 = forward.sgp41(z + 50, a([20.0, 20.0]), a([0.0, 40.0]), z, p, R)
    assert nox_o3[1] > nox_o3[0]  # oxidising O3 also raises NOx ticks (Fig. 3)


def test_device_aqi_matches_shared_implementation() -> None:
    rng = np.random.default_rng(5)
    n = 80
    pm25 = rng.uniform(20, 400, n)
    pm10 = pm25 * rng.uniform(1.1, 2.0, n)
    m = rng.random(n) > 0.2
    got = device.rolling_aqi(pm25, pm10, m, m)
    for i in range(23, n):
        w = slice(i - 23, i + 1)
        ref = aqi.aqi_pm_24h(pm25[w].tolist(), pm10[w].tolist(), m[w].tolist(), m[w].tolist())
        assert ref.valid == (not np.isnan(got["aqi_pm"][i]))
        if ref.valid:
            assert got["aqi_pm"][i] == pytest.approx(ref.aqi, abs=1e-9)
            assert got["aqi_pm_int"][i] == ref.aqi_int


def test_pm_state_machine() -> None:
    measured = np.array([True, False, False, True] + [False] * 26 + [True])
    streak = device.lost_streak(measured)
    st = device.pm_state(streak)
    assert st[:4].tolist() == ["ok", "filled_le_1h", "fallback_model_b", "ok"]
    assert (
        st[27] == "fallback_model_b" and st[28] == "unavailable" and st[-1] == "ok"
    )  # streak 24 | 25


def _fake_base(hours: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2021-01-01 00:30", periods=hours, freq="h", tz="UTC")
    local = ts.tz_convert("Asia/Kolkata")
    diurnal = np.sin(2 * np.pi * local.hour / 24)
    pm25 = np.exp(4.5 + 0.4 * diurnal + rng.normal(0, 0.2, hours))
    df = pd.DataFrame({
        "station_id": "s1", "ts": ts, "pm25": pm25, "pm10": pm25 * rng.uniform(1.4, 2.2, hours),
        "co_ppm": 1.5 + 0.5 * diurnal, "no2_ppb": 30 + 10 * diurnal, "o3_ppb": 20 - 10 * diurnal,
        "t": 20 + 6 * diurnal, "rh": 60 - 20 * diurnal, "p": 990.0, "wind_u": 1.0, "wind_v": -0.5,
        "precip": 0.0, "swrad": np.maximum(0, 600 * -np.cos(2 * np.pi * local.hour / 24)),
        "ist_hour": local.hour, "ist_dow": local.dayofweek, "ist_doy": local.dayofyear,
    })  # fmt: skip
    df["td"] = forward._B * 0 + 10.0
    df.loc[100:130, "pm25"] = np.nan  # CPCB gap
    df.loc[500:520, "co_ppm"] = np.nan  # gas gap -> climatology
    return df


def test_end_to_end_unit() -> None:
    df = _fake_base(24 * 40)
    truth = virtual.station_truth(df, CFG, virtual.gas_climatology(df), np.random.default_rng(1))
    assert truth.truth_co_ppm.notna().all() and set(truth.truth_co_ppm_src.unique()) <= {0, 1, 2, 3}
    assert truth.truth_kappa.between(*CFG["truth"]["kappa_aer"]["clip"]).all()
    rng = sp.unit_rng(CFG["seed"], "s1", 0)
    rec = virtual.simulate_unit(truth, sp.draw_unit(CFG, rng), CFG, pl.load_calibration(), rng)
    assert len(rec) == len(df)
    assert rec.sim_gap.iloc[100:131].all() and rec.pm25_corr.iloc[101:131].isna().all()
    assert rec.pm_state.iloc[100] == "filled_le_1h" and (rec.flag_pm.iloc[100] & pl.FILLED)
    assert (rec.pm_state.iloc[101:131] != "ok").all()
    ok = rec.pm_state == "ok"
    assert rec.loc[ok, "pm25_corr"].notna().all() and (rec.loc[ok, "flag_pm"] & pl.MEASURED).all()
    assert rec.aqi_pm_int.dropna().between(0, 500).all()
    assert (
        rec.base_voc.dropna().gt(0).all() and rec.Rbase7.iloc[:24].isna().all()
    )  # causal baseline
    with pytest.raises(ValueError):
        virtual.simulate_unit(
            truth.drop(index=[10]), sp.draw_unit(CFG, rng), CFG, pl.load_calibration(), rng
        )


def test_unit_draws_are_deterministic_and_in_range() -> None:
    a1 = sp.draw_unit(CFG, sp.unit_rng(1, "x", 0))
    a2 = sp.draw_unit(CFG, sp.unit_rng(1, "x", 0))
    assert a1 == a2
    lo, hi = CFG["sps30"]["scale_s"]["u"]
    assert lo <= a1["sps30"]["scale_s"] <= hi
