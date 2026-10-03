"""
simulator.py — the "sensor box":  X (true hourly air + weather of one station)  ->  Y (what one ESP32 node would log)

    import simulator
    Y = simulator.simulate(X, station="anand_vihar_delhi_dpcc", unit=0)

    or from the command line:
    uv run python simulator.py data/2_clean/anand_vihar_delhi_dpcc.parquet --unit 0 -o Y.parquet

Used ONLY to make training data while we have no real nodes (notebook 02), or to feed fake nodes for demos.
At real inference the real sensors replace this box: the ESP32 itself produces Y.

X  (one row per IST hour; = a table from data/2_clean/)
    ts                      UTC time of the IST hour start
    pm25, pm10              true PM (ug/m3, CPCB)            -> SPS30, BH1750 (smog dims light); kept as the answer
    co_ppm, no2_ppb, o3_ppb true gases (CPCB, gaps filled)    -> MQ-7, MQ-135, SGP41
    t, td, rh               true temperature, dew point, RH   -> HDC1080, SPS30 (humid particles), MQ sensors
    p                       true pressure (hPa)               -> BMP280
    wind_u, wind_v          true wind (m/s)                   -> anemometer, HDC1080 (wind cools the shield)
    precip                  rain (mm/h)                       -> rain board
    swrad                   sunshine (W/m2)                   -> BH1750, HDC1080 (sun heats the shield)
    (made inside the box: kappa = particle water uptake, VOC = 20 ppb + r * CO)

Y  (one row per hour)
    raw sensor readings:  pm1, pm25, pm10 (SPS30) | t, rh (HDC1080) | p (BMP280) | wind_speed, wind_u, wind_v
                          (anemometer) | rain_frac (rain board) | lux (BH1750) | mq7_rs, mq135_rs (ohm) |
                          sraw_voc, sraw_nox (SGP41 ticks)
    ESP32 processing:     pm_measured, pm_state, pm25_corr, pm10_corr (humidity-corrected), coarse_ratio, dust,
                          Rbase7, Rbase135, base_voc, base_nox (clean-air baselines), aqi_pm (CPCB AQI, 24 h)
    bookkeeping:          sim_outage (simulated SPS30 outage), true_pm25, true_pm10 (the ANSWER - never a model input)

Formulas and sources: notebook 02 (one section per sensor) and docs/simulator_sources.md.
"""

import argparse
import hashlib

import numpy as np
import pandas as pd

# a random-number generator that gives the same numbers every run for the same (station, unit)
def make_rng(station, unit):
    seed = int(hashlib.sha256(f"20260926/{station}/{unit}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed)


def U(rng, low, high):
    return float(rng.uniform(low, high))


# kappa: slowly changing aerosol water uptake
def kappa_series(n, rng):
    phi = np.exp(-1 / 48)
    z = np.zeros(n)
    z[0] = rng.normal()
    noise = rng.normal(0, np.sqrt(1 - phi**2), n)
    for i in range(1, n):
        z[i] = phi * z[i - 1] + noise[i]
    return np.clip(0.30 + 0.10 * z, 0.05, 0.77)


# gas sensors need a gas value every hour: fill short gaps, then use the usual value for that month and hour
def fill_gas(gas, ts, network_typical):
    month = ts.dt.tz_convert("Asia/Kolkata").dt.month
    hour = ts.dt.tz_convert("Asia/Kolkata").dt.hour
    key = pd.MultiIndex.from_arrays([month, hour])
    filled = gas.interpolate(limit=3, limit_area="inside")
    station_typical = gas.groupby([month, hour]).median()
    filled = filled.fillna(pd.Series(station_typical.reindex(key).to_numpy(), index=gas.index))
    filled = filled.fillna(pd.Series(network_typical.reindex(key).to_numpy(), index=gas.index))
    return filled.to_numpy()


# SPS30: draw this unit's own errors
def sps30_params(rng):
    return dict(scale=U(rng, 1.15, 1.55), dt_inside=U(rng, 1.5, 3.5), dust_eff=U(rng, 0.20, 0.40),
                pm10_q=U(rng, 0.02, 0.10), coarse_f=U(rng, 0.05, 0.25), pm1_r=U(rng, 0.75, 0.95),
                fog_bias=U(rng, 0.0, 0.20))


# SPS30: true PM -> what the sensor reports
def sps30(pm25_true, pm10_true, rh_inside, kappa, p, rng):
    coarse = np.maximum(pm10_true - pm25_true, 0)
    dust = np.minimum(0.25 * coarse, pm25_true)
    fine = pm25_true - dust
    aw = np.clip(rh_inside / 100, 0, 0.97)
    growth = 1 + (kappa / 1.65) / (1 / np.maximum(aw, 1e-9) - 1)
    fog = np.where(rh_inside > 95, 1 + rng.uniform(0, p["fog_bias"], len(rh_inside)), 1.0)

    pm25 = p["scale"] * (fine * growth + p["dust_eff"] * dust) * fog
    pm25 = pm25 + rng.normal(0, 0.05 * np.abs(pm25) + 1)
    pm10 = pm25 * (1 + p["pm10_q"]) + p["scale"] * p["coarse_f"] * coarse
    pm1 = p["pm1_r"] * p["scale"] * fine * growth * fog + rng.normal(0, 0.05 * np.abs(pm25) + 1)

    pm25 = np.clip(pm25, 0, 1000)
    pm10 = np.clip(np.maximum(pm10, pm25), 0, 1000)     # PM10 always includes PM2.5
    pm1 = np.clip(np.minimum(pm1, pm25), 0, 1000)       # PM1 is part of PM2.5
    return pm1, pm25, pm10


# RH (%) of air with dew point td at temperature t (Magnus formula)
def rh_from_dewpoint(td, t):
    return 100 * np.exp(17.625 * td / (243.04 + td) - 17.625 * t / (243.04 + t))


# HDC1080: draw errors / simulate
def hdc1080_params(rng):
    return dict(t_bias=U(rng, -0.5, 0.5), sun_heat=U(rng, 1.0, 2.5), rh_bias=U(rng, -2, 2))


def hdc1080(t_true, td_true, sun, wind_at_sensor, p, rng):
    n = len(t_true)
    heating = p["sun_heat"] * (np.nan_to_num(sun) / 1000) / (1 + np.nan_to_num(wind_at_sensor))
    t = t_true + p["t_bias"] + heating + rng.normal(0, 0.3, n)
    rh = rh_from_dewpoint(td_true, t) + p["rh_bias"] + rng.normal(0, 2, n)
    return t, np.clip(rh, 0, 100)


# BMP280: draw errors / simulate
def bmp280_params(rng):
    return dict(height=U(rng, 3, 25), bias=U(rng, -1, 1))


def bmp280(p_true, p, rng):
    return p_true - p["height"] / 8.3 + p["bias"] + rng.normal(0, 0.1, len(p_true))


# anemometer: draw errors / simulate
def wind_params(rng):
    return dict(shelter=U(rng, 0.4, 1.0), offset=U(rng, -15, 15))


def anemometer(ws_true, wd_true, p, rng):
    n = len(ws_true)
    ws = p["shelter"] * ws_true + rng.normal(0, 0.3, n)
    ws = np.where(ws < 0.3, 0, ws)
    wd = (wd_true + p["offset"] + rng.normal(0, 15, n)) % 360
    return ws, wd


# rain board: draw errors / simulate
def rain_params(rng):
    return dict(dry_hours=U(rng, 1, 2))


def rain_board(precip, rh, p, rng):
    raining = np.nan_to_num(precip) > 0.1
    hours_since_rain = np.full(len(precip), np.inf)
    last_rain = -np.inf
    for i in range(len(precip)):
        if raining[i]:
            last_rain = i
        hours_since_rain[i] = i - last_rain
    wet = np.where(raining, 1.0, np.clip(p["dry_hours"] - (hours_since_rain - 1), 0, 1))
    fog = (wet == 0) & (np.nan_to_num(rh) > 90) & (rng.random(len(precip)) < 0.02)
    return np.where(fog, rng.uniform(0.2, 1.0, len(precip)), wet)


# BH1750: draw errors / simulate
def bh1750_params(rng):
    return dict(gain=U(rng, 0.6, 1.2), smog_dim=U(rng, 0.0, 0.2))


def bh1750(sun, pm25_true, p, rng):
    lux = p["gain"] * 108 * sun * np.exp(-p["smog_dim"] * np.nan_to_num(pm25_true) / 100)
    lux = lux + rng.normal(0, 50, len(sun))
    lux = np.where(sun < 1, 0, lux)
    return np.clip(lux, 0, 121557)


# MQ-7 / MQ-135: draw errors
def mq_params(rng, r0_median):
    return dict(r0=r0_median * np.exp(rng.normal(0, 0.5)), b=None, c0=U(rng, 0.3, 1.0),
                g_t=U(rng, -0.016, -0.007), g_rh=U(rng, -0.008, -0.003), drift=U(rng, -0.001, 0.001))


def mq7_params(rng):
    p = mq_params(rng, 20000)
    p["b"] = U(rng, -0.70, -0.45)
    return p


def mq135_params(rng):
    p = mq_params(rng, 30000)
    p.update(b=U(rng, -0.55, -0.40), w_co=U(rng, 0.1, 0.5), w_voc=U(rng, 0.5, 1.5), g_ox=U(rng, 0.0, 0.15))
    return p


# metal-oxide sensor: gas -> resistance (formula above)
def mos_resistance(gas_ppm, t, rh, days, p, rng):
    ln_rs = (np.log(p["r0"]) + p["b"] * np.log(gas_ppm + p["c0"]) + p["g_t"] * (t - 20)
             + p["g_rh"] * (rh - 65) + p["drift"] * days + rng.normal(0, 0.05, len(gas_ppm)))
    return np.exp(ln_rs)


def mq7(co_ppm, t, rh, days, p, rng):
    return mos_resistance(co_ppm, t, rh, days, p, rng)


def mq135(co_ppm, voc_ppb, no2_ppb, t, rh, days, p, rng):
    gas = p["w_co"] * co_ppm + p["w_voc"] * voc_ppb / 1000
    return mos_resistance(gas, t, rh, days, p, rng) * np.exp(p["g_ox"] * np.log1p(np.nan_to_num(no2_ppb) / 20))


# SGP41: draw errors / simulate
def sgp41_params(rng):
    return dict(voc_base=U(rng, 25000, 35000), voc_k=U(rng, 1040, 1507), voc_k_ox=U(rng, 0, 700),
                nox_base=U(rng, 14000, 17000), nox_k=U(rng, 1437, 3039), nox_w_o3=U(rng, 2.0, 5.0),
                drift=U(rng, -5, 5))


def sgp41(voc_ppb, no2_ppb, o3_ppb, days, p, rng):
    n = len(voc_ppb)
    oxidants = np.nan_to_num(no2_ppb) + np.nan_to_num(o3_ppb)
    voc = (p["voc_base"] - p["voc_k"] * np.log1p(voc_ppb / 100) + p["voc_k_ox"] * np.log1p(oxidants / 50)
           + p["drift"] * days + rng.normal(0, 50, n))
    nox = (p["nox_base"] + p["nox_k"] * np.log1p((no2_ppb + p["nox_w_o3"] * o3_ppb) / 10)
           + p["drift"] * days + rng.normal(0, 30, n))
    return np.clip(np.round(voc), 0, 65535), np.clip(np.round(nox), 0, 65535)


# SPS30 faults: random missing hours + long outages
def sps30_faults(n, rng):
    missing = rng.random(n) < 0.01
    outage = np.zeros(n, dtype=bool)
    for _ in range(rng.poisson(n / 8766)):            # ~1 outage per year
        length = round(U(rng, 6, 48))
        start = int(rng.integers(0, n - length))
        outage[start:start + length] = True
    return missing, outage


# humidity correction (the device uses one fixed kappa = 0.4)
def humidity_correct(pm, rh):
    aw = np.clip(rh / 100, 0, 0.95)
    growth = 1 + (0.4 / 1.65) / (1 / np.maximum(aw, 1e-9) - 1)
    return pm / growth


# clean-air baseline of a gas sensor: quantile of the previous 7 days
def rolling_baseline(x, q):
    return pd.Series(x).shift(1).rolling(168, min_periods=24).quantile(q).to_numpy()


# PM state from how many hours the SPS30 has been silent
def pm_state(pm_measured):
    m = pd.Series(pm_measured)
    hours_lost = (~m).groupby(m.cumsum()).cumsum().to_numpy()
    return np.select([hours_lost == 0, hours_lost <= 1, hours_lost <= 24],
                     ["ok", "filled_le_1h", "fallback_model_b"], "unavailable")


# CPCB sub-index of one 24-h mean concentration
AQI_BANDS = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
PM25_LIMITS = [30, 60, 90, 120, 250, 380]
PM10_LIMITS = [50, 100, 250, 350, 430, 510]


def sub_index(c, limits):
    previous = 0.0
    for top, (i_lo, i_hi) in zip(limits, AQI_BANDS):
        if c <= top:
            return i_lo + (i_hi - i_lo) / (top - previous) * (c - previous)
        previous = top
    return 500.0


assert sub_index(30, PM25_LIMITS) == 50 and sub_index(60, PM25_LIMITS) == 100 and sub_index(100, PM10_LIMITS) == 100


# AQI of every hour from the last 24 hours (NaN if fewer than 16 measured hours)
def rolling_aqi(pm25, pm10, measured):
    p25 = pd.Series(np.where(measured, pm25, np.nan))
    p10 = pd.Series(np.where(measured, pm10, np.nan))
    mean25, mean10 = p25.rolling(24, min_periods=1).mean(), p10.rolling(24, min_periods=1).mean()
    count = p25.rolling(24, min_periods=1).count()
    aqi = np.full(len(p25), np.nan)
    for i in np.flatnonzero(count >= 16):
        aqi[i] = round(max(sub_index(mean25[i], PM25_LIMITS), sub_index(mean10[i], PM10_LIMITS)))
    return aqi


# the true values every unit at this station shares (air, weather, kappa, VOC)
def station_truth(clean, network_typical, station):
    rng = make_rng(station, -1)
    truth = clean[["ts", "pm25", "pm10", "t", "rh", "td", "p", "wind_u", "wind_v", "precip", "swrad"]].copy()
    for gas in ("co_ppm", "no2_ppb", "o3_ppb"):
        truth[gas] = fill_gas(clean[gas], clean.ts, network_typical[gas])
    truth["voc_ppb"] = 20 + U(rng, 0.03, 0.10) * truth.co_ppm * 1000
    truth["kappa"] = kappa_series(len(truth), rng)
    truth["ws"] = np.hypot(truth.wind_u, truth.wind_v)
    truth["wd"] = np.rad2deg(np.arctan2(-truth.wind_u, -truth.wind_v)) % 360
    return truth


# draw all errors of one unit (once, for its whole life)
def draw_unit(rng):
    return {"sps30": sps30_params(rng), "hdc1080": hdc1080_params(rng), "bmp280": bmp280_params(rng),
            "wind": wind_params(rng), "rain": rain_params(rng), "bh1750": bh1750_params(rng),
            "mq7": mq7_params(rng), "mq135": mq135_params(rng), "sgp41": sgp41_params(rng)}


# every sensor of one unit reads the true values
def read_sensors(truth, p, rng):
    days = ((truth.ts - truth.ts.iloc[0]).dt.total_seconds() / 86400).to_numpy()
    g = {c: truth[c].to_numpy(float) for c in truth.columns if c != "ts"}
    r = pd.DataFrame({"ts": truth.ts})
    rh_inside = np.clip(rh_from_dewpoint(g["td"], g["t"] + p["sps30"]["dt_inside"]), 0, 100)
    r["pm1"], r["pm25"], r["pm10"] = sps30(g["pm25"], g["pm10"], rh_inside, g["kappa"], p["sps30"], rng)
    ws, wd = anemometer(g["ws"], g["wd"], p["wind"], rng)
    r["t"], r["rh"] = hdc1080(g["t"], g["td"], g["swrad"], p["wind"]["shelter"] * g["ws"], p["hdc1080"], rng)
    r["p"] = bmp280(g["p"], p["bmp280"], rng)
    r["wind_speed"] = ws
    r["wind_u"], r["wind_v"] = -ws * np.sin(np.deg2rad(wd)), -ws * np.cos(np.deg2rad(wd))
    r["rain_frac"] = rain_board(g["precip"], g["rh"], p["rain"], rng)
    r["lux"] = bh1750(g["swrad"], g["pm25"], p["bh1750"], rng)
    r["mq7_rs"] = mq7(g["co_ppm"], r.t.to_numpy(), r.rh.to_numpy(), days, p["mq7"], rng)
    r["mq135_rs"] = mq135(g["co_ppm"], g["voc_ppb"], g["no2_ppb"], r.t.to_numpy(), r.rh.to_numpy(), days, p["mq135"], rng)
    r["sraw_voc"], r["sraw_nox"] = sgp41(g["voc_ppb"], g["no2_ppb"], g["o3_ppb"], days, p["sgp41"], rng)
    return r


# the ESP32's hourly processing (section 3)
def process_on_device(r, missing, outage, truth):
    have_truth = truth.pm25.notna().to_numpy() & truth.pm10.notna().to_numpy()
    r["pm_measured"] = have_truth & ~missing & ~outage
    r["pm_state"] = pm_state(r.pm_measured.to_numpy())
    one_hour_hole = r.pm_state.eq("filled_le_1h").to_numpy()
    for c in ("pm1", "pm25", "pm10"):
        value = r[c].where(r.pm_measured)
        r[c] = np.where(one_hour_hole, value.ffill(limit=1), value)
    r["pm25_corr"] = humidity_correct(r.pm25, r.rh)
    r["pm10_corr"] = humidity_correct(r.pm10, r.rh)
    r["coarse_ratio"] = (r.pm10 - r.pm25) / np.maximum(r.pm1, 1)
    r["dust"] = (r.coarse_ratio > 0.5) & (r.rh < 60) & r.pm_measured
    r["Rbase7"], r["Rbase135"] = rolling_baseline(r.mq7_rs, 0.90), rolling_baseline(r.mq135_rs, 0.90)
    r["base_voc"], r["base_nox"] = rolling_baseline(r.sraw_voc, 0.90), rolling_baseline(r.sraw_nox, 0.10)
    r["aqi_pm"] = rolling_aqi(r.pm25_corr.to_numpy(), r.pm10_corr.to_numpy(), r.pm_measured.to_numpy())
    r["sim_outage"] = outage
    return r


# one virtual unit: errors -> sensor readings -> faults -> device processing (+ true PM as the answer)
def simulate_unit(truth, station, unit):
    rng = make_rng(station, unit)
    params = draw_unit(rng)
    readings = read_sensors(truth, params, rng)
    missing, outage = sps30_faults(len(truth), rng)
    record = process_on_device(readings, missing, outage, truth)
    record["true_pm25"], record["true_pm10"] = truth.pm25.to_numpy(), truth.pm10.to_numpy()   # TARGET only
    return record, params


# the whole box for one station: X (clean hourly table) -> Y (one unit's log)
def simulate(X, station, unit, network_typical=None):
    """network_typical = typical gas level per (month, hour) over all stations, used when this station misses a gas.
    If not given, the station's own typical values are used."""
    if network_typical is None:
        local = X.ts.dt.tz_convert("Asia/Kolkata")
        network_typical = {g: X[g].groupby([local.dt.month, local.dt.hour]).median() for g in ("co_ppm", "no2_ppb", "o3_ppb")}
    truth = station_truth(X, network_typical, station)
    Y, _ = simulate_unit(truth, station, unit)
    return Y


# command line: python simulator.py <clean table> --unit 0 -o <output>
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="X (clean hourly station table) -> Y (simulated ESP32 log)")
    ap.add_argument("x_file", help="a table from data/2_clean/ (.parquet or .csv)")
    ap.add_argument("--station", help="station id (default: file name)")
    ap.add_argument("--unit", type=int, default=0, help="which virtual unit (0, 1, 2 ...)")
    ap.add_argument("-o", "--out", default="Y.parquet")
    a = ap.parse_args()
    X = pd.read_parquet(a.x_file) if a.x_file.endswith(".parquet") else pd.read_csv(a.x_file, parse_dates=["ts"])
    station = a.station or a.x_file.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
    Y = simulate(X, station, a.unit)
    Y.to_parquet(a.out, index=False) if a.out.endswith(".parquet") else Y.to_csv(a.out, index=False)
    print(f"{station} unit {a.unit}: {len(Y)} hours -> {a.out}")
