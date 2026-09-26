"""Forward sensor models: true atmosphere -> what our fixed sensor pool would report [S10.3].

Pure, vectorised (numpy), hourly. `p` is one unit's drawn parameters for that sensor (params.draw_unit),
`rng` supplies measurement noise. NaN truth -> NaN reading. Equations, sources and deviations from
[S10.3]: docs/research_notes/simulator_parameters.md, ADR 0007.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from aqi_shared.pipeline import growth_factor

F = npt.NDArray[np.float64]
P = dict[str, Any]

# Magnus constants (Alduchov & Eskridge 1996), same as the base-table dew point
_A, _B = 17.625, 243.04


def rh_from_dewpoint(td: F, t: F) -> F:
    """Relative humidity (%) of air with dew point td at temperature t (°C)."""
    out: F = 100.0 * np.exp(_A * td / (_B + td) - _A * t / (_B + t))
    return out


def sps30(pm25_t: F, pm10_t: F, rh: F, kappa: F, p: P, dust_fine_fraction: float,
          rng: np.random.Generator) -> tuple[F, F, F]:  # fmt: skip
    """SPS30 PM1, PM2.5, PM10 (µg/m³) from BAM truth.

    Fine (non-dust) PM2.5 is seen with hygroscopic growth C(RH, kappa_aer) and scale s; dust is
    largely invisible (efficiency e); PM10 is extrapolated from the fine signal plus a small coarse
    share f [Kuula2020, SCAQMD, Gautam2025, IITD2025].
    """
    coarse = np.maximum(pm10_t - pm25_t, 0.0)
    dust25 = np.minimum(dust_fine_fraction * coarse, pm25_t)
    fine = pm25_t - dust25
    c = growth_factor(rh, kappa, p["aw_max_true"])
    n = len(pm25_t)
    humid = np.where(rh > 95.0, 1.0 + rng.uniform(0.0, p["rh95_bias"], n), 1.0)
    s, e = p["scale_s"], p["dust_efficiency"]
    b25 = s * (fine * c + e * dust25) * humid
    b10_extra = s * p["coarse_visibility_f"] * coarse
    b1 = (p["pm1_ratio_r"] * s * fine * c * humid) + 0.3 * s * e * dust25

    def noise(x: F) -> F:
        out: F = rng.normal(0.0, p["noise_rel"] * np.abs(x) + p["noise_abs"])
        return out

    pm25 = b25 + noise(b25)
    pm10 = (
        pm25 * (1.0 + p["pm10_fine_ext_q"])
        + b10_extra
        + rng.normal(0.0, p["noise_rel"] * b10_extra + 1e-9)
    )
    pm1 = b1 + noise(b1)
    lo, hi = p["clip"]
    pm25 = np.clip(pm25, lo, hi)
    pm10 = np.clip(np.maximum(pm10, pm25), lo, hi)  # SPS30 outputs are cumulative
    pm1 = np.clip(np.minimum(pm1, pm25), lo, hi)
    return pm1, pm25, pm10


def hdc1080(
    t_true: F, td_true: F, sw: F, ws_local: F, p: P, rng: np.random.Generator
) -> tuple[F, F]:
    """T (°C) and RH (%) inside a naturally ventilated shield.

    Solar heating dT = c SW/1000 / (1 + ws_local) [NakamuraMahrt2005]; RH follows the true dew point
    at the sensor temperature ([S10.3]) plus datasheet bias/noise.
    """
    n = len(t_true)
    sw0 = np.nan_to_num(np.maximum(sw, 0.0))
    dt_rad = p["rad_coeff_c"] * (sw0 / 1000.0) / (1.0 + np.nan_to_num(np.maximum(ws_local, 0.0)))
    t = t_true + p["t_bias"] + dt_rad + rng.normal(0.0, p["t_noise"], n)
    rh = rh_from_dewpoint(td_true, t) + p["rh_bias"] + rng.normal(0.0, p["rh_noise"], n)
    return t, np.clip(rh, 0.0, 100.0)


def bmp280(p_true: F, p: P, rng: np.random.Generator) -> F:
    """Pressure (hPa) at rooftop height: hydrostatic -1 hPa per 8.3 m + datasheet bias/noise."""
    out: F = (
        p_true - p["mount_height_m"] / 8.3 + p["bias"] + rng.normal(0.0, p["noise"], len(p_true))
    )
    return out


def wind(ws_true: F, wd_true: F, p: P, rng: np.random.Generator) -> tuple[F, F]:
    """Rooftop cup anemometer + vane: sheltering k, noise, stall, vane alignment offset."""
    n = len(ws_true)
    ws = p["shelter_k"] * ws_true + rng.normal(0.0, p["speed_noise"], n)
    ws = np.where(ws < p["stall_ms"], 0.0, ws)
    wd = np.mod(wd_true + p["dir_offset_deg"] + rng.normal(0.0, p["dir_noise_deg"], n), 360.0)
    return ws, wd


def rain(precip: F, rh: F, p: P, rng: np.random.Generator) -> F:
    """Rain-board wet fraction per hour: wet while raining, dries over `dry_hours`, dew/fog false wets."""
    n = len(precip)
    raining = np.nan_to_num(precip) > p["wet_mm_h"]
    since = np.full(n, np.inf)  # hours since the last raining hour
    last = -np.inf
    for i in range(n):
        if raining[i]:
            last = i
        since[i] = i - last
    frac = np.where(raining, 1.0, np.clip(p["dry_hours"] - (since - 1.0), 0.0, 1.0))
    false_wet = (
        (frac == 0) & (np.nan_to_num(rh) > p["false_wet_rh"]) & (rng.random(n) < p["false_wet_p"])
    )
    frac = np.where(false_wet, rng.uniform(0.2, 1.0, n), frac)
    return np.where(np.isnan(precip), np.nan, frac)


def bh1750(sw: F, pm25_t: F, p: P, rng: np.random.Generator) -> F:
    """Illuminance (lux) = g * K * SW * exp(-beta PM2.5/100) + noise, 0 at night [S10.3]."""
    base = (
        p["gain_g"]
        * p["efficacy_lm_per_w"]
        * sw
        * np.exp(-p["pm_beta"] * np.nan_to_num(pm25_t) / 100.0)
    )
    lux = base + rng.normal(0.0, p["noise"], len(sw))
    lux = np.where(sw < 1.0, 0.0, lux)
    lo, hi = p["clip"]
    return np.clip(lux, lo, hi)


def _mos_ln_rs(eq_ppm: F, t: F, rh: F, days: F, p: P, rng: np.random.Generator) -> F:
    """ln Rs = ln R0 + b ln(eq + c0) + gT (T-20) + gRH (RH-65) + drift*days + noise (datasheet form)."""
    out: F = (np.log(p["r0_ohm"]) + p["b"] * np.log(eq_ppm + p["c0_ppm"]) + p["g_t"] * (t - 20.0)
              + p["g_rh"] * (rh - 65.0) + p["drift_per_day"] * days
              + rng.normal(0.0, p["noise"], len(eq_ppm)))  # fmt: skip
    return out


def mq7(co_ppm: F, t: F, rh: F, days: F, p: P, rng: np.random.Generator) -> F:
    """MQ-7 Rs (Ohm) at the end of the 1.4-1.5 V phase; c0 = reducing-gas background (H2)."""
    return np.exp(_mos_ln_rs(co_ppm, t, rh, days, p, rng))


def mq135(co_ppm: F, voc_ppb: F, no2_ppb: F, t: F, rh: F, days: F, p: P,
          rng: np.random.Generator) -> F:  # fmt: skip
    """MQ-135 Rs (Ohm): reducing gases lower Rs; oxidising NO2 raises it (n-type SnO2)."""
    eq = p["w_co"] * co_ppm + p["w_voc"] * voc_ppb / 1000.0
    ln_rs = _mos_ln_rs(eq, t, rh, days, p, rng) + p["g_ox"] * np.log1p(
        np.nan_to_num(no2_ppb) / 20.0
    )
    rs: F = np.exp(ln_rs)
    return rs


def sgp41(
    voc_ppb: F, no2_ppb: F, o3_ppb: F, days: F, p: P, rng: np.random.Generator
) -> tuple[F, F]:
    """SGP41 SRAW_VOC, SRAW_NOX ticks (∝ log R). VOC ticks fall with VOC, NOx ticks rise with NO2 + O3."""
    n = len(voc_ppb)
    ox = np.nan_to_num(no2_ppb) + np.nan_to_num(o3_ppb)
    drift = p["drift_ticks_per_day"] * days
    voc = (p["voc_base"] - p["voc_k"] * np.log1p(voc_ppb / p["voc_v0_ppb"])
           + p["voc_k_ox"] * np.log1p(ox / 50.0) + drift + rng.normal(0.0, p["voc_noise"], n))  # fmt: skip
    ox_eq = no2_ppb + p["nox_w_o3"] * o3_ppb
    nox = (p["nox_base"] + p["nox_k"] * np.log1p(ox_eq / p["nox_n0_ppb"]) + drift
           + rng.normal(0.0, p["nox_noise"], n))  # fmt: skip
    lo, hi = p["clip"]
    voc_t: F = np.clip(np.round(voc), lo, hi)
    nox_t: F = np.clip(np.round(nox), lo, hi)
    return voc_t, nox_t
