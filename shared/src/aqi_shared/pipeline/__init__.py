"""Hourly on-device processing shared by firmware (C twin), simulator and cloud ([S06.4]-[S06.7], [C06]).

Functions accept floats or numpy arrays. Calibration values come from calibration.json (I06).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

DEFAULT_CALIBRATION = Path(__file__).resolve().parents[3] / "calibration" / "default.json"

FloatArray = npt.NDArray[np.float64]
ArrayLike = float | FloatArray

# [S06.7] flag bits per variable
MEASURED, FILLED, MISSING, HIGH_HUMIDITY, DUST, WARMING, FAN_CLEANING, SENSOR_FAULT = (
    1 << i for i in range(8)
)
FLAG_NAMES = ("measured", "filled", "missing", "high_humidity", "dust", "warming", "fan_cleaning",
              "sensor_fault")  # fmt: skip


def load_calibration(path: Path = DEFAULT_CALIBRATION) -> dict[str, Any]:
    cal: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return cal


def growth_factor(rh: ArrayLike, kappa: ArrayLike, aw_max: float = 0.95) -> FloatArray:
    """kappa-Koehler mass growth C = 1 + (kappa/1.65) / (1/a_w - 1), a_w = clamp(RH/100, 0, aw_max).

    Crilley et al. 2018 AMT Eq. 6 (1.65 = dry particle density g/cm3) [S06.4].
    """
    aw = np.clip(np.asarray(rh, dtype=float) / 100.0, 0.0, aw_max)
    out: FloatArray = 1.0 + (np.asarray(kappa, dtype=float) / 1.65) / (
        1.0 / np.maximum(aw, 1e-9) - 1.0
    )
    return out


def correct_pm(pm: ArrayLike, rh: ArrayLike, cal: dict[str, Any]) -> FloatArray:
    """Humidity-corrected (dry) PM = PM_raw / C(RH; kappa_cal) [S06.4]."""
    out: FloatArray = np.asarray(pm, dtype=float) / growth_factor(rh, cal["kappa"], cal["aw_max"])
    return out


def coarse_ratio(pm1: ArrayLike, pm25: ArrayLike, pm10: ArrayLike) -> FloatArray:
    """(PM10 - PM2.5) / max(PM1, 1) on raw sensor values [S06.5]."""
    out: FloatArray = (np.asarray(pm10, float) - np.asarray(pm25, float)) / np.maximum(
        np.asarray(pm1, float), 1.0
    )
    return out


def dust_flag(ratio: ArrayLike, rh: ArrayLike, cal: dict[str, Any]) -> npt.NDArray[np.bool_]:
    """dust = ratio > R_dust AND RH < dust_rh_max [S06.5]."""
    out: npt.NDArray[np.bool_] = (np.asarray(ratio, float) > cal["R_dust"]) & (
        np.asarray(rh, float) < cal["dust_rh_max"]
    )
    return out


def rolling_baseline(
    values: FloatArray, quantile: float, window_h: int = 168, min_h: int = 24
) -> FloatArray:
    """Causal unit baseline [S08.1]: quantile of the previous `window_h` hourly values (excl. current).

    NaN until `min_h` valid past hours exist. Same rule on device (NVS) and in the simulator.
    """
    q = pd.Series(np.asarray(values, dtype=float)).shift(1).rolling(window_h, min_periods=min_h)
    out: FloatArray = q.quantile(quantile).to_numpy(dtype=float)
    return out
