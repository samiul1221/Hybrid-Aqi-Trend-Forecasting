"""CPCB NAQI (India) PM-based AQI [S06.6] and AQI trend roll [S07]. Invariant I01.

Reference implementation for Python (training, cloud). The firmware C twin must match
``shared/fixtures/aqi_vectors.json``. Interpretation choices: docs/decisions/0003-cpcb-aqi-rules.md
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Literal

Pollutant = Literal["pm25", "pm10"]
Trend = Literal["rising", "falling", "stable"]

_TABLE = json.loads(
    resources.files(__package__).joinpath("cpcb_breakpoints.json").read_text(encoding="utf-8")
)
CATEGORIES: tuple[str, ...] = tuple(_TABLE["categories"])
AQI_BANDS: tuple[tuple[int, int], ...] = tuple((a, b) for a, b in _TABLE["aqi_bands"])
BREAKPOINTS: dict[str, tuple[tuple[float, float], ...]] = {
    p: tuple((float(lo), float(hi)) for lo, hi in _TABLE[p]) for p in ("pm25", "pm10")
}
AQI_CAP: int = _TABLE["aqi_cap"]
MIN_MEASURED_HOURS: int = _TABLE["validity_min_measured_hours"]
WINDOW_HOURS: int = _TABLE["window_hours"]
TREND_THRESHOLD: float = float(_TABLE["trend_threshold"])


def sub_index(pollutant: Pollutant, c: float) -> float:
    """Sub-index for a 24 h mean concentration (ug/m3).

    Gap rule [S06.6]: band k>0 uses the previous band's B_hi as B_lo, so every C > 0 falls in
    exactly one band (upper bound inclusive) and I rises monotonically. C above the Severe upper
    limit is capped at 500. Negative C is clipped to 0.
    """
    c = max(c, 0.0)
    bands = BREAKPOINTS[pollutant]
    prev_hi = 0.0
    for (_, b_hi), (i_lo, i_hi) in zip(bands, AQI_BANDS, strict=True):
        if c <= b_hi:
            return (i_hi - i_lo) / (b_hi - prev_hi) * (c - prev_hi) + i_lo
        prev_hi = b_hi
    return float(AQI_CAP)


def category(aqi: float) -> str:
    """CPCB category name for an AQI value."""
    for (_, i_hi), name in zip(AQI_BANDS, CATEGORIES, strict=True):
        if aqi <= i_hi:
            return name
    return CATEGORIES[-1]


def category_index(aqi: float) -> int:
    """0 = Good ... 5 = Severe."""
    return CATEGORIES.index(category(aqi))


def round_aqi(aqi: float) -> int:
    """Integer AQI for display/wire: round half up (identical in C: floor(x + 0.5))."""
    return math.floor(aqi + 0.5)


@dataclass(frozen=True)
class AqiResult:
    valid: bool  # False = 'not_ready'
    aqi: float | None
    aqi_int: int | None
    sub_pm25: float | None
    sub_pm10: float | None
    valid_hours: int  # min over pollutants of measured hours in the window
    category: str | None


def _mean_measured(values: Sequence[float], measured: Sequence[bool]) -> tuple[float, int]:
    kept = [v for v, m in zip(values, measured, strict=True) if m and not math.isnan(v)]
    return (sum(kept) / len(kept) if kept else math.nan), len(kept)


def aqi_pm_24h(
    pm25: Sequence[float],
    pm10: Sequence[float],
    measured_pm25: Sequence[bool],
    measured_pm10: Sequence[bool],
) -> AqiResult:
    """PM-based AQI from 24 hourly slots (oldest first) of humidity-corrected PM.

    Means use measured hours only; each pollutant needs >= 16 measured hours, and both are
    required for a valid AQI (they come from the same SPS30). NaN counts as not measured.
    """
    for seq in (pm25, pm10, measured_pm25, measured_pm10):
        if len(seq) != WINDOW_HOURS:
            raise ValueError(f"expected {WINDOW_HOURS} hourly slots, got {len(seq)}")
    mean25, n25 = _mean_measured(pm25, measured_pm25)
    mean10, n10 = _mean_measured(pm10, measured_pm10)
    n = min(n25, n10)
    if n < MIN_MEASURED_HOURS:
        return AqiResult(False, None, None, None, None, n, None)
    s25, s10 = sub_index("pm25", mean25), sub_index("pm10", mean10)
    aqi = max(s25, s10)
    return AqiResult(True, aqi, round_aqi(aqi), s25, s10, n, category(aqi))


def aqi_full(result: AqiResult, gas_subindex: dict[str, float] | None) -> float | None:
    """max(PM sub-indices, cloud NO2/O3/CO sub-indices) [S06.6]; PM-based AQI if no gas data."""
    if not result.valid or result.aqi is None:
        return None
    gases = [float(v) for k, v in (gas_subindex or {}).items() if k in ("no2", "o3", "co")]
    return max([result.aqi, *gases])


def pm_from_y(pm_base: float, y: float) -> float:
    """[S07] step 1: PM[t+h] = exp(ln(1+PM_base) + y) - 1."""
    return math.exp(math.log1p(pm_base) + y) - 1.0


@dataclass(frozen=True)
class TrendResult:
    pm25: tuple[float, ...]  # predicted PM2.5 for h = 1..H
    pm10: tuple[float, ...]
    aqi_now: float | None
    aqi: tuple[float | None, ...]  # AQI[t+h]
    aqi_int: tuple[int | None, ...]
    delta: tuple[float | None, ...]  # AQI[t+h] - AQI[t]
    trend: Trend | None  # None when AQI[t] or AQI[t+H] is not ready
    band_warning: bool  # predicted category at any h worse than current


def aqi_trend(
    pm25: Sequence[float],
    pm10: Sequence[float],
    measured_pm25: Sequence[bool],
    measured_pm10: Sequence[bool],
    y: Sequence[float],
    pm25_base: float | None = None,
    pm10_base: float | None = None,
    threshold: float = TREND_THRESHOLD,
) -> TrendResult:
    """[S07] AQI trend: predicted hourly PM rolled into the 24 h buffer.

    ``y`` is pollutant-major ``[pm25_h1..hH, pm10_h1..hH]`` (log-change targets, I04).
    Bases default to the newest slot (Model A); Model B passes the last valid PM [S14].
    Predicted hours count as valid slots in the rolled window.
    """
    h_out = len(y) // 2
    if len(y) != 2 * h_out or h_out == 0:
        raise ValueError("y must be pollutant-major with 2*H values")
    b25 = pm25[-1] if pm25_base is None else pm25_base
    b10 = pm10[-1] if pm10_base is None else pm10_base
    f25 = tuple(pm_from_y(b25, y[h]) for h in range(h_out))
    f10 = tuple(pm_from_y(b10, y[h_out + h]) for h in range(h_out))

    now = aqi_pm_24h(pm25, pm10, measured_pm25, measured_pm10)
    aqis: list[float | None] = []
    for h in range(1, h_out + 1):
        r = aqi_pm_24h(
            [*pm25[h:], *f25[:h]],
            [*pm10[h:], *f10[:h]],
            [*measured_pm25[h:], *([True] * h)],
            [*measured_pm10[h:], *([True] * h)],
        )
        aqis.append(r.aqi)

    deltas = tuple(None if (a is None or now.aqi is None) else a - now.aqi for a in aqis)
    last = deltas[-1]
    trend: Trend | None
    if last is None:
        trend = None
    elif last >= threshold:
        trend = "rising"
    elif last <= -threshold:
        trend = "falling"
    else:
        trend = "stable"
    band_warning = now.aqi is not None and any(
        a is not None and category_index(a) > category_index(now.aqi) for a in aqis
    )
    return TrendResult(
        pm25=f25,
        pm10=f10,
        aqi_now=now.aqi,
        aqi=tuple(aqis),
        aqi_int=tuple(None if a is None else round_aqi(a) for a in aqis),
        delta=deltas,
        trend=trend,
        band_warning=band_warning,
    )
