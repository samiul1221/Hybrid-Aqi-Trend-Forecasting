"""Regenerate shared/fixtures/{aqi_vectors,ist_bucket_fixture}.json.

    uv run python shared/fixtures/make_fixtures.py

AQI expected values come from aqi_shared.aqi (regression + C parity target); their correctness is
checked by hand-computed anchors in shared/tests/test_aqi.py. IST expected values come from
zoneinfo('Asia/Kolkata'), independent of aqi_shared.time_utils.
Review the git diff of the JSON after regenerating: a changed value is a behaviour change.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from aqi_shared import aqi

HERE = Path(__file__).parent
NAN = float("nan")


def _dump(path: Path, obj: object) -> None:
    text = json.dumps(obj, indent=1, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _series(base: float, amp: float, n: int = 24, phase: float = 0.0) -> list[float]:
    return [round(base + amp * math.sin(i / 3.0 + phase), 3) for i in range(n)]


def _nan_to_none(xs: list[float]) -> list[float | None]:
    return [None if math.isnan(x) else x for x in xs]


def aqi_vectors() -> dict[str, object]:
    sub = []
    pm25_c = [0, 15, 30, 30.5, 31, 45, 60, 60.2, 75, 90, 90.5, 105, 120, 121, 185, 250, 250.1,
              300, 380, 381, 1000, -5]  # fmt: skip
    pm10_c = [0, 25, 50, 50.5, 51, 75, 100, 101, 175, 250, 250.5, 300, 350, 351, 390, 430,
              430.5, 470, 510, 600, -1]  # fmt: skip
    for pol, cs in (("pm25", pm25_c), ("pm10", pm10_c)):
        for c in cs:
            v = aqi.sub_index(pol, float(c))
            sub.append({"pollutant": pol, "c": float(c), "expected": v,
                        "category": aqi.category(v)})  # fmt: skip

    t, f = True, False
    full = [t] * 24
    cases_24h: list[tuple[str, list[float], list[float], list[bool], list[bool]]] = [
        ("constant_moderate", [75.0] * 24, [120.0] * 24, full, full),
        ("pm10_dominates", [20.0] * 24, [300.0] * 24, full, full),
        ("pm25_dominates", [140.0] * 24, [160.0] * 24, full, full),
        ("diurnal_series", _series(95, 40), _series(180, 70, phase=0.5), full, full),
        ("exactly_16_measured", _series(60, 20), _series(110, 30), [t] * 16 + [f] * 8, full),
        ("only_15_measured", _series(60, 20), _series(110, 30), [f] * 9 + [t] * 15, full),
        ("pm10_not_ready", [50.0] * 24, [90.0] * 24, full, [t] * 15 + [f] * 9),
        ("unmeasured_spike_ignored", [40.0] * 23 + [900.0], [70.0] * 23 + [950.0],
         [t] * 23 + [f], [t] * 23 + [f]),
        ("nan_slots_ignored", [NAN] * 4 + [55.0] * 20, [NAN] * 4 + [95.0] * 20, full, full),
        ("severe_cap", [600.0] * 24, [800.0] * 24, full, full),
        ("band_edge_250", [250.0] * 24, [100.0] * 24, full, full),
        ("clean_air", [8.0] * 24, [20.0] * 24, full, full),
    ]  # fmt: skip
    vec_24h = []
    for name, p25, p10, m25, m10 in cases_24h:
        r = aqi.aqi_pm_24h(p25, p10, m25, m10)
        vec_24h.append({
            "name": name, "pm25": _nan_to_none(p25), "pm10": _nan_to_none(p10),
            "measured_pm25": m25, "measured_pm10": m10,
            "expected": {"valid": r.valid, "aqi": r.aqi, "aqi_int": r.aqi_int,
                         "sub_pm25": r.sub_pm25, "sub_pm10": r.sub_pm10,
                         "valid_hours": r.valid_hours, "category": r.category},
        })  # fmt: skip

    cases_trend: list[tuple[str, list[float], list[float], list[bool], list[float], float | None]] = [
        ("rising", [80.0] * 24, [150.0] * 24, full, [0.3, 0.6, 0.9, 0.2, 0.4, 0.6], None),
        ("falling", [140.0] * 24, [260.0] * 24, full, [-0.5, -0.9, -1.2, -0.4, -0.7, -1.0], None),
        ("stable", [60.0] * 24, [100.0] * 24, full, [0.01, 0.02, 0.02, 0.0, 0.01, 0.01], None),
        ("band_warning_up", [58.0] * 24, [95.0] * 24, full, [0.8, 1.2, 1.5, 0.6, 0.9, 1.2], None),
        ("now_not_ready", [70.0] * 24, [120.0] * 24, [f] * 10 + [t] * 14,
         [0.1, 0.2, 0.3, 0.1, 0.2, 0.3], None),
        ("model_b_last_valid_base", [90.0] * 24, [170.0] * 24, full,
         [0.05, 0.1, 0.15, 0.05, 0.1, 0.15], 70.0),
    ]  # fmt: skip
    vec_trend = []
    for name, p25, p10, m, y, base in cases_trend:
        b10 = None if base is None else base * 1.8
        r = aqi.aqi_trend(p25, p10, m, m, y, pm25_base=base, pm10_base=b10)
        vec_trend.append({
            "name": name, "pm25": p25, "pm10": p10, "measured_pm25": m, "measured_pm10": m,
            "y": y, "pm25_base": base, "pm10_base": b10, "threshold": aqi.TREND_THRESHOLD,
            "expected": {"pm25": list(r.pm25), "pm10": list(r.pm10), "aqi_now": r.aqi_now,
                         "aqi": list(r.aqi), "aqi_int": list(r.aqi_int), "delta": list(r.delta),
                         "trend": r.trend, "band_warning": r.band_warning},
        })  # fmt: skip

    return {
        "_doc": "CPCB AQI test vectors [S06.6]/[S07]. Python and firmware C must both pass. "
        "Floats: abs tol 1e-6; ints/strings/bools exact. null in pm arrays = NaN.",
        "rounding": "aqi_int = floor(aqi + 0.5)",
        "subindex": sub,
        "aqi_24h": vec_24h,
        "trend": vec_trend,
    }


def ist_fixture() -> dict[str, object]:
    kolkata = ZoneInfo("Asia/Kolkata")
    instants: list[datetime] = []
    start = datetime(2024, 2, 28, 12, 0, 0, tzinfo=UTC)  # 48 h across the leap day (IST)
    step = timedelta(minutes=7, seconds=13)
    t = start
    while t < start + timedelta(hours=48):
        instants.append(t)
        t += step
    for s in ("2023-12-31T18:29:59Z", "2023-12-31T18:30:00Z", "2023-12-31T18:30:01Z",
              "2024-03-01T04:29:59Z", "2024-03-01T04:30:00Z", "2025-06-15T23:59:59Z",
              "2025-06-16T00:00:00Z", "2026-10-01T04:30:00Z", "2026-10-01T05:39:00Z"):  # fmt: skip
        instants.append(datetime.fromisoformat(s.replace("Z", "+00:00")))
    rows = []
    for dt in instants:
        local = dt.astimezone(kolkata)
        bucket = local.replace(minute=0, second=0, microsecond=0).astimezone(UTC)
        rows.append({
            "utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "epoch": int(dt.timestamp()),
            "bucket_ts": bucket.strftime("%Y-%m-%dT%H:%M:%SZ"), "bucket_epoch": int(bucket.timestamp()),
            "ist_hour": local.hour, "ist_dow": local.weekday(), "ist_doy": local.timetuple().tm_yday,
        })  # fmt: skip
    return {
        "_doc": "IST clock-hour buckets (I02). Expected from zoneinfo Asia/Kolkata. "
        "dow 0=Monday; doy 1-based; ist_* describe the input instant.",
        "rows": rows,
    }


if __name__ == "__main__":
    _dump(HERE / "aqi_vectors.json", aqi_vectors())
    _dump(HERE / "ist_bucket_fixture.json", ist_fixture())
    print("fixtures written")
