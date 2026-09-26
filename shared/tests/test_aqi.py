import importlib.util
import itertools
import json
import math
from pathlib import Path
from typing import Any

import pytest

from aqi_shared import aqi

TOL = 1e-6
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _nan(xs: list[float | None]) -> list[float]:
    return [math.nan if x is None else x for x in xs]


def _close(a: float | None, b: float | None) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= TOL)


# Hand-computed anchors, independent of the fixture generator (ADR 0003 gap rule).
@pytest.mark.parametrize(
    ("pol", "c", "expected"),
    [
        ("pm25", 30, 50.0),
        ("pm25", 30.5, 51 + 49 / 30 * 0.5),
        ("pm25", 60, 100.0),
        ("pm25", 75, 101 + 99 / 30 * 15),
        ("pm25", 250, 400.0),  # boundary belongs to Very Poor
        ("pm25", 380, 500.0),
        ("pm25", 5000, 500.0),
        ("pm10", 100, 100.0),
        ("pm10", 120, 101 + 99 / 150 * 20),
        ("pm10", 300, 201 + 99 / 100 * 50),
        ("pm10", 430, 400.0),
        ("pm10", -3, 0.0),
    ],
)
def test_subindex_anchors(pol: aqi.Pollutant, c: float, expected: float) -> None:
    assert aqi.sub_index(pol, c) == pytest.approx(expected, abs=TOL)


def test_subindex_monotonic() -> None:
    for pol in ("pm25", "pm10"):
        vals = [aqi.sub_index(pol, c / 10) for c in range(0, 7000)]
        assert all(b >= a for a, b in itertools.pairwise(vals))


def test_category_and_rounding() -> None:
    got = [aqi.category(x) for x in (0, 50, 51, 100.01, 200, 301, 400, 401, 500)]
    assert got == [
        "Good", "Good", "Satisfactory", "Moderate", "Moderate", "Very_Poor", "Very_Poor",
        "Severe", "Severe",
    ]  # fmt: skip
    assert [aqi.round_aqi(x) for x in (150.5, 150.49, 0.5, 399.5)] == [151, 150, 1, 400]


def test_vector_count(aqi_vectors: Any) -> None:
    n = len(aqi_vectors["subindex"]) + len(aqi_vectors["aqi_24h"]) + len(aqi_vectors["trend"])
    assert n >= 50


def test_subindex_vectors(aqi_vectors: Any) -> None:
    for v in aqi_vectors["subindex"]:
        got = aqi.sub_index(v["pollutant"], v["c"])
        assert _close(got, v["expected"]), v
        assert aqi.category(got) == v["category"]


def test_24h_vectors(aqi_vectors: Any) -> None:
    for v in aqi_vectors["aqi_24h"]:
        r = aqi.aqi_pm_24h(_nan(v["pm25"]), _nan(v["pm10"]), v["measured_pm25"], v["measured_pm10"])
        e = v["expected"]
        got = (r.valid, r.aqi_int, r.valid_hours, r.category)
        assert got == (e["valid"], e["aqi_int"], e["valid_hours"], e["category"]), v["name"]
        for k in ("aqi", "sub_pm25", "sub_pm10"):
            assert _close(getattr(r, k), e[k]), (v["name"], k)


def test_trend_vectors(aqi_vectors: Any) -> None:
    for v in aqi_vectors["trend"]:
        r = aqi.aqi_trend(
            v["pm25"], v["pm10"], v["measured_pm25"], v["measured_pm10"], v["y"],
            v["pm25_base"], v["pm10_base"], v["threshold"],
        )  # fmt: skip
        e = v["expected"]
        got = (r.trend, r.band_warning, list(r.aqi_int))
        assert got == (e["trend"], e["band_warning"], e["aqi_int"]), v["name"]
        for k in ("pm25", "pm10", "aqi", "delta"):
            assert all(_close(a, b) for a, b in zip(getattr(r, k), e[k], strict=True)), k
        assert _close(r.aqi_now, e["aqi_now"])


def test_pm_from_y_roundtrip() -> None:
    assert aqi.pm_from_y(80.0, math.log1p(100.0) - math.log1p(80.0)) == pytest.approx(100.0)


def test_aqi_full() -> None:
    r = aqi.aqi_pm_24h([75.0] * 24, [120.0] * 24, [True] * 24, [True] * 24)
    assert aqi.aqi_full(r, None) == r.aqi
    assert aqi.aqi_full(r, {"no2": 180.0, "o3": 41.0}) == 180.0


def test_window_length_checked() -> None:
    with pytest.raises(ValueError):
        aqi.aqi_pm_24h([1.0] * 23, [1.0] * 24, [True] * 24, [True] * 24)


def test_fixture_matches_generator() -> None:
    """Committed fixture must equal what the current code generates (catches silent drift)."""
    spec = importlib.util.spec_from_file_location("mk", FIXTURES / "make_fixtures.py")
    assert spec and spec.loader
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    on_disk = json.loads((FIXTURES / "aqi_vectors.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(mk.aqi_vectors())) == on_disk
