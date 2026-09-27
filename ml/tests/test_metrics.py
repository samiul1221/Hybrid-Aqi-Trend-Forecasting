"""[S21] metric sanity: PM inverse, skill, Diebold-Mariano direction, AQI/band scores."""

import numpy as np
import pytest

from aqi_ml.eval import metrics as m


def test_pm_from_y_and_skill() -> None:
    base = np.array([[100.0, 200.0]] * 400)
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 0.3, (400, 6))
    good = y_true + rng.normal(0, 0.05, y_true.shape)
    out = m.point_metrics(y_true, good, base, 3)
    assert m.pm_from_y(np.array([100.0]), np.array([np.log(201 / 101)]))[0] == pytest.approx(200.0)
    assert all(
        v["skill"] > 0.5 and v["dm_vs_persistence"] < 0 and v["dm_p"] < 1e-6 for v in out.values()
    )
    perfect = m.point_metrics(y_true, y_true, base, 3)
    assert all(
        v["rmse"] == pytest.approx(0.0, abs=1e-9) and v["skill"] == pytest.approx(1.0)
        for v in perfect.values()
    )


def test_dm_is_symmetric() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 500), rng.normal(0, 2, 500)
    s1, p1 = m.diebold_mariano(a, b)
    s2, p2 = m.diebold_mariano(b, a)
    assert s1 < 0 < s2 and p1 == pytest.approx(p2)


def test_aqi_metrics_perfect_and_band() -> None:
    n = 20
    hist = np.stack(
        [np.full((n, 24), 55.0), np.full((n, 24), 90.0)], axis=2
    )  # Satisfactory, near 60
    ok = np.ones((n, 24), bool)
    rise = np.tile([1.0, 1.2, 1.4, 0.2, 0.3, 0.4], (n, 1))  # pushes into Moderate
    res = m.aqi_metrics(hist, ok, rise, rise, 3)
    assert res["aqi_h1"]["mae"] == 0 and res["aqi_h3"]["category_accuracy"] == 1.0
    assert res["band_crossing"]["pod"] == 1.0 and res["band_crossing"]["far"] == 0.0
    miss = m.aqi_metrics(hist, ok, rise, np.zeros_like(rise), 3)
    assert miss["band_crossing"]["pod"] == 0.0
