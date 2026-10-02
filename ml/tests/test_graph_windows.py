"""N2 graph windows: shapes, leakage (G3), sensor-domain column guard (G2), sample selection."""

import numpy as np
import pytest

from aqi_ml.data import graph_windows as gw

T, N, C, W = 60, 3, 23, 7


def _grid() -> dict:
    rng = np.random.default_rng(0)
    pm = rng.uniform(20, 200, (T, N, 2)).astype(np.float32)
    pm[40:, 2] = np.nan  # station 2 loses its truth
    split = np.array(["train"] * 35 + ["val"] * 25)
    return {"ts": np.arange(T, dtype=np.int64) * 3600, "X": rng.normal(size=(T, N, C)).astype(np.float32),
            "M": np.ones((T, N, C), np.uint8), "W": rng.normal(size=(T, N, W)).astype(np.float32),
            "pm": pm, "split": split, "static": np.zeros((N, 6), np.float32),
            "ist_hour": (np.arange(T) % 24).astype(np.int8), "stations": np.array(["a", "b", "c"]),
            "holdout": np.array([False, False, True])}  # fmt: skip


def test_window_shapes_targets_and_masks() -> None:
    g = _grid()
    g["M"][30, 1, 0] = 0  # one invalid input
    idx = np.arange(N)
    s = gw.window(g, 30, idx)
    assert s["X"].shape == (N, 24, 30) and s["M"].shape == (N, 24, 30)
    assert s["X_fut"].shape == (N, 6, 9) and s["wind_t"].shape == (N, 2) and s["y"].shape == (N, 12)
    assert s["X"][1, -1, 0] == 0.0 and s["M"][1, -1, 0] == 0.0  # masked value zeroed
    np.testing.assert_allclose(s["X"][0, -1, :23], g["X"][30, 0])  # last history row = base hour
    want = np.log1p(g["pm"][33, 0, 1]) - np.log1p(g["pm"][30, 0, 1])  # pm10_h3 of station 0
    assert s["y"][0, 6 + 2] == pytest.approx(want, rel=1e-5)
    np.testing.assert_allclose(s["X_fut"][0, 0, :7], g["W"][31, 0])  # future covariate = t+1
    s2 = gw.window(g, 38, idx)
    assert s2["y_mask"][2, 5] == 0.0 and s2["y_mask"][2, 0] == 1.0  # target at t+6 = 44 missing


def test_future_never_in_history() -> None:
    g = _grid()
    idx = np.arange(N)
    base = gw.window(g, 30, idx)
    g2 = _grid()
    g2["X"][31:] = 999.0  # change only the future sensor values
    s = gw.window(g2, 30, idx)
    np.testing.assert_array_equal(s["X"][:, :, :23], base["X"][:, :, :23])
    with pytest.raises(AssertionError):
        gw.window(g, 10, idx)  # history would start before the grid


def test_column_guard_rejects_truth_and_targets() -> None:
    gw.check_input_columns(["ln_pm25", "m_ln_pm25", "t"])
    for bad in (["truth_pm25"], ["target_pm10"], ["pm25"], ["pm_now"]):
        with pytest.raises(ValueError):
            gw.check_input_columns(bad)


def test_sample_index_stays_inside_split() -> None:
    g = _grid()
    idx = np.arange(N)
    tr = gw.sample_index(g, "train", idx)
    assert tr.min() == 23 and tr.max() == 34 - 6  # targets of the last train sample stay in train
    va = gw.sample_index(g, "val", idx)
    assert va.min() == 35 and va.max() == T - 7
    stats = gw.train_stats(g, idx)
    assert stats["mean"].shape == (30,) and (stats["std"] > 0).all()
