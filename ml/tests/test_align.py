import numpy as np
import pytest

from aqi_ml.data import align

H = 3600
B = 1704846600  # 2024-01-10T00:30:00Z = IST 06:00 bucket start


def test_ist_aligned_hour_passes_through() -> None:
    out = align.intervals_to_ist(np.array([B]), np.array([B + H]), np.array([42.0]))
    assert out.bucket.tolist() == [B] and out.value.tolist() == [42.0]
    assert out.coverage.tolist() == [1.0]


def test_quarter_hours_are_averaged_with_coverage() -> None:
    s = B + np.arange(4) * 900
    out = align.intervals_to_ist(s, s + 900, np.array([10.0, 20.0, 30.0, 40.0]))
    assert out.value.tolist() == [25.0] and out.coverage.tolist() == [1.0]
    out3 = align.intervals_to_ist(s[:3], s[:3] + 900, np.array([10.0, 20.0, 30.0]))
    assert out3.coverage.tolist() == [0.75]


def test_utc_hours_are_split_by_overlap() -> None:
    s = np.array([B - 1800, B + 1800])  # [00:00Z, 01:00Z), [01:00Z, 02:00Z)
    out = align.intervals_to_ist(s, s + H, np.array([10.0, 30.0]))
    row = out.set_index("bucket").loc[B]
    assert row.value == 20.0 and row.coverage == 1.0
    assert out.set_index("bucket").loc[B - H].coverage == 0.5


def test_rejects_long_intervals_and_ignores_nan_duplicates() -> None:
    with pytest.raises(ValueError):
        align.intervals_to_ist(np.array([B]), np.array([B + 2 * H]), np.array([1.0]))
    out = align.intervals_to_ist(
        np.array([B, B, B]), np.array([B + H] * 3), np.array([5.0, 5.0, np.nan])
    )
    assert out.value.tolist() == [5.0]


def test_instantaneous_weights() -> None:
    t = np.array([B - 1800 - H, B + 1800 - H, B + 1800, B + 1800 + H])  # UTC hours
    out = align.instantaneous_to_ist(t, np.array([0.0, 0.0, 8.0, 0.0])).set_index("bucket")
    assert out.loc[B].value == pytest.approx(6.0)  # 3/4 * 8
    assert out.loc[B - H].value == pytest.approx(1.0)  # 1/8 * 8 (next sample is the peak)
    assert np.isnan(out.loc[B + H].value)  # edges need both neighbours


def test_accumulated_preceding_hour() -> None:
    t = np.array([B + 1800, B + 1800 + H])  # values cover (00:00,01:00] and (01:00,02:00]
    out = align.accumulated_to_ist(t, np.array([1.0, 3.0])).set_index("bucket")
    assert out.loc[B].value == 2.0  # half of each hour's sum -> 0.5 + 1.5
    assert np.isnan(out.loc[B - H].value)  # partially covered bucket is not reported


def test_wind_conventions() -> None:
    u, v = align.wind_uv(np.array([5.0]), np.array([270.0]))  # wind FROM west blows east
    assert u[0] == pytest.approx(5.0) and v[0] == pytest.approx(0.0, abs=1e-12)
    _, wd = align.wind_speed_dir(*align.wind_uv(np.array([2.0, 2.0]), np.array([350.0, 10.0])))
    assert wd[0] == pytest.approx(350.0) and wd[1] == pytest.approx(10.0)
    mu, mv = [x.mean() for x in align.wind_uv(np.array([2.0, 2.0]), np.array([350.0, 10.0]))]
    s, d = align.wind_speed_dir(np.array([mu]), np.array([mv]))
    assert min(d[0], 360 - d[0]) < 1e-9  # vector mean of 350 deg and 10 deg is north, not 180
    assert s[0] == pytest.approx(2 * np.cos(np.deg2rad(10)))
