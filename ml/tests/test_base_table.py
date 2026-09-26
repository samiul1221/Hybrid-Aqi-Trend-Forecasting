from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from aqi_ml.data.base_table import clean, merge_station
from aqi_ml.data.datacard import summarise
from aqi_shared.time_utils import ist_calendar

CFG = {
    "pm25_max": 1500,
    "pm25_over_pm10_max": 1.05,
    "flatline_hours": 6,
    "plausible": {"t": [-5, 50], "rh": [1, 100], "p": [850, 1100], "ws": [0, 30]},
}
B0 = 1704846600  # IST 2024-01-10 06:00


def _hours(n: int) -> np.ndarray:
    return B0 + np.arange(n) * 3600


def test_clean_rules() -> None:
    df = pd.DataFrame({
        "pm25": [80.0, 1600.0, 120.0, -3.0, *([55.0] * 6), 60.0],
        "pm10": [100.0, 1700.0, 100.0, 90.0, 90.0, 91, 92, 93, 94, 95, 99.0],
    })  # fmt: skip
    out, stats = clean(df, CFG)
    assert out.pm25.iloc[0] == 80.0
    assert np.isnan(out.pm25.iloc[1])  # > 1500
    assert np.isnan(out.pm25.iloc[2]) and np.isnan(out.pm10.iloc[2])  # pm25 > 1.05 pm10
    assert np.isnan(out.pm25.iloc[3])  # negative
    assert out.pm25.iloc[4:10].isna().all() and out.pm10.iloc[4:10].notna().all()  # flatline 6 h
    assert out.pm25.iloc[10] == 60.0
    assert stats["pm25_flatline"] == 6 and stats["pm25_gt_pm10"] == 1


def test_merge_station_every_ist_hour_and_sources() -> None:
    n = 6
    aq = pd.DataFrame({"bucket": _hours(n)[[0, 2, 3]], "pm25": [40.0, 50.0, 60.0],
                       "pm10": [80.0, 90.0, 100.0], "t_stn": [20.0, 99.0, np.nan]})  # fmt: skip
    om = pd.DataFrame({"bucket": _hours(n), "t2m": 15.0, "rh2m": 60.0, "sp": 990.0,
                       "u10": 1.0, "v10": 0.0, "precip": 0.0, "swrad": 100.0, "blh": np.nan})  # fmt: skip
    df, _ = merge_station("s1", aq, om, int(B0), int(B0 + (n - 1) * 3600), CFG)
    assert len(df) == n and df.ts.is_monotonic_increasing
    assert df.pm25.isna().tolist() == [False, True, False, False, True, True]  # no imputation
    assert df.t.tolist()[:4] == [20.0, 15.0, 15.0, 15.0]
    assert df.t_src.tolist()[:4] == ["stn", "om", "om", "om"]  # 99 °C implausible -> Open-Meteo
    assert df.wind_dir.iloc[0] == pytest.approx(270.0)  # u>0: blowing east, FROM west
    for ts, h, d, y in zip(df.ts, df.ist_hour, df.ist_dow, df.ist_doy, strict=True):
        assert ist_calendar(ts.to_pydatetime()) == (h, d, y)
    assert df.td.iloc[0] < df.t.iloc[0]


def test_datacard_summary() -> None:
    ts = pd.to_datetime([datetime(2024, 1, 1, h, 30, tzinfo=UTC) for h in range(4)])
    base = pd.DataFrame({"station_id": ["a", "a", "b", "b"], "ts": ts,
                         "pm25": [10.0, np.nan, 30.0, 40.0], "t_src": ["om", "stn", "om", None]})  # fmt: skip
    s = summarise(base)
    assert s["coverage"].loc["a", "pm25"] == 0.5 and s["coverage"].loc["b", "pm25"] == 1.0
    assert "winter" in s["pm25_season_q"].index
