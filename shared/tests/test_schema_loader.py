import copy
from pathlib import Path

import pytest
import yaml

from aqi_shared import schema_loader as sl


def test_counts_match_spec() -> None:
    fs = sl.load()
    assert (fs.version, fs.lookback_hours, fs.horizons) == (1, 24, 3)
    assert len(fs.channels) == 23  # [S08.1] X[24x23]
    assert len(fs.rf_features) == 50  # [S08.3] approx_count 50
    assert fs.channel_names[:2] == ("ln_pm25", "ln_pm10")
    assert fs.channel_names[-1] == "pm_mask"
    assert [c.name for c in fs.model_b_channels] == list(fs.channel_names[4:21])
    assert len(fs.rf_names_b) == 27 + 5
    assert fs.output_order == ("pm25_h1", "pm25_h2", "pm25_h3", "pm10_h1", "pm10_h2", "pm10_h3")
    assert fs.hash.startswith("sha256:") and len(fs.hash) == 7 + 64


def test_rf_expansion_order() -> None:
    names = sl.load().rf_names_a
    assert names[:8] == (
        "ln_pm25_lag0", "ln_pm25_lag1", "ln_pm25_lag2", "ln_pm25_lag3", "ln_pm25_lag6",
        "ln_pm25_lag12", "ln_pm25_lag23", "ln_pm25_rmean3",
    )  # fmt: skip
    assert "rain_flag_rainh3" in names
    assert names[-1] == "aqi_now_lag0"


def test_hash_ignores_formatting(tmp_path: Path) -> None:
    raw = sl.load().raw
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=True, default_flow_style=False), encoding="utf-8")
    assert sl.load(p).hash == sl.load().hash


def test_hash_changes_with_content() -> None:
    raw = copy.deepcopy(sl.load().raw)
    raw["channels"][6]["op"] = {"offset": {"value": -1013}}
    assert sl.parse(raw).hash != sl.load().hash


def test_rejects_bad_schema() -> None:
    raw = copy.deepcopy(sl.load().raw)
    raw["channels"][0]["op"] = {"sqrt": {}}
    with pytest.raises(ValueError):
        sl.parse(raw)
    raw = copy.deepcopy(sl.load().raw)
    raw["rf_features"].append({"src": "mics6814", "lags": [0]})  # not in the sensor pool
    with pytest.raises(ValueError):
        sl.parse(raw)
