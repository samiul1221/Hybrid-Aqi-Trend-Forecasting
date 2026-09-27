"""P8 distillation tests: teacher_store contract, KD targets, precondition, full_KD with a fake store."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aqi_ml.distill import kd
from aqi_ml.distill import teacher_store as ts

OUT = ["pm25_h1", "pm25_h2", "pm25_h3", "pm10_h1", "pm10_h2", "pm10_h3"]


def _rows(n: int = 60, with_arrays: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts_ = pd.date_range("2021-01-30T18:30Z", periods=n, freq="h")  # crosses a month boundary
    rows = [ts.row_from_arrays("v-s1-0", t, rng.normal(size=6).astype(np.float32), True, i % 5, OUT,
                               rng.normal(size=(24, 8)).astype(np.float32) if with_arrays else None,
                               rng.random((4, 24, 24)).astype(np.float32) if with_arrays else None)
            for i, t in enumerate(ts_)]  # fmt: skip
    return pd.DataFrame(rows)


def test_store_roundtrip_and_join(tmp_path: Path) -> None:
    df = _rows(with_arrays=True)
    ts.write(df, "fake-v1", OUT, root=tmp_path)
    assert len(list((tmp_path / "fake-v1" / "node_id=v-s1-0").glob("month=*"))) == 2
    back = ts.read("fake-v1", root=tmp_path)
    assert len(back) == len(df) and (back.d_t == 8).all() and (back.heads == 4).all()
    f = ts.decode(back.feat_seq.iloc[0], (24, 8))
    assert f is not None and f.shape == (24, 8)
    win = pd.DataFrame(
        {
            "node_id": ["v-s1-0", "v-s1-0"],
            "base_ts": [df.base_ts.iloc[3], pd.Timestamp("2030-01-01", tz="UTC")],
        }
    )
    y = ts.join(win, back, OUT)
    assert np.allclose(y[0], df[ts.y_columns(OUT)].iloc[3].to_numpy()) and np.isnan(y[1]).all()


def test_store_validation(tmp_path: Path) -> None:
    df = _rows()
    with pytest.raises(ValueError):
        ts.write(pd.concat([df, df]), "dup", OUT, root=tmp_path)
    bad = df.copy()
    bad.loc[0, "y_teacher_pm25_h1"] = np.nan
    with pytest.raises(ValueError):
        ts.write(bad, "nan", OUT, root=tmp_path)


def test_patch_aggregation() -> None:
    f = np.arange(24 * 2, dtype=np.float32).reshape(24, 2)
    a = np.ones((3, 24, 24), np.float32)
    fp, ap = ts.aggregate_to_patches(f, a)
    assert fp.shape == (12, 2) and fp[0, 0] == 1.0 and ap is not None and ap.shape == (3, 12, 12)


def test_rf_targets_and_model_b() -> None:
    y = np.array([[1.0] * 6, [2.0] * 6], np.float32)
    yt = np.array([[0.0] * 6, [np.nan] * 6], np.float32)
    assert np.allclose(kd.rf_targets(y, yt, 0.5)[0], 0.5) and np.allclose(
        kd.rf_targets(y, yt, 0.0)[1], 2.0
    )
    shifted = kd.model_b_teacher(
        np.zeros((1, 6), np.float32),
        np.array([[5.0, 6.0]], np.float32),
        np.array([[4.0, 4.5]], np.float32),
    )
    assert np.allclose(shifted, [[1, 1, 1, 1.5, 1.5, 1.5]])


def test_precondition() -> None:
    p = kd.precondition({"a": 0.1, "b": 0.3}, {"a": 0.2, "b": 0.2})
    assert p["per_output"]["a"]["pass"] and not p["per_output"]["b"]["pass"] and not p["all_pass"]


def test_full_kd_with_fake_teacher_store() -> None:
    """full_KD is ready for the graph teacher: loss parts, curriculum, AFD mask, training reduces loss."""
    from aqi_ml.models.edge.gdc_cnn.model import build

    rng = np.random.default_rng(0)
    n, d_t = 64, 8
    x = rng.normal(size=(n, 12, 46)).astype(np.float32)
    y = rng.normal(size=(n, 6)).astype(np.float32)
    f_t = np.stack(
        [
            ts.aggregate_to_patches(rng.normal(size=(24, d_t)).astype(np.float32), None)[0]
            for _ in range(n)
        ]
    )
    attn = np.stack(
        [
            ts.aggregate_to_patches(
                np.zeros((24, d_t), np.float32), rng.random((4, 24, 24)).astype(np.float32)
            )[1]
            for _ in range(n)
        ]
    )
    _, feat_model = build(12, 46, 16)
    tr = kd.FullKDTrainer(feat_model, d_t)
    m = kd.afd_mask(attn)
    assert np.allclose(np.asarray(m).sum(axis=1), 1.0)
    first = tr.step(x, y, f_t, attn, epoch=0)
    assert first["r"] == 0.0 and first["total"] == pytest.approx(
        kd.LAMBDA_P * first["l_pred"], rel=1e-5
    )
    hist = [tr.step(x, y, f_t, attn, epoch=10) for _ in range(30)]
    assert hist[-1]["total"] < hist[0]["total"] and hist[0]["r"] == 1.0 and hist[0]["l_afd"] > 0
    no_attn = kd.full_kd_loss(
        y, y, np.zeros((2, 12, 16), np.float32), np.zeros((2, 12, 16), np.float32), None, 10
    )
    assert float(no_attn[1]["l_afd"]) == 0.0  # teacher without attention -> out_KD + L_feat only
