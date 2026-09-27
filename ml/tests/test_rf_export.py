"""RF blob export ([S13.2]): exact float32 decisions, Python reference == C evaluator, bad blobs rejected."""

import struct
import subprocess

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from aqi_ml.export import rf_blob


def _forests(n_out: int = 3) -> tuple[list[RandomForestRegressor], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 7)).astype(np.float32)
    x[:, 3] = np.round(x[:, 3], 1)  # many ties: thresholds sit exactly between float32 values
    ys = [x[:, 0] * 2 + np.sin(3 * x[:, 3]) + 0.1 * i for i in range(n_out)]
    forests = [
        RandomForestRegressor(n_estimators=5, max_depth=6, random_state=i).fit(x, y)
        for i, y in enumerate(ys)
    ]
    return forests, x, np.column_stack([f.predict(x) for f in forests])


def test_blob_matches_sklearn_and_c() -> None:
    forests, x, y = _forests()
    blob = rf_blob.write_blob(forests, 7, "sha256:" + "a" * 64)
    assert np.max(np.abs(rf_blob.predict_blob(blob, x) - y)) < 1e-5
    assert rf_blob.host_parity(blob, x, y) < 1e-5


def test_threshold_round_down_is_exact() -> None:
    t = 0.1  # float64 midpoint-like value not representable in float32
    f = rf_blob._thr_down(t)
    assert (
        f <= t and float(np.nextafter(np.float32(f), np.float32(np.inf))) > t
    )  # compare in float64


def test_c_rejects_corrupt_blobs() -> None:
    forests, x, y = _forests(1)
    blob = bytearray(rf_blob.write_blob(forests, 7, "sha256:" + "a" * 64))
    bad_magic = b"XXXX" + bytes(blob[4:])
    with pytest.raises(subprocess.CalledProcessError):  # C loader rejects -> non-zero exit
        rf_blob.host_parity(bad_magic, x, y)
    cyc = bytearray(blob)
    node0 = 84 + 2 + 4  # first node of the first tree
    struct.pack_into("<H", cyc, node0 + 5, 0)  # left child -> itself (cycle)
    with pytest.raises(subprocess.CalledProcessError):  # C loader rejects -> non-zero exit
        rf_blob.host_parity(bytes(cyc), x, y)
