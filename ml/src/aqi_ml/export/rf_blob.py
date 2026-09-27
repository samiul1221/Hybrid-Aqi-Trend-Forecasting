"""RandomForest -> OTA-loadable blob for the firmware evaluator (rf_eval.c), plus host parity. ADR 0009.

Thresholds: sklearn splits float32 inputs with float64 thresholds (x <= thr). The blob stores the
largest float32 <= thr, which gives exactly the same decision for every float32 x.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parents[4]
RF_SRC = ROOT / "edge" / "firmware" / "aqi_node" / "src" / "models"
PARITY_TOOL_SRC = ROOT / "edge" / "tools" / "rf_parity.cpp"
LEAF = 255
NODE = struct.Struct("<BfHH")  # 9 bytes


def _thr_down(t: float) -> float:
    f = np.float32(t)
    if float(f) > t:
        f = np.nextafter(f, np.float32(-np.inf))
    return float(f)


def tree_bytes(tree: Any) -> bytes:
    """One fitted sklearn tree (tree_) -> u32 n_nodes + packed nodes."""
    t = tree.tree_
    n = t.node_count
    if n > 65535:
        raise ValueError("tree too large for u16 node indices")
    out = bytearray(struct.pack("<I", n))
    for i in range(n):
        if t.children_left[i] == -1:
            out += NODE.pack(LEAF, float(np.float32(t.value[i].ravel()[0])), 0, 0)
        else:
            out += NODE.pack(int(t.feature[i]), _thr_down(float(t.threshold[i])),
                             int(t.children_left[i]), int(t.children_right[i]))  # fmt: skip
    return bytes(out)


def write_blob(forests: Sequence[Any], n_features: int, schema_hash: str) -> bytes:
    """forests: one fitted RandomForestRegressor per output, in schema output order."""
    if n_features >= LEAF:
        raise ValueError("feature index must fit u8 < 255")
    h = schema_hash.encode("ascii")
    if len(h) >= 72:
        raise ValueError("schema hash too long")
    out = bytearray(
        b"AQRF" + struct.pack("<HHHH", 1, len(forests), n_features, 0) + h.ljust(72, b"\0")
    )
    for rf in forests:
        out += struct.pack("<H", len(rf.estimators_))
        for est in rf.estimators_:
            out += tree_bytes(est)
    return bytes(out)


def predict_blob(blob: bytes, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Python reference of rf_predict (same float32 arithmetic order) for tests."""
    n_out, n_feat = struct.unpack_from("<HH", blob, 6)
    off = 84
    x = np.asarray(x, dtype=np.float32)
    y = np.zeros((len(x), n_out), dtype=np.float32)
    for o in range(n_out):
        (n_trees,) = struct.unpack_from("<H", blob, off)
        off += 2
        acc = np.zeros(len(x), dtype=np.float32)
        for _ in range(n_trees):
            (n_nodes,) = struct.unpack_from("<I", blob, off)
            nodes = [NODE.unpack_from(blob, off + 4 + 9 * i) for i in range(n_nodes)]
            off += 4 + 9 * n_nodes
            for r in range(len(x)):
                i = 0
                while nodes[i][0] != LEAF:
                    f, thr, left, right = nodes[i]
                    i = left if x[r, f] <= np.float32(thr) else right
                acc[r] = np.float32(acc[r] + np.float32(nodes[i][1]))
        y[:, o] = acc / np.float32(n_trees)
    assert n_feat == x.shape[1]
    return y


def host_parity(
    blob: bytes, x: npt.NDArray[np.float32], y_ref: npt.NDArray[np.floating[Any]]
) -> float:
    """Compile rf_eval.c + edge/tools/rf_parity.cpp with host g++, run on x; max |C - y_ref|."""
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "model.bin").write_bytes(blob)
        x32 = np.ascontiguousarray(x, dtype=np.float32)
        hdr = struct.pack("<IHH", len(x32), x32.shape[1], y_ref.shape[1])
        (dp / "vectors.bin").write_bytes(hdr + x32.tobytes())
        exe, obj = dp / "rf_parity.exe", dp / "rf_eval.o"
        cc = [
            "gcc",
            "-std=c99",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{RF_SRC}",
            "-c",
            str(RF_SRC / "rf_eval.c"),
            "-o",
            str(obj),
        ]
        cxx = [
            "g++",
            "-std=c++14",
            "-O1",
            "-Wall",
            "-Werror",
            f"-I{RF_SRC}",
            str(PARITY_TOOL_SRC),
            str(obj),
            "-o",
            str(exe),
        ]
        for cmd in (cc, cxx):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"compile failed: {r.stderr[:2000]}")
        res = subprocess.run([str(exe), str(dp / "model.bin"), str(dp / "vectors.bin"), str(dp / "out.bin")],
                             check=True, capture_output=True, text=True)  # fmt: skip
        if "ok" not in res.stdout:
            raise RuntimeError(res.stdout + res.stderr)
        y_c = np.frombuffer((dp / "out.bin").read_bytes(), dtype=np.float32).reshape(y_ref.shape)
    return float(np.max(np.abs(y_c.astype(np.float64) - np.asarray(y_ref, dtype=np.float64))))
