"""teacher_store: versioned store of teacher outputs for distillation ([S12.2], cloud [C08.9]).

Layout: data/teacher_store/<teacher_version>/node_id=<id>/month=<YYYY-MM>/part-0.parquet
Row key: (teacher_version, node_id, base_ts). Columns (STORE_VERSION 1):
  y_teacher_<pm25_h1..pm10_h3>  float32  same definition/order as the students' targets (I04, I09)
  oof                           bool     True = produced by a model that never saw this window's time block
  fold_id                       int8     -1 = production model (window outside the teacher training period)
  feat_seq                      binary   optional float32 [24 x D_T] (full_KD; graph teacher, phase 2)
  attn                          binary   optional float32 [heads x 24 x 24] (L_afd)
  d_t, heads                    int16    shapes of feat_seq/attn (0 when absent)
Leakage rules ([S12.2]): teacher trained only on periods <= the student train period; OOF for training
windows; teacher inputs at base_ts only use data available at base_ts (except the documented LUPI weather).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

STORE_VERSION = 1
ROOT = Path(__file__).resolve().parents[4] / "data" / "teacher_store"


def y_columns(output_order: Iterable[str]) -> list[str]:
    return [f"y_teacher_{n}" for n in output_order]


def encode(arr: npt.NDArray[np.float32] | None) -> bytes | None:
    return None if arr is None else np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def decode(blob: bytes | None, shape: tuple[int, ...]) -> npt.NDArray[np.float32] | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32).reshape(shape)


def validate(df: pd.DataFrame, output_order: list[str]) -> None:
    need = {"node_id", "base_ts", "oof", "fold_id", *y_columns(output_order)}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"teacher_store rows miss {sorted(missing)}")
    if df.duplicated(["node_id", "base_ts"]).any():
        raise ValueError("duplicate (node_id, base_ts) rows")
    if not np.isfinite(df[y_columns(output_order)].to_numpy(float)).all():
        raise ValueError("non-finite y_teacher")


def write(
    df: pd.DataFrame, teacher_version: str, output_order: list[str], root: Path = ROOT
) -> Path:
    """Write teacher rows (one teacher_version) partitioned by node_id/month; overwrites those parts."""
    validate(df, output_order)
    df = df.copy()
    for c, dt in (("feat_seq", object), ("attn", object)):
        if c not in df:
            df[c] = pd.Series([None] * len(df), dtype=dt)
    for c in ("d_t", "heads"):
        if c not in df:
            df[c] = 0
    df["teacher_version"], df["store_version"] = teacher_version, STORE_VERSION
    df["month"] = df.base_ts.dt.tz_convert("UTC").dt.strftime("%Y-%m")
    out = root / teacher_version
    for (node, month), g in df.groupby(["node_id", "month"]):
        p = out / f"node_id={node}" / f"month={month}"
        p.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["node_id", "month"]).to_parquet(p / "part-0.parquet", index=False)
    return out


def read(teacher_version: str, node_ids: Iterable[str] | None = None, columns: list[str] | None = None,
         root: Path = ROOT) -> pd.DataFrame:  # fmt: skip
    """Rows of one teacher version (optionally for some nodes), keyed by node_id + base_ts."""
    base = root / teacher_version
    dirs = (
        [base / f"node_id={n}" for n in node_ids]
        if node_ids is not None
        else sorted(base.glob("node_id=*"))
    )
    frames = []
    for d in dirs:
        if not d.exists():
            continue
        f = pd.read_parquet(d, columns=columns)
        f["node_id"] = d.name.split("=", 1)[1]
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "store_version" in out and (out.store_version != STORE_VERSION).any():
        raise ValueError("teacher_store version mismatch")
    return out


def join(
    windows: pd.DataFrame, store: pd.DataFrame, output_order: list[str]
) -> npt.NDArray[np.float32]:
    """y_teacher aligned to a window index (node_id, base_ts); NaN where the teacher has no row."""
    cols = y_columns(output_order)
    key = ["node_id", "base_ts"]
    s = store[key + cols].copy()
    s["base_ts"] = pd.to_datetime(s.base_ts, utc=True)
    w = windows[key].copy()
    w["base_ts"] = pd.to_datetime(w.base_ts, utc=True)
    m = w.merge(s, on=key, how="left")
    return m[cols].to_numpy(np.float32)


def aggregate_to_patches(feat_seq: npt.NDArray[np.float32], attn: npt.NDArray[np.float32] | None,
                         patch: int = 2) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]:  # fmt: skip
    """[S12.2] aggregation: feat_seq [24 x D_T] -> [12 x D_T] (mean of each 2-hour patch);
    attn [heads x 24 x 24] -> [heads x 12 x 12] (mean over both axes' patch groups)."""
    t, d = feat_seq.shape
    f = feat_seq.reshape(t // patch, patch, d).mean(axis=1)
    a = None
    if attn is not None:
        h = attn.shape[0]
        a = attn.reshape(h, t // patch, patch, t // patch, patch).mean(axis=(2, 4))
    return f.astype(np.float32), a


def row_from_arrays(node_id: str, base_ts: Any, y: npt.NDArray[np.float32], oof: bool, fold_id: int,
                    output_order: list[str], feat_seq: npt.NDArray[np.float32] | None = None,
                    attn: npt.NDArray[np.float32] | None = None) -> dict[str, Any]:  # fmt: skip
    row: dict[str, Any] = {"node_id": node_id, "base_ts": base_ts, "oof": oof, "fold_id": fold_id}
    row.update({c: float(v) for c, v in zip(y_columns(output_order), y, strict=True)})
    row["feat_seq"], row["attn"] = encode(feat_seq), encode(attn)
    row["d_t"] = 0 if feat_seq is None else feat_seq.shape[1]
    row["heads"] = 0 if attn is None else attn.shape[0]
    return row
