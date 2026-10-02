"""Teacher graph windows (CR v2 [R07.2]): all stations at one base_ts, in the edge sensor domain.

    uv run python -m aqi_ml.data.graph_windows        # data/graph/<version>/

Why it looks like this
- Sensor domain (ADR 0019): the node features are read from the SAME `hourly.parquet` files the student
  uses (virtual unit u at every station). Clean CPCB PM enters only as the target `pm` array. Neighbours are
  the same unit u at the other stations, never reference values.
- Storage: one dense hourly grid per unit, `unit{u}.npz` = X [T, N, 23], M [T, N, 23], plus `common.npz`
  (ts, weather, target PM, split, static). Windows are sliced at load time by `window()`. This replaces
  [R07.2]'s month shards of [S, N, 24, F] windows, which would repeat every hour 24 times (ADR 0020).
- Weather = pure Open-Meteo ERA5 (data/interim/open_meteo), not the base table's station/ERA5 mix, so past
  channels and the "oracle" future covariates come from one consistent source (ADR 0012, I19).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from aqi_ml.data.align import wind_speed_dir
from aqi_ml.features.dataset import ROOT
from aqi_shared.schema_loader import load as load_schema

F32 = npt.NDArray[np.float32]
CONFIG = ROOT / "ml" / "configs" / "graph.yaml"
FORBIDDEN_PREFIXES = ("truth_", "target_")  # audit guard G2
FORBIDDEN_NAMES = {"pm25", "pm10", "pm_now"}


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg


def check_input_columns(cols: list[str]) -> None:
    """Guard G2: no ground-truth or target column may be read as a model input."""
    bad = [c for c in cols if c.startswith(FORBIDDEN_PREFIXES) or c in FORBIDDEN_NAMES]
    if bad:
        raise ValueError(f"forbidden teacher input columns (ADR 0019): {bad}")


# ---------------------------------------------------------------------------------------- build
def _stations(ds: Path) -> list[str]:
    return sorted({d.name.split("=", 1)[1].rsplit("-", 1)[0][2:] for d in ds.glob("node_id=v-*")})


def _static(stations: list[str], cfg: dict[str, Any]) -> F32:
    st = pd.read_parquet(ROOT / "data" / "interim" / "stations.parquet").set_index("station_id")
    c = cfg["static_constants"]
    rows = [[st.lat[s], st.lon[s], st.elevation_m[s], c["height_agl"], c["sheltering_factor"], c["is_reference"]]
            for s in stations]  # fmt: skip
    return np.asarray(rows, dtype=np.float32)  # [N, 6]


def build(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    schema = load_schema()
    chans = [c.name for c in schema.channels]  # the 23 student channels, schema order
    x_cols = chans + [f"m_{c}" for c in chans]
    check_input_columns(x_cols)
    ds = ROOT / "data" / "datasets" / cfg["dataset"]
    out: Path = ROOT / "data" / "graph" / str(cfg["version"])
    out.mkdir(parents=True, exist_ok=True)
    stations = _stations(ds)
    n = len(stations)

    # common time axis: every IST hour from the first to the last record of any node
    ts_all = [
        pd.read_parquet(d / "hourly.parquet", columns=["ts"]).ts for d in ds.glob("node_id=v-*")
    ]
    t0, t1 = min(s.min() for s in ts_all), max(s.max() for s in ts_all)
    ts = pd.date_range(t0, t1, freq="h")
    T = len(ts)
    pos = pd.Series(np.arange(T), index=ts)

    pm = np.full((T, n, 2), np.nan, np.float32)  # TRUE CPCB PM: target only (ADR 0019)
    split = np.full(T, "", dtype=object)
    holdout = np.zeros(n, dtype=bool)
    for u in cfg["units"]:
        X = np.full((T, n, len(chans)), np.nan, np.float32)
        M = np.zeros((T, n, len(chans)), np.uint8)
        for i, s in enumerate(stations):
            f = ds / f"node_id=v-{s}-{u}" / "hourly.parquet"
            if not f.exists():
                continue
            h = pd.read_parquet(f, columns=["ts", *x_cols])  # inputs: schema channels + masks only
            r = pos[h.ts].to_numpy()
            X[r, i] = h[chans].to_numpy(np.float32)
            M[r, i] = h[[f"m_{c}" for c in chans]].to_numpy(np.uint8)
            lab = pd.read_parquet(
                f, columns=["ts", "target_pm25", "target_pm10", "split", "spatial_holdout"]
            )
            prev = pm[r, i]
            new = lab[["target_pm25", "target_pm10"]].to_numpy(np.float32)
            if not np.allclose(prev[np.isfinite(prev)], new[np.isfinite(prev)], equal_nan=True):
                raise ValueError(
                    f"truth PM differs between units at {s}"
                )  # truth is shared per station
            pm[r, i] = new
            split[r] = lab.split.to_numpy()
            holdout[i] = bool(lab.spatial_holdout.iloc[0])
        np.savez(out / f"unit{u}.npz", X=X, M=M)
        print(f"unit {u}: X {X.shape}, measured PM2.5 {np.mean(M[..., 0]):.1%}", flush=True)

    W = np.full((T, n, len(cfg["om_columns"])), np.nan, np.float32)
    for i, s in enumerate(stations):
        om = pd.read_parquet(ROOT / "data" / "interim" / "open_meteo" / f"{s}.parquet")
        om_ts = pd.to_datetime(om.bucket, unit="s", utc=True)
        keep = om_ts.isin(pos.index)
        W[pos[om_ts[keep]].to_numpy(), i] = om.loc[keep.to_numpy(), cfg["om_columns"]].to_numpy(
            np.float32
        )

    ist_hour = ((ts + pd.Timedelta(hours=5, minutes=30)).hour).to_numpy()
    epoch = ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")).to_numpy(np.int64)
    np.savez(out / "common.npz", ts=epoch, W=W, pm=pm, split=split.astype(str),
             holdout=holdout, static=_static(stations, cfg), stations=np.array(stations),
             ist_hour=ist_hour.astype(np.int8))  # fmt: skip
    g1 = check_against_student(open_grid(out, cfg["units"][0]), ds, cfg["g1_check_windows"])
    card(out, cfg, open_grid(out, cfg["units"][0]), g1)
    return out


# ------------------------------------------------------------------------------------ reading
def open_grid(root: Path, unit: int) -> dict[str, Any]:
    g = dict(np.load(root / "common.npz", allow_pickle=False))
    g.update(np.load(root / f"unit{unit}.npz"))
    g["unit"] = unit
    return g


def window(
    g: dict[str, Any], t: int, idx: npt.NDArray[np.int64], L: int = 24, H: int = 6
) -> dict[str, Any]:
    """One graph sample at base hour t for stations idx (CR [R07.2]); raw units, not normalised.

    X [N, L, 30] = 23 sensor channels + 7 Open-Meteo past; M [N, L, 30]; S [N, 6]; X_fut [N, H, 9];
    wind_t [N, 2] (ws, wd_from); y / y_mask [N, 2H] pollutant-major; pm_now [N, 2] (EVALUATION ONLY).
    """
    hist = np.arange(t - L + 1, t + 1)
    fut = np.arange(t + 1, t + H + 1)
    assert hist[0] >= 0 and fut[-1] < len(g["ts"]), "window outside the grid"
    assert (
        g["ts"][hist].max() == g["ts"][t] and g["ts"][fut].min() > g["ts"][t]
    )  # G3: no future in inputs
    x_s = g["X"][hist][:, idx].transpose(1, 0, 2)  # [N, L, 23]
    m_s = g["M"][hist][:, idx].transpose(1, 0, 2)
    w = g["W"][hist][:, idx].transpose(1, 0, 2)  # [N, L, 7]
    X = np.concatenate([x_s, w], axis=2)  # [N, L, 30]
    M = np.concatenate([m_s, np.isfinite(w).astype(np.uint8)], axis=2)
    M = M * np.isfinite(X)  # a NaN is never "valid"
    X = np.where(M > 0, X, 0.0).astype(np.float32)

    hod = 2 * np.pi * g["ist_hour"][fut] / 24.0
    cyc = np.broadcast_to(np.stack([np.sin(hod), np.cos(hod)], 1), (len(idx), H, 2))  # [N, H, 2]
    w_f = g["W"][fut][:, idx].transpose(1, 0, 2)  # [N, H, 7] ERA5 "oracle" future weather (I19)
    X_fut = np.nan_to_num(np.concatenate([w_f, cyc], axis=2)).astype(np.float32)  # [N, H, 9]

    u10, v10 = g["W"][t, idx, 3], g["W"][t, idx, 4]
    ws, wd = wind_speed_dir(np.nan_to_num(u10), np.nan_to_num(v10))
    wind_t = np.stack([ws, wd], 1).astype(np.float32)  # [N, 2]

    lp_now = np.log1p(g["pm"][t, idx])  # [N, 2]
    lp_fut = np.log1p(g["pm"][fut][:, idx])  # [H, N, 2]
    y = (
        (lp_fut - lp_now[None]).transpose(1, 2, 0).reshape(len(idx), 2 * H)
    )  # [N, 12] pm25_h1..h6, pm10_h1..h6
    y_mask = np.isfinite(y)
    return {"X": X, "M": M.astype(np.float32), "S": g["static"][idx], "X_fut": X_fut, "wind_t": wind_t,
            "y": np.nan_to_num(y).astype(np.float32), "y_mask": y_mask.astype(np.float32),
            "pm_now": lp_now.astype(np.float32), "base_ts": int(g["ts"][t])}  # fmt: skip


def sample_index(
    g: dict[str, Any], split: str, idx: npt.NDArray[np.int64], L: int = 24, H: int = 6
) -> npt.NDArray[np.int64]:
    """Base hours of one split whose window fits, whose targets stay in the same split, and where >= 1
    station has a measured target (I16: masks replace the 70 % completeness rule)."""
    T = len(g["ts"])
    t = np.arange(L - 1, T - H)
    s = g["split"]
    ok = (s[t] == split) & (s[t + H] == split)
    lp = np.log1p(g["pm"][:, idx])  # [T, N, 2]
    has_y = np.zeros(T, dtype=bool)
    for h in range(1, H + 1):
        has_y[: T - h] |= np.isfinite(lp[h:] - lp[:-h]).any(axis=(1, 2))
    res: npt.NDArray[np.int64] = t[ok & has_y[t]]
    return res


def train_stats(g: dict[str, Any], idx: npt.NDArray[np.int64]) -> dict[str, F32]:
    """Per-column z-score statistics over TRAIN hours and valid values only ([C08.4] normalisation)."""
    tr = g["split"] == "train"
    X = np.concatenate([g["X"][tr][:, idx], g["W"][tr][:, idx]], axis=2).reshape(
        -1, g["X"].shape[2] + g["W"].shape[2]
    )
    M = np.concatenate(
        [g["M"][tr][:, idx], np.ones_like(g["W"][tr][:, idx], np.uint8)], axis=2
    ).reshape(X.shape)
    X = np.where((M > 0) & np.isfinite(X), X, np.nan)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    return {"mean": mu.astype(np.float32), "std": np.maximum(sd, 1e-3).astype(np.float32)}


# ---------------------------------------------------------------------------------- checks + card
def check_against_student(g: dict[str, Any], ds: Path, n: int) -> dict[str, Any]:
    """Guard G1: the teacher grid at (station, unit, base_ts) equals the student's X_seq window bit-for-bit."""
    from aqi_ml.features.dataset import load

    st = list(g["stations"])
    stu = load(ds, "A", split="train", seq=True, fraction=0.01, seed=0)
    rows = np.random.default_rng(0).choice(len(stu["y"]), size=min(n, len(stu["y"])), replace=False)
    t_of = {int(v): k for k, v in enumerate(g["ts"])}
    checked = 0
    for r in rows:
        node = stu["index"].node_id.iloc[r]
        if not node.endswith(f"-{g['unit']}"):
            continue
        s = node[2:].rsplit("-", 1)[0]
        t = t_of[int(pd.Timestamp(stu["index"].base_ts.iloc[r]).timestamp())]
        teacher = g["X"][t - 23 : t + 1, st.index(s)]
        if not np.array_equal(teacher, stu["X_seq"][r], equal_nan=True):
            raise AssertionError(
                f"G1 failed: teacher != student input at {node} {stu['index'].base_ts.iloc[r]}"
            )
        checked += 1
    if checked == 0:
        raise AssertionError("G1 checked no windows")
    return {"g1_windows_checked": checked, "g1_equal": True}


def card(out: Path, cfg: dict[str, Any], g: dict[str, Any], g1: dict[str, Any]) -> None:
    idx_tr = np.flatnonzero(~g["holdout"])
    idx_all = np.arange(len(g["stations"]))
    counts = {"train": len(sample_index(g, "train", idx_tr)), "val": len(sample_index(g, "val", idx_tr)),
              "test": len(sample_index(g, "test", idx_all))}  # fmt: skip
    meta = {"version": cfg["version"], "config": cfg, "schema_hash": load_schema().hash,
            "stations": list(g["stations"]), "holdout": list(g["stations"][g["holdout"]]),
            "hours": len(g["ts"]), "base_hours_per_unit": counts, "units": cfg["units"],
            "oracle_weather": cfg["oracle_weather"], **g1,
            "input_rule": "sensor-proxy features only; true CPCB PM is target/eval only (ADR 0019)"}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {k: meta[k] for k in ("hours", "base_hours_per_unit", "holdout", "g1_windows_checked")}
        )
    )


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(prog="aqi_ml.data.graph_windows").parse_args(argv)
    print(build())


if __name__ == "__main__":
    main()
