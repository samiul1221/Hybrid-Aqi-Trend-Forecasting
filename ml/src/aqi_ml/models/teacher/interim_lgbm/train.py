"""Interim LightGBM teacher ([S12.1], cloud [C08.10]) with out-of-fold predictions -> teacher_store.

    uv run python -m aqi_ml.models.teacher.interim_lgbm.train      # resumable stages

Privileged inputs (never available to the edge student):
- 5 nearest OTHER CPCB stations: ln(1+PM2.5) at t, t-1, t-3, ln(1+PM10) at t, distance, upwind score
  cos(bearing node->station - wind FROM at the node) (ADR 0011), and upwind-weighted neighbour PM;
- weather at t+1..t+3 (actual ERA5/station values: historical forecasts do not exist before 2022 ->
  documented fallback of cloud [C07]; allowed for a teacher as privileged information).
Plus the node's own 50 RF features. One LightGBM per output (6). OOF: 5 contiguous time blocks over the
teacher training period (train split, hold-out stations excluded); val/test/hold-out windows are
predicted by the model trained on the whole training period (oof = True by construction).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from aqi_ml.distill import teacher_store as ts
from aqi_ml.eval.metrics import rmse
from aqi_ml.features.dataset import ROOT
from aqi_shared.schema_loader import load as load_schema

F32 = npt.NDArray[np.float32]
N_NEIGHBOURS = 5
FOLDS = 5
PARAMS = {"objective": "huber", "alpha": 1.0, "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "verbose": -1, "num_threads": 8, "seed": 0}  # fmt: skip
MAX_ROUNDS, EARLY_STOP = 2000, 50
WEATHER = ("t", "rh", "wind_u", "wind_v", "precip", "swrad")


def haversine_km(
    lat1: float, lon1: float, lat2: npt.NDArray[np.float64], lon2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    )
    out: npt.NDArray[np.float64] = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    return out


def bearing_deg(
    lat1: float, lon1: float, lat2: npt.NDArray[np.float64], lon2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Initial bearing node -> station, degrees clockwise from north."""
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    out: npt.NDArray[np.float64] = np.rad2deg(np.arctan2(y, x)) % 360.0
    return out


class Context:
    """Station grid (hourly ln-PM of every CPCB station) and the neighbour lists."""

    def __init__(self) -> None:
        st = pd.read_parquet(ROOT / "data" / "interim" / "stations.parquet")
        base = pd.read_parquet(ROOT / "data" / "processed" / "base_table",
                               columns=["station_id", "ts", "pm25", "pm10", *WEATHER])  # fmt: skip
        base["station_id"] = base.station_id.astype(str)
        self.t0 = base.ts.min()
        self.ids = sorted(base.station_id.unique())
        col = {s: i for i, s in enumerate(self.ids)}
        n_t = int((base.ts.max() - self.t0) / pd.Timedelta(hours=1)) + 1
        row = ((base.ts - self.t0) / pd.Timedelta(hours=1)).astype(int).to_numpy()
        c = base.station_id.map(col).to_numpy()
        self.lp25 = np.full((n_t, len(self.ids)), np.nan, np.float32)
        self.lp10 = np.full((n_t, len(self.ids)), np.nan, np.float32)
        self.lp25[row, c] = np.log1p(base.pm25.to_numpy())
        self.lp10[row, c] = np.log1p(base.pm10.to_numpy())
        self.weather = {w: np.full((n_t, len(self.ids)), np.nan, np.float32) for w in WEATHER}
        for w in WEATHER:
            self.weather[w][row, c] = base[w].to_numpy()
        loc = st.set_index("station_id").loc[self.ids]
        self.lat, self.lon = loc.lat.to_numpy(float), loc.lon.to_numpy(float)
        self.col = col

    def neighbours(
        self, sid: str
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        i = self.col[sid]
        d = haversine_km(self.lat[i], self.lon[i], self.lat, self.lon)
        d[i] = np.inf
        nb = np.argsort(d)[:N_NEIGHBOURS]
        return nb, d[nb], bearing_deg(self.lat[i], self.lon[i], self.lat[nb], self.lon[nb])

    def features(self, sid: str, base_ts: pd.Series) -> tuple[F32, list[str]]:
        """Privileged teacher features for one station's windows (rows aligned to base_ts)."""
        t = (
            ((pd.to_datetime(base_ts, utc=True) - self.t0) / pd.Timedelta(hours=1))
            .astype(int)
            .to_numpy()
        )
        n_t = self.lp25.shape[0]

        def at(a: npt.NDArray[np.float32], lag: int) -> npt.NDArray[np.float64]:
            """a at row t - lag (lag < 0 = future), NaN outside the grid."""
            ok = (t - lag >= 0) & (t - lag < n_t)
            return np.where(ok, a[np.clip(t - lag, 0, n_t - 1)], np.nan)

        i = self.col[sid]
        nb, dist, brg = self.neighbours(sid)
        u, v = at(self.weather["wind_u"][:, i], 0), at(self.weather["wind_v"][:, i], 0)
        wd_from = np.rad2deg(np.arctan2(-u, -v)) % 360.0
        calm = np.hypot(u, v) < 0.3
        cols, names = [], []
        w_sum = np.zeros(len(t))
        w_pm = np.zeros(len(t))
        w_tr = np.zeros(len(t))
        for k, (j, dk, bk) in enumerate(zip(nb, dist, brg, strict=True)):
            l0, l1, l3 = at(self.lp25[:, j], 0), at(self.lp25[:, j], 1), at(self.lp25[:, j], 3)
            up = np.where(calm, 0.0, np.cos(np.deg2rad(bk - wd_from)))
            cols += [l0, l1, l3, at(self.lp10[:, j], 0), np.full(len(t), dk), up]
            names += [
                f"nb{k}_lp25_l0",
                f"nb{k}_lp25_l1",
                f"nb{k}_lp25_l3",
                f"nb{k}_lp10_l0",
                f"nb{k}_km",
                f"nb{k}_upwind",
            ]
            w = np.maximum(up, 0.0) * np.exp(-dk / 10.0) * np.isfinite(l0) * np.isfinite(l3)
            w_sum += w
            w_pm += w * np.nan_to_num(l0)
            w_tr += w * np.nan_to_num(l0 - l3)
        cols += [np.where(w_sum > 0, w_pm / np.maximum(w_sum, 1e-9), np.nan),
                 np.where(w_sum > 0, w_tr / np.maximum(w_sum, 1e-9), np.nan)]  # fmt: skip
        names += ["upwind_lp25", "upwind_lp25_trend3"]
        for w_name in WEATHER:
            for h in (1, 2, 3):
                cols.append(at(self.weather[w_name][:, i], -h))
                names.append(f"future_{w_name}_h{h}")
        return np.column_stack(cols).astype(np.float32), names


def node_table(
    ctx: Context, node_dir: Path, rf_names: list[str], outputs: list[str]
) -> pd.DataFrame:
    w = pd.read_parquet(node_dir / "windows_a.parquet")
    node = node_dir.name.split("=", 1)[1]
    sid = node[2:].rsplit("-", 1)[0]  # v-<station_id>-<unit>
    priv, names = ctx.features(sid, w.base_ts)
    out = pd.DataFrame(priv, columns=names)
    out.insert(0, "node_id", node)
    out.insert(1, "base_ts", w.base_ts.to_numpy())
    for c in ("split", "spatial_holdout"):
        out[c] = w[c].to_numpy()
    for n in rf_names:
        out[f"rf_{n}"] = w[f"rf_{n}"].to_numpy()
    for n in outputs:
        out[f"y_{n}"] = w[f"y_{n}"].to_numpy()
    return out


def train(dataset: str = "feat-v1", teacher_version: str | None = None) -> Path:
    t_start = time.time()
    schema = load_schema()
    outputs = list(schema.output_order)
    teacher_version = teacher_version or f"lgbm-interim-{dataset}"
    work = ROOT / "data" / "models" / dataset / "teacher-lgbm"
    work.mkdir(parents=True, exist_ok=True)
    table_f = work / "table.parquet"
    if not table_f.exists():
        ctx = Context()
        dirs = sorted((ROOT / "data" / "datasets" / dataset).glob("node_id=*"))
        pd.concat([node_table(ctx, d, list(schema.rf_names_a), outputs) for d in dirs],
                  ignore_index=True).to_parquet(table_f, index=False)  # fmt: skip
    tab = pd.read_parquet(table_f)
    feats = [c for c in tab.columns if c.startswith(("nb", "upwind", "future_", "rf_"))]
    trn = tab[(tab.split == "train") & ~tab.spatial_holdout]
    val = tab[(tab.split == "val") & ~tab.spatial_holdout]

    # production models: all training windows, early stopping on validation
    models_f = work / "models.json"
    if not models_f.exists():
        info = {}
        for j, name in enumerate(outputs):
            ds = lgb.Dataset(trn[feats], trn[f"y_{name}"])
            dv = lgb.Dataset(val[feats], val[f"y_{name}"], reference=ds)
            b = lgb.train(
                PARAMS,
                ds,
                MAX_ROUNDS,
                valid_sets=[dv],
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
            )
            b.save_model(str(work / f"prod_{j}.txt"))
            info[name] = {
                "best_iteration": b.best_iteration,
                "val_rmse": rmse(b.predict(val[feats]) - val[f"y_{name}"].to_numpy()),
            }
        models_f.write_text(json.dumps(info, indent=1))
    info = json.loads(models_f.read_text())

    # OOF: 5 contiguous time blocks over the teacher training period
    edges = np.quantile(trn.base_ts.astype("int64"), np.linspace(0, 1, FOLDS + 1))
    fold = np.clip(
        np.searchsorted(edges, trn.base_ts.astype("int64"), side="right") - 1, 0, FOLDS - 1
    )
    oof = np.full((len(trn), len(outputs)), np.nan, np.float32)
    for k in range(FOLDS):
        f = work / f"oof_fold{k}.npy"
        if not f.exists():
            tr_k, te_k = fold != k, fold == k
            pred = np.zeros((te_k.sum(), len(outputs)), np.float32)
            for j, name in enumerate(outputs):
                b = lgb.train(
                    PARAMS,
                    lgb.Dataset(trn[feats][tr_k], trn[f"y_{name}"][tr_k]),
                    info[name]["best_iteration"],
                )
                pred[:, j] = b.predict(trn[feats][te_k])
            np.save(f, pred)
        oof[fold == k] = np.load(f)

    prod = [lgb.Booster(model_file=str(work / f"prod_{j}.txt")) for j in range(len(outputs))]
    rest = tab.drop(index=trn.index)
    rest_pred = np.column_stack([b.predict(rest[feats]) for b in prod]).astype(np.float32)
    rows = pd.concat([
        pd.DataFrame({"node_id": trn.node_id.to_numpy(), "base_ts": trn.base_ts.to_numpy(), "oof": True, "fold_id": fold.astype(np.int8),
                      **{c: oof[:, j] for j, c in enumerate(ts.y_columns(outputs))}}),
        pd.DataFrame({"node_id": rest.node_id.to_numpy(), "base_ts": rest.base_ts.to_numpy(), "oof": True, "fold_id": np.int8(-1),
                      **{c: rest_pred[:, j] for j, c in enumerate(ts.y_columns(outputs))}}),
    ], ignore_index=True)  # fmt: skip
    rows["base_ts"] = pd.to_datetime(rows.base_ts, utc=True)
    store = ts.write(rows, teacher_version, outputs)

    # teacher quality on held-out data (feeds the [S12.1] precondition check)
    quality: dict[str, Any] = {}
    quality["train_oof"] = {
        n: rmse(oof[:, j] - trn[f"y_{n}"].to_numpy()) for j, n in enumerate(outputs)
    }
    for name, m in (("val", (rest.split == "val") & ~rest.spatial_holdout), ("test", (rest.split == "test") & ~rest.spatial_holdout),
                    ("test_holdout", (rest.split == "test") & rest.spatial_holdout)):  # fmt: skip
        mm = m.to_numpy()
        quality[name] = {
            n: rmse(rest_pred[mm, j] - rest[f"y_{n}"].to_numpy()[mm]) for j, n in enumerate(outputs)
        }
    meta = {"teacher_version": teacher_version, "store_version": ts.STORE_VERSION, "dataset": dataset,
            "features": feats, "params": PARAMS, "models": info, "folds": FOLDS,
            "train_period": [str(trn.base_ts.min()), str(trn.base_ts.max())], "rmse_log": quality,
            "rows": {"train_oof": len(trn), "production": len(rest)}, "store": str(store),
            "privileged": ["5 nearest other CPCB stations (PM, upwind)", "actual weather t+1..t+3 (ERA5/station)"],
            "seconds_last_run": round(time.time() - t_start, 1)}  # fmt: skip
    (work / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return work


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aqi_ml.models.teacher.interim_lgbm.train")
    ap.add_argument("--dataset", default="feat-v1")
    a = ap.parse_args(argv)
    print(f"wrote {train(a.dataset)}")


if __name__ == "__main__":
    main()
