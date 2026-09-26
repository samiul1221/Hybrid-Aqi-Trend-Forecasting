"""Golden feature fixtures for the firmware C twin (shared/fixtures/feature_golden/).

    uv run python -m aqi_ml.features.golden

A deterministic 60-hour record set exercising every rule (filled hour, 2 h gap -> interpolated,
3 h gap -> window invalid, baselines, calendar across IST midnight, rain, dust) and the builder's
outputs. Firmware (P10) must reproduce channels, masks, RF rows and targets within 1e-5.
Review the JSON diff after regenerating: a changed value is a behaviour change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aqi_ml.features.builder import build_node
from aqi_ml.features.dataset import load_config
from aqi_ml.models.edge.model_b.dropout import extras
from aqi_shared import pipeline as pl
from aqi_shared.schema_loader import load as load_schema

OUT = Path(__file__).resolve().parents[4] / "shared" / "fixtures" / "feature_golden"
N = 60


def golden_records() -> pd.DataFrame:
    i = np.arange(N, dtype=float)
    ts = pd.date_range("2026-01-01T18:30:00Z", periods=N, freq="h")  # IST 2026-01-02 00:00 onward
    pm25 = 60 + 30 * np.sin(i / 4) + i
    rec = pd.DataFrame({
        "ts": ts, "pm1": 0.8 * pm25, "pm25_corr": pm25, "pm10_corr": 1.7 * pm25,
        "coarse_ratio": (1.7 * pm25 - pm25) / (0.8 * pm25),
        "t": 12 + 8 * np.sin((i - 6) / 24 * 2 * np.pi), "rh": 70 - 20 * np.sin((i - 6) / 24 * 2 * np.pi),
        "p": 995 + 0.1 * i, "wind_u": np.cos(i / 5), "wind_v": np.sin(i / 7),
        "wind_speed": np.hypot(np.cos(i / 5), np.sin(i / 7)),
        "rain_frac": np.where((i >= 10) & (i <= 12), 1.0, 0.0),
        "lux": np.maximum(0, 40000 * np.sin((i - 6) / 24 * 2 * np.pi)),
        "mq7_rs": 20000 - 50 * i, "mq135_rs": 30000 + 20 * i,
        "sraw_voc": 30000 - 10 * i, "sraw_nox": 15000 + 15 * i,
        "Rbase7": 21000.0, "Rbase135": 35000.0, "base_voc": 30500.0, "base_nox": 15100.0,
        "aqi_pm": np.where(i >= 16, np.round(100 + i), np.nan),
        "dust": (i % 11) == 0, "pm_state": "ok",
    })  # fmt: skip
    for f in ("flag_pm", "flag_t", "flag_rh", "flag_p", "flag_wind", "flag_rain", "flag_lux",
              "flag_mq7", "flag_mq135", "flag_voc", "flag_nox"):  # fmt: skip
        rec[f] = pl.MEASURED
    rec.loc[20, "pm_state"] = "filled_le_1h"  # filled hour keeps the last value
    rec.loc[20, "flag_pm"] = pl.FILLED
    rec.loc[20, ["pm1", "pm25_corr", "pm10_corr"]] = rec.loc[
        19, ["pm1", "pm25_corr", "pm10_corr"]
    ].to_numpy()
    rec.loc[30:31, "t"] = np.nan  # 2 h gap -> interpolated, mask 0
    rec.loc[30:31, "flag_t"] = pl.MISSING
    rec.loc[50:52, "lux"] = np.nan  # 3 h gap -> windows covering it invalid
    rec.loc[50:52, "flag_lux"] = pl.MISSING
    return rec


def _clean(x: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), 9) for v in np.ravel(x)]


def build() -> dict[str, Any]:
    schema = load_schema()
    rec = golden_records()
    truth = (rec.pm25_corr.to_numpy() * 1.1, rec.pm10_corr.to_numpy() * 1.1)
    nf = build_node(rec, schema, truth, load_config())
    names = [c.name for c in schema.channels]
    rows_b = np.array([30, 45], dtype=np.int64)
    k_b = np.array([3, 24], dtype=np.int64)
    records = rec.assign(ts=rec.ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return {
        "_doc": "Feature golden [S08]: builder outputs for records.json. Tolerance 1e-5. null = NaN.",
        "schema_hash": schema.hash,
        "records": json.loads(records.to_json(orient="records")),
        "channels": names,
        "filled_values": [_clean(nf.hourly[names].to_numpy(float)[t]) for t in range(N)],
        "masks": nf.hourly[[f"m_{c}" for c in names]].to_numpy().tolist(),
        "rf_names": list(schema.rf_names_a),
        "rf": [_clean(nf.rf[t]) for t in range(N)],
        "targets_label_pm": {"pm25": _clean(truth[0]), "pm10": _clean(truth[1])},
        "y": [_clean(nf.y[t]) for t in range(N)],
        "valid_a": nf.valid_a.tolist(),
        "model_b_extras": {
            "rows": rows_b.tolist(),
            "k": k_b.tolist(),
            "names": list(schema.model_b_extra),
            "values": [_clean(r) for r in extras(nf.hourly, rows_b - k_b, k_b)],
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feature_golden.json").write_text(
        json.dumps(build(), indent=0) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {OUT / 'feature_golden.json'}")


if __name__ == "__main__":
    main()
