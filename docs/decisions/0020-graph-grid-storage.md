# 0020 — Teacher graph data stored as hourly grids, windows sliced at load

Status: accepted (2026-10-02). Deviation from CR v2 [R07.2] storage ("one shard per month per unit").

- [R07.2] month shards of [S, N, 24, F] windows repeat every hour 24 times (≈ 20 GB for 3 units).
  `aqi_ml.data.graph_windows` stores one dense grid per unit instead: `unit{u}.npz` (X [T, N, 23] float32,
  M uint8) + `common.npz` (ts, Open-Meteo W [T, N, 7], true PM [T, N, 2] = target only, split, static,
  stations) ≈ 0.8 GB total; `window(g, t, idx)` slices one sample with the leakage asserts (G3).
- Node features come from the student's own `hourly.parquet` (ADR 0019, guard G1 checked bit-for-bit at
  build time). Weather comes from pure Open-Meteo ERA5 (`data/interim/open_meteo`), not the base table,
  whose t/rh/wind mix CPCB station met (~10 % of hours) with ERA5.
- Samples: base hours whose window fits, whose targets stay in the same split, and with ≥ 1 measured target.
