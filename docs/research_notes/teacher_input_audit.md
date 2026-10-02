# Teacher input audit — teacher and student share the edge sensor domain (2026-10-02)

> **The teacher does not receive ground-truth future PM or privileged clean PM measurements as input.
> Ground-truth PM is used only as the supervised forecasting target. The teacher's privilege comes from
> access to the multi-station network and transport/physics information.** (ADR 0019)

## 1. How reference data becomes the 23 edge features (as implemented)

There is no learned model in this path. Every step is a documented physical/sensor model with parameters in
`ml/configs/sim_params.yaml` (sources: `simulator_parameters.md`, datasheets in `docs/papers/Datasheet`, ADR 0007).

```
base table (per station, IST hour)                     sim/virtual.py station_truth()
  CPCB PM2.5/PM10, CO, NO2, O3   ─┐                    truth_* columns (shared by the 3 units)
  ERA5: t, td, rh, p, wind, rain, SW ┘                          │
                                                                ▼
            sim/virtual.py simulate_unit() + sim/forward.py  (unit-specific drawn parameters, noise, drift)
              SPS30   pm1/pm25/pm10 ← truth PM, RH inside sensor (κ growth), dust response
              HDC1080 t, rh        ← ERA5 t/td + radiation/shelter error
              BMP280  p            ← ERA5 p + offset/noise
              anemometer wind_u/v  ← ERA5 wind + unit error
              rain    rain_frac    ← ERA5 precip/rh
              BH1750  lux          ← SW · exp(−β·PM2.5)  (real aerosol light extinction)
              MQ-7 / MQ-135 Rs     ← CO / VOC / NO2 (+H2 background), T/RH, drift
              SGP41 sraw_voc/nox   ← VOC / NO2 / O3, log response
              faults: random missing + outages → pm_measured
                                                                │
            sim/device.py process()  = on-device pipeline (aqi_shared.pipeline, calibration.json)
              pm25_corr, pm10_corr (κ humidity correction), coarse_ratio, dust flag, aqi_pm, pm_state
                                                                │
            features/builder.py channel_matrix()  (feature_schema.yaml srcs only)
              i0–i22: ln_pm25, ln_pm10, ln_pm1, coarse_ratio, t, rh, p, wind_u, wind_v, rain_frac, ln_lux,
              mq7_n, mq135_n, voc_n, nox_n, 6 time channels, aqi_now, pm_mask
                                                                │
            features/dataset.py → data/datasets/feat-v1/node_id=v-<station>-<u>/hourly.parquet
```

Labels: `features/dataset.py:66` passes `rec.truth_pm25 / truth_pm10` (true CPCB PM) as `target_pm`; they become
`y = ln(1+PM[t+h]) − ln(1+PM[t])` (I04) and are stored as `target_pm25/target_pm10` columns — **targets only**.

## 2. Findings

| # | where | what | verdict |
|---|---|---|---|
| A1 | `features/builder.py` `channel_matrix` | reads only schema `src` columns (sensor/device outputs), never `truth_*` | OK — student is in the sensor domain |
| A2 | `sim/virtual.py` `simulate_unit` | records keep `truth_*` columns next to sensor columns | risk — a builder that picked `truth_pm25` would leak; guard needed (G1) |
| A3 | `hourly.parquet` | `target_pm25/target_pm10` (true PM) stored next to the channels | risk — the graph window builder must read channel columns for X and targets only for y (G1, G2) |
| A4 | `sim/forward.py:118` BH1750 | lux depends on true PM2.5 through light extinction | OK — real physics the ESP32 also sees; same for teacher and student |
| A5 | **`models/teacher/interim_lgbm/train.py:71-83`** (v1 LightGBM teacher, ADR 0011) | neighbour features = **clean CPCB ln PM2.5 (t, t−1, t−3), ln PM10 (t)** of the 5 nearest other stations + upwind-weighted neighbour PM, read straight from the base table; own-node part = the student's 50 RF sensor features; future weather = ERA5 actuals | **violates the rule.** Not changed (frozen baseline, ADR 0016). Its precondition/KD numbers (ADR 0011, `e2_kd.md`) must be labelled "clean-neighbour teacher" and must not be compared to v2 teachers as if equivalent. |
| A6 | CR [R07.2] `pm_now` (true ln PM at t) | needed to turn ln-change forecasts into µg/m³ for metrics | allowed **only** in evaluation, never a model input (G2); student evaluation already uses the same true base (`pm_base`), so teacher and student are scored identically |
| A7 | CR [R07.2] X_node Open-Meteo past (7 ch), X_fut, wind prior | ERA5 actual weather (same source the HDC1080/BMP280/anemometer models start from) | declared cloud-side privileged covariates ([R07.2], ADR 0012); not PM; flag `use_om_past` lets the teacher run without them as an ablation |

## 3. Guards for the v2 graph window builder (N2)

- **G1 same pipeline:** the teacher's local block X_node[i, :, 0:23] for (station, unit u, base_ts) is read
  from the same `hourly.parquet` the student uses. Test: it equals the student's `X_seq` window bit-for-bit.
- **G2 column whitelist:** input columns = `feature_schema` channel names + masks + declared Open-Meteo columns;
  an assertion rejects any column starting with `truth_`, `target_`, or equal to `pm25`/`pm10`/`pm_now`.
- **G3 time:** max(input ts) == base_ts; targets strictly after base_ts (CR [R07.2]).
- **G4 neighbours are sensors too:** neighbour nodes are the same virtual unit u at the other stations
  (sensor-proxy), never the CPCB reference values.
