# 0012 — v2 teacher inputs and graph data (CR v2 I1, I2, I3, I11, I16–I19)

Status: accepted (2026-10-02, owner "proceed" on docs/specs/architecture_review_v2.md)

- **Node features (I1, I11):** teacher F = the 23 student channels of `feature_schema.yaml` (virtual unit u at
  every station, same builder as the student, so the teacher's edge is the network, not better sensors)
  + 7 Open-Meteo past channels (t2m, rh2m, sp, u10, v10, precip, swrad) = **30 columns**. Replaces cloud
  [C08.4]'s 32-name / 35-column list (dust, pm_state one-hot, festival flag dropped for simplicity).
- **Model B (I2):** 17 PM-free channels + 5 [S08.2] extras broadcast as constant channels = 22 (already built).
- **Static (I3):** [lat, lon, elevation, height_agl, sheltering_factor, is_reference] = 6.
- **Stations (I16):** no station reaches the CR's 70 % train PM completeness (best ≈ 40–45 %/yr, measured).
  Use all 30 simulated stations (simulator rule ≥ 8760 PM2.5 h); masks + masked loss; a sample needs ≥ 1
  measured target. Train graph N = 25 (5 spatial hold-out stations from the sim manifest); test N = 30.
- **Future weather (I17–I19):** `blh` is null for Delhi in Open-Meteo and there is no holiday calendar →
  X_fut = 9 columns (t2m, rh2m, sp, u10, v10, precip, swrad, hod_sin, hod_cos). Issued forecasts do not exist
  for 2018–21, so X_fut = ERA5 actuals in every split, flagged `oracle_weather: true` in the dataset card and
  reported as a limitation (same choice as ADR 0011).
