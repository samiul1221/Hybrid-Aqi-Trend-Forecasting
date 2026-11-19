# Design decisions (plain words)

> **Naming note:** this file describes version 1, where the big cloud graph model was called the "teacher" and the on-device model the "student". The current design (v3) is in `docs/ARCH_V3.md`; it no longer treats the box model as a copy of the cloud model.

Detailed plans: `docs/specs/` (edge plan, cloud plan, CHANGE_REQUEST_v2). This page says what we actually
built and why, in one place. "TODO_VERIFY" = check with a primary source or real hardware later.

## 1. Data
- **Truth PM and gases:** official CPCB hourly files (CPCB Data Repository, downloaded by hand), 33 Delhi-NCR
  stations, Jan 2018 - Sep 2026, complete years (2023-24 included). Most station-years have 90-99 % of hours.
  An earlier version used the OpenAQ archive (same monitors) but it had only ~28 % of hours and no 2023-24.
- **Time:** CPCB `Timestamp` = start of the IST hour (checked: shifting by ±1 h made the match with OpenAQ worse).
  We store `ts` = UTC instant of the IST hour start.
- **Units:** NO2 / O3 µg/m³ → ppb and CO mg/m³ → ppm with 24.45 / molar mass (sensor formulas need ppb / ppm).
- **Weather:** Open-Meteo ERA5 only (one source, as the cloud would have), requested in UTC (`timezone=GMT`,
  because its India time-zone mode mislabels the half hour) and converted to IST hours ourselves.
- **Cleaning (labels must be right):** negatives, PM2.5 > 1500, PM10 > 3000, PM2.5 > 1.05·PM10, 6-hour flat lines,
  and a Hampel filter on ln(PM) for short glitches (≤ 3 h).
- **Stations:** 33 downloaded; Sector 16A Faridabad has no PM10 at all → dropped, **32 used**. Pusa is the DPCC
  station (Naraina, 28.6396 N 77.1463 E, Yadav et al. 2023 IJMR Table I), not the old IMD one.
- **How much:** station-hours with PM2.5 + PM10: train 2018-23 1.32 M, val 2024 0.25 M, test 2025-26 0.41 M.

## 2. Simulator (notebook 02)
- No ESP32 data yet → 3 virtual units per station (32 stations → 96 units). Physics + datasheet formulas, not ML,
  one small function per sensor with its formula written above it.
- PM sensor: a PMS7003 is fitted now (`PM_SENSOR = "PMS7003"`, simulated slope vs truth 2.19, R² 0.61; TODO_VERIFY by
  co-location). The SPS30 option is calibrated to the IIT-Delhi field study (winter slope 1.46-1.86, R² ≈ 0.9): it needs
  the "sensor is warmer inside" effect (without it R² was 0.65); result on CPCB data: median slope 1.75, R² 0.86.
- Wind: two perpendicular fans with one hall sensor each → speed along each axis only, features |u|, |v|.
- Each unit draws its own errors once (bias, gain, drift), plus random missing hours and ~1 outage per year.
- After the sensors, the ESP32's own processing runs (humidity correction κ = 0.4, dust flag, gas baselines, AQI).

- All simulator code is in `simulator.py` (`simulate(X, station, unit) -> Y`); notebook 02 explains and runs it.
  The simulator is only for training data and demos; at inference the real sensors replace it.

## 3. Features and targets (notebook 03)
- 23 features from the ESP32's own sensors (PM, T/RH/P, wind, rain, light, gas sensors relative to their clean-air
  baseline, time of day/week/year, current AQI, PM mask). Model B (SPS30 broken) drops the 6 PM features and adds
  5 "last known PM" numbers.
- Target: `ln(1+PM[t+h]) − ln(1+PM[t])` of the TRUE PM; order [pm25 h1..h3, pm10 h1..h3] (teacher: h1..h6).
- Splits by time: train 2018-23, val 2024, test 2025 - Sep 2026. 5 stations are never trained on (test at a new place).
- Every table has the same complete hourly time line, so "h hours later" = "h rows later" and a station's row is
  the same hour in the teacher grid.
- Windows (Model A): train 2.60 M, val 0.50 M, test 0.80 M (+ the 5 hold-out stations). Distillation uses a fixed
  sample of 200 000 train + 40 000 val windows for every student variant.
- AQI breakpoints checked against the official CPCB NAQI table; the open "Severe" band runs to 380 / 510 µg/m³.

## 4. Fairness rule (owner's requirement)
**The teacher never gets clean PM as input.** Its station inputs are the very same simulated sensor features the
student gets (checked in notebook 03: identical numbers). True PM is only the target. The teacher's advantage is
the network: other stations, the wind graph, transport physics, ERA5 weather (past + "future"). So distillation
tests network knowledge, not "clean data teaching noisy data".

## 5. Teacher (notebook 04) — TI-STGNN, PyTorch
- Fixed order: Transformer over time (each station alone) → wind-aware graph attention → ADR → decoder.
- Wind prior from TransNet: `ReLU(ws·cos(bearing − direction))/distance`. **Fix:** TransNet uses the direction
  the wind comes FROM, which points upwind; we use the direction it blows TO. For a receiving station we use the
  edge *from* the neighbour (transpose), otherwise it would listen to downwind stations.
- ADR: advection re-uses the attention scores, normalised per sender (mass is conserved); diffusion on a fixed
  distance graph (exact step via eigen-decomposition); reaction = small local network. h = 0.1, 2 steps
  (TransNet uses 0.01 on its own scale).
- Future weather = ERA5 actuals ("oracle weather") because archived forecasts do not exist for most of 2018-23 →
  a limitation to state in the report. Boundary-layer height is empty for Delhi in Open-Meteo, so it is not used.
- **Inference inputs:** the student needs only its own 23 sensor features (no weather, no internet). The teacher needs
  every node's 23 features + Open-Meteo past weather, the 6-h forecast and the current wind. Mismatch to remember:
  training used ERA5 actuals as "future weather", live inference must use forecasts (less accurate). Plan: report as a
  limitation now; later fine-tune the teacher on archived Open-Meteo forecasts (available from ~2022).
- Variants T0-T3 switch graph / wind / ADR off and on. ~0.2 M parameters.
- Out-of-fold outputs: 3 teachers each skip one block of training hours (± 30 h gap) and predict it, so the
  student never learns from a teacher that saw the answer.

## 6. Student (notebook 04, Part B) — GDC-CNN, Keras → TFLite FP32
- Na et al. 2026 architecture: patches of 2 h, 3 gated dilated depthwise conv blocks. Width 16 (A) / 64 (B).
- S0 plain; S1 + teacher forecast; S2 + teacher time features & attention (Na's L_feat, L_AFD); S3 + teacher GAT
  state; S4 + teacher ADR state. Weights: λp = 0.4 on the true target (Na; TODO_VERIFY in the paper's code),
  KD terms share the rest and ramp up over 10 epochs. Teacher forecast copied only where it beats S0 (Shmuel).
- Only the student is exported; projection heads are thrown away. FP32 on the ESP32 (INT8 only as a test).

## 6b. Benchmarks, tuning and ablation (notebook 04)
- Benchmarks for every model and persistence, on the test period, at the 27 training stations and separately at the
  5 never-seen stations, per output (pollutant x horizon) and per season: RMSE / MAE of the ln change, R², skill vs
  persistence, RMSE / MAE in ug/m3, CPCB category accuracy, and (teacher) q10-q90 band coverage.
- Teacher ablation: add one block (T0 -> T3) and remove one block from T3 (no wind, no future weather; no ADR = T2);
  T3 is trained 3 times (seeds) to see what is noise.
- Student: lr x width tuned on S0 (validation year only); S0-S4 with every allowed teacher; remove one loss from
  S4-T3; remove one sensor from S0 (which sensors does the node need?). Every student 3 seeds -> mean +- spread.
- Baselines on the student's exact test windows: persistence, ridge regression, LightGBM, and the teachers
  themselves ("gain kept" = (student - S0) / (teacher - S0)). Extra metrics: bias, high-PM error, Poor hit /
  false-alarm rate, direction accuracy, and "_device" versions that start from the node's own PM (deployment view).
- Validation for early stopping is spread over all 3 units and the whole year; normalisation uses all 3 units;
  out-of-fold: 5 folds for T3, 3 for T0-T2; results labelled hindcast (oracle future weather); run_manifest.json.
- Saved: only the best teacher (cloud inference later) + teacher outputs for the student, and only the best
  student (.keras, .tflite, student_model.h). All in one Kaggle session: teacher <= 8.5 h, student until 11.3 h.
- Fixed on the way: S0 now uses the same loss as S1-S4 (fair comparison); the teacher's "unseen" score uses only
  the 5 hold-out stations; feature gaps are filled from the past only (no future leakage).

## 7. Training place
- Laptop: data, simulator, features (notebooks 01-03). Kaggle T4 GPU: all training in one session (notebook 04: teacher, then student; `kaggle/kaggle_run.py`).
  Not TPU: PyTorch on TPU needs torch_xla and our graph code is not a good fit.

## 8. Earlier results (version 1 on the OpenAQ data, kept for the report; synthetic sensors)
- RandomForest students: RF-A 362 KB, test PM2.5 skill vs persistence +0.09/+0.13/+0.17 (h1/h2/h3).
- GDC-CNN S0: A 34.7 KB, B 202 KB; TFLite parity < 3·10⁻⁶.
- Interim LightGBM teacher (used CLEAN neighbour PM, so it breaks rule 4): output KD made RF-A slightly worse,
  CNN-A λ = 0.5 better by 1.1 % (all 6 outputs significant). Not comparable with the new teachers.
- RandomForest and LightGBM code were removed in the simplification (still in git history).
