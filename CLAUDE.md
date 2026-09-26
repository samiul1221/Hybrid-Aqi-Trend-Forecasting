# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## How to communicate

- While working on a task, keep posting short, concise lines saying what you are doing and why.
- At the end of every response, explain what you did, why, and how, in a bit more detail. Use very simple language but stay concise.

## Project

B.Tech research project: edge–cloud hybrid AQI forecasting for 5–10 sensor nodes in Delhi.
- Edge: ESP32-S3 N16R8 node measures PM (SPS30), T/RH, pressure, wind, rain, lux, MQ-7, MQ-135, SGP41; computes India CPCB AQI locally.
- Edge students forecast humidity-corrected PM2.5/PM10 for T+1..T+3 → AQI trend. Model A uses PM; Model B drops PM (fallback when the SPS30 fails).
- Each student comes in two architectures: RandomForest (emlearn) and GDC-CNN (TFLite Micro, FP32).
- Cloud (phase 2): wind-aware graph "teacher" (TI-STGNN) over all nodes, T+1..T+6 with calibrated intervals, pushed to nodes by MQTT.
- Distillation: teacher outputs (out-of-fold) are stored in `teacher_store` and used to train students (Shmuel-style output KD for RF, Na-style full KD for CNN).
- Students are pretrained on a physics-based sensor simulator driven by real CPCB + Open-Meteo data, then fine-tuned on real logs.
- Edge must keep measuring, forecasting and alerting with no cloud at all.
- **Phase 1 (now):** edge firmware + all ML groundwork (data, simulator, features, students, interim LightGBM teacher, KD, ZenML). Progress: `docs/PROGRESS.md`.

## Source of truth

- `docs/specs/edge_implementation_plan.md` (sections `[S00]`–`[S26]`) — **read it before changing anything in `edge/`, `ml/` or `shared/`.**
- `docs/specs/cloud_implementation_plan.md` (`[C00]`–`[C21]`) — phase 2; read to keep interfaces compatible. Wire contracts (edge `[S16]`/`[S17]`, cloud `[C14]`) must stay byte-compatible.
- Cite section IDs when referencing decisions. YAML blocks = spec, JSON blocks = wire format. "(tune)" = start value; "(verify)" = check the primary source.
- Any deviation from the spec gets a short ADR in `docs/decisions/NNNN-*.md`. Never change behaviour silently. Spec paths map to repo paths via `docs/decisions/0001-repo-layout.md`.
- Uncertain constants (CPCB breakpoints, datasheet values, API variable names) are marked `TODO_VERIFY` with the source to check.

## Layout

```
shared/     aqi_shared: CPCB AQI, IST time utils, schema loader; schema/feature_schema.yaml;
            contracts/*.schema.json; fixtures/ (aqi_vectors, ist_bucket, feature_golden); codegen/ → C headers
edge/       firmware/ (ESP-IDF v5.x: main/, components/, test_host/ host-compiled C tests); tools/ (replay, parity, mock_cloud)
ml/         aqi_ml library: data/ sim/ features/ models/{edge,teacher}/ distill/ export/ eval/;
            pipelines/ (ZenML, thin wrappers over aqi_ml only); configs/ (yaml); tests/
cloud/ frontend/   phase-2 placeholders
infra/      docker-compose: MinIO, MLflow, ZenML server
data/       gitignored: raw/ interim/ processed/ synthetic/ teacher_store/
docs/       specs/ papers/ (incl. Datasheet/) decisions/ research_notes/ PROGRESS.md
```

Rules: no logic duplicated across folders; anything used by more than one component lives in `shared/`. The only allowed twins are the C and Python implementations of AQI and features, and both must pass the same fixtures in `shared/fixtures/`.

## Invariants checklist ([S01], cloud [C01])

- [ ] I01 AQI = India CPCB NAQI only, one implementation per language, both pass `shared/fixtures/aqi_vectors.json`.
- [ ] I02/I03 Timestamps UTC ISO-8601; buckets are IST clock hours; `ts` = UTC instant of the IST hour start (IST 10:00 → 04:30Z).
- [ ] I04 Target `y = ln(1+PM[t+h]) − ln(1+PM[t])`, h=1..3 edge (1..6 cloud); order `[pm25_h1..h3, pm10_h1..h3]`. Model B targets are relative to last valid PM ([S14]).
- [ ] I05 `feature_schema.yaml` is the only source of feature names/order/transforms; its hash is the schema identity; models embed it; firmware rejects mismatches.
- [ ] I06 `calibration.json` is the only calibration source (κ, scales, R_dust); every record carries `calibration_version`.
- [ ] I07 Never train on our own outputs: only `measured` values are targets (filled/imputed/forecast/cloud_estimate never).
- [ ] I08 Raw values immutable; processed values versioned (`pipeline_version`).
- [ ] I09 Teacher and students predict the same quantity, units, transform and hour buckets.
- [ ] I10 Edge works indefinitely with zero cloud.
- [ ] I11 Units: PM µg/m³, T °C, RH %, P hPa, wind m/s + degrees FROM, lux; gas sensors dimensionless.
- [ ] I12 MQ-7, MQ-135, SGP41 are model inputs only: never ppm, never AQI.
- [ ] Sensor pool is fixed (list above + DS3231 RTC). No MiCS-6814, no LDR, no invented sensors.
- [ ] No future leakage: inputs at `base_ts` use only `ts ≤ base_ts`; OOF teacher predictions; `teacher_train_end < student_test_start`.
- [ ] CNN ships FP32 on device; INT8 only as ablation.

## Commands

Tooling: Python 3.11 via uv (uv workspace: `shared`, `ml`), ruff, mypy (strict on `shared/`), pytest. Firmware: ESP-IDF v5.x. These become valid once milestone P1 lands.

```bash
uv sync --all-packages                       # setup (installs Python 3.11 if missing)
uv run pytest                                # all Python tests
uv run pytest ml/tests/test_x.py::test_y     # single test
uv run ruff check . && uv run ruff format --check .
uv run mypy shared/src                       # strict
idf.py -C edge/firmware build                # firmware (needs ESP-IDF env)
cmake -S edge/firmware/test_host -B build/test_host && cmake --build build/test_host && ctest --test-dir build/test_host   # host C tests
```

## Working rules

- Work milestone by milestone (P1–P10 in `docs/PROGRESS.md`). After each: run all tests, update PROGRESS.md, write ADRs for deviations, then stop for review.
- Small commits per milestone step. Seeds fixed, configs in yaml, secrets only via `.env` (commit `.env.example`).
- No fake implementations presented as done. Stub only what the spec defers to phase 2, and label it.
