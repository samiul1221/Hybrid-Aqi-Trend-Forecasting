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
- Teacher (CR v2, now in phase 1): multi-station TI-STGNN (Embedding → Transformer → wind-aware GAT → ADR → decoder), T+1..T+6, PyTorch, trained on Kaggle GPU. Variants T0–T3 (use_graph/use_wind/use_adr). It reads the SAME sensor-proxy features as the student (ADR 0019).
- Distillation: OOF teacher outputs + states (feat_seq, attention, h_spatial, h_adr) in `teacher_store` v2 → GDC-CNN student variants S0–S4 (output, temporal, spatial, physics KD). RF and the interim LightGBM teacher are frozen baselines (ADR 0016).
- Students are pretrained on a physics-based sensor simulator driven by real CPCB + Open-Meteo data, then fine-tuned on real logs.
- Edge must keep measuring, forecasting and alerting with no cloud at all.
- **Now:** milestones N0–N9 of `docs/specs/CHANGE_REQUEST_v2.md` [R10] (P0–P7 done). Progress: `docs/PROGRESS.md`; hand-off: `docs/HANDOFF.md`.

## Source of truth

Priority, high → low (CR v2 [R01]):
1. `docs/specs/research_spec_v2.md` (scientific design; **not in the repo yet** — ask the owner).
2. `docs/specs/CHANGE_REQUEST_v2.md` (`[R00]`–`[R14]`) + approved `docs/specs/architecture_review_v2.md` — supersede the ML-architecture parts of the old plans (banners mark the sections).
3. `docs/specs/edge_implementation_plan.md` (sections `[S00]`–`[S26]`) — hardware, firmware, pipeline, AQI, features, targets, Model B, contracts, OTA. **Read it before changing anything in `edge/`, `ml/` or `shared/`.**
4. `docs/specs/cloud_implementation_plan.md` (`[C00]`–`[C21]`) — backend/streaming/DB/API; its model section [C08] is superseded where it conflicts. Wire contracts (edge `[S16]`/`[S17]`, cloud `[C14]`) must stay byte-compatible.
- Cite section IDs when referencing decisions. YAML blocks = spec, JSON blocks = wire format. "(tune)" = start value; "(verify)" = check the primary source.
- Any deviation from the spec gets a short ADR in `docs/decisions/NNNN-*.md`. Never change behaviour silently. Spec paths map to repo paths via `docs/decisions/0001-repo-layout.md`.
- Uncertain constants (CPCB breakpoints, datasheet values, API variable names) are marked `TODO_VERIFY` with the source to check.

## Simplicity rules (CR v2 [R03], non-negotiable)

- Simplify code, never the architecture. One concept per file, < 250 lines, reads top-to-bottom.
- Every nn.Module/layer: docstring with what it does, paper + equation, input/output shapes; shape comments on reshaping lines; shape asserts at block boundaries.
- Dense tensors (N ≈ 30): no PyTorch Geometric, no sparse ops. No registries/factories/Hydra/Lightning; plain PyTorch loop; dataclass config from one YAML per experiment; variant switches are booleans.
- Comments say WHY (physics/paper), not WHAT. No silent defaults. `docs/walkthrough.md` explains the code reading order.

## Layout

```
shared/     aqi_shared: CPCB AQI, IST time utils, schema loader; schema/feature_schema.yaml;
            contracts/*.schema.json; fixtures/ (aqi_vectors, ist_bucket, feature_golden); codegen/ → C headers
edge/       firmware/aqi_node/ (Arduino sketch: aqi_node.ino + src/<component>/ per [S05]); firmware/test_host/ (host g++ tests);
            tools/ (replay, parity, mock_cloud)
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
- [ ] Teacher input = edge sensor domain (ADR 0019): teacher and student read the same sensor-proxy features (simulator → device pipeline → feature builder); clean CPCB PM is a target only, never an input (node or neighbours). Privilege = other stations, wind graph, ADR, declared weather.
- [ ] No future leakage: inputs at `base_ts` use only `ts ≤ base_ts`; OOF teacher predictions; `teacher_train_end < student_test_start`.
- [ ] CNN ships FP32 on device; INT8 only as ablation.

## Commands

Tooling: Python 3.11 via uv (workspace members `shared`, `ml`; deps pinned exactly with `uv add --bounds exact`), ruff, mypy (strict on `aqi_shared`), pytest. Firmware is an **Arduino sketch** (arduino-esp32 core 3.x via `arduino-cli`), not ESP-IDF — see `docs/decisions/0002-arduino-firmware.md`. Scripts are bash (Git Bash on Windows).

```bash
uv sync --all-packages                       # setup (installs Python 3.11 if missing)
bash scripts/test-all.sh                     # pytest + host C++ firmware tests
uv run pytest ml/tests/test_x.py::test_y     # single Python test
bash edge/firmware/test_host/run.sh          # host C++ tests only (g++, assert-based test_*.cpp)
bash scripts/lint.sh                         # ruff check + ruff format --check + mypy
bash scripts/build-firmware.sh               # arduino-cli compile, FQBN esp32:esp32:esp32s3:PSRAM=opi,FlashSize=16M
uv run python shared/codegen/gen_c_headers.py        # regenerate firmware headers after editing feature_schema.yaml or cpcb_breakpoints.json
uv run python shared/fixtures/make_fixtures.py       # regenerate AQI/IST fixtures; review the JSON diff (a changed value = behaviour change)
uv run python -m aqi_ml.data.build all-archive          # keyless CPCB (OpenAQ S3) -> weather -> base table -> data card (resumable; ADR 0006)
uv run python -m aqi_ml.data.build all --max-stations 2 # same via the OpenAQ API (needs OPENAQ_API_KEY in .env)
uv run python -m aqi_ml.sim.virtual                     # simulator [S10] -> data/synthetic/<version>/ (records, units, manifest)
uv run python -m aqi_ml.sim.validate                    # sim data card: SPS30 vs Delhi field calibration + signal checks
uv run python -m aqi_ml.features.dataset                # P5: records -> data/datasets/<version>/ (hourly + Model A/B windows)
uv run python -m aqi_ml.features.golden                 # regenerate shared/fixtures/feature_golden (firmware twin target)
uv run python -m aqi_ml.models.edge.rf.train --role A   # P6 RF student (resumable stages in data/models/<dataset>/rf-A/; --role B)
uv run python scripts/rf_summary.py data/models/feat-v1/rf-A/meta.json   # skill vs persistence per horizon
```

Data: config `ml/configs/data.yaml`; HTTP responses cached under `data/raw/cache/` (delete to refetch); base table `data/processed/base_table/station_id=*/`. Always request Open-Meteo with `timezone=GMT` — its Asia/Kolkata mode mislabels half-hour offsets (ADR 0005). pandas is 3.x: never convert datetimes with `astype(int64) // 10**9` (resolution may be seconds). Never use naive `datetime(...).timestamp()` (uses the PC's timezone).

Simulator: parameters only in `ml/configs/sim_params.yaml` (sources: `docs/research_notes/simulator_parameters.md`, deviations: ADR 0007). Hourly on-device processing (humidity correction, dust flag, baselines, flag bits) lives in `aqi_shared.pipeline`; calibration in `shared/calibration/default.json`. After changing physics, rerun `sim.validate` and keep the SPS30 winter slope inside the IIT-Delhi range.

Features (ADR 0008): `aqi_ml.features.builder` is the reference for firmware; `features.dataset.load(root, 'A'|'B', split=...)` returns X_seq/M_seq/X_rf/y (+X_extra for B), slicing X_seq from hourly.parquet by row. Synthetic labels = TRUE CPCB PM; real-log labels via `builder.real_log_targets` (measured only, I07).

Single sources: CPCB table `shared/src/aqi_shared/aqi/cpcb_breakpoints.json`; features `shared/schema/feature_schema.yaml` (hash = canonical JSON of parsed YAML); contracts `shared/contracts/*.schema.json` with golden payloads in `shared/fixtures/contract_examples.json`. Generated `*.gen.h` files are committed; a test fails if they are stale.

Firmware pure-logic code in `edge/firmware/aqi_node/src/{pipeline,aqi,features,models,decision}/` must not include Arduino headers, so host tests can compile it. Every GPIO number, I2C address and bus/electrical constant lives in `edge/firmware/aqi_node/src/config/pins.h` ([S04.2]/[S04.3]); drivers include it and never hard-code pins (`test_host/test_pins.cpp` checks forbidden/duplicate pins and ADC1-only analog inputs).

## Working rules

- Work milestone by milestone (N0–N9, CR v2 [R10]; listed in `docs/PROGRESS.md`). Claim a milestone at the top of PROGRESS.md; after each: run all tests, update PROGRESS.md + HANDOFF.md, write ADRs for deviations, then stop for review.
- GPU work (teacher T0–T3, OOF, student KD) runs on Kaggle via the Kaggle CLI (`KAGGLE_API_TOKEN` in `.env`; ADR 0018; `ml/kaggle/README.md` once N4 adds it). Data, simulator, RF, tests, export and parity stay on the laptop.
- Small commits per milestone step. Seeds fixed, configs in yaml, secrets only via `.env` (commit `.env.example`).
- No fake implementations presented as done. Stub only what the spec defers to phase 2, and label it.
