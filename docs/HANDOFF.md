# HANDOFF — state of the project and what remains (2026-09-27)

Read first: `CLAUDE.md` (rules, commands, invariants), `docs/specs/*.md` (source of truth),
`docs/PROGRESS.md`, and ADRs 0001–0011 in `docs/decisions/`.

## Done (committed)
| Milestone | Where | Key result |
|---|---|---|
| P1 scaffold | repo root | uv workspace (`shared`, `ml`), ruff/mypy/pytest, host g++ tests, Arduino sketch (ADR 0002) |
| P2 shared | `shared/` | CPCB AQI + 61 vectors, IST buckets, feature schema + hash, JSON Schemas [S16], C-header codegen |
| P3 data | `ml/src/aqi_ml/data/` | keyless CPCB via OpenAQ S3 archive (33 Delhi-NCR stations), Open-Meteo ERA5, base table, data card, Hampel label cleaning (ADR 0005, 0006, 0009) |
| P4 simulator | `ml/src/aqi_ml/sim/` | physics forward models from datasheets + Delhi literature; SPS30 calibrated to IIT-Delhi co-location (slope 1.66 vs field 1.46–1.86) (ADR 0007) |
| P5 features | `ml/src/aqi_ml/features/`, `models/edge/model_b/` | 1.16M Model A / 1.09M Model B windows; golden fixture for firmware (ADR 0008) |
| P6 RF | `models/edge/rf/`, `export/rf_blob.py`, firmware `src/models/rf_eval.{h,c}` | RF-A 362 KB, parity 1.3e-7; test PM2.5 skill +0.09/+0.13/+0.17 (ADR 0009) |
| P7 CNN | `models/edge/gdc_cnn/`, `export/tflite.py` | CNN-A 34.7 KB, CNN-B 202 KB, TFLM ops only, host parity < 3e-6 (ADR 0010) |
| P8 (partial) | `models/teacher/interim_lgbm/`, `distill/` | LightGBM teacher + OOF teacher_store; precondition; RF out_KD A/B done; full_KD implemented + tested (ADR 0011) |
| Firmware pins | `edge/firmware/aqi_node/src/config/pins.h` | [S04.2] map + host check `test_host/test_pins.cpp` |

## Data is NOT in git (`/data` is gitignored) — rebuild it (all keyless, resumable)
```bash
uv sync --all-packages
uv run python -m aqi_ml.data.build all-archive     # ~1 h: OpenAQ S3 archive + Open-Meteo (paced for rate limits)
uv run python -m aqi_ml.sim.virtual                # ~10 min: 90 virtual nodes
uv run python -m aqi_ml.sim.validate               # SPS30 field-calibration check (sim_data_card.md)
uv run python -m aqi_ml.features.dataset           # ~5 min: data/datasets/feat-v1
uv run python -m aqi_ml.models.edge.rf.train --role A   # and --role B
uv run python -m aqi_ml.models.edge.gdc_cnn.train --role A   # rerun until "done" (resumable), and B
uv run python -m aqi_ml.models.teacher.interim_lgbm.train
uv run python -m aqi_ml.distill.run_kd --arch rf --role A    # also B; --arch cnn --role A/B
```
Memory note: the original PC had ~2.5 GB free; all steps are chunked/resumable and read column subsets.

## Remaining work (owner request, 2026-09-27)
1. **Finish P8**: CNN out_KD A (3 of 9 runs left) and B; E2 summary table; choose deployed variants per role/arch.
2. **P9 ZenML** (`ml/pipelines/`): data_snapshot, simulate, interim_teacher (incl. OOF → teacher_store),
   student_training(role, arch, kd), export_and_verify — thin wrappers over `aqi_ml`; local stack + MLflow;
   log schema hash, data snapshot hash, git sha, sim_params version, metrics per horizon ([C11]).
3. **P10 firmware** (Arduino, ADR 0002; use `src/config/pins.h` for every pin): [S05] tasks, [S06] pipeline,
   AQI + features C twins passing `shared/fixtures/*` (host tests), RF evaluator (done) + TFLM runner,
   [S15] decision logic, [S16] HTTPS ingest/events + MQTT downlink, store-and-forward, OTA; `edge/tools/mock_cloud`.
4. **Cloud (phase 2, `cloud/`)** per `docs/specs/cloud_implementation_plan.md`:
   - FastAPI backend: device APIs [C14.1] (ingest, forecast, manifest, events, artifacts), frontend APIs
     [C14.3], WebSocket [C12]; Pydantic models from `shared/contracts/*.schema.json` (contract tests exist).
   - Database: **local PostgreSQL + TimescaleDB** (hypertables per [C05]; docker-compose in `infra/`);
     keep it Supabase-compatible (plain Postgres + Timescale extension, SQL migrations via Alembic) — the
     owner will move to Supabase later.
   - Inference worker [C08.8]: hourly job; **serve the interim LightGBM teacher now** (6 h horizons: extend
     the teacher to h1..h6 per [C08.10]), CQR intervals [C08.8b], neighbour estimate, persistence fallback,
     outbox → MQTT (Mosquitto) + Redis → WebSocket.
   - MLOps [C11]: ZenML + MLflow + MinIO (compose), teacher/student pipelines, registry, manifests, retraining [C10].
   - **Graph neural network teacher TI-STGNN** [C08.5] (PyTorch + PyG): code it, unit tests [C08.5], pretrain on
     the virtual-node graph (P0 of [C08.7]), write feat_seq/attn to teacher_store (enables real full_KD).
   - `cloud/tools/node_sim` [C17] to replay simulated nodes against the API.
5. **Frontend**: Flutter app (`frontend/`) consuming [C14.3] REST + WebSocket: map of nodes, readings,
   cloud vs edge forecasts, AQI trend, events/alerts.
6. Keep every rule in CLAUDE.md: ADR for each deviation, TODO_VERIFY for unverified constants, tests +
   `docs/PROGRESS.md` update + commit per milestone, never train on own outputs (I07), wire contracts identical.

## Open questions for the owner (unchanged)
See `docs/PROGRESS.md` → "Open questions" (AQI gap rule, PM2.5 scale in calibration.json, MQ-7 part,
SGP41 > 90 % RH, core update).
