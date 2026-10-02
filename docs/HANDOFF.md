# HANDOFF — state of the project and what remains (2026-10-02, CR v2)

Read first: `CLAUDE.md` (rules, invariants, source-of-truth order), `docs/specs/CHANGE_REQUEST_v2.md`,
`docs/specs/architecture_review_v2.md` (approved decisions I1–I25), `docs/research_notes/teacher_input_audit.md`,
`docs/PROGRESS.md`, ADRs 0001–0019.

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
| P8 (frozen v1) | `baselines/lgbm_teacher/`, `distill/` | LightGBM "clean-neighbour" teacher + OOF store v1; RF out_KD; CNN-A out_KD results in `research_notes/e2_kd.md` (ADR 0011, 0016, 0019) |
| N0 step 0 | `docs/specs/architecture_review_v2.md`, ADRs 0012–0019 | approved review: tensor shapes, literature/novelty tables, I1–I25 |
| N1 cleanup | `baselines/`, spec banners | LightGBM teacher moved + frozen; LightGBM-teacher CNN KD runner removed; `export/verify.py`, `lineage.py` kept from the stopped cloud session |
| Firmware pins | `edge/firmware/aqi_node/src/config/pins.h` | [S04.2] map + host check `test_host/test_pins.cpp` |


## Data is NOT in git (`/data` is gitignored) — rebuild it (all keyless, resumable)
```bash
uv sync --all-packages
uv run python -m aqi_ml.data.build all-archive     # ~1 h: OpenAQ S3 archive + Open-Meteo (paced for rate limits)
uv run python -m aqi_ml.sim.virtual                # ~10 min: 90 virtual nodes (30 stations x 3 units)
uv run python -m aqi_ml.sim.validate               # SPS30 field-calibration check (sim_data_card.md)
uv run python -m aqi_ml.features.dataset           # ~5 min: data/datasets/feat-v1
uv run python -m aqi_ml.models.edge.rf.train --role A   # baselines; also --role B
uv run python -m aqi_ml.models.edge.gdc_cnn.train --role A   # S0 student; rerun until "done"; also B
uv run python -m aqi_ml.baselines.lgbm_teacher.train          # frozen v1 baseline only
```

## Next (CR v2 [R10])
1. **N2 graph windows** (`ml/src/aqi_ml/data/graph_windows.py`): per base_ts and unit u, [N,24,30] node features read
   from the SAME `hourly.parquet` as the student (guards G1–G4 of the audit), masks, static [N,6], X_fut [N,6,9]
   (ERA5 oracle, ADR 0012), wind_t, y/y_mask [N,12] from true CPCB PM (targets only), month shards, dataset card.
2. **N3 teacher code** (`ml/src/aqi_ml/models/teacher/`: config, graph, temporal, gat, adr, decoder, model, train, oof)
   in PyTorch (add torch as a pinned dependency), R07 unit tests on a tiny random graph, overfit test, `docs/walkthrough.md`.
3. **N4 Kaggle** (ADR 0018): upload `aqi-graph-windows`, `aqi-ml-src`; T3 smoke run; T0–T3; teacher-vs-S0 gate per horizon.
4. **N5** OOF + teacher_store v2 (ADR 0015). **N6** S0–S4 for Model A, S0–S1 for B, export parity. **N7** experiment matrix.
5. **N8** deploy chosen student + P10 firmware (Arduino, `pins.h`). **N9** thin ZenML wrappers (ADR 0017).
6. Phase-2 cloud backend (FastAPI, TimescaleDB, MQTT/WS, serving, Flutter) stays as in the cloud plan [C02]–[C16].

## Open questions for the owner
- `docs/specs/research_spec_v2.md` is not in the repo (needed for §39 item 7 and novelty wording); LightST PDF missing.
- Carry-over (PROGRESS.md): AQI gap rule, PM2.5 scale in calibration.json, SGP41 kn_u, MQ-7 part, SGP41 > 90 % RH, core update.
