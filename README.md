# aqi-hybrid

Edge–cloud hybrid AQI forecasting for Delhi: ESP32-S3 sensor nodes run small "student" models (RandomForest, GDC-CNN) that forecast PM2.5/PM10 and the CPCB AQI trend 1–3 h ahead; a cloud graph "teacher" (phase 2) forecasts 1–6 h for all nodes and distils its knowledge into the students.

- Specs (source of truth): `docs/specs/edge_implementation_plan.md`, `docs/specs/cloud_implementation_plan.md`
- Progress: `docs/PROGRESS.md` · Decisions: `docs/decisions/`

## Setup

Needs [uv](https://docs.astral.sh/uv/) and, for firmware, `arduino-cli` with the `esp32:esp32` core 3.x.

```bash
uv sync --all-packages        # installs Python 3.11 + all workspace packages
cp .env.example .env          # fill OPENAQ_API_KEY
bash scripts/test-all.sh      # Python tests + host C++ firmware tests
bash scripts/lint.sh          # ruff + mypy
bash scripts/build-firmware.sh
```

## Layout

`shared/` cross-component logic and fixtures · `ml/` ML library + ZenML pipelines · `edge/firmware/aqi_node/` Arduino sketch · `edge/firmware/test_host/` host tests · `infra/` MLOps compose · `cloud/`, `frontend/` phase 2.
