# backend/ — later phase (not built yet)

FastAPI backend + TimescaleDB (PostgreSQL, Supabase later) + MQTT/WebSocket streaming; serves the teacher's
forecasts to nodes and the dashboard. Plan: `docs/specs/cloud_implementation_plan.md`.

`mlops/` — training/deployment tooling for later: ZenML pipelines (`mlops/pipelines/`) and a docker-compose
with MinIO + MLflow + ZenML server (`mlops/docker-compose.yml`, not run yet). The pipelines will just call the
same functions the notebooks use (`src/`).
