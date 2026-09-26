# 0001 — Repository layout differs from spec [S23] / [C18]

Status: accepted (2026-09-26)

## Context
The two specs disagree on paths, and the owner's phase-1 brief gives a third layout:
- edge [S23]: `edge/shared/`, `edge/schema/`, `edge/training/`, `edge/sim/`, `edge/eval/`, fixtures at `tests/aqi_vectors.json`
- cloud [C18]: `/shared/aqi`, `/shared/tests/aqi_vectors.json`, `/shared/feature_schema.yaml`, cloud ML in `/cloud/ml/`
- phase-1 brief: `shared/` package `aqi_shared`, all Python ML in `ml/` package `aqi_ml`, ZenML in `ml/pipelines/`

## Decision
Follow the phase-1 brief. Path mapping from spec text to repo:

| Spec says | Repo path |
|---|---|
| `tests/aqi_vectors.json`, `/shared/tests/aqi_vectors.json` | `shared/fixtures/aqi_vectors.json` |
| `tests/ist_bucket_fixture.json` | `shared/fixtures/ist_bucket_fixture.json` |
| `schema/feature_schema.yaml`, `/shared/feature_schema.yaml` | `shared/schema/feature_schema.yaml` |
| `shared/contracts/`, `/shared/contracts/` | `shared/contracts/*.schema.json` |
| `/shared/aqi`, `/shared/time_utils` | `shared/src/aqi_shared/aqi/`, `.../time_utils/` |
| edge `data/ sim/ training/ eval/` | `ml/src/aqi_ml/{data,sim,models,distill,export,eval}/` |
| edge `tools/` (parity, stub cloud) | `edge/tools/` |
| cloud `ml/students/`, `ml/export/`, `pipelines/` | `ml/` (shared by cloud in phase 2; cloud imports `aqi_ml`) |

The repo root is the existing git folder, not a new `aqi-hybrid/` subfolder; `aqi-hybrid` is the workspace name in `pyproject.toml`.

## Consequences
- Student training lives in one place (`ml/`), so phase-2 cloud pipelines call `aqi_ml` instead of duplicating it.
- Spec text keeps its old paths; read them through this table.
