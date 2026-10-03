# CLAUDE.md

Guidance for Claude Code in this repository.

## How to communicate
- While working, keep posting short lines saying what you are doing and why.
- At the end of every response, explain what you did, why, and how, in simple, concise language.

## What this is
A **student-style research prototype** (B.Tech): cloud graph teacher (TI-STGNN, PyTorch) distils network
knowledge into an ESP32-S3 student (GDC-CNN, Keras → TFLite FP32) that forecasts PM2.5/PM10 1-3 h ahead.
Start with `README.md`, then `docs/DESIGN.md` (all decisions in plain words).

## Code style (owner's request — most important)
- Easy to read and present beats "professional". Write like a careful student: plain functions, short comments
  that say WHY, no frameworks, no factories/registries/config classes, no type-checking ceremony.
- All code lives in the notebooks `notebooks/01..05`, each self-contained: many SMALL cells, every code cell starts
  with a `# what this cell does` comment, formulas in markdown above the code (one function per sensor / block).
  Each notebook reads the previous step from `data/` and writes the next; there is no shared `src/` package.
- One exception: `simulator.py` (repo root) is the sensor box `simulate(X, station, unit) -> Y`, used by notebook 02
  (which explains it sensor by sensor). One function per sensor, formulas in notebook 02's markdown.
- Don't add files or folders unless needed.
- Notebooks find their data on Kaggle automatically (`/kaggle/input`) and locally (`../data`).

## Hard rules (research correctness)
- Teacher and student inputs come only from simulated sensor features (same grids, checked in notebook 03).
  True CPCB PM is a TARGET only, never an input (node or neighbours).
- Target `y = ln(1+PM[t+h]) − ln(1+PM[t])`, order `[pm25_h1..h3, pm10_h1..h3]` (teacher h1..h6).
- No future leakage: inputs use hours ≤ t; teacher outputs for student training are out-of-fold.
- Times: IST clock hours; `ts` is the UTC instant of the IST hour start. Open-Meteo: always request `timezone=GMT`.
- pandas 3: never `astype(int64) // 10**9` for datetimes; use `(ts - Timestamp(0, tz="UTC")) // Timedelta("1s")`.
- Sensor pool is fixed (SPS30, HDC1080, BMP280, anemometer, rain board, BH1750, MQ-7, MQ-135, SGP41, DS3231).
- CPCB AQI only. Unverified constants get `TODO_VERIFY`. Never present a stub as done.

## Where things are
- `data/` (git-ignored): `1_download → 2_clean → 3_sensors → 4_features → 5_models` (see README).
  Truth = official CPCB hourly files (`data/1_download/cpcb/`, timestamp = start of the IST hour), 32 stations used.
  Splits: train 2018-2023, val 2024, test 2025-Sep 2026; 5 hold-out stations never trained on.
- Training runs only on Kaggle (never train models on the laptop; local runs are tiny smoke tests at most).
- `kaggle/kaggle_run.py`: upload / run / download on a Kaggle T4 GPU (`KAGGLE_API_TOKEN` in `.env`).
- `firmware/aqi_node/` Arduino sketch; every pin in `src/config/pins.h`.
- `backend/` (+ `backend/mlops/`: ZenML, MLflow, MinIO) and `frontend/`: later phase, keep as is.
- `docs/specs/` original plans (reference), `docs/simulator_sources.md`, `docs/papers/`.

## Commands
```bash
uv sync                                   # setup (Python 3.11)
uv add --bounds exact <pkg>               # add a dependency (pinned)
uv run python kaggle/kaggle_run.py upload | teacher | student | status teacher | download teacher
```
