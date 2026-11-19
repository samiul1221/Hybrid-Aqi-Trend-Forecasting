# Hybrid AQI forecasting for Delhi: cloud-assisted forecasting on an ESP32 node

B.Tech research prototype. A tiny **box model** runs on an ESP32 node and forecasts PM2.5 / PM10 (→ CPCB AQI trend)
for the next 1-3 hours. The **cloud** sees all nodes and sends every node a few bytes each hour: the forecast for the
shared city-wide change. The box model adds only its own local difference, helped by a small memory of its own daily
habits. Without internet it falls back to a box-alone model. Current design: `docs/ARCH_V3.md`.

**Research question:** how much better can a tiny on-device forecaster get when the nodes share a few bytes per
hour through the cloud and each node remembers its own habits, and how should it behave when the link fails?
(Version 1 tried to copy the big cloud graph model into the box model; that gained little, which led to this design.)

## The pipeline: five notebooks, each one step

Every notebook is self-contained (all code inside, small cells, each starting with a `#` line saying what it does),
reads the previous step from `data/` and writes the next.

| step | notebook | runs on | in → out |
|---|---|---|---|
| 1 | `notebooks/01_data.ipynb` | laptop | CPCB hourly files + Open-Meteo weather → `data/2_clean/` (one clean table per station) |
| 2 | `notebooks/02_simulator.ipynb` | laptop | explains `simulator.py` sensor by sensor, then runs it: true air + weather → what 3 virtual ESP32 units per station would read → `data/3_sensors/` |
| 3 | `notebooks/03_dataset.ipynb` | laptop | 23 features, targets, windows, graph grids → `data/4_features/` |
| 4 | `notebooks/04_train.ipynb` | Kaggle GPU (one session) | **Part A** cloud graph model T0-T3 + ablation runs (Transformer → wind graph → ADR physics → decoder), benchmarks, outputs for the box model; **Part B** box model: tuning, S0-S4 (with and without matching the cloud model), loss + sensor ablation, benchmarks → best box model as `.tflite` + `student_model.h` for the ESP32 |

**`simulator.py` — the sensor box.** `Y = simulator.simulate(X, station, unit)`: X = true hourly air + weather of one
station (a `data/2_clean` table), Y = what one ESP32 would log (raw sensor readings + its hourly processing). Only
needed to make training data (and for demos) while there are no real nodes; at real inference the real sensors
produce Y. Command line: `uv run python simulator.py data/2_clean/<station>.parquet --unit 0 -o Y.parquet`.

**What the models need at inference**

| | box model (ESP32) | cloud (graph model + regional forecaster) |
|---|---|---|
| sensor data | its own 23 features, last 24 h | the 23 features of every node, last 24 h |
| weather | none | Open-Meteo: past 24 h, forecast for the next 6 h, wind now |
| internet / cloud / simulator | not needed | internet needed; no simulator |

## Data folder (not in git)

```
data/
  1_download/   cpcb/<station>/  official CPCB hourly CSVs (downloaded by hand, 2018 - Sep 2026)
                weather/         Open-Meteo ERA5 weather per station (downloaded by notebook 01)
                stations.csv     33 stations: id, name, lat, lon, elevation
  2_clean/      one table per station, one row per IST hour: true PM2.5/PM10, CO/NO2/O3, ERA5 weather
  3_sensors/    one table per virtual unit (32 stations x 3): simulated ESP32 readings (+ true PM as the answer)
  4_features/   23 features per unit, grid_u0..2.npz (all stations side by side), window lists
  5_models/     results downloaded from Kaggle
```

## Run it

```bash
uv sync                                                    # Python 3.11 + packages
# run notebooks/01, 02, 03 top to bottom (laptop)
uv run python kaggle/kaggle_run.py upload                  # send data/4_features to Kaggle (private, 1.2 GB)
uv run python kaggle/kaggle_run.py train --smoke           # quick test on Kaggle (a few minutes)
uv run python kaggle/kaggle_run.py train                   # notebook 04 on a T4 GPU: cloud graph model, then box model (<= 12 h)
uv run python kaggle/kaggle_run.py status
uv run python kaggle/kaggle_run.py download                # results + firmware/aqi_node/student_model.h
```
Kaggle needs `KAGGLE_API_TOKEN=...` in `.env` (see `.env.example`).

## Other folders

- `kaggle/kaggle_run.py` — upload / run / download on Kaggle.
- `firmware/aqi_node/` — ESP32-S3 Arduino sketch (`src/config/pins.h` = every pin). Work starts after the box model is chosen.
- `backend/` — later: FastAPI + TimescaleDB + MQTT; `backend/mlops/` — later: ZenML + MLflow + MinIO.
- `frontend/` — later: dashboard.
- `docs/DESIGN.md` — every design decision in plain words. `docs/simulator_sources.md` — where every simulator number
  comes from. `docs/specs/` — original plans and change request. `docs/papers/` — papers and datasheets.
