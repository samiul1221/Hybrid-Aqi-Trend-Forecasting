# CLOUD SPEC — Backend, Graph Teacher Model, Streaming and MLOps for the AQI Node Network

<doc_meta>
scope: "CLOUD PORTION: FastAPI backend, database, MQTT/WebSocket streaming, cloud graph teacher model (training + hourly inference), edge-student retraining pipelines (ZenML), events, frontend-facing APIs. Not the device firmware."
companion_doc: "edge_implementation_plan.md. Wire contracts in edge [S16]/[S17] are shared and MUST stay byte-compatible. When this doc says 'edge [Sxx]' it refers to that file."
audience: "engineer or coding agent with zero prior context"
format_conventions:
  - "Sections have stable IDs like [C08]. Cross-references use these IDs."
  - "YAML blocks = machine-readable specs (source of truth). JSON blocks = wire formats."
  - "MUST = required; SHOULD = strongly recommended; MAY = optional."
  - "(tune) = starting value to tune on data. (verify) = check against primary source before release."
read_order: "C00 → C01 → C02 → C08 (model, most important) → then build in milestone order [C19]"
monorepo: "Use one repo: /edge (firmware + edge training), /cloud (this doc), /shared (AQI code, contracts, fixtures, feature_schema). Where the edge doc says edge/shared or edge/schema, use /shared."
</doc_meta>

---

## [C00] HOW TO USE THIS DOCUMENT

1. Read [C01] invariants. They override everything else.
2. Read [C02] for the system picture and hourly timeline.
3. Read [C08] fully before writing any ML code. It defines inputs, architecture, training, inference, intervals and teacher outputs.
4. Build in the milestone order of [C19]. Each milestone lists acceptance tests.
5. Never change a wire contract ([C14]) without changing the edge doc in the same commit.

---

## [C01] INVARIANTS

```yaml
shared_with_edge:
  - "Edge invariants I01–I12 (edge [S01]) apply unchanged: CPCB AQI only; IST clock-hour buckets with ts = UTC of IST hour start; target = ln(1+PM[t+h]) - ln(1+PM[t]); single feature_schema; single calibration source; never train on own model outputs; raw immutable; teacher and students predict the same quantities; edge autonomy; units; gas sensors are proxies."
cloud_specific:
  CI01_one_aqi_library: "Cloud imports /shared/aqi (Python). It MUST pass /shared/tests/aqi_vectors.json in CI."
  CI02_targets_only_measured: "Loss masks and evaluation use only values whose flag is 'measured'. filled / missing / sensor_fault / warming / cloud_estimate values are never targets."
  CI03_flagged_rows_are_kept: "Every ingested row is stored with its flags, including filled and fallback rows. They are valid model INPUTS (with masks) and are the source of real outage windows for edge Model B training."
  CI04_no_future_leakage: "Inference and training windows at base_ts use only data with ts ≤ base_ts, plus weather FORECASTS issued ≤ base_ts + 1 h. Never observed future weather at inference; for training use archived forecasts when available (C07)."
  CI05_teacher_student_alignment: "Teacher horizons h1..h6; edge students use the h1..h3 subset. Output layout is pollutant-major: [pm25_h1..h6, pm10_h1..h6]; edge subset [pm25_h1..h3, pm10_h1..h3] (edge [S07] output_order)."
  CI06_deterministic_runs: "Each hourly inference run is keyed by (base_ts, run_kind). Re-running produces a new issued_at, never overwrites history."
  CI07_versions_everywhere: "Every forecast row stores teacher_version, feature_schema hash, calibration_version and run_id. Every student artifact stores teacher_version and training data range."
  CI08_edge_independence: "Cloud failure MUST NOT break nodes. Cloud never sends commands that disable edge fallback."
```

---

## [C02] SYSTEM CONTEXT

### C02.1 What the cloud does

```yaml
cloud_responsibilities:
  ingest: "receive hourly node records (HTTPS POST /api/v1/ingest) and immediate events (POST /api/v1/events); validate, store with flags"
  history: "keep full history in PostgreSQL+TimescaleDB; ML needs 24 h input windows and months of training data"
  inference: "every hour: build graph inputs from history + weather forecasts → teacher model → per-node forecast T+1..T+6 with intervals"
  streaming:
    to_nodes: "MQTT push per node (retained), HTTP GET fallback"
    to_frontend: "WebSocket stream of forecasts, events, node status, latest readings"
  events: "store edge events, re-check edge-vs-cloud disagreement server-side (cloud_confirmed), stream to frontend as alerts"
  training:
    teacher: "periodic retraining of the graph model"
    students: "periodic retraining of edge Model A (with PM) and Model B (PM dropped), RF and GDC-CNN, with distillation from the teacher"
    delivery: "export → verify → register → manifest → OTA to nodes (edge [S18])"
  mlops: "ZenML pipelines + MLflow tracking + object storage + model registry"
  frontend_api: "REST + WebSocket for a client dashboard that shows readings, forecasts, trends, events"
two_main_workloads:
  inference_worker: "hourly forecast job + late re-run + event checks + streaming"
  training_worker: "ZenML pipelines for teacher and students on schedules/triggers"
```

### C02.2 Services

```yaml
services:
  api:            {tech: "FastAPI + Uvicorn (async, asyncpg)", role: "device API, frontend REST, WebSocket, admin"}
  inference_worker: {tech: "Python process (PyTorch, PyG)", role: "hourly teacher inference, AQI post-processing, intervals, publish"}
  training_worker:  {tech: "ZenML client + pipelines", role: "teacher + student training, export, registry, manifests"}
  scheduler:      {tech: "APScheduler (inside a small service) + Redis locks", role: "cron triggers for inference, weather fetch, retraining, housekeeping"}
  db:             {tech: "PostgreSQL 16 + TimescaleDB", role: "all history, forecasts, events, registry mirror"}
  redis:          {tech: "Redis 7", role: "cache, distributed locks, pub/sub fan-out to WebSocket, rate limiting, small job queue (Arq or RQ)"}
  mqtt_broker:    {tech: "Mosquitto (small) or EMQX (large)", role: "push to nodes; node presence (LWT)"}
  object_store:   {tech: "MinIO (S3 API)", role: "model artifacts, datasets snapshots, teacher_store parquet, ZenML artifact store"}
  mlflow:         {tech: "MLflow tracking server", role: "experiment tracking (ZenML integration)"}
  zenml_server:   {tech: "ZenML server", role: "pipeline metadata, model control plane"}
  monitoring:     {tech: "Prometheus + Grafana (+ Loki optional)", role: "metrics, dashboards, alerts"}
  frontend:       {tech: "any SPA (out of scope)", role: "consumes REST + WebSocket"}
```

### C02.3 Hourly timeline (IST; must match edge [S02.2])

```yaml
hourly_timeline:
  "HH:00:00": "IST hour H closes. Nodes build record for hour H-1 (base_ts = UTC of H-1 start)."
  "HH:00:10-HH:02:00": "Nodes POST /ingest (jittered). API stores rows."
  "HH:05:00": "scheduler: fetch/refresh Open-Meteo forecasts for all nodes (cache)"
  "HH:08:00": "ingest deadline for this base_ts. Inference job starts (Redis lock 'infer:{base_ts}')."
  "HH:08:00-HH:10:00": "build inputs → teacher → post-process → store → MQTT publish per node → WebSocket fan-out → server-side event checks"
  "HH:10:00": "SLO: ≥95% of active nodes have a published forecast"
  "HH:20:00": "late re-run if any record for base_ts arrived after HH:08 (run_kind=late). Re-publish same based_on_ts with newer issued_at."
  "any time": "POST /events handled immediately → stored → WebSocket"
```

### C02.4 Data flow diagram

```
[ESP32 nodes] --HTTPS POST /ingest (hourly, backlog)--> [FastAPI api] --> [PostgreSQL/Timescale]
[ESP32 nodes] --HTTPS POST /events (immediate)-------> [FastAPI api] --> events table --> Redis pub/sub --> WebSocket --> [Frontend]
[Open-Meteo]  <--scheduler fetch-- weather cache (db)
[scheduler HH:08] --> [inference_worker] --reads--> db (24h history + weather fcst + registry)
                                         --runs---> teacher model (production version from registry/object store)
                                         --writes-> cloud_forecasts + outbox
[outbox publisher] --MQTT retained--> aqi/v1/nodes/{id}/forecast --> [ESP32 nodes]
                   --Redis pub/sub--> WebSocket --> [Frontend]
[scheduler weekly/monthly or trigger] --> [training_worker / ZenML]
      teacher pipeline --> registry (staging→production) --> inference_worker hot-reload
      student pipelines --> artifacts (MinIO) --> manifest --> MQTT manifest notify --> nodes OTA
```

---

## [C03] GLOSSARY (cloud additions; see edge [S03] for the rest)

```yaml
glossary:
  base_ts: "UTC instant of the IST hour start that a record/forecast is based on"
  run_kind: "scheduled | late | backfill | manual"
  node_inputs_complete: "true if the node's record for base_ts arrived before the run started"
  teacher_version: "registry version id of the production graph model"
  teacher_store: "offline store of teacher outputs (predictions, features, attention) per node per base_ts for distillation"
  OOF: "out-of-fold: prediction for a window made by a model not trained on that window's time block"
  CQR: "conformalized quantile regression (interval calibration)"
  outbox: "db table of messages to publish; guarantees a stored forecast is eventually published"
  reference_node: "optional CPCB station node in the graph (real pollutant data), used in pretraining and optionally at runtime"
  virtual_node: "simulated low-cost node at a CPCB station location (edge [S10] simulator)"
```

---

## [C04] TECH STACK AND DEPLOYMENT

```yaml
stack:
  language: "Python 3.11"
  api: "FastAPI, Pydantic v2, Uvicorn/Gunicorn, asyncpg, SQLAlchemy 2 (core) or SQLModel, Alembic migrations"
  ml: "PyTorch 2.x, PyTorch Geometric (PyG) for graph ops, scikit-learn, LightGBM, emlearn (RF export), TensorFlow/Keras (CNN student → TFLite), numpy, pandas, pyarrow"
  mlops: "ZenML (pipelines, model control plane), MLflow (experiment tracker via ZenML integration), MinIO (artifact store)"
  messaging: "paho-mqtt / aiomqtt for publishing; Redis pub/sub for WebSocket fan-out"
  scheduling: "APScheduler with Redis-backed locks (single-run guarantee)"
  observability: "prometheus-client, Grafana, structured JSON logging"
deployment:
  target: "docker-compose on one VM for ≤100 nodes; Kubernetes optional later"
  sizing_small: "4 vCPU, 16 GB RAM, 200 GB SSD; GPU optional (only speeds up pretraining)"
  compose_services: [api, inference_worker, training_worker, scheduler, outbox_publisher, db, redis, mqtt_broker, minio, mlflow, zenml_server, prometheus, grafana]
  ports: {api: 8000, mqtt_tls: 8883, grafana: 3000, minio: 9000, mlflow: 5000, zenml: 8080}
  config: "12-factor env vars; secrets via .env (dev) / secret manager (prod)"
```

---

## [C05] DATA MODEL (PostgreSQL + TimescaleDB)

```yaml
tables:
  nodes:
    pk: node_id
    cols: {lat: float8, lon: float8, height_agl_m: float4, elevation_m: float4, site_notes: text, sensors: jsonb,
           wind_trusted: bool, sheltering_factor: float4, install_date: date, burn_in_until: timestamptz,
           status: "active|burn_in|maintenance|retired", cohort: "canary|stable", calibration_version: text,
           token_hash: text, created_at: timestamptz}
  calibrations:
    pk: calibration_version
    cols: {kappa: float4, pm10_scale: float4, r_dust: float4, scope: "global|node", node_id: "text|null", valid_from: timestamptz, source: jsonb, artifact_uri: text}
  readings_hourly:            # hypertable on ts, chunk 7 days, compress after 30 days
    pk: [node_id, ts]
    cols: {raw: jsonb, pm1: float4, pm25_raw: float4, pm10_raw: float4, pm25_corr: float4, pm10_corr: float4,
           t: float4, rh: float4, p: float4, wind_u: float4, wind_v: float4, wind_speed: float4, wind_dir: float4,
           rain_frac: float4, lux: float4, mq7_rs: float4, mq135_rs: float4, sraw_voc: int4, sraw_nox: int4,
           coarse_ratio: float4, dust: bool, aqi_pm: int2, aqi_valid_hours: int2, baselines: jsonb,
           flags: jsonb, pm_state: "ok|filled_le_1h|fallback_model_b|unavailable",
           calibration_version: text, pipeline_version: text, schema_hash: text, fw_version: text,
           cloud_pm25_corr: float4, cloud_pm10_corr: float4, pipeline_mismatch: bool, outlier_flags: jsonb,
           received_at: timestamptz, is_backlog: bool, late: bool, sys: jsonb}
  edge_forecasts:
    pk: [node_id, base_ts]
    cols: {model_role: "A|B", arch: "rf|cnn", model_version: text, y: "float4[6]", pm25: "float4[3]", pm10: "float4[3]", aqi: "int2[3]", trend: text, inference_ms: int4}
  forecast_runs:
    pk: run_id
    cols: {base_ts: timestamptz, run_kind: text, started_at: timestamptz, finished_at: timestamptz, teacher_version: text,
           nodes_total: int4, nodes_complete: int4, status: "ok|partial|failed|fallback", error: text}
    unique: [base_ts, run_kind, started_at]
  cloud_forecasts:            # hypertable on base_ts
    pk: [node_id, base_ts, issued_at]
    cols: {run_id: uuid, teacher_version: text, forecast_source: "teacher|persistence_fallback",
           node_inputs_complete: bool, y_pm25: "float4[6]", y_pm10: "float4[6]", pm25: "float4[6]", pm10: "float4[6]",
           q10_pm25: "float4[6]", q90_pm25: "float4[6]", q10_pm10: "float4[6]", q90_pm10: "float4[6]",
           aqi: "int2[6]", aqi_lower: "int2[6]", aqi_upper: "int2[6]", interval_coverage_target: float4, interval_quality: "calibrated|provisional", trend: text,
           gas_subindex: jsonb, neighbour_pm_estimate: jsonb, schema_hash: text, calibration_version: text}
  weather_fcst:               # hypertable on valid_ts
    pk: [node_id, issue_ts, valid_ts]
    cols: {t2m: float4, rh2m: float4, sp: float4, ws10: float4, wd10: float4, u10: float4, v10: float4, precip: float4, swrad: float4, blh: float4, source: text}
  weather_obs:                # reanalysis/actuals for training and backfill
    pk: [node_id, ts]
    cols: {same_vars_as_weather_fcst: true, source: text}
  ref_stations:   {pk: station_id, cols: {name: text, lat: float8, lon: float8, source: "CPCB"}}
  ref_obs:        {pk: [station_id, ts], cols: {pm25: float4, pm10: float4, no2: float4, o3: float4, co: float4, so2: float4, nh3: float4, t: float4, rh: float4, ws: float4, wd: float4, flags: jsonb}}
  neighbour_estimates: {pk: [node_id, ts], cols: {pm25: float4, pm10: float4, method: text, valid: bool, flag: "cloud_estimate"}}
  events:
    pk: event_id
    cols: {node_id: text, type: "local_event|sensor_check|pm_sensor_fault", status: "open|update|closed", severity: "low|medium|high",
           source: "edge|cloud", ts_detected: timestamptz, base_ts: timestamptz, evidence: jsonb, cloud_confirmed: "bool|null",
           ack_by: "text|null", ack_at: "timestamptz|null", closed_at: "timestamptz|null", created_at: timestamptz, updated_at: timestamptz}
  outbox:
    pk: id
    cols: {channel: "mqtt|ws", topic: text, payload: jsonb, qos: int2, retained: bool, created_at: timestamptz, sent_at: "timestamptz|null", attempts: int4}
  models:
    pk: model_id
    cols: {kind: "teacher|student", role: "teacher|A|B", arch: "tistgnn|rf|cnn|lgbm_interim", version: text, schema_hash: text,
           teacher_version: "text|null", data_range: tstzrange, kd_variant: "no_KD|out_KD|full_KD|null", alpha: "float4|null",
           metrics: jsonb, artifact_uri: text, sha256: text, size_bytes: int8, arena_bytes: "int4|null",
           stage: "staging|canary|production|archived", zenml_model_version_id: text, created_at: timestamptz}
  manifests:        {pk: manifest_version, cols: {content: jsonb, etag: text, created_at: timestamptz, rollout: jsonb}}
  node_deployments: {pk: [node_id, reported_at], cols: {manifest_version: int4, reported_versions: jsonb}}
  teacher_store_index: {pk: [teacher_version, node_id, month], cols: {uri: text, oof: bool, rows: int4}}
  training_runs:    {pk: run_id, cols: {pipeline: text, params: jsonb, status: text, metrics: jsonb, started_at: timestamptz, finished_at: timestamptz}}
  schema_versions:  {pk: schema_hash, cols: {version: int4, content: text, created_at: timestamptz}}
  users:            {pk: user_id, cols: {email: text, role: "viewer|admin", password_hash: text}}
  audit_log:        {pk: id, cols: {actor: text, action: text, target: text, details: jsonb, ts: timestamptz}}
  ingest_rejects:   {pk: id, cols: {node_id: text, ts: timestamptz, reason: text, payload: jsonb, received_at: timestamptz}}
views:
  v_targets: "readings_hourly rows where PM flags = measured and pm_state='ok' (training targets only)"
  v_inputs: "all rows with per-variable masks (1=measured, 0 otherwise); used to build model inputs"
  v_outage_windows: "contiguous runs of pm_state ∈ {fallback_model_b, unavailable, filled_le_1h} with hours_since_pm_lost; used for Model B training (C09)"
indexes: ["readings_hourly (node_id, ts DESC)", "cloud_forecasts (node_id, base_ts DESC)", "events (status, updated_at)", "outbox (sent_at) WHERE sent_at IS NULL"]
retention: {readings_hourly: "forever (compressed)", cloud_forecasts: "forever (compressed)", outbox_sent: "7 days", ingest_rejects: "90 days"}
backups: "daily pg_basebackup + WAL archiving to MinIO; MinIO bucket versioning; monthly restore drill"
```

---

## [C06] INGESTION PIPELINE (device → DB)

```yaml
ingest_post:
  endpoint: "POST /api/v1/ingest (contract: edge [S16.2])"
  auth: "Bearer node token → lookup node_id; token node_id MUST equal payload node_id"
  validation:
    - "Pydantic model generated from /shared/contracts/ingest.schema.json"
    - "schema_hash MUST exist in schema_versions (else reject record with reason 'unknown_schema')"
    - "ts on IST hour boundary (minute = 30 in UTC). Reject otherwise."
    - "clock checks (edge C14): reject ts > now + 1 h; accept older than 30 days only with is_backlog semantics"
    - "range checks: PM 0–1000, T -10–55, RH 0–100, P 850–1100, lux 0–65535; out-of-range values set to null + flag 'invalid'"
  idempotency: "upsert on (node_id, ts). Identical payload → no-op. Different payload → overwrite, log in audit_log."
  derived_fields:
    - "late = received_at > (ts + 1 h + 8 min)"
    - "is_backlog = ts < floor_ist_hour(now) - 1 h"
    - "cloud re-computes pm25_corr/pm10_corr from raw with the record's calibration_version (/shared pipeline). If |cloud − edge| > 1%: pipeline_mismatch=true; training uses cloud values."
    - "outlier_flags via Hampel filter (centered 24 h window, κ=3, scale 1.4826·MAD; Wu et al. 2026) computed in a nightly job; flag only, never delete"
  side_effects:
    - "store edge_fc into edge_forecasts"
    - "store events block (if any) into events (same upsert rules as /events)"
    - "publish 'reading' message to Redis channel for WebSocket"
    - "if late for an already-run base_ts → mark base_ts for late re-run"
    - "if is_backlog → enqueue teacher_store backfill for that node/time (C08.9)"
  response: '{"accepted": [ts...], "rejected": [{"ts": ..., "reason": ...}]}'
  performance: "p99 < 300 ms for 48-record batch; batch insert via COPY or executemany"
flag_handling_summary:
  measured: "input (mask=1) and target"
  filled: "input (mask=0, value kept for display); never target"
  missing: "input mask=0; no value"
  sensor_fault / warming: "input mask=0; never target"
  high_humidity / dust: "input + feature; target allowed (flag becomes a feature)"
  cloud_estimate: "never input to the node's own PM channels, never target; display only"
```

---

## [C07] EXTERNAL DATA

```yaml
open_meteo:
  runtime_forecast:
    api: "Open-Meteo forecast API at each node lat/lon, hourly, next 12 h (verify variable names)"
    vars: [temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m, wind_direction_10m, precipitation, shortwave_radiation, boundary_layer_height]
    schedule: "every hour at HH:05 IST; also at HH:07 retry if failed"
    cache: "weather_fcst (issue_ts = fetch time rounded down to hour)"
    failure: "use most recent cached issue ≤ 6 h old, shifted; if none → mark weather_missing and let model use masks"
  training_history:
    preferred: "historical-forecast archive (forecasts as issued) → honest future covariates (CI04)"
    fallback: "reanalysis actuals (ERA5-based archive) with flag future_covariates='actual'; report as limitation"
  alignment: "Open-Meteo is on UTC hours; IST buckets are offset 30 min. Instantaneous vars: mean of the two bracketing UTC values. Accumulated/averaged vars (precipitation, shortwave): proportional overlap. Verify how the API handles +05:30."
  wind_for_graph: "graph edges use Open-Meteo 10 m wind at each node (consistent, unsheltered). Node wind used only if nodes.wind_trusted=true (edge C05)."
cpcb_reference (optional):
  use: ["pretraining dataset (virtual nodes + reference nodes)", "optional runtime reference nodes if a live feed is available", "gas sub-indices for edge AQI_full"]
  source: "CPCB CCR portal or OpenAQ API (mirror)"
  time: "IST From–To hourly → ts = From converted to UTC"
calendar:
  vars: [ist_hour, ist_dow, ist_doy, is_holiday, is_festival_window]   # Diwali ±3 days, New Year, etc. (maintain a yaml list)
static:
  elevation: "Open-Meteo elevation API or SRTM at node coords"
  node_meta: "height_agl_m, sheltering_factor, node_type"
fires (optional): "NASA FIRMS daily fire counts in Punjab/Haryana box, lagged 1–2 days, as a global feature Oct–Nov"
```

---

## [C08] THE CLOUD MODEL (graph teacher) — MOST IMPORTANT

### C08.1 Task definition

```yaml
task:
  name: "hourly multi-node multi-horizon PM forecasting with calibrated intervals"
  at_each_base_ts:
    input_history: "L = 24 hourly steps (t-23 .. t) for every node"
    input_future_covariates: "weather forecast for t+1 .. t+6 per node"
    input_graph: "static distance graph + dynamic wind graph for every hour in the window"
    output: "for every node i, h = 1..6, pollutant p ∈ {pm25, pm10}: point ŷ[i,h,p] and quantiles q_0.1, q_0.9"
    target: "y[i,h,p] = ln(1+PM_p[t+h]) - ln(1+PM_p[t])   (identical to edge [S07])"
    output_layout: "pollutant-major: [pm25_h1..h6, pm10_h1..h6] (CI05)"
  derived_products: ["PM forecasts", "AQI T+1..T+6 via shared CPCB code (rolling 24 h)", "AQI intervals", "trend label", "neighbour PM estimate for failed nodes", "teacher outputs for distillation"]
  horizons_rationale: "Edge compares h1..h3; frontend shows h1..h6. TransNet showed graph transport models are strongest ≤48 h, so 6 h is well inside the reliable range."
```

### C08.2 Literature basis (what each paper contributes)

```yaml
papers:
  Gu2024_AtmosEnv:
    link: "https://www.sciencedirect.com/science/article/abs/pii/S1352231024006873"
    facts_from_abstract: ["dynamic wind-driven graph: edge weights change with wind to model inter-station PM2.5 transport", ">70% of station-level predictions beat leading models", "ablation: inter-station transport module mattered most", "transfer learning cut training data needed by ~80% (verify exact figure in full text)"]
    we_use: ["dynamic wind edges", "transfer learning: pretrain on a large station graph, fine-tune on our small network (C08.7)"]
  Wang2024_STOTEN:
    link: "https://www.sciencedirect.com/science/article/abs/pii/S0048969724043778"
    facts_from_abstract: ["directed graph rebuilt each hour from wind direction", "graph attention assigns dynamic neighbour weights", "sequence-to-sequence temporal model with attention", "meteorological forecasts used as inputs", "target: ozone"]
    we_use: ["directed hourly wind graph", "attention over neighbours", "weather FORECASTS as future covariates in the decoder (C08.5 step 7)"]
  Wu2026_IJGI_WSDST_GAT:
    link: "https://www.mdpi.com/2220-9964/15/3/101"
    method:
      preprocessing: "Hampel filter (centered 24 h window, κ=3); ffill/bfill then 24 h rolling median for gaps; min–max scaling; mutual-information input screening"
      edge_weight: "[A_t]ij = exp(-d_ij/λ) · max(0, cos(θ_ij,t − φ_t)); θ_ij = azimuth i→j; φ_t = wind direction (downwind); d_ij great-circle distance"
      spatial: "dynamic GAT with edge-biased attention: e_ij = φ(W h_i ∥ W h_j) + β·g([A_τ]_{i→j}); softmax over neighbours; multi-head, residual, layer norm; applied at every hour τ in the window"
      temporal: "per-station pre-norm Transformer encoder with sinusoidal PE; non-autoregressive decoder with H learned queries"
      fusion: "GLU(s_i ∥ u_i) = (s_i∥u_i)W_h ⊙ σ((s_i∥u_i)W_g)"
      heads: "point head + quantile heads τ ∈ {0.1, 0.5, 0.9}"
      loss: "λ_MSE·MSE + λ_Q·Σ_τ pinball_τ"
      training: "Adam lr 1e-3, batch 32, early stopping on val RMSE; chronological 70/15/15 split; 35 Beijing stations, 2022"
    results: "MAE 5.25, RMSE 9.58 µg/m3, R² 0.957; PICP 94.01%, PINAW 0.174; beat LSTM, GRU, GCN-LSTM, ST-GAT, ST-Transformer, STD-GCN"
    evaluation_idea: "meteorological volatility index V_t = rolling std of wind speed; test hours >P80 = high volatility, <P20 = steady"
    we_use: ["edge-weight prior (C08.3)", "edge-biased attention", "temporal Transformer + H-query decoder", "GLU fusion", "quantile heads + pinball loss", "volatility-stratified evaluation", "Hampel outlier flags"]
  TransNet2026_npjCleanAir:
    link: "https://doi.org/10.1038/s44407-026-00052-x"
    method:
      graph_edges: "w_ij(t) = ReLU(ws_i(t) · cos(θ_ij − wd_i(t)) / d_ij)"
      embeddings: "state embedding U (pollutants + position + time); history embedding V (24 h wind + position + time)"
      advection: "U_i ← U_i + h(Σ_j V_{j→i} ⊙ U_j − U_i Σ_j V_{i→j}); directional weights V_{i→j} = ReLU(A4(Z_ij − Z_ji)), Z_ij = A3·ReLU(V_i A1 + V_j A2); softmax per source (Σ_j V_{i→j} = 1, mass conservation)"
      diffusion: "implicit step (I + h·κ_c·L̂ + εI)^-1 U per channel c; κ = hardtanh(θ, 0, 1); L̂ = normalized Laplacian of the wind graph; ε = 1e-6"
      reaction: "f = ReLU(Rθ1(U) + tanh(Rθ2(U)) ⊙ U + Rθ3(U_hist) + Rθ4(Met)); U ← U + h·f"
      splitting: "advection → diffusion → reaction per step; h = 0.01"
      features: "43 features incl. VMD modes of PM2.5, PCA of pollutants, wind (U, V, cos, sin), 8 cyclical time, 16 positional, WRF met (PBL height, T2, Q2, precip)"
      training: "negative IOA loss; Adamax lr 0.01, weight decay 5e-4, batch 32; ReduceLROnPlateau(patience 3, factor 0.2, min 1e-6); grad clip 1.0; early stop patience 5 (δ=1e-4); ≤100 epochs; chronological train 2018–19 / val 2020 / test 2021; 24 h sliding windows; stations with >10% missing excluded"
    results: "170 stations, +1 h MAE 2.60 vs 4.27 µg/m3 (AGATNet, CTM-based); better to ~+48 h; removing the coupled advection–diffusion system raised +1 h MAE by 153.5%; advection-only or diffusion-only ≈ +72–73%"
    limitations: "weak on rare extremes (recall 53.9% at +1 h for ≥76 µg/m3, near 0 beyond +12 h) but very high precision (94%); degrades beyond +48 h"
    we_use: ["ADR transport block (C08.5 step 5)", "wind-projected edge prior with speed", "mass-conserving directional advection", "IOA as metric (+ optional loss term)", "extreme-event evaluation (POD/FAR/CSI) and optional extreme weighting"]
  TopoFlow2026_npjCAS:
    link: "https://doi.org/10.1038/s41612-026-01417-5"
    facts: ["physics in the architecture beats physics as loss/inputs", "wind-guided patch reordering (sort patches along wind direction per sector)", "topography attention bias J[i,j] = −α·ReLU((h_j − h_i)/h0), α learnable init 2.0, h0 = 1000 m, clamp [−10, 0]", "ablation: wind reordering −2.7% val loss, elevation bias −3.1% more", "no log transform to keep extremes", "wind-quality errors propagate into the ordering (limitation)"]
    we_use: ["optional elevation penalty on edge prior (Delhi is flat → expected small effect; keep for generality)", "use Open-Meteo wind for edges (wind quality)", "ablation: log-residual target vs raw target on extremes"]
  Ali2025_EnvPollut:
    link: "https://www.sciencedirect.com/science/article/pii/S0269749125016987"
    facts: ["Delhi daily AQI 2017–2024; VAR + Granger", "AQI mostly explained by its own past (~93% at short horizons)", "temperature the strongest weather driver (~13%); RH < 2%"]
    we_use: ["residual target + strong history encoder", "temperature and forecast temperature as key covariates"]
  ZenderSwiercz2024_AtmosRes:
    link: "https://www.sciencedirect.com/science/article/pii/S0169809524005155"
    facts: ["PMS7003 sensors, 9 sites, Poland winter", "PM vs RH r≈0.5, vs T r≈−0.2", "relationship depends on AQI category"]
    we_use: ["non-linear, regime-dependent model (not linear corrections)", "RH needed for sensor correction and as input", "evaluate per AQI category"]
  Na2026_IEEEAccess:
    link: "https://doi.org/10.1109/ACCESS.2026.3667131"
    we_use: ["teacher must expose per-node temporal features + temporal self-attention maps for the student's L_feat and L_afd (C08.9)", "LUPI idea: teacher sees privileged info (here: neighbours, wind graph, weather forecasts)"]
  MoralesGarcia2025_JBHI:
    link: "https://doi.org/10.1109/JBHI.2024.3508466"
    we_use: ["heterogeneous graph (different node types, close-to edges weighted by distance)", "synthetic data to help small datasets (we use the physics simulator instead)"]
```

### C08.3 Graph construction

```yaml
graph:
  nodes:
    types:
      sensor: "our ESP32 nodes (runtime)"
      virtual: "simulated sensor nodes at CPCB station locations (pretraining only)"
      reference: "CPCB stations with real pollutant data (pretraining; runtime only if live feed enabled)"
    node_type_embedding: "learned vector per type added to node embeddings (Morales-García heterogeneous graph idea)"
  candidate_edges: "all ordered pairs (i,j), i≠j, with d_ij ≤ R_max = 25 km (tune), else k = 8 nearest"
  static_features_per_edge: {d_ij_km: "great-circle distance", theta_ij: "azimuth i→j (degrees)", dh_ij: "elevation_j − elevation_i (m)"}
  dynamic_prior_weight_per_hour:
    formula: "w_ij(τ) = exp(−d_ij/λ) · max(0, cos(θ_ij − φ_i(τ))) · (ws_i(τ)/ws_ref) · elev_ij"
    phi_i: "direction the wind blows TOWARD at node i = (met_wind_dir_from + 180) mod 360"
    ws_ref: "2 m/s (tune); clip ws_i/ws_ref to [0, 3]"
    lambda: "10 km (tune in {5, 10, 20})"
    elev_ij: "exp(−α_e · ReLU(dh_ij / 1000 m)), α_e learnable init 2.0 (TopoFlow form); optional, default ON"
    source: "Wu 2026 Eq. 21 (distance decay × downwind alignment) combined with TransNet Eq. 3 (wind-speed gain)"
    calm: "if ws_i < 0.3 m/s → directional term replaced by 0.2 (weak isotropic mixing) (tune)"
  static_graph: "A_static_ij = exp(−d_ij²/σ²), σ = 10 km — used for diffusion Laplacian and as a fallback when wind is missing"
  missing_nodes: "node with no record at τ keeps its edges, but its features are masked; its outgoing messages are multiplied by its mask (no information from unknown state)"
  wind_source_per_hour: "Open-Meteo actual/analysis for past hours; Open-Meteo forecast for future hours (decoder)"
```

### C08.4 Input tensors

```yaml
tensors:
  N: "number of nodes in the graph for this run"
  L: 24
  H: 6
  X_node: "[N, L, F_n] node-hour features"
  M_node: "[N, L, F_n] masks (1 = measured/valid)"
  X_fut: "[N, H, F_f] future covariates (weather forecast)"
  S_node: "[N, F_s] static node features"
  E_dyn: "[L, N, N] prior edge weights w_ij(τ) (sparse; store as edge_index + edge_attr per τ)"
  E_static: "[N, N] static weights (sparse)"
  V_wind: "[N, L, 4] wind history (u, v, cos φ, sin φ) for the history embedding"
node_hour_features_F_n:   # sensor nodes; virtual nodes have the same; reference nodes fill PM from CPCB and gases masked
  - {name: ln_pm25, from: "pm25_corr (cloud recomputed)", mask: "flag measured"}
  - {name: ln_pm10, from: "pm10_corr", mask: measured}
  - {name: ln_pm1, from: pm1}
  - {name: coarse_ratio}
  - {name: dust}
  - {name: t_node, from: "node HDC1080"}
  - {name: rh_node}
  - {name: p_node}
  - {name: rain_frac}
  - {name: ln_lux}
  - {name: mq7_n}
  - {name: mq135_n}
  - {name: voc_n}
  - {name: nox_n}
  - {name: aqi_now_scaled}
  - {name: t2m_om, from: "Open-Meteo past-hour value"}
  - {name: rh2m_om}
  - {name: sp_om}
  - {name: u10_om}
  - {name: v10_om}
  - {name: precip_om}
  - {name: swrad_om}
  - {name: blh_om, note: "boundary layer height; key Delhi driver"}
  - {name: hod_sin}
  - {name: hod_cos}
  - {name: dow_sin}
  - {name: dow_cos}
  - {name: doy_sin}
  - {name: doy_cos}
  - {name: is_holiday}
  - {name: is_festival_window}
  - {name: pm_state_onehot, width: 4}
node_hour_features_note: "Same channel definitions/transforms as edge feature_schema.yaml for the node-sensor part (reuse /shared/feature_schema.yaml); Open-Meteo channels are cloud-only."
future_covariates_F_f: [t2m, rh2m, sp, u10, v10, precip, swrad, blh, hod_sin, hod_cos, is_holiday]
static_F_s: [lat_norm, lon_norm, elevation_norm, height_agl_norm, sheltering_factor, node_type_onehot]
normalisation: "per-feature z-score with TRAINING statistics (stored with the model); ln-PM features z-scored too; masks not normalised"
input_gap_policy: "gaps ≤ 2 h: linear interpolation with mask=0; longer: value 0 (after z-score) with mask=0. Never interpolate targets."
```

### C08.5 Architecture — TI-STGNN (Transport-Informed Spatio-Temporal GNN)

Design rationale: Wu et al. show wind-aware dynamic graph attention plus a Transformer temporal encoder gives strong multi-step forecasts with good intervals; TransNet shows that an explicit, mass-conserving advection–diffusion–reaction (ADR) block on a wind graph is the key driver of short-horizon (≤48 h) accuracy; Wang et al. show weather forecasts help; TopoFlow shows physics built into the architecture beats physics as a loss term. TI-STGNN combines these: Transformer per node for time (and to give the student attention maps), ADR block for space, forecast weather in the decoder, quantile heads for intervals. A simpler wind-GAT spatial block is kept as variant B for ablation.

```yaml
ti_stgnn:
  hyperparams_default: {D: 64, heads: 4, temporal_layers: 2, adr_steps_K: 2, h: 0.01, dropout: 0.1, decoder_layers: 1}
  step1_embed:
    op: "e[i,τ] = MLP_in([X_node[i,τ] ⊙ M_node[i,τ], M_node[i,τ]]) + NodeTypeEmb(type_i) + MLP_static(S_node[i])"
    shape: "[N, L, D]"
  step2_spatial_per_hour_variantB_only:
    op: "for each τ: wind-biased GAT (Wu Eq.14–16): logits_ij = LeakyReLU(a^T[W e_i ∥ W e_j]) + β·log(w_ij(τ)+1e-3); α = softmax_j; e_i ← e_i + LN(Σ_j α_ij W e_j)"
    note: "Variant A skips this and does space once in step 5 (cheaper). Variant B = Wu-style (GAT every hour). Both MUST be implemented for ablation E-C3."
  step3_temporal_encoder:
    op: "Z_i = TransformerEncoder_prenorm(e[i,·] + sinusoidal_PE)  (shared across nodes; causal mask not needed on history)"
    shape: "Z: [N, L, D]"
    exports: {feat_seq: "Z_i (last layer output) [L, D]", attn: "last-layer self-attention weights [heads, L, L] (average over batch not needed; per node)"}
  step4_embeddings_for_transport:
    state: "U_i = MLP_state(Z_i[L-1] ∥ pos_i ∥ time_t)   [N, D]"
    history: "V_i = MLP_hist(flatten(V_wind[i]) ∥ pos_i ∥ time_t)   [N, D_v]"
    U_hist: "MLP_uhist(flatten(Z_i))  [N, D]"
    Met: "MLP_met(Open-Meteo channels at t for node i)  [N, D]"
  step5_ADR_transport_block (repeat K times, TransNet operator splitting):
    advection:
      scores: "Z_ij = A3·ReLU(V_i A1 + V_j A2) + β·log(w_ij(t)+1e-3)   # learned transport + physical prior (Wu edge bias)"
      direction: "V_{i→j} = ReLU(A4·(Z_ij − Z_ji)); V_{j→i} = ReLU(A4·(Z_ji − Z_ij))   # only one direction non-zero per pair"
      normalise: "softmax over j ∈ N(i) of V_{i→j} (masked by node validity) so Σ_j V_{i→j} = 1  (mass conservation)"
      update: "U_i ← U_i + h·(Σ_j V_{j→i} ⊙ U_j − U_i ⊙ Σ_j V_{i→j})"
    diffusion:
      laplacian: "L̂ = normalized Laplacian of symmetrised (A_static + mean_τ w(τ))"
      coef: "κ_c = hardtanh(θ_c, 0, 1) per channel c"
      update: "U_c ← (I + h·κ_c·L̂ + 1e-6·I)^{-1} U_c   # implicit, solve per channel (torch.linalg.solve; N small); fallback lstsq/pinv"
    reaction:
      f: "ReLU(Rθ1(U) + tanh(Rθ2(U)) ⊙ U + Rθ3(U_hist) + Rθ4(Met))"
      update: "U ← U + h·f"
    output: "S_i = U_i after K steps   [N, D]"
  step6_fusion:
    op: "h_i = GLU(S_i ∥ Z_i[L-1]) = (S_i∥Z_i)W_h ⊙ σ((S_i∥Z_i)W_g)   (Wu Eq.18)   [N, D]"
  step7_decoder:
    queries: "q_{i,k} = PE(k) + MLP_fut(X_fut[i,k]) + h_i,  k = 1..H   # weather forecast enters here (Wang)"
    op: "decoder_layers × (self-attn over k, cross-attn to Z_i, FFN), non-autoregressive (Wu Eq.17)"
    shape: "[N, H, D]"
  step8_heads:
    point: "Linear(D → 2) → ŷ[i,h,(pm25,pm10)]"
    quantile: "Linear(D → 2×3) → q_τ for τ ∈ {0.1, 0.5, 0.9}; enforce monotone: q0.1 = q0.5 − softplus(a), q0.9 = q0.5 + softplus(b)"
  output: {y_hat: "[N, 6, 2]", q: "[N, 6, 2, 3]", feat_seq: "[N, 24, D]", attn: "[N, heads, 24, 24]"}
  params_estimate: "~0.2–0.6 M (tiny graph; CPU inference < 1 s for N ≤ 100)"
  unit_tests:
    - "Σ_j V_{i→j} = 1 for valid i (advection mass conservation)"
    - "κ ∈ [0,1]; diffusion step does not increase max |U| (maximum principle, numeric check)"
    - "masked node contributes zero outgoing flux"
    - "permutation of node order permutes outputs identically (equivariance)"
    - "w_ij = 0 when wind blows away from j"
```

### C08.6 Loss

```yaml
loss:
  total: "L = λ_H · Huber_δ(ŷ, y; mask) + λ_Q · Σ_τ Pinball_τ(q_τ, y; mask) + λ_IOA · (1 − IOA(ŷ, y; mask))"
  defaults: {λ_H: 1.0, λ_Q: 0.5, λ_IOA: 0.0, δ: 1.0}
  mask: "target measured AND node active AND not burn-in (CI02)"
  horizon_weights: "equal (tune: slight upweight h1..h3 because edge uses them)"
  extreme_weighting_optional: "weight = 1 + γ·1[PM_t+h > 120 µg/m3], γ ∈ {0, 1, 3}  — addresses TransNet's weak extreme recall; ablation E-C8"
  node_type_weights: {sensor: 1.0, virtual: 1.0, reference: 0.5}
  notes: ["Huber for robustness (Na teacher, edge consistency)", "pinball for quantiles (Wu)", "IOA term optional (TransNet trained on −IOA)"]
```

### C08.7 Training data and stages

```yaml
training_stages:
  P0_pretrain_virtual_graph:
    graph: "~40 Delhi CPCB station locations as virtual sensor nodes (edge [S10] simulator, 3 units per station → 3 parallel graph samples) + optional reference nodes"
    labels: "true CPCB PM (edge [S10] label rule)"
    period: "2018 → latest full year; chronological split train ≤2023, val 2024, test 2025 (adjust)"
    purpose: "learn transport + meteorology relations at scale (Gu transfer learning)"
  P1_finetune_real_network:
    graph: "our sensor nodes (+ reference nodes if runtime feed enabled; train with reference-node dropout p=0.5 so the model works with or without them)"
    init: "P0 weights"
    schedule: "first 5 epochs freeze temporal encoder, then unfreeze all; lr × 0.3"
    min_data: "start when ≥ 14 days of real data from ≥ 3 nodes; before that, serve the P0 model on the real graph"
    splits: "expanding window: train = all real data up to T_val_start; val = next 7 days; test = last 7 days (never used for selection)"
  node_inclusion_rule: "a node contributes training windows only after burn-in and if its PM target completeness in the window period ≥ 70% (TransNet excluded stations with >10% missing; we mask instead but drop very sparse nodes)"
  windows: "sliding, stride 1 h; a window is a training sample if ≥ 1 node has ≥ 1 measured target"
  optimiser: {name: AdamW, lr: 1e-3, weight_decay: 1e-4, batch: 32, grad_clip: 1.0}
  scheduler: "ReduceLROnPlateau(patience 3, factor 0.2, min 1e-6) (TransNet)"
  early_stopping: "patience 8 on val loss, max 100 epochs"
  augmentation: ["random node dropout 10% (robustness to offline nodes)", "random PM masking of the last k∈{1..6} hours on 10% of nodes (robustness to PM faults; teaches neighbour inference)"]
  hyperparam_search: "Optuna 30 trials on P0 val: D {32,64,128}, heads {2,4}, temporal_layers {1,2}, K {1,2,4}, h {0.01,0.1}, λ {5,10,20} km, lr {1e-3,3e-4}, dropout {0.1,0.2}, variant {A,B}"
  reproducibility: "seed everything; log data snapshot hash, schema hash, calibration versions, code git sha"
```

### C08.8 Inference (hourly)

```yaml
inference_job:
  trigger: "scheduler at HH:08 IST (run_kind=scheduled); HH:20 if late data (run_kind=late); admin manual"
  lock: "Redis SET NX infer:{base_ts}:{run_kind} TTL 15 min"
  steps:
    1_select_nodes: "nodes with status in (active, burn_in); reference nodes if enabled"
    2_build_inputs: "read last 24 h from v_inputs + weather_obs/fcst; build tensors (C08.4); node_inputs_complete[i] = record(base_ts) exists"
    3_load_model: "production teacher from registry (cached; hot-reload on Redis message 'teacher_promoted')"
    4_forward: "torch.no_grad(), CPU; timeout 60 s"
    5_postprocess_pm: "PM_p[t+h] = exp(ln(1+PM_p[t]) + ŷ) − 1; if PM_p[t] not measured → use last measured PM within 6 h and set node_inputs_complete=false; else skip node (no forecast, publish 'unavailable')"
    6_intervals: "quantiles → CQR calibration (C08.8b) → PM lower/upper per h"
    7_aqi: "shared CPCB code on node's 24 h measured buffer + predicted hours (edge [S07] roll); AQI bounds from PM bounds (AQI is monotone in PM)"
    8_neighbour_estimate: "for nodes with pm_state ≠ ok (C08.8c)"
    9_gas_subindex: "if CPCB live enabled: NO2/O3/CO sub-indices from nearest reference station"
    10_store: "insert cloud_forecasts rows + outbox rows in ONE transaction"
    11_publish: "outbox publisher sends MQTT retained per node + Redis 'forecast' channel"
    12_event_check: "server-side disagreement check (C13.2) for nodes with an edge_fc for base_ts"
    13_metrics: "run duration, nodes_complete, publish lag"
  fallback: "if model load/forward fails → persistence forecast (ŷ = 0, intervals from last 14 days' persistence errors), forecast_source='persistence_fallback', alert developer"
  late_rerun: "only nodes whose inputs changed + their graph neighbours get new rows; others unchanged"
cqr_intervals_C08_8b:
  method: "Conformalized quantile regression per node, per horizon, per pollutant"
  residual: "E = max(q0.1 − y, y − q0.9) on the last 14 days of measured targets (rolling)"
  correction: "Q = quantile_{(1−α)(1+1/n)}(E), α = 0.1"
  interval: "[q0.1 − Q, q0.9 + Q] in ln-change space → convert to PM → AQI bounds"
  small_n: "if n < 200 for that node/horizon: use pooled residuals of all nodes, multiply Q by 1.5 and set interval_quality='provisional' (edge suppresses local_event; edge C08)"
  quality_flag: "interval_quality = 'calibrated' when every h1..h3 has n ≥ 200 for the node, else 'provisional'"
  coverage_target: 0.9
  monitoring: "rolling PICP per node; alert if < 0.8"
neighbour_estimate_C08_8c:
  when: "node i PM not measured at t"
  method: "PM̂_i = Σ_j ω_ij · r_ij · PM_j,  ω_ij ∝ w_ij(t)+A_static_ij over neighbours with measured PM, r_ij = median(PM_i/PM_j) over the last 14 days when both measured (default 1.0)"
  output: "neighbour_pm_estimate {pm25, pm10, valid}; stored in neighbour_estimates with flag cloud_estimate (never a target)"
```

### C08.9 Teacher outputs for distillation (teacher_store)

```yaml
teacher_store:
  contract: "edge [S12.2]"
  per_row_key: [teacher_version, node_id, base_ts]
  fields:
    y_teacher: "float[6] = [pm25_h1..h3, pm10_h1..h3]  (edge subset, CI05)"
    y_teacher_full: "float[12] = [pm25_h1..h6, pm10_h1..h6]"
    q_teacher: "float[12][3] quantiles (optional use)"
    feat_seq: "float[24][D] = Z_i (C08.5 step3)"
    attn: "float[heads][24][24] = last temporal layer self-attention for node i"
    oof: "bool"
    fold_id: "int"
  oof_generation:
    method: "k = 5 contiguous time blocks over the teacher training period; train 5 fold models (same hyperparams, fewer epochs allowed); each window is predicted by the model that did not see its block"
    recent_windows: "windows after the teacher training period are predicted by the production teacher (oof=true by construction)"
    cost_control: "if too slow, k = 3"
  virtual_nodes: "teacher_store also covers virtual-node history (P0) so students can pretrain with KD on synthetic data"
  backlog: "backfilled node data (is_backlog) triggers teacher_store backfill for affected windows"
  format: "parquet in MinIO: s3://aqi/teacher_store/{teacher_version}/node_id={id}/month={YYYY-MM}/part-*.parquet; indexed in teacher_store_index"
  leakage_rules: ["teacher training period end < student test period start", "fold models never predict their own block", "inputs at base_ts use only data ≤ base_ts"]
  precondition_check: "per horizon, teacher RMSE on node data < no-KD student RMSE, else students are trained without KD for that release (Shmuel precondition, edge [S12.1])"
```

### C08.10 Interim teacher (until TI-STGNN is ready)

```yaml
interim_teacher:
  model: "LightGBM, one model per (pollutant, horizon) → 12 models"
  inputs: "node features (edge RF feature set) + lagged PM of 5 nearest nodes/reference stations + upwind indicator cos(θ_ij − φ_i) + Open-Meteo forecast at t+h"
  outputs: "same targets; quantiles via LightGBM quantile objective (α 0.1, 0.9)"
  teacher_store: "y_teacher only (no feat_seq/attn) → students use out_KD only"
```

### C08.11 Baselines, ablations and evaluation (cloud model)

```yaml
baselines: [persistence, "diurnal persistence (14-day hourly mean change)", "LightGBM interim teacher", "GCN-GRU with static distance graph", "Wu-style WSDST-GAT (variant B without ADR)", "TransNet-style ADR only (no temporal Transformer)"]
ablations:
  E-C1: "static distance edges vs dynamic wind edges (Gu, Wu)"
  E-C2: "prior weight without wind speed (Wu Eq.21) vs with speed (TransNet Eq.3)"
  E-C3: "variant A (ADR) vs variant B (hourly wind-GAT) vs A+B"
  E-C4: "remove advection / remove diffusion / reaction only (TransNet operator ablation)"
  E-C5: "with vs without future weather covariates (Wang)"
  E-C6: "with vs without P0 pretraining (Gu transfer learning); learning curve vs days of real data"
  E-C7: "with vs without reference nodes at runtime"
  E-C8: "extreme weighting γ; log-residual vs raw-residual target (TopoFlow keeps raw scale)"
  E-C9: "elevation penalty on/off (expected ≈ 0 for Delhi)"
  E-C10: "raw quantiles vs CQR-calibrated intervals"
metrics:
  point: ["MAE", "RMSE", "R²", "IOA (TransNet)", "skill vs persistence"]
  per: ["horizon h1..h6", "pollutant", "node", "season", "AQI category (Zender-Świercz)", "volatility stratum: wind-speed rolling std >P80 vs <P20 (Wu)"]
  intervals: ["PICP", "PINAW (Wu)", "per-node coverage"]
  extremes: ["POD, FAR, CSI for hourly PM2.5 > 120 and > 250 µg/m3 and for AQI band crossings"]
  significance: ["Diebold–Mariano vs persistence and vs best baseline"]
  operational: ["publish lag", "fraction of nodes forecast by HH:10", "fallback rate"]
```

---

## [C09] EDGE STUDENT TRAINING (cloud-side pipelines)

```yaml
students:
  roles: {A: "with PM features", B: "PM features dropped (fallback)"}
  archs: {rf: "edge [S13.2]", cnn: "edge [S13.4] GDC-CNN"}
  kd_variants: {no_KD: "true labels", out_KD: "blend with y_teacher (α grid)", full_KD: "Na L_pred + L_feat + L_afd using feat_seq/attn (CNN only; needs TI-STGNN teacher)"}
  datasets:
    model_A: "windows from v_inputs where targets measured (edge [S11.1] validity rules)"
    model_B:
      synthetic_dropout: "from healthy windows: hide last k∈{1..24} h of PM; features per edge [S08.2]; targets relative to last valid PM (edge [S14])"
      real_outages: "windows from v_outage_windows: real hours_since_pm_lost; inputs = real non-PM sensors; targets = PM at t+h only when PM is measured again (CI03)"
      teacher_targets: "re-express y_teacher relative to last valid PM: y_T' = ln(1+PM̂_T[t+h]) − ln(1+PM_last_valid)"
    synthetic_pretrain: "virtual-node data + teacher_store virtual rows (edge [S10], [S11.3] P0)"
    weighting: "real rows weight 5× while real < 3 months; drop synthetic after (edge [S11.3])"
  per_node_vs_global: "one global student per (role, arch) trained on all nodes + node_id NOT used as a feature (portability); optional per-node fine-tune later"
  export:
    rf: "emlearn arrays → binary blob (edge RF generic evaluator format) + sha256"
    cnn: "Keras → TFLite FP32 (+ INT8 variant for ablation) + arena size measured by a TFLM host build"
  verification:
    - "parity: host C evaluator (same code as firmware, compiled for x86) vs Python on 1,000 vectors, max diff < 1e-4"
    - "schema_hash matches production feature_schema"
    - "size budgets: RF ≤ 400 KB total; CNN ≤ 300 KB, arena ≤ 128 KB"
    - "optional HIL: flash a lab device and run self-test vectors"
  gates:
    - "RMSE on last real week ≤ current production student (per horizon, ≥2 of 3 better, none >5% worse)"
    - "skill vs persistence > 0 for h1..h3"
    - "Model B: RMSE vs hours_since_pm_lost curve not worse than current"
  publish: "register (stage=staging) → manifest for canary cohort → after 48 h canary metrics OK → production manifest for all"
```

---

## [C10] RETRAINING POLICY AND TRIGGERS

```yaml
retraining:
  teacher:
    schedule: "monthly, 1st day 02:00 IST"
    data_trigger: "new measured node-hours since last training ≥ 20% of the last training set, or ≥ 2,000 node-hours"
    drift_trigger: "7-day rolling skill vs persistence < 0.05 on h1..h3 for ≥ 30% of nodes, or PICP < 0.8 on ≥ 30% of nodes"
    topology_trigger: "a new node reaches 14 days of data (so it is learned, not only inferred)"
    min_interval: "7 days between teacher trainings (unless admin forces)"
    after_promotion: "rebuild teacher_store (OOF) → trigger student pipelines"
  students:
    schedule: "weekly, Sunday 03:00 IST (all 4: A_rf, A_cnn, B_rf, B_cnn; CNN may be skipped if not in use)"
    data_trigger: "≥ 500 new measured node-hours since last student training"
    drift_trigger: "edge_fc skill vs persistence < 0 for 3 consecutive days on any node (edge [S19])"
    after_teacher_promotion: true
  conformal_recalibration: "hourly rolling (no retraining)"
  guards: ["never train during HH:05–HH:15 (inference window) on shared CPU; training_worker nice/cgroup-limited", "skip if data completeness of last week < 50%"]
promotion_and_rollback:
  teacher_gate: "on held-out last 14 days: RMSE ≤ production on ≥ 4 of 6 horizons and no horizon > 5% worse; skill > 0 all horizons; calibrated PICP within [0.85, 0.95]"
  teacher_shadow: "new teacher runs in shadow for 72 h (forecasts stored, not published); promote if shadow metrics hold"
  rollback: "one-click admin endpoint → previous production version; inference hot-reloads"
```

---

## [C11] MLOPS WITH ZENML

```yaml
zenml:
  stack:
    orchestrator: "local_docker (dev/single VM); kubernetes (scale)"
    artifact_store: "s3 → MinIO bucket s3://aqi-zenml"
    experiment_tracker: "mlflow"
    model_registry: "ZenML Model Control Plane (model 'aqi_teacher', 'edge_student_A_rf', ...) mirrored into db.models"
    container_registry: "optional (needed for kubernetes)"
  scheduling: "APScheduler triggers pipelines via ZenML Python client (local orchestrators do not support native schedules); if a scheduling-capable orchestrator is used, native ZenML schedules MAY replace this"
  pipelines:
    data_snapshot_pipeline:
      steps: [export_readings_to_parquet, export_weather, export_reference, validate_snapshot (great_expectations-style checks), hash_snapshot]
      output: "dataset artifact (versioned, hash)"
    teacher_training_pipeline:
      params: {stage: "P0|P1", variant: "A|B|A+B", hparams: "dict"}
      steps: [load_snapshot, build_graph_windows, fit_normalizer, train_teacher, evaluate_teacher, calibrate_cqr, compare_with_production, register_teacher(stage=staging)]
    teacher_oof_pipeline:
      steps: [make_time_folds, train_fold_models, predict_oof, predict_recent_with_candidate, write_teacher_store, index_teacher_store]
    teacher_promotion_pipeline:
      steps: [shadow_metrics_check, promote(stage=production), publish_redis('teacher_promoted'), audit]
    student_training_pipeline:
      params: {role: "A|B", arch: "rf|cnn", kd: "no_KD|out_KD|full_KD", alpha_grid: [1.0, 0.5, 0.0]}
      steps: [load_snapshot, load_teacher_store, check_teacher_precondition, build_student_dataset, train_student, evaluate_student, export_student, verify_parity, register_student(stage=staging)]
    manifest_pipeline:
      steps: [select_candidate_students, build_manifest(cohort=canary), store_manifest, mqtt_notify_manifest]
    rollout_pipeline:
      steps: [collect_canary_metrics_48h, gate, build_manifest(cohort=all), mqtt_notify_manifest]
  tracking: "log params, metrics per horizon, plots (skill curves, reliability diagrams, Model B curve), artifacts, git sha, snapshot hash"
  caching: "ZenML step caching ON for deterministic steps (snapshot, windows); OFF for train steps with new seeds"
```

---

## [C12] STREAMING

```yaml
to_nodes_mqtt:
  contract: "edge [S16.7]"
  publisher: "outbox_publisher service drains outbox (FOR UPDATE SKIP LOCKED), publishes QoS1 retained, marks sent_at; retries with backoff; dead-letter after 20 attempts"
  broker_auth: "per-node username/password (token); ACL per node topic; TLS"
  presence: "subscribe aqi/v1/nodes/+/status; update nodes.last_seen/online; stream node_status to frontend"
  commands: "admin API writes command to outbox (non-retained)"
to_frontend_websocket:
  endpoint: "WS /api/v1/stream?channels=forecasts,events,node_status,readings (JWT in Sec-WebSocket-Protocol or query)"
  fanout: "Redis pub/sub channel per type; each API worker subscribes and forwards to its connected clients (horizontal scaling)"
  envelope: '{"type": "forecast|event|node_status|reading|run", "ts": "ISO-8601", "data": {...}}'
  backfill_on_connect: "server sends latest forecast per node + open events"
  heartbeat: "ping every 20 s; client reconnect with exponential backoff; resume with ?since=ISO to fetch missed events via REST"
  sse_alternative: "GET /api/v1/stream/sse MAY be offered with the same envelope"
```

---

## [C13] EVENTS PIPELINE

### C13.1 Ingest (edge-pushed)

```yaml
events_ingest:
  endpoint: "POST /api/v1/events (contract edge [S16.6]); also accepted inside /ingest records"
  upsert: "on (node_id, event_id); status transitions open → update* → closed"
  dedupe: "if an open event of the same type exists for the node, attach as update"
  auto_close: "no update for 3 h → closed (reason timeout)"
  response: '{"accepted": true, "cloud_confirmed": null|true|false}'
  stream: "Redis 'event' → WebSocket; frontend shows alert"
```

### C13.2 Server-side confirmation (cloud matches edge trend vs cloud trend)

```yaml
server_check:
  when: "after each inference run, for nodes with edge_fc(base_ts) and cloud forecast(base_ts) with node_inputs_complete=true"
  rule: "identical to edge [S15] step6: disagree = edge AQI[h] ∉ [cloud.aqi_lower[h], cloud.aqi_upper[h]] for ≥ 2 of h=1..3; local_event if disagree AND gas/PM evidence (z(sraw_nox_max) > 3 OR pm25_max_minute > 2×mean OR mq7 jump > 3σ) using the record's raw block"
  outcomes:
    edge_event_and_server_agrees: "cloud_confirmed = true"
    edge_event_and_server_disagrees: "cloud_confirmed = false (keep event; frontend shows 'unconfirmed')"
    server_detects_edge_silent: "create event source=cloud, type=local_event, severity=low (e.g. edge had stale cloud forecast)"
  neighbour_context: "attach to evidence: neighbours' latest PM and whether neighbours also disagree (widespread → likely not local; single node → local)"
  new_node_rule: "if interval_quality = provisional (C08.8b) → only sensor_check events, no local_event (edge C08)"
frontend_alerts:
  - "event open with severity ≥ medium"
  - "node offline > 2 h (LWT/last_seen)"
  - "cloud forecast crosses into Poor / Very Poor / Severe within 6 h"
  - "sensor_check or pm_sensor_fault open > 6 h (maintenance)"
events_are_not_labels: "never used as training targets; used for evaluation (edge E11) and maintenance"
```

---

## [C14] API CONTRACTS

### C14.1 Device APIs (identical to edge [S16])

```yaml
device_api:
  - {method: POST, path: /api/v1/ingest,   contract: "edge S16.2", auth: node_token}
  - {method: GET,  path: /api/v1/forecast, query: "node_id, base_ts", contract: "edge S16.3", auth: node_token, not_ready: "202"}
  - {method: GET,  path: /api/v1/manifest, query: "node_id", contract: "edge S16.4", auth: node_token, caching: "ETag / If-None-Match → 304"}
  - {method: POST, path: /api/v1/events,   contract: "edge S16.6", auth: node_token}
  - {method: GET,  path: "/api/v1/artifacts/{artifact_id}", auth: node_token, returns: "302 to pre-signed MinIO URL (expires 15 min)"}
forward_compatibility: "cloud MAY add fields to responses; edge MUST ignore unknown fields. Cloud MUST NOT remove or rename fields without a new /api/v2."
```

### C14.2 Forecast payload (as published on MQTT and returned by GET)

```json
{
  "node_id": "delhi-n01",
  "based_on_ts": "2026-10-01T04:30:00Z",
  "issued_at": "2026-10-01T05:39:00Z",
  "teacher_version": "tistgnn-2026-10-03",
  "forecast_source": "teacher",
  "node_inputs_complete": true,
  "horizons": [1, 2, 3, 4, 5, 6],
  "y": [0.03, 0.06, 0.09, 0.11, 0.12, 0.13],
  "y10": [0.01, 0.03, 0.05, 0.06, 0.07, 0.08],
  "pm25": [73.1, 75.4, 77.7, 79.3, 80.1, 80.9],
  "pm10": [97.6, 99.6, 101.6, 102.6, 103.6, 104.6],
  "aqi": [199, 202, 204, 206, 207, 208],
  "aqi_lower": [190, 191, 190, 188, 186, 184],
  "aqi_upper": [208, 214, 219, 224, 229, 233],
  "interval_coverage_target": 0.9,
  "interval_quality": "calibrated",
  "trend": "rising",
  "gas_subindex": {"no2": 62, "o3": 41, "co": 88, "station": "DL-ITO", "station_ts": "2026-10-01T04:30:00Z"},
  "neighbour_pm_estimate": {"pm25": 74.0, "pm10": 101.0, "valid": true},
  "updates": {"manifest_etag": "W/\"m-17\""}
}
```

```yaml
forecast_payload_notes:
  identical_to_edge_S16_3: "all edge fields present with same names/units; 'forecast_source' and 'trend' are additive"
  trend_rule: "same as edge [S07] on ΔAQI[3] (±5)"
```

### C14.3 Frontend APIs

```yaml
frontend_api:
  auth: "OAuth2 password flow → JWT (roles viewer|admin); refresh tokens"
  rest:
    - {method: GET,  path: /api/v1/nodes, returns: "list with location, status, online, last_seen, latest AQI"}
    - {method: GET,  path: "/api/v1/nodes/{node_id}", returns: "metadata + latest reading + latest forecasts (cloud & edge)"}
    - {method: GET,  path: "/api/v1/nodes/{node_id}/readings", query: "from, to, fields, include_flags", returns: "hourly series with flags"}
    - {method: GET,  path: "/api/v1/nodes/{node_id}/forecasts", query: "from, to, source=cloud|edge|both", returns: "forecast history for verification plots"}
    - {method: GET,  path: /api/v1/forecasts/latest, returns: "latest forecast per node (map view)"}
    - {method: GET,  path: /api/v1/events, query: "status, node_id, type, from, to, since"}
    - {method: POST, path: "/api/v1/events/{event_id}/ack", role: viewer}
    - {method: GET,  path: /api/v1/metrics/skill, query: "node_id, window_days, source", returns: "skill vs persistence per horizon, PICP"}
  websocket: "WS /api/v1/stream (C12)"
  pagination: "cursor-based (ts, id)"
  rate_limits: "viewer 60 req/min; WebSocket 1 connection per token per tab"
admin_api:
  - {method: POST, path: /api/v1/admin/nodes, body: "registry entry (edge S16.5)", returns: "node token (shown once)"}
  - {method: POST, path: "/api/v1/admin/nodes/{node_id}/rotate_token"}
  - {method: POST, path: /api/v1/admin/calibrations, body: "calibration.json"}
  - {method: POST, path: /api/v1/admin/retrain, body: '{"pipeline": "teacher|students|manifest", "params": {}}'}
  - {method: POST, path: /api/v1/admin/models/promote, body: '{"model_id": "...", "stage": "production"}'}
  - {method: POST, path: /api/v1/admin/models/rollback, body: '{"kind": "teacher|student", "role": "..."}'}
  - {method: POST, path: "/api/v1/admin/nodes/{node_id}/command", body: '{"cmd": "...", "args": {}}'}
  - {method: GET,  path: /api/v1/admin/runs, returns: "forecast_runs + training_runs"}
```

---

## [C15] SYSTEM DESIGN CONCERNS

```yaml
reliability:
  single_run_jobs: "Redis locks with TTL; forecast_runs unique keys; idempotent writes"
  outbox_pattern: "forecast + outbox rows in one transaction → no stored-but-unpublished forecasts"
  retries: "Open-Meteo: 3 retries, jitter, circuit breaker 10 min; MQTT publish: backoff; DB: pooled connections, statement timeout 10 s"
  degradation_ladder:
    - "weather API down → cached forecast ≤6 h old → masks"
    - "teacher failure → persistence fallback forecasts (forecast_source flagged)"
    - "broker down → nodes poll GET (edge fallback) ; outbox keeps messages"
    - "DB down → API returns 503 for ingest; nodes keep store-and-forward (edge 30 days)"
  late_data: "HH:20 late re-run; backlog never re-issues past forecasts but updates teacher_store"
consistency:
  time: "all services UTC internally; IST only for bucketing/feature computation via /shared time utils; NTP on host"
  schema: "Pydantic models generated from /shared/contracts JSON Schemas; CI contract tests against edge fixtures"
  aqi: "/shared/aqi used by API, inference, training, and edge parity tests"
scalability:
  now: "≤100 nodes on one VM"
  path: "stateless API behind load balancer; Redis fan-out; EMQX cluster; Timescale compression and continuous aggregates (daily means); graph partition by region with halo nodes if > 1,000 nodes"
  expected_load: "≤100 ingest req/h per node batch, trivial; inference < 5 s for 100 nodes on CPU"
security:
  transport: "TLS everywhere (API, MQTT, MinIO pre-signed URLs)"
  device_tokens: "random 256-bit; stored as argon2 hash; rotate via admin; revoke on retirement"
  mqtt_acl: "per-node topic ACL (edge S16.7)"
  frontend: "JWT, role checks, CORS allow-list"
  artifacts: "sha256 in manifest; optional signing of model blobs (ed25519) verified on device"
  input_hardening: "Pydantic strict types, max body 1 MB, rate limit per node (120 req/h)"
  audit: "promotions, rollbacks, token ops, calibration changes → audit_log"
observability:
  metrics:
    - "ingest_requests_total{status}, ingest_latency_seconds, records_late_total, records_backlog_total"
    - "inference_run_seconds, inference_nodes_complete_ratio, forecast_publish_lag_seconds, fallback_forecasts_total"
    - "mqtt_outbox_pending, mqtt_publish_failures_total, nodes_online"
    - "events_open{type,severity}, events_confirmed_ratio"
    - "model_skill{node,horizon,source}, picp{node}, training_run_seconds{pipeline}"
  dashboards: ["network map & AQI", "inference health", "model quality (skill, PICP, edge vs cloud)", "data completeness & flags", "training runs"]
  alerts_to_dev: ["no successful inference for base_ts by HH:15", "fallback rate > 10%", "outbox pending > 100", "node completeness < 80% 24 h", "PICP < 0.8", "disk > 80%"]
  logging: "JSON logs with run_id, base_ts, node_id; retention 30 days"
ci_cd:
  - "lint + type check + unit tests + contract tests + AQI vectors + IST fixture"
  - "model tests (C08.5 unit_tests) on tiny synthetic graph"
  - "docker images per service; alembic migrations run on deploy"
  - "staging environment fed by the node simulator (C20)"
```

---

## [C16] CONFIGURATION (single YAML, env-overridable)

```yaml
config:
  timezone_bucket: "Asia/Kolkata"
  ingest_deadline_min: 8
  late_rerun_min: 20
  horizons: 6
  lookback_h: 24
  graph: {r_max_km: 25, k_nearest: 8, lambda_km: 10, ws_ref: 2.0, calm_ws: 0.3, calm_weight: 0.2, elevation_penalty: true}
  model: {variant: "A", D: 64, heads: 4, temporal_layers: 2, adr_K: 2, h: 0.01}
  intervals: {alpha: 0.1, window_days: 14, min_n: 200, new_node_multiplier: 1.5}
  events: {horizons_outside_min: 2, nox_z: 3.0, pm_spike_ratio: 2.0, mq7_sigma: 3.0, auto_close_h: 3}
  retrain: {teacher_cron: "0 2 1 * *", students_cron: "0 3 * * 0", teacher_data_growth: 0.2, student_min_new_hours: 500}
  features: {open_meteo_vars: [...], cpcb_runtime_enabled: false, firms_enabled: false}
```

---

## [C17] TEST PLAN

```yaml
tests:
  unit: ["ingest validation & flags mapping", "IST bucketing fixture", "AQI vectors", "edge prior weight formula (downwind, calm, elevation)", "ADR invariants (C08.5 unit_tests)", "CQR coverage on synthetic data", "neighbour estimate ratio logic", "outbox publisher idempotency"]
  contract: ["validate edge fixtures (ingest, events) and cloud payloads (forecast, manifest) against /shared/contracts", "MQTT payload byte-equal to GET payload"]
  integration:
    node_simulator: "tools/node_sim replays synthetic/real logs as N virtual ESP32s: HTTPS ingest, MQTT subscribe, events; configurable faults (late, offline, PM fault, backlog)"
    scenarios: ["normal hour", "2 nodes late → late re-run", "Open-Meteo down", "teacher crash → persistence fallback", "broker restart → retained forecast delivered", "PM fault → Model B + neighbour estimate + event", "backlog after 24 h outage → teacher_store backfill"]
  ml:
    - "overfit a tiny graph (sanity)"
    - "leakage test: shuffle future weather → performance must not collapse unless future covariates are truly used; assert window builder never reads ts > base_ts"
    - "OOF test: fold model never predicts its training block"
    - "student parity tests (C09)"
  load: "1,000 simulated nodes ingest within 2 min; WebSocket 200 clients"
  chaos: "kill inference_worker mid-run → lock expiry → re-run succeeds; DB restart"
```

---

## [C18] REPOSITORY LAYOUT

```
/shared
  aqi/                      # CPCB AQI (Python) + test vectors
  contracts/                # JSON Schemas: ingest, events, forecast, manifest, registry, ws envelope
  feature_schema.yaml
  time_utils/               # IST bucketing
  tests/aqi_vectors.json, tests/ist_bucket_fixture.json
/edge                       # see edge doc [S23]
/cloud
  api/                      # FastAPI app: routers device/, frontend/, admin/, ws/; deps; auth
  core/                     # config, db session, redis, mqtt client, logging, metrics
  db/                       # SQLAlchemy models, alembic migrations, views
  ingest/                   # validation, flag mapping, cloud recompute, Hampel job
  weather/                  # Open-Meteo client, alignment, cache
  inference/                # worker, input builder, post-processing, CQR, neighbour estimate, fallback
  events/                   # server check, lifecycle
  streaming/                # outbox publisher, ws fan-out
  ml/
    teacher/                # TI-STGNN (variants A/B), dataset (graph windows), losses, train loop
    interim_teacher/        # LightGBM
    students/               # wrappers calling /edge/training (rf, cnn, model_b builders)
    export/                 # emlearn blob writer, tflite export, host parity runner
    eval/                   # metrics, DM test, volatility strata, reliability plots
  pipelines/                # ZenML pipelines + steps (C11)
  scheduler/                # APScheduler jobs
  infra/                    # docker-compose.yml, mosquitto.conf + ACL, grafana dashboards, prometheus.yml
  tools/node_sim/           # virtual node simulator
  tests/
```

---

## [C19] MILESTONES AND ACCEPTANCE

```yaml
milestones:
  CM1_infra: {done_when: ["compose stack up", "migrations applied", "health endpoints green", "MinIO/MLflow/ZenML reachable"]}
  CM2_ingest: {done_when: ["edge fixtures accepted", "flags stored", "idempotent upserts", "late/backlog marking", "node_sim 10 nodes × 7 days ingested"]}
  CM3_weather: {done_when: ["Open-Meteo fetch + IST alignment tests", "historical backfill for training period"]}
  CM4_streaming: {done_when: ["outbox → MQTT retained per node", "LWT presence", "WebSocket fan-out with backfill", "GET forecast fallback"]}
  CM5_interim_teacher: {done_when: ["LightGBM teacher trained on P0 data", "hourly inference end-to-end publishes forecasts by HH:10", "persistence fallback path tested"]}
  CM6_tistgnn: {done_when: ["C08.5 unit tests pass", "P0 pretraining beats interim teacher on P0 test", "exports feat_seq/attn", "CQR PICP within [0.85,0.95] on val"]}
  CM7_teacher_store: {done_when: ["OOF generation", "parquet + index", "leakage tests pass"]}
  CM8_students: {done_when: ["4 student pipelines run", "parity tests pass", "manifest + MQTT notify", "canary rollout on a lab node"]}
  CM9_events: {done_when: ["edge events stored/streamed", "server-side confirmation", "frontend alert messages", "auto-close"]}
  CM10_retraining: {done_when: ["schedules + data/drift triggers", "shadow + promotion gates", "rollback endpoint"]}
  CM11_observability_security: {done_when: ["dashboards", "alerts", "TLS + ACL + token rotation", "backup/restore drill"]}
  CM12_experiments: {done_when: ["E-C1..E-C10 tables/plots generated from scripts", "edge E4 skill-retained computed jointly"]}
```

---

## [C20] ASSUMPTIONS AND OPEN ITEMS

```yaml
assumptions:
  - "Gu 2024 and Wang 2024 details are from abstracts; confirm formulas and numbers in the full texts before citing specifics."
  - "Open-Meteo variable names, historical-forecast availability for Delhi, and +05:30 handling must be verified."
  - "CPCB live feed is optional; the system is fully functional with sensor nodes + Open-Meteo only."
  - "Small real network (5–10 nodes): TI-STGNN depends on P0 pretraining; expect gains over LightGBM mainly after fine-tuning."
  - "ADR step size h = 0.01 follows TransNet; with our embedding scales, h ∈ {0.01, 0.1} is searched."
open_items:
  - "Whether reference nodes are used at runtime (depends on a reliable live CPCB feed)."
  - "Frontend technology and hosting."
  - "Kubernetes vs single VM for final deployment."
```

---

## [C21] REFERENCES (journal articles) AND WHAT EACH SUPPORTS

```yaml
references:
  - {key: Gu2024, cite: "Atmospheric Environment, 2024 (ScienceDirect pii S1352231024006873)", supports: "dynamic wind-driven graph edges; transfer learning"}
  - {key: Wang2024, cite: "Science of the Total Environment, 2024 (pii S0048969724043778)", supports: "hourly directed wind graph + GAT + seq2seq; weather forecasts as inputs"}
  - {key: Wu2026, cite: "W. Wu, X. Mo, H. Li, ISPRS Int. J. Geo-Inf. 15:101, 2026, doi:10.3390/ijgi15030101", supports: "edge-weight prior, edge-biased GAT, Transformer temporal encoder, GLU fusion, quantile heads, PICP/PINAW, volatility strata, Hampel filter"}
  - {key: Dimri2026, cite: "R. Dimri, Y. Choi, D. Singh, J. Park, N. Khorshidian, npj Clean Air 2:12, 2026, doi:10.1038/s44407-026-00052-x", supports: "ADR operator splitting on a wind graph, mass-conserving advection, implicit diffusion, reaction MLP, IOA metric, extreme-event limits"}
  - {key: Kheder2026, cite: "A. Kheder et al., npj Climate and Atmospheric Science 9:143, 2026, doi:10.1038/s41612-026-01417-5", supports: "physics in architecture; elevation attention penalty; wind-quality caveat; raw-scale extremes"}
  - {key: Ali2025, cite: "D. Ali et al., Environmental Pollution 388, 2026, doi:10.1016/j.envpol.2025.127324", supports: "Delhi AQI autoregressive dominance; temperature strongest weather driver"}
  - {key: ZenderSwiercz2024, cite: "E. Zender-Świercz et al., Atmospheric Research 312, 2024, doi:10.1016/j.atmosres.2024.107733", supports: "non-linear, AQI-category-dependent T/RH effects; PMS7003 humidity sensitivity"}
  - {key: Na2026, cite: "S. Na, Y. Cho, Y. Baek, IEEE Access 14, 2026, doi:10.1109/ACCESS.2026.3667131", supports: "teacher feature/attention export for heterogeneous KD"}
  - {key: Shmuel2026, cite: "A. Shmuel, T. Lazebnik, O. Glickman, IEEE Access 14, 2026, doi:10.1109/ACCESS.2026.3657793", supports: "output-level KD precondition (teacher must beat student)"}
  - {key: MoralesGarcia2025, cite: "J. Morales-García et al., IEEE JBHI 29(12), 2025, doi:10.1109/JBHI.2024.3508466", supports: "heterogeneous node types; synthetic data for small datasets"}
```
