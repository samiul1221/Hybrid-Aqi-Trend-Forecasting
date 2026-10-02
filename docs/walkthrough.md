# Walkthrough — reading the code (CR v2 [R03])

Read in this order. Each file is short and reads top to bottom.

## A. Data: from CPCB/ERA5 to sensor features (shared by teacher and student)

1. `ml/src/aqi_ml/sim/virtual.py` + `sim/forward.py` — turns true CPCB PM/gases and ERA5 weather into what a
   virtual ESP32 unit would *read* (SPS30, HDC1080, BMP280, anemometer, BH1750, MQ-7, MQ-135, SGP41), with
   per-unit errors, drift and faults. A sensor model, not a learned model.
2. `sim/device.py` — the on-device hourly pipeline (humidity correction, AQI, PM state), same code as the node.
3. `features/builder.py` — the 23 edge channels of `shared/schema/feature_schema.yaml`; targets from true PM.
4. `features/dataset.py` — per-node `hourly.parquet`; the student slices 24 h windows from it.
5. `data/graph_windows.py` — the teacher's view: the **same** `hourly.parquet` of unit u at all 30 stations,
   stacked into a grid [T, N, 23] (+ Open-Meteo weather). `window(g, t, idx)` cuts one graph sample.
   Guards: G1 (teacher input == student input, checked at build), G2 (no truth/target column as input),
   G3 (no future in the history). See `docs/research_notes/teacher_input_audit.md`.

## B. Teacher (TI-STGNN), `ml/src/aqi_ml/models/teacher/`

1. `config.py` — one YAML per variant (`ml/configs/teacher/T0..T3.yaml`); booleans use_graph/use_wind/use_adr.
2. `graph.py` — distances/bearings, TransNet wind prior (direction corrected to "toward", ADR 0013),
   1/d prior for T1, diffusion Laplacian basis. Convention: w[i, j] = source i → target j.
3. `temporal.py` — embedding of [value·mask, mask] + static; small pre-norm Transformer per station that
   also returns its attention (needed later for the student's L_AFD).
4. `gat.py` — one graph-attention layer: receiver i attends to senders j, biased by β·log(prior of j→i).
5. `adr.py` — advection (mass-conserving, reuses GAT logits), implicit diffusion, reaction; K = 2 steps.
6. `decoder.py` — GLU fusion, 6 horizon queries with future weather, one decoder layer, point + q10/q90 heads.
7. `model.py` — wires 3→4→5→6 and returns the predictions plus every state the student will distil.
8. `train.py` — data batches (z-score, station dropout), masked Huber + pinball, AdamW loop, early stopping,
   resumable checkpoints, metrics.json.

## C. One sample through the teacher (real shapes; batch B = 16, N = 25 training stations, D = 64)

| step | tensor | shape |
|---|---|---|
| `window()` for one base hour t | X (23 sensor + 7 weather), M | [25, 24, 30] |
| | S static, X_fut, wind_t, y, y_mask | [25, 6], [25, 6, 9], [25, 2], [25, 12], [25, 12] |
| `Data.batch` stacks 16 samples, z-scores | X, M | [16, 25, 24, 30] |
| `Embedding` | e | [16, 25, 24, 64] |
| `Temporal` (stations share weights: reshaped to [400, 24, 64]) | feat_seq / attn | [16, 25, 24, 64] / [16, 25, 4, 24, 24] |
| last hour | h_t | [16, 25, 64] |
| `graph.wind_prior` (T2/T3) | w (source → target) | [16, 25, 25] |
| `WindGAT` | h_spatial, logits | [16, 25, 64], [16, 25, 25] |
| `ADR` (T3): P = softmax over receivers, eigh of L̂ | h_adr | [16, 25, 64] |
| `Decoder`: GLU(h_adr ‖ h_t) + queries | q | [16, 25, 6, 64] |
| heads → pollutant-major | y, q10, q90 | [16, 25, 12] |

Loss = masked Huber(y, targets) + 0.5 · pinball(q10, q90). Targets are ln-changes of TRUE CPCB PM
(h1..h6); true PM never appears in X.

## D. Student

GDC-CNN (`models/edge/gdc_cnn/`) is unchanged. The KD training wrapper (S0–S4, CR [R08]) comes in N6.
