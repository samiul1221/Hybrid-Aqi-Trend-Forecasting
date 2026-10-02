# CHANGE REQUEST v2 — Migrate to the "network teacher → isolated edge student" architecture

<doc_meta>
audience: "Claude Code (local CLI sessions AND Claude Code cloud/web sessions) working in this monorepo"
status: "ACTIVE. Supersedes all ML-architecture parts of the older specs (see R01)."
place_in_repo: "docs/specs/CHANGE_REQUEST_v2.md"
companion: "docs/specs/research_spec_v2.md = the owner's pasted research specification (sections 1–42). It is the scientific source of truth."
format: "IDs [R00]..[R14]. YAML = machine-readable. MUST/SHOULD/MAY as usual. (proposal) = recommended default that needs owner approval in step 0."
read_order: "R00 → R01 → R03 → R04 (step 0, do this first) → rest"
</doc_meta>

---

## [R00] WHY THIS DOCUMENT EXISTS

```yaml
summary:
  old_direction: "temporal teacher → temporal KD → small CNN; cloud teacher was a separate later concern"
  new_direction: "multi-station cloud teacher (Transformer → wind-aware GAT → ADR → decoder) whose internal network-level knowledge is distilled OFFLINE into the SAME isolated single-station GDC-CNN student that runs on the ESP32-S3"
  research_question: "How much of a multi-station teacher's wind-driven transport and physics-informed knowledge survives distillation into a local student that has no neighbour data, no graph and no teacher at inference?"
  consequence_for_repo:
    - "The cloud teacher (TI-STGNN) moves from 'phase 2' to the CENTRE of phase 1 ML work."
    - "The student, data pipeline, simulator, features, AQI code and export path mostly stay."
    - "KD grows from 2 levels (output, temporal) to 5 student variants S0–S4 and 4 teacher variants T0–T3."
    - "Teacher training runs on Kaggle GPU notebooks (owner's laptop has no GPU)."
    - "Code must stay SIMPLE and READABLE: the owner must be able to read and understand every file."
```

---

## [R01] SOURCE-OF-TRUTH HIERARCHY

```yaml
priority_high_to_low:
  1: "docs/specs/research_spec_v2.md (scientific design, novelty limits, experiment matrix)"
  2: "docs/specs/CHANGE_REQUEST_v2.md (this file: how to implement it in this repo, simply)"
  3: "docs/specs/edge_implementation_plan.md — still authoritative for: hardware, firmware, measurement pipeline, AQI, quality flags, feature schema [S08], targets [S07], Model B rules [S14], contracts [S16]/[S17], OTA [S18]"
  4: "docs/specs/cloud_implementation_plan.md — still authoritative for backend/streaming/DB/API ([C02]–[C07], [C12]–[C16]); its model section [C08] is SUPERSEDED where it conflicts with research_spec_v2 (ordering, graph formula, exported states)"
superseded_items:
  - "cloud [C08.3] Wu-style exp(-d/λ)·cos·ws/ws_ref·elev prior → replaced by the TransNet-style prior in [R07.3]"
  - "cloud [C08.5] variants A/B ordering → replaced by fixed order Temporal → GAT → ADR → Decoder"
  - "edge [S12.2]/cloud [C08.9] teacher_store fields → extended by teacher_store v2 [R07.7]"
  - "edge [S13.0] KD variants (no_KD/out_KD/full_KD) → replaced by S0–S4 [R08]"
  - "interim LightGBM teacher as THE teacher → demoted to a baseline [R05]"
action: "Add a short banner at the top of the superseded sections in both old plans: '> SUPERSEDED by docs/specs/research_spec_v2.md + CHANGE_REQUEST_v2.md [Rxx]'. Do NOT rewrite the old plans."
```

---

## [R02] WHAT CHANGES, AT A GLANCE

| Area | Before | After |
|---|---|---|
| Teacher | Interim LightGBM (P8) now, TI-STGNN later | **TI-STGNN now** (PyTorch, Kaggle GPU); LightGBM kept only as a baseline |
| Teacher order | Variant A (Transformer→ADR) / B (GAT per hour→Transformer) | **Fixed: Embedding → Temporal Transformer → wind-aware GAT → ADR → Decoder** |
| Graph prior | Wu-style exp-distance | **TransNet-style** `ReLU(ws·cos(θ−φ))/d` with safeguards |
| Teacher exports | y_teacher, feat_seq, attn | + **h_spatial** (GAT output) and **h_adr** (ADR output) per node |
| Student | GDC-CNN A/B (Keras) | **Unchanged architecture**; training wrapper adds projection heads (train-only) |
| KD | no_KD / out_KD / full_KD | **S0–S4** (output, temporal, spatial, physics) + teacher ablation **T0–T3** |
| Training compute | Laptop CPU | **Kaggle GPU notebooks** via Kaggle MCP (CLI fallback) |
| MLOps | ZenML pipelines next (P9) | **Deferred**: plain scripts first; thin ZenML wrappers after results |
| Firmware | P10 next | **After** the student variant is chosen (student shape does not change) |

---

## [R03] SIMPLICITY RULES (non-negotiable)

```yaml
code_style:
  goal: "Implement EVERY idea in research_spec_v2, with the simplest readable code. Simplify code, never the architecture."
  rules:
    - "One concept per file; target < 250 lines per file. Files read top-to-bottom like a tutorial."
    - "Every nn.Module/Keras layer: a docstring saying what it does, which paper it comes from ([TransNet Eq.7] etc.), and input/output shapes."
    - "Shape comments on every tensor line that changes shape, e.g. `h = h[:, -1]  # [B, N, D]`; plus `assert` on shapes at block boundaries."
    - "Dense tensors only. N ≈ 33 stations → use dense [B, N, N] matrices; NO PyTorch Geometric, no sparse ops."
    - "No registries, no factory patterns, no plugin systems, no Hydra, no Lightning. Plain PyTorch training loop (~100 lines) and a small dataclass config loaded from YAML."
    - "Variant switches are plain booleans in the config (use_graph, use_wind, use_adr), not class hierarchies."
    - "Prefer functions over classes for data code; prefer pandas/numpy that a student can follow."
    - "Comments explain WHY (physics/paper reason), not WHAT the syntax does."
    - "No silent defaults: every hyperparameter lives in one YAML file per experiment."
  explainability_deliverable: "docs/walkthrough.md: a guided reading order of the code with 3–5 lines per file and a worked example of one sample flowing through teacher and student with real shapes."
```

---

## [R04] STEP 0 — ARCHITECTURE REVIEW BEFORE ANY CODE (research_spec_v2 §39)

```yaml
step0:
  do_first:
    - "Read research_spec_v2.md fully, this file, edge plan [S07][S08][S12]–[S14], cloud plan [C08], docs/PROGRESS.md, docs/HANDOFF.md, all ADRs."
    - "Read the PDFs in docs/papers/ that research_spec_v2 cites (TransNet, Na 2026, Wang 2024, WSDST-GAT, KDAQnet, REDNet, ST-BernT, AIoT virtual sensors, TSMNet). Verify every bibliographic detail you write (titles, venues, DOIs) from the PDFs."
  produce: "docs/specs/architecture_review_v2.md containing items 1–8 of research_spec_v2 §39 (text diagram, full tensor-shape table, fixed-vs-hypothesis table, literature→component table, novelty-vs-counterexamples table, inconsistency list, verification of the 10 items, minimal implementable architecture)"
  then: "STOP. Wait for owner approval. Record approved decisions as ADRs."
  seed_inconsistencies_already_found: "R04.1 — confirm or correct each; do not silently resolve."
```

### R04.1 Inconsistencies found while writing this file (verify each)

```yaml
inconsistencies:
  I1_cloud_feature_count:
    finding: "cloud [C08.4] lists 32 named node-hour features; pm_state_onehot has width 4 → 35 numeric columns. '32 vs 35' is names vs columns."
    also: "cloud list has NO node wind (uses Open-Meteo u10/v10); student list HAS node wind_u/wind_v."
    proposal: "State both numbers explicitly (32 features / 35 columns) in the schema; keep node wind out of the teacher graph prior (Open-Meteo wind), but MAY include node wind as a teacher node feature."
  I2_student_channels:
    finding: "Model A = 23 channels (edge [S08.1] i0..i22). Model B = 23 − 6 PM-derived (ln_pm25, ln_pm10, ln_pm1, coarse_ratio, aqi_now, pm_mask) = 17 + 5 extras [S08.2] = 22 channels (matches PROGRESS open question 6)."
    proposal: "Confirm 22 channels with the 5 extras broadcast as constant channels."
  I3_static_features:
    finding: "research_spec_v2 says static [N, 6]; cloud [C08.4] static = lat, lon, elevation, height_agl, sheltering_factor + node_type one-hot(3) = 8 columns."
    proposal: "With only CPCB stations (one node type) and no real nodes yet: static = [lat, lon, elevation, height_agl, sheltering_factor, is_reference] = 6. Decide and document."
  I4_graph_direction_convention:
    finding: "TransNet Eq.3 uses meteorological wind direction with a 270−atan2 conversion; meteorological WD is where wind comes FROM."
    proposal: "Use φ_i = direction the wind blows TOWARD = (WD_from + 180) mod 360, θ_ij = bearing i→j, so a DOWNWIND j gets positive weight. Unit test: wind blowing east → east neighbour gets weight > 0, west neighbour 0."
  I5_ordering_and_what_GAT_sees:
    finding: "Spec fixes Temporal → GAT → ADR but does not say whether GAT runs on every hour or only on the last state."
    proposal: "GAT and ADR run ONCE at base_ts on h_t = feat_seq[:, -1] (the temporal summary per node). Simplest, matches 'export temporal before spatial mixing'."
  I6_advection_weights_source:
    finding: "TransNet learns separate directional advection weights (A1..A4) with per-source softmax (mass conservation). Spec also has a GAT."
    proposal: "Reuse the GAT's logit matrix for advection but normalise it PER SOURCE (softmax over receivers, self-loop excluded) → mass-conserving, wind-asymmetric, no extra network. Document as a simplification of TransNet's advection operator; ablation option to use TransNet's separate A1..A4 later."
  I7_S_variant_definitions:
    finding: "§22 lists A–D as cumulative (B = output + temporal, C = + spatial, D = + physics); §27 lists S3 'output + spatial' and S4 'output + spatial + physics' (no temporal)."
    proposal: "Make S0–S4 cumulative: S0 none, S1 output, S2 S1+temporal(L_feat+L_AFD), S3 S2+spatial, S4 S3+physics. Optionally add S3' = output+spatial only as an extra ablation."
  I8_loss_weighting_with_more_terms:
    finding: "Na's formula has two KD terms ((1−λp)/2)·r(e)·(L_feat+L_AFD). S3/S4 add terms."
    proposal: "L = λp·L_pred + ((1−λp)/n_kd)·r(e)·Σ active KD terms (n_kd = number of active KD terms incl. L_out). λp = 0.4, E_warmup = 10."
  I9_representation_scale:
    finding: "Huber between student projections and raw teacher states depends on arbitrary scale."
    proposal: "z-score each exported teacher state per dimension with TRAIN statistics stored in teacher_store metadata, before computing KD losses."
  I10_horizons:
    finding: "Spec §27 evaluates 1h, 2h, 3h, 6h, but the student predicts h1..h3 only (6 outputs = 3 h × 2 pollutants)."
    proposal: "Student evaluated at h1–h3; teacher at h1–h6; 'skill retained' computed on h1–h3."
  I11_teacher_input_domain:
    finding: "No real nodes exist yet. Teacher trains on the CPCB station network (OpenAQ, 33 stations). Student inputs are ESP32-sensor-proxy features (simulator virtual units at those stations)."
    proposal: "Teacher node features = the SAME sensor-proxy features as the student (virtual unit u at every station) + Open-Meteo past weather. Privileged info = neighbours + wind graph + ADR + future weather, NOT better sensors. Build 3 graph 'realities' (unit u = 0,1,2 at all stations); teacher_store keyed by (station, unit, base_ts)."
  I12_ADR_step_size:
    finding: "TransNet uses h = 0.01 with its own state scale; our states are LayerNorm-scaled embeddings."
    proposal: "Start h = 0.1, K = 2 (config), test h ∈ {0.01, 0.1}; record in ADR."
  I13_novelty_citations:
    finding: "LightST (AAAI 2025) is a conference paper. The owner prefers journal articles for main citations."
    proposal: "Cite LightST only as the strongest counterexample in related work; keep all architectural foundations journal-based."
  I14_modelB_KD_scope:
    finding: "Spec does not say whether Model B gets spatial/physics KD."
    proposal: "Model B: S0 and S1 first (teacher outputs re-expressed relative to last valid PM, edge [S14]); S2–S4 for Model B only if time permits."
  I15_OOF_cost:
    finding: "4 teacher variants × 5 folds + 4 final teachers = 24 teacher trainings."
    proposal: "Keep 5 folds for T3 (main); allow 3 folds for T0–T2 if Kaggle GPU quota is tight. Measure one fold's runtime first."
```

---

## [R05] DISPOSITION OF EXISTING WORK

```yaml
keep_unchanged:
  - "P0–P2: shared/ (AQI, IST bucketing, feature_schema loader, contracts, codegen, fixtures)"
  - "P3: OpenAQ archive downloader (33 stations), Open-Meteo ERA5 downloader, base table, data card"
  - "P4: simulator + virtual nodes (90 units) — now ALSO the source of teacher node features (I11)"
  - "P6 export path: emlearn blob + host C evaluator + parity tests"
  - "P7: GDC-CNN student model code, TFLite FP32 export, static-batch export graph fix (ADR 0010)"
  - "Arduino firmware decision (ADR 0002)"
adapt:
  - {item: "P5 feature builder", change: "add a GRAPH window builder: for each base_ts, aligned [N, 24, F] tensors for all stations + masks + Open-Meteo past + future weather [N, 6, 11] + wind at t; deterministic; leakage assertion max(input_ts) ≤ base_ts except declared future covariates"}
  - {item: "P7 GDC-CNN training", change: "split into (a) export model = unchanged inference graph, (b) training wrapper that also returns F_S [12, D_S] and owns train-only projection heads ψ, P_spatial, P_physics. Export MUST contain only (a)."}
  - {item: "P8 teacher_store", change: "schema v2 [R07.7]; keep reader/writer; add teacher_variant and new state fields"}
  - {item: "P8 full_KD losses", change: "keep L_feat/L_AFD code; add L_spatial, L_physics, cumulative S0–S4 [R08]"}
  - {item: "Open-Meteo", change: "add historical-forecast download for future-weather covariates (fallback: ERA5 actuals with an explicit 'oracle weather' flag, reported as a limitation)"}
demote_to_baseline_and_freeze:
  - "LightGBM interim teacher + its OOF store → move under ml/src/aqi_ml/baselines/lgbm_teacher/, do not extend"
  - "RF students A/B (no_KD, out_KD) → baselines only (RF cannot take feature KD); keep results in the comparison table"
delete:
  - "Any CNN out_KD/full_KD experiment configs and half-finished runs from P8 that assume the LightGBM teacher (keep their results in an ADR/results note, delete the configs/scripts)"
  - "ZenML pipeline stubs, if any exist that wire the OLD P8 flow (keep ml/pipelines/README.md: 'deferred, see R10')"
  - "Unused code paths for cloud [C08.5] variant A/B switching if any were started"
rule: "Before deleting, list every file to delete in the step-0 report; delete only after owner approval; one commit titled 'chore: remove v1-teacher artifacts (CR v2 R05)'."
```

---

## [R06] REPO CHANGES (minimal)

```
ml/src/aqi_ml/
  data/graph_windows.py         # NEW  [R07.2] aligned station windows → npz/parquet shards
  models/teacher/               # NEW  TI-STGNN, one block per file
    config.py                   #   dataclass + YAML loader (use_graph, use_wind, use_adr, D, heads, K, h, ...)
    graph.py                    #   distance/bearing matrices, wind prior, static prior  [R07.3]
    temporal.py                 #   embedding + tiny pre-norm Transformer that returns attention  [R07.4]
    gat.py                      #   one wind-biased GAT layer  [R07.5]
    adr.py                      #   advection, diffusion (eigendecomposition), reaction  [R07.6]
    decoder.py                  #   horizon queries + future weather + GLU fusion + heads  [R07.6b]
    model.py                    #   TISTGNN = wiring of the blocks; returns predictions + exports
    train.py                    #   plain loop: masked Huber (+ optional pinball), early stopping, checkpoint
    oof.py                      #   chronological folds, OOF prediction, teacher_store v2 writer  [R07.7]
  baselines/lgbm_teacher/       # MOVED (frozen)
  models/edge/gdc_cnn/          # KEEP; add train_wrapper.py (projection heads, returns F_S)
  distill/                      # KEEP; add losses_spatial.py, variants.py (S0–S4)  [R08]
ml/kaggle/                      # NEW
  teacher_train/notebook.ipynb + kernel-metadata.json
  student_kd/notebook.ipynb + kernel-metadata.json
  README.md                     # how to push/run/pull with Kaggle MCP or CLI  [R09]
ml/configs/teacher/{T0,T1,T2,T3}.yaml, ml/configs/student/{S0..S4}_{A,B}.yaml
docs/specs/research_spec_v2.md, CHANGE_REQUEST_v2.md, architecture_review_v2.md
docs/walkthrough.md             # NEW  [R03]
```

---

## [R07] TEACHER (TI-STGNN) — SIMPLE IMPLEMENTATION SPEC

### R07.1 Data split (verify against data card)

```yaml
split:
  train: "2018-01-01 .. 2021-12-31 (IST hours)"
  val: "2022-01-01 .. 2022-12-31"
  test: "2025 .. latest (base table gap 2023–2024 is acceptable; document it)"
  rule: "chronological only; test never used for teacher or student training or model selection"
  stations: "all OpenAQ Delhi stations in the base table with ≥ 70% PM target completeness in train; 5 stations held out as 'unseen stations' only if N stays ≥ 25"
```

### R07.2 Graph window sample (one sample = one base_ts, all stations)

```yaml
sample:
  X_node: "[N, 24, F]   F = final teacher schema (I1); sensor-proxy features of virtual unit u + Open-Meteo past weather + time"
  M_node: "[N, 24, F]   1 = measured/valid"
  S_node: "[N, F_s]     (I3)"
  X_fut:  "[N, 6, 11]   Open-Meteo forecast for t+1..t+6 (t2m, rh2m, sp, u10, v10, precip, swrad, blh, hod_sin, hod_cos, is_holiday)"
  wind_t: "[N, 2]       ws (m/s), wd_from (deg) at base_ts from Open-Meteo"
  y:      "[N, 12]      [pm25_h1..h6, pm10_h1..h6] ln-change targets (edge [S07]) from TRUE CPCB PM"
  y_mask: "[N, 12]      1 only if target measured"
  pm_now: "[N, 2]       ln(1+PM) at t for reconstruction"
  keys:   "base_ts, unit u, station_ids order (fixed list saved with the dataset)"
storage: "one .npz (or parquet) shard per month per unit; float32; index file with base_ts list"
leakage_asserts: ["max(history ts) == base_ts", "X_fut issue time ≤ base_ts + 1 h", "targets strictly after base_ts"]
```

### R07.3 Graph priors (graph.py)

```yaml
geometry: "d_ij km (haversine), θ_ij bearing i→j (deg); computed once from station coordinates"
wind_prior_T2_T3: "w_ij = ReLU(ws_i · cos(θ_ij − φ_i)) / max(d_ij, d_min);  φ_i = (wd_from_i + 180) mod 360;  d_min = 1 km;  w_ii = 0"
static_prior_T1: "w_ij = 1 / max(d_ij, d_min)  (no wind)"
diffusion_graph: "A_s = exp(−d_ij² / σ²), σ = 10 km, zero diagonal → L̂ = I − D^-½ A_s D^-½ (static, all variants with ADR)"
calm_wind: "if ws_i < 0.3 m/s then w_ij = 0 for all j (only self/diffusion act)"
unit_tests: ["downwind > 0, upwind == 0", "symmetric static prior", "L̂ eigenvalues in [0, 2]"]
```

### R07.4 Embedding + temporal Transformer (temporal.py)

```yaml
embedding: "e = Linear(2F → D)([X ⊙ M, M]) + Linear(F_s → D)(S)  → [N, 24, D]; D = 64"
transformer: "2 pre-norm layers, 4 heads, sinusoidal positions; built from nn.MultiheadAttention(need_weights=True, average_attn_weights=False) so attention can be exported"
outputs: {feat_seq: "[N, 24, D]", attn: "[N, 4, 24, 24] from the last layer", h_t: "feat_seq[:, -1] → [N, D]"}
```

### R07.5 Wind-biased GAT (gat.py) — used by T1, T2, T3

```yaml
gat:
  logits: "L[i,j] = LeakyReLU(aᵀ [W h_i ‖ W h_j]) + β · log(w_ij + ε)   ε = 1e-3, β learnable init 1.0;  self-loop allowed with w_ii = max_j w_ij"
  attention: "α = softmax_j(L)   (receiver i attends to senders j)"
  output: "h_spatial = h_t + W_o Σ_j α_ij W h_j   → [N, D]   (export)"
  T0: "skip GAT; h_spatial := h_t"
  note: "α is a learned weighting biased by a physics prior, NOT a physical transport coefficient (spec §10)."
```

### R07.6 ADR block (adr.py) — used by T3 only

```yaml
adr:
  state: "U = h_spatial [N, D]"
  repeat: "K = 2 steps, step size h = 0.1 (I12)"
  advection: "P[j→i] = softmax over receivers i of L[i,j] with diagonal masked (I6) → each source sends total weight 1 (mass conservation). U ← U + h · (Pᵀ-weighted inflow − U)"
  diffusion: "κ_c = hardtanh(θ_c, 0, 1) per channel; precompute L̂ = V Λ Vᵀ once; U ← V · diag(1/(1 + h·κ_c·λ)) · Vᵀ U per channel (exact implicit step, TransNet Eq.13–14 without a linear solver)"
  reaction: "f = ReLU(R1(U) + tanh(R2(U)) ⊙ U + R3(mean_time(feat_seq)) + R4(met_t)); U ← U + h · f   (TransNet Eq.15–16)"
  output: "h_adr = U → [N, D] (export)"
  T0_T1_T2: "skip ADR; h_adr := h_spatial"
  unit_tests: ["Σ_i P[j→i] = 1 for every j", "κ ∈ [0,1]", "diffusion step never increases max |U|", "permuting stations permutes outputs"]
```

### R07.6b Decoder + heads (decoder.py)

```yaml
fusion: "z = GLU([h_adr ‖ h_t]) → [N, D]   (Wu Eq.18)"
queries: "q_k = PosEmb(k) + Linear(11 → D)(X_fut[:, k]) + z,  k = 1..6 → [N, 6, D]"
decoder: "1 layer: self-attention over k, cross-attention to feat_seq, FFN"
heads: {point: "Linear(D → 2) → [N, 6, 2] → reorder to [N, 12] pollutant-major", quantiles_optional: "Linear(D → 2×3), monotone via softplus offsets"}
loss: "masked Huber(δ=1) on y; + 0.5 · pinball(q10, q90) if quantiles enabled; conformal calibration is a SEPARATE post-training step (spec §12)"
```

### R07.6c Training (train.py)

```yaml
training:
  batch: "16 samples (each sample = whole graph at one base_ts)"
  optimiser: "AdamW lr 1e-3, weight_decay 1e-4, grad clip 1.0"
  schedule: "ReduceLROnPlateau(patience 3, factor 0.2); early stopping patience 8 on val loss; ≤ 60 epochs"
  augment: "random station dropout 10% (mask whole node)"
  seeds: "fixed; log everything listed in spec §38"
  gate: "teacher must beat the local S0 student (same stations, same test hours, h1–h3) before any KD (spec §24 stage 4). Report per horizon."
variants:
  T0: {use_graph: false, use_wind: false, use_adr: false}
  T1: {use_graph: true,  use_wind: false, use_adr: false}
  T2: {use_graph: true,  use_wind: true,  use_adr: false}
  T3: {use_graph: true,  use_wind: true,  use_adr: true}
```

### R07.7 teacher_store v2 (oof.py)

```yaml
key: [teacher_variant, teacher_version, station_id, unit, base_ts]
fields:
  y_teacher: "float[6]  [pm25_h1..h3, pm10_h1..h3]"
  y_teacher_full: "float[12]"
  feat_seq: "float[24][D]"
  attn: "float[4][24][24]"
  h_spatial: "float[D]"
  h_adr: "float[D]"
  oof: bool
  fold_id: int
metadata_file: "teacher_store/{variant}/{version}/meta.json: D, heads, station list, folds, train-period z-score stats for feat_seq/h_spatial/h_adr (I9), config hash, git sha, data hash"
oof_procedure: "5 contiguous time blocks over the TRAIN period; fold model k trains on the other blocks (+ uses val for early stopping); predicts block k. Val and test windows are predicted by the final teacher (trained on all train blocks)."
size_control: "store float16 for feat_seq/attn; partition by variant/unit/month"
```

---

## [R08] STUDENT KD VARIANTS (distill/variants.py)

```yaml
student: "GDC-CNN unchanged (edge [S13.4]): input [24, 23] (A) or [24, 22] (B); F_S = last GDC output [12, D_S]; pooled f_S = mean over patches [D_S]"
train_only_heads: {psi: "Linear(D_T → D_S) on patch-averaged teacher feat_seq", P_spatial: "Linear(D_S → D_T)", P_physics: "Linear(D_S → D_T)"}
alignment:
  temporal: "teacher feat_seq [24, D_T] → average pairs → [12, D_T] → ψ → [12, D_S] vs F_S"
  afd_mask: "w_j = Σ_i mean_heads A[i,j] → z-score → softmax → M24 → sum pairs → M12 (renormalise to 1)"
losses:
  L_pred: "Huber(y_S, y)"
  L_out: "Huber(y_S, y_T)"
  L_feat: "Huber(F_S, ψ(F_T12))"
  L_afd: "(1/(12·D_S)) Σ_p M12_p Σ_d Huber(F_S[p,d], ψ(F_T12)[p,d])"
  L_spatial: "Huber(P_spatial(f_S), z(h_spatial_T))"
  L_physics: "Huber(P_physics(f_S), z(h_adr_T))"
  note: "all output losses in raw ln-change space (after the student's de-normalisation), never in instance-normalised space"
variants_cumulative:
  S0: "L_pred"
  S1: "λ·L_pred + (1−λ)·L_out, λ ∈ {0.5, 0.7}"
  S2: "λp·L_pred + ((1−λp)/3)·r(e)·(L_out + L_feat + L_afd)"
  S3: "λp·L_pred + ((1−λp)/4)·r(e)·(L_out + L_feat + L_afd + L_spatial)"
  S4: "λp·L_pred + ((1−λp)/5)·r(e)·(L_out + L_feat + L_afd + L_spatial + L_physics)"
  params: {λp: 0.4, E_warmup: 10, "r(e)": "min(1, e/E_warmup)"}
valid_teacher_pairs: "S3 needs T1–T3 (T0 has no spatial state); S4 needs T3. Central ablation (spec §28): same student config S2 (and S3 where valid) across T0–T3."
precondition: "per horizon, teacher RMSE < S0 RMSE on the same windows; otherwise report and skip KD for that horizon (Shmuel precondition)"
training: "synthetic/virtual pretrain Adam 1e-3, batch 128, ≤ 100 epochs, patience 10; real fine-tune 1e-4 (when real nodes exist)"
export_rule: "only the inference graph is exported to TFLite; heads ψ/P_* are discarded; parity test must still pass"
```

---

## [R09] KAGGLE WORKFLOW (teacher and student KD training)

```yaml
why: "owner's laptop has no GPU; Kaggle provides free GPU notebooks"
tools: "use the Kaggle MCP server if available in this Claude Code session (list its tools first); fallback: kaggle CLI (kaggle datasets create/version, kaggle kernels push/status/output) with KAGGLE_USERNAME/KAGGLE_KEY from .env"
datasets_to_upload:
  aqi-graph-windows: "graph window shards + index + station list + dataset card (version per data hash)"
  aqi-teacher-store: "teacher_store v2 outputs (after teacher runs) for the student KD notebook"
  aqi-ml-src: "a zip of ml/src/aqi_ml + shared/src/aqi_shared + configs, tagged with git sha (single source of code; the notebook imports it, it does NOT copy code)"
notebook_rules:
  - "kernel-metadata.json: id '<user>/<slug>', language python, kernel_type notebook, is_private true, enable_gpu true, enable_internet false (use packages preinstalled on Kaggle: torch, numpy, pandas, pyarrow, scikit-learn, tensorflow; add internet only if a package is missing), dataset_sources [the three datasets above]"
  - "Cell 1: print versions, GPU name, git sha; sys.path.insert(0, '/kaggle/input/aqi-ml-src/...')"
  - "Cells are SHORT and readable: load config → build loaders → train → evaluate → write outputs. All logic lives in aqi_ml."
  - "Reads only /kaggle/input; writes only /kaggle/working (checkpoints, teacher_store, metrics.json, plots)"
  - "Checkpoint every epoch; resumable if the session stops; must pass 'Save & Run All' non-interactively"
  - "Respect Kaggle session-time and weekly GPU quotas (verify current limits); first run a 1-epoch smoke test on 1 fold"
  - "Seeds fixed; log the spec §38 reproducibility fields into metrics.json"
pull_back: "download outputs (kaggle kernels output) into data/kaggle_runs/{run_id}/; register paths in docs/PROGRESS.md"
local_cpu_ok: "data building, simulator, RF baselines, unit tests, export, parity — stay on the laptop"
```

---

## [R10] NEW MILESTONES (replace P8-remaining, P9, P10 ordering)

```yaml
milestones:
  N0_step0: {done_when: ["architecture_review_v2.md written", "owner approved", "ADRs for each approved decision (I1–I15)"]}
  N1_cleanup: {done_when: ["R05 moves/deletions done after approval", "banners added to superseded spec sections", "tests green"]}
  N2_graph_windows: {done_when: ["graph window builder + leakage asserts + unit tests", "historical weather forecasts (or oracle flag) added", "shards built for units 0–2", "dataset card"]}
  N3_teacher_code: {done_when: ["all R07 unit tests pass on CPU with a tiny random graph", "overfit test on 50 samples", "walkthrough.md teacher part"]}
  N4_kaggle_teacher: {done_when: ["datasets uploaded", "T3 smoke run", "T0–T3 final teachers trained", "teacher-vs-S0 gate reported per horizon"]}
  N5_teacher_store: {done_when: ["OOF folds for T0–T3 (R04 I15)", "teacher_store v2 + meta (z-score stats)", "leakage test: fold model never predicts its own block"]}
  N6_student_kd: {done_when: ["train wrapper + heads", "S0–S4 for Model A on Kaggle", "S0–S1 for Model B", "export parity still < 1e-4 (heads excluded)"]}
  N7_experiments: {done_when: ["spec §27 matrix: RMSE/MAE/R² by horizon/station/regime/extremes/missing-sensor", "spec §28 teacher ablation T0–T3", "spec §29 sensor-failure experiment", "skill retained", "plots + tables from scripts in ml/src/aqi_ml/eval"]}
  N8_deploy: {done_when: ["chosen student exported FP32 (+ INT8 ablation)", "P10 firmware resumes (Arduino, ADR 0002)", "Flash/RAM/latency/energy measured on ESP32-S3"]}
  N9_mlops: {done_when: ["thin ZenML pipelines wrapping the SAME functions used by the notebooks (no duplicate logic)"], note: "deferred until N7 results exist"}
```

---

## [R11] CLAUDE CODE CLOUD / WEB SESSIONS — RECONCILE AND STEER

```yaml
reconcile_steps:
  1: "git fetch --all; list remote branches and open PRs created by Claude Code cloud/web sessions (gh pr list if GitHub CLI is available)"
  2: "for each branch/PR: summarise changes, map to R05 (keep/adapt/freeze/delete) and R10 milestones; write the table into docs/architecture_review_v2.md §'In-flight work'"
  3: "do NOT merge work that implements the superseded teacher/KD design (LightGBM-as-teacher KD, variant A/B teacher, old teacher_store) — close it with a comment linking this file; cherry-pick only parts that R05 keeps"
  4: "rebase any kept branch on main after N1 cleanup"
standing_rules_for_every_session:
  - "Read docs/specs/research_spec_v2.md and this file before working; follow R03"
  - "One active session per milestone; claim it at the top of docs/PROGRESS.md ('N3 — owner: cloud session <date>') and release it when done"
  - "End every session by updating docs/PROGRESS.md and docs/HANDOFF.md (what changed, what is next, open questions)"
  - "Never edit superseded spec sections except to add the banner"
```

---

## [R12] DOCUMENTATION UPDATES

```yaml
update:
  CLAUDE.md: "add: source-of-truth hierarchy (R01), simplicity rules (R03), Kaggle workflow pointer (R09), milestone list (R10)"
  docs/PROGRESS.md: "replace P8-remaining/P9/P10 with N0–N9; keep P0–P7 history; carry over open questions (pm25_scale, AQI gap rule, SGP41 kn_u, MQ-7 vendor, SGP41 RH flag at 90%, arduino core update)"
  docs/HANDOFF.md: "rewrite for v2"
  ADRs: "one per approved I1–I15 decision; one for 'LightGBM teacher demoted'; one for 'ZenML deferred'; one for 'Kaggle training'"
  docs/walkthrough.md: "new (R03)"
```

---

## [R13] WHAT MUST NOT CHANGE

```yaml
frozen:
  - "Edge invariants I01–I12 (edge [S01]); targets and output order"
  - "Student architecture and its input restriction (spec §18): no neighbour data, no graph, no teacher at inference"
  - "Fixed sensor pool; features derivable only from those sensors + time"
  - "Novelty language (spec §3, §31): 'candidate research gap', 'to our knowledge'"
  - "Contracts with the future cloud backend (edge [S16])"
```

---

## [R14] OPEN QUESTIONS FOR THE OWNER (answer in step 0)

```yaml
questions:
  - "Approve proposals I1–I15 (or change them)."
  - "Teacher node features: sensor-proxy (virtual units) + Open-Meteo, as in I11? (This keeps the comparison fair: the teacher's advantage is the network, not better sensors.)"
  - "Is the Kaggle MCP server configured in Claude Code, or should the kaggle CLI be used?"
  - "Should RF baselines get any further work, or stay frozen?"
  - "Carry-over questions from PROGRESS (pm25_scale, AQI gap rule, SGP41 kn_u range, MQ-7 vendor, SGP41 RH flag, arduino core update)."
```
