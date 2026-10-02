# Architecture review v2 (CR v2 step 0, [R04])

Status: **APPROVED 2026-10-02** (owner: "proceed") with every proposal in §6 as written, ERA5 oracle weather (I19), RF frozen. Still open: B1 (research_spec_v2.md), LightST PDF, carry-over questions. ADRs 0012–0018.
Inputs read: `CHANGE_REQUEST_v2.md`, edge plan [S07][S08][S12]–[S14], cloud plan [C08], `docs/PROGRESS.md`,
`docs/HANDOFF.md`, ADRs 0001–0011, data card, sim manifest, and the PDFs listed in §5. Facts marked
**(measured)** were computed from `data/` today; **(PDF)** were read from the paper in `docs/papers/`.

## 0. Blockers and readiness

| # | item | state |
|---|---|---|
| B1 | `docs/specs/research_spec_v2.md` (priority-1 source, §1–42) | **missing** — never added to the repo or pasted in a session. §39 items 7 ("verification of the 10 items"), the novelty language §3/§31, the experiment matrix §27–§29 and the reproducibility list §38 cannot be checked until it is added. This review uses the CR's quotations of the spec only. |
| B2 | Cited PDFs | all on disk except **LightST (AAAI 2025)** (update 2026-10-02: owner added Wang 2024, REDNet, KDAQnet, TSMNet, ST-BernT, AIoT/GTA-Net, + Zhao 2025 DGGNN). DOIs in §4/§5 read from the PDFs. |
| B3 | Kaggle | **ready**: Kaggle CLI 2.2.4 authenticated as `samiulseikh` via `KAGGLE_API_TOKEN` in `.env` (checked with `kaggle kernels list --mine`). No Kaggle MCP server in this session → CLI path of [R09]. Weekly GPU quota: TODO_VERIFY on kaggle.com. |

## 1. Text diagram

```
                        OFFLINE (Kaggle GPU)                                      ON DEVICE (ESP32-S3)
 all stations, one base_ts                                                 one node, no neighbours
 X_node [N,24,F] ─┐
 M_node [N,24,F] ─┼─ Embedding ─► Temporal Transformer ─► feat_seq [N,24,D], attn [N,4,24,24]
 S_node [N,F_s] ──┘   (per node)        │  h_t = feat_seq[:, -1] [N,D]
                                         ▼
 wind_t [N,2] ─► wind prior w [N,N] ─► wind-biased GAT ─► h_spatial [N,D]          (T1–T3; T0: = h_t)
                                         ▼
 static L̂ (eig once) ───────────────► ADR (K=2: adv → diff → react) ─► h_adr [N,D]  (T3; else = h_spatial)
                                         ▼
 X_fut [N,6,F_f] ─► GLU(h_adr‖h_t) + horizon queries ─► decoder ─► ŷ [N,12] (+ q10/q90)
                                         │
             OOF per fold ──► teacher_store v2 (per station, unit, base_ts):
                              y_teacher[6], feat_seq→[12,D], AFD mask M12[12], z(h_spatial), z(h_adr)
                                         │
                                         ▼  (train-only heads ψ, P_spatial, P_physics; discarded)
 local window [24,23] (A) / [24,22] (B) ─► GDC-CNN student (unchanged) ─► ŷ_S [6] ─► TFLite FP32 ─► firmware
```

## 2. Tensor shapes (proposed numbers; D = 64, heads = 4, L = 24, H = 6)

| tensor | shape | notes |
|---|---|---|
| N (train graph) | 25 | 30 simulated stations (sim rule ≥ 8760 PM2.5 h) − 5 spatial hold-out (sim manifest: patparganj, rohini, sector_125_noida, sirifort, sonia_vihar). Test graph: N = 30 (dense ops accept any N). |
| X_node, M_node | [N, 24, F], F = 30 | 23 student channels i0–i22 (unit u, sensor-proxy, same builder) + 7 Open-Meteo past (t2m, rh2m, sp, u10, v10, precip, swrad). See I1/I17. |
| S_node | [N, 6] | lat, lon, elevation, height_agl, sheltering_factor, is_reference (I3) |
| X_fut | [N, 6, 9] | t2m, rh2m, sp, u10, v10, precip, swrad, hod_sin, hod_cos (blh and is_holiday dropped, I17/I18) |
| wind_t | [N, 2] | ws, wd_from at base_ts (Open-Meteo) |
| y, y_mask | [N, 12] | [pm25_h1..h6, pm10_h1..h6] ln-change from TRUE CPCB PM |
| pm_now | [N, 2] | ln(1+PM) at t |
| e (embedding) | [N, 24, 64] | Linear(2F→D)([X⊙M, M]) + Linear(6→D)(S) |
| feat_seq / attn | [N, 24, 64] / [N, 4, 24, 24] | last Transformer layer |
| w, GAT logits L, α | [N, N] | dense |
| h_t, h_spatial, h_adr, z | [N, 64] | |
| L̂ = VΛVᵀ | [N, N] | precomputed per station set |
| queries / decoder out | [N, 6, 64] | |
| ŷ | [N, 6, 2] → [N, 12] | pollutant-major |
| batch | [B=16, …] | one sample = whole graph at one base_ts |
| student input (as built) | [12, 46] A / [12, 44] B | after the existing 2-hour patching ([24,23]/[24,22] before); ADR 0010 |
| F_S | [12, D_S], D_S = 16 (A) / 64 (B) | D chosen by the existing search (ADR 0010) |

Sample count **(measured)**: 19,686 train IST hours (2018–2021) have ≥ 1 station with both PM targets → ≈ 59k graph samples over 3 units.

## 3. Fixed vs hypothesis

| fixed (do not change) | hypothesis (tested by ablation) |
|---|---|
| CPCB AQI, IST buckets, target definition + output order (I01–I04) | the graph helps (T1 vs T0) |
| student architecture + input restriction (no neighbours/graph/teacher at inference) [R13] | wind direction in the prior helps (T2 vs T1) |
| feature schema hash; sensor pool | ADR physics helps (T3 vs T2) |
| chronological splits train 2018–21 / val 2022 / test 2025–26 | output KD helps the student (S1 vs S0) |
| OOF teacher outputs for student training windows | temporal / spatial / physics KD each add skill (S2, S3, S4) |
| FP32 on device | spatial knowledge "survives" into an isolated student (research question) |

## 4. Literature → component (only PDFs on disk verified)

| component | source (PDF) | what we take | deviation |
|---|---|---|---|
| wind edge prior `ReLU(ws·cos(θ−φ))/d` | TransNet, Dimri et al., npj Clean Air 2:12 (2026), doi 10.1038/s44407-026-00052-x | form of the prior | φ = wd_from + 180 (I4/I19); d_min = 1 km; calm cut-off |
| advection with per-source softmax | TransNet Eq. 8–11 | mass conservation, U ← U + h(inflow − U) | weights from GAT logits, not A1–A4 (I6) |
| implicit diffusion, κ = hardtanh | TransNet Eq. 12–14 | exact implicit step | static Gaussian graph, eigendecomposition (I20) |
| reaction | TransNet Eq. 15–16 | ReLU(R1 + tanh(R2)⊙U + R3 + R4) | R3 input = mean(feat_seq); no PBL height (I17) |
| h = 0.01 | TransNet (PDF) | — | start h = 0.1, test {0.01, 0.1} (I12) |
| edge-biased GAT, Transformer, H-query decoder, GLU fusion, quantile heads | Wu, Mo, Li, IJGI 15(3):101 (2026), doi 10.3390/ijgi15030101 (WSDST-GAT) | architecture blocks | GAT once at base_ts (I5). **Note: Wu forecasts ozone (Beijing 2022), not PM.** |
| L_pred, L_feat, L_AFD, r(e) | Na et al., IEEE Access 14 (2026), doi 10.1109/ACCESS.2026.3667131, Eq. 16–17 | Huber, AFD, curriculum | adds L_out, L_spatial, L_physics (I8/I21) |
| output KD + precondition | Shmuel, Lazebnik, Glickman, IEEE Access (2026), doi 10.1109/ACCESS.2026.3657793 | α-blend, "teacher must beat student" | — |
| physics in architecture > physics as loss | TopoFlow, npj Clim. Atmos. Sci. (2026), doi 10.1038/s41612-026-01417-5 | design rationale | elevation bias skipped (Delhi flat) |
| directed hourly wind graph + GAT + weather forecasts in decoder | Wang, Sun, Gu, Cao, Shi, He, Sci. Total Environ. 946 (2024) 174229, doi 10.1016/j.scitotenv.2024.174229 (WDDSTG-Net, **ozone**) | forecast covariates in the decoder; GAT over a wind-directed graph | we use ERA5 "oracle" weather (I19) |
| dynamic wind graph + static geographic graph for PM2.5 | Zhao, Liu, Yang, Qi, Lian, Environ. Model. Softw. 186 (2025) 106351, doi 10.1016/j.envsoft.2025.106351 (DGGNN, GCN+LSTM) | supports wind prior + static diffusion graph (I20) | — |
| two-stage feature + response KD to an edge student | Fu et al., Pattern Recognit. Lett. (2026), doi 10.1016/j.patrec.2026.08.011 (REDNet) | projected-feature alignment, z-scored soft targets (I9) | our teacher is a graph, not a foundation model |

## 5. Novelty vs counterexamples

Candidate research gap (wording per [R13]): *to our knowledge*, no journal work distils the **network-level
(spatial, wind-transport, physics) internal states** of a multi-station graph teacher into an **isolated
single-station student that runs on a microcontroller with no neighbour data at inference**, and measures how
much of that knowledge survives (T0–T3 × S0–S4). The spec's own wording (§3, §31) is still unchecked (B1).

| closest work | what it does | why it is not our setting |
|---|---|---|
| LightST (AAAI 2025, conference) | graph → light model distillation for spatio-temporal forecasting (per CR) | **PDF not on disk — must be read before any novelty sentence**; strongest counterexample |
| KDAQnet, Kumar et al., Remote Sens. Appl. Soc. Environ. 43 (2026) 102203, doi 10.1016/j.rsase.2026.102203 | KD (ResNet+FCN student) for multi-pollutant forecasting from satellite data | teacher and student see the same gridded spatial input; no graph, no isolated node, no MCU |
| REDNet (above) | foundation-model teacher → MLP-BiGRU edge student, feature + response KD | single-series teacher; no spatial privileged information |
| Na 2026 (above) | Transformer → GDC-CNN with privileged *future* data, L_feat + L_AFD, MCU deployment | privileged info is temporal, not spatial; single series |
| An, Msigwa, Yun, IEEE IoT J. 12(19) (2025), doi 10.1109/JIOT.2025.3591268 (GTA-Net virtual sensors) | GAT + Transformer estimates PM2.5 at unmonitored sites from neighbours | needs neighbours **at inference** (opposite of our student) |
| Qi et al., IEEE IoT J. 13(2) (2026), doi 10.1109/JIOT.2025.3617922 (ST-BernT) | Gaussian distance + correlation graph, Bernstein filter + Transformer, multisite | graph model, no distillation, network needed at inference |
| TransNet, WSDST-GAT, WDDSTG-Net, DGGNN | wind-aware graph forecasters | teacher candidates only; no distillation |
| Tang & Guan, Appl. Soft Comput. 178 (2025) 113208, doi 10.1016/j.asoc.2025.113208 (TSMNet) | lightweight 2D-dilated-conv forecaster | lightweight baseline idea, no KD, no spatial info |

## 6. Inconsistencies

### 6a. Seed list R04.1 — verdicts

| id | verdict | evidence / proposed decision |
|---|---|---|
| I1 | **confirmed** | [C08.4]: 32 names, 35 columns (pm_state_onehot = 4); no node wind. Proposal changed for simplicity: teacher F = the 23 student channels (reuse the student builder, includes node wind) + 7 Open-Meteo past = **30** (no dust/pm_state one-hot/festival). Graph prior uses Open-Meteo wind only. |
| I2 | **confirmed** | `feature_schema.yaml`: 6 channels have `model_b: false` → 17 + 5 extras = 22; `preprocess.py` already broadcasts the extras as constant channels. |
| I3 | confirmed | 6 static columns as proposed; height_agl/sheltering constant for CPCB stations (kept for real nodes later). |
| I4 | **confirmed, and TransNet is literally upwind** | TransNet converts to meteorological WD (FROM, `270 − atan2(V,U)`) and puts it straight into cos(θ_ij − WD) — that maximises the edge toward the upwind station. Our φ = WD_from + 180 is a deliberate correction → ADR + the CR's unit test. |
| I5 | agree | GAT and ADR once at base_ts on h_t. |
| I6 | agree | documented simplification of TransNet Eq. 8–11. |
| I7 | agree | S0–S4 cumulative; S3′ optional. |
| I8 | agree | see I21. |
| I9 | agree | z-score with train stats in meta.json. |
| I10 | agree | |
| I11 | agree | teacher_store keyed by (station, unit, base_ts); simulator already masks hours without CPCB truth (`sim_gap`). |
| I12 | agree | TransNet h = 0.01 confirmed (PDF). |
| I13 | agree | LightST PDF not on disk. |
| I14 | agree | |
| I15 | agree | with N = 25 dense, one fold should be minutes on a T4; measure first. |

### 6b. New findings

| id | finding | proposal |
|---|---|---|
| I16 | **No station reaches 70% PM completeness in train (measured):** best ≈ 40–45%/yr; 2018–21 average ≥ 30% for only 16 stations, ≥ 20% for 24. [R07.1]'s rule would give an empty graph. | Keep all 30 simulated stations; rely on node masks + masked loss; a sample needs ≥ 1 measured target (cloud [C08.7] rule). |
| I17 | `blh` is null for Delhi in Open-Meteo archive and historical-forecast (data card, verified 2026-09-26). X_fut and the reaction "Met" both list it; TransNet uses PBL height. | Drop blh; X_fut = 9 columns; report as a limitation. |
| I18 | `is_holiday` has no calendar anywhere in the repo; the student schema has none. | Drop from X_fut now (9 columns); optional later with a cited holiday list. |
| I19 | Future weather: `hforecast` download step exists but was never run; Open-Meteo historical-forecast archive starts around 2021–22 (TODO_VERIFY), so train 2018–21 has no issued forecasts. | Use ERA5 actuals for X_fut in **all** splits with flag `oracle_weather: true` (consistent train/test), report as a limitation; optional check on 2025 test with real historical forecasts. |
| I20 | TransNet diffuses on the Laplacian of the (dynamic) wind graph; [R07.3] uses a static Gaussian graph. | Accept (one eigendecomposition, simple); ADR. |
| I21 | Na Eq. 17 has only L_feat + L_AFD with weight (1−λp)/2 and **no L_out**; λp = 0.4 is from edge [S13.3], not found in the Na text. | S2–S4 are our extension (ADR); λp = 0.4 TODO_VERIFY against Na's implementation details. |
| I22 | teacher_store size: full feat_seq + attn in float16 ≈ 7.7 KB/row; ≈ 3.1M train rows (30 stations × 3 units × 35k h) → ≈ 24 GB **per variant**; laptop has 36 GB free. | Store what the student uses: feat_seq patch-averaged [12, D] + AFD mask M12 [12] + h_spatial/h_adr [D] → ≈ 2 KB/row, and only for hours with a student window → < 1 GB per variant. Changes [R07.7] field shapes (ADR). |
| I23 | CR says student input [24, 23]; the built model takes [12, 46] (2-hour patches). | Wording only; F_S = [12, D_S] as in the CR. |
| I25 | Owner directive 2026-10-02: teacher input must match the edge sensor domain. Audit (`docs/research_notes/teacher_input_audit.md`): student path is sensor-only; v1 LightGBM teacher used clean neighbour CPCB PM. | ADR 0019; graph builder guards G1–G4. |
| I26 | (found in N3) [R07.3] w[i,j] = source→target but [R07.5] adds log w_ij to receiver-i logits → literal reading favours downwind senders. | GAT uses w[j, i]; ADR 0021. |
| I24 | CNN-A D_S = 16 is small for 3 projection heads from D_T = 64. | Keep (student is frozen per [R13]); heads are train-only. |

## 7. Verification of the 10 items (spec §39 item 7)

Blocked on B1 — the 10 items are defined in `research_spec_v2.md`.

## 8. Minimal implementable architecture

Exactly [R06]–[R08] with the decisions above: dense PyTorch, files `graph.py temporal.py gat.py adr.py
decoder.py model.py train.py oof.py config.py`, booleans use_graph/use_wind/use_adr, plain loop,
one YAML per variant. Student code unchanged except `train_wrapper.py`. Teacher ≈ 0.2–0.4 M params.

## 9. In-flight work ([R11])

| source | content | map to R05 | proposal |
|---|---|---|---|
| remote | `origin/main` only; no other branches; `gh pr list` empty | — | — |
| worktree `agent-aac41351…` (uncommitted, cloud agent stopped on usage limit) | ZenML `ml/pipelines/{pipelines,run,steps}.py` + zenml/mlflow deps | delete (old P8 flow, R10 defers MLOps) | drop |
| same | `run_kd.py` (+185: summaries, `select_variant`), CNN/LGBM train tweaks | superseded KD | keep **results only** |
| same | `docs/research_notes/e2_kd.md`: RF-A out_KD hurts (+0.3…+1.0%); RF-B α0.0 −0.35%; **CNN-A λ0.5 −1.13%, all 6 outputs p < 0.05**; CNN-B 2/3 seeds | keep as results note | copy to main |
| same | `export/verify.py` (schema hash, size budget, parity re-check), `lineage.py` | keep (export path) | review + cherry-pick in N1 |

## 10. Files to move / delete in N1 (needs approval; one commit "chore: remove v1-teacher artifacts (CR v2 R05)")

- move `ml/src/aqi_ml/models/teacher/interim_lgbm/` → `ml/src/aqi_ml/baselines/lgbm_teacher/` (frozen); update imports in `distill/run_kd.py`, tests.
- delete CNN out_KD path in `distill/run_kd.py` (`run_cnn`, `_pack`) and `kd.out_kd_loss` after the results note is in; keep RF out_KD (baseline) and `precondition`.
- keep `distill/kd.py` full_KD losses (become S2), `teacher_store.py` (→ v2).
- delete the worktree branch `worktree-agent-aac41351ca97c83e4` after cherry-picking `verify.py`, `lineage.py`, `e2_kd.md`.
- add `ml/pipelines/README.md`: "deferred, see CR v2 R10".
- add SUPERSEDED banners: cloud [C08.3], [C08.5], [C08.9]; edge [S12.2], [S13.0].

## 11. Questions for the owner

1. Add `docs/specs/research_spec_v2.md` (B1) and the missing PDFs (B2).
2. Approve I1–I15 as in §6a (note the I1 change to F = 30) and I16–I24.
3. Future weather: ERA5 "oracle" everywhere (I19), or spend time on historical forecasts for 2022+ only?
4. RF baselines: frozen (proposed).
5. Carry-over: pm25_scale, AQI gap rule, SGP41 kn_u, MQ-7 vendor, SGP41 RH flag 90%, arduino core update.
