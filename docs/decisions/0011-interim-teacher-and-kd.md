# 0011 — Interim LightGBM teacher, teacher_store, KD experiments (P8)

Status: accepted (2026-09-27); CNN out_KD runs partially done (see HANDOFF)

## Teacher ([S12.1], cloud [C08.10])
- One LightGBM (Huber) per output (6: pm25/pm10 × h1..h3 — the edge subset; the cloud teacher will add h4..h6).
- Privileged inputs: 5 nearest OTHER CPCB stations (ln PM2.5 at t, t−1, t−3; ln PM10 at t; distance;
  upwind score cos(bearing node→station − wind FROM at the node), 0 when calm) + upwind-weighted neighbour
  PM and 3 h trend + weather at t+1..t+3. Historical weather *forecasts* do not exist before 2022, so actual
  ERA5/station weather is used (cloud [C07] fallback; acceptable as teacher-only privileged information).
  Plus the node's own 50 RF features.
- OOF: 5 contiguous time blocks over the training period (2018-04 … 2021-12, hold-out stations excluded);
  val/test/hold-out windows predicted by the model trained on the whole training period.
- teacher_store v1 (`aqi_ml.distill.teacher_store`): parquet per node/month, y_teacher_* + oof + fold_id +
  optional feat_seq/attn blobs (shape columns) — ready for the graph teacher. Leakage rules per [S12.2].
- Validation log-RMSE: pm25 h1/h2/h3 0.240/0.326/0.387, pm10 0.197/0.282/0.345.

## Precondition ([S12.1], E3) — teacher vs no-KD student, validation log RMSE
- RF-A: teacher better on all 6 outputs, but only marginally (e.g. 0.240 vs 0.252).
- CNN-A: teacher better only on pm25 h1/h2 and pm10 h1 → KD gated per output (others keep true targets).
- RF-B / CNN-B: teacher much better (0.24 vs 0.48–0.56) — the teacher still sees PM (neighbours).

## KD results (E2; 5 seeds × 5 test time blocks, paired t-test vs α = 1.0; synthetic data)
- RF-A: KD **hurts** slightly: α=0.5 +0.5 % RMSE, α=0.0 +1.2…1.8 % (p < 1e-4). Consistent with Shmuel
  et al.: gains need a teacher ≫ student. Deploy RF-A no_KD.
- RF-B: small gains on PM10 (α=0.5/0.0: −0.5…−1.0 %, significant at h2/h3), PM2.5 ≈ unchanged. The
  teacher's edge comes from neighbour PM that the student cannot observe.
- CNN out_KD (λ ∈ {0.5, 0.7}, 3 seeds): runner `aqi_ml.distill.run_kd --arch cnn` is resumable; A was
  6/9 runs done when handed off; B not started.
- full_KD (Na L_pred + L_feat + L_afd, curriculum r(e), λp = 0.4, trainable ψ) is implemented and
  unit-tested with a fake store holding feat_seq/attn; with LightGBM (no features/attention) it reduces to
  out_KD ([S12.2]). Real full_KD needs the graph teacher (phase 2).
