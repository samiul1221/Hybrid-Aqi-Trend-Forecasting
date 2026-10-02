# 0014 — v2 student KD variants S0–S4 and losses (CR v2 I7–I10, I14, I21, I24)

Status: accepted (2026-10-02). Replaces edge [S13.0] no_KD/out_KD/full_KD.

- **Cumulative variants (I7):** S0 L_pred; S1 + L_out; S2 + L_feat + L_AFD; S3 + L_spatial; S4 + L_physics.
- **Weighting (I8, I21):** L = λp·L_pred + ((1−λp)/n_kd)·r(e)·Σ active KD terms; S1 keeps the Shmuel
  λ-blend (λ ∈ {0.5, 0.7}). Na et al. Eq. 17 has only L_feat + L_AFD with (1−λp)/2 and no L_out: S2–S4 are
  our extension. λp = 0.4 and E_warmup = 10 come from edge [S13.3] (TODO_VERIFY against Na's implementation details).
- **Scale (I9):** teacher states are z-scored per dimension with train statistics stored in teacher_store meta.
- **Horizons (I10):** student h1–h3, teacher h1–h6, "skill retained" on h1–h3.
- **Model B (I14):** S0, S1 first (teacher re-expressed relative to last valid PM, [S14]); S2–S4 only if time permits.
- **Student size (I24):** CNN-A keeps D_S = 16 (student frozen per [R13]); heads ψ / P_* are train-only.
