# 0016 — LightGBM interim teacher demoted; RF students frozen (CR v2 R05)

Status: accepted (2026-10-02)

- The interim LightGBM teacher (ADR 0011) moves to `aqi_ml.baselines.lgbm_teacher` and is frozen; it stays in
  the comparison table as a non-graph privileged teacher.
- RF students A/B are baselines only (RF cannot take feature KD); no further work.
- CNN out_KD with the LightGBM teacher is removed from code; its results are kept in
  `docs/research_notes/e2_kd.md` (CNN-A λ = 0.5: −1.13 % RMSE, all 6 outputs p < 0.05; CNN-B 2/3 seeds).
