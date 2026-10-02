# 0015 — teacher_store v2 and OOF cost (CR v2 R07.7, I15, I22)

Status: accepted (2026-10-02)

- Key (teacher_variant, teacher_version, station_id, unit, base_ts).
- **Size (I22):** raw feat_seq [24, 64] + attn [4, 24, 24] in float16 ≈ 7.7 KB/row ≈ 24 GB per variant
  (30 stations × 3 units × 35k h) — too big (36 GB free on the laptop, Kaggle output limits). The store keeps
  exactly what the student losses use: feat_seq patch-averaged **[12, D]**, AFD mask **M12 [12]** (attention
  → column importance → patch sum, [R08]), z-scored h_spatial [D], h_adr [D], y_teacher [6], y_teacher_full [12],
  oof, fold_id; float16 arrays. ≈ 2 KB/row, and only hours that have a student window.
- **OOF (I15):** 5 contiguous train blocks for T3; T0–T2 may use 3 if Kaggle GPU time is tight (measure one fold first).
