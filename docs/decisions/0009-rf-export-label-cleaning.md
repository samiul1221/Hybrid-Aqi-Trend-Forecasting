# 0009 — RF export format, RF training protocol, label spike cleaning (P6)

Status: accepted (2026-09-27)

## RF export: own blob + generic C evaluator (instead of emlearn's generated code)
[S05]/[S13.2] prefer "serialised arrays file + generic evaluator (update via model OTA)". emlearn's main
path generates compiled-in C code. We write a small versioned blob (`aqi_ml.export.rf_blob`) read by a
pure-C evaluator (`edge/firmware/aqi_node/src/models/rf_eval.{h,c}`), shared by firmware and host:
- node = 9 bytes (u8 feature / 255 leaf, f32 threshold or leaf value, u16 left, u16 right) → the 400 KB
  budget holds ~44k nodes; header carries the feature-schema hash (I05, checked by `rf_load`).
- exact decisions: sklearn compares float32 inputs with float64 thresholds; the blob stores the largest
  float32 ≤ threshold, which yields identical left/right decisions for every float32 input.
- `rf_load` validates magic, version, sizes, feature indices and that children follow parents (no cycles),
  so a corrupt OTA blob cannot hang the device.
Result: device-vs-Python parity 1.3e-7 (A) and 1.4e-7 (B) on 1000 test vectors (target < 1e-4).

## Training protocol
- Grid [S13.2] (54 configs) searched on a 25 % train subsample with a multi-output forest as a speed proxy;
  candidates must fit the 400 KB budget; best validation RMSE (mean of 6 log targets) wins; final model
  = 6 single-output forests on all training windows, real blob size re-checked.
- Training excludes the 5 spatial hold-out stations; reports val, test and test on hold-out stations.
- Stages cached in `data/models/<dataset>/rf-<role>/` so a run resumes after interruption.
- Scores are on synthetic data (not results per [S11.3]); the 10 seeds × 5 blocks protocol is used for the
  KD comparison in P8.

## Label spike cleaning ([S09] + cloud [C06] Hampel)
The first RF evaluation exposed garbage labels: PM10 up to 7.4e13 µg/m³ and 3157 one-hour PM2.5 jumps > 7×
(0.1 % tail: 72×). Cleaning now adds: `pm10 > 3000 → NaN`, and a Hampel filter on ln(1+PM) (centred 24 h,
k = 3, 1.4826·MAD with a 0.2 floor — [C06]/Wu 2026 parameters) that removes only flagged runs of ≤ 3 h
(glitches); longer flagged runs are kept as real episodes (a 12 h step episode is kept, Diwali 2025 at
Anand Vihar untouched). Effect: 2.1 % of PM2.5 hours removed, > 7× jumps 3157 → 832. The cloud keeps its
flag-only rule for stored readings; this cleaning applies to training labels in the base table.
