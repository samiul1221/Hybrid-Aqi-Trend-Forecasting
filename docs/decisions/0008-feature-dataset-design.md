# 0008 — Feature builder and training-dataset layout ([S07], [S08], [S11.1], [S14])

Status: accepted (2026-09-27)

## Rules the spec leaves open (now fixed in `aqi_ml.features.builder`, mirrored by firmware in P10)
- Channel masks: MEASURED bit of the record flag (`flag_pm` for the four PM channels, `flag_t`, …).
  `aqi_now` is valid only on a measured PM hour with a ready 24 h AQI; calendar and `pm_mask` always 1.
- Filled PM hours (`filled_le_1h`) keep the carried value with mask 0 ([S14]); Model A still applies.
- ≤ 2 h rule: only interior NaN runs of length ≤ 2 are interpolated; longer runs and edges stay NaN.
  Masks are never changed by filling. Inference: last value forward, mask 0.
- A Model A training window at base hour t is valid when: 24 consecutive hours with no NaN after the fill
  (all 23 channels), `pm_state[t]` ∈ {ok, filled_le_1h}, ≥ 20 measured PM hours in the window ([S11.1]),
  finite RF row and finite 6 targets. Consequence: `aqi_now` is missing until the 24 h AQI is ready
  (16 measured hours), so every long PM gap costs ~16 extra hours of windows — kept spec-literal.
- RF windows include hour t; `roll_std` is population std; `rain_hours_last` counts `rain_frac > 0.1`
  hours; `dust` is the record flag as 0/1.
- Targets: synthetic nodes use TRUE CPCB PM for both PM[t] and PM[t+h] ([S10.3]); real logs use the
  node's own corrected PM with non-measured hours set to NaN ([S11.1], I07).
- Model B ([S14]): dropout k ~ U{1..24} per valid Model A window (seeded per node) plus real outage
  windows (simulated SPS30 outages / real fallback hours; CPCB truth gaps excluded). Last valid PM hour
  = t − k; it and t − k − 3 must be measured. Targets relative to label PM at t − k. Model B RF =
  27 PM-free RF features at t + 5 extras = 32 (schema `rf_names_b`).
- Open (P7): how the 5 scalar extras enter the GDC-CNN is not specified; the dataset provides them as
  `X_extra` next to the 17-channel sequence.

## Layout
`data/datasets/<version>/node_id=*/{hourly,windows_a,windows_b}.parquet` + `manifest.json`.
X_seq is sliced from `hourly.parquet` by row index at load time (`features.dataset.load`) instead of being
stored per window (~9 GB for the full simulator output). Row index = base hour t; X_seq = rows t−23..t.

## Golden fixture
`shared/fixtures/feature_golden/feature_golden.json`: 60 designed hours (filled hour, 2 h gap
interpolated, 3 h gap invalidating, AQI warm-up, rain, dust, IST midnight) with filled values, masks,
RF rows, targets, validity and Model B extras. Firmware must match within 1e-5.
