# 0003 — CPCB AQI interpretation choices ([S06.6], [S07])

Status: accepted (2026-09-26). Breakpoints still TODO_VERIFY against the official CPCB NAQI document.

The spec leaves several details open. One implementation (`aqi_shared.aqi`, C twin in P10) and one
table (`cpcb_breakpoints.json`) fix them. Edge and cloud must agree exactly, because a formula
drift would create false local events ([S17] C02).

| Topic | Spec text | Decision |
|---|---|---|
| Gap rule | "values between listed integer bands (e.g. 30.5) use previous band's B_hi as B_lo" | Applied to the **whole band**: band k>0 uses B_lo = B_hi[k−1], I_lo as listed (51, 101, …). The only reading that is monotonic. Differs from the official calculator by up to ≈3.3 AQI points just above a band start (PM2.5 61 → 104.3 vs 101; PM2.5 45 → 75.5 vs 74.7), never across a category. Alternatives: a fully continuous version (I_lo = previous I_hi) or the official integer B_lo with gaps clamped. |
| Band boundary | PM2.5 Very Poor [121, 250], Severe [250, …] overlap at 250 | Upper bound inclusive: 250 → 400 (Very Poor). Same for PM10 430. |
| Severe upper limit | "from official CPCB calculator (verify); cap 500" | PM2.5 380, PM10 510 (slope of the Very Poor band continued, as in common CPCB calculator implementations), cap 500 above. TODO_VERIFY. |
| Integer AQI | not specified | `floor(aqi + 0.5)` (round half up; same in C). Float kept internally. |
| Validity | "≥16 measured hours of 24" | Per pollutant; **both** PM2.5 and PM10 need ≥ 16 (same SPS30). `aqi_valid_hours` = min of the two. NaN counts as not measured. Negative C clipped to 0. |
| Trend roll | "roll 24 h buffer forward: append predicted hours" | Predicted hours count as valid slots (otherwise the roll has no effect). Trend is `null` when AQI[t] or AQI[t+3] is not ready. Threshold from table/manifest (default 5), compared on float ΔAQI. |
| Category | bands by AQI | From float AQI: ≤50 Good, ≤100 Satisfactory, ≤200 Moderate, ≤300 Poor, ≤400 Very Poor, else Severe. |
| Calendar | not specified | dow 0 = Monday … 6 = Sunday; doy 1 … 366 (used by features, `feature_schema.yaml` calendar_conventions). |

Changing any row changes `shared/fixtures/aqi_vectors.json`; regenerate and review the diff.
