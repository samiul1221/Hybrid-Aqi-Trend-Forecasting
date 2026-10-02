# Graph data card — graph-v1 (2026-10-02)

Built by `uv run python -m aqi_ml.data.graph_windows` from `data/datasets/feat-v1` (ADR 0012, 0019, 0020).

| item | value |
|---|---|
| stations N | 30 (25 train graph + 5 spatial hold-out: patparganj, rohini, sector_125_noida, sirifort, sonia_vihar) |
| hours T | 75,960 IST hours (2018-03 … 2026-08; 2023–24 archive gap is empty, ADR 0006) |
| units | 0, 1, 2 (one sensor "reality" per unit index; same truth, different simulated sensors) |
| base hours per unit | train 19,289 · val 2,761 (2022 is sparse in the archive) · test 12,560 |
| measured PM2.5 (node-hours) | 27.8 % (dominated by missing CPCB truth, not sensor faults) |
| node features | 23 student channels (sensor domain) + 7 Open-Meteo ERA5 past = 30 |
| future covariates | 7 ERA5 at t+1..t+6 + hod sin/cos = 9 — **oracle weather** (actuals, not issued forecasts): limitation |
| targets | ln-change of TRUE CPCB PM2.5/PM10, h1..h6 (inputs never contain true PM, ADR 0019) |
| G1 sensor-domain check | 182 student windows equal to the teacher grid bit-for-bit |
| size | 81 MB common + 3 × 250 MB units |
