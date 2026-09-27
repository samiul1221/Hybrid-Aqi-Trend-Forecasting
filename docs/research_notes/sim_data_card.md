# Simulator data card (virtual nodes) [S10]

Generated 2026-09-27 01:42 UTC by `python -m aqi_ml.sim.validate`. Sim `sim-v1` (sha256:32290c471665…), seed 20260926, calibration `default-kappa-0.4-rdust-0.5-v1`, schema `sha256:49657c558950…`, base table `sha256:ba28f343a7f1…`, git `9323133f`.

- Stations: 30; spatial hold-out: patparganj_delhi_dpcc, rohini_delhi_dpcc, sector_125_noida_uppcb, sirifort_delhi_cpcb, sonia_vihar_delhi_dpcc
- Records: 4,054,176 unit-hours in 90 virtual units; split share {'train': 0.591, 'test': 0.288, 'val': 0.121}
- pm_state share: {'ok': 0.469, 'unavailable': 0.315, 'fallback_model_b': 0.163, 'filled_le_1h': 0.053} (CPCB truth gaps = 52.5%, simulated SPS30 outages = 0.32%)

## PM calibration check vs Delhi field data [IITD2025]

Target: winter hourly raw SPS30 vs BAM slope (1.46, 1.86), R² (0.85, 0.95). Units inside the slope range: 79% of 87.

slope median 1.66 (P10-P90 1.50-1.88); R² median 0.85 (P10-P90 0.80-0.90)

## Signal checks (ok hours, correlation of the model feature with its physical driver)

| feature vs driver | r |
|---|---|
| mq7_n vs ln CO | -0.74 |
| mq135_n vs ln CO | -0.64 |
| voc_n vs ln VOC | -0.77 |
| nox_n vs ln(1+(NO2+3.5 O3)/10) | +0.72 |
| ln_lux vs SW (day) | +0.84 |
| t vs truth | +1.00 |
| rh vs truth | +0.99 |

## Rates

- dust flag (R_dust=0.5) by season: {'monsoon': 0.132, 'post': 0.074, 'pre': 0.398, 'winter': 0.037}
- corrected/true PM2.5 median by season: {'monsoon': 0.6, 'post': 0.81, 'pre': 0.75, 'winter': 0.75}
- gas truth source share (CO; 0 meas, 1 interp, 2 station clim, 3 network clim): {0: 0.51, 2: 0.409, 1: 0.081}
- AQI valid share: 0.437

## Raw channel ranges (P1 / P50 / P99)

| channel | P1 | P50 | P99 |
|---|---|---|---|
| pm25_corr | 1.422 | 45.38 | 401 |
| pm10_corr | 4.026 | 62.22 | 443 |
| t | 6.74 | 26.99 | 41.43 |
| rh | 17.78 | 69.12 | 100 |
| p | 968.1 | 981.7 | 995.1 |
| wind_speed | 0 | 1.442 | 4.716 |
| lux | 0 | 2296 | 9.823e+04 |
| mq7_rs | 1133 | 1.343e+04 | 1.639e+05 |
| mq135_rs | 1882 | 3.041e+04 | 5.736e+05 |
| sraw_voc | 1.778e+04 | 3.041e+04 | 4.357e+04 |
| sraw_nox | 8455 | 2.029e+04 | 3.393e+04 |

Interpretation and parameter sources: docs/research_notes/simulator_parameters.md, ADR 0007.
