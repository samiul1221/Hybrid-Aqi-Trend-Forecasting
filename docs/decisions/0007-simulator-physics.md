# 0007 — Simulator physics deviations from [S10.3] and calibration defaults

Status: accepted (2026-09-27). Evidence: `docs/research_notes/simulator_parameters.md`;
check results: `docs/research_notes/sim_data_card.md`.

The spec's forward models are generic low-cost-sensor models. Our sensors' datasheets and Delhi field
data contradict several of them. Each change below is backed by a source; the result is checked against
an independent Delhi co-location (IITD2025).

| # | Spec [S10.3] | Now | Why |
|---|---|---|---|
| 1 | κ_u ~ U(0.2, 0.7) per unit | κ(t) per station, AR(1) mean 0.30, sd 0.10, 0.05–0.77 | κ is an aerosol property; Delhi HTDMA 0.05–0.56 (Mandariya2024), PM1 0.13–0.77 (Arub2020) |
| 2 | RH seen by SPS30 = ambient | RH at T + ΔT_u, ΔT_u ~ U(1.5, 3.5) °C | ambient-RH growth gives R² 0.65 vs field 0.91; warming inside the sensor reproduces slope and R² together |
| 3 | s_u ~ U(0.85, 1.25) | U(1.15, 1.55) | SPS30 reads 1.45× (SCAQMD) to 1.5–1.9× (IITD2025, incl. growth) high |
| 4 | PM10 = PM2.5 + f·s·coarse, f ~ U(0.3, 0.7) | dust mostly invisible (e ~ U(0.2, 0.4)); PM10 = PM2.5·(1+q) + s·f·coarse, f ~ U(0.05, 0.25) | SPS30 bins > 1 µm only see 0.7–1.3 µm (Kuula2020); PM10 R² 0.07 (SCAQMD); dust slope ≈ 0.29 (Gautam2025) |
| 5 | T error 0.8·SW/1000 | c·SW/1000/(1 + ws_local), c ~ U(1, 2.5) | shield errors depend on wind (NakamuraMahrt2005) |
| 6 | P bias U(−3, 3) | −h/8.3 + U(−1, 1), h ~ U(3, 25) m | same total, split into hydrostatics + datasheet accuracy |
| 7 | lux clip 65535 | 121 557 | firmware auto-ranges MTreg on saturation [S06.1]; otherwise midday lux carries no information |
| 8 | MQ offset +0.1 ppm | c0 ~ U(0.3, 1.0) ppm | MQ-7 responds to H2 like CO; air holds ~0.53 ppm H2 (Novelli1999) |
| 9 | MQ-7 b U(−0.75, −0.55), gT U(−0.02, 0), gRH U(−0.01, 0) | b U(−0.70, −0.45), gT U(−0.016, −0.007), gRH U(−0.008, −0.003) | read from Winsen/Hanwei Figs. 3–4; Winsen α ≤ 0.6 holds for all draws (test) |
| 10 | MQ-135: 10·NO2 inside the reducing-gas sum | NO2 raises Rs via g_ox·ln(1+NO2/20); NH3 dropped | NO2 is oxidising (n-type SnO2); NH3 not in the archive and ≪ MQ-135 range |
| 11 | SGP41 NOx kn U(200, 800), NO2 only | kn U(1437, 3039) incl. O3 (w U(2, 5)) | Table 1 / Fig. 3 of the SGP41 datasheet; ranges reproduce Table 1 exactly (test) |
| 12 | SGP41 VOC kv U(500, 2000) | U(1040, 1507) + oxidant term | Table 1 / Fig. 2 |
| 13 | voc_proxy mixing units | 20 ppb + U(0.03, 0.10)·CO[ppb] | co-emission with CO; TODO_VERIFY |

Calibration default (`shared/calibration/default.json`, I06):
- κ stays **0.4** (spec); it lies inside the Delhi range and co-location [S11.2] will fit it.
- **R_dust 1.5 → 0.5.** With an SPS30 the coarse ratio (PM10−PM2.5)/PM1 is ≈ 0.1–0.3 normally and ≈ 1
  in dust, so 1.5 would never fire. The simulated dust-flag rate is highest pre-monsoon, as expected.
- Consequence to note: before co-location the device's corrected PM is ≈ 0.6–0.8× true (κ 0.4 over-corrects
  the internally drier SPS30, and nothing removes the ×1.15–1.55 scale). The spec's calibration.json has
  `pm10_scale` but no PM2.5 scale; co-location should fit κ_eff and a PM2.5 scale (open question).

Also implemented: device processing is done per contiguous segment (a gap of more than an hour means
re-deployment); `sim_gap` marks CPCB truth gaps so they are never mistaken for sensor outages
(Model B uses `sim_outage` only).
