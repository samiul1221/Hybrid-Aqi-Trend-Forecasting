# Simulator parameters — where every number comes from

The values are written in `src/sensors.py` (section 1, `PARAMS`); notebook `02_simulator.ipynb` runs the
simulator and checks the SPS30 against the IIT-Delhi field study (winter slope 1.46–1.86).
`TODO_VERIFY` = not yet confirmed by a primary source or by our own hardware; check during the first real
week / co-location. Section numbers like [S10.3] refer to `docs/specs/edge_implementation_plan.md`.

## Sources

| key | reference | used for |
|---|---|---|
| Mandariya2024 | A. K. Mandariya et al., "Hygroscopicity of size-selected aerosol particles in the heavily polluted urban atmosphere of Delhi", ACP 24:3627–3647, 2024, doi:10.5194/acp-24-3627-2024 | Delhi κ: 100/150/200 nm = 0.14±0.04 / 0.18±0.06 / 0.22±0.07, range 0.05–0.56; events: chloride 0.35, biomass 0.18, clean 0.27 (winter, Feb–Mar 2020, HTDMA at 90 % RH) |
| Arub2020 | Arub et al. 2020 (cited in Mandariya2024) | Delhi PM1 κ ≈ 0.13–0.77 — TODO_VERIFY primary text |
| Crilley2018 | L. R. Crilley et al., AMT 11:709–720, 2018, doi:10.5194/amt-11-709-2018, Eq. 6 | growth factor C = 1 + (κ/1.65)/(−1 + 1/a_w), 1.65 g cm⁻³ dry density — confirms [S06.4] |
| Crilley2019 | L. R. Crilley et al., AMT 13:1181–1193, 2020, doi:10.5194/amt-13-1181-2020 | literature κ gives OPC PM within 50 % of reference; in-situ κ best |
| IITD2025 | G. Kumar, P. Kumar D., J. Dhariwal, S. Srirangarajan, "Reliability Assessment of Low-Cost PM Sensors under High Humidity and High PM Level Outdoor Conditions", arXiv:2504.06615, 2025 | **Delhi winter (Nov 2021–Mar 2022), SPS30 vs BAM 1022**: hourly slope 1.46–1.86, R² 0.69–0.92 (0.91–0.92 for healthy units), inter-unit CV 3.5 %; bias grows with RH, drops > 90 % RH (BAM heater effect); PM2.5 6–611 µg m⁻³, RH 25–95 % |
| SCAQMD | South Coast AQMD AQ-SPEC, "Field Evaluation Sensirion SPS30 Evaluation Kit" (2019) | GRIMM PM2.5 ≈ 0.69 × SPS30 (SPS30 ≈ 1.45× high), R² ~0.80; PM10 R² 0.07–0.12, underestimated; inter-unit spread 1–2.4 % |
| Kuula2020 | J. Kuula et al., AMT 13:2413–2423, 2020, doi:10.5194/amt-13-2413-2020, Table 2 | SPS30 bins 2–4 (nominal 1–10 µm) all respond only to 0.7–1.3 µm particles → PM4/PM10 are extrapolated |
| Gautam2025 | P. Gautam et al., Atmosphere 16:502, 2025, doi:10.3390/atmos16050502 | SPS30 does not size coarse particles; in a dust storm SPS30 slope ≈ 0.29 vs reference for PM2.5 and PM10, PM10/PM2.5 ≈ 1.6 — **TODO_VERIFY** (values from an automated page query, check Table/Fig in the PDF) |
| SPS30-DS | Sensirion SPS30 datasheet v0.9 (Nov 2018) | accuracy ±10 µg m⁻³ (0–100) / ±10 % (100–1000); range 0–1000 µg m⁻³; operating RH 0–95 %; best 20–80 % RH |
| SGP41-DS | Sensirion SGP41 datasheet v1.0 (Dec 2021), Table 1, Figs 2–3 | SRAW ∝ log R; VOC −870 (−710…−1030) ticks per doubling of ethanol (5000→10000 ppb); NOx +1480 (+950…+2010) per doubling of NO2 (150→300 ppb); O3 raises the NOx pixel ~2 doublings more than NO2 at equal ppb; oxidants raise the VOC pixel (~+1700 at 500 ppb); RH 0–90 % non-condensing; NOx spec reached < 7 d after first power-on |
| MQ7-W | Winsen MQ-7 manual v1.3 (2014) | 1.5 V / 90 s low phase; α = Rs(300)/Rs(50) ≤ 0.6; Fig. 3 CO slope ≈ −0.5 (log–log); Fig. 4 ≈ −0.011 ln/°C, ≈ −0.005 ln/%RH; preheat ≥ 48 h |
| MQ7-H | Hanwei MQ-7 technical data | 1.4 V / 90 s; Rs 2–20 kΩ at 100 ppm CO (RL 10 kΩ); Rs(300)/Rs(100) < 0.5; Fig. 3 slope 50→4000 ppm ≈ −0.66; H2 curve close to CO |
| MQ135-W | Winsen MQ135 manual v1.4 (2015) | heater 5 V; Fig. 3 slopes: H2 −0.48, NH3 −0.48, toluene −0.43; Fig. 4 T/RH as MQ-7 |
| NakamuraMahrt2005 | R. Nakamura, L. Mahrt, J. Atmos. Oceanic Technol. 22(7), 2005 | naturally ventilated shield errors 0.5–2.5 °C, largest at high SW and low wind |
| Littlefair1985 | P. J. Littlefair, Lighting Res. Technol. 17(4), 1985 | global luminous efficacy ≈ 105 lm W⁻¹ |
| Li1997 | D. H. W. Li et al., ANZASCA 1997 | yearly mean global efficacy 110.7 lm W⁻¹ (100–140 for 80 % of the year) |
| BH1750-DS | ROHM BH1750FVI datasheet | 1–65535 lx at default MTreg; lx = counts/1.2 × 69/MTreg (MTreg 31 → ≈121.6 klx); accuracy ×1.2 ±20 % |
| HDC1080-DS | TI HDC1080 datasheet | ±0.2 °C, ±2 %RH typical |
| BMP280-DS | Bosch BMP280 datasheet | absolute ±1 hPa, relative ±0.12 hPa |
| Novelli1999 | P. C. Novelli et al., JGR 104(D23), 1999 | tropospheric H2 ≈ 0.53 ppm (reducing-gas background for MQ sensors) |
| Baker2008 | A. K. Baker et al., Atmos. Environ. 42:170–182, 2008 | urban NMHC/CO emission ratios → VOC ∝ CO — TODO_VERIFY magnitude |

## Parameter table

| parameter | value / range | equation | source | notes |
|---|---|---|---|---|
| aerosol κ(t) | AR(1), mean 0.30, sd 0.10, clip 0.05–0.77, τ = 48 h | C = 1 + (κ/1.65)/(1/a_w − 1) | Mandariya2024, Arub2020, Crilley2018 | property of the air, shared by units at a station (spec drew κ per unit); τ TODO_VERIFY |
| SPS30 scale s | U(1.15, 1.55) | PM2.5 = s·(fine·C + e·dust) | IITD2025, SCAQMD | jointly calibrated with ΔT (see card) |
| SPS30 internal warming ΔT | U(1.5, 3.5) °C | RH_sps from dew point at T + ΔT | IITD2025 R² 0.91 needs it (0.65 without) | TODO_VERIFY on a node |
| forward-model a_w cap | 0.97 | | | device clamps 0.95 [S06.4] |
| dust PM2.5 fraction | 0.25 × (PM10−PM2.5) | dust25 = 0.25·coarse | typical 0.2–0.35 | TODO_VERIFY |
| dust efficiency e | U(0.20, 0.40) | | Gautam2025 (0.29) | TODO_VERIFY |
| PM10 extension q | U(0.02, 0.10) | PM10 = PM2.5·(1+q) + s·f·coarse | Kuula2020 | |
| coarse visibility f | U(0.05, 0.25) | | Kuula2020, SCAQMD | spec U(0.3,0.7) is for other OPCs |
| PM1 ratio r | U(0.75, 0.95) | PM1 = r·s·fine·C + 0.3·s·e·dust | | TODO_VERIFY |
| PM noise | N(0, 0.05·PM + 1) | | [S10.3] | |
| RH>95 bias | +U(0, 20) % | | [S10.3] | |
| missing / outages | 1 % hours; 1 yr⁻¹ × U(6, 48) h | | [S10.3] | outages feed Model B |
| T bias / noise | U(−0.5, 0.5) / 0.3 °C | T = T_true + b + ΔT_rad + ε | [S10.3], HDC1080-DS | |
| radiation error | c ∈ U(1.0, 2.5) | ΔT_rad = c·SW/1000/(1 + k·ws) | NakamuraMahrt2005 | spec had no wind term |
| RH | bias U(−2, 2), noise 2 % | RH from true Td at sensor T | [S10.3], HDC1080-DS | |
| pressure | bias U(−1, 1), noise 0.1, height U(3, 25) m | P = P_true − h/8.3 + b + ε | BMP280-DS, hydrostatics | spec b U(−3,3) split into physics |
| wind | k U(0.4, 1.0), noise 0.3, stall 0.3 m/s, dir ±15° noise, offset U(−15, 15)° | | [S10.3] | offset = vane alignment, TODO_VERIFY |
| rain | wet > 0.1 mm h⁻¹, dries U(1, 2) h, false wet 2 % if RH > 90 | | [S10.3] | |
| lux | K = 108 lm W⁻¹, g U(0.6, 1.2), β U(0, 0.2), clip 121 557 | lux = g·K·SW·e^(−β·PM/100) | Littlefair1985, Li1997, BH1750-DS | clip reflects MTreg auto-range [S06.1] |
| MQ-7 | R0 ln(20 kΩ, 0.5), b U(−0.70, −0.45), c0 U(0.3, 1.0) ppm, gT U(−0.016, −0.007), gRH U(−0.008, −0.003), drift ±0.001 d⁻¹, noise 0.05 | ln Rs = ln R0 + b ln(CO + c0) + gT(T−20) + gRH(RH−65) + drift·d + ε | MQ7-W, MQ7-H, Novelli1999 | c0 = H2 background (spec 0.1) |
| MQ-135 | b U(−0.55, −0.40), w_CO U(0.1, 0.5), w_VOC U(0.5, 1.5), g_ox U(0, 0.15) | ln Rs = … b ln(w_CO·CO + w_VOC·VOC + c0) + g_ox ln(1 + NO2/20) | MQ135-W | NO2 is oxidising → raises Rs (spec added it to reducing gases); no NH3 in archive |
| SGP41 VOC | base U(25k, 35k), k U(1040, 1507), v0 100 ppb, k_ox U(0, 700) | SRAW = base − k ln(1 + VOC/v0) + k_ox ln(1 + ox/50) | SGP41-DS | k gives exactly −710…−1030 per doubling in the datasheet window |
| SGP41 NOx | base U(14k, 17k), k U(1437, 3039), n0 10 ppb, w_O3 U(2, 5) | SRAW = base + k ln(1 + (NO2 + w·O3)/n0) | SGP41-DS | spec k U(200, 800) is ~3× below the datasheet |
| VOC truth | 20 + U(0.03, 0.10)·CO[ppb] ppb | per station | Baker2008 | TODO_VERIFY; spec's mixed-unit proxy replaced |
| gas truth gaps | ≤ 3 h interpolate; else station month×hour median; else network | `truth_*_src` column | | inputs only, never targets (I07) |
