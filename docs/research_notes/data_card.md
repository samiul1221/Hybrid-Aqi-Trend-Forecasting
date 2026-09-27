# Data card — base table [S09]

Generated 2026-09-27 01:39 UTC by `python -m aqi_ml.data.build card`. Regenerate after every data build; numbers below come from the parquet files.

## Sources
- CPCB stations via OpenAQ v3 (raw measurements, provider regex `(?i)cpcb|caaqm|central pollution`). Providers seen in the bbox: n/a.
- Weather: Open-Meteo archive, model `era5`, timezone=GMT, aligned to IST hours (ADR 0005). Station met used when present and plausible (`*_src`).
- Attribution: OpenAQ (CC BY 4.0 platform terms; data licences per provider), Open-Meteo (CC BY 4.0), ERA5 (Copernicus C3S).
- Period: 2018-01-01 .. 2026-08-31; bbox [76.84, 28.4, 77.35, 28.88].
- CCR fallback: stations/periods missing in OpenAQ can be exported manually from https://airquality.cpcb.gov.in/ccr (15-min or hourly CSV) and loaded with the same columns.

## Stations (33)

| station | name | lat | lon | elev m | PM2.5 from | to | PM2.5 % | PM10 % | NO2 % | CO % | IST-aligned PM2.5 periods % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| alipur_delhi_dpcc | Alipur, Delhi - DPCC | 28.8153 | 77.1530 | 215 | 2018-12-14 | 2026-08-31 | 30 | 30 | 31 | 30 | 100 |
| anand_vihar_delhi_dpcc | Anand Vihar, Delhi - DPCC | 28.6488 | 77.3156 | 207 | 2018-01-01 | 2026-08-31 | 29 | 29 | 29 | 28 | 100 |
| arya_nagar_bahadurgarh_hspcb | Arya Nagar, Bahadurgarh - HSPCB | 28.6701 | 76.9254 | 213 | 2019-02-12 | 2026-08-31 | 21 | 21 | 25 | 25 | 100 |
| aya_nagar_delhi_imd | Aya Nagar, Delhi - IMD | 28.4707 | 77.1099 | 270 | 2018-03-13 | 2026-08-31 | 31 | 33 | 35 | 33 | nan |
| bawana_delhi_dpcc | Bawana, Delhi - DPCC | 28.7762 | 77.0511 | 219 | 2018-08-10 | 2026-08-31 | 33 | 33 | 34 | 33 | 100 |
| burari_crossing_delhi_imd | Burari Crossing, Delhi - IMD | 28.7257 | 77.2012 | 208 | 2018-03-21 | 2026-08-31 | 20 | 22 | 21 | 21 | nan |
| crri_mathura_road_delhi_imd | CRRI Mathura Road, Delhi - IMD | 28.5512 | 77.2736 | 218 | 2018-03-09 | 2026-08-31 | 35 | 35 | 36 | 37 | nan |
| dr_karni_singh_shooting_range_delhi_dpcc | Dr. Karni Singh Shooting Range, Delhi - DPCC | 28.4986 | 77.2648 | 251 | 2018-08-10 | 2026-08-31 | 30 | 30 | 30 | 30 | 100 |
| dtu_delhi_cpcb | DTU, Delhi - CPCB | 28.7500 | 77.1113 | 216 | 2018-03-09 | 2026-08-31 | 36 | 37 | 40 | 40 | nan |
| dwarka_sector_8_delhi_dpcc | Dwarka-Sector 8, Delhi - DPCC | 28.5710 | 77.0719 | 217 | 2018-08-10 | 2026-08-31 | 35 | 35 | 35 | 35 | 100 |
| ito_delhi_cpcb | ITO, Delhi - CPCB | 28.6302 | 77.2452 | 208 | 2018-03-14 | 2026-08-27 | 35 | 32 | 38 | 37 | nan |
| jawaharlal_nehru_stadium_delhi_dpcc | Jawaharlal Nehru Stadium, Delhi - DPCC | 28.5803 | 77.2338 | 209 | 2018-08-10 | 2026-08-31 | 33 | 33 | 34 | 32 | 100 |
| lodhi_road_delhi_imd | Lodhi Road, Delhi - IMD | 28.5918 | 77.2273 | 214 | 2018-03-09 | 2026-08-31 | 31 | 32 | 32 | 33 | nan |
| major_dhyan_chand_national_stadium_delhi_dpcc | Major Dhyan Chand National Stadium, Delhi - DPCC | 28.6113 | 77.2377 | 213 | 2018-08-10 | 2026-08-31 | 32 | 32 | 33 | 31 | 100 |
| mundka_delhi_dpcc | Mundka, Delhi - DPCC | 28.6847 | 77.0766 | 213 | 2018-12-14 | 2026-08-31 | 31 | 31 | 32 | 31 | 100 |
| najafgarh_delhi_dpcc | Najafgarh, Delhi - DPCC | 28.5702 | 76.9338 | 213 | 2018-12-14 | 2026-08-31 | 29 | 29 | 30 | 28 | 100 |
| narela_delhi_dpcc | Narela, Delhi - DPCC | 28.8228 | 77.1020 | 215 | 2018-08-10 | 2026-08-31 | 33 | 33 | 34 | 32 | 100 |
| north_campus_du_delhi_imd | North Campus, DU, Delhi - IMD | 28.6574 | 77.1585 | 233 | 2018-03-12 | 2026-08-31 | 32 | 33 | 34 | 33 | nan |
| nsit_dwarka_delhi_cpcb | NSIT Dwarka, Delhi - CPCB | 28.6091 | 77.0325 | 216 | 2018-03-09 | 2026-08-31 | 37 | 24 | 39 | 39 | nan |
| patparganj_delhi_dpcc | Patparganj, Delhi - DPCC | 28.6238 | 77.2872 | 205 | 2018-08-10 | 2026-08-31 | 33 | 33 | 34 | 16 | 100 |
| punjabi_bagh_delhi_dpcc | Punjabi Bagh, Delhi - DPCC | 28.6671 | 77.1301 | 220 | 2018-03-09 | 2018-05-13 | 0 | 0 | 0 | 0 | nan |
| pusa_new_delhi_imd | Pusa, New Delhi - IMD | 28.6103 | 77.0997 | 217 | 2018-03-09 | 2018-05-14 | 1 | 1 | 1 | 1 | nan |
| r_k_puram_new_delhi_dpcc | R K Puram, New Delhi - DPCC | 28.5646 | 77.1670 | 238 | 2018-03-10 | 2018-05-12 | 0 | 0 | 0 | 0 | nan |
| rohini_delhi_dpcc | Rohini, Delhi - DPCC | 28.7325 | 77.1199 | 214 | 2018-08-10 | 2026-08-31 | 32 | 33 | 33 | 33 | 100 |
| sector_125_noida_uppcb | Sector - 125, Noida - UPPCB | 28.5448 | 77.3231 | 201 | 2018-03-09 | 2026-08-31 | 36 | 36 | 35 | 32 | nan |
| sector_16a_faridabad_hspcb | Sector- 16A, Faridabad - HSPCB | 28.4088 | 77.3099 | 201 | 2018-03-09 | 2022-10-31 | 18 | 0 | 9 | 15 | nan |
| sector_30_faridabad_hspcb | Sector 30, Faridabad - HSPCB | 28.4417 | 77.3217 | 201 | 2020-03-11 | 2026-08-31 | 13 | 12 | 13 | 14 | 100 |
| sector_51_gurugram_hspcb | Sector-51, Gurugram - HSPCB | 28.4227 | 77.0667 | 233 | 2020-03-11 | 2026-08-31 | 19 | 19 | 20 | 20 | 100 |
| shadipur_delhi_cpcb | Shadipur, Delhi - CPCB | 28.6515 | 77.1473 | 223 | 2018-03-09 | 2026-08-31 | 39 | 26 | 41 | 41 | nan |
| sirifort_delhi_cpcb | Sirifort, Delhi - CPCB | 28.5504 | 77.2159 | 228 | 2018-03-09 | 2026-08-31 | 34 | 38 | 38 | 39 | nan |
| sonia_vihar_delhi_dpcc | Sonia Vihar, Delhi - DPCC | 28.7105 | 77.2495 | 208 | 2018-08-10 | 2026-08-31 | 33 | 33 | 33 | 31 | 100 |
| sri_aurobindo_marg_delhi_dpcc | Sri Aurobindo Marg, Delhi - DPCC | 28.5313 | 77.1902 | 237 | 2018-12-14 | 2026-08-31 | 30 | 30 | 31 | 31 | 100 |
| vivek_vihar_delhi_dpcc | Vivek Vihar, Delhi - DPCC | 28.6723 | 77.3153 | 205 | 2018-12-14 | 2026-08-31 | 31 | 31 | 32 | 31 | 100 |

## Coverage by variable (mean over stations, % of IST hours in period)

| variable | coverage % |
|---|---|
| pm25 | 28 |
| pm10 | 27 |
| pm1 | 0 |
| no2_ppb | 28 |
| no_ppb | 13 |
| nox_ppb | 0 |
| o3_ppb | 29 |
| so2_ppb | 23 |
| nh3_ppb | 0 |
| co_ppm | 28 |
| benzene_ugm3 | 0 |
| t | 100 |
| rh | 100 |
| p | 100 |
| wind_u | 100 |
| wind_v | 100 |
| td | 100 |
| precip | 100 |
| swrad | 100 |
| blh | 94 |

## Weather source share (fraction of hours)

|  | t_src | rh_src | p_src | wind_src |
|---|---|---|---|---|
| om | 0.899 | 0.9 | 1.0 | 0.94 |
| stn | 0.101 | 0.1 | 0.0 | 0.06 |

## PM2.5 distribution by season (µg/m³, cleaned, all stations)

| season | 0.05 | 0.5 | 0.95 |
|---|---|---|---|
| monsoon | 11.0 | 34.2 | 86.0 |
| post-monsoon | 28.0 | 132.0 | 387.5 |
| pre-monsoon | 17.5 | 58.9 | 170.0 |
| winter | 47.2 | 138.0 | 358.5 |

## Cleaning (values set to NaN, all stations) [S09]

| rule | count |
|---|---|
| co_ppm_flatline | 7752 |
| no2_ppb_flatline | 5285 |
| no_ppb_flatline | 790 |
| o3_ppb_flatline | 3323 |
| pm10_flatline | 1476 |
| pm10_hampel | 12107 |
| pm10_over_max | 190 |
| pm25_flatline | 4579 |
| pm25_gt_pm10 | 11334 |
| pm25_hampel | 14608 |
| pm25_over_max | 409 |
| so2_ppb_flatline | 2289 |

## Unit-label repairs (ADR 0006)

| parameter | label | used | groups | values | median_raw |
|---|---|---|---|---|---|
| co | ppb | mg/m³ | 36 | 705200 | 0.925000011920929 |
| co | µg/m³ | mg/m³ | 5 | 26790 | 1.2999999523162842 |
| nox | ppb | ppm | 2 | 23751 | 0.04465 |

## Known limitations

- `blh` (boundary layer height) is not served by the Open-Meteo archive/historical-forecast for Delhi (verified 2026-09-26): almost always null (see coverage table).
- No imputation in the base table; gaps stay NaN (feature builder applies the <=2 h rule).
- OpenAQ S3 archive has no Delhi data for 2023-2024; NOx not used (inconsistent with NO + NO2); no NH3/benzene (ADR 0006).
- Historical-forecast (teacher covariates) is stitched from the first hours of successive runs, not forecasts as issued at base_ts (ADR 0005).
