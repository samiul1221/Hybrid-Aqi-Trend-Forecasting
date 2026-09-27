# 0006 — CPCB data from the keyless OpenAQ S3 archive ([S09])

Status: accepted (2026-09-27)

## Context
No OpenAQ API key was available. OpenAQ also publishes its data on AWS
(`s3://openaq-data-archive`, anonymous HTTPS), one gzip CSV per location-day.

## Verified facts (2026-09-26/27)
- Delhi-NCR reference monitors sit in OpenAQ id ranges 5530–5640, 6920–6970, 8460–8480, 10480–10495,
  10820–10835 and 235 (found via explore.openaq.org, each confirmed by coordinates inside the bbox):
  35 ids → 33 stations (Anand Vihar 235 + 10487 and ITO 5613 + 10489 are the same sites).
- `datetime` is the **end** of the averaging period: a CPCB day file holds 15-min values labelled
  00:15 … 24:00 IST; `+05:30` offsets. Some years are hourly.
- **No files exist for 2023–2024** at any Delhi station; data cover 2018–2022 and 2025–2026.
- Unit labels are wrong in places: 2025+ `co "ppb"` values ≈ 1.5 are mg/m³ (CPCB's native unit);
  `nox "ppb"` values ≈ 0.05 are ppm. CPCB NOx is < NO + NO2 in most hours even after repair.
- Station T/RH/wind appear only in recent years; NH3 and benzene are absent.

## Decisions
1. `aqi_ml.data.openaq_archive` + `build archive|all-archive`: probe candidate ids, keep bbox, merge
   same-name ids within 1 km, mirror raw files byte-for-byte under `data/raw/openaq_archive/` (I08).
2. Periods from the modal timestamp spacing, labels = period end; alignment reuses `align.intervals_to_ist`.
3. Gas unit repair: per (station, gas, unit label, year) group, if the median is implausible in the
   canonical unit, re-read with the first plausible unit (CO: mg/m³ first; others: ppm first). Every
   repair is logged in `data/interim/openaq/_unit_repairs.json` and the data card.
4. NOx is not downloaded (inconsistent; the simulator uses NO2 and O3).
5. Splits move to train 2018–2021, val 2022, test 2025-01 … 2026-08 ([S10.2] "adjust to availability").
   The API path (P3) can fill 2023–2024 later with a key; nothing else changes.
6. Memory: a full run with 16 workers was stopped by the OS for low memory; the step now uses compact
   dtypes, 8 workers and resumes per station.

## Addendum (2026-09-27): 15-min snapshots and coverage
- 2021-2022 files hold **one 15-min value every ~2 h** (labels :15). Inferring the period from the
  spacing (2 h, capped at 1 h) plus a 0.75 hour-coverage rule dropped almost all of 2021-22 (2022, the
  validation year, came out empty). Now: a label that is not a full IST hour is a native 15-min value;
  full-hour labels use the sensor's modal spacing. Minimum hour coverage is 0.25 (one 15-min value) —
  CPCB PM is a BAM hourly value repeated over its 15-min slots, so one slot represents the hour.
  `pm25_cov`/`pm10_cov` (0.25-1.0) are kept in the base table; ~19 % of PM hours have coverage 0.25.
- Result: PM2.5 station-hours per year 2018: 47k, 2019: 101k, 2020: 127k, 2021: 72k, 2022: 42k,
  2025: 182k, 2026 (to Aug): 137k. Punjabi Bagh, Pusa and R K Puram only have spring 2018 data and
  are below the simulator's minimum (8760 PM hours).
