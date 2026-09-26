# 0005 — Data sources and IST alignment ([S09], cloud [C07])

Status: accepted (2026-09-26)

## Verified facts (live API probes, 2026-09-26, Delhi 28.63 N 77.22 E)
- Open-Meteo variable names in [S09] are correct; units: °C, %, hPa, mm, W/m², m; wind speed needs
  `wind_speed_unit=ms` (default is km/h).
- **`timezone=Asia/Kolkata` is wrong for us:** the value at 01:00Z (06:30 IST) is labelled `06:00`
  local, i.e. timestamps shift by +5 h and drop the half hour. We always request `timezone=GMT`
  and align ourselves.
- `boundary_layer_height` is null for Delhi in the archive and historical-forecast APIs for every
  model tried (best_match, era5, era5_land, era5_seamless, gfs_seamless, ecmwf_ifs025) — ~1 % of
  hours non-null. The column is kept and reported in the data card; it matters only for the
  phase-2 teacher.
- Archive models disagree strongly (same hour: best_match 4.5 °C, era5_land 5.8, era5 8.1,
  gfs 11.1). For recent dates `best_match` equals the historical-forecast API output.
- OpenAQ v3 needs `X-API-Key`; limits 60/min, 2000/h; location 8118 "New Delhi" is AirNow (US
  Embassy), not CPCB, so stations are filtered by provider name.

## Decisions
1. **Weather truth for the simulator = Open-Meteo archive with `models=era5`** (one consistent
   0.25° reanalysis for 2018→present), not `best_match` (mixes products over time).
2. **OpenAQ raw `/measurements`, not `/hours`**, then our own IST alignment, so CPCB's native
   IST From–To hours pass through unchanged and 15-min or UTC-hour data are re-binned the same way.
   The data card reports the share of periods that were already IST hours.
3. **Alignment rules** (`aqi_ml.data.align`):
   - interval data → time-weighted overlap mean, bucket kept if coverage ≥ 0.75 (CPCB 75 % rule);
   - Open-Meteo accumulated (precipitation sum, shortwave mean over (T−1h, T]) → proportional
     overlap = mean of the two overlapping hours (spec rule);
   - Open-Meteo instantaneous → **exact mean of the linearly interpolated series over the IST hour**
     = 1/8·v(T−1h) + 3/4·v(T) + 1/8·v(T+1h), bucket centred on T. *Deviation:* [S09] says "average
     the two bracketing UTC values", which estimates the value at the bucket **start** (30 min
     before its centre). The bucket is an hourly mean like CPCB data, so the interval mean is used.
   - wind → vector mean of u, v; speed/direction derived from the means ([S06.2]).
4. **Station weather preferred when present and plausible** ([S09]); per-variable `*_src` column
   records `stn`/`om`. Dew point is recomputed from the chosen T and RH (Magnus) so T/RH/Td agree.
5. **Historical-forecast API** is downloadable (`build hforecast`, from 2022) for the teacher, but it
   is stitched from the first hours of successive runs, not forecasts as issued at `base_ts`.
   Honest lead-time covariates (Open-Meteo Previous Runs API) are a phase-2 item (cloud CI04).
6. No imputation in the base table; the ≤ 2 h interpolate+mask rule lives in the feature builder (P5).
