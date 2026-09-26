# EDGE NODE SPEC — Local AQI Trend Forecasting on ESP32-S3 (cloud-aware)

<doc_meta>
scope: EDGE PORTION ONLY. The cloud (graph teacher, server, dashboards) is a separate work item. This doc defines everything the edge needs, plus the exact contracts and guarantees the edge requires from the cloud so the two plug together later.
audience: engineer or coding agent with zero prior context
format_conventions:
  - Sections have stable IDs like [S07]. Cross-references use these IDs.
  - YAML blocks = machine-readable specs (treat as source of truth).
  - JSON blocks = wire formats / schemas.
  - "MUST" = required; "SHOULD" = strongly recommended; "MAY" = optional.
  - Numbers marked "(tune)" are starting values to be tuned on data.
  - Numbers marked "(verify)" come from memory/literature and must be checked against the primary source before release.
read_order: S00 → S01 → S02 → then build in milestone order [S22]
companion_doc: "cloud_implementation_plan.md (backend + cloud teacher + MLOps). Wire contracts in [S16]/[S17] are shared and MUST stay identical in both docs."
</doc_meta>

---

## [S00] HOW TO USE THIS DOCUMENT

1. Read [S01] invariants. They override anything else in the doc.
2. Read [S02] for the system picture and where the edge sits.
3. Build in the order of milestones in [S22]. Each milestone lists acceptance tests.
4. Anything the cloud must provide is in [S16] (wire contracts) and [S17] (integration guarantees). Do not invent cloud behaviour; if a contract is missing, stub it exactly as specified.

---

## [S01] INVARIANTS (MUST hold everywhere, edge and cloud)

```yaml
invariants:
  I01_aqi_standard: "India CPCB NAQI only. Never US-EPA. One shared implementation with shared test vectors (S06.6) used by firmware, training code, and cloud."
  I02_time_base: "All timestamps stored/transmitted in UTC ISO-8601. Hour buckets are IST clock hours [HH:00, HH:00+1h) IST = [HH-5:30 UTC ...). Record ts = UTC instant of the IST hour START (e.g. IST 10:00 → 04:30Z)."
  I03_why_IST_buckets: "CPCB reference data is hourly on IST clock hours. UTC-hour buckets would be offset 30 min from all labels."
  I04_prediction_target: "Hourly humidity-corrected PM2.5 and PM10; target = ln(1+PM[t+h]) - ln(1+PM[t]), h=1..3 on edge (cloud h=1..6, same definition). AQI trend is derived deterministically from predicted PM (S07)."
  I05_single_feature_schema: "feature_schema.yaml is the single source of truth for feature names/order/transforms/normalisation. Models embed its hash. Firmware refuses a model whose schema_hash != active schema."
  I06_single_calibration_source: "kappa, scales, dust threshold live in calibration.json (versioned). Every record carries calibration_version. Raw values are always stored so any party can reprocess."
  I07_no_self_training: "Never use values produced by our own models (filled PM, forecasts, imputations) as training targets. Only 'measured' values are targets."
  I08_raw_immutable: "Raw sensor values are immutable. Processed values are derived and versioned (pipeline_version)."
  I09_teacher_student_same_target: "Teacher (cloud) and students (edge) predict the same quantities in the same units and same transform, on the same hour buckets."
  I10_edge_autonomy: "Edge MUST keep measuring, computing AQI, forecasting and alerting with zero cloud connectivity indefinitely."
  I11_units: "PM µg/m3; T °C; RH %; P hPa; wind m/s and degrees FROM (meteorological); lux; gas sensors as dimensionless normalised raw signals (never ppm)."
  I12_gas_sensors_are_proxies: "MQ-7, MQ-135, SGP41 are model inputs only. They are never converted to concentrations and never used to compute AQI."
```

---

## [S02] SYSTEM CONTEXT

### S02.1 Roles

```yaml
system:
  nodes:
    count: "N nodes (target 5-10) across Delhi, spacing 1-5 km"
    device: ESP32-S3 N16R8
    responsibilities:
      - measure, clean, humidity-correct, flag
      - compute PM-based CPCB AQI locally
      - run local student model (RF or GDC-CNN): forecast PM T+1..T+3 → AQI trend
      - run no-PM fallback model (Model B) when PM sensor fails
      - compare with cloud forecast → local event / sensor check flags
      - push local_event / sensor_check / pm_sensor_fault immediately to cloud (POST /api/v1/events, S16.6)
      - receive cloud forecasts by MQTT push (S16.7); HTTP GET poll is fallback
      - alerts, display, store-and-forward, OTA
  cloud:
    role: teacher + aggregator (built separately)
    responsibilities_relevant_to_edge:
      - ingest node hourly records
      - wind-aware spatio-temporal graph model over all nodes (+ CPCB reference stations as optional extra nodes for training)
      - return per-node forecast T+1..T+6 with calibrated intervals
      - return gas sub-indices (NO2/O3/CO) from nearest reference station
      - return neighbour-based PM estimate for a node with failed PM
      - produce teacher outputs for distillation (offline store)
      - host training pipelines for students, model registry, OTA manifests
      - monitor drift per node
      - stream forecasts to nodes (MQTT) and forecasts/events/node status to the client frontend (WebSocket)
      - store edge-pushed events; re-check edge-vs-cloud disagreement server-side (cloud_confirmed)
  reference_data:
    CPCB: used for synthetic bootstrap, co-location calibration, optional validation, optional gas sub-indices
    runtime_dependency_on_CPCB: "none for edge; optional for cloud (gas sub-indices)"
```

### S02.2 Hourly data flow (timeline, IST)

```yaml
hourly_timeline:
  "HH:00:00": "IST hour H closes. hour_task builds record for hour H-1 (ts = start of H-1)."
  "HH:00:05": "Edge inference (Model A or B), AQI trend, local decision with last-known cloud forecast."
  "HH:00:10-HH:02:00": "Upload record (+backlog). Jittered start (random 0-110 s) to avoid thundering herd."
  "HH:02:00-HH:10:00": "Cloud waits for node records (deadline HH:08), runs graph model, writes forecasts."
  "HH:08:00-HH:10:00": "Cloud publishes forecast JSON to MQTT topic aqi/v1/nodes/{node_id}/forecast (QoS1, retained). Edge receives push, re-runs decision logic (S15)."
  "HH:12:00-HH:20:00": "Fallback: if no push for this base_ts by HH:12, edge polls GET /api/v1/forecast every 60 s until HH:20."
  "HH:20:00": "Cloud may re-publish an updated forecast (same based_on_ts, newer issued_at) if late node records arrived; edge replaces the older one."
  "any time": "On local_event / sensor_check / pm_sensor_fault the edge POSTs /api/v1/events immediately (does not wait for the hourly upload)."
  note: "Cloud forecast for hour H-1 base is 'fresh' only if cloud.based_on_ts == edge record ts."
```

### S02.3 Text diagram

```
[Sensors] → sensor_task → minute agg → hour_task ─┬→ AQI (CPCB, PM-based) ─→ display/alerts
                                                 ├→ features → Model A|B → PM T+1..3 → AQI trend
                                                 ├→ log (LittleFS) → comms_task → POST /ingest ─→ [Cloud]
                                                 └→ decision ← MQTT push forecast (GET poll fallback) ← [Cloud graph teacher]
                                                        └→ local_event → POST /events ─→ [Cloud] → WebSocket → [Client frontend]
[Cloud] → teacher_store (offline) → student training → model registry → OTA manifest → node
```

---

## [S03] GLOSSARY

```yaml
glossary:
  AQI_PM: "CPCB AQI computed from PM2.5 and PM10 only (24h means). Labelled 'PM-based'."
  AQI_full: "max(AQI_PM sub-indices, cloud-supplied NO2/O3/CO sub-indices). Only when cloud data fresh."
  teacher: "cloud graph model; sees all nodes + wind edges + forecast weather (privileged info)."
  student: "edge model (RF or GDC-CNN); sees only local sensors."
  LUPI: "Learning Using Privileged Information: teacher trained with info unavailable to student."
  KD: "knowledge distillation."
  out_KD: "student trained on (blend of) teacher predictions (Shmuel-style)."
  full_KD: "out_KD + feature distillation + attention-guided feature distillation (Na-style)."
  GDC: "Gated Dilated depthwise separable Convolution block (Na et al. 2026)."
  GLU: "Gated Linear Unit: A ⊙ sigmoid(B)."
  RevIN / instance norm: "per-window per-channel standardisation, reversed on output."
  Model_A: "student with PM inputs."
  Model_B: "student without PM inputs (fallback when PM sensor failed > 1 h)."
  kappa: "hygroscopicity parameter for humidity correction of optical PM."
  measured/filled/missing: "quality states of a value (S06.7)."
  base_ts: "UTC instant of the IST hour start the record/forecast is based on."
```

---

## [S04] HARDWARE

### S04.1 Bill of materials

```yaml
bom:
  mcu: {part: "ESP32-S3-DevKitC-1 N16R8", flash: 16MB, psram: 8MB octal, why: "same chip as Na et al.; RAM for buffers+2 models"}
  sensors:
    - {id: pm, part: "Sensirion SPS30", measures: "PM1, PM2.5, PM4, PM10 mass; number conc; typical size", range: "0-1000 µg/m3", iface: "UART SHDLC 115200 (SEL floating) | I2C 0x69 ≤100kHz", supply: 5V, role: "main AQI input"}
    - {id: th, part: "HDC1080", measures: "T, RH", iface: "I2C 0x40", mount: "inside radiation shield near SPS30 inlet, away from exhausts/heaters", role: "humidity correction + input"}
    - {id: baro, part: "BMP280", measures: "pressure", iface: "I2C 0x76|0x77", role: input}
    - {id: wind, part: "weather-meter kit (cup anemometer + vane)", measures: "speed (reed pulses), direction (resistor ladder)", iface: "GPIO interrupt + ADC", role: "input; cloud graph wind edges"}
    - {id: rain, part: "existing rain board", measures: "wet/dry (DO), optional AO", iface: GPIO, role: input}
    - {id: lux, part: "BH1750", measures: "lux 0-65535", iface: "I2C 0x23", role: "sunlight/mixing proxy (replaces LDR)"}
    - {id: mq7, part: "MQ-7", measures: "raw resistance (CO-like)", iface: "ADC via divider + heater MOSFET", role: "combustion proxy"}
    - {id: mq135, part: "MQ-135", measures: "raw resistance (general gases)", iface: "ADC via divider", role: "pollution proxy"}
    - {id: sgp41, part: "SGP41", measures: "SRAW_VOC, SRAW_NOX ticks", iface: "I2C 0x59", role: "combustion proxy + local-event confirmation"}
    - {id: rtc, part: "DS3231", measures: time, iface: "I2C 0x68", role: "offline timekeeping"}
  ui: [button, buzzer, optional small display]
  removed: {LDR: "no real unit", MiCS-6814: "cost; covered by MQ-7 + SGP41 NOx"}
  optional_recommended:
    - {part: INA219, why: "energy per inference (paper)", note: "default addr 0x40 CONFLICTS with HDC1080 → strap A0 high → 0x41"}
    - {part: "Plantower PMSA003", why: "backup PM + cross-check fault detection"}
```

### S04.2 Pin map (verify against actual devkit)

```yaml
pins:
  avoid: {flash: "GPIO26-32", octal_psram: "GPIO33-37", usb: "GPIO19-20", strapping: [0, 3, 45, 46]}
  adc1_only: "GPIO1-10 (ADC2 unusable while Wi-Fi active)"
  map:
    i2c_sda: 8
    i2c_scl: 9            # 100 kHz, 4.7k pull-ups to 3V3
    sps30_uart_tx: 17     # ESP TX → SPS30 RX
    sps30_uart_rx: 18
    mq7_aout: 1           # via divider
    mq7_heater_pwm: 12    # LEDC → logic-level MOSFET
    mq135_aout: 2         # via divider
    wind_vane_adc: 4
    rain_do: 5
    rain_ao: 6            # optional
    anemometer_pulse: 11  # interrupt, pull-up, 10 ms debounce
    buzzer: 13
    button: 14            # pull-up, active low
    rtc_sqw: 15           # optional 1 Hz
```

### S04.3 Electrical rules

```yaml
electrical:
  mq_divider: "MQ AOUT up to 5V; ESP ADC max ~3.1V @12dB. Use 10k/20k (scale 0.667). Firmware rescales to sensor-side voltage."
  adc: "ESP-IDF adc_cali curve-fitting, 12 dB atten, average 16 samples."
  mq_rs: "Rs = RL*(Vc - Vout)/Vout; RL = measured load resistor on the module (often 1-20k)."
  mq7_heater_cycle: "60 s @5V then 90 s @1.4V; read at END of 1.4V phase (verify datasheet). 5V PWM duty for 1.4V-equivalent heater power ≈ (1.4/5)^2 ≈ 7.8%, PWM ≥1 kHz. Many breakouts hard-wire heater to 5V → modify or use bare sensor."
  mq_burn_in: "48 h continuous heating before data is valid (flag 'warming')."
  power_budget_mA: {sps30_measuring: 60, mq7: 150, mq135: 150, esp32s3_wifi_peak: 350, others: 30}
  psu: "5V 2A"
  siting: "record lat, lon, height above ground (m), photo, obstruction notes → node registry (S16.5)."
```

---

## [S05] FIRMWARE ARCHITECTURE (ESP-IDF v5.x, FreeRTOS)

```yaml
firmware:
  tasks:
    sensor_task: {period: "1 s tick", job: "SPS30 & SGP41 @1 Hz; HDC1080, BMP280, BH1750, MQ-135, rain, vane @0.1 Hz; count anemometer pulses; MQ-7 heater state machine + read at end of low phase"}
    minute_task: {period: "60 s", job: "1-min aggregates → minute ring buffer (PSRAM, 180 min)"}
    hour_task: {trigger: "IST hour boundary via RTC", job: "hourly record, corrections, flags, 24h buffers, AQI, features, inference, decision, log append"}
    comms_task: {trigger: event, job: "HTTPS upload + backlog; POST events (queued offline); MQTT client (esp-mqtt, TLS, persistent session, LWT) for forecast/manifest/command push; HTTP poll fallback; OTA manifest check; time sync"}
    ui_task: {period: "250 ms", job: "display, buzzer, button"}
  stacks: "measure high-water marks; Na et al. used ~15 KB total for 3 tasks"
  storage:
    littlefs: {size: "4 MB", holds: ["hourly logs (30 days store-and-forward)", "model blobs", "feature_schema.yaml", "calibration.json", "manifest cache"]}
    nvs: {holds: ["24h PM buffers + states", "unit baselines", "config", "last cloud forecast", "active versions", "counters"], write_rate: "≤1/hour per key"}
  time:
    rtc: "DS3231 holds UTC"
    sync: "SNTP when online; update RTC if |Δ| > 2 s"
    features_use: "IST local time (UTC+05:30) for hour/dow/doy"
  model_runtime:
    rf: "emlearn C evaluator; trees either compiled in (update via firmware OTA) OR serialised arrays file + generic evaluator (update via model OTA) — prefer the latter"
    cnn: "esp-tflite-micro, FP32, static tensor arena in internal SRAM (start 64 KB, measure)"
    preprocessing_in_C: "instance norm, patching, de-norm, AQI roll are done in C outside the TFLM graph"
```

---

## [S06] MEASUREMENT PIPELINE (no ML)

### S06.1 Per-sensor acquisition

```yaml
acquisition:
  sps30: {rate: 1 Hz, warmup_discard_s: 30, status_check: "every 60 s (fan/laser/speed flags)", fan_cleaning: "keep default weekly; flag cleaning minutes"}
  hdc1080: {rate: 0.1 Hz, resolution: 14bit, heater: "if RH>95% for >30 min run built-in heater briefly; flag next 5 min"}
  bmp280: {rate: 0.1 Hz, mode: normal, oversampling: x16}
  bh1750: {rate: 0.1 Hz, mode: "continuous high-res", saturation: "if 65535 → lower MTreg, flag"}
  wind_speed: {method: "pulses per 60 s × kit_factor", kit_factor: "(verify from kit datasheet; common: 2.4 km/h per 1 pulse/s)"}
  wind_dir: {rate: 0.1 Hz, method: "ADC → nearest of 16 kit positions → degrees FROM"}
  rain: {rate: 0.1 Hz, method: "DO wet=1"}
  mq135: {rate: 0.1 Hz, value: Rs}
  mq7: {rate: "1 per 150 s cycle", value: "Rs at end of low phase", per_hour: 24}
  sgp41: {rate: 1 Hz, compensation: "RH/T from HDC1080", nox_conditioning_s: 10, values: [SRAW_VOC, SRAW_NOX], note: "Sensirion gas index MAY be computed for display only; never a model input"}
```

### S06.2 Minute aggregation

```yaml
minute_aggregation:
  valid_minute_min_samples: {sps30: 45, others: 4}
  pm: "median of valid 1 s readings"
  t_rh_p_lux_mq135_sgp41: "mean"
  wind: "vector mean: u=-ws*sin(θ), v=-ws*cos(θ); speed=hypot(u,v); dir=atan2(-u,-v) mod 360"
  rain: "fraction wet"
```

### S06.3 Hour aggregation

```yaml
hour_aggregation:
  bucket: "IST clock hour (I02)"
  valid_hour_min_minutes: 45
  values: {pm: "mean of minute medians", wind: "vector mean", rain_frac: "fraction wet minutes", rain_flag: "rain_frac > 0.1", mq7: "mean of cycle values", others: mean}
  extra: {pm25_max_minute: max, sraw_nox_max_minute: max}
```

### S06.4 Humidity correction (κ-Köhler)

```yaml
humidity_correction:
  a_w: "clamp(RH/100, 0, 0.95)"
  growth_factor: "C = 1 + (kappa/1.65) / (1/a_w - 1)"   # Crilley et al. 2018 form (verify)
  corrected: "PM_dry = PM_raw / C  (PM2.5 and PM10 separately)"
  kappa_default: 0.4   # literature start (verify); Delhi likely higher (Chen et al. 2022)
  kappa_source: "calibration.json; fitted from co-location (S11.2)"
  high_humidity_flag: "RH > 95%"
```

### S06.5 Dust flag

```yaml
dust_flag:
  ratio: "(PM10 - PM2.5) / max(PM1, 1)"
  rule: "dust = ratio > R_dust AND RH < 60"
  R_dust_default: 1.5   # (tune) on co-location
  use: "model input only; no PM alteration until a correction factor is fitted (Kaur et al. 2026 approach)"
```

### S06.6 CPCB AQI (shared implementation; I01)

```yaml
cpcb_aqi:
  buffers: "24 hourly slots each for corrected PM2.5, PM10 with state"
  validity: "≥16 'measured' hours of 24 → valid else 'not_ready'"
  mean_uses: "measured hours only"
  subindex: "I = (I_hi - I_lo)/(B_hi - B_lo) * (C - B_lo) + I_lo"
  aqi_pm: "max(I_pm25, I_pm10)"
  label: "PM-based"
  aqi_full_when_online: "max(I_pm25, I_pm10, cloud.no2, cloud.o3, cloud.co)"
  continuous_gaps: "values between listed integer bands (e.g. 30.5) use previous band's B_hi as B_lo"
  severe_band: "upper interpolation limit from official CPCB calculator (verify); cap 500"
  breakpoints_ug_m3_24h:   # (verify against official CPCB NAQI document)
    - {aqi: [0, 50],    category: Good,         pm25: [0, 30],    pm10: [0, 50]}
    - {aqi: [51, 100],  category: Satisfactory, pm25: [31, 60],   pm10: [51, 100]}
    - {aqi: [101, 200], category: Moderate,     pm25: [61, 90],   pm10: [101, 250]}
    - {aqi: [201, 300], category: Poor,         pm25: [91, 120],  pm10: [251, 350]}
    - {aqi: [301, 400], category: Very_Poor,    pm25: [121, 250], pm10: [351, 430]}
    - {aqi: [401, 500], category: Severe,       pm25: [250, null], pm10: [430, null]}
  test_vectors: "tests/aqi_vectors.json (≥50 cases incl. band edges, 16/24 validity, severe cap) shared by firmware, Python, cloud"
```

### S06.7 Quality flags

```yaml
flags_bitmask_per_variable:
  bit0: measured
  bit1: filled          # carried forward (never a training target)
  bit2: missing
  bit3: high_humidity
  bit4: dust
  bit5: warming         # burn-in / startup
  bit6: fan_cleaning
  bit7: sensor_fault
record_level:
  model_used: [A_rf, A_cnn, B_rf, B_cnn]
  pm_state: [ok, filled_le_1h, fallback_model_b, unavailable]
```

---

## [S07] PREDICTION TARGET AND AQI TREND

```yaml
target:
  outputs: 6
  definition: "y[p,h] = ln(1+PM_p[t+h]) - ln(1+PM_p[t]), p ∈ {pm25, pm10}, h ∈ {1,2,3}"
  PM: "hourly mean, humidity-corrected (S06.4)"
  why_log_residual: "Delhi PM spans ~10–1000 µg/m3; residual avoids tree extrapolation failure; matches cloud target (I09)"
  H_OUT: 3   # config
  output_order: "[pm25_h1, pm25_h2, pm25_h3, pm10_h1, pm10_h2, pm10_h3] — identical order in edge_fc.y, teacher_store.y_teacher and training labels (cloud uses h1..h6 in the same pollutant-major layout)"
aqi_trend_on_device:
  step1: "PM_p[t+h] = exp(ln(1+PM_p[t]) + ŷ[p,h]) - 1"
  step2: "roll 24h buffer forward: append predicted hours, drop oldest"
  step3: "AQI[t+h] = cpcb_aqi(rolled buffer)"
  step4: "ΔAQI[h] = AQI[t+h] - AQI[t]"
  trend_label: {rising: "ΔAQI[3] ≥ +5", falling: "ΔAQI[3] ≤ -5", stable: otherwise}   # (tune)
  band_warning: "predicted category at any h worse than current"
  note: "AQI is a 24h mean; hours leaving the window are known exactly — rolling predicted hourly PM uses that exactly."
```

---

## [S08] FEATURES (feature_schema.yaml content)

### S08.1 Sequence channels (CNN input; also base for RF features)

```yaml
feature_schema:
  version: 1
  lookback_hours: 24          # ablation 48
  channels:
    - {i: 0,  name: ln_pm25,       src: pm25_corr,       tf: "ln(1+x)",               instnorm: true,  model_b: false}
    - {i: 1,  name: ln_pm10,       src: pm10_corr,       tf: "ln(1+x)",               instnorm: true,  model_b: false}
    - {i: 2,  name: ln_pm1,        src: pm1,             tf: "ln(1+x)",               instnorm: true,  model_b: false}
    - {i: 3,  name: coarse_ratio,  src: "(pm10-pm25)/max(pm1,1)", tf: "clip(0,10)",   instnorm: true,  model_b: false}
    - {i: 4,  name: t,             src: t,               tf: identity,                instnorm: true,  model_b: true}
    - {i: 5,  name: rh,            src: rh,              tf: identity,                instnorm: true,  model_b: true}
    - {i: 6,  name: p,             src: p,               tf: "x-1000",                instnorm: true,  model_b: true}
    - {i: 7,  name: wind_u,        src: wind_u,          tf: identity,                instnorm: true,  model_b: true}
    - {i: 8,  name: wind_v,        src: wind_v,          tf: identity,                instnorm: true,  model_b: true}
    - {i: 9,  name: rain_frac,     src: rain_frac,       tf: identity,                instnorm: false, model_b: true}
    - {i: 10, name: ln_lux,        src: lux,             tf: "ln(1+x)",               instnorm: true,  model_b: true}
    - {i: 11, name: mq7_n,         src: mq7_rs,          tf: "ln(Rs/Rbase7)",         instnorm: true,  model_b: true}
    - {i: 12, name: mq135_n,       src: mq135_rs,        tf: "ln(Rs/Rbase135)",       instnorm: true,  model_b: true}
    - {i: 13, name: voc_n,         src: sraw_voc,        tf: "(x-base_voc)/1000",     instnorm: true,  model_b: true}
    - {i: 14, name: nox_n,         src: sraw_nox,        tf: "(x-base_nox)/1000",     instnorm: true,  model_b: true}
    - {i: 15, name: hod_sin,       src: ist_hour,        tf: "sin(2πh/24)",           instnorm: false, model_b: true}
    - {i: 16, name: hod_cos,       src: ist_hour,        tf: "cos(2πh/24)",           instnorm: false, model_b: true}
    - {i: 17, name: dow_sin,       src: ist_dow,         tf: "sin(2πd/7)",            instnorm: false, model_b: true}
    - {i: 18, name: dow_cos,       src: ist_dow,         tf: "cos(2πd/7)",            instnorm: false, model_b: true}
    - {i: 19, name: doy_sin,       src: ist_doy,         tf: "sin(2πd/365)",          instnorm: false, model_b: true}
    - {i: 20, name: doy_cos,       src: ist_doy,         tf: "cos(2πd/365)",          instnorm: false, model_b: true}
    - {i: 21, name: aqi_now,       src: aqi_pm,          tf: "x/500",                 instnorm: true,  model_b: false}
    - {i: 22, name: pm_mask,       src: pm_state,        tf: "1 if measured else 0",  instnorm: false, model_b: false}
  unit_baselines:
    Rbase7:   "rolling 7-day 90th percentile of hourly mq7_rs (Rs high in clean air)"
    Rbase135: "rolling 7-day 90th percentile of hourly mq135_rs"
    base_voc: "rolling 7-day 90th percentile of sraw_voc (verify direction on real data)"
    base_nox: "rolling 7-day 10th percentile of sraw_nox (verify direction on real data)"
    storage: "NVS; included in every record; simulator MUST apply identical baselining"
  missing_input_policy: "gap ≤2 h: linear interpolate + mask; >2 h: window invalid for training; at inference use last value + mask"
```

### S08.2 Model B extra features

```yaml
model_b_extra:
  - {name: last_valid_ln_pm25}
  - {name: last_valid_ln_pm10}
  - {name: last_valid_aqi, tf: "x/500"}
  - {name: hours_since_pm_lost, tf: "clip(0,24)/24"}
  - {name: pm25_trend_before_loss, def: "ln_pm25[last_valid] - ln_pm25[last_valid-3h]"}
```

### S08.3 RF engineered features (from the same hourly buffer)

```yaml
rf_features:
  ln_pm25: {lags: [0,1,2,3,6,12,23], roll_mean: [3,6,24], roll_std: [6], diff: [1,3]}
  ln_pm10: {lags: [0,1,3], roll_mean: [6,24], diff: [1]}
  ln_pm1: {lags: [0]}
  coarse_ratio: {lags: [0]}
  dust_flag: {lags: [0]}
  t_rh_p: {lags: [0], diff: [3]}
  wind_u_v: {lags: [0]}
  wind_speed: {roll_mean: [3]}
  rain_frac: {lags: [0], rain_hours_last: 3}
  ln_lux: {lags: [0], diff: [1]}
  gases_mq7_mq135_voc_nox: {lags: [0], diff: [1]}
  time: [hod_sin, hod_cos, dow_sin, dow_cos, doy_sin, doy_cos]
  aqi_now: {lags: [0]}
  approx_count: 50
  model_b_variant: "drop all PM-derived + aqi_now; add S08.2"
```

---

## [S09] DATA SOURCES AND BASE TABLE

```yaml
data_sources:
  cpcb:
    access: "CPCB CCR portal or OpenAQ API (mirrors CPCB)"
    content: "hourly PM2.5, PM10, NO2, NO, NOx, CO, O3, SO2, NH3, benzene; AT, RH, WS, WD, BP, SR, RF (station-dependent)"
    coverage: "Delhi ≈40 stations, 2018→present"
    time_convention: "IST hourly averages labelled From–To; ts = From (IST) → convert to UTC per I02"
  open_meteo:
    apis: ["historical weather (ERA5-based)", "historical forecast (for honest forecast-weather training)", "forecast (runtime, cloud only)"]
    vars: "temperature_2m, relative_humidity_2m, dew_point_2m, surface_pressure, wind_speed_10m, wind_direction_10m, precipitation, shortwave_radiation, boundary_layer_height (verify names/availability)"
    alignment: "Open-Meteo hourly is on UTC hours; IST hours are offset 30 min. Instantaneous vars → average the two bracketing UTC values; accumulated/mean vars (precipitation, shortwave) → take proportional overlap. Verify how the API treats half-hour timezones."
  nasa_firms: "optional, cloud teacher only"
  own_logs: "all S08 channels + raw"
unit_conversions_25C_1atm: {co_mg_m3_to_ppm: 0.873, no2_ug_m3_to_ppb: 0.532, nh3_ug_m3_to_ppb: 1.436}
base_table_build:
  - "download CPCB hourly for all Delhi stations"
  - "fetch Open-Meteo at each station lat/lon; prefer CPCB weather when present and plausible"
  - "clean: drop negatives; drop PM2.5 > 1.05×PM10; drop flatlines ≥6 h; drop PM2.5 > 1500"
  - "never impute target hours; input gaps ≤2 h interpolate+mask"
  - "output: parquet partitioned by station, one row per IST hour"
```

---

## [S10] SYNTHETIC BOOTSTRAP (physics-based sensor simulator)

### S10.1 Principle

Public data contains true pollutant concentrations and weather but not our sensors' outputs. Simulate each sensor with a forward model and randomised per-unit parameters (domain randomisation), apply the exact on-device processing, and train on it. Labels come from true CPCB PM. Later fine-tune on real logs (S11). Contrast with Morales-García et al. (GAN-generated data): ours is sensor-physics-based.

### S10.2 Virtual nodes

```yaml
virtual_nodes:
  per_station_units: 3            # ≈120 virtual nodes for Delhi
  unit_params: "drawn once per virtual unit; fixed over its lifetime except drift"
  splits:
    time: {train: "2018-2023", val: "2024", test: "2025+"}   # adjust to availability
    spatial_holdout_stations: 5
```

### S10.3 Forward sensor models

```yaml
forward_models:
  pm25_sps30:
    eq: "PM25_meas = s_u * PM25_true * C(RH; kappa_u) + N(0, 0.05*PM + 1)"
    params: {s_u: "U(0.85,1.25)", kappa_u: "U(0.2,0.7)"}
  pm10_sps30:
    eq: "PM10_meas = PM25_meas + f_u * s_u * (PM10_true - PM25_true) + noise"
    params: {f_u: "U(0.3,0.7)"}   # coarse undersizing (Kaur 2023/2026, Ouimette 2023, Gautam 2025)
  pm1_sps30: {eq: "PM1_meas = r_u * PM25_meas", params: {r_u: "U(0.6,0.85)"}}
  pm_common: {clip: [0, 1000], rh_gt_95_bias: "+U(0,20)%", random_missing_hours: "1%", outages_per_year: "1 × U(6,48) h"}
  t_hdc1080: {eq: "T_true + N(0,0.3) + b_T_u + 0.8*(SW/1000)", params: {b_T_u: "U(-0.5,0.5)"}}
  rh_hdc1080: {eq: "RH from true dewpoint at simulated T + N(0,2), clip 0-100"}
  p_bmp280: {eq: "P_true + N(0,0.5) + b_P_u", params: {b_P_u: "U(-3,3) hPa"}}
  wind_speed: {eq: "k_u*WS_true + N(0,0.3); 0 if <0.3", params: {k_u: "U(0.4,1.0) rooftop sheltering"}}
  wind_dir: {eq: "WD_true + N(0,15°) mod 360"}
  rain: {eq: "wet if precip>0.1 mm/h; stays wet U(1,2) h after; false-wet p=0.02 if RH>90"}
  lux_bh1750:
    eq: "min(65535, g_u * 110 * SW * exp(-beta_u*PM25_true/100)) + N(0,50); 0 at night"
    params: {g_u: "U(0.6,1.2)", beta_u: "U(0,0.2)"}
    note: "110 lux per W/m2 ≈ daylight luminous efficacy (approximate)"
  mq7:
    eq: "ln Rs = ln R0_u + b7_u*ln(CO_ppm+0.1) + gT_u*(T-20) + gRH_u*(RH-65) + drift_u*t_days + N(0,0.05)"
    params: {R0_u: "lognormal σ=0.5", b7_u: "U(-0.75,-0.55)", gT_u: "U(-0.02,0)", gRH_u: "U(-0.01,0)", drift_u: "U(-0.001,0.001)/day"}
  mq135:
    eq: "same form with eq_conc = w1*NH3_ppm + w2*CO_ppm + w3*10*NO2_ppm + w4*VOC_proxy; exponent b135_u"
    params: {b135_u: "U(-0.5,-0.3)", w1_to_w4: "U(0.5,1.5) each"}
  sgp41_voc: {eq: "base_u - kv_u*ln(1+VOC_proxy) + N(0,50)", params: {base_u: "U(25000,35000)", kv_u: "U(500,2000)"}}
  sgp41_nox: {eq: "base_u + s_n*kn_u*ln(1+NO2_ppb) + N(0,30)", params: {kn_u: "U(200,800)", s_n: "+1 default (verify sign on real logs)"}}
  voc_proxy: "0.5*10*CO_ppm + 0.3*NO2_ppb/10 + 0.2*benzene(if available) + 0.1*PM25_true/10   # assumption, document it"
post_simulation:
  - "apply on-device pipeline exactly: kappa_default correction, dust flag, unit baselines, features"
  - "labels from TRUE CPCB PM (S07 definition)"
  - "save as training shards: (node_id, ts, X_seq[24×23], X_rf[~50], y[6], masks, teacher fields empty)"
tests:
  - "each forward model unit-tested for monotonicity and ranges"
  - "simulated distributions (mean/std/quantiles per season) compared with first real week once available; refit placeholders"
```

---

## [S11] REAL LOGS, CO-LOCATION, TRAINING SCHEDULE

### S11.1 Real-log labels (self-supervised)

```yaml
real_labels:
  target_source: "node's own future corrected PM (measured only)"
  window_valid_model_a: "all 3 target hours measured AND ≥20 measured PM hours in the 24h input"
  forbidden_targets: "filled, missing, model outputs (I07)"
  reprocessing: "if calibration changes, recompute processed values from raw before retraining"
```

### S11.2 Co-location (SHOULD before deployment; 2–4 weeks)

```yaml
colocation:
  placement: "≤50 m from a CPCB station inlet"
  fit_kappa: "argmin Σ (PM_raw/C(RH;κ) - PM_ref)^2 over hours RH<95"
  fit_optional: [pm10_scale, R_dust]
  verify: ["SGP41 tick directions", "MQ response directions", "simulator placeholder ranges"]
  output: "calibration.json {version, kappa, pm10_scale, R_dust, station, date_range, per_node|global}"
  distribute: "OTA to all nodes; cloud stores and uses same version (I06)"
```

### S11.3 Training schedule

```yaml
training_schedule:
  P0_before_deploy: {data: synthetic, cnn: pretrain, rf: train}
  P1_first_2to4_weeks: {data: "synthetic + real", cnn: "fine-tune lr×0.1, early stop on latest real week", rf: "retrain, real rows weight 5×"}
  P2_weekly: {data: "growing real", cnn: "fine-tune", rf: "retrain; raise real weight; drop synthetic after >3 months real"}
  evaluation_rule: "ALWAYS evaluate on latest held-out real period; never report synthetic scores as results"
  required_experiment: "synthetic-only vs real-only vs synthetic-pretrain+real-finetune on the same real test period"
  where_training_runs: "offline / cloud ML pipeline; edge only runs inference"
```

---

## [S12] TEACHER AND DISTILLATION (edge requirements on cloud outputs)

### S12.1 Teacher definition

```yaml
teacher:
  main: "cloud wind-aware spatio-temporal graph model TI-STGNN (defined in cloud doc [C08]; built separately)"
  privileged_info: ["other nodes' data", "wind-directed edges", "forecast weather", "optional: future ground truth (Na-style LUPI variant)"]
  interim_teacher_until_graph_ready:
    model: "LightGBM/XGBoost (cf. Shmuel et al.: XGBoost teacher)"
    inputs: "node features + same-hour & lagged PM of 5 nearest CPCB stations + upwind indicator (station bearing vs wind dir)"
    target: "same 6 targets (I09)"
  precondition: "teacher RMSE < no-KD student RMSE on validation, per horizon. If not, do NOT distil; report it (Shmuel: weak KNN teacher gave no gain)."
```

### S12.2 Teacher artifacts the edge training pipeline needs

```yaml
teacher_store:
  granularity: "per (node_id, base_ts)"
  fields:
    y_teacher: "float[6] — same definition as S07 (h=1..3 subset of cloud's 1..6)"
    oof: "bool — MUST be out-of-fold (5 time-block folds) for training windows"
    feat_seq: "float[L_T=24][D_T] — teacher per-node hidden sequence for the 24 input hours (for full_KD)"
    attn: "float[heads][24][24] — temporal self-attention map from teacher's final 1-layer temporal Transformer encoder for this node (for AFD)"
    teacher_version: "string"
  aggregation_for_student: "average feat_seq/attn into the student's L=12 two-hour patches (attn: average over both axes' patch groups)"
  if_teacher_lacks_attention: "full_KD degrades to out_KD + L_feat only (no L_afd)"
  leakage_rules:
    - "teacher trained only on periods ≤ student train period; never on student test period"
    - "OOF predictions for every training window"
    - "teacher inputs for window base_ts use only data available at base_ts (no future except in explicit LUPI variant during teacher training)"
  format: "parquet, partitioned by node_id/month"
```

---

## [S13] EDGE MODELS

### S13.0 Overview

```yaml
models:
  roles:
    Model_A: {when: "PM healthy", inputs: "all features"}
    Model_B: {when: "PM failed > 1 h", inputs: "no PM features + S08.2", details: S14}
  architectures: [RF, GDC_CNN, optional CNN_GRU]
  distillation_variants: [no_KD, out_KD, full_KD(cnn only)]
  baselines: {persistence: "ŷ=0", diurnal_persistence: "mean change for that IST hour over last 14 days"}
  output_count: 6
```

### S13.1 Paper reference: Shmuel et al. 2026 (output distillation into a non-neural student)

```yaml
paper_shmuel_2026:
  citation: "A. Shmuel, T. Lazebnik, O. Glickman, 'Follow the Forest Trail: Distillation by Gradient Boosting Models to Enhance Symbolic Regression Performance', IEEE Access 14:19701-19712, 2026, doi:10.1109/ACCESS.2026.3657793"
  problem: "symbolic regression (SR) is interpretable but less accurate than tree ensembles"
  method:
    - "train teacher XGBoost (via TPOT AutoML, 15 min optimisation) on train set"
    - "predict teacher outputs on the TRAIN set"
    - "train student SR on (X_train, ŷ_teacher) instead of (X_train, y_train)"
    - "evaluate student on TRUE y_test"
  protocol: "80/20 random splits ×50 repetitions per dataset; 10 datasets + wildfire case; paired t-tests on per-run RMSE differences"
  results:
    - "mean RMSE improvement 3.3–6.6% (QLattice 4.88%, PySR 3.33%, gplearn 6.58%); wins in 7–8 of 10 datasets"
    - "gain largest when teacher >> student; KNN teacher (not better than student) → no gain / slight harm; RF teacher slightly worse than XGB"
    - "longer XGB optimisation → larger gains; student complexity unchanged"
  mechanism_claimed: "target smoothing / denoising, not structural-bias transfer"
  what_we_take: ["output-level KD works for non-neural students", "teacher must beat student (precondition)", "paired-test evaluation protocol"]
  what_we_change: ["student = RandomForest", "blend α instead of pure teacher target", "out-of-fold teacher predictions (avoid teacher-memorised noise)", "time-series splits, not random splits"]
```

### S13.2 Student 1 — Random Forest

```yaml
rf_student:
  inputs: "S08.3 engineered features"
  outputs: "6 separate RandomForestRegressor (one per target) — simplest emlearn export"
  kd_target: "y_kd = α*y_true + (1-α)*y_teacher_oof"
  alpha_grid: [1.0, 0.5, 0.0]    # 1.0 = no_KD, 0.0 = pure Shmuel
  hyperparam_search:
    n_estimators: [10, 20, 30]
    max_depth: [6, 8, 10]
    min_samples_leaf: [5, 20, 50]
    max_features: [0.3, 0.5]
  size_budget: "all 6 forests < 400 KB flash"
  selection: "best on latest real validation week (or synthetic val in P0)"
  export: "emlearn → C arrays; prefer serialised arrays file + generic evaluator for OTA"
  parity_test: "device vs Python on 1,000 vectors, max abs diff < 1e-4"
  evaluation: "10 seeds × 5 time-block splits; paired t-test no_KD vs KD per horizon; Diebold–Mariano vs persistence"
```

### S13.3 Paper reference: Na et al. 2026 (heterogeneous distillation into a tiny CNN on ESP32-S3)

```yaml
paper_na_2026:
  citation: "S. Na, Y. Cho, Y. Baek, 'Distilling Privileged Knowledge From Transformers to Lightweight CNNs for On-Device Time Series Forecasting', IEEE Access 14:30373-30386, 2026, doi:10.1109/ACCESS.2026.3667131"
  problem: "accurate Transformer/LLM forecasters cannot run on MCUs; small models miss long dependencies"
  preprocessing:
    instance_norm: "X̄ = (X - μ)/σ per window per variable; reversible de-norm on output"
    patching: "non-overlapping patches of length P (PatchTST style); L = H/P; X_P ∈ R^{L×P×N}"
  teacher:
    backbone: "frozen Llama-2-7B; numerical patches (P=96) linearly projected to D_LLM=4096 (tokenizer bypassed)"
    privileged_input: "X_priv = concat(X, Y) (history + future ground truth) — LUPI"
    SCA: "Subtractive Cross-Attention: E_future = Q - Attention(Q,K,V); Q = privileged future embedding, K=V = historical embedding (removes history info)"
    encoder: "1-layer Transformer encoder → feature maps F_T ∈ R^{L×D×N}; decoder reconstructs Y"
    projection: "ψ(·) linear projection of teacher features to student dim D=512 (inside teacher)"
    loss: "Huber(Ŷ_T, Y)"
  student:
    structure: "projection encoder → N=3 GDC blocks (k=3, d=1,2,4) → decoder → de-norm"
    gdc_block:
      1: "X_norm = LayerNorm(X_in)"
      2: "X_dw = DilatedDepthwiseConv(X_norm, kernel=3, dilation=d)   # per-channel temporal filter; dilation widens receptive field at no param cost"
      3: "X_gate = GLU(PWConv1(X_dw)) = A ⊙ σ(B)   # 1×1 conv, split in two halves, sigmoid gate (replaces ReLU)"
      4: "X_out = X_in + PWConv2(X_gate)   # 1×1 conv back to D + residual"
    hidden_D: 512
    params: "0.64 M; 0.009 GFLOPs"
  training:
    stage1: "pretrain teacher; freeze; store F_T and attention maps A_T offline (final Transformer layer)"
    stage2_losses:
      L_pred: "Huber(Ŷ_S, Y)"
      L_feat: "Huber(F_S, ψ(F_T))"
      L_afd: "w_j = Σ_i mean_heads A_T[i,j]; ŵ = (w-μ_w)/(σ_w+ε); M = softmax(ŵ); L_afd = (1/(L·D·N)) Σ_i M_i Σ_j Σ_k Huber(F_S[i,j,k], F_T[i,j,k])"
      curriculum: "r(e) = min(1, e/E_warmup), linear 0→1 over all 10 epochs"
      total: "L_total = λp·L_pred + ((1-λp)/2)·r(e)·(L_feat + L_afd), λp = 0.4"
    optimiser: "Adam lr 1e-4, batch 128, 10 epochs, A100 GPU"
    horizons: "single model trained for T=96; longer horizons by autoregressive rolling"
    lookback: "H=672 long-term, H=168 short-term"
  results:
    - "MSE −11.5% (long-term) and −12.4% (short-term) vs baseline average; beats AutoTimes (7B) with <0.01% params"
    - "ablations: removing KD, AFD, dilation, GLU, or depthwise-separable conv each worsened accuracy"
  deployment:
    device: "ESP32-S3, 240 MHz dual LX7, 512 KB SRAM, 8 MB PSRAM"
    runtime: "TensorFlow Lite for Microcontrollers"
    precision: "FP32 chosen. Dynamic INT8 raised MSE at long horizons (0.431→0.458 @T=720) and latency 256→416 ms (type-casting in patching)"
    resources: {flash: "737 KB", sram_total: "≈201 KB", tensor_arena: "175 KB used of 200 KB static", io_buffers: "11 KB", freertos_stacks: "15 KB (3 tasks: inference, send, receive)", latency: "256 ms"}
  what_we_take: ["GDC block exactly", "instance norm + patching", "two-stage offline KD with L_pred+L_feat+L_afd and curriculum", "FP32 on ESP32-S3 via TFLM", "preprocessing outside the model"]
  what_we_change: ["teacher = graph model; privileged info = spatial (neighbours, wind) instead of future truth", "much shorter lookback/horizon", "smaller D", "early stopping with more epochs", "multivariate channel-mixing patches"]
```

### S13.4 Student 2 — GDC-CNN (our configuration)

```yaml
gdc_cnn_student:
  input: "X ∈ R^{24×23}"
  step1_instance_norm: "channels with instnorm=true (S08.1): per-window μ,σ (ε=1e-5). Others pass through."
  step2_patching: "P=2 → L=12 patches; flatten each patch to 2×23=46 values"   # ablation: P=1, L=24, add 4th block d=8
  step3_encoder: "Dense(46→D) shared across patches + learned positional embedding [L×D]"
  step4_blocks: "3 × GDC (k=3, d=1,2,4), CAUSAL left padding along patch axis"
  receptive_field: "1 + 2·(1+2+4) = 15 patches ≥ 12"
  step5_features: "F_S = last GDC output [L×D]"
  step6_decoder: "Flatten(L·D) → Dense → 6"
  step7_denorm: "targets are differences in ln space → multiply by σ of channel ln_pm25 / ln_pm10 of the window (no mean shift)"
  D_search: [16, 32, 64]
  losses:
    no_KD: "Huber(ŷ_S, y)"
    out_KD: "λ·Huber(ŷ_S, y) + (1-λ)·Huber(ŷ_S, ŷ_T)   # Shmuel-style; λ ∈ {0.5, 0.7}"
    full_KD: "Na total: λp=0.4; L_feat vs ψ(F_T patch-averaged); L_afd with teacher attn mask; ψ = trainable Linear(D_T→D) during student training, discarded after"
  huber_delta: 1.0   # on normalised targets
  optimiser:
    pretrain_synthetic: {opt: Adam, lr: 1e-3, batch: 128, epochs: "≤100", early_stop_patience: 10, E_warmup: 10}
    finetune_real: {lr: 1e-4}
  framework: "Keras/TensorFlow recommended (direct TFLite). PyTorch allowed via ONNX→TF→TFLite."
  tflm_rules:
    conv1d: "Conv2D with height 1"
    depthwise: "DEPTHWISE_CONV_2D with dilation"
    pointwise: "1×1 CONV_2D or FULLY_CONNECTED"
    glu: "split + LOGISTIC + MUL"
    layernorm: "MEAN, SQUARED_DIFFERENCE, RSQRT, MUL, ADD — if unsupported in esp-tflite-micro build, replace by learned per-channel scale/shift and document"
    outside_graph: "instance norm, patching, de-norm, AQI roll in C"
  step0_mandatory: "convert and run a 1-block toy model on the board before training the real one; confirm every op"
  precision: {default: FP32, ablation: "static full-INT8 PTQ with representative dataset"}
  measure_on_device: ["flash bytes", "arena_used_bytes()", "latency mean of 100 runs (esp_timer)", "energy (INA219 if fitted)"]
  parity_test: "device vs Python on 1,000 vectors, FP32 max abs diff < 1e-4"
```

### S13.5 Optional Student 3 — CNN-GRU

```yaml
cnn_gru_optional:
  reference: "Mazinani et al., IEEE Access 2025, doi:10.1109/ACCESS.2025.3603920 — CNN-BiGRU best for PM2.5 on STM32; 8-bit PTQ −66% size, ~1% accuracy loss"
  build_only_if: "GRU converts to TFLM (usually requires unrolling)"
  role: "extra comparison row, not a deliverable"
```

---

## [S14] MODEL B — NO-PM FALLBACK (instead of PM imputation)

```yaml
why_not_impute_pm:
  - "gas/weather sensors cannot see dust, transported smoke, secondary aerosol (most of Delhi PM)"
  - "imputing PM k hours after loss == k-hour forecast from a stale value; chaining adds error"
  - "prior test: PM from other sensors without recent PM → R² = −4.28"
pm_failure_detection_single_sps30:
  any_of:
    - "no valid SPS30 frame for 60 s, or status fan/laser error"
    - "minute PM2.5 identical ≥30 consecutive minutes"
    - "PM2.5 > 1.05 × PM10 (outputs are cumulative; impossible)"
    - "PM outside [0,1000] or one-minute jump > 500 not mirrored in PM1"
    - "(if backup PMSA003 fitted) disagreement > 40% for > 1 h"
runtime_policy:
  - {since_loss: "≤1 h", pm_shown: "last value (filled)", model: "A with pm_mask=0 for filled hours", aqi: "valid (measured hours only)"}
  - {since_loss: ">1 h", pm_shown: "unavailable", model: "B (low confidence)", aqi: "valid while ≥16/24 measured (~8 h), then not_ready"}
  - {since_loss: "any, online", extra: "show cloud neighbour-based PM estimate (labelled)", forecast: "cloud preferred"}
  - {since_loss: ">24 h offline", model: "B only", alerts: disabled}
model_b_training_pm_dropout:
  - "from each valid Model A window, sample k ~ U{1..24}"
  - "hide last k hours of PM channels; compute S08.2 fallback features from the hour before hidden part"
  - "targets relative to last valid PM: y = ln(1+PM[t+h]) - ln(1+PM_last_valid)"
  - "train RF and GDC-CNN variants with same KD variants as Model A (teacher Model-B-compatible outputs: teacher still forecasts PM, so y_teacher re-expressed relative to PM_last_valid)"
paper_plot: "RMSE vs hours_since_pm_lost (1..24): Model A (true PM), Model B, persistence-from-last-valid"
```

---

## [S15] RUNTIME DECISION LOGIC (hourly state machine)

```yaml
decision_logic:
  step1: "build record; update buffers; compute AQI_PM"
  step2: "select model (S14 runtime_policy); infer; AQI trend T+1..3 (edge_fc)"
  step3: "upload record + edge_fc"
  step4_receive_cloud: "wait for MQTT forecast push for base_ts; if none by HH:12 IST poll GET /api/v1/forecast (S02.2, S16.7)"
  step5_display_source:
    - {if: "cloud_fc.fresh AND cloud_fc.based_on_ts == record.ts", show: cloud_fc, source: cloud}
    - {elif: "cloud_fc age ≤ 2 h", show: cloud_fc, source: "cloud (stale ≤2h)"}
    - {else: true, show: edge_fc, source: "edge (fallback)"}
  step6_event_detection:
    only_if: "cloud_fc.fresh AND cloud_fc.based_on_ts == record.ts AND cloud_fc.node_inputs_complete"
    local_event_requires: "cloud_fc.interval_quality == 'calibrated'"
    disagree: "edge AQI[h] ∉ [cloud.lower[h], cloud.upper[h]] for ≥2 of h=1..3"
    local_event: "disagree AND (z(sraw_nox_max vs 7-day) > 3 OR pm25_max_minute > 2×hourly mean OR Δ mq7_n > 3σ)"
    sensor_check: "disagree AND gas signals flat"
    on_flag: "immediately POST /api/v1/events (S16.6) with evidence; include in hourly record too; de-duplicate: one open event per type per node; send status=update every hour while active, status=closed when cleared"
    note: "Cloud re-checks the same rule server-side and marks the event cloud_confirmed (cloud doc C13)."
  step6b_pm_fault_event: "when PM state changes to sensor_fault or recovers, POST event type pm_sensor_fault (open/closed)"
  step7_alerts:
    triggers: ["current category worsens", "displayed forecast shows worse category within 3 h (edge-source → 'low confidence')"]
    rate_limit: "1 buzzer alert/hour; button silences 2 h"
  step8_health_message: "per CPCB category"
  step9_log: "write full record (S16.2)"
```

---

## [S16] CLOUD INTERFACE CONTRACTS (edge implements client side; cloud implements server side)

### S16.1 Transport and auth

```yaml
transport:
  protocol: "HTTPS (TLS 1.2+), CA bundle in firmware"
  auth: "per-node bearer token (provisioned at onboarding) in Authorization header; rotateable via OTA config"
  base_url: "config"
  api_version: v1
  payload_encoding: "JSON (gzip if > 4 KB)"
  idempotency: "records keyed by (node_id, ts); server upserts; client may resend safely"
```

### S16.2 POST /api/v1/ingest (edge → cloud)

```json
{
  "node_id": "delhi-n01",
  "fw_version": "1.3.0",
  "schema_hash": "sha256:...",
  "pipeline_version": "1.0",
  "calibration_version": "2026-10-colo-v1",
  "records": [
    {
      "ts": "2026-10-01T04:30:00Z",
      "raw": {"pm1": 61.2, "pm25": 88.4, "pm4": 95.1, "pm10": 120.5, "typ_size_um": 0.62,
              "t": 27.4, "rh": 71.0, "p": 993.2, "lux": 18000,
              "wind_u": -0.8, "wind_v": 1.1, "wind_speed": 1.36, "wind_dir": 324,
              "rain_frac": 0.0, "mq7_rs": 18450.0, "mq135_rs": 32100.0,
              "sraw_voc": 28110, "sraw_nox": 15620,
              "pm25_max_minute": 140.2, "sraw_nox_max_minute": 15900},
      "proc": {"pm25_corr": 70.9, "pm10_corr": 96.6, "coarse_ratio": 0.42, "dust": false,
               "aqi_pm": 197, "aqi_valid_hours": 24, "kappa": 0.4,
               "baselines": {"Rbase7": 21000, "Rbase135": 35000, "base_voc": 30500, "base_nox": 15100}},
      "flags": {"pm25": ["measured"], "rh": ["measured", "high_humidity"]},
      "pm_state": "ok",
      "edge_fc": {"model": "A_cnn", "model_version": "cnn-A-2026-10-05", "y": [0.02, 0.05, 0.08, 0.01, 0.03, 0.05],
                  "pm25": [72.4, 74.6, 76.8], "pm10": [97.6, 99.6, 101.6], "aqi": [199, 201, 203],
                  "trend": "stable", "inference_ms": 38},
      "cloud_fc_used": {"based_on_ts": "2026-10-01T03:30:00Z", "source": "cloud (stale ≤2h)"},
      "events": {"local_event": false, "sensor_check": false, "alerts": []},
      "sys": {"heap_free": 181000, "psram_free": 7300000, "arena_used": 41200, "rssi": -67, "uptime_s": 86400, "energy_mJ": null}
    }
  ]
}
```

```yaml
ingest_rules:
  batch: "backlog oldest-first, ≤48 records/request"
  response_200: '{"accepted": [ts...], "rejected": [{"ts":..., "reason":...}]}'
  mark_uploaded: "only ts listed in accepted"
  retention_on_device: "30 days, FIFO"
```

### S16.3 GET /api/v1/forecast?node_id=&base_ts= (cloud → edge)

```json
{
  "node_id": "delhi-n01",
  "based_on_ts": "2026-10-01T04:30:00Z",
  "issued_at": "2026-10-01T05:39:00Z",
  "teacher_version": "graph-2026-10-03",
  "node_inputs_complete": true,
  "horizons": [1, 2, 3, 4, 5, 6],
  "y": [0.03, 0.06, 0.09, 0.11, 0.12, 0.13],
  "y10": [0.01, 0.03, 0.05, 0.06, 0.07, 0.08],
  "pm25": [73.1, 75.4, 77.7, 79.3, 80.1, 80.9],
  "pm10": [97.6, 99.6, 101.6, 102.6, 103.6, 104.6],
  "aqi": [199, 202, 204, 206, 207, 208],
  "aqi_lower": [190, 191, 190, 188, 186, 184],
  "aqi_upper": [208, 214, 219, 224, 229, 233],
  "interval_coverage_target": 0.9,
  "interval_quality": "calibrated",
  "gas_subindex": {"no2": 62, "o3": 41, "co": 88, "station": "DL-ITO", "station_ts": "2026-10-01T04:30:00Z"},
  "neighbour_pm_estimate": {"pm25": 74.0, "pm10": 101.0, "valid": true},
  "updates": {"manifest_etag": "W/\"m-17\""}
}
```

```yaml
forecast_rules:
  fresh: "based_on_ts == edge record ts"
  delivery: "primary: MQTT push (S16.7) with byte-identical JSON; GET is fallback and used after reboot"
  unknown_fields: "ignore (forward compatible; cloud may add fields such as forecast_source, trend)"
  interval_quality: "'calibrated' | 'provisional' (new node, few residuals). Edge MUST NOT raise local_event when provisional (sensor_check still allowed)."
  replace_rule: "same based_on_ts with newer issued_at replaces the stored forecast"
  not_ready: "HTTP 202 → retry every 60 s until HH:20 IST"
  edge_uses: "aqi, aqi_lower, aqi_upper for h=1..3 in decision logic; gas_subindex for AQI_full; neighbour_pm_estimate when PM failed"
```

### S16.4 GET /api/v1/manifest?node_id= (model / calibration / schema updates)

```json
{
  "manifest_version": 17,
  "schema": {"version": 1, "hash": "sha256:...", "url": "...", "sha256": "..."},
  "calibration": {"version": "2026-10-colo-v1", "url": "...", "sha256": "..."},
  "models": [
    {"role": "A", "arch": "rf",  "version": "rf-A-2026-10-05",  "schema_hash": "sha256:...", "url": "...", "sha256": "...", "size": 312000},
    {"role": "A", "arch": "cnn", "version": "cnn-A-2026-10-05", "schema_hash": "sha256:...", "url": "...", "sha256": "...", "size": 98000, "arena_bytes": 48000},
    {"role": "B", "arch": "rf",  "version": "rf-B-2026-10-05",  "schema_hash": "sha256:...", "url": "...", "sha256": "...", "size": 280000},
    {"role": "B", "arch": "cnn", "version": "cnn-B-2026-10-05", "schema_hash": "sha256:...", "url": "...", "sha256": "...", "size": 90000,  "arena_bytes": 46000}
  ],
  "active": {"A": "cnn", "B": "rf"},
  "rollout": {"cohort": "canary", "percent": 20},
  "_note": "manifest change is also announced on MQTT topic aqi/v1/nodes/{node_id}/manifest (payload = {manifest_version, manifest_etag}); node then GETs /api/v1/manifest",
  "config": {"trend_threshold": 5, "R_dust": 1.5, "alerts_enabled": true}
}
```

### S16.5 Node registry record (created at onboarding; cloud-owned, edge supplies facts)

```yaml
node_registry_entry:
  node_id: "delhi-n01"
  lat: 28.6139
  lon: 77.2090
  height_agl_m: 12
  site_notes: "rooftop, 3 m from parapet, open to NW"
  sensors: {pm: SPS30, th: HDC1080, baro: BMP280, wind: "kit-X", rain: "board-Y", lux: BH1750, gas: [MQ-7, MQ-135, SGP41]}
  wind_quality: {sheltering_factor_est: 0.6, trusted_for_graph_edges: false}
  install_date: "2026-10-01"
  burn_in_until: "2026-10-03"
  calibration_version: "default-kappa-0.4"
  token_id: "..."
```

### S16.6 POST /api/v1/events (edge → cloud, immediate)

```json
{
  "node_id": "delhi-n01",
  "event_id": "7f1c2a9e-3b0d-4c1e-9a55-1c2d3e4f5a6b",
  "type": "local_event",
  "status": "open",
  "ts_detected": "2026-10-01T05:41:12Z",
  "base_ts": "2026-10-01T04:30:00Z",
  "severity": "medium",
  "evidence": {
    "edge_aqi": [205, 214, 222], "cloud_aqi": [199, 202, 204],
    "cloud_lower": [190, 191, 190], "cloud_upper": [208, 214, 219],
    "horizons_outside": 2, "sraw_nox_z": 3.8,
    "pm25_max_minute": 180.4, "pm25_hour_mean": 88.2, "mq7_delta_sigma": 2.1
  },
  "fw_version": "1.3.0",
  "model_version": "cnn-A-2026-10-05"
}
```

```yaml
event_rules:
  types: [local_event, sensor_check, pm_sensor_fault]
  status: [open, update, closed]
  idempotency: "event_id generated on device (UUIDv4); server upserts on (node_id, event_id)"
  offline: "queue in LittleFS; send oldest-first on reconnect before hourly backlog"
  retry: "exponential backoff 5 s → 5 min, max 24 h"
  severity_rule: {high: "horizons_outside==3 AND (sraw_nox_z>4 OR pm25_max_minute>3×mean)", medium: "local_event default", low: "sensor_check"}
  response_200: '{"accepted": true, "cloud_confirmed": null|true|false}'
```

### S16.7 MQTT downlink (cloud → edge push)

```yaml
mqtt:
  broker: "MQTT 3.1.1/5 over TLS, port 8883 (Mosquitto or EMQX)"
  client_id: "{node_id}"
  auth: "username = node_id, password = node token (same token as HTTPS)"
  session: {clean_session: false, keepalive_s: 60, reconnect_backoff: "1 s → 60 s"}
  acl: {subscribe: "aqi/v1/nodes/{node_id}/#", publish: "aqi/v1/nodes/{node_id}/status only"}
  topics:
    forecast: {topic: "aqi/v1/nodes/{node_id}/forecast", qos: 1, retained: true, payload: "exact JSON of S16.3"}
    manifest: {topic: "aqi/v1/nodes/{node_id}/manifest", qos: 1, retained: true, payload: '{"manifest_version": 17, "manifest_etag": "W/\"m-17\""}'}
    command: {topic: "aqi/v1/nodes/{node_id}/command", qos: 1, retained: false, payload: '{"cmd": "upload_now|sync_time|rollback_model|reboot|recheck_manifest", "id": "uuid", "args": {}}'}
    status: {topic: "aqi/v1/nodes/{node_id}/status", qos: 1, retained: true, online_payload: '{"state":"online","fw":"1.3.0","models":{...},"ts":"..."}', lwt_payload: '{"state":"offline"}'}
  edge_handling:
    - "validate forecast JSON schema; ignore if based_on_ts in the future (C14)"
    - "store newest per based_on_ts; trigger S15 re-evaluation"
    - "commands are idempotent by id; acknowledge by including last_command_id in next record sys block"
```

---

## [S17] CLOUD-INTEGRATION CONSIDERATIONS (what the hybrid needs; edge-side responsibilities marked EDGE)

```yaml
integration_checklist:
  C01_time_alignment:
    rule: "Both sides bucket by IST clock hour; ts = UTC of IST hour start (I02)."
    edge: "EDGE computes buckets from RTC/SNTP UTC + 05:30 offset; no DST in India."
    cloud: "must convert CPCB From-To (IST) and Open-Meteo (UTC-hourly) to the same buckets (S09 alignment)."
    test: "shared fixture: 48 h of timestamps → identical buckets in firmware, Python, cloud."
  C02_same_aqi_code:
    rule: "One CPCB implementation + tests/aqi_vectors.json used by edge firmware, training code, cloud."
    why: "event detection compares edge AQI vs cloud AQI; any formula drift creates false 'local events'."
  C03_cloud_uses_corrected_pm_from_node:
    rule: "Cloud graph inputs for our nodes = node's pm25_corr/pm10_corr with same calibration_version; cloud also stores raw to reprocess."
    edge: "EDGE always sends raw + proc + calibration_version + baselines."
  C04_masking_quality:
    rule: "Cloud MUST mask node inputs flagged filled/missing/sensor_fault/warming; never treat filled as observed."
    edge: "EDGE sends per-variable flags."
  C05_wind_quality:
    rule: "Rooftop wind is often sheltered/turbulent. Cloud SHOULD build wind edges from Open-Meteo wind at node coords, using node wind only if registry.trusted_for_graph_edges=true."
    edge: "EDGE sends vector-averaged wind + registry sheltering estimate."
  C06_same_targets:
    rule: "Teacher predicts ln-change of PM2.5/PM10 per I04 for h=1..6 and also returns absolute pm and aqi; edge uses h=1..3 subset for KD and comparison."
  C07_freshness_semantics:
    rule: "Cloud response carries based_on_ts and node_inputs_complete. Edge compares only when based_on_ts == its record ts and inputs complete; otherwise display-only."
  C08_calibrated_intervals_per_node:
    rule: "Disagreement test uses cloud [lower, upper]. Intervals MUST be calibrated per node (e.g. conformal, target 90%) and widened for new nodes (few residuals)."
    new_node: "while a node has < 200 calibration residuals per horizon (≈ 8–14 days), cloud sends pooled intervals ×1.5 with interval_quality='provisional'; edge then suppresses local_event (only sensor_check). Cloud doc [C08.8b]."
  C09_teacher_store_contract:
    rule: "Teacher outputs per S12.2, out-of-fold, versioned, leakage rules respected."
  C10_model_lifecycle:
    rule: "Students trained in cloud/offline pipeline → registry → manifest → OTA (S18). Each model carries schema_hash; edge rejects mismatches."
  C11_calibration_lifecycle:
    rule: "calibration.json versioned; cloud reprocesses history from raw on change; retraining only on reprocessed data."
  C12_node_offline_in_graph:
    rule: "Cloud graph must run with missing nodes (mask). Edge is unaffected (I10)."
  C13_cloud_or_cpcb_down:
    rule: "If CPCB feed down → cloud omits gas_subindex (edge shows PM-based AQI). If cloud down → edge fallback per S15."
  C14_clock_skew:
    rule: "Edge rejects cloud responses with based_on_ts in the future; cloud rejects records with ts > now+1h or older than 30 days (unless backlog flag)."
  C15_backlog_semantics:
    rule: "Backlogged records (after outage) are ingested for storage/training but not used to issue past forecasts; cloud re-runs teacher on backlog for teacher_store."
  C16_monitoring_feedback:
    edge: "EDGE sends its own forecasts so cloud can score students online (S19)."
  C17_onboarding_new_node:
    steps:
      - "flash firmware + token; register S16.5 entry (coords, height, photos)"
      - "48 h burn-in (warming flags); default calibration"
      - "cloud adds node to graph with edges from coordinates; interim: node excluded from teacher training until ≥7 days data"
      - "edge runs global student models (trained on synthetic + other nodes) until node-specific fine-tune after ≥14 days"
  C18_security:
    rule: "TLS, per-node tokens, signed OTA images, SHA-256 on every downloaded artifact (S20)."
  C19_bandwidth:
    note: "≈1.5–2 KB per record JSON; 24/day ≈ 50 KB/day/node; gzip for backlog."
  C20_evaluation_coupling:
    rule: "Skill-retained metric (edge skill / cloud skill) computed on identical hours, identical targets, identical AQI code."
  C21_events:
    rule: "Edge pushes events immediately (S16.6); cloud stores, re-checks server-side (cloud_confirmed), streams to frontend. Events are never training labels; they feed evaluation E11."
  C22_mqtt_downlink:
    rule: "Forecast/manifest/command are pushed via MQTT (S16.7); payload identical to HTTP. Node presence via LWT status topic. Uplink stays HTTPS (idempotent, backlog-friendly)."
  C23_flagged_rows_are_kept:
    rule: "Cloud stores filled/fallback/sensor_fault rows with flags. They are valid INPUTS for Model B training (real non-PM sensors, real hours_since_pm_lost) but never TARGETS."
```

---

## [S18] OTA AND MODEL LIFECYCLE

```yaml
ota:
  firmware: "ESP-IDF dual app partitions, signed images, auto-rollback if new app fails health check within 10 min"
  artifacts: "schema, calibration, models downloaded to temp → verify sha256 + schema_hash → atomic rename → keep previous for rollback"
  check_interval: "on MQTT manifest notify (S16.7); also every 6 h and whenever a forecast carries a changed manifest_etag"
  activation: "new models activate at next hour boundary; log active versions in every record"
  rollback_triggers: ["parity self-test fails (10 stored vectors per model)", "inference exception", "latency > 2× expected", "cloud instructs rollback"]
  rollout: "canary cohort (≥1 node) 48 h → all nodes if online metrics not worse"
lifecycle:
  retrain_cadence: weekly
  promote_rule: "new student promoted only if RMSE on last real week ≤ current model and ≥ persistence skill"
  model_ids: "{arch}-{role}-{date}; schema_hash; teacher_version used; training data range; kd_variant; alpha"
```

---

## [S19] MONITORING AND DRIFT

```yaml
monitoring:
  per_node_daily:
    - "edge forecast RMSE vs realised PM (h=1..3), rolling 7 days"
    - "cloud forecast RMSE vs realised, same hours"
    - "skill vs persistence"
    - "interval coverage of cloud intervals at the node"
    - "local_event / sensor_check counts"
    - "data completeness per sensor; flag rates (high_humidity, dust, fault)"
    - "gas baseline drift (Rbase7, Rbase135, base_voc, base_nox trends)"
  alerts_to_developer:
    - "edge skill < 0 for 3 consecutive days → investigate/retrain"
    - "completeness < 80% for any sensor for 24 h"
    - "MQ baseline drift > 30% in 30 days → recalibrate/replace"
    - "interval coverage < 80% → cloud recalibration"
```

---

## [S20] SECURITY

```yaml
security:
  tls: "HTTPS only; pinned CA bundle"
  device_identity: "per-node token in encrypted NVS (flash encryption SHOULD be enabled)"
  ota: "signed firmware (secure boot v2 SHOULD); sha256 for every artifact"
  no_secrets_in_logs: true
  local_ui: "no network-exposed services on the node"
```

---

## [S21] EVALUATION AND EXPERIMENTS (paper)

```yaml
evaluation:
  test_data: "held-out latest real period per node (time split), h=1..3"
  metrics:
    - "RMSE, MAE on PM2.5, PM10 (µg/m3) and AQI"
    - "skill = 1 - RMSE_model/RMSE_persistence"
    - "category accuracy"
    - "band-crossing alerts: POD, FAR, CSI"
    - "Diebold–Mariano vs persistence; paired t-test KD vs no-KD (Shmuel protocol)"
    - "on-device: flash, peak arena, latency, energy"
experiments:
  E1: "architectures: persistence, diurnal persistence, RF, GDC-CNN (, CNN-GRU)"
  E2: "distillation: no_KD vs out_KD (α grid) vs full_KD (CNN)"
  E3: "teacher-quality precondition: teacher vs student RMSE"
  E4: "skill retained = edge skill / cloud skill per horizon (C20)"
  E5: "bootstrap: synthetic-only vs real-only vs synthetic+fine-tune"
  E6: "PM failure: Model A vs Model B vs last-value persistence vs hours since loss"
  E7: "precision: FP32 vs static INT8 (accuracy, latency, size) — cf. Na et al."
  E8: "GDC ablation: no dilation, GLU→ReLU, depthwise→standard conv — cf. Na et al."
  E9: "lookback 24 vs 48"
  E10: "outage simulation 10/20/30% hours: forecast availability & accuracy, cloud-only vs cloud+edge"
  E11: "local events: hand-check flagged events against notes/photos/news"
  E12: "sensor physics: raw vs κ-corrected PM vs co-located reference"
reporting: "all tables/plots produced by scripts in eval/ directly from logs + teacher_store"
```

---

## [S22] MILESTONES AND ACCEPTANCE TESTS

```yaml
milestones:
  M1_hardware:
    done_when: ["I2C scan shows 0x23,0x40,0x59,0x68,0x76", "SPS30 streaming", "MQ-7 heater cycle verified with meter", "24 h stable run, no resets"]
  M2_pipeline:
    done_when: ["minute/hour aggregates + flags", "κ correction + dust flag", "CPCB AQI passes tests/aqi_vectors.json", "Python reference matches firmware on recorded data (AQI diff < 0.5)", "IST bucket fixture passes (C01)"]
  M3_logging_comms:
    done_when: ["records match S16.2 schema", "store-and-forward survives 24 h Wi-Fi outage + reboot", "RTC keeps time offline", "forecast poll + stub server works", "MQTT forecast push received + LWT status visible", "event POST idempotent + queued when offline"]
  M4_data_simulator:
    done_when: ["base table built (S09)", "simulator (S10) with unit tests", "feature_schema.yaml final + hash", "virtual-node shards generated"]
  M5_rf_students_no_kd:
    done_when: ["A and B trained", "beat persistence on synthetic test", "exported; device parity < 1e-4", "size < budget"]
  M6_cnn_students_no_kd:
    done_when: ["toy op-check on device passed", "A and B trained", "FP32 on device parity < 1e-4", "latency + arena measured"]
  M7_distillation:
    done_when: ["interim teacher trained; OOF predictions in teacher_store", "precondition checked (E3)", "out_KD RF + CNN trained", "full_KD once graph teacher provides feat_seq/attn"]
  M8_decision_ota:
    done_when: ["A/B switching incl. PM-dropout scenarios", "cloud/edge display selection", "event flags with stub cloud intervals", "alerts + rate limit", "OTA model update + rollback tested"]
  M9_colocation_finetune:
    done_when: ["κ, R_dust fitted → calibration.json", "P1 fine-tune done", "real-data evaluation run"]
  M10_experiments:
    done_when: ["E1–E12 produced by scripts"]
```

---

## [S23] REPOSITORY LAYOUT

```
edge/
  firmware/                      # ESP-IDF project
    main/                        # tasks, config, boot
    components/
      sensors/                   # sps30, hdc1080, bmp280, bh1750, sgp41, mq7, mq135, wind, rain, ds3231
      pipeline/                  # minute/hour aggregation, κ correction, dust flag, flags, baselines
      aqi/                       # CPCB AQI + trend roll (mirrors shared/aqi)
      features/                  # generated from schema/feature_schema.yaml
      models/                    # RF generic evaluator, TFLM runner, instance norm/patching in C
      decision/                  # model selection, cloud/edge choice, events, alerts
      storage/                   # LittleFS logs, NVS buffers
      comms/                     # HTTPS client, ingest/forecast/manifest, store-and-forward
      ota/                       # firmware + artifact updates, rollback, parity self-test
      ui/
  shared/
    aqi/                         # Python reference CPCB implementation
    tests/aqi_vectors.json       # shared with cloud
    tests/ist_bucket_fixture.json
    contracts/                   # JSON Schemas for S16.2–S16.5 (shared with cloud)
  schema/feature_schema.yaml
  data/                          # CPCB/OpenAQ + Open-Meteo downloaders, cleaning, base table
  sim/                           # forward sensor models, virtual-node generator
  training/
    rf/                          # training, α grid, emlearn export
    cnn/                         # GDC-CNN, losses (pred/feat/afd), TFLite export
    teacher_interim/             # LightGBM teacher + OOF predictions → teacher_store
    distill/                     # KD dataset builder from teacher_store
    model_b/                     # PM-dropout dataset builder
  eval/                          # metrics, DM test, paired t-tests, plots, experiment runners
  tools/                         # parity checker, log parser, calibration fitter, stub cloud server
  docs/
```

---

## [S24] TEST PLAN

```yaml
tests:
  unit_firmware: ["sensor frame parsing (SPS30 SHDLC CRC)", "Rs computation", "vector wind mean", "κ correction", "AQI vectors", "IST bucketing", "flag bitmasks", "feature builder vs Python golden"]
  unit_python: ["simulator monotonicity/ranges", "feature builder == firmware golden", "target construction", "PM-dropout builder", "OOF teacher join (no leakage: assert teacher_train_end < student_test_start)"]
  parity: ["RF device vs Python 1,000 vectors", "CNN device vs Python 1,000 vectors"]
  hil_hardware_in_loop:
    - "replay 7 days of logged minute data into hour_task via UART → compare records to Python"
    - "simulate SPS30 failure (unplug) → A→filled→B transitions and AQI validity countdown"
    - "Wi-Fi off 24 h → backlog upload order and idempotency"
    - "stub cloud: stale/complete/incomplete forecasts → display source + event logic"
    - "OTA: good model, corrupted sha, schema mismatch, rollback"
  contract_tests: "validate all payloads against shared/contracts JSON Schemas in CI"
```

---

## [S25] ASSUMPTIONS AND OPEN ITEMS

```yaml
assumptions:
  - "Anemometer kit includes wind vane; factors from kit datasheet."
  - "SGP41 tick directions and MQ exponents in simulator are placeholders until first real week."
  - "No backup PM and no INA219 in base build (both recommended)."
  - "CPCB breakpoints and Severe-band interpolation limit to be verified from official sources."
  - "Graph teacher is separate; interim LightGBM teacher used until it exists."
  - "Open-Meteo variable names and half-hour-timezone handling to be verified."
  - "κ default 0.4 and R_dust 1.5 are starting values; Delhi κ likely higher."
open_items:
  - "Decide RF-serialised-file vs compiled-in trees (affects OTA path)."
  - "Decide whether CPCB stations are also graph nodes during teacher training (cloud decision; affects teacher_store coverage)."
  - "Choose display hardware."
```

---

## [S26] REFERENCES (journal) AND WHAT EACH SUPPORTS

```yaml
references:
  - {key: Na2026, cite: "S. Na, Y. Cho, Y. Baek, IEEE Access 14:30373–30386, 2026, doi:10.1109/ACCESS.2026.3667131", supports: "GDC-CNN student, heterogeneous KD (L_feat, L_afd, curriculum), ESP32-S3 FP32 deployment"}
  - {key: Shmuel2026, cite: "A. Shmuel, T. Lazebnik, O. Glickman, IEEE Access 14:19701–19712, 2026, doi:10.1109/ACCESS.2026.3657793", supports: "output KD into non-neural student; teacher-must-beat-student precondition; paired-test protocol"}
  - {key: Mazinani2025, cite: "A. Mazinani et al., IEEE Access 13, 2025, doi:10.1109/ACCESS.2025.3603920", supports: "embedded PM2.5 forecasting, CNN-BiGRU, PTQ on STM32"}
  - {key: Wardana2023, cite: "I. N. K. Wardana, S. A. Fahmy, J. W. Gardner, IEEE Sensors Letters 7(11), 2023, doi:10.1109/LSENS.2023.3315249", supports: "TinyML predictor + imputer on MCU"}
  - {key: MoralesGarcia2025, cite: "J. Morales-García et al., IEEE JBHI 29(12), 2025, doi:10.1109/JBHI.2024.3508466", supports: "heterogeneous graph + synthetic data for small datasets (contrast with our physics simulator)"}
  - {key: Patel2024, cite: "Patel et al., Atmospheric Measurement Techniques, 2024", supports: "κ-Köhler humidity correction for low-cost PM"}
  - {key: Crilley2018, cite: "Crilley et al., Atmospheric Measurement Techniques, 2018", supports: "κ-Köhler correction formula form"}
  - {key: Kaur2026, cite: "Kaur et al., Atmospheric Measurement Techniques, 2026", supports: "dust detection/correction from sensor-only data"}
  - {key: Ouimette2023, cite: "Ouimette et al., Aerosol Science and Technology, 2023", supports: "optical sensor physics; coarse undersizing"}
  - {key: Gautam2025, cite: "Gautam et al., Atmosphere, 2025", supports: "SPS30 behaviour in dust"}
  - {key: Chen2022, cite: "Chen et al., Environmental Science & Technology, 2022", supports: "Delhi aerosol water → local κ needed"}
  - {key: Oancea2026, cite: "Oancea, Electronics, 2026", supports: "KD gains across model families incl. trees (honest limits)"}
  - {key: Ali2025, cite: "D. Ali et al., Environmental Pollution 388, 2026, doi:10.1016/j.envpol.2025.127324", supports: "Delhi AQI dominated by own past values; temperature strongest weather driver"}
  - {key: ZenderSwiercz2024, cite: "E. Zender-Świercz et al., Atmospheric Research 312, 2024, doi:10.1016/j.atmosres.2024.107733", supports: "non-linear, pollution-level-dependent T/RH effects; PMS7003 humidity sensitivity"}
```
