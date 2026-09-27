# PROGRESS

Last updated: 2026-09-26

## Phase-1 milestones

| # | Milestone | Status |
|---|---|---|
| P0 | First actions: specs/papers moved to `docs/`, CLAUDE.md, PROGRESS.md, ADR 0001 | done |
| P1 | Repo scaffold + tooling + placeholders | done — uv workspace (Py 3.11.13, pinned deps), ruff/mypy/pytest green, host g++ test runner, Arduino sketch compiles for ESP32-S3 (ADR 0002) |
| P2 | `shared/`: CPCB AQI + ≥50 vectors, IST bucketing + fixture, feature_schema + loader + hash, JSON Schemas [S16.2]–[S16.7], C-header codegen | done — 61 AQI vectors, 409-row IST fixture (vs zoneinfo), schema hash `sha256:49657c55…`, 9 contract schemas + examples, 2 generated headers; 53 pytest + host C++ test green (ADR 0003, 0004) |
| P3 | Data downloaders + base table + data card | done — keyless OpenAQ S3 archive (33 stations) + Open-Meteo ERA5 (rate-limit safe); base table 2018–2022 + 2025–26; `docs/research_notes/data_card.md` (ADR 0005, 0006) |
| P4 | Simulator with researched parameters + unit tests + virtual-node shards | done — 30 stations × 3 units = 90 virtual nodes, 4.05M unit-hours; SPS30 winter slope vs BAM median 1.64 (P10–P90 1.46–1.85) vs IIT-Delhi 1.46–1.86, R² 0.85; `sim_data_card.md` (ADR 0007) |
| P5 | Python feature builder (golden fixtures) + target builder + Model B PM-dropout builder | done — `data/datasets/feat-v1`: 1.16M Model A / 1.09M Model B windows (3.3k real-outage); A: train 329k, val 22k, test 567k (+ hold-out stations); golden fixture for firmware (ADR 0008) |
| P6 | Baselines + RF students A/B (no_KD) + [S21] metrics + emlearn export + host C parity < 1e-4 | done — RF-A 362 KB, parity 1.3e-7, test skill PM2.5 h1/h2/h3 +0.09/+0.13/+0.17 (hold-out stations +0.12/+0.14/+0.18), beats diurnal persistence; RF-B 232 KB, parity 1.4e-7, skill +0.17 vs last-valid persistence; own blob + C evaluator (ADR 0009); label Hampel cleaning |
| P7 | GDC-CNN A/B per [S13.3]/[S13.4] + TFLite FP32 export + 1-block TFLM op check | todo |
| P8 | teacher_store + interim LightGBM teacher + OOF + out_KD/full_KD + precondition check | todo |
| P9 | ZenML pipelines wiring P3–P8 | todo |
| P10 | Firmware (Arduino sketch, ADR 0002) components + host tests against shared fixtures + mock_cloud | todo |

## Spec acceptance checklist ([S22]) → phase-1 milestone

- [ ] M1 hardware — I2C scan 0x23,0x40,0x59,0x68,0x76; SPS30 streaming; MQ-7 heater cycle verified with meter; 24 h stable run (**needs hardware**, after P10)
- [ ] M2 pipeline — minute/hour aggregates + flags; κ correction + dust flag; AQI passes vectors; Python vs firmware AQI diff < 0.5; IST fixture passes (P2, P10)
- [ ] M3 logging/comms — records match [S16.2]; store-and-forward survives 24 h outage + reboot; RTC offline; forecast poll + stub server; MQTT push + LWT; event POST idempotent + queued offline (P10; outage/reboot parts need hardware)
- [ ] M4 data/simulator — base table [S09]; simulator [S10] with unit tests; ~~feature_schema final + hash~~ (P2 done; final once P5/P7 confirm); virtual-node shards (P3–P4)
- [ ] M5 RF no-KD — A and B trained; beat persistence on synthetic test; exported; parity < 1e-4; size < 400 KB (P6)
- [ ] M6 CNN no-KD — toy op-check on device; A and B trained; FP32 parity < 1e-4; latency + arena measured (P7; on-device parts need hardware)
- [ ] M7 distillation — interim teacher + OOF in teacher_store; precondition checked (E3); out_KD RF + CNN; full_KD when graph teacher exists (P8; full_KD real run is phase 2)
- [ ] M8 decision/OTA — A/B switching; cloud/edge display selection; events with stub cloud intervals; alerts + rate limit; OTA update + rollback (P10)
- [ ] M9 co-location fine-tune — κ, R_dust → calibration.json; P1 fine-tune; real-data eval (**needs deployed nodes**)
- [ ] M10 experiments — E1–E12 produced by scripts (partly phase 1, E4/E10/E11 need cloud/real data)

## Early findings from datasheets (feed into P4 `docs/research_notes/simulator_parameters.md`)

- **SGP41** (Sensirion datasheet v1.0, Table 1): SRAW ticks ∝ log(sensing-layer resistance). VOC: −870 ticks per doubling of ethanol (−710…−1030) ≈ −1255 ticks per e-fold → spec `kv_u ~ U(500,2000)` is consistent. NOx: **+1480 ticks per doubling of NO2 (+950…+2010) ≈ +2135 per e-fold (1370–2900)** → spec `kn_u ~ U(200,800)` is ~3× below the datasheet; sign `s_n = +1` confirmed. Baseline directions in [S08.1] confirmed (VOC clean = high ticks → 90th pct; NOx clean = low → 10th pct). Operating RH 0–90 % non-condensing; SRAW_NOX needs up to 7 days after first power-on to meet spec; conditioning ≤ 10 s; avoid bright light on the sensor.
- **MQ-7**: Hanwei sheet says low phase 1.4 V / 90 s, RL 10 kΩ, Rs 2–20 kΩ at 100 ppm CO, α = Rs(300)/Rs(100) < 0.5 → exponent ≲ −0.63. Winsen v1.3 says 1.5 V / 90 s, α = Rs(300)/Rs(50) ≤ 0.6. Fig. 4 slopes: ≈ −0.011 ln-units/°C and ≈ −0.006 per %RH → inside spec ranges `gT U(-0.02,0)`, `gRH U(-0.01,0)`. Which vendor's part is fitted decides 1.4 vs 1.5 V (TODO_VERIFY on hardware). RH limit < 95 %. Preheat ≥ 48 h (matches [S04.3]).
- **MQ-135** (Winsen v1.4): heater constant 5 V; NH3 curve slope ≈ −0.48, H2 ≈ −0.5 on log–log → spec `b135 U(-0.5,-0.3)` plausible, upper end may need −0.55. Same T/RH shape as MQ-7.

## Open questions for the owner

- **PM2.5 scale in calibration.json:** before co-location the device's corrected PM is ≈ 0.6–0.8× true (SPS30 reads high, κ 0.4 over-corrects). Add a `pm25_scale` next to the spec's `pm10_scale` so co-location can fit it? (ADR 0007)
0. **OpenAQ API key needed** (free: explore.openaq.org → Settings). Put `OPENAQ_API_KEY=...` in `.env`, then run `uv run python -m aqi_ml.data.build all` (up to ≈ 40 stations × 16 params × 96 months ≈ 60k requests; the 2000/h cap makes that ≤ ≈ 30 h, fewer when stations lack sensors; resumable — start with `--max-stations 2` to check).
1. **AQI gap rule** (ADR 0003): the spec's literal rule ("previous band's B_hi as B_lo") gives up to ≈3.3 AQI points more than the official CPCB calculator just above a band start (PM2.5 61 → 104.3 vs 101). Implemented literally. Keep, or switch to the official integer breakpoints? (One-line change + fixture regen.)
2. Resolved by default: PM2.5 = 250 → Very Poor (upper-inclusive); Severe upper limits 380 / 510; all breakpoints TODO_VERIFY against the official CPCB NAQI document.
3. SGP41 `kn_u` range: move to datasheet-based U(1400,2900) per e-fold, or keep the spec value? (Proposed: datasheet value + ADR in P4.)
4. Which MQ-7 module do you have (Hanwei 1.4 V or Winsen 1.5 V low phase), and is its heater hard-wired to 5 V?
5. Delhi fog/monsoon RH regularly exceeds SGP41's 90 % limit: flag SGP41 readings `high_humidity` above 90 % RH? (Spec flags at 95 %.)
6. (P7) How should the 5 scalar Model B extras [S08.2] enter the GDC-CNN? Proposal: broadcast as 5 constant channels over the 24 h window (17 + 5 = 22 channels).
7. arduino-esp32 core 3.3.3 installed, 3.3.11 latest: update before P10?

## Environment notes

- Host: Windows 11, uv 0.8.6 (Python 3.11.13 managed by uv), git 2.50, g++/mingw32-make (MinGW 6.3, C++14 OK for host tests).
- Firmware toolchain: arduino-cli 1.5.1, core esp32:esp32 3.3.3 (latest 3.3.11 — update before P10), libraries already installed: TensorFlowLite_ESP32, Chirale_TensorFlowLite, ClosedCube HDC1080, Adafruit BMP280, RTClib, PubSubClient, ArduinoJson.
- Not installed / not needed in phase 1: ESP-IDF (replaced by Arduino, ADR 0002), CMake (host tests use plain g++), Docker (`infra/docker-compose.yml` written, not run; ZenML uses the local stack).
- `papers/Datasheet/` still holds two MQ manuals locked by a PDF viewer; copies are in `docs/papers/Datasheet/`. Delete `papers/` after closing the viewer.
- Removed duplicate `Improving_Air_Quality_Zoning... (1).pdf` (byte-identical, same sha256).
