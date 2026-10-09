# How the system works (in plain words)

## 1. The idea in one paragraph

A small box with cheap sensors sits on a roof in Delhi. Every hour it guesses how dirty the air will be in the
next 1, 2 and 3 hours (PM2.5 and PM10) and shows it on its screen. It does this **by itself**, on a tiny chip
(ESP32), without internet. To make the small model on the chip smarter, we first train a **big model in the
cloud** (the *teacher*) that sees all 32 monitoring stations of Delhi at once and knows how the wind carries
pollution between them. The small model (the *student*) then learns from the teacher. This is called
**knowledge distillation**.

```
 official CPCB data (8 years, 32 stations) + weather
              │
              ▼
   simulator: "what would OUR cheap sensors have read?"      (notebook 02, simulator.py)
              │
              ▼
   23 numbers per hour per box ("features")                  (notebook 03)
              │
       ┌──────┴───────────────────────┐
       ▼                              ▼
  TEACHER (cloud, big)  ── teaches ─▶ STUDENT (chip, tiny)   (notebook 04, on a Kaggle GPU)
  all stations + wind + physics       one box only, 71 KB
                                              │
                                              ▼
                                ESP32 box: sensors → features → student → screen   (firmware/aqi_node)
```

## 2. The pieces

### 2.1 The box (hardware)
| sensor | measures | used for |
|---|---|---|
| PMS7003 | PM1, PM2.5, PM10 (laser) | the pollution itself |
| HDC1080 | temperature, humidity | weather; humid air makes PM sensors read too high |
| BMP280 | air pressure | weather |
| 2 small fans with a magnet | wind speed along east-west and north-south | wind blows pollution away |
| rain plate | wet / dry | rain washes PM out |
| BH1750 | light | sunshine, mixing of the air |
| MQ-7, MQ-135, SGP41 | gases (CO, traffic exhaust, VOC, NOx) | traffic and burning come with PM |
| DS3231 | battery clock | time of day, survives power cuts |

The board is an ESP32-WROOM-32 (8 MB flash) with a 2.8" colour screen, a button and a buzzer.

### 2.2 The data
- **Truth:** official hourly data from CPCB (the government network), 2018 – Sep 2026, 32 stations.
- **Weather:** ERA5 (a global weather archive) via Open-Meteo.
- We have **no** years of data from our own box yet. So we **simulate** it.

### 2.3 The simulator (`simulator.py`, notebook 02)
It takes the true air and weather of a station and works out what *our* cheap sensors would have shown:
cheap PM sensors read too high in humid air and miss dust, gas sensors give a resistance (not ppm) that also
changes with temperature and humidity, each unit has its own small errors, sensors sometimes fail. Every
formula comes from a datasheet or a published study (`docs/simulator_sources.md`). It makes 3 virtual boxes per
station = 96 boxes, 8.75 years each.

**Why:** the models must learn from the same kind of messy readings the real box will produce. The clean
government PM is only used as the **answer** to learn, never as an input.

### 2.4 The features (notebook 03)
Every hour, every box turns its readings into 23 numbers: PM (log scale, humidity-corrected), weather, light,
each gas sensor compared with its own "clean air" level of the last week, time of day / week / year, the
current AQI, and whether the PM sensor worked. The model sees the **last 24 hours** of these.

The ESP32 computes the very same numbers in C (`firmware/aqi_node/features.cpp`); `firmware/check_features.py`
proves they match the training data (largest difference 0.0001).

### 2.5 What the models predict
Not the PM itself but its **change**: "in 2 hours PM2.5 will be 15 % higher than now". Changes are easier to
learn than levels and work the same in summer and winter.

### 2.6 The teacher (notebook 04, part A) — TI-STGNN
Looks at all 27 training stations at once, in 5 steps:
1. turns each hour into numbers it can work with,
2. a **Transformer** reads each station's last 24 hours,
3. a **graph** lets each station listen to its neighbours, more to the ones **upwind**,
4. a small **physics** block moves pollution between stations (wind, spreading, local sources),
5. a decoder adds the weather of the next 6 hours and forecasts 1 – 6 hours ahead.

### 2.7 The student (notebook 04, part B) — GDC-CNN
A tiny network (14 278 numbers, 71 KB) that only sees its own box. It learns:
- **S0:** from the true answer only,
- **S1:** + the teacher's forecast,
- **S2:** + how the teacher "thinks" about time (its inner features and attention),
- **S3 / S4:** + the teacher's view of neighbours / physics.

Fairness rule: the teacher's answers for the student's training data come from teacher copies that never saw
those hours ("out-of-fold"), so the student cannot learn the answer through the back door.

### 2.8 On the box (firmware)
Every 10 s it reads the sensors; every hour it averages them, builds the 23 features, runs the student
(TensorFlow Lite) and shows: the CPCB AQI, a chart and a table of the next 3 hours, every sensor's value and
health. If a cloud server is set up later, it also shows how far its forecast is from the cloud teacher's.

## 3. Results (test period 2025 – Sep 2026, never used for training)

**Skill** = how much better than simply saying "PM stays the same" (0 % = no better, higher = better).
"New stations" = 5 stations no model ever saw while training.

### 3.1 Teacher (1 – 6 hours ahead)
| model | skill | skill at new stations | right direction (up / down) | catches "Poor" hours |
|---|---|---|---|---|
| T0: time only | 21.7 % | 23.0 % | 73.9 % | 81.0 % |
| T1: + neighbours | 23.6 % | 25.5 % | 75.7 % | 82.7 % |
| T2: + wind | 23.6 % | 25.4 % | 75.6 % | 81.3 % |
| **T3: + physics (full)** | **23.8 %** | **25.9 %** | **75.7 %** | **82.6 %** |
| T3 without future weather | 22.7 % | 24.6 % | 75.3 % | 81.7 % |

- **Neighbours help clearly** (T0 → T1). Wind and physics add very little: the differences are as small as the
  difference between two training runs of the same model (T3 three times: 23.7 – 23.8 %).
- Knowing the real future weather helps only a little (1 point), so a live system with ordinary weather forecasts
  should lose little.
- The uncertainty band holds 75 % of true values (aim: 80 %): slightly too narrow.

### 3.2 Student (1 – 3 hours ahead, same test windows for every model)
| model | skill | ± (3 runs) | error (µg/m³) | right direction | catches "Poor" hours |
|---|---|---|---|---|---|
| straight-line model (linear) | 11.4 % | | 42.5 | 68.6 % | 84.1 % |
| LightGBM (classic ML) | 14.2 % | | 40.8 | 70.2 % | 84.9 % |
| S0: student alone | 14.0 % | 0.2 | 41.2 | 69.7 % | 84.8 % |
| **S2-T0: student taught by T0 (best)** | **15.0 %** | 0.1 | 40.6 | 70.3 % | 84.7 % |
| S2-T3: taught by the full teacher | 14.8 % | 0.2 | 40.7 | 70.2 % | 84.9 % |
| teacher T3 on the same windows | 20.8 % | | 37.3 | 74.4 % | 86.4 % |

- **Teaching helps, a little:** +0.9 skill points over S0, about 5× the run-to-run spread, so it is real, but
  the student keeps only **12 – 24 %** of the teacher's advantage. Most of the teacher's edge comes from seeing
  other stations, which one box simply cannot see.
- The tiny student is as good as LightGBM and clearly better than a straight line, and it fits on the chip.
- At the 5 new stations the scores are as good or better: the models are not tied to the stations they learned on.

### 3.3 Ablation: what matters?
- **Teaching:** copying the teacher's forecast (S1) gives most of the gain. Copying its "thinking" (S2) adds a bit
  (removing both feature terms from S4 costs 0.5 points). Copying its neighbour / physics state (S3, S4) does
  **not** help more.
- **Sensors** (skill lost when the sensor is removed): MQ-7 0.45 > fans 0.26 > SGP41 0.14 > BMP280 0.09 > MQ-135,
  light, rain ≈ 0. So the MQ-7 and the wind fans earn their place; MQ-135, BH1750 and the rain plate add nothing
  measurable here (in simulated data — check again with real data).

### 3.4 What the box would really show
The numbers above start each forecast from the **true** current PM, to compare models fairly. On the box, the
forecast starts from **its own** PM sensor. Without calibration the shown PM is about 38 µg/m³ too low on
average and the AQI colour is right only 43 % of the time (71 % with a perfect start). **A calibration next to a
CPCB station is essential** (one number: `PM_CALIBRATION` in `firmware/aqi_node/aqi_node.h`).

## 4. Honest limits
1. All box data is **simulated**; real sensors will differ. Next step: put the box next to a CPCB station for a few
   weeks, compare, recalibrate the simulator, fine-tune.
2. The teacher was given the **real** weather of the next 6 hours (archived forecasts do not exist for 2018 – 23):
   a hindcast. Live, it would use weather forecasts (a bit worse; the "no future weather" run shows about 1 point).
3. Wind direction and the physics block did not help measurably; the network of stations did.
4. The forecast is for hourly averages; the AQI shown is the official 24-hour CPCB AQI.

## 5. How to run it
```bash
uv sync                                         # Python 3.11 + packages
# notebooks 01 -> 02 -> 03 on the laptop (data, simulator, features)
uv run python kaggle/kaggle_run.py upload       # send the features to Kaggle
uv run python kaggle/kaggle_run.py train        # notebook 04 on a free Kaggle GPU (~7 h)
uv run python kaggle/kaggle_run.py download     # results + firmware/aqi_node/student_model.h
```
Firmware: open `firmware/aqi_node/aqi_node.ino` in the Arduino IDE (board "ESP32 Dev Module", Flash 8 MB,
partition "8M with spiffs"), create `secrets.h` with your Wi-Fi, upload.

## 6. Where to look
| file | what |
|---|---|
| `notebooks/01_data.ipynb` | download and clean CPCB + weather |
| `notebooks/02_simulator.ipynb` + `simulator.py` | the sensor simulator, sensor by sensor |
| `notebooks/03_dataset.ipynb` | the 23 features, targets, training windows |
| `notebooks/04_train.ipynb` | teacher + student training, all results and plots (runs on Kaggle) |
| `firmware/aqi_node/` | the box: sensors, features, model, screen |
| `docs/DESIGN.md` | every design decision, in short |
| `docs/simulator_sources.md` | where every simulator number comes from |
