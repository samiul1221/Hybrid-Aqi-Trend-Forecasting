# What the ablation study found (and what to do about it)

> **Naming note:** this file describes version 1, where the big cloud graph model was called the "teacher" and the on-device model the "student". The current design (v3) is in `docs/ARCH_V3.md`; it no longer treats the box model as a copy of the cloud model.

All numbers come from the **latest successful Kaggle run** (`aqi-train`, finished 9 Oct 2026, no failed runs;
files in `data/5_models/kaggle/`). Test period: Jan 2025 – Sep 2026, never used for training.

## How to read the numbers

- **Skill %** = how much better than the lazy guess "PM stays the same" (persistence).
  0 % = no better, 20 % = error 20 % smaller. Higher is better.
- **±** = how much the skill moves when the same model is trained again with another random start (3 runs).
  A difference smaller than about 2 × this spread can be luck.
- **Seen stations** = the 27 used for training (but test years). **New stations** = 5 stations never used.
- An **ablation** = remove or add one part and see how much the result changes. If removing a part changes
  nothing, that part is not doing useful work.
- Everything is on **simulated** sensor data (see `docs/HOW_IT_WORKS.md`): real-world checks come later.

---

## Finding 1 — Neighbouring stations are the teacher's real strength

| teacher | what it has | skill (1-6 h) |
|---|---|---|
| T0 | only its own station's history | 21.7 % |
| T1 | + other stations (graph) | **23.6 %** |
| T2 | + wind direction on the graph | 23.6 % |
| T3 | + physics block (transport) | 23.8 % |
| T3, trained 3 times | | 23.7 – 23.8 % |

**What it means:** adding the other stations (T0 → T1) is the one clear jump (+1.9 points, far above the
0.05-point spread). Wind and physics add **0.2 points or less**, which is the same size as the noise between
runs. So in this data, *knowing what the neighbours see* matters; *knowing the wind direction* does not add more.

**Why could wind not help?** The wind for each station comes from one weather-grid point (ERA5, ~25 km), so it
is smooth and almost the same for all of Delhi. The graph attention can already learn "which neighbours matter"
from the data itself, without the wind hint.

**What to do:**
1. Report it as a finding: "the graph helps, the wind prior and ADR do not add measurable skill here".
2. For the live cloud, **T1 is enough** (simpler, same accuracy).
3. Optional research step: try wind from the real boxes' fans or a finer weather source; test again.

## Finding 2 — The real future weather matters more for longer forecasts

| skill, PM10 | +1 h | +3 h | +6 h |
|---|---|---|---|
| T3 (got the real weather of the next 6 h) | 12.2 % | 23.2 % | 28.8 % |
| T3 without future weather | 12.2 % | 22.1 % | **26.0 %** |

**What it means:** for the next hour the future weather does not matter; at 6 hours it is worth ~3 points.
Our teacher was trained with the **real** future weather (a "hindcast"), which a live system does not have.

**What to do:**
1. Label the teacher results as **hindcast** in the report (already done in the notebook).
2. In the live cloud, use weather **forecasts** and measure the loss; expect about 1 point on average, up to ~3 at 6 h.
3. Later: retrain with archived Open-Meteo forecasts (available from ~2022) instead of the real weather.

## Finding 3 — Distillation helps the student, but only a little

| model (1-3 h, same test windows) | skill | ± |
|---|---|---|
| straight-line model (linear) | 11.4 % | |
| S0: student alone | 14.0 % | 0.2 |
| LightGBM (classic ML) | 14.2 % | |
| **S2-T0: student taught by teacher T0 (best)** | **15.0 %** | 0.1 |
| S2-T3: taught by the full teacher | 14.8 % | 0.2 |
| teacher T3 itself | 20.8 % | |

**What it means:**
- Teaching is a **real** gain (+0.9 points, about 5 × the spread), visible at every horizon and every season.
- But the student keeps only **12 – 24 %** of the teacher's advantage over S0
  ("kept" = (student − S0) / (teacher − S0)). The gap S0 → teacher is 6.8 points; the student closes ~1 of them.
- **Why:** the teacher's edge is *seeing other stations* (Finding 1). A single box cannot see them, so no amount
  of teaching can give it that information. The student can only learn better ways to read its own history.
- The tiny student (71 KB) equals LightGBM and beats the straight line: the network itself is fine.

**What to do (this is the most important design conclusion):**
1. **Use the cloud for what only the cloud can do.** When the box is online, show the teacher's forecast
   (20.8 %); when offline, use its own student (15.0 %). This *is* the edge–cloud hybrid of the project title.
   The firmware already receives the cloud forecast and shows "edge − cloud"; the missing piece is the cloud
   endpoint in `backend/`.
2. Keep the distilled student (S1/S2) as the offline model: it is better than S0 for free.
3. Optional: give the student a cheap hint of the neighbours (for example the cloud's last forecast as an extra
   input when it is fresh) — this would change the model, so only as a separate experiment.

## Finding 4 — What the student should copy from the teacher

| student (all taught by T3) | what it copies | skill vs S0 |
|---|---|---|
| S1 | the teacher's forecast | +0.8 |
| S2 | + the teacher's "thinking" over time (features + attention) | +0.8 |
| S3 | + the teacher's neighbour state | +0.5 |
| S4 | + the teacher's physics state | +0.4 |
| S4 without copying the forecast | | +0.1 |
| S4 without the time features **and** attention | | −0.2 |

**What it means:**
- **Copying the forecast** is the main source of the gain (remove it: +0.4 → +0.1).
- Copying the teacher's time features / attention helps a little (remove both: +0.4 → −0.2).
- Copying the **neighbour and physics states makes it slightly worse** (S3, S4 < S2): the student is asked to
  imitate information it cannot see, which only distracts it.

**What to do:** use S1 or S2 as the final student (S2-T0 / S1-T0 are best). Report S3/S4 as a negative result:
"distilling spatial/physical states into a single-station student does not help".

## Finding 5 — Which sensors matter (sensor ablation)

Skill lost when one sensor is removed from the student (S0, mean of 3 runs, spread ~0.1-0.2):

| sensor removed | skill lost (training stations) | new stations |
|---|---|---|
| MQ-7 (CO) | **−0.45** | **−0.68** |
| 2 wind fans | **−0.26** | −0.34 |
| SGP41 (VOC / NOx) | −0.14 | −0.08 |
| BMP280 (pressure) | −0.09 | −0.18 |
| rain plate | −0.05 | +0.04 |
| MQ-135 | −0.02 | +0.03 |
| BH1750 (light) | −0.01 | −0.05 |

**What it means:** the **MQ-7** (traffic CO) and the **wind fans** clearly earn their place. SGP41 and BMP280 help
a little. **MQ-135, light and rain add nothing measurable** — the MQ-135 mostly reacts to humidity and
temperature in outdoor air (we saw this in notebook 02), and rain/light information is already in the other
inputs.

**What to do:** keep all sensors for the real test (they are cheap), but **check this again with real data** before
removing any: the simulator decides how informative each sensor is, so this result is only as good as the
simulator. If it holds, the MQ-135 and the rain plate are the first to drop (saves power and pins).

## Finding 6 — What the box would really show (deployment view)

| | starts from the true PM now | starts from the box's own PM sensor |
|---|---|---|
| error of the shown PM (µg/m³) | 40.6 | **82.6** |
| average bias | −1.5 | **−38** (shows too low) |
| AQI colour right | 71 % | **43 %** |

**What it means:** the forecast of the *change* is good, but the box starts from its own uncalibrated PM sensor
(after its humidity correction). That starting value is off, so the shown numbers are off.

**What to do (must do before anyone reads the screen):** a **co-location calibration** — run the box next to a
CPCB station for 2 – 4 weeks, find the factor between the two, set `PM_CALIBRATION` in
`firmware/aqi_node/aqi_node.h`. Better later: a small calibration per humidity range.

## Finding 7 — Smaller observations

- **Forecasting further is relatively easier to beat persistence:** PM2.5 skill goes 12.7 % (+1 h) → 24.7 % (+3 h)
  → 32.2 % (+6 h) for the teacher, because "no change" gets worse with time.
- **Monsoon is the hardest season** (teacher 17 %, student 9 – 10 %), winter and summer the easiest (~29 % / ~19 %).
  Rain makes PM change suddenly; our rain input is coarse.
- **New stations:** the teacher scores as well or better there (25.9 % vs 23.8 %): it learned general rules, not
  station-specific ones. Good sign for adding real boxes in new places.
- **High-pollution hours** (true "Poor" or worse): error 76 µg/m³ for persistence, 65 for the student,
  60 for the teacher. The student catches ~85 % of "Poor" hours with ~13 % false alarms.
- **Uncertainty band:** 75 % of true values fall inside the teacher's 10-90 % band (should be 80 %): slightly
  over-confident. Fix: widen it using the 2024 validation year (simple calibration).
- **Tuning:** learning rate 0.001 with width 32 was best, but all 6 settings were within 0.0007 of each other:
  the student is not sensitive to these settings.

---

## Priority list

| # | do this | why | effort |
|---|---|---|---|
| 1 | Build the box, co-locate with a CPCB station 2-4 weeks, set `PM_CALIBRATION` | without it the shown AQI is right only 43 % of the time (Finding 6); also tests the simulator | hardware + weeks |
| 2 | Cloud endpoint in `backend/` (teacher T1 + live Open-Meteo forecasts) | the cloud forecast is clearly better (Findings 1, 3); the box already shows it | medium |
| 3 | Fine-tune the student on the real box data; re-run the sensor ablation on real data | everything so far is simulated (Finding 5) | small, after #1 |
| 4 | Calibrate the teacher's uncertainty band on 2024 | 75 % instead of 80 % (Finding 7) | small |
| 5 | Train Model B (student for a broken PM sensor) | the box has no fallback yet | one Kaggle run |
| 6 | Report: graph helps, wind/ADR do not here; distillation +0.9 points; S3/S4 negative; hindcast label | these are the honest research results | writing |
| 7 | (optional) teacher with archived weather forecasts; finer wind | turns the hindcast into a real forecast setup | research |
