"""
Check that the ESP32 code (firmware/aqi_node/features.h) gives the student exactly the input it was trained on.

    uv run python firmware/check_features.py        # needs g++ (MinGW) and data/3_sensors + data/4_features

It feeds one simulated unit's hourly log (data/3_sensors, = what a real node logs) hour by hour into features.h,
keeping only the 194 hours the node keeps, and compares with the notebooks:
  - which hours are usable          (notebook 03, model_a_rows without the targets)
  - the 24 x 23 feature window      (notebook 03 tables in data/4_features)
  - the model input [12, 46]        (notebook 04 Part B, preprocess)
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIT = "anand_vihar_delhi_dpcc_u1"
EVERY = 53                     # check every 53rd hour (~1400 hours, all seasons)
COLS = ["pm1", "pm25", "pm10", "t", "rh", "p", "wind_u", "wind_v", "rain_frac", "lux",
        "mq7_rs", "mq135_rs", "sraw_voc", "sraw_nox"]
FEATURES = ["ln_pm25", "ln_pm10", "ln_pm1", "coarse_ratio", "t", "rh", "p", "wind_u", "wind_v", "rain_frac",
            "ln_lux", "mq7_n", "mq135_n", "voc_n", "nox_n", "hod_sin", "hod_cos", "dow_sin", "dow_cos",
            "doy_sin", "doy_cos", "aqi_now", "pm_mask"]
NO_NORM = ["rain_frac", "hod_sin", "hod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "pm_mask"]

# small C++ program: read the hours, add them one by one, print the window of the hours to check
HARNESS = r"""
#include "features.h"
int main() {
  FILE* in = fopen("hours.txt", "r");
  FILE* out = fopen("out.txt", "w");
  long start; int measured, check;
  static float x[WINDOW][N_FEAT];
  float v[14], input[N_PATCH * PATCH_LEN], scale[N_OUT];
  while (fscanf(in, "%ld %d %d", &start, &measured, &check) == 3) {
    for (int k = 0; k < 14; k++) { fscanf(in, "%f", &v[k]); if (v[k] == -9999.0f) v[k] = NAN; }
    HourRec r = {(int32_t)start, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
                 v[12], v[13], (uint8_t)measured};
    addHour(r);
    if (!check) continue;
    const char* why = buildWindow(x);
    fprintf(out, "%ld %d", start, why[0] == 0);
    if (why[0] == 0) {
      preprocess(x, input, scale);
      for (int i = 0; i < WINDOW; i++) for (int c = 0; c < N_FEAT; c++) fprintf(out, " %.7g", x[i][c]);
      for (int i = 0; i < N_PATCH * PATCH_LEN; i++) fprintf(out, " %.7g", input[i]);
      for (int i = 0; i < N_OUT; i++) fprintf(out, " %.7g", scale[i]);
    }
    fprintf(out, "\n");
  }
  return 0;
}
"""


# notebook 04 (Part B) preprocess (Model A), for one window
def preprocess(x):
    norm = np.array([f not in NO_NORM for f in FEATURES])
    mean, std = x.mean(0, keepdims=True), np.sqrt(x.var(0, keepdims=True) + 1e-5)
    xn = np.where(norm, (x - mean) / std, x)
    s25, s10 = max(std[0, FEATURES.index("ln_pm25")], 0.1), max(std[0, FEATURES.index("ln_pm10")], 0.1)
    return xn.reshape(12, -1), np.array([s25] * 3 + [s10] * 3)


log = pd.read_parquet(ROOT / "data" / "3_sensors" / f"{UNIT}.parquet")
table = pd.read_parquet(ROOT / "data" / "4_features" / f"{UNIT}.parquet")
n = len(log)
check = np.zeros(n, bool)
check[200::EVERY] = True

# notebook 03's usable-hour rule (model_a_rows without "has targets")
x_all = table[FEATURES].to_numpy(np.float64)
complete_24h = pd.Series(np.isfinite(x_all).all(axis=1)).rolling(24).sum().to_numpy() == 24
measured_24h = table.pm_measured.astype(float).rolling(24).sum().to_numpy()
usable = complete_24h & (measured_24h >= 20) & table.pm_state.isin(["ok", "filled_le_1h"]).to_numpy()

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    start = (log.ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")
    values = log[COLS].to_numpy(np.float64)
    with open(tmp / "hours.txt", "w") as f:
        for i in range(n):
            row = " ".join("-9999" if np.isnan(v) else f"{v:.7g}" for v in values[i])
            f.write(f"{start.iloc[i]} {int(log.pm_measured.iloc[i])} {int(check[i])} {row}\n")
    (tmp / "harness.cpp").write_text(HARNESS)
    subprocess.run(["g++", "-O2", "-std=c++11", "-I", str(ROOT / "firmware" / "aqi_node"), "harness.cpp", "-o", "harness"],
                   cwd=tmp, check=True)
    subprocess.run([str(tmp / "harness")], cwd=tmp, check=True)
    lines = (tmp / "out.txt").read_text().splitlines()

rows = np.flatnonzero(check)
assert len(lines) == len(rows)
# aqi_now is a rounded whole AQI / 500: when the 24-h mean sits right on .5, float32 (ESP32) and float64 (pandas)
# can round to neighbouring integers -> allow 1 AQI point there, and compare the other 22 features strictly
AQI = FEATURES.index("aqi_now")
OTHERS = [c for c in range(23) if c != AQI]
worst_x, worst_in, worst_scale, worst_aqi, agree, aqi_off_by_one = 0.0, 0.0, 0.0, 0.0, 0, 0
for t, line in zip(rows, lines):
    parts = line.split()
    ok_c = parts[1] == "1"
    agree += ok_c == usable[t]
    if not (ok_c and usable[t]):
        continue
    nums = np.array(parts[2:], float)
    x_c, in_c, scale_c = nums[:24 * 23].reshape(24, 23), nums[24 * 23:-6].reshape(12, 46), nums[-6:]
    x_py = x_all[t - 23: t + 1]
    in_py, scale_py = preprocess(x_py)
    worst_x = max(worst_x, np.abs(x_c - x_py)[:, OTHERS].max())
    worst_in = max(worst_in, np.abs(in_c - in_py).reshape(12, 2, 23)[:, :, OTHERS].max())
    aqi_diff = np.abs(x_c[:, AQI] - x_py[:, AQI]).max() * 500                      # in AQI points
    worst_aqi, aqi_off_by_one = max(worst_aqi, aqi_diff), aqi_off_by_one + (aqi_diff > 0.01)
    worst_scale = max(worst_scale, np.abs(scale_c / scale_py - 1).max())

print(f"{len(rows)} hours checked, {usable[rows].sum()} usable in notebook 03")
print(f"usable-hour rule agrees on {agree}/{len(rows)} hours")
print(f"largest difference (22 features): features {worst_x:.1e}, model input {worst_in:.1e}, output scale {worst_scale:.1e} (relative)")
print(f"aqi_now: largest difference {worst_aqi:.2f} AQI points, windows with a 1-point rounding difference: {aqi_off_by_one}")
assert agree == len(rows), "the ESP32 and notebook 03 disagree on which hours can be forecast"
assert worst_x < 1e-3 and worst_in < 1e-2 and worst_scale < 1e-3 and worst_aqi < 1.01, "the ESP32 input differs from the training input"
print("OK: features.h gives the student the same input it was trained on")
