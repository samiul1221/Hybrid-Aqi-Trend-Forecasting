// features.h - what the ESP32 does with its hourly records, in plain C++ (no Arduino code), so the laptop can
// check it: firmware/check_features.py compiles features.cpp and compares it with the notebooks.
// features.h = what other files can use; features.cpp = the code.
// It is a line-by-line copy of:
//   simulator.process_on_device   (notebook 02, section 3: humidity correction, gas baselines, AQI)
//   make_features + fill_short_gaps + model_a_rows   (notebook 03: the 23 features, which hours are usable)
//   preprocess                    (notebook 04 Part B: instance norm + 2-hour patches)
// If you change one of those, change this file too and run check_features.py.
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <algorithm>

// one closed IST hour: the same columns simulator.py logs (hourly means of the raw sensor readings)
struct HourRec {
  int32_t start;                 // UTC epoch of the IST hour start
  float pm1, pm25, pm10;         // PM sensor (ug/m3, raw)
  float t, rh, p;                // HDC1080 (C, %), BMP280 (hPa)
  float wind_u, wind_v;          // fan speeds along x and y (m/s, no sign)
  float rain_frac, lux;          // fraction of the hour the plate was wet, BH1750 (lux)
  float mq7_rs, mq135_rs;        // MQ resistances from the ADC (ohm)
  float sraw_voc, sraw_nox;      // SGP41 raw ticks
  uint8_t pm_measured;           // 1 if the PM sensor worked this hour
};

const int WINDOW = 24;                            // the student looks at the last 24 hours
const int FILL = 2;                               // gaps of <= 2 h get the last value (notebook 03)
const int BASE_HOURS = 168;                       // clean-air baselines: the previous 7 days
const int HISTORY = WINDOW + FILL + BASE_HOURS;   // 194 hours kept (enough for every number below)
const int N_FEAT = 23;
const int N_PATCH = 12, PATCH_LEN = 2 * N_FEAT;   // model input [12][46]
const int N_OUT = 6;                              // pm25 +1..+3 h, pm10 +1..+3 h
const int32_t IST_OFFSET_S = 19800;               // +5:30

// the 23 features, in the order the model was trained on (notebook 03 FEATURES)
enum { F_LN_PM25, F_LN_PM10, F_LN_PM1, F_COARSE, F_T, F_RH, F_P, F_WIND_U, F_WIND_V, F_RAIN, F_LN_LUX,
       F_MQ7, F_MQ135, F_VOC, F_NOX, F_HOD_SIN, F_HOD_COS, F_DOW_SIN, F_DOW_COS, F_DOY_SIN, F_DOY_COS,
       F_AQI, F_PM_MASK };
const char* const FEATURE_NAMES[N_FEAT] = {
  "ln_pm25", "ln_pm10", "ln_pm1", "coarse_ratio", "t", "rh", "p", "wind_u", "wind_v", "rain_frac",
  "ln_lux", "mq7_n", "mq135_n", "voc_n", "nox_n", "hod_sin", "hod_cos", "dow_sin", "dow_cos",
  "doy_sin", "doy_cos", "aqi_now", "pm_mask"};
// not instance-normalised (notebook 04 NO_NORM): rain, the time codes and the PM mask
const bool NO_NORM[N_FEAT] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1};

// the stored history, oldest first (defined in features.cpp)
extern HourRec hist[HISTORY];
extern int histCount;

// device processing of every stored hour (+3 spare rows for the forecast hours)
extern float d_pm1[HISTORY + 3], d_pm25[HISTORY + 3], d_pm10[HISTORY + 3];   // raw PM after the 1-hour hold
extern float d_c25[HISTORY + 3], d_c10[HISTORY + 3];                         // humidity-corrected PM
extern float d_rc7[HISTORY], d_rc135[HISTORY], d_voc[HISTORY], d_nox[HISTORY];
extern bool d_meas[HISTORY + 3];

// CPCB sub-index of a 24-h mean concentration (limits = top of each band)
const float AQI_LO[6] = {0, 51, 101, 201, 301, 401}, AQI_HI[6] = {50, 100, 200, 300, 400, 500};
const float PM25_LIMITS[6] = {30, 60, 90, 120, 250, 380};
const float PM10_LIMITS[6] = {50, 100, 250, 350, 430, 510};

// NaN-safe helpers (pandas keeps NaN where C's fmaxf would not)
inline float clipNan(float x, float lo, float hi) { return isnan(x) ? NAN : std::min(std::max(x, lo), hi); }
inline float maxNan(float x, float lo) { return isnan(x) ? NAN : std::max(x, lo); }
inline float finiteOrNan(float x) { return isfinite(x) ? x : NAN; }

// what features.cpp offers
HourRec emptyHour(int32_t start);
void addHour(const HourRec& r);
float humidityCorrect(float pm, float rh);
float subIndex(float c, const float* limits);
float aqiAt(int i, float cal);
float baselineAt(const float* x, int i, float q);
void processHistory();
void featuresAt(int i, float* f);
const char* buildWindow(float x[WINDOW][N_FEAT]);
void preprocess(const float x[WINDOW][N_FEAT], float* input, float* scale);
void makeForecast(const float* out, const float* scale, float cal, float* pm25, float* pm10, float* aqi);
