// features.cpp - the code behind features.h (see the comment there). Plain C++, no Arduino.
#include "features.h"

// the stored history, oldest first
HourRec hist[HISTORY];
int histCount = 0;

// device processing of every stored hour (+3 spare rows for the forecast hours)
float d_pm1[HISTORY + 3], d_pm25[HISTORY + 3], d_pm10[HISTORY + 3];   // raw PM after the 1-hour hold
float d_c25[HISTORY + 3], d_c10[HISTORY + 3];                         // humidity-corrected PM
float d_rc7[HISTORY], d_rc135[HISTORY], d_voc[HISTORY], d_nox[HISTORY];
bool d_meas[HISTORY + 3];

HourRec emptyHour(int32_t start) {
  HourRec r;
  r.start = start;
  r.pm1 = r.pm25 = r.pm10 = r.t = r.rh = r.p = r.wind_u = r.wind_v = r.rain_frac = r.lux = NAN;
  r.mq7_rs = r.mq135_rs = r.sraw_voc = r.sraw_nox = NAN;
  r.pm_measured = 0;
  return r;
}

// add a closed hour at the end (the oldest falls out). Hours the node was off become empty rows,
// so row i is always exactly one hour after row i-1.
void addHour(const HourRec& r) {
  if (histCount > 0) {
    int32_t last = hist[histCount - 1].start;
    if (r.start <= last) return;                          // clock went backwards: ignore
    long missing = std::min((long)(r.start - last) / 3600 - 1, (long)HISTORY);
    for (long k = 1; k <= missing; k++) {
      if (histCount == HISTORY) { memmove(hist, hist + 1, (HISTORY - 1) * sizeof(HourRec)); histCount--; }
      hist[histCount++] = emptyHour(last + (int32_t)(k * 3600));
    }
  }
  if (histCount == HISTORY) { memmove(hist, hist + 1, (HISTORY - 1) * sizeof(HourRec)); histCount--; }
  hist[histCount++] = r;
}

// humidity correction of the PM reading (kappa-Koehler, kappa = 0.4, a_w <= 0.95)
float humidityCorrect(float pm, float rh) {
  float aw = std::min(std::max(rh / 100.0f, 0.0f), 0.95f);
  float growth = 1.0f + (0.4f / 1.65f) / (1.0f / std::max(aw, 1e-9f) - 1.0f);
  return pm / growth;
}

float subIndex(float c, const float* limits) {
  float previous = 0;
  for (int k = 0; k < 6; k++) {
    if (c <= limits[k]) return AQI_LO[k] + (AQI_HI[k] - AQI_LO[k]) / (limits[k] - previous) * (c - previous);
    previous = limits[k];
  }
  return 500;
}

// CPCB AQI of the 24 hours ending at row i (NaN if fewer than 16 measured hours); cal scales the PM
float aqiAt(int i, float cal) {
  double s25 = 0, s10 = 0;
  int count = 0;
  for (int k = std::max(0, i - 23); k <= i; k++)
    if (d_meas[k] && isfinite(d_c25[k]) && isfinite(d_c10[k])) { s25 += d_c25[k]; s10 += d_c10[k]; count++; }
  if (count < 16) return NAN;
  float a = std::max(subIndex(cal * s25 / count, PM25_LIMITS), subIndex(cal * s10 / count, PM10_LIMITS));
  return rintf(a);                                // rint = Python's round (half to even)
}

// clean-air baseline of row i: q-quantile of the previous 168 hours (>= 24 values), like pandas
// x.shift(1).rolling(168, min_periods=24).quantile(q)
float baselineAt(const float* x, int i, float q) {
  static float buf[BASE_HOURS];
  int m = 0;
  for (int k = std::max(0, i - BASE_HOURS); k < i; k++)
    if (isfinite(x[k])) buf[m++] = x[k];
  if (m < 24) return NAN;
  std::sort(buf, buf + m);
  float pos = q * (m - 1);
  int lo = (int)pos;
  return lo + 1 < m ? buf[lo] + (pos - lo) * (buf[lo + 1] - buf[lo]) : buf[lo];
}

// the ESP32's hourly processing of the whole history (= simulator.process_on_device)
void processHistory() {
  for (int i = 0; i < histCount; i++) {
    const HourRec& h = hist[i];
    bool measured = h.pm_measured;
    bool hole = !measured && i > 0 && hist[i - 1].pm_measured;   // first silent hour: hold the last value
    const HourRec& pm = hole ? hist[i - 1] : h;
    d_meas[i] = measured;
    d_pm1[i] = (measured || hole) ? pm.pm1 : NAN;
    d_pm25[i] = (measured || hole) ? pm.pm25 : NAN;
    d_pm10[i] = (measured || hole) ? pm.pm10 : NAN;
    d_c25[i] = humidityCorrect(d_pm25[i], h.rh);
    d_c10[i] = humidityCorrect(d_pm10[i], h.rh);
    // MQ temperature / humidity compensation with the datasheets' typical slopes
    d_rc7[i] = h.mq7_rs * expf(0.007f * (h.t - 20) + 0.0028f * (h.rh - 65));
    d_rc135[i] = h.mq135_rs * expf(0.0105f * (h.t - 20) + 0.0055f * (h.rh - 55));
    d_voc[i] = h.sraw_voc;
    d_nox[i] = h.sraw_nox;
  }
}

// the 23 features of history row i (= make_features in notebook 03); NaN when an input is missing
void featuresAt(int i, float* f) {
  const HourRec& h = hist[i];
  f[F_LN_PM25] = log1pf(maxNan(d_c25[i], 0));
  f[F_LN_PM10] = log1pf(maxNan(d_c10[i], 0));
  f[F_LN_PM1] = log1pf(maxNan(d_pm1[i], 0));
  f[F_COARSE] = clipNan((d_pm10[i] - d_pm25[i]) / maxNan(d_pm1[i], 1), 0, 10);
  f[F_T] = h.t;
  f[F_RH] = h.rh;
  f[F_P] = h.p - 1000;
  f[F_WIND_U] = h.wind_u;
  f[F_WIND_V] = h.wind_v;
  f[F_RAIN] = h.rain_frac;
  f[F_LN_LUX] = log1pf(maxNan(h.lux, 0));
  f[F_MQ7] = finiteOrNan(logf(d_rc7[i] / baselineAt(d_rc7, i, 0.90f)));     // gas level vs clean air
  f[F_MQ135] = finiteOrNan(logf(d_rc135[i] / baselineAt(d_rc135, i, 0.90f)));
  f[F_VOC] = (d_voc[i] - baselineAt(d_voc, i, 0.90f)) / 1000;
  f[F_NOX] = (d_nox[i] - baselineAt(d_nox, i, 0.10f)) / 1000;
  time_t ist = (time_t)h.start + IST_OFFSET_S;                            // IST clock of the hour start
  struct tm* tm = gmtime(&ist);
  int dow = (tm->tm_wday + 6) % 7, doy = tm->tm_yday + 1;                  // pandas: Monday = 0, 1st Jan = 1
  const float TWO_PI_F = 6.2831853f;
  f[F_HOD_SIN] = sinf(TWO_PI_F * tm->tm_hour / 24);
  f[F_HOD_COS] = cosf(TWO_PI_F * tm->tm_hour / 24);
  f[F_DOW_SIN] = sinf(TWO_PI_F * dow / 7);
  f[F_DOW_COS] = cosf(TWO_PI_F * dow / 7);
  f[F_DOY_SIN] = sinf(TWO_PI_F * doy / 365);
  f[F_DOY_COS] = cosf(TWO_PI_F * doy / 365);
  f[F_AQI] = aqiAt(i, 1.0f) / 500;
  f[F_PM_MASK] = d_meas[i] ? 1.0f : 0.0f;
}

// the student's 24-hour input for the newest hour. Returns "" if the hour is usable (the same rules as
// model_a_rows in notebook 03), otherwise a short reason for the screen.
static char windowWhy[48];
const char* buildWindow(float x[WINDOW][N_FEAT]) {
  if (histCount < WINDOW + FILL) return "collecting history";
  processHistory();
  float rows[WINDOW + FILL][N_FEAT];
  int t = histCount - 1, first = t - (WINDOW + FILL) + 1;
  for (int r = 0; r < WINDOW + FILL; r++) featuresAt(first + r, rows[r]);
  // gaps of <= 2 h get the last value (pandas ffill(limit=2)): past only, the same as in training
  for (int c = 0; c < N_FEAT; c++) {
    float last = NAN;
    int run = 0;
    for (int r = 0; r < WINDOW + FILL; r++) {
      if (!isnan(rows[r][c])) { last = rows[r][c]; run = 0; }
      else if (++run <= FILL) rows[r][c] = last;
    }
  }
  for (int r = 0; r < WINDOW; r++)
    for (int c = 0; c < N_FEAT; c++) {
      x[r][c] = rows[r + FILL][c];
      if (isnan(x[r][c])) {
        snprintf(windowWhy, sizeof(windowWhy), "%s missing %dh ago", FEATURE_NAMES[c], WINDOW - 1 - r);
        return windowWhy;
      }
    }
  int measured = 0;
  for (int k = t - WINDOW + 1; k <= t; k++) measured += d_meas[k];
  if (measured < 20) return "PM measured < 20 of 24 h";
  if (!d_meas[t] && !d_meas[t - 1]) return "PM sensor silent 2+ h";
  return "";
}

// notebook 04 (Part B) preprocessing: instance norm per feature over the 24 h (not for NO_NORM), then 2-hour
// patches -> input [12][46]. scale = how the model's outputs are multiplied back.
void preprocess(const float x[WINDOW][N_FEAT], float* input, float* scale) {
  float mean[N_FEAT], sd[N_FEAT];
  for (int c = 0; c < N_FEAT; c++) {
    double s = 0, v = 0;
    for (int r = 0; r < WINDOW; r++) s += x[r][c];
    mean[c] = s / WINDOW;
    for (int r = 0; r < WINDOW; r++) v += (x[r][c] - mean[c]) * (x[r][c] - mean[c]);
    sd[c] = sqrtf(v / WINDOW + 1e-5f);
  }
  for (int r = 0; r < WINDOW; r++)                // patch k = hours 2k and 2k+1 side by side
    for (int c = 0; c < N_FEAT; c++)
      input[r * N_FEAT + c] = NO_NORM[c] ? x[r][c] : (x[r][c] - mean[c]) / sd[c];
  float s25 = std::max(sd[F_LN_PM25], 0.1f), s10 = std::max(sd[F_LN_PM10], 0.1f);
  for (int h = 0; h < 3; h++) { scale[h] = s25; scale[3 + h] = s10; }
}

// model output (change of ln(1+PM) / scale) -> PM forecast for +1..+3 h (sensor scale, ug/m3) and the CPCB
// AQI those hours would give (24-h mean with the forecast hours added). cal = display calibration.
void makeForecast(const float* out, const float* scale, float cal, float* pm25, float* pm10, float* aqi) {
  int t = histCount - 1;
  for (int h = 0; h < 3; h++) {
    pm25[h] = (1 + d_c25[t]) * expf(out[h] * scale[h]) - 1;
    pm10[h] = (1 + d_c10[t]) * expf(out[3 + h] * scale[3 + h]) - 1;
    d_c25[t + 1 + h] = pm25[h];
    d_c10[t + 1 + h] = pm10[h];
    d_meas[t + 1 + h] = true;
  }
  for (int h = 0; h < 3; h++) aqi[h] = aqiAt(t + 1 + h, cal);
}
