// hourly.cpp - one record per IST hour (hourly means of the 10-s readings, the same columns as simulator.py),
// the saved history (survives a power cut) and the forecast that runs when an hour closes.
#include <LittleFS.h>
#include "aqi_node.h"

static void closeHour();
static void saveHistory();

// sums of the current hour (only readings from healthy sensors are added)
struct HourSum {
  int32_t start = 0;                            // UTC epoch of the IST hour start (0 = no hour open)
  int ticks = 0, nPm = 0, nTh = 0, nP = 0, nLux = 0, nSgp = 0, nMq7 = 0, nMq135 = 0, nWind = 0, nRain = 0, wet = 0;
  double pm1 = 0, pm25 = 0, pm10 = 0, t = 0, rh = 0, p = 0, lux = 0, voc = 0, nox = 0, mq7 = 0, mq135 = 0, u = 0, v = 0;
};
static HourSum cur;

static const char* HISTORY_FILE = "/history.bin";
static const uint32_t HISTORY_MAGIC = 0xA0C00000 | sizeof(HourRec);   // changes if HourRec changes -> old file ignored

// the IST hour a moment belongs to (UTC epoch of its start)
int32_t hourStartOf(time_t t) { return (int32_t)(((int64_t)t + IST_OFFSET_S) / 3600 * 3600 - IST_OFFSET_S); }

// hourly mean, or NaN if the sensor gave nothing this hour
static float meanOr(double sum, int n) { return n > 0 ? sum / n : NAN; }

// add the latest 10-s reading to the current hour; a new IST hour closes the old one
void addSample() {
  if (!clockValid()) return;                    // no hours without a real clock
  int32_t start = hourStartOf(time(nullptr));
  if (cur.start != 0 && start != cur.start) closeHour();
  if (cur.start != start) { cur = HourSum(); cur.start = start; }
  cur.ticks++;
  if (health[S_PM] == H_OK) { cur.pm1 += live.pm1; cur.pm25 += live.pm25; cur.pm10 += live.pm10; cur.nPm++; }
  if (health[S_HDC] == H_OK) { cur.t += live.t; cur.rh += live.rh; cur.nTh++; }
  if (health[S_BMP] == H_OK) { cur.p += live.p; cur.nP++; }
  if (health[S_LUX] == H_OK) { cur.lux += live.lux; cur.nLux++; }
  if (health[S_SGP] == H_OK) { cur.voc += live.voc_raw; cur.nox += live.nox_raw; cur.nSgp++; }
  if (health[S_MQ7] == H_OK && mq7Readable()) { cur.mq7 += live.mq7_rs; cur.nMq7++; }
  if (health[S_MQ135] == H_OK) { cur.mq135 += live.mq135_rs; cur.nMq135++; }
  if (health[S_FANS] == H_OK) { cur.u += live.wind_u; cur.v += live.wind_v; cur.nWind++; }
  if (health[S_RAIN] == H_OK) { cur.wet += live.wet; cur.nRain++; }
}

// close the hour: record -> history -> forecast -> cloud
static void closeHour() {
  HourRec r;
  r.start = cur.start;
  r.pm1 = meanOr(cur.pm1, cur.nPm);
  r.pm25 = meanOr(cur.pm25, cur.nPm);
  r.pm10 = meanOr(cur.pm10, cur.nPm);
  r.t = meanOr(cur.t, cur.nTh);
  r.rh = meanOr(cur.rh, cur.nTh);
  r.p = meanOr(cur.p, cur.nP);
  r.wind_u = meanOr(cur.u, cur.nWind);
  r.wind_v = meanOr(cur.v, cur.nWind);
  r.rain_frac = meanOr(cur.wet, cur.nRain);
  r.lux = meanOr(cur.lux, cur.nLux);
  r.mq7_rs = meanOr(cur.mq7, cur.nMq7);
  r.mq135_rs = meanOr(cur.mq135, cur.nMq135);
  r.sraw_voc = meanOr(cur.voc, cur.nSgp);
  r.sraw_nox = meanOr(cur.nox, cur.nSgp);
  r.pm_measured = cur.nPm >= cur.ticks / 2 && cur.nPm > 0;      // PM worked at least half the hour
  addHour(r);                                   // features.h: empty rows are added for hours the node was off
  saveHistory();
  runForecast();
  sendHourToCloud(r);
  checkAlert();
  dataChanged = true;
}

// ---- history on the flash (LittleFS) -------------------------------------------------------------------------
static void saveHistory() {
  File f = LittleFS.open(HISTORY_FILE, "w");
  if (!f) return;
  f.write((const uint8_t*)&HISTORY_MAGIC, 4);
  f.write((const uint8_t*)&histCount, sizeof(histCount));
  f.write((const uint8_t*)hist, histCount * sizeof(HourRec));
  f.close();
}

void loadHistory() {
  if (!LittleFS.begin(true)) return;            // true = format the partition the first time
  File f = LittleFS.open(HISTORY_FILE, "r");
  if (!f) return;
  uint32_t magic = 0;
  int count = 0;
  f.read((uint8_t*)&magic, 4);
  f.read((uint8_t*)&count, sizeof(count));
  if (magic == HISTORY_MAGIC && count > 0 && count <= HISTORY &&
      f.read((uint8_t*)hist, count * sizeof(HourRec)) == count * sizeof(HourRec)) histCount = count;
  f.close();
}

// ---- the forecast (student on this board) --------------------------------------------------------------------
static void noForecast(const char* why) {
  edge.ok = false;
  strncpy(edge.why, why, sizeof(edge.why) - 1);
  edge.why[sizeof(edge.why) - 1] = 0;
}

void runForecast() {
  if (histCount > 0) processHistory();          // also needed for the AQI and gas baselines on screen
  if (!modelReady()) return noForecast(modelStatus);
  if (!clockValid()) return noForecast("waiting for the clock");
  if (histCount == 0 || hist[histCount - 1].start != hourStartOf(time(nullptr)) - 3600)
    return noForecast("waiting for the hour to end");
  static float x[WINDOW][N_FEAT];
  const char* why = buildWindow(x);             // features.h: the same rules as notebook 03
  if (why[0]) return noForecast(why);
  static float input[N_PATCH * PATCH_LEN];
  float scale[N_OUT], out[N_OUT];
  preprocess(x, input, scale);                  // features.h: the same as notebook 04
  uint32_t t0 = micros();
  if (!runModel(input, out)) return noForecast("model failed to run");
  edge.runMs = (micros() - t0) / 1000;
  makeForecast(out, scale, PM_CALIBRATION, edge.pm25, edge.pm10, edge.aqi);
  int t = histCount - 1;
  edge.base = hist[t].start;
  edge.now25 = d_c25[t];
  edge.now10 = d_c10[t];
  edge.ok = true;
  strcpy(edge.why, "");
}

// ---- numbers the screen needs -------------------------------------------------------------------------------------
// CPCB AQI (24 h) of the last closed hour, with the display calibration
float aqiNow() { return histCount > 0 ? aqiAt(histCount - 1, PM_CALIBRATION) : NAN; }
