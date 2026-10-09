
// ui.cpp - the dashboard. Every size is a fraction of the screen, so it fits any panel size and rotation.
//
// Pages (tap the button for the next one):
//   NOW          CPCB AQI, PM now, next 3 hours        PM2.5 / PM10  chart: measured, edge and cloud forecast
//   FORECAST     table: values, change vs now, edge - cloud
//   WEATHER      HDC1080, BMP280, fans, rain, BH1750   GASES   PM raw, MQ-7, MQ-135, SGP41
//   HEALTH       every sensor's state + system (Wi-Fi, clock, history, model, cloud)
// Hold the button: alert mode (tap = next alert level, hold = done).
//
// Two rules keep the screen clean:
//   1. every text goes through fitText(): the biggest font that fits its box, else cut with ".." -> no overflow;
//   2. a text is only redrawn when it changed (and its box is cleared first) -> no flicker, no overlap.
#include <WiFi.h>
#include "aqi_node.h"

// ======================= colours ===========================================================================
constexpr uint16_t rgb(uint8_t r, uint8_t g, uint8_t b) { return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3); }
const uint16_t C_BG = rgb(8, 12, 22), C_CARD = rgb(20, 28, 44), C_LINE = rgb(48, 60, 84), C_GRID = rgb(34, 44, 64);
const uint16_t C_TEXT = rgb(236, 240, 248), C_DIM = rgb(128, 142, 168);
const uint16_t C_MEAS = rgb(236, 240, 248), C_EDGE = rgb(0, 200, 255), C_CLOUD = rgb(255, 150, 40);
const uint16_t C_OK = rgb(40, 200, 110), C_WARN = rgb(255, 190, 40), C_BAD = rgb(240, 70, 70);

// CPCB AQI categories
const char* const AQI_NAMES[6] = {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"};
const uint16_t AQI_COLORS[6] = {rgb(0, 176, 80), rgb(146, 208, 80), rgb(255, 215, 0),
                                rgb(255, 140, 0), rgb(230, 30, 30), rgb(150, 0, 40)};
const int ALERT_LEVELS[4] = {101, 201, 301, 401};   // start of Moderate, Poor, Very Poor, Severe

static int aqiBand(float aqi) {
  for (int k = 0; k < 6; k++)
    if (aqi <= AQI_HI[k]) return k;
  return 5;
}

// dark text on light colours (yellow, light green), white text on dark ones
static uint16_t textOn(uint16_t bg) {
  int r = (bg >> 11) << 3, g = ((bg >> 5) & 0x3F) << 2, b = (bg & 0x1F) << 3;
  return (r * 299 + g * 587 + b * 114) / 1000 > 150 ? C_BG : C_TEXT;
}

static uint16_t healthColor(Health h) { return h == H_OK ? C_OK : h == H_WARMING ? C_WARN : h == H_ABSENT ? C_DIM : C_BAD; }
static const char* healthText(Health h) {
  const char* const names[] = {"OK", "WARMING", "NO DATA", "OUT OF RANGE", "NOT FOUND"};
  return names[h];
}

// ======================= layout: boxes as fractions ========================================================
struct Box { int x, y, w, h; };

static int W, H, GAP, RADIUS;
static Box HEADER, BODY, FOOTER;

// part of a box: fx, fy, fw, fh are fractions (0..1) of the box
static Box sub(const Box& b, float fx, float fy, float fw, float fh) {
  return {b.x + (int)(fx * b.w), b.y + (int)(fy * b.h), (int)(fw * b.w), (int)(fh * b.h)};
}

// cell (col, row) of a grid of cols x rows inside a box, with GAP between cells
static Box cell(const Box& b, int col, int row, int cols, int rows) {
  int w = (b.w - (cols - 1) * GAP) / cols, h = (b.h - (rows - 1) * GAP) / rows;
  return {b.x + col * (w + GAP), b.y + row * (h + GAP), w, h};
}

// ======================= text that always fits =============================================================
const uint8_t FONTS[] = {6, 4, 2, 1};            // 48, 26, 16, 8 px high (font 6 has digits only)

static bool digitsOnly(const String& s) {
  if (!s.length()) return false;
  for (char c : s)
    if (!isdigit(c) && c != '.' && c != '-' && c != ':') return false;
  return true;
}

// what is on screen at each box, so unchanged text is not redrawn
struct Drawn { int x, y, w; uint32_t key; };
static Drawn drawn[96];
static int nDrawn = 0;

static uint32_t textKey(const String& s, uint16_t fg, uint16_t bg, uint8_t datum) {
  uint32_t k = 2166136261u ^ fg ^ (bg << 16) ^ (datum << 8);
  for (char c : s) k = (k ^ (uint8_t)c) * 16777619u;   // FNV-1a hash
  return k;
}

static bool unchanged(const Box& b, uint32_t key) {
  for (int i = 0; i < nDrawn; i++)
    if (drawn[i].x == b.x && drawn[i].y == b.y && drawn[i].w == b.w) {
      if (drawn[i].key == key) return true;
      drawn[i].key = key;
      return false;
    }
  if (nDrawn < 96) drawn[nDrawn++] = {b.x, b.y, b.w, key};
  return false;
}

// write `s` inside box b: biggest font up to maxFont that fits, else font 1 cut with ".."
static void fitText(const Box& b, String s, uint8_t maxFont, uint16_t fg, uint16_t bg, uint8_t datum = MC_DATUM) {
  if (b.w <= 0 || b.h <= 0 || unchanged(b, textKey(s, fg, bg, datum))) return;
  uint8_t font = 1;
  for (uint8_t f : FONTS) {
    if (tft.fontHeight(f) > tft.fontHeight(maxFont) || (f == 6 && !digitsOnly(s))) continue;
    if (tft.textWidth(s, f) <= b.w && tft.fontHeight(f) <= b.h) { font = f; break; }
  }
  while (s.length() > 2 && tft.textWidth(s, font) > b.w) s = s.substring(0, s.length() - 3) + "..";
  tft.fillRect(b.x, b.y, b.w, b.h, bg);
  tft.setTextColor(fg, bg);
  tft.setTextDatum(datum);
  int x = datum == ML_DATUM ? b.x : datum == MR_DATUM ? b.x + b.w - 1 : b.x + b.w / 2;
  tft.drawString(s, x, b.y + b.h / 2, font);
}

// ======================= small building blocks =============================================================
static String num(float v, int decimals = 0) { return isnan(v) ? String("--") : String(v, decimals); }

// signed number: "+12" / "-3"
static String signedNum(float v, int decimals = 0) {
  if (isnan(v)) return "--";
  return (v >= 0 ? "+" : "") + String(v, decimals);
}

// IST clock "HH:MM" of an epoch, or of now
static String clockText(time_t t) {
  if (t < 1704067200) return "--:--";
  t += IST_OFFSET_S;
  struct tm tm;
  gmtime_r(&t, &tm);
  char s[8];
  snprintf(s, sizeof(s), "%02d:%02d", tm.tm_hour, tm.tm_min);
  return s;
}

// "14-15h": the IST hour slot that starts `hoursAfter` hours after the hour `base` starts
static String slotText(int32_t base, int hoursAfter) {
  time_t t = (time_t)base + hoursAfter * 3600 + IST_OFFSET_S;
  struct tm tm;
  gmtime_r(&t, &tm);
  return String(tm.tm_hour) + "-" + String((tm.tm_hour + 1) % 24) + "h";
}

// humidity-corrected live PM on the display scale
static float shownLive(float pm) { return humidityCorrect(pm, isnan(live.rh) ? 50 : live.rh) * PM_CALIBRATION; }

// the cloud forecast belongs to the same hour as the edge one?
static bool cloudMatches() { return cloud.ok && histCount > 0 && cloud.base == hist[histCount - 1].start; }

// a card: rounded panel + small title (+ health dot). Returns the area under the title.
static Box card(const Box& b, const String& title, bool full, uint16_t accent = C_LINE, int dot = -1) {
  int titleH = constrain(b.h / 5, 10, 20);
  if (full) {
    tft.fillRoundRect(b.x, b.y, b.w, b.h, RADIUS, C_CARD);
    tft.drawRoundRect(b.x, b.y, b.w, b.h, RADIUS, C_LINE);
    tft.fillRect(b.x + RADIUS, b.y + 1, b.w - 2 * RADIUS, 2, accent);       // coloured line on top
  }
  int dotR = max(3, titleH / 4);
  fitText({b.x + GAP, b.y + 3, b.w - 2 * GAP - (dot >= 0 ? 3 * dotR : 0), titleH}, title, 2, C_DIM, C_CARD, ML_DATUM);
  if (dot >= 0) tft.fillCircle(b.x + b.w - GAP - dotR, b.y + 3 + titleH / 2, dotR, healthColor((Health)dot));
  return {b.x + GAP, b.y + 3 + titleH, b.w - 2 * GAP, b.h - titleH - 3 - GAP / 2};
}

// a value tile: title, big value, small line under it
static void tile(const Box& b, const String& title, const String& value, const String& small, Health h, bool full,
                 uint16_t valueColor = C_TEXT) {
  Box in = card(b, title, full, C_LINE, h);
  fitText(sub(in, 0, 0, 1, 0.68), value, 6, valueColor, C_CARD);
  fitText(sub(in, 0, 0.68, 1, 0.32), small, 2, C_DIM, C_CARD);
}

// ======================= header and footer =================================================================
enum Page { P_NOW, P_CHART25, P_CHART10, P_TABLE, P_WEATHER, P_GASES, P_HEALTH, N_PAGES };
const char* const PAGE_TITLES[N_PAGES] = {"Air now", "PM2.5 forecast", "PM10 forecast", "Forecast",
                                          "Weather", "Gas sensors", "Health"};
static int page = P_NOW;
static bool alertMode = false;
static uint32_t alertModeMs = 0;

static void drawHeader(bool full) {
  if (full) tft.drawFastHLine(0, HEADER.h - 1, W, C_LINE);
  fitText(sub(HEADER, 0.02, 0.1, 0.46, 0.8), PAGE_TITLES[page], 4, C_TEXT, C_BG, ML_DATUM);
  // status: Wi-Fi, cloud, model (green = working)
  Box s = sub(HEADER, 0.49, 0.2, 0.33, 0.6);
  fitText(cell(s, 0, 0, 3, 1), "WiFi", 2, WiFi.status() == WL_CONNECTED ? C_OK : C_DIM, C_BG);
  fitText(cell(s, 1, 0, 3, 1), "Cloud", 2, cloudMatches() ? C_OK : C_DIM, C_BG);
  fitText(cell(s, 2, 0, 3, 1), "Model", 2, edge.ok ? C_OK : modelReady() ? C_WARN : C_BAD, C_BG);
  fitText(sub(HEADER, 0.83, 0.1, 0.15, 0.8), clockText(time(nullptr)), 4, C_TEXT, C_BG, MR_DATUM);
}

static void drawFooter(bool full) {
  if (full) tft.drawFastHLine(0, FOOTER.y, W, C_LINE);
  String hint = alertMode ? "ALERT at AQI " + String(alertAqi) + ":  tap = next level,  hold = done"
                          : "tap: next page   hold: alert (AQI " + String(alertAqi) + ")";
  fitText(sub(FOOTER, 0.02, 0.15, 0.68, 0.7), hint, 2, alertMode ? C_WARN : C_DIM, C_BG, ML_DATUM);
  if (full) {                                   // page dots
    int r = max(2, FOOTER.h / 7), x0 = W - GAP - (2 * N_PAGES - 1) * 2 * r, cy = FOOTER.y + FOOTER.h / 2;
    for (int i = 0; i < N_PAGES; i++) {
      if (i == page) tft.fillCircle(x0 + i * 4 * r, cy, r, C_EDGE);
      else tft.drawCircle(x0 + i * 4 * r, cy, r, C_DIM);
    }
  }
}

// ======================= page: NOW =========================================================================
static void pageNow(bool full) {
  bool wide = W >= H;
  Box left = wide ? sub(BODY, 0, 0, 0.48, 1) : sub(BODY, 0, 0, 1, 0.45);
  Box right = wide ? sub(BODY, 0.51, 0, 0.49, 1) : sub(BODY, 0, 0.47, 1, 0.53);

  // AQI: CPCB 24-h value; before 16 h of PM data, an estimate from the PM right now
  float aqi = aqiNow();
  bool estimate = isnan(aqi);
  if (estimate && !isnan(live.pm25))
    aqi = roundf(max(subIndex(shownLive(live.pm25), PM25_LIMITS), subIndex(shownLive(live.pm10), PM10_LIMITS)));
  int band = aqiBand(isnan(aqi) ? 0 : aqi);
  Box in = card(left, estimate ? "AQI (estimate from PM now)" : "AQI  (CPCB, last 24 h)", full, AQI_COLORS[band]);
  fitText(sub(in, 0, 0, 1, 0.52), num(aqi), 6, isnan(aqi) ? C_DIM : AQI_COLORS[band], C_CARD);
  fitText(sub(in, 0, 0.52, 1, 0.22), isnan(aqi) ? "waiting for PM" : AQI_NAMES[band], 4, C_TEXT, C_CARD);
  // colour scale 0-500 with a marker
  static float drawnAqi = -1;
  Box bar = sub(in, 0.04, 0.8, 0.92, 0.1);
  if (full || aqi != drawnAqi) {
    drawnAqi = aqi;
    tft.fillRect(bar.x - 4, bar.y - 6, bar.w + 8, bar.h + 7, C_CARD);
    for (int k = 0; k < 6; k++) {
      int x0 = bar.x + bar.w * AQI_LO[k] / 500, x1 = bar.x + bar.w * AQI_HI[k] / 500;
      tft.fillRect(x0, bar.y, x1 - x0, bar.h, AQI_COLORS[k]);
    }
    if (!isnan(aqi)) {
      int mx = bar.x + bar.w * constrain(aqi, 0, 500) / 500;
      tft.fillTriangle(mx - 4, bar.y - 6, mx + 4, bar.y - 6, mx, bar.y - 1, C_TEXT);
    }
  }

  // PM right now
  Box pm25 = cell(sub(right, 0, 0, 1, 0.64), 0, 0, 2, 1), pm10 = cell(sub(right, 0, 0, 1, 0.64), 1, 0, 2, 1);
  tile(pm25, "PM2.5 now", num(shownLive(live.pm25)), "ug/m3", health[S_PM], full);
  tile(pm10, "PM10 now", num(shownLive(live.pm10)), "ug/m3", health[S_PM], full);

  // next 3 hours (edge model): coloured AQI chips
  Box next = card(sub(right, 0, 0.67, 1, 0.33), "Next 3 hours (AQI, this node)", full, C_EDGE);
  for (int h = 0; h < 3; h++) {
    Box chip = cell(next, h, 0, 3, 1);
    if (!edge.ok) {
      if (h == 0) fitText(next, edge.why, 2, C_DIM, C_CARD);
      continue;
    }
    int b = aqiBand(edge.aqi[h]);
    if (full) tft.fillRoundRect(chip.x, chip.y, chip.w, chip.h, RADIUS, AQI_COLORS[b]);
    fitText(sub(chip, 0.05, 0.05, 0.9, 0.35), "+" + String(h + 1) + "h", 2, textOn(AQI_COLORS[b]), AQI_COLORS[b]);
    fitText(sub(chip, 0.05, 0.4, 0.9, 0.55), num(edge.aqi[h]), 4, textOn(AQI_COLORS[b]), AQI_COLORS[b]);
  }
}

// ======================= page: chart (PM2.5 or PM10) =======================================================
// legend symbols
static void legendItem(const Box& b, int kind, const String& text) {
  int s = max(3, b.h / 3), cx = b.x + s + 2, cy = b.y + b.h / 2;
  if (kind == 0) { tft.fillRect(cx - s, cy - 1, 2 * s, 3, C_MEAS); }
  if (kind == 1) { tft.drawLine(cx - s, cy, cx + s, cy, C_EDGE); tft.fillCircle(cx, cy, s / 2 + 1, C_EDGE); }
  if (kind == 2) { tft.drawLine(cx - s, cy, cx + s, cy, C_CLOUD); tft.fillRect(cx - s / 2 - 1, cy - s / 2 - 1, s + 2, s + 2, C_CLOUD); }
  fitText({cx + s + 3, b.y, b.w - (cx + s + 3 - b.x), b.h}, text, 2, C_TEXT, C_CARD, ML_DATUM);
}

// dashed line (for "now" and the Poor level)
static void dashedLine(int x0, int y0, int x1, int y1, uint16_t c) {
  int n = max(abs(x1 - x0), abs(y1 - y0)) / 4;
  for (int i = 0; i < n; i += 2)
    tft.drawLine(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, x0 + (x1 - x0) * (i + 1) / n, y0 + (y1 - y0) * (i + 1) / n, c);
}

// value label of a point, kept inside the plot; above = true puts it over the point
static void pointLabel(const Box& plot, int x, int y, int r, float v, uint16_t c, bool above, uint8_t font) {
  String s = num(v);
  int w = tft.textWidth(s, font) + 2, h = tft.fontHeight(font);
  int ly = above ? y - r - 2 - h : y + r + 2;
  ly = constrain(ly, plot.y, plot.y + plot.h - h);
  int lx = constrain(x - w / 2, plot.x, plot.x + plot.w - w);
  tft.fillRect(lx, ly, w, h, C_CARD);
  tft.setTextColor(c, C_CARD);
  tft.setTextDatum(TL_DATUM);
  tft.drawString(s, lx + 1, ly, font);
}

static void drawChart(const Box& b, int pollutant) {
  const float* hourly = pollutant == 0 ? d_c25 : d_c10;
  const float* edgeV = pollutant == 0 ? edge.pm25 : edge.pm10;
  const float* cloudV = pollutant == 0 ? cloud.pm25 : cloud.pm10;
  float poor = pollutant == 0 ? PM25_LIMITS[2] : PM10_LIMITS[2];     // "Poor" starts above this

  // 9 slots: 6 measured hours (-5 h ... now) and +1, +2, +3 h
  float meas[9], fe[9], fc[9];
  for (int i = 0; i < 9; i++) meas[i] = fe[i] = fc[i] = NAN;
  int t = histCount - 1;
  for (int i = 0; i < 6; i++) {
    int row = t - 5 + i;
    if (row >= 0 && d_meas[row]) meas[i] = hourly[row] * PM_CALIBRATION;
  }
  for (int h = 0; h < 3; h++) {
    if (edge.ok) fe[6 + h] = edgeV[h] * PM_CALIBRATION;
    if (cloudMatches()) fc[6 + h] = cloudV[h] * PM_CALIBRATION;
  }

  // legend
  int legendH = constrain(b.h / 8, 10, 18);
  Box legend = {b.x, b.y, b.w, legendH};
  legendItem(cell(legend, 0, 0, 3, 1), 0, "measured");
  legendItem(cell(legend, 1, 0, 3, 1), 1, "edge (this node)");
  legendItem(cell(legend, 2, 0, 3, 1), 2, cloudMatches() ? String("cloud ") + cloud.model : String("cloud (none)"));

  // y range from every value (+ the Poor line if it is near), with room for labels
  float lo = 1e9, hi = -1e9;
  for (int i = 0; i < 9; i++)
    for (float v : {meas[i], fe[i], fc[i]})
      if (!isnan(v)) { lo = min(lo, v); hi = max(hi, v); }
  if (hi < lo) {
    fitText({b.x, b.y + legendH, b.w, b.h - legendH}, "collecting hourly data...", 4, C_DIM, C_CARD);
    return;
  }
  if (poor < hi * 1.4f && poor > lo * 0.6f) { lo = min(lo, poor); hi = max(hi, poor); }
  float span = max(hi - lo, 10.0f);
  lo = max(0.0f, lo - 0.12f * span);
  hi = hi + 0.22f * span;

  uint8_t labelFont = b.h > 140 ? 2 : 1;
  int yLabelW = tft.textWidth(String((int)hi), 1) + 4, xLabelH = tft.fontHeight(1) + 3;
  Box plot = {b.x + yLabelW, b.y + legendH + 4, b.w - yLabelW - 2, b.h - legendH - 4 - xLabelH};
  auto X = [&](int i) { return plot.x + i * (plot.w - 1) / 8; };
  auto Y = [&](float v) { return plot.y + plot.h - 1 - (int)((v - lo) / (hi - lo) * (plot.h - 1)); };

  // grid + y labels (3 levels)
  tft.setTextDatum(MR_DATUM);
  for (int k = 0; k < 3; k++) {
    float v = lo + (hi - lo) * k / 2;
    tft.drawFastHLine(plot.x, Y(v), plot.w, C_GRID);
    tft.setTextColor(C_DIM, C_CARD);
    tft.drawString(String((int)roundf(v)), plot.x - 3, Y(v), 1);
  }
  // "now" line and the Poor level
  dashedLine(X(5), plot.y, X(5), plot.y + plot.h - 1, C_DIM);
  if (poor > lo && poor < hi) {
    dashedLine(plot.x, Y(poor), plot.x + plot.w - 1, Y(poor), C_WARN);
    tft.setTextDatum(BR_DATUM);
    tft.setTextColor(C_WARN, C_CARD);
    tft.drawString("Poor", plot.x + plot.w - 2, Y(poor) - 1, 1);
  }
  // x labels: -5h ... now ... +3h (every second past hour on narrow screens)
  int step = tft.textWidth("-5h", 1) + 4 > plot.w / 8 ? 2 : 1;
  tft.setTextDatum(TC_DATUM);
  tft.setTextColor(C_DIM, C_CARD);
  for (int i = 0; i < 9; i++) {
    if (i < 5 && (5 - i) % step) continue;
    String s = i < 5 ? "-" + String(5 - i) + "h" : i == 5 ? String("now") : "+" + String(i - 5) + "h";
    tft.drawString(s, X(i), plot.y + plot.h + 2, 1);
  }

  // measured: thick white line + dots
  for (int i = 1; i < 6; i++)
    if (!isnan(meas[i - 1]) && !isnan(meas[i])) {
      tft.drawLine(X(i - 1), Y(meas[i - 1]), X(i), Y(meas[i]), C_MEAS);
      tft.drawLine(X(i - 1), Y(meas[i - 1]) + 1, X(i), Y(meas[i]) + 1, C_MEAS);
    }
  for (int i = 0; i < 6; i++)
    if (!isnan(meas[i])) tft.fillCircle(X(i), Y(meas[i]), 2, C_MEAS);

  // forecasts start at the last measured value
  int r = max(3, plot.h / 40);
  float start = meas[5];
  for (int i = 6; i < 9; i++) {
    float pe = i == 6 ? start : fe[i - 1], pc = i == 6 ? start : fc[i - 1];
    if (!isnan(pe) && !isnan(fe[i])) tft.drawLine(X(i - 1), Y(pe), X(i), Y(fe[i]), C_EDGE);
    if (!isnan(pc) && !isnan(fc[i])) dashedLine(X(i - 1), Y(pc), X(i), Y(fc[i]), C_CLOUD);
  }
  for (int i = 6; i < 9; i++) {
    if (!isnan(fe[i])) tft.fillCircle(X(i), Y(fe[i]), r, C_EDGE);
    if (!isnan(fc[i])) tft.fillRect(X(i) - r, Y(fc[i]) - r, 2 * r + 1, 2 * r + 1, C_CLOUD);
  }
  // labels: the higher point gets its label above, the lower one below -> they never overlap
  if (!isnan(start)) pointLabel(plot, X(5), Y(start), 2, start, C_MEAS, true, labelFont);
  for (int i = 6; i < 9; i++) {
    bool edgeHigher = isnan(fc[i]) || (!isnan(fe[i]) && Y(fe[i]) <= Y(fc[i]));
    if (!isnan(fe[i])) pointLabel(plot, X(i), Y(fe[i]), r, fe[i], C_EDGE, edgeHigher, labelFont);
    if (!isnan(fc[i])) pointLabel(plot, X(i), Y(fc[i]), r, fc[i], C_CLOUD, !edgeHigher, labelFont);
  }
}

static void pageChart(bool full, int pollutant) {
  if (!full) return;                           // charts change once an hour: drawn only on a full redraw
  Box chart = card(sub(BODY, 0, 0, 1, 0.76), pollutant == 0 ? "PM2.5  (ug/m3, hourly mean)" : "PM10  (ug/m3, hourly mean)",
                   full, C_EDGE);
  drawChart(chart, pollutant);
  // deviation of the edge forecast from the cloud forecast
  Box dev = card(sub(BODY, 0, 0.79, 1, 0.21), "edge - cloud  (ug/m3)", full, C_CLOUD);
  const float* e = pollutant == 0 ? edge.pm25 : edge.pm10;
  const float* c = pollutant == 0 ? cloud.pm25 : cloud.pm10;
  for (int h = 0; h < 3; h++) {
    String text = "+" + String(h + 1) + "h  ";
    uint16_t color = C_DIM;
    if (edge.ok && cloudMatches()) {
      float d = (e[h] - c[h]) * PM_CALIBRATION, pct = 100 * (e[h] / c[h] - 1);
      text += signedNum(d) + " (" + signedNum(pct) + "%)";
      color = fabsf(pct) < 15 ? C_OK : fabsf(pct) < 35 ? C_WARN : C_BAD;   // close / some / far apart
    } else {
      text += "--";
    }
    fitText(cell(dev, h, 0, 3, 1), text, 2, color, C_CARD);
  }
}

// ======================= page: forecast table ===============================================================
static void pageTable(bool full) {
  if (!full) return;
  Box in = card(BODY, edge.ok ? "Forecast from hour " + slotText(edge.base, 0) + "  (edge model on this node)"
                              : "Forecast  (edge model on this node)", full, C_EDGE);
  if (!edge.ok) {
    fitText(in, edge.why, 4, C_DIM, C_CARD);
    return;
  }
  const char* heads[] = {"hour", "PM2.5", "change", "PM10", "change", "AQI", "vs cloud"};
  const float widths[] = {0.13f, 0.13f, 0.14f, 0.13f, 0.14f, 0.13f, 0.20f};
  int rows = 5;
  auto at = [&](int col, int row) {
    float x = 0;
    for (int k = 0; k < col; k++) x += widths[k];
    return sub(in, x, row / (float)rows, widths[col], 1.0f / rows);
  };
  for (int c = 0; c < 7; c++) fitText(at(c, 0), heads[c], 2, C_DIM, C_CARD);
  tft.drawFastHLine(in.x, in.y + in.h / rows - 1, in.w, C_LINE);

  float now25 = edge.now25 * PM_CALIBRATION, now10 = edge.now10 * PM_CALIBRATION;
  fitText(at(0, 1), "now", 2, C_TEXT, C_CARD);
  fitText(at(1, 1), num(now25), 4, C_TEXT, C_CARD);
  fitText(at(3, 1), num(now10), 4, C_TEXT, C_CARD);
  float aqi = aqiNow();
  fitText(at(5, 1), num(aqi), 4, isnan(aqi) ? C_DIM : AQI_COLORS[aqiBand(aqi)], C_CARD);
  for (int h = 0; h < 3; h++) {
    int row = 2 + h;
    float p25 = edge.pm25[h] * PM_CALIBRATION, p10 = edge.pm10[h] * PM_CALIBRATION;
    float c25 = 100 * (p25 / now25 - 1), c10 = 100 * (p10 / now10 - 1);
    fitText(at(0, row), "+" + String(h + 1) + "h " + slotText(edge.base, h + 1), 2, C_TEXT, C_CARD);
    fitText(at(1, row), num(p25), 4, C_EDGE, C_CARD);
    fitText(at(2, row), signedNum(c25) + "%", 2, c25 > 0 ? C_BAD : C_OK, C_CARD);
    fitText(at(3, row), num(p10), 4, C_EDGE, C_CARD);
    fitText(at(4, row), signedNum(c10) + "%", 2, c10 > 0 ? C_BAD : C_OK, C_CARD);
    fitText(at(5, row), num(edge.aqi[h]), 4, AQI_COLORS[aqiBand(edge.aqi[h])], C_CARD);
    String vs = cloudMatches() ? signedNum((edge.pm25[h] - cloud.pm25[h]) * PM_CALIBRATION) + " / " +
                                 signedNum((edge.pm10[h] - cloud.pm10[h]) * PM_CALIBRATION)
                               : String("--");
    fitText(at(6, row), vs, 2, C_CLOUD, C_CARD);
  }
}

// ======================= pages: weather and gases ===========================================================
static int tileCols() { return W >= H ? 3 : 2; }

static void pageWeather(bool full) {
  int cols = tileCols(), rows = 6 / cols;
  auto at = [&](int i) { return cell(BODY, i % cols, i / cols, cols, rows); };
  tile(at(0), "Temperature", num(live.t, 1), "C   HDC1080", health[S_HDC], full);
  tile(at(1), "Humidity", num(live.rh, 0), "%   HDC1080", health[S_HDC], full);
  tile(at(2), "Pressure", num(live.p, 0), "hPa   BMP280", health[S_BMP], full);
  float speed = isnan(live.wind_u) ? NAN : sqrtf(live.wind_u * live.wind_u + live.wind_v * live.wind_v);
  tile(at(3), "Wind", num(speed, 1), "m/s  x " + num(live.wind_u, 1) + "  y " + num(live.wind_v, 1), health[S_FANS], full);
  tile(at(4), "Rain plate", live.wet ? "WET" : "DRY", "now", health[S_RAIN], full, live.wet ? C_EDGE : C_TEXT);
  tile(at(5), "Light", num(live.lux, 0), "lux   BH1750", health[S_LUX], full);
}

// gas reading vs its clean-air baseline of the last 7 days (what the model sees); "--" before 24 h of data
static String vsCleanAir(const float* series, float value, float q) {
  float base = histCount > 0 ? baselineAt(series, histCount, q) : NAN;
  return isnan(base) || isnan(value) ? String("baseline after 24 h") : String(value / base, 2) + "x clean air";
}

static void pageGases(bool full) {
  int cols = tileCols(), rows = 6 / cols;
  auto at = [&](int i) { return cell(BODY, i % cols, i / cols, cols, rows); };
  tile(at(0), "PM sensor (raw)", num(live.pm25), "PM1 " + num(live.pm1) + "  PM10 " + num(live.pm10), health[S_PM], full);
  // MQ resistances, compensated for T / RH like the model sees them
  float t = isnan(live.t) ? 20 : live.t, rh = isnan(live.rh) ? 60 : live.rh;
  float rc7 = live.mq7_rs * expf(0.007f * (t - 20) + 0.0028f * (rh - 65));
  float rc135 = live.mq135_rs * expf(0.0105f * (t - 20) + 0.0055f * (rh - 55));
  String heater = MQ7_HEATER_CYCLE ? (mq7CycleSeconds() < 60 ? "  [heating]" : "  [sensing]") : "";
  tile(at(1), "MQ-7 (CO)  kOhm" + heater, num(live.mq7_rs / 1000, 1), vsCleanAir(d_rc7, rc7, 0.90f), health[S_MQ7], full);
  tile(at(2), "MQ-135  kOhm", num(live.mq135_rs / 1000, 1), vsCleanAir(d_rc135, rc135, 0.90f), health[S_MQ135], full);
  float vocBase = histCount > 0 ? baselineAt(d_voc, histCount, 0.90f) : NAN;
  float noxBase = histCount > 0 ? baselineAt(d_nox, histCount, 0.10f) : NAN;
  tile(at(3), "SGP41 VOC (raw)", num(live.voc_raw), "vs clean air " + signedNum(live.voc_raw - vocBase), health[S_SGP], full);
  tile(at(4), "SGP41 NOx (raw)", num(live.nox_raw), "vs clean air " + signedNum(live.nox_raw - noxBase), health[S_SGP], full);
  tile(at(5), "MQ ADC (V)", num(live.mq7_v, 2) + " " + num(live.mq135_v, 2), "MQ-7  MQ-135  (ok 0.05-3.1)",
       health[S_MQ7] == H_RANGE || health[S_MQ135] == H_RANGE ? H_RANGE : H_OK, full);
}

// ======================= page: health =======================================================================
static String ago(uint32_t ms) {
  if (!ms) return "never";
  uint32_t s = (millis() - ms) / 1000;
  return s < 60 ? String(s) + " s" : s < 3600 ? String(s / 60) + " min" : String(s / 3600) + " h";
}

static void pageHealth(bool full) {
  bool wide = W >= H;
  Box left = wide ? sub(BODY, 0, 0, 0.56, 1) : sub(BODY, 0, 0, 1, 0.58);
  Box right = wide ? sub(BODY, 0.58, 0, 0.42, 1) : sub(BODY, 0, 0.6, 1, 0.4);

  Box list = card(left, "Sensors", full, C_EDGE);
  for (int s = 0; s < N_SENSORS; s++) {
    Box row = cell(list, 0, s, 1, N_SENSORS);
    fitText(sub(row, 0, 0, 0.36, 1), SENSOR_NAMES[s], 2, C_TEXT, C_CARD, ML_DATUM);
    fitText(sub(row, 0.37, 0, 0.38, 1), healthText(health[s]), 2, healthColor(health[s]), C_CARD, ML_DATUM);
    fitText(sub(row, 0.76, 0, 0.24, 1), ago(lastOkMs[s]), 1, C_DIM, C_CARD, MR_DATUM);
  }

  Box sys = card(right, "System", full, C_EDGE);
  String lines[8][2] = {
    {"Wi-Fi", WiFi.status() == WL_CONNECTED ? String(WiFi.RSSI()) + " dBm" : String("offline")},
    {"Clock", String(clockSource) + " " + clockText(time(nullptr))},
    {"History", String(histCount) + " / " + String(HISTORY) + " h"},
    {"Model", modelStatus},
    {"Forecast", edge.ok ? "ok, " + String(edge.runMs) + " ms" : String(edge.why)},
    {"Cloud", cloudMatches() ? String("ok, ") + cloud.model : String(cloud.why)},
    {"Memory", String(ESP.getFreeHeap() / 1024) + " KB free"},
    {"Uptime", String(millis() / 3600000) + " h " + String(millis() / 60000 % 60) + " min"},
  };
  for (int i = 0; i < 8; i++) {
    Box row = cell(sys, 0, i, 1, 8);
    fitText(sub(row, 0, 0, 0.34, 1), lines[i][0], 2, C_DIM, C_CARD, ML_DATUM);
    fitText(sub(row, 0.35, 0, 0.65, 1), lines[i][1], 2, C_TEXT, C_CARD, ML_DATUM);
  }
}

// ======================= screen ===============================================================================
void setupScreen() {
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_BUZZER, OUTPUT);
  tft.init();
  tft.setRotation(SCREEN_ROTATION);
  W = tft.width();
  H = tft.height();
  GAP = max(3, min(W, H) / 60);
  RADIUS = max(3, min(W, H) / 40);
  int headerH = max(20, H * 11 / 100), footerH = max(14, H * 8 / 100);
  HEADER = {0, 0, W, headerH};
  FOOTER = {0, H - footerH, W, footerH};
  BODY = {GAP, headerH + GAP, W - 2 * GAP, H - headerH - footerH - 2 * GAP};
  tft.fillScreen(C_BG);
}

// start-up screen with a progress bar
void showSplash(const char* text, float progress) {
  static bool drawnOnce = false;
  if (!drawnOnce) {
    tft.fillScreen(C_BG);
    nDrawn = 0;
    drawnOnce = true;
  }
  Box all = {0, 0, W, H};
  fitText(sub(all, 0.1, 0.25, 0.8, 0.2), "AQI node", 4, C_TEXT, C_BG);
  fitText(sub(all, 0.1, 0.45, 0.8, 0.1), "edge forecast of PM2.5 / PM10", 2, C_DIM, C_BG);
  fitText(sub(all, 0.1, 0.62, 0.8, 0.08), text, 2, C_EDGE, C_BG);
  Box bar = sub(all, 0.15, 0.74, 0.7, 0.03);
  tft.drawRect(bar.x - 1, bar.y - 1, bar.w + 2, bar.h + 2, C_LINE);
  tft.fillRect(bar.x, bar.y, (int)(bar.w * constrain(progress, 0.0f, 1.0f)), bar.h, C_EDGE);
}

void drawScreen(bool full) {
  if (full) {
    tft.fillScreen(C_BG);
    nDrawn = 0;                                // everything is new
  }
  drawHeader(full);
  switch (page) {
    case P_NOW: pageNow(full); break;
    case P_CHART25: pageChart(full, 0); break;
    case P_CHART10: pageChart(full, 1); break;
    case P_TABLE: pageTable(full); break;
    case P_WEATHER: pageWeather(full); break;
    case P_GASES: pageGases(full); break;
    case P_HEALTH: pageHealth(full); break;
  }
  drawFooter(full);
}

// ======================= button and buzzer ==================================================================
static void beep(int freq, int ms) { tone(PIN_BUZZER, freq, ms); }

// tap = next page (or next alert level); hold 1 s = alert mode on / off; alert mode ends after 10 s idle
void handleButton() {
  static bool down = false, held = false;
  static uint32_t downMs = 0;
  bool pressed = digitalRead(PIN_BUTTON) == LOW;
  if (pressed && !down) { down = true; held = false; downMs = millis(); }
  if (pressed && down && !held && millis() - downMs > 1000) {
    held = true;
    alertMode = !alertMode;
    alertModeMs = millis();
    if (!alertMode) prefs.putInt("alert", alertAqi);
    beep(alertMode ? 2000 : 1400, 80);
    dataChanged = true;
  }
  if (!pressed && down) {
    down = false;
    if (!held && millis() - downMs > 40) {      // a real tap, not a bounce
      if (alertMode) {
        int k = 0;
        while (k < 4 && ALERT_LEVELS[k] <= alertAqi) k++;
        alertAqi = ALERT_LEVELS[k % 4];
        alertModeMs = millis();
        beep(1800, 40);
      } else {
        page = (page + 1) % N_PAGES;
      }
      dataChanged = true;
    }
  }
  if (alertMode && millis() - alertModeMs > 10000) {
    alertMode = false;
    prefs.putInt("alert", alertAqi);
    dataChanged = true;
  }
}

// after each forecast: rising tones when the next 3 hours reach the alert level (at most once an hour)
void checkAlert() {
  if (!edge.ok) return;
  float worst = max(edge.aqi[0], max(edge.aqi[1], edge.aqi[2]));
  float now = aqiNow();
  if (worst >= alertAqi && (isnan(now) || now < alertAqi)) {
    for (int f : {1400, 1800, 2200}) { beep(f, 120); delay(150); }
  }
}
