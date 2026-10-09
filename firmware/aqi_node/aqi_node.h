// aqi_node.h - the settings and everything the .cpp files share (data + which file offers which function).
#pragma once
#include <Arduino.h>
#include <Preferences.h>
#include <TFT_eSPI.h>
#include "pins.h"
#include "features.h"
#if __has_include("secrets.h")                  // your Wi-Fi (+ optional cloud URL), not in git
#include "secrets.h"
#endif

// ======================= settings =========================================================================
#define PM_SENSOR_PMS7003                     // PM_SENSOR_PMS7003 (now) or PM_SENSOR_SPS30 (later).
                                              // Must match PM_SENSOR in simulator.py (the model learned that sensor).
#ifdef PM_SENSOR_SPS30
#error "SPS30 driver not written yet: use PM_SENSOR_PMS7003"
#endif
const float PM_CALIBRATION = 1.0f;            // shown PM = sensor PM x this, from a co-location test. TODO_VERIFY
const uint8_t SCREEN_ROTATION = 3;            // 1 / 3 = landscape, 0 / 2 = portrait: the layout adapts
const uint32_t SAMPLE_MS = 10000;             // sensors are read every 10 s
const uint32_t SCREEN_MS = 1000;              // live values on screen refresh every second
const int DEFAULT_ALERT_AQI = 201;            // buzzer when the forecast AQI reaches this ("Poor" starts at 201)

// ======================= shared data (defined in aqi_node.ino) ============================================
// latest readings (every 10 s)
struct Live {
  float pm1 = NAN, pm25 = NAN, pm10 = NAN;    // PM sensor, raw (ug/m3)
  float t = NAN, rh = NAN, p = NAN, lux = NAN;
  float wind_u = NAN, wind_v = NAN;           // fan speeds along x / y (m/s, no sign)
  float mq7_rs = NAN, mq135_rs = NAN;         // MQ resistances (ohm)
  float mq7_v = NAN, mq135_v = NAN;           // their ADC voltages
  float voc_raw = NAN, nox_raw = NAN;         // SGP41 ticks
  bool wet = false;
};

// sensor health: shown on the HEALTH page and as dots on the sensor pages
enum Health : uint8_t { H_OK, H_WARMING, H_NO_DATA, H_RANGE, H_ABSENT };
enum Sensor { S_PM, S_HDC, S_BMP, S_LUX, S_SGP, S_MQ7, S_MQ135, S_FANS, S_RAIN, S_RTC, N_SENSORS };
const char* const SENSOR_NAMES[N_SENSORS] = {"PMS7003", "HDC1080", "BMP280", "BH1750", "SGP41",
                                             "MQ-7", "MQ-135", "Wind fans", "Rain plate", "DS3231"};

// the edge forecast (the student on this board)
struct Forecast {
  bool ok = false;
  char why[48] = "starting";                  // why there is no forecast (shown on screen)
  int32_t base = 0;                           // IST hour it starts from (UTC epoch of the hour's start)
  float pm25[3], pm10[3], aqi[3];             // +1, +2, +3 h (PM on the sensor scale, AQI with calibration)
  float now25 = NAN, now10 = NAN;             // humidity-corrected PM of the base hour
  uint32_t runMs = 0;                         // how long the model took
};

// the cloud forecast (the teacher on the server), for the same base hour
struct Cloud {
  bool ok = false;
  char why[40] = "cloud not set up";
  int32_t base = 0;
  float pm25[3], pm10[3];
  char model[12] = "";
};

extern Live live;
extern Health health[N_SENSORS];
extern uint32_t lastOkMs[N_SENSORS];
extern Forecast edge;
extern Cloud cloud;
extern int alertAqi;
extern bool dataChanged;                      // new hour / forecast / page: redraw everything
extern char modelStatus[40];
extern size_t arenaUsedBytes;
extern char clockSource[8];
extern uint32_t bootMs;
extern TFT_eSPI tft;
extern Preferences prefs;

// ======================= functions, by file ===============================================================
// sensors.cpp
void setupSensors();
void readSensors();                           // every 10 s
void readPmStream();                          // every loop
void updateMq7Heater();                       // every loop
bool mq7Readable();
uint32_t mq7CycleSeconds();
bool i2cPresent(uint8_t address);
void setHealth(Sensor s, Health h);

// hourly.cpp
void addSample();                             // every 10 s, closes the hour when a new one starts
void loadHistory();
void runForecast();
float aqiNow();
int32_t hourStartOf(time_t t);

// model.cpp
bool setupModel();
bool runModel(const float* input, float* output);
bool modelReady();

// cloud.cpp
void setupClock();
void keepClock();                             // every loop
bool clockValid();
void sendHourToCloud(const HourRec& r);

// ui.cpp
void setupScreen();
void showSplash(const char* text, float progress);
void drawScreen(bool full);
void handleButton();                          // every loop
void checkAlert();
