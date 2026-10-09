// aqi_node.ino - AQI edge node: ESP32-WROOM-32 + 2.8" TFT. Forecasts PM2.5 / PM10 for the next 3 hours ON the board.
//
// Every 10 s it reads the sensors; every IST hour it closes one hourly record (the same columns simulator.py
// produces), computes the 23 student features exactly like the notebooks (features.cpp, checked by
// firmware/check_features.py), runs the GDC-CNN student (student_model.h, TensorFlow Lite Micro) and, if a cloud
// server is set up, sends the hour and gets the cloud teacher's forecast back to compare.
//
// Files (the Arduino IDE compiles the whole folder; you open and upload this .ino):
//   aqi_node.ino   shared data, setup(), loop()            aqi_node.h   settings + what the files share
//   sensors.cpp    one function per sensor + its health    hourly.cpp   hourly record, saved history, forecast
//   model.cpp      the student on TensorFlow Lite Micro    cloud.cpp    Wi-Fi, clock (NTP + DS3231), cloud forecast
//   ui.cpp         the dashboard (7 pages, any screen size) + button + buzzer
//   features.h/.cpp  on-device processing + 23 features + preprocessing (= notebooks 02-04)
//   pins.h         every pin                               tft_setup.h  the display driver + SPI pins
//   student_model.h  the trained student (`kaggle_run.py download` puts it here; without it: no forecast)
//   secrets.h      yours, not in git (optional):
//                    #define WIFI_SSID "..."
//                    #define WIFI_PASS "..."
//                    #define CLOUD_URL "http://192.168.1.10:8000/api/node/hour"   // optional, later phase
//
// Libraries: TFT_eSPI, Adafruit BMP280, ClosedCube HDC1080, RTClib, ArduinoJson 7, Chirale_TensorFlowLite.
// Board settings: "ESP32 Dev Module", Flash Size 8MB, Partition Scheme "8M with spiffs" (the history is stored there).

#include "aqi_node.h"

// ======================= shared data ======================================================================
Live live;
Health health[N_SENSORS];
uint32_t lastOkMs[N_SENSORS];
Forecast edge;
Cloud cloud;
int alertAqi = DEFAULT_ALERT_AQI;
bool dataChanged = true;
char modelStatus[40] = "no model: add student_model.h";
size_t arenaUsedBytes = 0;
TFT_eSPI tft;
Preferences prefs;

// ======================= setup / loop =====================================================================
void setup() {
  Serial.begin(115200);
  prefs.begin("aqi", false);
  alertAqi = prefs.getInt("alert", DEFAULT_ALERT_AQI);
  setupScreen();
  showSplash("starting sensors", 0.1f);
  setupSensors();                             // includes 10 s SGP41 conditioning (progress on the splash)
  showSplash("clock + Wi-Fi", 0.8f);
  setupClock();
  showSplash("history + model", 0.9f);
  loadHistory();
  setupModel();
  runForecast();                              // a forecast right away if the saved history is fresh
  showSplash("ready", 1.0f);
  dataChanged = true;
}

void loop() {
  static uint32_t lastSample = 0, lastScreen = 0;
  readPmStream();                             // PMS7003 frames arrive every second
  updateMq7Heater();
  handleButton();
  keepClock();                                // Wi-Fi reconnect, NTP -> DS3231
  if (millis() - lastSample >= SAMPLE_MS) {
    lastSample = millis();
    readSensors();
    addSample();                              // closes the hour (forecast + cloud) when a new IST hour starts
  }
  if (millis() - lastScreen >= SCREEN_MS || dataChanged) {
    lastScreen = millis();
    bool full = dataChanged;
    dataChanged = false;
    drawScreen(full);
  }
}
