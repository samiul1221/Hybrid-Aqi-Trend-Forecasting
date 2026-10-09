// cloud.cpp - Wi-Fi, the clock (NTP + DS3231 battery clock) and the cloud forecast.
//
// The cloud server (backend/, a later phase) gets every closed hour and answers with its teacher's forecast for
// the same hour, so the screen can show how far the edge forecast is from the cloud one. Contract:
//   POST CLOUD_URL   {"node": "<MAC>", "start": <UTC epoch of the IST hour>, "pm1": .., "pm25": .., ... every HourRec field}
//   reply 200        {"start": <the same epoch>, "model": "T3", "pm25": [+1h, +2h, +3h], "pm10": [+1h, +2h, +3h]}
// PM in the reply is on the node's own scale (humidity-corrected sensor PM), like the edge forecast.
// Without WIFI_SSID / CLOUD_URL in secrets.h the node works alone and the screen says "cloud not set up".
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <RTClib.h>
#include <sys/time.h>
#include "esp_sntp.h"
#include "aqi_node.h"

static RTC_DS3231 rtc;
static bool hasRtc = false;
static volatile bool ntpJustSynced = false;
char clockSource[8] = "none";

// the clock is good after 1 Jan 2024
bool clockValid() { return time(nullptr) > 1704067200; }

// called by the NTP task: only raise a flag (the DS3231 is set from loop(), never from two tasks at once)
static void onNtpSync(struct timeval*) { ntpJustSynced = true; }

void setupClock() {
  hasRtc = i2cPresent(0x68) && rtc.begin();
  if (hasRtc && !rtc.lostPower()) {             // start from the battery clock (UTC)
    struct timeval now = {(time_t)rtc.now().unixtime(), 0};
    settimeofday(&now, nullptr);
    strcpy(clockSource, "RTC");
  }
  setHealth(S_RTC, !hasRtc ? H_ABSENT : rtc.lostPower() ? H_RANGE : H_OK);
#ifdef WIFI_SSID
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  sntp_set_time_sync_notification_cb(onNtpSync);
  configTime(0, 0, "pool.ntp.org", "time.google.com");   // UTC; IST is added where needed
#endif
}

// every loop: copy a fresh NTP time into the DS3231, reconnect Wi-Fi now and then
void keepClock() {
  if (ntpJustSynced) {
    ntpJustSynced = false;
    strcpy(clockSource, "NTP");
    if (hasRtc) {
      rtc.adjust(DateTime((uint32_t)time(nullptr)));
      setHealth(S_RTC, H_OK);
    }
  }
#ifdef WIFI_SSID
  static uint32_t lastTry = 0;
  if (WiFi.status() != WL_CONNECTED && millis() - lastTry > 30000) {
    lastTry = millis();
    WiFi.reconnect();
  }
#endif
}

// send the closed hour, read the cloud forecast for it
void sendHourToCloud(const HourRec& r) {
#if defined(WIFI_SSID) && defined(CLOUD_URL)
  cloud.ok = false;
  if (WiFi.status() != WL_CONNECTED) { strcpy(cloud.why, "cloud: no Wi-Fi"); return; }
  JsonDocument doc;                              // NaN values become null
  doc["node"] = WiFi.macAddress();
  doc["start"] = r.start;
  doc["pm1"] = r.pm1; doc["pm25"] = r.pm25; doc["pm10"] = r.pm10; doc["pm_measured"] = r.pm_measured;
  doc["t"] = r.t; doc["rh"] = r.rh; doc["p"] = r.p; doc["wind_u"] = r.wind_u; doc["wind_v"] = r.wind_v;
  doc["rain_frac"] = r.rain_frac; doc["lux"] = r.lux; doc["mq7_rs"] = r.mq7_rs; doc["mq135_rs"] = r.mq135_rs;
  doc["sraw_voc"] = r.sraw_voc; doc["sraw_nox"] = r.sraw_nox;
  String body;
  serializeJson(doc, body);

  HTTPClient http;
  http.begin(CLOUD_URL);
  http.setTimeout(5000);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  if (code != 200) {
    snprintf(cloud.why, sizeof(cloud.why), "cloud: HTTP %d", code);
    http.end();
    return;
  }
  JsonDocument reply;
  DeserializationError err = deserializeJson(reply, http.getString());
  http.end();
  if (err || reply["pm25"].size() != 3 || reply["pm10"].size() != 3) { strcpy(cloud.why, "cloud: bad reply"); return; }
  cloud.base = reply["start"] | 0;
  for (int h = 0; h < 3; h++) {
    cloud.pm25[h] = reply["pm25"][h] | NAN;
    cloud.pm10[h] = reply["pm10"][h] | NAN;
  }
  strlcpy(cloud.model, reply["model"] | "teacher", sizeof(cloud.model));
  cloud.ok = cloud.base == r.start;
  strcpy(cloud.why, cloud.ok ? "" : "cloud: other hour");
#endif
}
