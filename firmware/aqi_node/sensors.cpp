// sensors.cpp - one function per sensor. Each one fills `live` and sets the sensor's health.
#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <ClosedCube_HDC1080.h>
#include "aqi_node.h"

uint32_t bootMs;                                 // millis() at power-on (MQ warm-up, heater cycle)
static HardwareSerial pmSerial(2);
static ClosedCube_HDC1080 hdc;
static Adafruit_BMP280 bmp;
static bool hasHdc, hasBmp, hasLux, hasSgp;
static volatile uint32_t fanPulses[2], fanLastUs[2];
static uint32_t pmLastMs;

// is there a device at this I2C address?
bool i2cPresent(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

void setHealth(Sensor s, Health h) {
  health[s] = h;
  if (h == H_OK) lastOkMs[s] = millis();
}

// ---- wind fans: count hall pulses (debounced) -------------------------------------------------------------
void IRAM_ATTR fanXPulse() {
  uint32_t now = micros();
  if (now - fanLastUs[0] > FAN_DEBOUNCE_MS * 1000UL) { fanPulses[0]++; fanLastUs[0] = now; }
}
void IRAM_ATTR fanYPulse() {
  uint32_t now = micros();
  if (now - fanLastUs[1] > FAN_DEBOUNCE_MS * 1000UL) { fanPulses[1]++; fanLastUs[1] = now; }
}

// wind speed along each fan's axis since the last call (m/s); a fan does not turn below ~0.3 m/s
void readFans() {
  static uint32_t lastMs = millis();
  float seconds = (millis() - lastMs) / 1000.0f;
  lastMs = millis();
  noInterrupts();
  uint32_t x = fanPulses[0], y = fanPulses[1];
  fanPulses[0] = fanPulses[1] = 0;
  interrupts();
  if (seconds <= 0) return;
  live.wind_u = x * FAN_M_PER_PULSE / seconds;
  live.wind_v = y * FAN_M_PER_PULSE / seconds;
  if (live.wind_u < 0.3f) live.wind_u = 0;
  if (live.wind_v < 0.3f) live.wind_v = 0;
  setHealth(S_FANS, H_OK);                     // a broken fan looks like calm air: cannot be detected here
}

// ---- PMS7003: 32-byte frames, start 0x42 0x4D, checksum at the end ----------------------------------------
void readPmStream() {
  static uint8_t buf[32];
  static int n = 0;
  while (pmSerial.available()) {
    uint8_t c = pmSerial.read();
    if ((n == 0 && c != 0x42) || (n == 1 && c != 0x4D)) { n = 0; continue; }
    buf[n++] = c;
    if (n < 32) continue;
    n = 0;
    uint16_t sum = 0;
    for (int i = 0; i < 30; i++) sum += buf[i];
    if (sum != ((buf[30] << 8) | buf[31])) continue;
    live.pm1 = (buf[10] << 8) | buf[11];       // "atmospheric environment" values (ug/m3)
    live.pm25 = (buf[12] << 8) | buf[13];
    live.pm10 = (buf[14] << 8) | buf[15];
    pmLastMs = millis();
  }
}

void checkPm() {
  bool fresh = pmLastMs && millis() - pmLastMs < 10000;
  setHealth(S_PM, !fresh ? H_NO_DATA : (live.pm25 > 1000 || live.pm10 < live.pm25) ? H_RANGE : H_OK);
}

// ---- HDC1080: temperature + humidity ------------------------------------------------------------------------
void readHdc1080() {
  if (!hasHdc) return setHealth(S_HDC, H_ABSENT);
  float t = hdc.readTemperature(), rh = hdc.readHumidity();
  bool ok = t > -40 && t < 85 && rh >= 0 && rh <= 100;
  live.t = ok ? t : NAN;
  live.rh = ok ? rh : NAN;
  setHealth(S_HDC, ok ? H_OK : H_RANGE);
}

// ---- BMP280: pressure -------------------------------------------------------------------------------------------
void readBmp280() {
  if (!hasBmp) return setHealth(S_BMP, H_ABSENT);
  float p = bmp.readPressure() / 100.0f;
  live.p = (p > 800 && p < 1100) ? p : NAN;
  setHealth(S_BMP, isnan(live.p) ? H_RANGE : H_OK);
}

// ---- BH1750: light. Continuous high-resolution mode; in bright sun the measuring time is shortened
// (MTreg 69 -> 31) so it reaches 121 557 lux like the simulator, instead of stopping at 54 612 ----------------
static uint8_t luxMtreg = 69;

void bh1750Command(uint8_t c) {
  Wire.beginTransmission(0x23);
  Wire.write(c);
  Wire.endTransmission();
}

void bh1750SetMtreg(uint8_t mt) {
  luxMtreg = mt;
  bh1750Command(0x40 | (mt >> 5));             // measuring time, high bits
  bh1750Command(0x60 | (mt & 0x1F));           // measuring time, low bits
  bh1750Command(0x10);                         // continuous high-resolution mode
}

void readBh1750() {
  if (!hasLux) return setHealth(S_LUX, H_ABSENT);
  if (Wire.requestFrom(0x23, 2) != 2) return setHealth(S_LUX, H_NO_DATA);
  uint16_t raw = (Wire.read() << 8) | Wire.read();
  live.lux = raw / 1.2f * 69.0f / luxMtreg;
  if (raw > 60000 && luxMtreg == 69) bh1750SetMtreg(31);       // nearly full: measure shorter
  else if (raw < 25000 && luxMtreg == 31) bh1750SetMtreg(69);  // dark again: back to normal
  setHealth(S_LUX, H_OK);
}

// ---- SGP41: raw VOC / NOx ticks, compensated with the HDC1080's T / RH ------------------------------------
uint8_t crc8(const uint8_t* data, int len) {    // Sensirion CRC: polynomial 0x31, start 0xFF
  uint8_t crc = 0xFF;
  for (int i = 0; i < len; i++) {
    crc ^= data[i];
    for (int b = 0; b < 8; b++) crc = (crc & 0x80) ? (crc << 1) ^ 0x31 : crc << 1;
  }
  return crc;
}

// send a command with RH / T and read `words` 16-bit answers
bool sgp41Command(uint16_t cmd, float rh, float t, int words, uint16_t* out) {
  rh = isnan(rh) ? 50 : constrain(rh, 0, 100);
  t = isnan(t) ? 25 : constrain(t, -45, 130);
  uint16_t rhTicks = rh * 65535 / 100, tTicks = (t + 45) * 65535 / 175;
  uint8_t msg[8] = {uint8_t(cmd >> 8), uint8_t(cmd), uint8_t(rhTicks >> 8), uint8_t(rhTicks), 0,
                    uint8_t(tTicks >> 8), uint8_t(tTicks), 0};
  msg[4] = crc8(msg + 2, 2);
  msg[7] = crc8(msg + 5, 2);
  Wire.beginTransmission(0x59);
  Wire.write(msg, 8);
  if (Wire.endTransmission() != 0) return false;
  delay(50);                                    // measuring time
  if (Wire.requestFrom(0x59, words * 3) != words * 3) return false;
  for (int w = 0; w < words; w++) {
    uint8_t d[3] = {uint8_t(Wire.read()), uint8_t(Wire.read()), uint8_t(Wire.read())};
    if (crc8(d, 2) != d[2]) return false;
    out[w] = (d[0] << 8) | d[1];
  }
  return true;
}

void readSgp41() {
  if (!hasSgp) return setHealth(S_SGP, H_ABSENT);
  uint16_t raw[2];
  if (!sgp41Command(0x2619, live.rh, live.t, 2, raw)) return setHealth(S_SGP, H_NO_DATA);
  live.voc_raw = raw[0];
  live.nox_raw = raw[1];
  setHealth(S_SGP, H_OK);
}

// ---- MQ-7 / MQ-135: Rs from the ADC, exactly like simulator.read_through_adc ------------------------------
// V_out = 5 V * RL / (Rs + RL) -> divider -> ADC;  Rs = RL * (5 V - V_out) / V_out with the nominal RL
float mqResistance(uint8_t pin, float rl, float* adcVolts) {
  uint32_t mv = 0;
  for (int i = 0; i < 16; i++) mv += analogReadMilliVolts(pin);
  *adcVolts = mv / 16.0f / 1000.0f;
  float v = constrain(*adcVolts, 0.05f, 3.10f) / MQ_DIVIDER_SCALE;    // the ADC's readable range
  return rl * (MQ_VC_VOLTS - v) / v;
}

Health mqHealth(float adcVolts) {
  if (adcVolts < 0.05f || adcVolts > 3.10f) return H_RANGE;           // unplugged or saturated
  return (millis() - bootMs) / 1000 < MQ_WARMUP_S ? H_WARMING : H_OK;
}

// MQ-7 heater: 60 s hot (cleans the surface), 90 s at 1.4 V (senses CO); read in the last 10 s of the low phase
uint32_t mq7CycleSeconds() { return ((millis() - bootMs) / 1000) % 150; }
bool mq7Readable() { return !MQ7_HEATER_CYCLE || mq7CycleSeconds() >= 140; }

void updateMq7Heater() {
  if (!MQ7_HEATER_CYCLE) return;
  static int lastDuty = -1;
  int duty = mq7CycleSeconds() < 60 ? 1023 : (int)(MQ7_LOW_DUTY * 1023);
  if (duty != lastDuty) ledcWrite(PIN_MQ7_HEATER, duty);
  lastDuty = duty;
}

void readMq7() {
  if (!mq7Readable()) return;                   // keep the last value until the next sensing window
  live.mq7_rs = mqResistance(PIN_MQ7_AOUT, MQ7_RL_OHM, &live.mq7_v);
  setHealth(S_MQ7, mqHealth(live.mq7_v));
}

void readMq135() {
  live.mq135_rs = mqResistance(PIN_MQ135_AOUT, MQ135_RL_OHM, &live.mq135_v);
  setHealth(S_MQ135, mqHealth(live.mq135_v));
}

// ---- rain plate: 1 = wet ------------------------------------------------------------------------------------
void readRain() {
  live.wet = digitalRead(PIN_RAIN_DO) == RAIN_WET_LEVEL;
  setHealth(S_RAIN, H_OK);
}

// ---- all sensors (every 10 s) ---------------------------------------------------------------------------------
void readSensors() {
  checkPm();
  readHdc1080();                                // first: the SGP41 needs T / RH
  readBmp280();
  readBh1750();
  readSgp41();
  readMq7();
  readMq135();
  readFans();
  readRain();
}

// ---- start everything -------------------------------------------------------------------------------------------
void setupSensors() {
  bootMs = millis();
  for (int s = 0; s < N_SENSORS; s++) health[s] = H_NO_DATA;
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  pmSerial.begin(9600, SERIAL_8N1, PIN_PM_RX, PIN_PM_TX);
  pinMode(PIN_RAIN_DO, INPUT);
  pinMode(PIN_FAN_X, INPUT_PULLUP);
  pinMode(PIN_FAN_Y, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_FAN_X), fanXPulse, FALLING);
  attachInterrupt(digitalPinToInterrupt(PIN_FAN_Y), fanYPulse, FALLING);
  analogSetAttenuation(ADC_11db);               // ADC range up to ~3.1 V
  if (MQ7_HEATER_CYCLE) ledcAttach(PIN_MQ7_HEATER, 1000, 10);

  hasHdc = i2cPresent(0x40);
  if (hasHdc) hdc.begin(0x40);
  hasBmp = bmp.begin(0x76) || bmp.begin(0x77);
  hasLux = i2cPresent(0x23);
  if (hasLux) { bh1750Command(0x01); bh1750SetMtreg(69); }     // power on, normal measuring time
  hasSgp = i2cPresent(0x59);

  // SGP41: 10 s of conditioning after power-on (the NOx pixel needs it), once per second
  for (int s = 0; hasSgp && s < 10; s++) {
    uint16_t voc;
    sgp41Command(0x2612, 50, 25, 1, &voc);
    showSplash("SGP41 conditioning", 0.1f + 0.06f * s);
    delay(950);
  }
}
