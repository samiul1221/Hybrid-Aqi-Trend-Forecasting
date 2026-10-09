// pins.h - every pin and hardware constant of the AQI node.
// Board: ESP32-WROOM-32 DevKit, 38-pin, 8 MB flash, no PSRAM (an ESP32-S3 version comes later).
// Sensor pins are the ones of the old box (env-anamoly-3-RF.ino); parts that are new are marked NEW.
// Rules: GPIO 34-39 are input-only (no pull-ups); analog inputs must be on ADC1 (GPIO 32-39) because Wi-Fi
// blocks ADC2; GPIO 6-11 belong to the flash; GPIO 12 must not be HIGH at boot (a button to GND is fine).
// TODO_VERIFY every pin against the wiring before the first power-on.
#pragma once

// ---- I2C bus (the ESP32's default pins) ---------------------------------------------------------------
// HDC1080 0x40, BMP280 0x76 / 0x77, BH1750 0x23 (NEW), SGP41 0x59 (NEW), DS3231 0x68 (NEW)
#define PIN_I2C_SDA 21
#define PIN_I2C_SCL 22

// ---- PM sensor: PMS7003 on UART2, 9600 baud (old box) ---------------------------------------------------
#define PIN_PM_RX 16                 // ESP RX <- PMS TX
#define PIN_PM_TX 17                 // ESP TX -> PMS RX

// ---- MQ gas sensors: module AOUT -> 10k/20k divider -> ADC1 ----------------------------------------------
#define PIN_MQ135_AOUT 36            // old box: MQ-135 on GPIO 36 (VP)
#define PIN_MQ7_AOUT 39              // NEW: GPIO 39 held the LDR in the old box (the BH1750 replaces it)
#define PIN_MQ7_HEATER 27            // NEW: PWM -> logic-level MOSFET -> MQ-7 heater
#define MQ7_HEATER_CYCLE 1           // 1 = datasheet cycle 60 s at 5 V / 90 s at 1.4 V (needs the MOSFET; the
                                     //     training data assumes it). 0 = the module's own constant heater
#define MQ7_LOW_DUTY 0.078f          // (1.4 V / 5 V)^2: same heater power as 1.4 V
#define MQ_VC_VOLTS 5.0f             // module supply voltage
#define MQ_DIVIDER_SCALE 0.6667f     // ADC volts = AOUT volts x this (10k over 20k). 1.0 if AOUT goes straight in at 3.3 V
#define MQ7_RL_OHM 10000.0f          // load resistor on the module. TODO_VERIFY with a multimeter
#define MQ135_RL_OHM 10000.0f        // TODO_VERIFY
#define MQ_WARMUP_S 180              // readings count as "warming" for 3 min after power-on. TODO_VERIFY

// ---- wind: two perpendicular fans, one hall sensor each (old box) -----------------------------------------
#define PIN_FAN_X 26                 // fan 1, axis east-west   -> |u|
#define PIN_FAN_Y 25                 // fan 2, axis north-south -> |v|
#define FAN_M_PER_PULSE 0.1131f      // metres of wind per pulse = 2*pi*0.018 m (old sketch). TODO_VERIFY (calibrate)
#define FAN_DEBOUNCE_MS 10

// ---- rain plate: digital output of the rain board (old box pin; it was read as analog there) -------------
#define PIN_RAIN_DO 35
#define RAIN_WET_LEVEL LOW           // most boards pull DO low when wet. TODO_VERIFY

// ---- user interface (old box) ------------------------------------------------------------------------------
#define PIN_BUTTON 12                // push button to GND, internal pull-up
#define PIN_BUZZER 5
// the display's SPI pins are in tft_setup.h
