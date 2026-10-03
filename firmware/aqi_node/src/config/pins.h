// Hardware pin map and bus constants for the AQI node (ESP32-S3-DevKitC-1 N16R8).
// Source: docs/specs/edge_implementation_plan.md [S04.2] pin map, [S04.3] electrical rules.
// Single place for every GPIO/bus number: drivers must include this header, never hard-code pins.
// TODO_VERIFY: check every GPIO against the actual devkit silkscreen before first power-on.
#pragma once

// ---- Pins to avoid on ESP32-S3 N16R8 [S04.2] -------------------------------------------------
// GPIO26-32: SPI flash | GPIO33-37: octal PSRAM | GPIO19-20: USB D-/D+ | strapping: 0, 3, 45, 46
// ADC2 is unusable while Wi-Fi is active -> all analog inputs are on ADC1 (GPIO1-10).

// ---- I2C bus (100 kHz, 4.7 kOhm pull-ups to 3V3) ---------------------------------------------
#define PIN_I2C_SDA 8
#define PIN_I2C_SCL 9
#define I2C_FREQ_HZ 100000        // SPS30 on I2C would need <= 100 kHz; all devices share this bus

// I2C addresses (M1 acceptance: scan shows 0x23, 0x40, 0x59, 0x68, 0x76)
#define I2C_ADDR_BH1750 0x23      // lux (ADDR pin low)
#define I2C_ADDR_HDC1080 0x40     // T/RH (fixed)
#define I2C_ADDR_SGP41 0x59       // VOC/NOx raw (fixed)
#define I2C_ADDR_DS3231 0x68      // RTC (fixed)
#define I2C_ADDR_BMP280 0x76      // pressure (SDO low; 0x77 if SDO high) TODO_VERIFY module strap
#define I2C_ADDR_INA219 0x41      // optional energy monitor: A0 strapped HIGH to avoid 0x40 clash

// ---- SPS30 on UART (SHDLC 115200 8N1, SEL pin floating) -------------------------------------
#define PIN_SPS30_TX 17           // ESP TX -> SPS30 RX
#define PIN_SPS30_RX 18           // ESP RX <- SPS30 TX
#define SPS30_BAUD 115200

// ---- Analog inputs (ADC1, 12 dB attenuation, 16-sample average, analogReadMilliVolts) -------
#define PIN_MQ7_AOUT 1            // via 10k/20k divider
#define PIN_MQ135_AOUT 2          // via 10k/20k divider
#define PIN_WIND_VANE_ADC 4       // resistor ladder, 16 positions
#define PIN_RAIN_AO 6             // optional analog wetness
#define ADC_SAMPLES 16

// MQ divider and load resistor [S04.3]: Rs = RL * (Vc - Vout) / Vout, Vout = V_adc / DIVIDER_SCALE
#define MQ_DIVIDER_SCALE 0.6667f  // 10k over 20k: sensor-side 5 V -> 3.33 V at the ADC
#define MQ_VC_VOLTS 5.0f
#define MQ7_RL_OHM 10000.0f       // TODO_VERIFY: measure the load resistor on the actual module
#define MQ135_RL_OHM 10000.0f     // TODO_VERIFY: measure the load resistor on the actual module

// ---- MQ-7 heater (LEDC PWM -> logic-level MOSFET) [S04.3] ------------------------------------
#define PIN_MQ7_HEATER_PWM 12
#define MQ7_PWM_FREQ_HZ 1000      // >= 1 kHz
#define MQ7_PWM_BITS 10
#define MQ7_HIGH_PHASE_S 60       // 5.0 V
#define MQ7_LOW_PHASE_S 90        // 1.4 V (Hanwei) / 1.5 V (Winsen): TODO_VERIFY which part is fitted
#define MQ7_LOW_DUTY 0.078f       // (1.4/5)^2 heater-power equivalent; 0.09 for 1.5 V
#define MQ_BURN_IN_H 48           // readings flagged 'warming' until then

// ---- Digital inputs / outputs ------------------------------------------------------------------
#define PIN_ANEMOMETER 11         // reed pulses, INPUT_PULLUP, interrupt, 10 ms debounce
#define ANEMO_DEBOUNCE_MS 10
#define ANEMO_MS_PER_HZ 0.6667f   // 2.4 km/h per pulse/s: TODO_VERIFY from the kit datasheet
#define PIN_RAIN_DO 5             // wet = 1 (TODO_VERIFY polarity of the board)
#define PIN_BUZZER 13
#define PIN_BUTTON 14             // INPUT_PULLUP, active low
#define PIN_RTC_SQW 15            // optional 1 Hz from DS3231
