// pins.h stays consistent with the ESP32-S3 constraints in [S04.2].
#include <cassert>
#include <cstdio>

#include "config/pins.h"

static bool forbidden(int g) {
  return (g >= 26 && g <= 37) || g == 19 || g == 20 || g == 0 || g == 3 || g == 45 || g == 46;
}

int main() {
  const int gpios[] = {PIN_I2C_SDA, PIN_I2C_SCL, PIN_SPS30_TX, PIN_SPS30_RX, PIN_MQ7_AOUT, PIN_MQ135_AOUT,
                       PIN_WIND_VANE_ADC, PIN_RAIN_AO, PIN_MQ7_HEATER_PWM, PIN_ANEMOMETER, PIN_RAIN_DO,
                       PIN_BUZZER, PIN_BUTTON, PIN_RTC_SQW};
  const int n = sizeof(gpios) / sizeof(gpios[0]);
  for (int i = 0; i < n; ++i) {
    assert(!forbidden(gpios[i]));
    for (int j = i + 1; j < n; ++j) assert(gpios[i] != gpios[j]);  // no pin used twice
  }
  const int adc[] = {PIN_MQ7_AOUT, PIN_MQ135_AOUT, PIN_WIND_VANE_ADC, PIN_RAIN_AO};
  for (int g : adc) assert(g >= 1 && g <= 10);  // ADC1 only (ADC2 blocked by Wi-Fi)
  assert(I2C_ADDR_INA219 != I2C_ADDR_HDC1080);
  std::puts("test_pins: ok");
  return 0;
}
