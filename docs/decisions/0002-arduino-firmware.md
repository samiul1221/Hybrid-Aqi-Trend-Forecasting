# 0002 — Firmware in Arduino (arduino-esp32 core), not ESP-IDF

Status: accepted (2026-09-26, owner decision)

## Context
[S05], [S13.4] and [S23] assume an ESP-IDF v5.x project (`main/`, `components/`, `idf.py`). The owner will write the firmware as an Arduino sketch (`.ino`) and will not use ESP-IDF directly.

## Decision
- Toolchain: arduino-esp32 core 3.x (built on ESP-IDF 5.x, so FreeRTOS, LittleFS, NVS, esp_timer and mbedTLS are still available) via `arduino-cli`; Arduino IDE also works.
- Board: FQBN `esp32:esp32:esp32s3` with OPI PSRAM, 16 MB flash, custom `partitions.csv` in the sketch folder (TODO_VERIFY exact FQBN options in P10).
- Layout: one sketch `edge/firmware/aqi_node/aqi_node.ino` plus `aqi_node/src/<component>/` (Arduino compiles `src/` recursively). Components keep the [S23] names: sensors, pipeline, aqi, features, models, decision, storage, comms, ota, ui.
- Pure logic (pipeline, aqi, features, models evaluator, decision) is plain C/C++ with no Arduino headers, so it compiles on the host.
- Host tests: `edge/firmware/test_host/` built with the host g++ and small assert-based test programs (no CMake, no Unity) run by `scripts/test-all.sh`. They check the same `shared/fixtures/` as Python.
- Spec → Arduino mapping:

| Spec (IDF) | Arduino equivalent |
|---|---|
| FreeRTOS tasks [S05] | `xTaskCreatePinnedToCore` (available in the Arduino core) |
| LittleFS / NVS | `LittleFS.h` / `Preferences.h` |
| esp-mqtt (TLS, LWT, persistent session) | `PubSubClient` over `WiFiClientSecure` (TODO_VERIFY QoS1 subscribe + LWT cover [S16.7]; PubSubClient publishes QoS0 only, which is fine since the node only publishes `status`) |
| esp_http_client | `HTTPClient` + `WiFiClientSecure` |
| esp-tflite-micro | Arduino TFLite library (`TensorFlowLite_ESP32` or `Chirale_TensorFlowLite`; pick in P7 by op coverage incl. DEPTHWISE_CONV_2D with dilation) |
| emlearn C evaluator | emlearn headers vendored into `src/models/` (same C code as the host parity build) |
| OTA dual partitions + rollback [S18] | `Update.h` / `HTTPUpdate` + app rollback API (TODO_VERIFY in P10) |
| adc_cali curve fitting | `analogReadMilliVolts()` (uses eFuse calibration) |

## Consequences
- The [S05] architecture, [S15] logic and [S16] contracts are unchanged; only the build system and driver APIs change.
- MQTT library choice must be re-checked against [S16.7] in P10.
