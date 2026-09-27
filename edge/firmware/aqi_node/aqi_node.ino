// AQI edge node (ESP32-S3 N16R8, arduino-esp32 core 3.x). Spec: docs/specs/edge_implementation_plan.md
// Architecture [S05] and component layout: docs/decisions/0002-arduino-firmware.md
// Placeholder until milestone P10: tasks, pipeline, models and comms land in src/<component>/.
// All GPIO numbers and bus constants live in src/config/pins.h.
#include "src/config/pins.h"

void setup() {
  Serial.begin(115200);
  Serial.println("aqi_node: scaffold only (P10 pending)");
}

void loop() {
  delay(1000);
}
