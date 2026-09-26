#!/usr/bin/env bash
# Compile the Arduino sketch for ESP32-S3 N16R8 (OPI PSRAM, 16 MB flash). ADR 0002.
# ponytail: default partition scheme until P10 adds partitions.csv (then PartitionScheme=custom).
set -euo pipefail
cd "$(dirname "$0")/.."
FQBN="${FQBN:-esp32:esp32:esp32s3:PSRAM=opi,FlashSize=16M}"
arduino-cli compile --fqbn "$FQBN" --build-path build/firmware "$@" edge/firmware/aqi_node
