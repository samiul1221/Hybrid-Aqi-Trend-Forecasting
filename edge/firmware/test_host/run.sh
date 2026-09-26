#!/usr/bin/env bash
# Build and run every test_host/test_*.cpp with the host g++ against aqi_node/src (pure-logic code only).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/../../../build/test_host"
mkdir -p "$out"
for t in "$here"/test_*.cpp; do
  name="$(basename "$t" .cpp)"
  g++ -std=c++14 -Wall -Wextra -Werror -O1 -I"$here/../aqi_node/src" "$t" -o "$out/$name"
  "$out/$name"
done
