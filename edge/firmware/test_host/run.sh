#!/usr/bin/env bash
# Build and run every test_host/test_*.cpp with the host compiler, linked against the pure-logic C
# sources of the sketch (src/{pipeline,aqi,features,models,decision}/*.c — no Arduino headers there).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/../aqi_node/src"
out="$here/../../../build/test_host"
mkdir -p "$out"
objs=()
while IFS= read -r -d '' c; do  # NUL-separated: the project path contains spaces
  o="$out/$(basename "$c" .c).o"
  gcc -std=c99 -Wall -Wextra -Werror -O1 -I"$src" -c "$c" -o "$o"
  objs+=("$o")
done < <(find "$src/pipeline" "$src/aqi" "$src/features" "$src/models" "$src/decision" -name '*.c' -print0 2>/dev/null)
for t in "$here"/test_*.cpp; do
  name="$(basename "$t" .cpp)"
  g++ -std=c++14 -Wall -Wextra -Werror -O1 -I"$src" "$t" "${objs[@]}" -o "$out/$name"
  "$out/$name"
done
