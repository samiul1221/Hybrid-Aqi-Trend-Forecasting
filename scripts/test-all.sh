#!/usr/bin/env bash
# Python tests + host C/C++ firmware tests (both read shared/fixtures/).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run pytest "$@"
bash edge/firmware/test_host/run.sh
