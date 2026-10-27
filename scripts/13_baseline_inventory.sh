#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
BASELINE_ROOT="${RIDE_BASELINE_ROOT:-/data0/senzeyu2/dataset/RIDE/research_baselines}"
mkdir -p "$RIDE_OUTPUT_ROOT"
python -m ride.baseline_inventory --root "$BASELINE_ROOT" \
  --output "$RIDE_OUTPUT_ROOT/baseline_inventory.json"
