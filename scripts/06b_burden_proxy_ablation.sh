#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
PRED="${1:?Usage: bash scripts/06b_burden_proxy_ablation.sh /path/to/predictions.csv}"
mkdir -p "$RIDE_OUTPUT_ROOT/frozen"
python -m ride.burden --predictions "$PRED" \
  --output "$RIDE_OUTPUT_ROOT/frozen/burden_on_off_PROXY_only.json"
echo 'This is a proxy selector ablation, NOT a full planner checkpoint comparison.'
