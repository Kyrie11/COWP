#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
mkdir -p "$RIDE_OUTPUT_ROOT/diagnostics"
python -m ride.doctor --carla-root "$CARLA_ROOT" "$@" \
  2>&1 | tee "$RIDE_OUTPUT_ROOT/diagnostics/doctor.log"
python -m ride.datasets \
  --raw-root "$RAW_WOMD_ROOT" \
  --existing-ids "$RIDE_HOME/references/prior_development_exclusion_ids.txt" \
  --output "$RIDE_OUTPUT_ROOT/diagnostics/datasets.json"
