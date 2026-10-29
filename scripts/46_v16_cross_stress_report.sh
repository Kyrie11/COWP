#!/usr/bin/env bash
# Independent analysis only: accepts completed ZIPs or CARLA raw panel JSONL.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_V16_BASELINE:?completed baseline CARLA panel or ZIP required}"
: "${RIDE_V16_CAP:?completed cap-only CARLA panel or ZIP required}"
: "${RIDE_V16_DELAY:?completed delay-only CARLA panel or ZIP required}"
: "${RIDE_V16_BOTH:?completed combined panel or ZIP required}"
: "${RIDE_V16_REPORT:?new JSON output path required}"
extra=()
if [[ -n "${RIDE_V16_LOCKED:-}" ]]; then extra+=(--locked "$RIDE_V16_LOCKED"); fi
python -m ride.v16_cross_stress report \
  --baseline "$RIDE_V16_BASELINE" \
  --cap "$RIDE_V16_CAP" \
  --delay "$RIDE_V16_DELAY" \
  --both "$RIDE_V16_BOTH" \
  "${extra[@]}" --output "$RIDE_V16_REPORT"
