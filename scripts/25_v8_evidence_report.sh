#!/usr/bin/env bash
# Read only original CARLA trial outputs. First path is DEV; rest CONFIRM.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_DEV_TRIALS:?Set RIDE_DEV_TRIALS=<dev>/carla/cut_in.jsonl}"
: "${RIDE_CONFIRM_ROOT:?Set RIDE_CONFIRM_ROOT}"
: "${RIDE_REPORT_JSON:?Set RIDE_REPORT_JSON output path}"
files=("$RIDE_DEV_TRIALS")
for seed in ${RIDE_CONFIRM_SEEDS:-23 41 73}; do
  files+=("$RIDE_CONFIRM_ROOT/seed_${seed}/carla/cut_in.jsonl")
done
python -m ride.v8_gate_audit --panels "${files[@]}" --output "$RIDE_REPORT_JSON" "$@"
