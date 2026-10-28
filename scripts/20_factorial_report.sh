#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
root="${RIDE_FACTORIAL_OUTPUT_ROOT:-$DATASET_ROOT/RIDE/evidence/rq12_v6_factorial}"
python -m ride.carla.pilot_report --root "$root" --output "$root/all_cells_audit.json"
