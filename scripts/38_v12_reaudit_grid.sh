#!/usr/bin/env bash
# Read-only sensitivity to predeclared display grid, never overwrite old results.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_V11_GRID_ROOT:?path to EXISTING v11_chain_grid_town04_seed17}"
: "${RIDE_V12_AUDIT_OUTPUT:?NEW JSON output path}"
paths=()
for bc in 20 25; do
  for rear in 35 45; do
    file="$RIDE_V11_GRID_ROOT/bc_${bc}_rear_${rear}/carla/chain_cut_in.jsonl"
    [[ -s "$file" ]] || { echo "MISSING_CELL $file" >&2; exit 4; }
    paths+=("$file")
  done
done
python -m ride.v12_sensitivity --input "${paths[@]}" --output "$RIDE_V12_AUDIT_OUTPUT"
