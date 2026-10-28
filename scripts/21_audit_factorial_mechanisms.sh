#!/usr/bin/env bash
# Recompute mechanism timing/burden diagnostics for every frozen v6 panel.
# This is an audit only; it never changes the primary ranking endpoint.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
root="${RIDE_FACTORIAL_OUTPUT_ROOT:-$DATASET_ROOT/RIDE/evidence/rq12_v6_factorial}"
found=0
while IFS= read -r -d '' panel; do
  found=1
  out="${panel%.jsonl}_mechanism_audit.json"
  python -m ride.carla.mechanism_audit --input "$panel" --output "$out"
done < <(find "$root" -type f -path '*/carla/cut_in.jsonl' -print0 | sort -z)
if [[ "$found" -eq 0 ]]; then
  echo "No factorial panels under $root" >&2
  exit 2
fi
python -m ride.carla.pilot_report --root "$root" --output "$root/all_cells_audit.json"
