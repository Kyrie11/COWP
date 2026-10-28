#!/usr/bin/env bash
# Explore geometry ONLY, no outcome rollout. Usage: bash scripts/27_probe_disjoint_v8.sh 9901 9920
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
start="${1:-9901}"; stop="${2:-9920}"
[[ "$start" =~ ^[0-9]+$ && "$stop" =~ ^[0-9]+$ ]] || { echo 'Use integer seed range' >&2; exit 2; }
[[ "$stop" -ge "$start" ]] || exit 2
base="${RIDE_V8_EVIDENCE_ROOT:-$DATASET_ROOT/RIDE/evidence/rq12_v8_locked}"
reference="$base/dev/seed_17/rear_gap_35/rear_delta_0/frozen/cut_in_anchors.jsonl"
[[ -f "$reference" ]] || { echo "Dev frozen anchors not found: $reference" >&2; exit 3; }
export RIDE_CUTIN_LATE_SWITCH_S=3.5
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M=20
export RIDE_CUTIN_EGO_GAP_M=20
export RIDE_CUTIN_REAR_GAP_M=35
export RIDE_REAR_SPEED_DELTA_MPS=0
export RIDE_SEED=17
export RIDE_ANCHORS=8
export CARLA_TOWN=Town04
export RIDE_WARMUP_TICKS=20
output="$base/prospective_anchor_search"
mkdir -p "$output"
for ((seed=start;seed<=stop;seed++)); do
  export RIDE_ANCHOR_SELECTION_SEED="$seed"
  export RIDE_OUTPUT_ROOT="$output/anchor_seed_${seed}"
  [[ ! -e "$RIDE_OUTPUT_ROOT/frozen/cut_in_anchors.jsonl" ]] || { echo "Skipping existing seed=$seed" >&2; continue; }
  bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" cut_in
  if python -m ride.anchor_disjoint --reference "$reference" \
     --candidate "$RIDE_OUTPUT_ROOT/frozen/cut_in_anchors.jsonl" \
     --radius-m 25 --enforce \
     --output "$RIDE_OUTPUT_ROOT/frozen/disjointness.json"; then
    printf 'Geometry-only candidate: export RIDE_V8_CONFIRM_ANCHOR_SEED=%s\n' "$seed"
    printf 'Review the entire road/lane corridor topology before committing; coordinate-only check is not a proof.\n'
    exit 0
  fi
done
echo 'No location-disjoint seed in this predeclared range. Expand map/seed search before rollout.' >&2
exit 3
