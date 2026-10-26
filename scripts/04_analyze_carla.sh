#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
mkdir -p "$RIDE_OUTPUT_ROOT/carla"
FILES=()
for scene in convoy cut_in intersection support_gap; do
  f="$RIDE_OUTPUT_ROOT/carla/${scene}.jsonl"
  if [[ -f "$f" ]]; then FILES+=("$f"); fi
done
if [[ ${#FILES[@]} -eq 0 ]]; then echo "No CARLA JSONLs; run cases first" >&2; exit 3; fi
python -m ride.combine "${FILES[@]}" --output "$RIDE_OUTPUT_ROOT/carla/all_scenes.jsonl"
python -m ride.evidence --input "$RIDE_OUTPUT_ROOT/carla/all_scenes.jsonl" --output "$RIDE_OUTPUT_ROOT/carla/all_scenes_audit.json"
python -m ride.ranking --input "$RIDE_OUTPUT_ROOT/carla/all_scenes.jsonl" --output "$RIDE_OUTPUT_ROOT/carla/risk_ranking_disagreement.json"
python -m ride.gates --ranking "$RIDE_OUTPUT_ROOT/carla/risk_ranking_disagreement.json" --audit "$RIDE_OUTPUT_ROOT/carla/all_scenes_audit.json" --output "$RIDE_OUTPUT_ROOT/carla/science_gates.json"
