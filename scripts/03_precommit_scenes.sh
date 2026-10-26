#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
CASE="${1:?Usage: bash scripts/03_precommit_scenes.sh convoy|cut_in|intersection|support_gap}"
case "$CASE" in convoy|cut_in|intersection|support_gap) ;; *) exit 2;; esac
mkdir -p "$RIDE_OUTPUT_ROOT/frozen"
cd "$RIDE_HOME"
OUT="$RIDE_OUTPUT_ROOT/frozen/${CASE}_anchors.jsonl"
if [[ -e "$OUT" ]]; then echo "Frozen anchor manifest already exists: $OUT" >&2; exit 3; fi
python -m research_probe.carla_h2.run_carla \
  --archetype "$CASE" --town "${CARLA_TOWN:-Town04}" \
  --anchors "${RIDE_ANCHORS:-1}" --seed "${RIDE_SEED:-17}" \
  --manifest-only --no-rendering --output "$OUT"
echo "Now run external planners on these frozen, OUTCOME-BLIND scene identities."
