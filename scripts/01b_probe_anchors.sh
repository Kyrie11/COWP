#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
CASE="${1:?Usage: bash scripts/01b_probe_anchors.sh convoy|cut_in|intersection|support_gap [Town04]}"
case "$CASE" in convoy|cut_in|intersection|support_gap) ;; *) echo "Unknown case: $CASE" >&2; exit 2;; esac
TOWN="${2:-${CARLA_TOWN:-Town04}}"
cd "$RIDE_HOME"
# Read-only scientific output; loads the requested CARLA map but writes no file.
# Do not run while another CARLA trial is using the same server.
python -m ride.carla.run \
  --archetype "$CASE" --town "$TOWN" \
  --anchors "${RIDE_ANCHORS:-1}" --seed "${RIDE_SEED:-17}" \
  --cruise-mps "${RIDE_CRUISE_MPS:-9}" \
  --inspect-anchors --no-rendering
