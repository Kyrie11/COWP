#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
CASE="${1:?Usage: bash scripts/03_precommit_scenes.sh convoy|cut_in|intersection|support_gap}"
case "$CASE" in convoy|cut_in|intersection|support_gap) ;; *) echo "Unknown case: $CASE" >&2; exit 2;; esac
mkdir -p "$RIDE_OUTPUT_ROOT/frozen"
OUT="$RIDE_OUTPUT_ROOT/frozen/${CASE}_anchors.jsonl"
if [[ -e "$OUT" ]]; then echo "REFUSING TO OVERWRITE frozen anchors: $OUT" >&2; exit 3; fi
cd "$RIDE_HOME"
python -m ride.carla.ports --mode client --host "$CARLA_HOST" \
  --rpc-port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT"
python -m ride.carla.run \
  --host "$CARLA_HOST" --port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT" \
  --archetype "$CASE" --town "${CARLA_TOWN:-Town04}" \
  --anchors "${RIDE_ANCHORS:-1}" --seed "${RIDE_SEED:-17}" \
  --cruise-mps "${RIDE_CRUISE_MPS:-9}" \
  --warmup-ticks "${RIDE_WARMUP_TICKS:-20}" \
  --timeout "${RIDE_CARLA_TIMEOUT:-120}" --manifest-only --no-rendering --output "$OUT"
echo "Frozen scene locations BEFORE outcome rollouts: $OUT"
