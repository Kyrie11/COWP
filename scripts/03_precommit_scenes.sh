#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
CASE="${1:?Usage: bash scripts/03_precommit_scenes.sh convoy|cut_in|chain_cut_in|intersection|support_gap}"
case "$CASE" in convoy|cut_in|chain_cut_in|intersection|support_gap) ;; *) echo "Unknown case: $CASE" >&2; exit 2;; esac
mkdir -p "$RIDE_OUTPUT_ROOT/frozen"
OUT="$RIDE_OUTPUT_ROOT/frozen/${CASE}_anchors.jsonl"
if [[ -e "$OUT" ]]; then echo "REFUSING TO OVERWRITE frozen anchors: $OUT" >&2; exit 3; fi
read -r -a _EXCLUDE_FILES <<< "${RIDE_EXCLUDE_ANCHORS_FROM:-}"
_EXCLUSION_ARGS=()
if (( ${#_EXCLUDE_FILES[@]} )); then
  _EXCLUSION_ARGS+=(--exclude-anchors-from "${_EXCLUDE_FILES[@]}" --exclude-radius-m "${RIDE_EXCLUDE_RADIUS_M:-90}")
fi
cd "$RIDE_HOME"
python -m ride.carla.ports --mode client --host "$CARLA_HOST" \
  --rpc-port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT"
python -m ride.carla.run \
  --host "$CARLA_HOST" --port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT" \
  --archetype "$CASE" --town "${CARLA_TOWN:-Town04}" \
  --anchors "${RIDE_ANCHORS:-1}" --seed "${RIDE_SEED:-17}" \
  --cruise-mps "${RIDE_CRUISE_MPS:-8}" \
  --rear-speed-delta-mps "${RIDE_REAR_SPEED_DELTA_MPS:-0}" \
  --cutin-ego-gap-m "${RIDE_CUTIN_EGO_GAP_M:-20}" \
  --cutin-rear-gap-m "${RIDE_CUTIN_REAR_GAP_M:-45}" \
  --chain-bc-gap-m "${RIDE_CHAIN_BC_GAP_M:-25}" \
  --chain-c-speed-delta-mps "${RIDE_CHAIN_C_SPEED_DELTA_MPS:-0}" \
  --behavior-speed-protocol "${RIDE_BEHAVIOR_SPEED_PROTOCOL:-nominal_cap}" \
  --behavior-route-protocol "${RIDE_BEHAVIOR_ROUTE_PROTOCOL:-native_shortest}" \
  --cutin-paired-amplitude-mps "${RIDE_CUTIN_PAIRED_AMPLITUDE_MPS:-0.5}" \
  --cutin-late-switch-s "${RIDE_CUTIN_LATE_SWITCH_S:-4.5}" \
  --cutin-lateral-lookahead-m "${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-15}" \
  --warmup-ticks "${RIDE_WARMUP_TICKS:-20}" \
  --timeout "${RIDE_CARLA_TIMEOUT:-120}" --manifest-only --no-rendering \
  "${_EXCLUSION_ARGS[@]}" --output "$OUT"
echo "Frozen scene locations BEFORE outcome rollouts: $OUT"
