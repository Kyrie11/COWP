#!/usr/bin/env bash
# Read-only, no world load/reload, no actor destruction, no scene generation.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
TOWN="${1:-${CARLA_TOWN:-Town04}}"
case "$TOWN" in Town[0-9][0-9]|Town[0-9][0-9]_Opt) ;; *) echo "Invalid CARLA town $TOWN" >&2;exit 2;;esac
printf 'RIDE_V9_SAFE_PREFLIGHT requested=%s cruise=%s mps rear_delta=%s mps\n' \
  "$TOWN" "${RIDE_CRUISE_MPS:-8}" "${RIDE_REAR_SPEED_DELTA_MPS:-0}" >&2
python -m ride.carla.ports --mode client --host "$CARLA_HOST" --rpc-port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT"
python -m ride.carla.health --mode inspect --host "$CARLA_HOST" \
  --port "$CARLA_RPC_PORT" --timeout "${RIDE_CARLA_TIMEOUT:-60}" \
  --expected-town "$TOWN" --require-empty
# No load_world calls. No validity claim until real M0 trials run.
echo 'RIDE_V9_PREFLIGHT_PASS: loaded map and empty CARLA world checked; not physics validation'
