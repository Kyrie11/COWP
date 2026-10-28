#!/usr/bin/env bash
# Explicit ONE-TIME setup only. Never called by the read-only preflight or trial scripts.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
TOWN="${1:-${CARLA_TOWN:-Town04}}"
case "$TOWN" in Town[0-9][0-9]|Town[0-9][0-9]_Opt) ;; *) echo "Invalid CARLA town $TOWN" >&2; exit 2 ;; esac
if [[ "${RIDE_ALLOW_INITIAL_MAP_LOAD:-0}" != 1 ]]; then
  echo 'RIDE_MAP_SETUP_REFUSED: map changes are destructive. On your own dedicated idle CARLA server, explicitly run:' >&2
  echo "  RIDE_ALLOW_INITIAL_MAP_LOAD=1 bash scripts/02c_prepare_carla_town.sh $TOWN" >&2
  exit 2
fi
python -m ride.carla.ports --mode client --host "$CARLA_HOST" \
  --rpc-port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT"
python -m ride.carla.map_setup --host "$CARLA_HOST" --port "$CARLA_RPC_PORT" \
  --timeout "${RIDE_CARLA_TIMEOUT:-180}" --town "$TOWN" --allow-world-load
# Independent read-only verification after the one-time, explicitly approved load.
bash "$RIDE_HOME/scripts/29_v9_carla_preflight.sh" "$TOWN"
