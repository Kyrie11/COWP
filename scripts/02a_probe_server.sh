#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$RIDE_HOME"
# Default is non-destructive. For isolated reload test only:
# RIDE_HEALTH_MODE=reload RIDE_ALLOW_DESTRUCTIVE_RELOAD=1 bash scripts/02a_probe_server.sh
EXTRA=()
if [[ "${RIDE_HEALTH_MODE:-inspect}" == reload ]]; then
  [[ "${RIDE_ALLOW_DESTRUCTIVE_RELOAD:-0}" == 1 ]] || {
    echo 'ERROR: reload destroys all actors; set RIDE_ALLOW_DESTRUCTIVE_RELOAD=1 on an empty dedicated server' >&2; exit 2;
  }
  EXTRA+=(--allow-destructive)
fi
python -m ride.carla.health --host "$CARLA_HOST" --port "$CARLA_RPC_PORT" \
  --timeout "${RIDE_CARLA_TIMEOUT:-120}" --mode "${RIDE_HEALTH_MODE:-inspect}" "${EXTRA[@]}"
