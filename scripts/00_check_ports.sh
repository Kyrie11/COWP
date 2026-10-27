#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
python -m ride.carla.ports --mode status --host "$CARLA_HOST" \
  --rpc-port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT"
