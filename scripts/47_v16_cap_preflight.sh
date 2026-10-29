#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_V16_CAP_REFERENCE_PANEL:?Set earlier completed normal C reference CARLA panel or ZIP}"
: "${RIDE_CHAIN_C_BRAKE_COMMAND_CAP:?Set proposed command cap [0.15,1]}"
python -m ride.v16_cross_stress cap-preflight \
  --panel "$RIDE_V16_CAP_REFERENCE_PANEL" \
  --cap "$RIDE_CHAIN_C_BRAKE_COMMAND_CAP"
