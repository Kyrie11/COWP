#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
if [[ -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then exe="$CARLA_ROOT/CarlaUE4.sh"
elif [[ -x "$CARLA_ROOT/CarlaUnreal.sh" ]]; then exe="$CARLA_ROOT/CarlaUnreal.sh"
else echo "ERROR: No executable CarlaUE4.sh or CarlaUnreal.sh in $CARLA_ROOT" >&2; exit 3
fi
# Headless off-screen != no-rendering; no rendering enabled in Python client
# for physics-only tests. Offscreen still supports GPU camera data if needed.
cd "$CARLA_ROOT"
exec "$exe" -RenderOffScreen -quality-level=Low -carla-rpc-port=2000 -nosound "$@"
