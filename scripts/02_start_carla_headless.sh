#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
if [[ -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then exe="$CARLA_ROOT/CarlaUE4.sh"
elif [[ -x "$CARLA_ROOT/CarlaUnreal.sh" ]]; then exe="$CARLA_ROOT/CarlaUnreal.sh"
else echo "ERROR: No executable CarlaUE4.sh or CarlaUnreal.sh in $CARLA_ROOT" >&2; exit 3
fi
# Server start must never silently bind onto an existing user's simulator.
python -m ride.carla.ports --mode launch --host "$CARLA_HOST" \
  --rpc-port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT"
GPU_ARGS=()
if [[ -n "${CARLA_GPU_ADAPTER:-}" ]]; then
  if [[ ! "$CARLA_GPU_ADAPTER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: CARLA_GPU_ADAPTER must be a non-negative Vulkan graphics adapter index" >&2
    exit 2
  fi
  GPU_ARGS+=("-graphicsadapter=$CARLA_GPU_ADAPTER")
fi
printf 'RIDE CARLA: root=%s rpc=%s streaming=%s tm=%s graphics_adapter=%s\n' \
  "$CARLA_ROOT" "$CARLA_RPC_PORT" "$((CARLA_RPC_PORT+1))" \
  "$CARLA_TM_PORT" "${CARLA_GPU_ADAPTER:-auto}" >&2
# Offscreen is headless, but still initializes rendering/Vulkan. RIDE Python
# trials enable no_rendering_mode for their state-only physical rollouts.
cd "$CARLA_ROOT"
exec "$exe" -RenderOffScreen -quality-level=Low \
  "-carla-rpc-port=$CARLA_RPC_PORT" -nosound "${GPU_ARGS[@]}" "$@"
