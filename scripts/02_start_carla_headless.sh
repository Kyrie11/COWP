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
# UE4.26/CARLA has documented crashes when switching to Town05 after a
# -quality-level=Low launch. Default is Epic. To experiment, set
# RIDE_CARLA_QUALITY=Low explicitly in a separate named diagnostic session.
QUALITY="${RIDE_CARLA_QUALITY:-Epic}"
case "$QUALITY" in
  Epic|Low) ;;  *) echo "Unsupported RIDE_CARLA_QUALITY=$QUALITY" >&2; exit 2;;
esac
# A map may be provided as a CARLA UE4 startup asset. Its availability must
# be verified with python -m ride.carla.health before any batch experiment.
STARTUP_ARGS=()
if [[ -n "${RIDE_CARLA_START_MAP:-}" ]]; then
  case "$RIDE_CARLA_START_MAP" in
    Town[0-9][0-9]|Town[0-9][0-9]_Opt)
      STARTUP_ARGS+=("/Game/Carla/Maps/$RIDE_CARLA_START_MAP") ;;
    *) echo "Invalid RIDE_CARLA_START_MAP=$RIDE_CARLA_START_MAP" >&2;exit 2;;
  esac
fi
printf 'RIDE_CARLA_LAUNCH_CONFIG quality=%s startup_map=%s\n' "$QUALITY" "${RIDE_CARLA_START_MAP:-default}" >&2
# In some packaged 0.9.16 builds the UE4 positional map asset is ignored.
# Never claim that CARLA_TOWN or RIDE_CARLA_START_MAP implies the world is loaded.
echo 'RIDE_CARLA_MAP_UNVERIFIED: after startup check the actual server map with scripts/29_v9_carla_preflight.sh; if it differs, use the explicit one-time 02c_prepare_carla_town.sh on your own idle server.' >&2
cd "$CARLA_ROOT"
exec "$exe" "${STARTUP_ARGS[@]}" -RenderOffScreen "-quality-level=$QUALITY" \
  "-carla-rpc-port=$CARLA_RPC_PORT" -nosound "${GPU_ARGS[@]}" "$@"
