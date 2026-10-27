#!/usr/bin/env bash
# Source from any working directory. RIDE_CARLA_ROOT is the explicit override.
# Inherited 0.9.15 values must not silently poison a 0.9.16 experiment.
_RIDE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RIDE_HOME="$_RIDE_ROOT"
_RIDE_DEFAULT_CARLA_ROOT=/data0/senzeyu2/dataset/CARLA_0.9.16
if [[ -n "${RIDE_CARLA_ROOT:-}" ]]; then
  export CARLA_ROOT="${RIDE_CARLA_ROOT%/}"
elif [[ -z "${CARLA_ROOT:-}" ]]; then
  export CARLA_ROOT="$_RIDE_DEFAULT_CARLA_ROOT"
elif [[ "${CARLA_ROOT}" == *CARLA_0.9.15* ]]; then
  printf 'RIDE WARNING: inherited CARLA_ROOT=%s is 0.9.15; switching to %s\n' \
    "$CARLA_ROOT" "$_RIDE_DEFAULT_CARLA_ROOT" >&2
  export CARLA_ROOT="$_RIDE_DEFAULT_CARLA_ROOT"
else
  export CARLA_ROOT="${CARLA_ROOT%/}"
fi
# Fail clearly for any other mismatched install rather than silently mixing APIs.
if [[ "$(basename "$CARLA_ROOT")" != 'CARLA_0.9.16' && \
      "${RIDE_ALLOW_CUSTOM_CARLA_DIR:-0}" != 1 ]]; then
  printf 'RIDE ERROR: expected CARLA 0.9.16 root (directory CARLA_0.9.16); got %s\n' "$CARLA_ROOT" >&2
  printf 'Use RIDE_CARLA_ROOT for a custom root and RIDE_ALLOW_CUSTOM_CARLA_DIR=1 if renamed.\n' >&2
  return 2 2>/dev/null || exit 2
fi
export DATASET_ROOT="${DATASET_ROOT:-/data0/senzeyu2/dataset}"
export RAW_WOMD_ROOT="${RAW_WOMD_ROOT:-$DATASET_ROOT/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
export RIDE_OUTPUT_ROOT="${RIDE_OUTPUT_ROOT:-$DATASET_ROOT/RIDE/evidence}"
export CARLA_HOST="${CARLA_HOST:-127.0.0.1}"
export CARLA_RPC_PORT="${CARLA_RPC_PORT:-2000}"
# Dedicated TM port prevents collisions with other default-8000 clients.
export CARLA_TM_PORT="${CARLA_TM_PORT:-8100}"

# Remove stale CARLA navigation paths from the inherited Python search path.
# Preserve unrelated packages and ensure RIDE and the matching agents win.
_RIDE_CLEAN_PYTHONPATH=''
_RIDE_OLD_IFS="$IFS"; IFS=':'
for _RIDE_ENTRY in ${PYTHONPATH:-}; do
  [[ -n "$_RIDE_ENTRY" ]] || continue
  case "$_RIDE_ENTRY" in
    *CARLA_0.9.15*|"$RIDE_HOME"|"$CARLA_ROOT/PythonAPI/carla")
      [[ "$_RIDE_ENTRY" != *CARLA_0.9.15* ]] || \
        printf 'RIDE WARNING: dropped legacy PYTHONPATH entry: %s\n' "$_RIDE_ENTRY" >&2
      ;;
    *) _RIDE_CLEAN_PYTHONPATH="${_RIDE_CLEAN_PYTHONPATH:+$_RIDE_CLEAN_PYTHONPATH:}$_RIDE_ENTRY" ;;
  esac
done
IFS="$_RIDE_OLD_IFS"
export PYTHONPATH="$RIDE_HOME${_RIDE_CLEAN_PYTHONPATH:+:$_RIDE_CLEAN_PYTHONPATH}"
if [[ -f "$CARLA_ROOT/PythonAPI/carla/agents/navigation/behavior_agent.py" ]]; then
  export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH"
fi
unset _RIDE_ROOT _RIDE_DEFAULT_CARLA_ROOT _RIDE_CLEAN_PYTHONPATH _RIDE_ENTRY _RIDE_OLD_IFS
