#!/usr/bin/env bash
# SOURCE this from an experiment shell. Fail on accidental dev/confirm drift.
# New protocol is defined before observing any v8 outcome.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"
ride_lock_variable() {
  local name="$1" expected="$2" actual="${!name:-}"
  if [[ -n "$actual" && "$actual" != "$expected" ]]; then
    printf 'V8 PROTOCOL DRIFT: %s is %s (expected %s). Explicitly unset stale value and retry.\n' "$name" "$actual" "$expected" >&2
    return 3
  fi
  export "$name=$expected"
}
ride_lock_variable CARLA_TOWN Town04
ride_lock_variable RIDE_CUTIN_LATE_SWITCH_S 3.5
ride_lock_variable RIDE_CUTIN_LATERAL_LOOKAHEAD_M 20
ride_lock_variable RIDE_CUTIN_EGO_GAP_M 20
ride_lock_variable RIDE_ANCHOR_SELECTION_SEED "${RIDE_V8_ANCHOR_SEED_EXPECTED:-9901}"
ride_lock_variable RIDE_REAR_GAP_GRID 35
ride_lock_variable RIDE_REAR_SPEED_GRID 0
ride_lock_variable RIDE_FAMILIES 'tm behavior'
ride_lock_variable RIDE_PROFILES 'cruise early_slow late_slow mild_early mild_late'
ride_lock_variable RIDE_WARMUP_TICKS 20
ride_lock_variable RIDE_WORLD_RESET actors
ride_lock_variable RIDE_REPEATS 1
# Freeze the response-model comparison and task-matching definition.
export RIDE_V8_LOCKED_PROTOCOL=ride_lane_corridor_v8_20261028
unset -f ride_lock_variable
printf 'RIDE v8 protocol frozen: late=3.5s lookahead=20m gap=35m speed_delta=0 anchor_seed=%s\n' "$RIDE_ANCHOR_SELECTION_SEED" >&2
