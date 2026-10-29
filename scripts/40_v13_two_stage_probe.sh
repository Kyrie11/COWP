#!/usr/bin/env bash
# RIDE v13 two-stage pilot. Source model completes hypothetical rollouts first;
# its choices are hashed BEFORE target-family outcomes are permitted to exist.
# This is a CARLA model-transfer DIAGNOSTIC, not an official planner comparison.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_V13_ROOT:?set a NEW directory, never use previous evidence root}"
: "${CARLA_TOWN:?prepare an empty map explicitly}"
[[ ! -e "$RIDE_V13_ROOT" ]] || { echo "REFUSE_EXISTING_ROOT: $RIDE_V13_ROOT" >&2; exit 3; }
export RIDE_V13_SOURCE_FAMILY="${RIDE_V13_SOURCE_FAMILY:-behavior}"
export RIDE_V13_TARGET_FAMILY="${RIDE_V13_TARGET_FAMILY:-tm}"
if [[ "$RIDE_V13_SOURCE_FAMILY" == "$RIDE_V13_TARGET_FAMILY" ]]; then
  if [[ "${RIDE_V13_PLACEBO:-0}" != 1 ]]; then
    echo 'Same-family test requires RIDE_V13_PLACEBO=1 and is only a repeatability control' >&2; exit 3
  fi
else
  if [[ "${RIDE_V13_PLACEBO:-0}" == 1 ]]; then
    echo 'A placebo must use the SAME source and target family' >&2; exit 3
  fi
fi
case "$RIDE_V13_SOURCE_FAMILY:$RIDE_V13_TARGET_FAMILY" in
 behavior:tm|tm:behavior|behavior:behavior|tm:tm) ;; *) echo 'Only tm/behavior supported' >&2; exit 3;;
esac
# Every parameter below is a preregistered pilot protocol. Keep unchanged
# throughout source and target. Confirm on new corridors, not this dev panel.
export RIDE_CRUISE_MPS="${RIDE_CRUISE_MPS:-7.5}"
export RIDE_CUTIN_EGO_GAP_M="${RIDE_CUTIN_EGO_GAP_M:-20}"
export RIDE_CUTIN_REAR_GAP_M="${RIDE_CUTIN_REAR_GAP_M:-35}"
export RIDE_CHAIN_BC_GAP_M="${RIDE_CHAIN_BC_GAP_M:-20}"
export RIDE_CHAIN_C_SPEED_DELTA_MPS="${RIDE_CHAIN_C_SPEED_DELTA_MPS:-0.5}"
export RIDE_REAR_SPEED_DELTA_MPS="${RIDE_REAR_SPEED_DELTA_MPS:-0}"
export RIDE_WARMUP_TICKS="${RIDE_WARMUP_TICKS:-20}"
export RIDE_CUTIN_LATE_SWITCH_S="${RIDE_CUTIN_LATE_SWITCH_S:-3.5}"
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M="${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-20}"
export RIDE_CUTIN_PAIRED_AMPLITUDE_MPS="${RIDE_CUTIN_PAIRED_AMPLITUDE_MPS:-0.5}"
export RIDE_BEHAVIOR_ROUTE_PROTOCOL=frozen_corridor
export RIDE_BEHAVIOR_SPEED_PROTOCOL=nominal_cap
export RIDE_WORLD_RESET="${RIDE_WORLD_RESET:-actors}"
export RIDE_SMOKE_REQUIRE_VALID=0
export RIDE_REPEATS=1
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late speed_early speed_late'

bash "$RIDE_HOME/scripts/29_v9_carla_preflight.sh" "$CARLA_TOWN"
mkdir -p "$RIDE_V13_ROOT"/{freeze,source,target,locks}
export RIDE_OUTPUT_ROOT="$RIDE_V13_ROOT/freeze"
# Fail if the map cannot produce enough distinct anchors, rather than
# silently dropping corridors or changing the chosen risk threshold.
bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" chain_cut_in
FROZEN="$RIDE_V13_ROOT/freeze/frozen/chain_cut_in_anchors.jsonl"
for side in source target; do
  mkdir -p "$RIDE_V13_ROOT/$side/frozen"
  cp "$FROZEN" "$RIDE_V13_ROOT/$side/frozen/chain_cut_in_anchors.jsonl"
done
export RIDE_OUTPUT_ROOT="$RIDE_V13_ROOT/source"
export RIDE_FAMILIES="$RIDE_V13_SOURCE_FAMILY"
bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
SOURCE="$RIDE_V13_ROOT/source/carla/chain_cut_in.jsonl"
TARGET="$RIDE_V13_ROOT/target/carla/chain_cut_in.jsonl"
CHOICES="$RIDE_V13_ROOT/locks/source_model_choices.json"
COMMIT="$RIDE_V13_ROOT/locks/source_model_choices.commit.json"
# Absolutely no target execution before this commit has succeeded.
PLACEBO_ARG=()
if [[ "${RIDE_V13_PLACEBO:-0}" == 1 ]]; then PLACEBO_ARG+=(--placebo); fi
python -m ride.v13_model_transfer lock \
 --source "$SOURCE" --frozen "$FROZEN" --target "$TARGET" \
 --choices "$CHOICES" --commit "$COMMIT" \
 --source-family "$RIDE_V13_SOURCE_FAMILY" --target-family "$RIDE_V13_TARGET_FAMILY" \
 --tie-resolution 0.05 --meaningful-delta 0.25 --comfort-limit 2.0 --min-decisions 1 \
 "${PLACEBO_ARG[@]}"
export RIDE_OUTPUT_ROOT="$RIDE_V13_ROOT/target"
export RIDE_FAMILIES="$RIDE_V13_TARGET_FAMILY"
bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
python -m ride.v13_model_transfer evaluate \
 --source "$SOURCE" --frozen "$FROZEN" --target "$TARGET" \
 --choices "$CHOICES" --commit "$COMMIT" \
 --output "$RIDE_V13_ROOT/v13_target_decision_audit.json"
echo "V13_TWO_STAGE_COMPLETE: $RIDE_V13_ROOT/v13_target_decision_audit.json"
