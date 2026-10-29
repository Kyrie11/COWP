#!/usr/bin/env bash
# Strict same-geometry v13 two-stage evaluation. New output root each invocation.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_V14_ORIGINAL_FROZEN:?point to freeze/frozen/chain_cut_in_anchors.jsonl of an ORIGINAL v13 run}"
: "${RIDE_V14_ANCHORS:?space separated original anchor ids (e.g. '3' or '4 5')}"
: "${RIDE_V14_SEED:?explicit seed (reusing same seed and new seed are separate tests)}"
: "${RIDE_V14_ROOT:?NEW, never-used output root}"
[[ ! -e "$RIDE_V14_ROOT" ]] || { echo "REFUSE_EXISTING_ROOT $RIDE_V14_ROOT" >&2; exit 3; }
SOURCE_FAMILY="${RIDE_V14_SOURCE_FAMILY:-behavior}"
TARGET_FAMILY="${RIDE_V14_TARGET_FAMILY:-behavior}"
if [[ "$SOURCE_FAMILY" != "behavior" && "$SOURCE_FAMILY" != "tm" ]] || \
   [[ "$TARGET_FAMILY" != "behavior" && "$TARGET_FAMILY" != "tm" ]]; then
  echo 'v13 diagnostic only supports TM/BehaviorAgent families' >&2; exit 3
fi
PLACEBO_ARG=()
if [[ "$SOURCE_FAMILY" == "$TARGET_FAMILY" ]]; then
  if [[ "${RIDE_V14_PLACEBO:-0}" != 1 ]]; then
    echo 'Same-family replay must set RIDE_V14_PLACEBO=1' >&2;exit 3
  fi
  PLACEBO_ARG+=(--placebo)
elif [[ "${RIDE_V14_PLACEBO:-0}" == 1 ]]; then
  echo 'A placebo cannot be cross-family' >&2;exit 3
fi
mkdir -p "$RIDE_V14_ROOT"/{freeze/frozen,source/frozen,target/frozen,locks}
FROZEN="$RIDE_V14_ROOT/freeze/frozen/chain_cut_in_anchors.jsonl"
read -r -a ANCHOR_LIST <<< "$RIDE_V14_ANCHORS"
python -m ride.v14_frozen_replay derive \
 --input "$RIDE_V14_ORIGINAL_FROZEN" --output "$FROZEN" \
 --anchors "${ANCHOR_LIST[@]}" --seed "$RIDE_V14_SEED"
# All exports are shell-quoted by Python; current frozen geometry determines
# exact speed/merge/time/route parameters instead of silent fallbacks.
eval "$(python -m ride.v14_frozen_replay emit-env --input "$FROZEN")"
export RIDE_WORLD_RESET="${RIDE_WORLD_RESET:-actors}"
export RIDE_SMOKE_REQUIRE_VALID=0
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late speed_early speed_late'
cp "$FROZEN" "$RIDE_V14_ROOT/source/frozen/chain_cut_in_anchors.jsonl"
cp "$FROZEN" "$RIDE_V14_ROOT/target/frozen/chain_cut_in_anchors.jsonl"
bash "$RIDE_HOME/scripts/29_v9_carla_preflight.sh" "$CARLA_TOWN"
export RIDE_OUTPUT_ROOT="$RIDE_V14_ROOT/source"
export RIDE_FAMILIES="$SOURCE_FAMILY"
bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
SOURCE="$RIDE_V14_ROOT/source/carla/chain_cut_in.jsonl"
TARGET="$RIDE_V14_ROOT/target/carla/chain_cut_in.jsonl"
CHOICES="$RIDE_V14_ROOT/locks/source_model_choices.json"
COMMIT="$RIDE_V14_ROOT/locks/source_model_choices.commit.json"
python -m ride.v13_model_transfer lock --source "$SOURCE" --frozen "$FROZEN" \
  --target "$TARGET" --choices "$CHOICES" --commit "$COMMIT" \
  --source-family "$SOURCE_FAMILY" --target-family "$TARGET_FAMILY" \
  --tie-resolution 0.05 --meaningful-delta 0.25 --comfort-limit 2.0 --min-decisions 1 \
  "${PLACEBO_ARG[@]}"
export RIDE_OUTPUT_ROOT="$RIDE_V14_ROOT/target"
export RIDE_FAMILIES="$TARGET_FAMILY"
bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
python -m ride.v13_model_transfer evaluate --source "$SOURCE" --frozen "$FROZEN" \
  --target "$TARGET" --choices "$CHOICES" --commit "$COMMIT" \
  --output "$RIDE_V14_ROOT/v13_target_decision_audit.json"
python -m ride.v14_counterfactual_audit --roots "$RIDE_V14_ROOT" \
  --output "$RIDE_V14_ROOT/v14_readonly_physics_audit.json"
echo "FROZEN_REPLAY_COMPLETE $RIDE_V14_ROOT"
