#!/usr/bin/env bash
# Prospective source-model choice lock BEFORE delayed-C target CARLA physics.
# Controlled comparison: A/B TM, C BehaviorAgent vs same with C-only brake delay.
# Not a learned-planner baseline and not a human-calibrated driver.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_V14_LOCKED_STRESS_ROOT:?New output root, never used}"
: "${RIDE_CHAIN_C_CONTROL_DELAY_S:?Explicit C brake delay, 0 for identity check}"
: "${RIDE_CHAIN_C_BRAKE_COMMAND_CAP:?Explicit C brake command cap, e.g. 0.85}"
: "${CARLA_TOWN:?CARLA_TOWN=Town04 or another verified town}"
[[ ! -e "$RIDE_V14_LOCKED_STRESS_ROOT" ]] || { echo "Refuse existing output root" >&2; exit 3; }
export RIDE_REPEATS=1
export RIDE_SMOKE_REQUIRE_VALID=0
export RIDE_CRUISE_MPS="${RIDE_CRUISE_MPS:-7.5}"
export RIDE_CUTIN_EGO_GAP_M="${RIDE_CUTIN_EGO_GAP_M:-20}"
export RIDE_CUTIN_REAR_GAP_M="${RIDE_CUTIN_REAR_GAP_M:-35}"
export RIDE_CHAIN_BC_GAP_M="${RIDE_CHAIN_BC_GAP_M:-20}"
export RIDE_CHAIN_C_SPEED_DELTA_MPS="${RIDE_CHAIN_C_SPEED_DELTA_MPS:-0.5}"
export RIDE_CUTIN_LATE_SWITCH_S="${RIDE_CUTIN_LATE_SWITCH_S:-3.5}"
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M="${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-20}"
export RIDE_CUTIN_PAIRED_AMPLITUDE_MPS="${RIDE_CUTIN_PAIRED_AMPLITUDE_MPS:-0.5}"
export RIDE_BEHAVIOR_ROUTE_PROTOCOL=frozen_corridor
export RIDE_BEHAVIOR_SPEED_PROTOCOL=nominal_cap
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late speed_early speed_late'
export RIDE_WORLD_RESET="${RIDE_WORLD_RESET:-actors}"
python -m ride.v15_protocol_check --mode locked
bash "$RIDE_HOME/scripts/29_v9_carla_preflight.sh" "$CARLA_TOWN"
mkdir -p "$RIDE_V14_LOCKED_STRESS_ROOT"/{freeze,source/frozen,target/frozen,locks}
export RIDE_OUTPUT_ROOT="$RIDE_V14_LOCKED_STRESS_ROOT/freeze"
export RIDE_FAMILIES='mixed_b_tm_c_behavior mixed_b_tm_c_delayed_behavior'
bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" chain_cut_in
FROZEN="$RIDE_OUTPUT_ROOT/frozen/chain_cut_in_anchors.jsonl"
cp "$FROZEN" "$RIDE_V14_LOCKED_STRESS_ROOT/source/frozen/chain_cut_in_anchors.jsonl"
cp "$FROZEN" "$RIDE_V14_LOCKED_STRESS_ROOT/target/frozen/chain_cut_in_anchors.jsonl"
SOURCE_FAMILY=mixed_b_tm_c_behavior
TARGET_FAMILY=mixed_b_tm_c_delayed_behavior
export RIDE_OUTPUT_ROOT="$RIDE_V14_LOCKED_STRESS_ROOT/source"
export RIDE_FAMILIES="$SOURCE_FAMILY"
export RIDE_V14_SOURCE_INACTIVE_STRESS_METADATA=1
bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
unset RIDE_V14_SOURCE_INACTIVE_STRESS_METADATA
SOURCE="$RIDE_V14_LOCKED_STRESS_ROOT/source/carla/chain_cut_in.jsonl"
TARGET="$RIDE_V14_LOCKED_STRESS_ROOT/target/carla/chain_cut_in.jsonl"
CHOICES="$RIDE_V14_LOCKED_STRESS_ROOT/locks/source_model_choices.json"
COMMIT="$RIDE_V14_LOCKED_STRESS_ROOT/locks/source_model_choices.commit.json"
python -m ride.v13_model_transfer lock --source "$SOURCE" --frozen "$FROZEN" \
 --target "$TARGET" --choices "$CHOICES" --commit "$COMMIT" \
 --source-family "$SOURCE_FAMILY" --target-family "$TARGET_FAMILY" \
 --tie-resolution 0.05 --meaningful-delta 0.25 --comfort-limit 2.0 --min-decisions 1
# Precommit transparent courtesy selectors BEFORE the target branch exists.
# Not a claim to replicate original RACP/HYPE/GameFormer planner.
python -m ride.v16_transparent_selectors lock \
 --source "$SOURCE" --frozen "$FROZEN" --target "$TARGET" \
 --choices "$RIDE_V14_LOCKED_STRESS_ROOT/locks/v16_transparent_choices.json" \
 --source-family "$SOURCE_FAMILY" --target-family "$TARGET_FAMILY" \
 --budget "${RIDE_V16_COURTESY_BUDGET:-0.4}" --tie 0.05
# At this point the target output does not exist and the SHA lock has committed.
export RIDE_OUTPUT_ROOT="$RIDE_V14_LOCKED_STRESS_ROOT/target"
export RIDE_FAMILIES="$TARGET_FAMILY"
bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
# A source-target choice difference is not scientific evidence if the target
# brake intervention never actuates. Require raw C request/applied validation.
python -m ride.v16_cross_stress actuation-gate \
 --panel "$TARGET" \
 --output "$RIDE_V14_LOCKED_STRESS_ROOT/v16_target_C_actuation_gate.json"
python -m ride.v13_model_transfer evaluate --source "$SOURCE" --frozen "$FROZEN" \
 --target "$TARGET" --choices "$CHOICES" --commit "$COMMIT" \
 --output "$RIDE_V14_LOCKED_STRESS_ROOT/v13_target_decision_audit.json"
python -m ride.v16_transparent_selectors evaluate \
 --source "$SOURCE" --frozen "$FROZEN" --target "$TARGET" \
 --choices "$RIDE_V14_LOCKED_STRESS_ROOT/locks/v16_transparent_choices.json" \
 --source-family "$SOURCE_FAMILY" --target-family "$TARGET_FAMILY" \
 --output "$RIDE_V14_LOCKED_STRESS_ROOT/v16_transparent_selector_audit.json"
python -m ride.v14_counterfactual_audit --roots "$RIDE_V14_LOCKED_STRESS_ROOT" \
 --output "$RIDE_V14_LOCKED_STRESS_ROOT/v14_readonly_physics_audit.json"
MERGED="$RIDE_V14_LOCKED_STRESS_ROOT/v14_merged_stress_panel.jsonl"
python -m ride.v14_merge_carla_panels --source "$SOURCE" --target "$TARGET" --output "$MERGED"
python -m ride.v14_safety_stress_audit --panel "$MERGED" \
 --output "$RIDE_V14_LOCKED_STRESS_ROOT/v14_physical_safety_audit.json"
python -m ride.v14_safety_envelope --panel "$MERGED" \
 --output "$RIDE_V14_LOCKED_STRESS_ROOT/v14_hypothetical_margin.json" \
 --reaction-s 0.6 --c-brake-mps2 3.5 --b-brake-mps2 5
printf 'PROSPECTIVE_DELAYED_C_STRESS_COMPLETE %s\n' "$RIDE_V14_LOCKED_STRESS_ROOT"
