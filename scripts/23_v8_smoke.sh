#!/usr/bin/env bash
# Physical M0 smoke: only then scale up. Results never overwrite previous panels.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_OUTPUT_ROOT:?choose a fresh RIDE_OUTPUT_ROOT}" 
export RIDE_CUTIN_LATE_SWITCH_S="${RIDE_CUTIN_LATE_SWITCH_S:-3.5}"
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M="${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-20}"
export RIDE_REAR_SPEED_DELTA_MPS="${RIDE_REAR_SPEED_DELTA_MPS:-0}"
export RIDE_CUTIN_REAR_GAP_M="${RIDE_CUTIN_REAR_GAP_M:-35}"
export RIDE_ANCHORS="${RIDE_ANCHORS:-2}"
export RIDE_REPEATS="${RIDE_REPEATS:-1}"
export RIDE_FAMILIES='tm behavior'
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late'
export RIDE_SMOKE_REQUIRE_COMPLETE=1
export RIDE_SMOKE_REQUIRE_VALID=1
export RIDE_SMOKE_REQUIRE_ALL=0
bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" cut_in
bash "$RIDE_HOME/scripts/03_run_carla_case.sh" cut_in
bash "$RIDE_HOME/scripts/04_analyze_carla.sh"
bash "$RIDE_HOME/scripts/18_reaudit_mechanism.sh" cut_in
bash "$RIDE_HOME/scripts/12_research_status.sh"
