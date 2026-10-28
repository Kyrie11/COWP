#!/usr/bin/env bash
# Isolated four-car B-C third-party chain / physics-validation panel.
# Never modifies Unreal's map; use operator-approved one-time 02c map setup.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_OUTPUT_ROOT:?Choose NEW output root for chain scenario}"
export RIDE_CRUISE_MPS="${RIDE_CRUISE_MPS:-8}"
export RIDE_CHAIN_BC_GAP_M="${RIDE_CHAIN_BC_GAP_M:-25}"
export RIDE_CHAIN_C_SPEED_DELTA_MPS="${RIDE_CHAIN_C_SPEED_DELTA_MPS:-0}"
export RIDE_BEHAVIOR_SPEED_PROTOCOL="${RIDE_BEHAVIOR_SPEED_PROTOCOL:-nominal_cap}"
export RIDE_CUTIN_LATE_SWITCH_S="${RIDE_CUTIN_LATE_SWITCH_S:-3.5}"
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M="${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-20}"
export RIDE_CUTIN_REAR_GAP_M="${RIDE_CUTIN_REAR_GAP_M:-35}"
export RIDE_REAR_SPEED_DELTA_MPS="${RIDE_REAR_SPEED_DELTA_MPS:-0}"
export RIDE_FAMILIES='tm behavior'
# Existing action labels are historical; all cut-in variants ONLY vary merge timing.
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late'
export RIDE_SMOKE_REQUIRE_COMPLETE=1
export RIDE_SMOKE_REQUIRE_ALL=0
export RIDE_SMOKE_REQUIRE_VALID=1
bash "$RIDE_HOME/scripts/29_v9_carla_preflight.sh" "${CARLA_TOWN:-Town04}"
bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" chain_cut_in
set +e
bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
RC=$?
set -e
FILE="$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl"
if [[ -s "$FILE" ]]; then
  python -m ride.carla.m0_gate --input "$FILE" --output "$RIDE_OUTPUT_ROOT/carla/chain_m0_gate.json" || true
  python -m ride.v10_mechanism_report --input "$FILE" --output "$RIDE_OUTPUT_ROOT/carla/chain_v10_mechanism_report.json" || true
fi
if (( RC != 0 )); then
  echo "RIDE_CHAIN_M0_INCOMPLETE: sim_rc=$RC, review stored raw results; NEVER treat as absence of risk" >&2
  exit "$RC"
fi
bash "$RIDE_HOME/scripts/04_analyze_carla.sh"
bash "$RIDE_HOME/scripts/18_reaudit_mechanism.sh" chain_cut_in
