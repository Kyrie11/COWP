#!/usr/bin/env bash
# Small, PRE-REGISTERED parameter screen. No autoscaling from outcome signals.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
if [[ -z "${RIDE_CUTIN_LATE_SWITCH_S:-}" || -z "${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-}" ]]; then
  echo "FATAL: pre-register RIDE_CUTIN_LATE_SWITCH_S and RIDE_CUTIN_LATERAL_LOOKAHEAD_M explicitly" >&2
  exit 4
fi
base="${RIDE_FACTORIAL_OUTPUT_ROOT:-$DATASET_ROOT/RIDE/evidence/rq12_v6_factorial}"
read -r -a grid <<< "${RIDE_REAR_SPEED_GRID:-0 1.0}"
read -r -a gaps <<< "${RIDE_REAR_GAP_GRID:-35 45}"
export RIDE_CUTIN_EGO_GAP_M=20
for rear_gap in "${gaps[@]}"; do
for delta in "${grid[@]}"; do
  [[ "$delta" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Invalid rear speed delta: $delta" >&2; exit 2; }
  export RIDE_OUTPUT_ROOT="$base/rear_gap_${rear_gap}/rear_delta_${delta//./p}"
  export RIDE_CUTIN_REAR_GAP_M="$rear_gap"
  export RIDE_REAR_SPEED_DELTA_MPS="$delta"
  export RIDE_FAMILIES='tm behavior'
  # Cruise is a mechanistic NEGATIVE CONTROL; both merge arms are task matched.
  export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late'
  export RIDE_SMOKE_REQUIRE_ALL=0
  export RIDE_SMOKE_REQUIRE_MECHANISM=0
  export RIDE_SMOKE_REQUIRE_COMPLETE=1
  export RIDE_SMOKE_REQUIRE_VALID=1
  echo "========== rear_gap=$rear_gap delta=$delta output=$RIDE_OUTPUT_ROOT ========== " >&2
  bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" cut_in
  bash "$RIDE_HOME/scripts/03_run_carla_case.sh" cut_in
  bash "$RIDE_HOME/scripts/04_analyze_carla.sh"
  bash "$RIDE_HOME/scripts/18_reaudit_mechanism.sh" cut_in
  bash "$RIDE_HOME/scripts/12_research_status.sh"
done
done
