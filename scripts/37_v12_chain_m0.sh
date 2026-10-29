#!/usr/bin/env bash
# New protocol! Full freeze and M0; no fallback to v11 BehaviorAgent routes.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_OUTPUT_ROOT:?set a NEW path, never overwrite v11 evidence}"
: "${CARLA_TOWN:?prepare map explicitly before this script}"
export RIDE_WORLD_RESET="${RIDE_WORLD_RESET:-actors}"
export RIDE_CRUISE_MPS="${RIDE_CRUISE_MPS:-7.5}"
export RIDE_CHAIN_BC_GAP_M="${RIDE_CHAIN_BC_GAP_M:-20}"
export RIDE_CUTIN_REAR_GAP_M="${RIDE_CUTIN_REAR_GAP_M:-35}"
export RIDE_CHAIN_C_SPEED_DELTA_MPS="${RIDE_CHAIN_C_SPEED_DELTA_MPS:-0.5}"
export RIDE_BEHAVIOR_SPEED_PROTOCOL=nominal_cap
export RIDE_BEHAVIOR_ROUTE_PROTOCOL=frozen_corridor
export RIDE_CUTIN_PAIRED_AMPLITUDE_MPS="${RIDE_CUTIN_PAIRED_AMPLITUDE_MPS:-0.5}"
export RIDE_CUTIN_LATE_SWITCH_S="${RIDE_CUTIN_LATE_SWITCH_S:-3.5}"
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M="${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-20}"
export RIDE_CUTIN_EGO_GAP_M=20
export RIDE_REAR_SPEED_DELTA_MPS=0
export RIDE_FAMILIES='tm behavior'
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late speed_early speed_late'
export RIDE_SMOKE_REQUIRE_COMPLETE=1
export RIDE_SMOKE_REQUIRE_VALID=0
bash "$RIDE_HOME/scripts/29_v9_carla_preflight.sh" "$CARLA_TOWN"
if [[ -e "$RIDE_OUTPUT_ROOT" ]]; then
  echo "REFUSE_EXISTING_OUTPUT_ROOT: $RIDE_OUTPUT_ROOT" >&2;exit 3
fi
bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" chain_cut_in
# Do not suppress failures or exceptions: partial data is useful for diagnosis.
bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
python -m ride.carla.m0_gate --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
 --output "$RIDE_OUTPUT_ROOT/carla/chain_m0_gate.json" || echo 'M0 gate failed; see report (not an RQ negative).'
python -m ride.v11_chain_evidence --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
 --output "$RIDE_OUTPUT_ROOT/carla/v11_frozen_primary_evidence.json"
python -m ride.v12_sensitivity --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
 --output "$RIDE_OUTPUT_ROOT/carla/v12_exploratory_sensitivity.json"
python -m ride.v12_speed_pair_audit --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
 --output "$RIDE_OUTPUT_ROOT/carla/v12_speed_pair_audit.json"
