#!/usr/bin/env bash
# Exploratory heterogeneous controller mix. It is NOT the official RQ1 test.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_OUTPUT_ROOT:?Set new output root}"
: "${CARLA_TOWN:?Explicit map required}"
export RIDE_WORLD_RESET="${RIDE_WORLD_RESET:-actors}"
export RIDE_FAMILIES='tm behavior mixed_b_tm_c_behavior mixed_b_behavior_c_tm'
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late'
export RIDE_SMOKE_REQUIRE_ALL=0
export RIDE_SMOKE_REQUIRE_COMPLETE=1
export RIDE_SMOKE_REQUIRE_VALID=0
bash "$RIDE_HOME/scripts/29_v9_carla_preflight.sh" "$CARLA_TOWN"
bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" chain_cut_in
bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
python -m ride.v11_chain_evidence --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
 --output "$RIDE_OUTPUT_ROOT/carla/v11_evidence.json"
