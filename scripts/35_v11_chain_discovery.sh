#!/usr/bin/env bash
# Small, explicitly exploratory 2x2 stress panel; NO selection of favorable cells.
# Each cell gets a fresh frozen manifest and a separate results directory.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_V11_GRID_ROOT:?Set new dedicated output root}"
: "${CARLA_TOWN:?Set CARLA_TOWN after operator-approved map preparation}"
export RIDE_WORLD_RESET="${RIDE_WORLD_RESET:-actors}"
export RIDE_CUTIN_LATE_SWITCH_S="${RIDE_CUTIN_LATE_SWITCH_S:-3.5}"
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M="${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-20}"
export RIDE_CUTIN_EGO_GAP_M=20
export RIDE_REAR_SPEED_DELTA_MPS=0
export RIDE_CRUISE_MPS="${RIDE_CRUISE_MPS:-7.5}"
export RIDE_CHAIN_C_SPEED_DELTA_MPS="${RIDE_CHAIN_C_SPEED_DELTA_MPS:-0.5}"
export RIDE_BEHAVIOR_SPEED_PROTOCOL=nominal_cap
export RIDE_REPEATS=1
export RIDE_SEED="${RIDE_SEED:-17}"
export RIDE_FAMILIES='tm behavior'
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late'
export RIDE_ANCHORS="${RIDE_ANCHORS:-4}"
export RIDE_SMOKE_REQUIRE_COMPLETE=1
export RIDE_SMOKE_REQUIRE_VALID=0
# Set grid BEFORE running, don't inspect outcomes and then change cell inclusion.
read -r -a _BC <<< "${RIDE_V11_BC_GRID:-20 25}"
read -r -a _REAR <<< "${RIDE_V11_REAR_GRID:-35 45}"
if [[ -e "$RIDE_V11_GRID_ROOT" ]]; then
  echo "Refuse an existing grid root: $RIDE_V11_GRID_ROOT" >&2; exit 3
fi
mkdir -p "$RIDE_V11_GRID_ROOT"
python - "$RIDE_V11_GRID_ROOT/pilot_protocol.json" "${_BC[*]}" "${_REAR[*]}" <<'PYCODE'
import json,sys,os
from pathlib import Path
p=Path(sys.argv[1]);document={
    'purpose':'EXPLORATORY_GENERATION_ONLY_NOT_CONFIRMATION',
    'archetype':'chain_cut_in','response_families':['tm','behavior'],
    'precommitted_bc_gap_grid_m':list(map(float,sys.argv[2].split())),
    'precommitted_rear_gap_grid_m':list(map(float,sys.argv[3].split())),
    'seed':int(os.environ['RIDE_SEED']),
    'anchors':int(os.environ['RIDE_ANCHORS']),
    'cruise_speed_mps':float(os.environ['RIDE_CRUISE_MPS']),
    'c_speed_delta_mps':float(os.environ['RIDE_CHAIN_C_SPEED_DELTA_MPS']),
    'map':os.environ['CARLA_TOWN'],
    'action_semantics':'lateral_switch_timing_only',
    'report_all_cells_including_invalid':True}
p.write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n')
PYCODE
for bc in "${_BC[@]}"; do
  for rear in "${_REAR[@]}"; do
    export RIDE_CHAIN_BC_GAP_M="$bc" RIDE_CUTIN_REAR_GAP_M="$rear"
    export RIDE_OUTPUT_ROOT="$RIDE_V11_GRID_ROOT/bc_${bc}_rear_${rear}"
    if [[ -e "$RIDE_OUTPUT_ROOT" ]]; then
      echo "Refuse to overwrite discovery cell $RIDE_OUTPUT_ROOT" >&2; exit 3
    fi
    bash "$RIDE_HOME/scripts/29_v9_carla_preflight.sh" "$CARLA_TOWN"
    bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" chain_cut_in
    bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
    python -m ride.carla.m0_gate --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
      --output "$RIDE_OUTPUT_ROOT/carla/chain_m0_gate.json" || true
    python -m ride.v11_chain_evidence --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
      --output "$RIDE_OUTPUT_ROOT/carla/v11_evidence.json"
  done
done
