#!/usr/bin/env bash
# Conservative chain-cut-in confirmation: block if development did not pass M0.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_DEV_FROZEN:?set frozen/chain_cut_in_anchors.jsonl from chain dev}"
: "${RIDE_CONFIRM_ROOT:?set a NEW and unused confirm root}"
python - "$RIDE_DEV_FROZEN" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads(p.read_text().splitlines()[0])
assert m['archetype']=='chain_cut_in', 'wrong development archetype'
report=p.parents[1]/'carla'/'chain_m0_gate.json'
r=json.loads(report.read_text())
assert r.get('batch_completed') and r.get('valid_cross_family_task_matched_pairs',0)>0, 'STOP: failed dev chain M0'
print('development chain M0 pairs:',r['valid_cross_family_task_matched_pairs'])
PY
export RIDE_CRUISE_MPS="${RIDE_CRUISE_MPS:-8}"
export RIDE_REAR_SPEED_DELTA_MPS="${RIDE_REAR_SPEED_DELTA_MPS:-0}"
export RIDE_CHAIN_C_SPEED_DELTA_MPS="${RIDE_CHAIN_C_SPEED_DELTA_MPS:-0}"
export RIDE_CHAIN_BC_GAP_M="${RIDE_CHAIN_BC_GAP_M:-25}"
export RIDE_BEHAVIOR_SPEED_PROTOCOL="${RIDE_BEHAVIOR_SPEED_PROTOCOL:-nominal_cap}"
export RIDE_CUTIN_LATE_SWITCH_S="${RIDE_CUTIN_LATE_SWITCH_S:-3.5}"
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M="${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-20}"
export RIDE_CUTIN_REAR_GAP_M="${RIDE_CUTIN_REAR_GAP_M:-35}"
python -m ride.protocol_lock --frozen "$RIDE_DEV_FROZEN" \
 --cutin-late-switch-s "$RIDE_CUTIN_LATE_SWITCH_S" \
 --cutin-lateral-lookahead-m "$RIDE_CUTIN_LATERAL_LOOKAHEAD_M" \
 --cutin-rear-gap-m "$RIDE_CUTIN_REAR_GAP_M" \
 --cutin-ego-gap-m "${RIDE_CUTIN_EGO_GAP_M:-20}" \
 --rear-speed-delta-mps "$RIDE_REAR_SPEED_DELTA_MPS" \
 --cruise-mps "$RIDE_CRUISE_MPS" \
 --warmup-ticks "${RIDE_WARMUP_TICKS:-20}" \
 --chain-bc-gap-m "$RIDE_CHAIN_BC_GAP_M" \
 --chain-c-speed-delta-mps "$RIDE_CHAIN_C_SPEED_DELTA_MPS" \
 --behavior-speed-protocol "$RIDE_BEHAVIOR_SPEED_PROTOCOL"
bash "$RIDE_HOME/scripts/29_v9_carla_preflight.sh" "${CARLA_TOWN:-Town04}"
export RIDE_EXCLUDE_ANCHORS_FROM="$RIDE_DEV_FROZEN"
export RIDE_EXCLUDE_RADIUS_M="${RIDE_EXCLUDE_RADIUS_M:-90}"
export RIDE_REPEATS=1
export RIDE_FAMILIES='tm behavior'
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late'
export RIDE_SMOKE_REQUIRE_COMPLETE=1
export RIDE_SMOKE_REQUIRE_VALID=1
export RIDE_SMOKE_REQUIRE_ALL=0
for seed in ${RIDE_CONFIRM_SEEDS:-23 41 73}; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed=$seed" >&2;exit 2;}
  export RIDE_SEED="$seed"
  export RIDE_OUTPUT_ROOT="$RIDE_CONFIRM_ROOT/seed_${seed}"
  bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" chain_cut_in
  bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
  python -m ride.carla.m0_gate --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
      --output "$RIDE_OUTPUT_ROOT/carla/chain_m0_gate.json" || true
  python -m ride.v10_mechanism_report --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
      --output "$RIDE_OUTPUT_ROOT/carla/chain_v10_mechanism_report.json"
done
