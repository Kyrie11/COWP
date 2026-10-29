#!/usr/bin/env bash
# NEW v12 fresh heldout confirmation only after an independently run clean dev M0.
# Do not select the best outcome cell from the four-cell grid as a confirmation dataset.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_DEV_FROZEN:?new v12 development frozen/chain_cut_in_anchors.jsonl required}"
: "${RIDE_CONFIRM_ROOT:?set a NEW confirm root}"
: "${CARLA_TOWN:?operator must explicitly prepare this CARLA map}"
[[ ! -e "$RIDE_CONFIRM_ROOT" ]] || { echo "CONFIRM_ROOT_EXISTS: $RIDE_CONFIRM_ROOT" >&2; exit 4; }
export RIDE_WORLD_RESET="${RIDE_WORLD_RESET:-actors}"
export RIDE_CRUISE_MPS="${RIDE_CRUISE_MPS:-7.5}"
export RIDE_CHAIN_BC_GAP_M="${RIDE_CHAIN_BC_GAP_M:-20}"
export RIDE_CUTIN_REAR_GAP_M="${RIDE_CUTIN_REAR_GAP_M:-35}"
export RIDE_CHAIN_C_SPEED_DELTA_MPS="${RIDE_CHAIN_C_SPEED_DELTA_MPS:-0.5}"
export RIDE_CUTIN_EGO_GAP_M=20
export RIDE_REAR_SPEED_DELTA_MPS=0
export RIDE_BEHAVIOR_ROUTE_PROTOCOL=frozen_corridor
export RIDE_BEHAVIOR_SPEED_PROTOCOL=nominal_cap
export RIDE_CUTIN_LATE_SWITCH_S="${RIDE_CUTIN_LATE_SWITCH_S:-3.5}"
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M="${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-20}"
export RIDE_CUTIN_PAIRED_AMPLITUDE_MPS="${RIDE_CUTIN_PAIRED_AMPLITUDE_MPS:-0.5}"
# FAIL CLOSED before any new CARLA trials: dev M0 must have no exceptions,
# >=2 valid cross-family pairs on >=2 distinct anchors. NO claim of RQ1/RQ2.
python - "$RIDE_DEV_FROZEN" <<'PY'
import sys,json
from pathlib import Path
f=Path(sys.argv[1]); manifest=json.loads(f.read_text().splitlines()[0])
if manifest.get('archetype')!='chain_cut_in' or manifest.get('anchor_protocol_version')!='ride_lane_corridor_v12_20261028':
    raise SystemExit('STOP: development manifest is not newly frozen chain v12')
if manifest.get('behavior_route_protocol')!='frozen_corridor':
    raise SystemExit('STOP: development route protocol was not v12 frozen corridor')
report=f.parents[1]/'carla'/'chain_m0_gate.json'
if not report.is_file():raise SystemExit('STOP: missing dev M0 report '+str(report))
r=json.loads(report.read_text())
if not r.get('batch_completed') or r.get('status')!='M0_CROSS_FAMILY_PAIRS_AVAILABLE_NOT_RQ1_OR_RQ2_PROOF':
    raise SystemExit('STOP: dev M0 invalid ('+str(r.get('status'))+'); never bypass this')
valid=[x for x in r.get('per_anchor_pairs',[]) if x.get('rq1_pair_eligible')]
if len(valid)<2 or len({x['anchor'] for x in valid})<2:
    raise SystemExit('STOP: dev matched pairs must come from >=2 anchors')
print('DEV_V12_M0_GATE_OK',len(valid),'pairs,',len({x['anchor'] for x in valid}),'anchors')
PY
python -m ride.protocol_lock --required-version ride_lane_corridor_v12_20261028 \
 --frozen "$RIDE_DEV_FROZEN" \
 --cutin-late-switch-s "$RIDE_CUTIN_LATE_SWITCH_S" \
 --cutin-lateral-lookahead-m "$RIDE_CUTIN_LATERAL_LOOKAHEAD_M" \
 --cutin-rear-gap-m "$RIDE_CUTIN_REAR_GAP_M" \
 --cutin-ego-gap-m "$RIDE_CUTIN_EGO_GAP_M" \
 --rear-speed-delta-mps "$RIDE_REAR_SPEED_DELTA_MPS" \
 --cruise-mps "$RIDE_CRUISE_MPS" \
 --warmup-ticks "${RIDE_WARMUP_TICKS:-20}" \
 --chain-bc-gap-m "$RIDE_CHAIN_BC_GAP_M" \
 --chain-c-speed-delta-mps "$RIDE_CHAIN_C_SPEED_DELTA_MPS" \
 --behavior-speed-protocol "$RIDE_BEHAVIOR_SPEED_PROTOCOL" \
 --behavior-route-protocol "$RIDE_BEHAVIOR_ROUTE_PROTOCOL" \
 --cutin-paired-amplitude-mps "$RIDE_CUTIN_PAIRED_AMPLITUDE_MPS"
bash "$RIDE_HOME/scripts/29_v9_carla_preflight.sh" "$CARLA_TOWN"
export RIDE_EXCLUDE_ANCHORS_FROM="$RIDE_DEV_FROZEN"
export RIDE_EXCLUDE_RADIUS_M="${RIDE_EXCLUDE_RADIUS_M:-90}"
export RIDE_REPEATS=1
export RIDE_FAMILIES='tm behavior'
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late speed_early speed_late'
export RIDE_SMOKE_REQUIRE_COMPLETE=1
export RIDE_SMOKE_REQUIRE_VALID=0
DEV_SEED="$(python - "$RIDE_DEV_FROZEN" <<'PY'
import json,sys
print(json.loads(open(sys.argv[1]).readline())['seed'])
PY
)"
for seed in ${RIDE_CONFIRM_SEEDS:-23 41 73}; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "INVALID_CONFIRM_SEED: $seed" >&2;exit 3; }
  [[ "$seed" != "$DEV_SEED" ]] || { echo 'Refuse development seed in confirmation' >&2;exit 3; }
done
mkdir -p "$RIDE_CONFIRM_ROOT"
printf '%s\n' "study=V12_CONFIRM; dev_frozen=$RIDE_DEV_FROZEN; seeds=${RIDE_CONFIRM_SEEDS:-23 41 73}; town=$CARLA_TOWN; all failures retained" \
 > "$RIDE_CONFIRM_ROOT/study_protocol.txt"
for seed in ${RIDE_CONFIRM_SEEDS:-23 41 73}; do
  export RIDE_SEED="$seed" RIDE_OUTPUT_ROOT="$RIDE_CONFIRM_ROOT/seed_${seed}"
  bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" chain_cut_in
  bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
  python -m ride.carla.m0_gate --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
    --output "$RIDE_OUTPUT_ROOT/carla/chain_m0_gate.json" || echo 'Confirm M0 incomplete; report retained.'
  python -m ride.v11_chain_evidence --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
    --output "$RIDE_OUTPUT_ROOT/carla/v11_frozen_primary_evidence.json"
  python -m ride.v12_sensitivity --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
    --output "$RIDE_OUTPUT_ROOT/carla/v12_exploratory_sensitivity.json"
  python -m ride.v12_speed_pair_audit --input "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
    --output "$RIDE_OUTPUT_ROOT/carla/v12_speed_pair_audit.json"
done
