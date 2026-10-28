#!/usr/bin/env bash
# Confirm previously frozen controls; exclude dev corridors by map coordinates.
# Requires env RIDE_DEV_FROZEN=<dev>/frozen/cut_in_anchors.jsonl.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_DEV_FROZEN:?set RIDE_DEV_FROZEN to development frozen/cut_in_anchors.jsonl}"
test -s "$RIDE_DEV_FROZEN" || { echo "Missing development freeze $RIDE_DEV_FROZEN" >&2;exit 3; }
: "${RIDE_CONFIRM_ROOT:?set a NEW RIDE_CONFIRM_ROOT}"
# Confirmation is prohibited when development M0 is incomplete or blocked.
# Earlier v8 allowed confirm directly after a failed smoke, which tested an
# uncalibrated TM and triggered a risky Town05 map switch without science gain.
RIDE_DEV_M0_REPORT="${RIDE_DEV_M0_REPORT:-$(dirname "$(dirname "$RIDE_DEV_FROZEN")")/carla/m0_physics_gate.json}"
python - "$RIDE_DEV_M0_REPORT" <<'PY_GATE'
import json,sys
from pathlib import Path
f=Path(sys.argv[1])
if not f.is_file():
    raise SystemExit('STOP_CONFIRM: missing completed development M0 gate: '+str(f))
r=json.loads(f.read_text())
if not r.get('batch_completed') or r.get('status')!='M0_CROSS_FAMILY_PAIRS_AVAILABLE_NOT_RQ1_OR_RQ2_PROOF':
    raise SystemExit('STOP_CONFIRM: development M0 did not produce eligible cross-family pairs: '+str(r.get('status')))
print('RIDE_DEV_M0_GATE_ACCEPTED',f)
PY_GATE
export RIDE_CRUISE_MPS="${RIDE_CRUISE_MPS:-8}"
export RIDE_CUTIN_LATE_SWITCH_S="${RIDE_CUTIN_LATE_SWITCH_S:-3.5}"
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M="${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-20}"
export RIDE_CUTIN_REAR_GAP_M="${RIDE_CUTIN_REAR_GAP_M:-35}"
export RIDE_REAR_SPEED_DELTA_MPS="${RIDE_REAR_SPEED_DELTA_MPS:-0}"
# Control parameters must exactly match the development freeze; defaulting from
# inherited environment silently created the v7 3.5/20 versus 4.5/15 drift.
python -m ride.protocol_lock --frozen "$RIDE_DEV_FROZEN" \
  --cutin-late-switch-s "$RIDE_CUTIN_LATE_SWITCH_S" \
  --cutin-lateral-lookahead-m "$RIDE_CUTIN_LATERAL_LOOKAHEAD_M" \
  --cutin-rear-gap-m "$RIDE_CUTIN_REAR_GAP_M" \
  --cutin-ego-gap-m "${RIDE_CUTIN_EGO_GAP_M:-20}" \
  --rear-speed-delta-mps "$RIDE_REAR_SPEED_DELTA_MPS" \
  --cruise-mps "${RIDE_CRUISE_MPS:-8}" \
  --warmup-ticks "${RIDE_WARMUP_TICKS:-20}" \
  --chain-bc-gap-m "${RIDE_CHAIN_BC_GAP_M:-25}" \
  --chain-c-speed-delta-mps "${RIDE_CHAIN_C_SPEED_DELTA_MPS:-0}" \
  --behavior-speed-protocol "${RIDE_BEHAVIOR_SPEED_PROTOCOL:-nominal_cap}"
# Critical: never silently switch Unreal's loaded map in a batch. Verify here,
# BEFORE creating any seed/frozen directory. Use a dedicated Town05 server.
bash "$RIDE_HOME/scripts/29_v9_carla_preflight.sh" "${CARLA_TOWN:-Town04}"
export RIDE_EXCLUDE_ANCHORS_FROM="$RIDE_DEV_FROZEN"
export RIDE_EXCLUDE_RADIUS_M="${RIDE_EXCLUDE_RADIUS_M:-90}"
export RIDE_REPEATS=1
export RIDE_FAMILIES='tm behavior'
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late'
export RIDE_SMOKE_REQUIRE_VALID=1
export RIDE_SMOKE_REQUIRE_COMPLETE=1
export RIDE_SMOKE_REQUIRE_ALL=0
for seed in ${RIDE_CONFIRM_SEEDS:-23 41 73}; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2;exit 2; }
  export RIDE_SEED="$seed"
  export RIDE_OUTPUT_ROOT="$RIDE_CONFIRM_ROOT/seed_${seed}"
  echo "RIDE HOLDOUT: seed=$seed map=${CARLA_TOWN:-Town04} excludes=$RIDE_DEV_FROZEN"
  bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" cut_in
  bash "$RIDE_HOME/scripts/03_run_carla_case.sh" cut_in
  bash "$RIDE_HOME/scripts/04_analyze_carla.sh"
  bash "$RIDE_HOME/scripts/18_reaudit_mechanism.sh" cut_in
done
