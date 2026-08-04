#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026}"
MECH="$OUT_ROOT/eval/learned_offline/mechanism_verification.json"
"$PYTHON_BIN" - "$MECH" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert bool(x.get('pass',False)), f'mechanism gate failed: {sys.argv[1]}'
assert bool(x.get('calibration_feasible',False)), f'calibration is infeasible: {sys.argv[1]}'
assert x.get('gate_role') == 'development_continuation_not_paper_claim'
print('v16.8.1 mechanism and calibration gates passed; launching Waymax probe')
PY
TRANSFER_MANIFEST="$OUT_ROOT/configs/natural_attribution_transfer_manifest.json"
[[ -s "$TRANSFER_MANIFEST" ]] || { echo "missing transfer manifest: $TRANSFER_MANIFEST" >&2; exit 2; }
mapfile -t NATURAL_TRANSFER < <("$PYTHON_BIN" - "$TRANSFER_MANIFEST" <<'PY'
import json,sys
from pathlib import Path
x=json.load(open(sys.argv[1],encoding='utf-8'))
ckpt=Path(x['natural_checkpoint'])
hist=Path(x.get('natural_history') or (ckpt.parent/'history_natural.json'))
assert ckpt.is_file(), f'missing transferred natural checkpoint: {ckpt}'
assert hist.is_file(), f'missing transferred natural history: {hist}'
print(ckpt)
print(hist)
PY
)
export NATURAL_CKPT="${NATURAL_TRANSFER[0]}"
export NATURAL_HISTORY="${NATURAL_TRANSFER[1]}"
export TRANSPORT_CKPT="${TRANSPORT_CKPT:-$OUT_ROOT/checkpoints/transport/cowp_witness_best.pt}"
export CKPT="${CKPT:-$OUT_ROOT/checkpoints/planner/cowp_planner_best.pt}"
[[ -s "$TRANSPORT_CKPT" ]] || { echo "missing transport checkpoint: $TRANSPORT_CKPT" >&2; exit 2; }
[[ -s "$CKPT" ]] || { echo "missing planner checkpoint: $CKPT" >&2; exit 2; }
export RUN_DIAGNOSE=0
export RUN_NATURAL=0
export RUN_TRANSPORT=0
export RUN_PLANNER=0
export RUN_OFFLINE=0
export RUN_PROBE=1
export RUN_FULL=0
export FORCE_TRAIN=0
export FORCE_EVAL="${FORCE_EVAL:-1}"
export STOP_AFTER_STAGE=probe
export REQUIRE_INIT_CKPT=0
export REQUIRE_WAYMAX_PREFLIGHT=1
export ALLOW_QUALITY_GATE_FAILURE=0
export BACKGROUND="${BACKGROUND:-1}"
exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_8_CN.sh"
