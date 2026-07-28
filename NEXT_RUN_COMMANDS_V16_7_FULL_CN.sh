#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_7_mechanism_v9labels_seed2026}"
MECH="$OUT_ROOT/eval/learned_offline/mechanism_verification.json"
"$PYTHON_BIN" - "$MECH" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert bool(x.get('pass',False)), f'mechanism gate failed: {sys.argv[1]}'
assert x.get('gate_role') == 'development_continuation_not_paper_claim'
print('v16.7 mechanism continuation gate passed; launching real Waymax closed loop')
PY
# Reuse checkpoints and calibrated budget under the exact same provenance root.
export RUN_DIAGNOSE=0
export RUN_NATURAL=0
export RUN_TRANSPORT=0
export RUN_PLANNER=0
export RUN_OFFLINE=0
export RUN_PROBE="${RUN_PROBE:-1}"
export RUN_FULL="${RUN_FULL:-1}"
export FORCE_TRAIN=0
export FORCE_EVAL="${FORCE_EVAL:-1}"
export STOP_AFTER_STAGE=none
export REQUIRE_INIT_CKPT=0
export REQUIRE_WAYMAX_PREFLIGHT=1
export ALLOW_QUALITY_GATE_FAILURE=0
export BACKGROUND="${BACKGROUND:-1}"
exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_7_CN.sh"
