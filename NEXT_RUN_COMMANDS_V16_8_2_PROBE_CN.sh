#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_2_certificate_consistent_v9base_seed2026}"
MECH="$OUT_ROOT/eval/learned_offline/mechanism_verification.json"
"$PYTHON_BIN" - "$MECH" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x.get('pass') is True, f'mechanism gate failed: {sys.argv[1]}'
assert x.get('calibration_feasible') is True, f'calibration infeasible: {sys.argv[1]}'
assert x.get('heldout_certificate_semantics_current') is True
assert x.get('calibration_certificate_semantics_current') is True
print('v16.8.2 mechanism/calibration gates passed; launching Waymax probe')
PY
TRANSFER="$OUT_ROOT/configs/natural_attribution_transfer_manifest.json"
REUSED="$OUT_ROOT/configs/reused_checkpoint_manifest.json"
if [[ -s "$REUSED" ]]; then
  mapfile -t PATHS < <("$PYTHON_BIN" - "$REUSED" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))['artifacts']
for k in ['natural_checkpoint','natural_history','transport_checkpoint','planner_checkpoint']:
    print(x[k]['path'])
PY
)
  export NATURAL_CKPT="${NATURAL_CKPT:-${PATHS[0]}}"
  export NATURAL_HISTORY="${NATURAL_HISTORY:-${PATHS[1]}}"
  export TRANSPORT_CKPT="${TRANSPORT_CKPT:-${PATHS[2]}}"
  export CKPT="${CKPT:-${PATHS[3]}}"
elif [[ -s "$TRANSFER" ]]; then
  mapfile -t PATHS < <("$PYTHON_BIN" - "$TRANSFER" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
print(x['natural_checkpoint']); print(x['natural_history'])
PY
)
  export NATURAL_CKPT="${NATURAL_CKPT:-${PATHS[0]}}"
  export NATURAL_HISTORY="${NATURAL_HISTORY:-${PATHS[1]}}"
  export TRANSPORT_CKPT="${TRANSPORT_CKPT:-$OUT_ROOT/checkpoints/transport/cowp_witness_best.pt}"
  export CKPT="${CKPT:-$OUT_ROOT/checkpoints/planner/cowp_planner_best.pt}"
else
  echo "missing checkpoint transfer/reuse manifest under $OUT_ROOT/configs" >&2; exit 2
fi
for p in "$NATURAL_CKPT" "$NATURAL_HISTORY" "$TRANSPORT_CKPT" "$CKPT"; do
  [[ -s "$p" ]] || { echo "missing artifact: $p" >&2; exit 2; }
done
export RUN_DIAGNOSE=0 RUN_NATURAL=0 RUN_TRANSPORT=0 RUN_PLANNER=0 RUN_OFFLINE=0
export RUN_PROBE=1 RUN_FULL=0 FORCE_TRAIN=0 FORCE_EVAL="${FORCE_EVAL:-1}"
export STOP_AFTER_STAGE=probe REQUIRE_INIT_CKPT=0 REQUIRE_WAYMAX_PREFLIGHT=1
export ALLOW_QUALITY_GATE_FAILURE=0 BACKGROUND="${BACKGROUND:-1}"
exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_8_CN.sh"
