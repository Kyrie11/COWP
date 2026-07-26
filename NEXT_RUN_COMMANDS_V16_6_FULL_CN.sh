#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_NATURAL_ROOT="${SOURCE_NATURAL_ROOT:-outputs/cowp_v16_5_natural_recovery_v9labels_seed2026}"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_6_full_pipeline_v9labels_seed2026}"
NAT_BASIS_GATE="$SOURCE_NATURAL_ROOT/eval/learned_offline/natural_basis_gate.json"
NAT_EFFECT_GATE="$SOURCE_NATURAL_ROOT/eval/learned_offline/natural_effectiveness_gate.json"
MAIN_REPORT="$SOURCE_NATURAL_ROOT/eval/learned_offline/learned_natural_effectiveness.json"
ATTR_GATE="${ATTR_GATE:-outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json}"
"$PYTHON_BIN" - "$NAT_BASIS_GATE" "$NAT_EFFECT_GATE" "$ATTR_GATE" <<'PY'
import json,sys
for p in sys.argv[1:]:
    x=json.load(open(p,encoding='utf-8'))
    assert bool(x.get('pass',x.get('passed',False))), f'gate failed: {p}'
print('natural and protocol-aligned attribution gates passed')
PY
TARGET_EPOCH="$($PYTHON_BIN - "$MAIN_REPORT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
e=x.get('checkpoint_epoch')
assert isinstance(e,int) and e>=0
print(e)
PY
)"
printf -v TARGET_TAG '%03d' "$TARGET_EPOCH"
CANDIDATE="$SOURCE_NATURAL_ROOT/checkpoints/natural/cowp_natural_epoch${TARGET_TAG}.pt"
BEST="$SOURCE_NATURAL_ROOT/checkpoints/natural/cowp_natural_best.pt"
if [[ -s "$CANDIDATE" ]]; then
  export NATURAL_CKPT="$CANDIDATE"
elif [[ -s "$BEST" ]]; then
  "$PYTHON_BIN" - "$BEST" "$TARGET_EPOCH" <<'PY'
import sys,torch
x=torch.load(sys.argv[1],map_location='cpu')
assert int(x.get('epoch',-1))==int(sys.argv[2]), (x.get('epoch'),sys.argv[2])
PY
  export NATURAL_CKPT="$BEST"
else
  echo "missing selected natural checkpoint under $SOURCE_NATURAL_ROOT/checkpoints/natural" >&2; exit 2
fi
export NATURAL_HISTORY="$SOURCE_NATURAL_ROOT/checkpoints/natural/history_natural.json"
[[ -s "$NATURAL_HISTORY" ]] || { echo "missing NATURAL_HISTORY=$NATURAL_HISTORY" >&2; exit 2; }

# Use a new v16.6 root so strict provenance remains valid.  The validated v16.5
# natural checkpoint is imported explicitly; it is not mixed with modified code
# under the old experiment root.
export STOP_AFTER_STAGE=none
export RUN_DIAGNOSE="${RUN_DIAGNOSE:-1}"
export RUN_NATURAL=0
export RUN_TRANSPORT=1
export RUN_PLANNER=1
export RUN_OFFLINE=1
export RUN_PROBE=1
export RUN_FULL="${RUN_FULL:-1}"
export REQUIRE_INIT_CKPT=0
export REQUIRE_WAYMAX_PREFLIGHT=1
export ALLOW_QUALITY_GATE_FAILURE=0
export BACKGROUND="${BACKGROUND:-1}"
exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_6_CN.sh"
