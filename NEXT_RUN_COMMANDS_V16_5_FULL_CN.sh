#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_5_natural_recovery_v9labels_seed2026}"
NAT_BASIS_GATE="$OUT_ROOT/eval/learned_offline/natural_basis_gate.json"
NAT_EFFECT_GATE="$OUT_ROOT/eval/learned_offline/natural_effectiveness_gate.json"
ATTR_GATE="${ATTR_GATE:-outputs/cowp_v16_5_natural_ablations_v9labels_seed2026/natural_component_attribution_gate.json}"
python - "$NAT_BASIS_GATE" "$NAT_EFFECT_GATE" "$ATTR_GATE" <<'PY2'
import json,sys
for p in sys.argv[1:]:
    x=json.load(open(p,encoding='utf-8'))
    assert bool(x.get('pass',x.get('passed',False))), f'gate failed: {p}'
print('natural and attribution gates passed')
PY2
export STOP_AFTER_STAGE=none
export RUN_DIAGNOSE="${RUN_DIAGNOSE:-0}"
export RUN_NATURAL=1
export RUN_TRANSPORT=1
export RUN_PLANNER=1
export RUN_OFFLINE=1
export RUN_PROBE=1
export RUN_FULL="${RUN_FULL:-1}"
export REQUIRE_WAYMAX_PREFLIGHT=1
export ALLOW_QUALITY_GATE_FAILURE=0
export BACKGROUND="${BACKGROUND:-1}"
exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_5_CN.sh"
