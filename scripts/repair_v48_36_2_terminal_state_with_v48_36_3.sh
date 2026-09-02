#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_36_1_ocaf_cuda_hotfix_48361}"

python tools/repair_v48_36_2_terminal_state_failure.py \
  --run "$OUTPUTDIR" --repo "$REPO" \
  --output "$OUTPUTDIR/V48_36_3_TERMINAL_STATE_REPAIR.json"

python - "$OUTPUTDIR" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
repair=json.loads((root/'V48_36_3_TERMINAL_STATE_REPAIR.json').read_text())
state=json.loads((root/'AUTHORITATIVE_RUN_STATUS.json').read_text())
complete=json.loads((root/'V48_36_COMPLETE.json').read_text())
assert repair.get('valid') is True, repair
assert repair.get('algorithm_changed') is False
assert repair.get('retraining_performed') is False
assert repair.get('recalibration_performed') is False
assert state.get('valid') is True and state.get('authoritative_exit_code') == 20, state
assert complete.get('pipeline_valid') is True and complete.get('pipeline_exit_code') == 20, complete
print('v48.36.3 terminal-state repair PASS: authoritative RC=20; no retraining/recalibration')
PY
