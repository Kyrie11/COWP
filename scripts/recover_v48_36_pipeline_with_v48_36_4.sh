#!/usr/bin/env bash
# Unified fail-closed recovery entrypoint for known v48.36 engineering signatures.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_36_1_ocaf_cuda_hotfix_48361}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
mkdir -p "$OUTPUTDIR/logs"

# First resolve idempotent terminal-state cases. This covers the uploaded result:
# a valid RC=20 was archived and later overwritten by a refused resume attempt.
set +e
OUTPUTDIR="$OUTPUTDIR" OCRAP_REPO="$REPO" \
  bash scripts/repair_v48_36_3_resume_clobber_with_v48_36_4.sh
terminal_rc=$?
set -e
if [[ "$terminal_rc" == 0 || "$terminal_rc" == 20 ]]; then
  exit "$terminal_rc"
fi

failure_stage="$(python - "$OUTPUTDIR" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'PIPELINE_FAILED.json'
try: print(json.loads(p.read_text()).get('stage',''))
except Exception: print('')
PY
)"

if [[ "$failure_stage" == terminal_state_contract ]]; then
  # Exact v48.36.2 attempt-ID mismatch. Requires original checkpoint bytes.
  set +e
  OUTPUTDIR="$OUTPUTDIR" OCRAP_REPO="$REPO" \
    bash scripts/repair_v48_36_2_terminal_state_with_v48_36_3.sh
  rc=$?
  set -e
  [[ "$rc" == 0 ]] || exit 30
  exit 20
fi

stage_repair_valid="$(python - "$OUTPUTDIR" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'V48_36_2_STAGE_TRANSFER_REPAIR.json'
try: print('1' if json.loads(p.read_text()).get('valid') is True else '0')
except Exception: print('0')
PY
)"
if [[ "$stage_repair_valid" != 1 ]]; then
  set +e
  OUTPUTDIR="$OUTPUTDIR" SOURCE_RUN="$SOURCE_RUN" OCRAP_ROOT="$OCRAP_ROOT" \
  PROTOCOL_ROOT="$PROTOCOL_ROOT" CAL_SAFE="$CAL_SAFE" GPU0="$GPU0" GPU1="$GPU1" \
  REPAIR_ONLY=1 bash scripts/repair_v48_36_1_stage_transfer_with_v48_36_2.sh
  rc=$?
  set -e
  [[ "$rc" == 0 ]] || exit 30
fi

set +e
OUTPUTDIR="$OUTPUTDIR" OCRAP_ROOT="$OCRAP_ROOT" SOURCE_RUN="$SOURCE_RUN" \
PROTOCOL_ROOT="$PROTOCOL_ROOT" CAL_SAFE="$CAL_SAFE" GPU0="$GPU0" GPU1="$GPU1" \
RESUME_AFTER_ADAPTATION=1 REBUILD_ADAPT_INDEX=0 REBUILD_ADAPT_DEV_INDEX=0 \
OCRAP_IMPLEMENTATION_VERSION=v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX \
bash scripts/run_v48_36_ocaf_dedicated.sh
rc=$?
set -e
exit "$rc"
