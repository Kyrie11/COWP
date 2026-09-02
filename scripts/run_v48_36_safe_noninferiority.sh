#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

BASE_RUN="${BASE_RUN:?BASE_RUN is required}"
RUN="${RUN:?RUN is required}"
# BASE_RUN is normally <controller>/candidates/<variant>. Allow an explicit OUT
# for nonstandard layouts, but never authorize from a checkpoint alone.
OUT="${OUT:-$(cd -- "$BASE_RUN/../.." && pwd)}"
STATUS="$OUT/V48_36_COMPLETE.json"
mkdir -p "$OUT/logs"
[[ -s "$STATUS" ]] || { echo "missing $STATUS" >&2; exit 30; }
set +e
python tools/resolve_v48_36_authoritative_result.py --run "$OUT" --output "$OUT/AUTHORITATIVE_RUN_STATUS.json" --expect-exit-code 0 \
  >"$OUT/logs/safe_authorization_state.log" 2>&1
auth_rc=$?
set -e
[[ "$auth_rc" == 0 ]] || { echo "Safe run is not authorized by authoritative run state" >&2; exit 30; }
[[ -s "$OUT/NEXT_COMMANDS.txt" ]] || { echo "Safe run is not authorized: missing $OUT/NEXT_COMMANDS.txt" >&2; exit 20; }
python - "$STATUS" "$BASE_RUN" <<'PY_STATUS'
import json,pathlib,sys
status=json.load(open(sys.argv[1],encoding='utf-8'))
base=pathlib.Path(sys.argv[2]).resolve()
variants={pathlib.Path(v.get('checkpoint','')).resolve().parent.parent for v in (status.get('variants') or {}).values() if v.get('checkpoint')}
if not (status.get('pipeline_valid') and status.get('certificate_executed') and
        status.get('gate_evaluated') and status.get('gate_passed') and
        status.get('certificate_exit_code') == 0 and status.get('next_commands_generated')):
    raise SystemExit('v48.36 run is not Safe-evaluation authorized')
if base not in variants:
    raise SystemExit(f'BASE_RUN is not an authorized v48.36 candidate: {base}')
PY_STATUS
[[ -f "$BASE_RUN/model_v48_trac_sr/best.pt" ]] || { echo "missing checkpoint in $BASE_RUN" >&2; exit 30; }

BASE_RUN="$BASE_RUN" RUN="$RUN" \
SAFE_NOMINAL_ONLY=1 RUN_OFFLINE_EVAL=0 RUN_AUDITS=0 \
RUN_SAFE_CLOSED_LOOP=1 RUN_SAFE_PAIRED_SCALAR="${RUN_SAFE_PAIRED_SCALAR:-1}" \
SAFE_TEST="${SAFE_TEST:-/data0/senzeyu2/dataset/OCRAP/calibration_safe}" \
SAFE_BUCKET_SPLIT="${SAFE_BUCKET_SPLIT:-calibration}" \
SAFE_WOMD_SOURCE="${SAFE_WOMD_SOURCE:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}" \
SAFE_RAW_MAX_SCENARIOS="${SAFE_RAW_MAX_SCENARIOS:-0}" \
GPU_SAFE_BASELINE="${GPU_SAFE_BASELINE:-0}" GPU_SAFE="${GPU_SAFE:-1}" \
  bash scripts/run_ocrap_v48_trac_sr.sh
