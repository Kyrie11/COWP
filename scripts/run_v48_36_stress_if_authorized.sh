#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="${OUT:?OUT is required}"
mkdir -p "$OUT/logs"
[[ -s "$OUT/NEXT_COMMANDS.txt" ]] || { echo "Stress run is not authorized: missing $OUT/NEXT_COMMANDS.txt" >&2; exit 20; }
[[ -s "$OUT/V48_36_COMPLETE.json" ]] || { echo "missing $OUT/V48_36_COMPLETE.json" >&2; exit 30; }
set +e
python tools/resolve_v48_36_authoritative_result.py --run "$OUT" --output "$OUT/AUTHORITATIVE_RUN_STATUS.json" --expect-exit-code 0 \
  >"$OUT/logs/stress_authorization_state.log" 2>&1
auth_rc=$?
set -e
[[ "$auth_rc" == 0 ]] || { echo "Stress run is not authorized by authoritative run state" >&2; exit 30; }
python - "$OUT/V48_36_COMPLETE.json" <<'PY_STATUS'
import json,sys
status=json.load(open(sys.argv[1],encoding='utf-8'))
if not (status.get('pipeline_valid') and status.get('certificate_executed') and
        status.get('gate_evaluated') and status.get('gate_passed') and
        status.get('certificate_exit_code') == 0 and status.get('next_commands_generated')):
    raise SystemExit('v48.36 run is not stress-authorized')
PY_STATUS
[[ -s "$OUT/chosen_base_run_dedicated.txt" ]] || { echo "missing chosen dedicated run" >&2; exit 30; }
BASE_RUN="$(cat "$OUT/chosen_base_run_dedicated.txt")"
[[ -f "$BASE_RUN/model_v48_trac_sr/best.pt" ]] || { echo "missing checkpoint in $BASE_RUN" >&2; exit 30; }
for f in \
  "$BASE_RUN/calibration/direct_value_risk_near_v48.json" \
  "$BASE_RUN/calibration/direct_value_risk_contact_v48.json" \
  "$BASE_RUN/calibration/dev_frozen_shared_rule_v48.json"; do
  [[ -s "$f" ]] || { echo "missing shared-certificate artifact $f" >&2; exit 30; }
done
python - \
  "$BASE_RUN/calibration/direct_value_risk_near_v48.json" \
  "$BASE_RUN/calibration/direct_value_risk_contact_v48.json" \
  "$BASE_RUN/calibration/dev_frozen_shared_rule_v48.json" <<'PY_SHARED'
import hashlib,json,sys
near=json.load(open(sys.argv[1],encoding='utf-8'))
contact=json.load(open(sys.argv[2],encoding='utf-8'))
shared_path=sys.argv[3]
shared_sha=hashlib.sha256(open(shared_path,'rb').read()).hexdigest()
rules=[]
for name,doc in [('near',near),('contact',contact)]:
    source=doc.get('frozen_rule_source') or {}
    if source.get('sha256') != shared_sha:
        raise SystemExit(f'{name} certificate was not verified with the shared frozen rule')
    rules.append(doc.get('selector_overrides') or doc.get('diagnostic_selector_overrides') or {})
if rules[0] != rules[1]:
    raise SystemExit('Near and Contact certificates do not expose the same shared selector rule')
PY_SHARED
BASE_RUN="$BASE_RUN" \
RUN="${RUN:-$OUT/stress_closed_loop}" \
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}" \
NEAR_TEST="${NEAR_TEST:-/data0/senzeyu2/dataset/OCRAP/test_near_contact}" \
CONTACT_TEST="${CONTACT_TEST:-/data0/senzeyu2/dataset/OCRAP/test_contact}" \
RUN_OFFLINE_EVAL=1 RUN_AUDITS=1 RUN_SAFE_CLOSED_LOOP=0 RUN_SCALAR_BASELINES=0 RUN_DIRECT_VALUE=true \
GPU0="${GPU0:-0}" GPU1="${GPU1:-1}" CL_RESUME="${CL_RESUME:-0}" \
  bash scripts/run_ocrap_v48_trac_sr.sh
