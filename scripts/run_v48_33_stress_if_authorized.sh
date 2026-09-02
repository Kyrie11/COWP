#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="${OUT:?OUT is required}"
[[ -s "$OUT/NEXT_COMMANDS.txt" ]] || { echo "Natural gate is not authorized: missing $OUT/NEXT_COMMANDS.txt" >&2; exit 20; }
[[ -s "$OUT/V48_33_COMPLETE.json" ]] || { echo "missing $OUT/V48_33_COMPLETE.json" >&2; exit 30; }
python - "$OUT/V48_33_COMPLETE.json" <<'PY_STATUS'
import json,sys
d=json.load(open(sys.argv[1]))
if not (d.get('pipeline_valid') and d.get('gate_evaluated') and d.get('gate_passed') and d.get('certificate_exit_code')==0 and d.get('next_commands_generated')):
    raise SystemExit('v48.33 run is not stress-authorized')
PY_STATUS
[[ -s "$OUT/chosen_base_run_dedicated.txt" ]] || { echo "missing chosen dedicated run" >&2; exit 30; }
BASE_RUN="$(cat "$OUT/chosen_base_run_dedicated.txt")"
[[ -f "$BASE_RUN/model_v48_trac_sr/best.pt" ]] || { echo "missing checkpoint in $BASE_RUN" >&2; exit 30; }
BASE_RUN="$BASE_RUN" \
RUN="${RUN:-$OUT/stress_closed_loop}" \
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}" \
NEAR_TEST="${NEAR_TEST:-/data0/senzeyu2/dataset/OCRAP/test_near_contact}" \
CONTACT_TEST="${CONTACT_TEST:-/data0/senzeyu2/dataset/OCRAP/test_contact}" \
RUN_OFFLINE_EVAL=1 RUN_AUDITS=1 RUN_SAFE_CLOSED_LOOP=0 RUN_SCALAR_BASELINES=0 RUN_DIRECT_VALUE=true \
GPU0="${GPU0:-0}" GPU1="${GPU1:-1}" CL_RESUME="${CL_RESUME:-0}" \
  bash scripts/run_ocrap_v48_trac_sr.sh
