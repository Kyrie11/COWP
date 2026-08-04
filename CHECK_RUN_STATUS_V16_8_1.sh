#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026}"
PID_FILE="$OUT_ROOT/logs/driver.pid"
LOG_FILE="$OUT_ROOT/logs/driver.nohup.log"

echo "OUT_ROOT=$OUT_ROOT"
if [[ -s "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "driver_status=RUNNING pid=$pid"
  else
    echo "driver_status=NOT_RUNNING stale_pid=$pid"
  fi
else
  echo "driver_status=NO_PID_FILE"
fi

for p in \
  "$OUT_ROOT/eval/pipeline_preflight.json" \
  "$OUT_ROOT/eval/causal_protocol_audit.json" \
  "$OUT_ROOT/eval/learned_offline/natural_basis_gate.json" \
  "$OUT_ROOT/eval/learned_offline/natural_effectiveness_gate.json" \
  "$OUT_ROOT/eval/learned_offline/bcot_calibration.json" \
  "$OUT_ROOT/eval/learned_offline/mechanism_verification.json" \
  "$OUT_ROOT/eval/probe/delta_conventional_vs_root_transport.json" \
  "$OUT_ROOT/eval/waymax/delta_conventional_vs_cowp.json" \
  "$OUT_ROOT/eval/pipeline_completion_report.json"; do
  if [[ -s "$p" ]]; then
    python - "$p" <<'PY'
import json,sys
p=sys.argv[1]
x=json.load(open(p,encoding='utf-8'))
keys=(
    'pass','status','calibration_feasible','calibration_status',
    'priority_accept_ncf_recall','priority_accept_ncf_precision',
    'priority_root_transport_auprc','learned_accepted_candidate_rate',
    'fallback_rate','priority_burden_transfer_rate',
    'priority_transfer_improvement','global_false_safe_improvement',
    'completion_level'
)
summary={k:x[k] for k in keys if k in x}
print(f"JSON {p}: {json.dumps(summary,ensure_ascii=False)}")
PY
  else
    echo "MISSING $p"
  fi
done

if [[ -s "$LOG_FILE" ]]; then
  echo "--- tail $LOG_FILE ---"
  tail -n "${TAIL_LINES:-40}" "$LOG_FILE"
fi
