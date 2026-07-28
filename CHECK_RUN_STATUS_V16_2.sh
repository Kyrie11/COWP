#!/usr/bin/env bash
set -euo pipefail
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_2_pipeline_v9labels_seed2026}"
for kind in driver full_driver smoke_driver; do
  pid_file="$OUT_ROOT/logs/${kind}.pid"
  log_file="$OUT_ROOT/logs/${kind}.nohup.log"
  if [[ -s "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then state=RUNNING; else state=EXITED; fi
    echo "$kind: $state pid=$pid log=$log_file"
  fi
done
if [[ -s "$OUT_ROOT/eval/pipeline_completion_report.json" ]]; then
  python -m json.tool "$OUT_ROOT/eval/pipeline_completion_report.json"
fi
