#!/usr/bin/env bash
set -euo pipefail
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_1_cnob_dynamics_v9labels_seed2026}"
PID_FILE="$OUT_ROOT/logs/driver.pid"
LOG_FILE="$OUT_ROOT/logs/driver.nohup.log"
if [[ -s "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "RUNNING pid=$pid"
  else
    echo "NOT_RUNNING last_pid=$pid"
  fi
else
  echo "NO_PID_FILE"
fi
if [[ -s "$LOG_FILE" ]]; then
  echo "--- tail $LOG_FILE ---"
  tail -n "${TAIL_LINES:-80}" "$LOG_FILE"
fi
