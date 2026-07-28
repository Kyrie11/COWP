#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_1_full_data_audit_v9}"
mkdir -p "$OUT_ROOT/logs"
export BACKGROUND="${BACKGROUND:-1}"
if [[ "$BACKGROUND" == "1" && "${COWP_V16_1_AUDIT_BACKGROUND_CHILD:-0}" != "1" ]]; then
  export COWP_V16_1_AUDIT_BACKGROUND_CHILD=1
  nohup env BACKGROUND=0 bash "$0" > "$OUT_ROOT/logs/driver.nohup.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$OUT_ROOT/logs/driver.pid"
  echo "[cowp_v16.1 full audit] background pid=$pid log=$OUT_ROOT/logs/driver.nohup.log"
  exit 0
fi
FULL_CACHE_AUDIT=1 RUN_NATURAL=0 RUN_TRANSPORT=0 RUN_PLANNER=0 RUN_OFFLINE=0 RUN_PROBE=0 RUN_FULL=0 \
RUN_DIAGNOSE=1 DIAG_PROFILE=full REQUIRE_INIT_CKPT=0 BACKGROUND=0 \
STOP_AFTER_STAGE=diagnose bash "$ROOT/NEXT_RUN_COMMANDS_V16_1_CN.sh"
