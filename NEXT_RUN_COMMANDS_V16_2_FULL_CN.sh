#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_2_pipeline_v9labels_seed2026}"
mkdir -p "$OUT_ROOT/logs"
export BACKGROUND="${BACKGROUND:-1}"
if [[ "$BACKGROUND" == "1" && "${COWP_V16_2_FULL_BACKGROUND_CHILD:-0}" != "1" ]]; then
  export COWP_V16_2_FULL_BACKGROUND_CHILD=1
  nohup env BACKGROUND=0 STOP_AFTER_STAGE=none RUN_DIAGNOSE="${RUN_DIAGNOSE:-0}" \
    RUN_PROBE="${RUN_PROBE:-1}" RUN_FULL="${RUN_FULL:-1}" REQUIRE_WAYMAX_PREFLIGHT=1 \
    bash "$0" > "$OUT_ROOT/logs/full_driver.nohup.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$OUT_ROOT/logs/full_driver.pid"
  echo "[cowp_v16.2 full] background pid=$pid"
  echo "[cowp_v16.2 full] log=$OUT_ROOT/logs/full_driver.nohup.log"
  exit 0
fi
STOP_AFTER_STAGE=none RUN_DIAGNOSE="${RUN_DIAGNOSE:-0}" RUN_PROBE="${RUN_PROBE:-1}" \
RUN_FULL="${RUN_FULL:-1}" REQUIRE_WAYMAX_PREFLIGHT=1 BACKGROUND=0 \
bash "$ROOT/NEXT_RUN_COMMANDS_V16_2_CN.sh"
