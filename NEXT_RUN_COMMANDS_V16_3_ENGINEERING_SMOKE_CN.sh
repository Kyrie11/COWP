#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_3_engineering_smoke_v9labels_seed2026_ancdatafix}"
mkdir -p "$OUT_ROOT/logs"
export BACKGROUND="${BACKGROUND:-1}"
if [[ "$BACKGROUND" == "1" && "${COWP_V16_3_SMOKE_BACKGROUND_CHILD:-0}" != "1" ]]; then
  export COWP_V16_3_SMOKE_BACKGROUND_CHILD=1
  nohup env BACKGROUND=0 bash "$0" > "$OUT_ROOT/logs/smoke_driver.nohup.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$OUT_ROOT/logs/smoke_driver.pid"
  echo "[cowp_v16.2 smoke] background pid=$pid"
  echo "[cowp_v16.2 smoke] log=$OUT_ROOT/logs/smoke_driver.nohup.log"
  exit 0
fi
# Engineering-only end-to-end code-path test. Outputs must not be used as paper evidence.
NATURAL_AMP=0 AMP_DTYPE=auto ALLOW_QUALITY_GATE_FAILURE=1 \
NATURAL_EPOCHS="${NATURAL_EPOCHS:-1}" TRANSPORT_EPOCHS="${TRANSPORT_EPOCHS:-1}" PLANNER_EPOCHS="${PLANNER_EPOCHS:-1}" \
EARLY_STOP_PATIENCE=1 PROBE_SCENARIOS="${PROBE_SCENARIOS:-20}" RUN_FULL=0 RUN_PROBE=1 \
STOP_AFTER_STAGE=probe RUN_DIAGNOSE=0 REQUIRE_WAYMAX_PREFLIGHT=1 BACKGROUND=0 \
bash "$ROOT/NEXT_RUN_COMMANDS_V16_3_CN.sh"
