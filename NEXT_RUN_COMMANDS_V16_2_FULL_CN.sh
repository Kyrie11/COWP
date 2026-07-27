#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_2_pipeline_v9labels_seed2026}"
export STOP_AFTER_STAGE=none
export RUN_DIAGNOSE="${RUN_DIAGNOSE:-0}"
export RUN_NATURAL=1
export RUN_TRANSPORT=1
export RUN_PLANNER=1
export RUN_OFFLINE=1
export RUN_PROBE=1
export RUN_FULL="${RUN_FULL:-1}"
export REQUIRE_WAYMAX_PREFLIGHT=1
export ALLOW_QUALITY_GATE_FAILURE=0
export BACKGROUND="${BACKGROUND:-1}"
exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_2_CN.sh"
