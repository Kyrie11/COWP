#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_3_rmr_bcte}"
export COWP_ROOT="${COWP_ROOT:-$DATA_ROOT}"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_3_rmr_bcte_seed2026}"
exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_8_2_FULL_CN.sh"
