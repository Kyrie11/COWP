#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_5_bcs_rmr_bcte}"
SEEDS="${SEEDS:-2026 2027 2028}"
for seed in $SEEDS; do
  echo "===== mechanism seed=$seed ====="
  TRAIN_SEED="$seed" \
  DATA_ROOT="$DATA_ROOT" \
  OUT_ROOT="${OUT_BASE:-outputs/cowp_v16_8_5_bcs_rmr_fast}_seed${seed}" \
  BACKGROUND=0 \
  bash "$ROOT/NEXT_RUN_COMMANDS_V16_8_5_MECHANISM_CN.sh"
done
