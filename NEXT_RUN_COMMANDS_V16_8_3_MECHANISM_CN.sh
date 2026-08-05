#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_3_rmr_bcte}"
export COWP_ROOT="${COWP_ROOT:-$DATA_ROOT}"
export RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-$DATA_ROOT/tensor_cache_train_waymax}"
export RAW_VAL_CACHE="${RAW_VAL_CACHE:-$DATA_ROOT/tensor_cache_val_waymax}"
export TRAIN_CACHE="${TRAIN_CACHE:-$DATA_ROOT/tensor_cache_train_waymax_transport_v16_8_3}"
export VAL_CACHE="${VAL_CACHE:-$DATA_ROOT/tensor_cache_val_waymax_transport_v16_8_3}"
export DATA_PROTOCOL="${DATA_PROTOCOL:-v16_8_3_fresh}"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_3_rmr_bcte_seed2026}"
exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_8_2_MECHANISM_CN.sh"
