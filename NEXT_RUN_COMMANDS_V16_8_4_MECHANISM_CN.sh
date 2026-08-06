#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_4_bcs_rmr_bcte}"
export COWP_ROOT="${COWP_ROOT:-$DATA_ROOT}"
export RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-$DATA_ROOT/tensor_cache_train_waymax}"
export RAW_VAL_CACHE="${RAW_VAL_CACHE:-$DATA_ROOT/tensor_cache_val_waymax}"
export TRAIN_CACHE="${TRAIN_CACHE:-$DATA_ROOT/tensor_cache_train_waymax_transport_v16_8_4}"
export VAL_CACHE="${VAL_CACHE:-$DATA_ROOT/tensor_cache_val_waymax_transport_v16_8_4}"
export DATA_PROTOCOL="${DATA_PROTOCOL:-v16_8_4_fresh}"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_4_bcs_rmr_bcte_seed2026}"
mkdir -p "$OUT_ROOT/configs" "$OUT_ROOT/logs"

# Fail before expensive training/Waymax if the caller accidentally exports the old
# v16.8 transport overlay.  Overlay labels cannot replace the inherited ego candidate bank.
"$PYTHON_BIN" -u -m cowp.scripts.47_gate_fresh_v16_8_4_cache_protocol \
  --cowp-root "$COWP_ROOT" \
  --raw-train "$RAW_TRAIN_CACHE" --raw-val "$RAW_VAL_CACHE" \
  --transport-train "$TRAIN_CACHE" --transport-val "$VAL_CACHE" \
  --sample-scenes "${FRESH_CACHE_GATE_SAMPLE_SCENES:-256}" \
  --output "$OUT_ROOT/configs/fresh_cache_protocol_gate_v16_8_4.json" \
  | tee "$OUT_ROOT/logs/fresh_cache_protocol_gate_v16_8_4.log"

exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_8_2_MECHANISM_CN.sh"
