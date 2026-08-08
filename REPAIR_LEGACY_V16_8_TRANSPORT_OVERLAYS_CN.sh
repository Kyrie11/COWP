#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
TRAIN_BASE="${TRAIN_BASE:-$COWP_ROOT/tensor_cache_train}"
VAL_BASE="${VAL_BASE:-$COWP_ROOT/tensor_cache_val}"
OLD_TRAIN_OVERLAY="${OLD_TRAIN_OVERLAY:-$COWP_ROOT/tensor_cache_train_waymax_transport_v16_8}"
OLD_VAL_OVERLAY="${OLD_VAL_OVERLAY:-$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8}"
NEW_TRAIN_OVERLAY="${NEW_TRAIN_OVERLAY:-$COWP_ROOT/tensor_cache_train_transport_v16_8_rebased}"
NEW_VAL_OVERLAY="${NEW_VAL_OVERLAY:-$COWP_ROOT/tensor_cache_val_transport_v16_8_rebased}"

"$PYTHON_BIN" -m cowp.scripts.54_rebase_transport_overlay \
  --base-cache "$TRAIN_BASE" --old-overlay "$OLD_TRAIN_OVERLAY" \
  --output-dir "$NEW_TRAIN_OVERLAY" --verify-all-sidecars --force
"$PYTHON_BIN" -m cowp.scripts.54_rebase_transport_overlay \
  --base-cache "$VAL_BASE" --old-overlay "$OLD_VAL_OVERLAY" \
  --output-dir "$NEW_VAL_OVERLAY" --verify-all-sidecars --force

echo "Legacy diagnostic TRAIN_CACHE=$NEW_TRAIN_OVERLAY"
echo "Legacy diagnostic VAL_CACHE=$NEW_VAL_OVERLAY"
echo "These caches intentionally contain no waymax/* outcomes. Set USE_WAYMAX_OUTCOME_LABELS=0."
echo "They remain legacy v16.8 proposal banks and are NOT paper-grade v16.8.6 data."
