#!/usr/bin/env bash
set -euo pipefail

COWP_CODE_ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$COWP_CODE_ROOT"

export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax}"
export RAW_VAL_CACHE="${RAW_VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax}"
export TRAIN_CACHE="${TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax_transport_v9}"
export VAL_CACHE="${VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax_transport_v9}"
export AUDIT_OUT="${AUDIT_OUT:-outputs/cache_reuse_audit_v9}"
mkdir -p "$AUDIT_OUT"

python -u -m cowp.scripts.19_diagnose_waymax_cache_sufficiency \
  --train-cache "$RAW_TRAIN_CACHE" \
  --val-cache "$RAW_VAL_CACHE" \
  --workers "${CACHE_AUDIT_WORKERS:-8}" \
  --output-json "$AUDIT_OUT/cache_sufficiency_current.json"

python -u -m cowp.scripts.38_gate_cache_reuse \
  --raw-train "$RAW_TRAIN_CACHE" \
  --raw-val "$RAW_VAL_CACHE" \
  --transport-train "$TRAIN_CACHE" \
  --transport-val "$VAL_CACHE" \
  --sample-scenes "${CACHE_GATE_SAMPLE_SCENES:-1024}" \
  --min-train-scenes "${MIN_TRAIN_SCENES:-20000}" \
  --min-val-scenes "${MIN_VAL_SCENES:-5000}" \
  --output "$AUDIT_OUT/cache_reuse_gate_v9.json"

echo "Cache audit passed. Reports:"
echo "  $AUDIT_OUT/cache_sufficiency_current.json"
echo "  $AUDIT_OUT/cache_reuse_gate_v9.json"
