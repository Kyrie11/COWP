#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax}"
RAW_VAL_CACHE="${RAW_VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax}"
TRAIN_CACHE="${TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax_transport_v16_8}"
VAL_CACHE="${VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8}"
WORKERS_TRAIN="${AUG_WORKERS_TRAIN:-12}"
WORKERS_VAL="${AUG_WORKERS_VAL:-8}"
LABEL_CFG="configs/label_cowp_v16_8.yaml"

for d in "$RAW_TRAIN_CACHE" "$RAW_VAL_CACHE"; do
  [[ -d "$d" ]] || { echo "missing raw cache: $d" >&2; exit 2; }
done

"$PYTHON_BIN" -u -m cowp.scripts.26_augment_transport_labels \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --input-dir "$RAW_TRAIN_CACHE" --output-dir "$TRAIN_CACHE" \
  --num-workers "$WORKERS_TRAIN" --chunksize 2 --storage-mode overlay \
  --sidecar-subdir .transport_v16_8 --force
"$PYTHON_BIN" -u -m cowp.scripts.26_augment_transport_labels \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --input-dir "$RAW_VAL_CACHE" --output-dir "$VAL_CACHE" \
  --num-workers "$WORKERS_VAL" --chunksize 2 --storage-mode overlay \
  --sidecar-subdir .transport_v16_8 --force

"$PYTHON_BIN" -u -m cowp.scripts.27_diagnose_transport_labels \
  --cache-dir "$TRAIN_CACHE" --workers 8 \
  --output "$TRAIN_CACHE/transport_diagnostics_v16_8.json"
"$PYTHON_BIN" -u -m cowp.scripts.27_diagnose_transport_labels \
  --cache-dir "$VAL_CACHE" --workers 8 \
  --output "$VAL_CACHE/transport_diagnostics_v16_8.json"

"$PYTHON_BIN" -u -m cowp.scripts.33_diagnose_cache_alignment \
  --raw-cache "$RAW_TRAIN_CACHE" --transport-cache "$TRAIN_CACHE" \
  --max-scenes 2000 --workers 8 --hash-mode sampled \
  --output "$TRAIN_CACHE/cache_alignment_v16_8.json"
"$PYTHON_BIN" -u -m cowp.scripts.33_diagnose_cache_alignment \
  --raw-cache "$RAW_VAL_CACHE" --transport-cache "$VAL_CACHE" \
  --max-scenes 2000 --workers 8 --hash-mode sampled \
  --output "$VAL_CACHE/cache_alignment_v16_8.json"

echo "TRAIN_CACHE=$TRAIN_CACHE"
echo "VAL_CACHE=$VAL_CACHE"
