#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:=/data0/senzeyu2/dataset/COWP/formal}"
: "${COWP_ROOT:=$DATA_ROOT}"
: "${RAW_TRAIN_CACHE:=$DATA_ROOT/tensor_cache_train_waymax}"
: "${RAW_VAL_CACHE:=$DATA_ROOT/tensor_cache_val_waymax}"
: "${TRAIN_CACHE:=$DATA_ROOT/tensor_cache_train_waymax_transport_v16_8}"
: "${VAL_CACHE:=$DATA_ROOT/tensor_cache_val_waymax_transport_v16_8}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
: "${PROBE_ROOT:=$DATA_ROOT/proposal_probe_v16_8_4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PROBE_ROOT"

python "$SCRIPT_DIR/tools/probe_cache_schema.py" \
  --cache-dir "$VAL_CACHE" \
  --sample 16 \
  --output "$PROBE_ROOT/current_val_schema.json"

python "$SCRIPT_DIR/tools/analyze_proposal_cache.py" \
  --cache-dir "$VAL_CACHE" \
  --sample 0 \
  --promotion-config "$SCRIPT_DIR/configs/proposal_promotion_v16_8_4.yaml" \
  --output "$PROBE_ROOT/current_val_proposal_audit.json"

python "$SCRIPT_DIR/tools/select_probe_scenarios.py" \
  --old-cache "$VAL_CACHE" \
  --index-jsonl "$COWP_ROOT/index_val.jsonl" \
  --hard-count 400 \
  --random-count 800 \
  --seed 2026 \
  --output-index-jsonl "$PROBE_ROOT/probe_index_val.jsonl" \
  --output-manifest "$PROBE_ROOT/probe_manifest.json"

cat <<EOF

Current-cache audit and paired scenario selection are complete.

Next, from the actual COWP source repository, build the two label-only arms using:
  $SCRIPT_DIR/configs/label_cowp_v16_8_4_single_region_control.yaml
  $SCRIPT_DIR/configs/label_cowp_v16_8_4_rmr_bcte.yaml

Exact build commands are in:
  $SCRIPT_DIR/README_CN.md

Do not start a full rebuild until compare_proposal_caches.py reports both:
  promote_to_full_rebuild=true
  algorithm_increment_demonstrated=true
EOF
