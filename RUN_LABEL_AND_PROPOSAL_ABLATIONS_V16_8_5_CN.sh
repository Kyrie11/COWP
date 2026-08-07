#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_5_bcs_rmr_bcte}"
LABELS_VAL="${LABELS_VAL:-$DATA_ROOT/labels_val}"
VAL_CACHE="${VAL_CACHE:-$DATA_ROOT/tensor_cache_val_transport_v16_8_4}"
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_5_bcs_rmr_fast_seed2026}"
OUT_DIR="$OUT_ROOT/eval/ablation"
mkdir -p "$OUT_DIR"

# 1) Label/certificate-space causal-module ablations.  These use source-resolved
# labels and are valid without retraining, but should be reported as oracle/label-
# space mechanism diagnostics rather than learned architecture ablations.
"$PYTHON_BIN" -u -m cowp.scripts.05_make_tables \
  --data-config configs/data.yaml \
  --label-config configs/label_cowp_v16_8.yaml \
  --eval-config configs/eval_cowp_v16_8.yaml \
  --labels-dir "$LABELS_VAL" --output-dir "$OUT_DIR/label_space"

# 2) Proposal-bank source ablation: isolate the marginal coverage/floor effect of
# BCS-RMR-BCTE without retraining or running Waymax.
"$PYTHON_BIN" -u -m cowp.scripts.50_ablate_proposal_sources \
  --cache-dir "$VAL_CACHE" --output "$OUT_DIR/proposal_sources_full.json"
"$PYTHON_BIN" -u -m cowp.scripts.50_ablate_proposal_sources \
  --cache-dir "$VAL_CACHE" --subset-modulo 2 --subset-remainder 0 \
  --output "$OUT_DIR/proposal_sources_calibration.json"
"$PYTHON_BIN" -u -m cowp.scripts.50_ablate_proposal_sources \
  --cache-dir "$VAL_CACHE" --subset-modulo 2 --subset-remainder 1 \
  --output "$OUT_DIR/proposal_sources_heldout.json"

echo "Wrote ablations under $OUT_DIR"
