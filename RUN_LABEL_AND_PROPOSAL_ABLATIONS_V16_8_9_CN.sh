#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit}"
LABELS_VAL="${LABELS_VAL:-$DATA_ROOT/labels_val}"
VAL_CACHE="${VAL_CACHE:-$DATA_ROOT/tensor_cache_val}"
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_9_causal_audit_seed2026}"
OUT_DIR="$OUT_ROOT/eval/ablation"; mkdir -p "$OUT_DIR"

# Oracle/label-space causal mechanisms: no learned checkpoint and no fake shared-forward branch removal.
"$PYTHON_BIN" -u -m cowp.scripts.05_make_tables \
  --data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml \
  --eval-config configs/eval_cowp_v16_8.yaml --labels-dir "$LABELS_VAL" \
  --output-dir "$OUT_DIR/label_space" --load-workers "${LABEL_TABLE_LOAD_WORKERS:-8}"

# Proposal source contribution on the actual fresh bank.
"$PYTHON_BIN" -u -m cowp.scripts.50_ablate_proposal_sources \
  --cache-dir "$VAL_CACHE" --output "$OUT_DIR/proposal_sources_full.json"
"$PYTHON_BIN" -u -m cowp.scripts.50_ablate_proposal_sources \
  --cache-dir "$VAL_CACHE" --subset-modulo 2 --subset-remainder 0 \
  --output "$OUT_DIR/proposal_sources_calibration.json"
"$PYTHON_BIN" -u -m cowp.scripts.50_ablate_proposal_sources \
  --cache-dir "$VAL_CACHE" --subset-modulo 2 --subset-remainder 1 \
  --output "$OUT_DIR/proposal_sources_heldout.json"

# New v16.8.9 dataset/certificate contract diagnostics.
"$PYTHON_BIN" -u -m cowp.scripts.57_diagnose_causal_audit \
  --cache-dir "$VAL_CACHE" --output "$OUT_DIR/causal_audit_full.json"
"$PYTHON_BIN" -u -m cowp.scripts.45_diagnose_proposal_ceiling \
  --cache-dir "$VAL_CACHE" --output "$OUT_DIR/proposal_ceiling_full.json" \
  --hard-count 0 --random-count 0 --control-count 0 --seed 2026

echo "Wrote v16.8.9 label/proposal/audit ablations under $OUT_DIR"
