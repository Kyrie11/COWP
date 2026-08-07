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
  --labels-dir "$LABELS_VAL" --output-dir "$OUT_DIR/label_space" \
  --load-workers "${LABEL_TABLE_LOAD_WORKERS:-8}"

# 2) Proposal-source ablation is meaningful ONLY for fresh caches that contain
# proposal provenance.  Old v16.8 overlays have no proposal_source tensor; the
# previous script silently labeled all such candidates as PAD and produced a
# fictitious "RMR increment".  Detect and skip instead.
if "$PYTHON_BIN" - "$VAL_CACHE" <<'PYPROV'
import sys
from cowp.data.dataset import COWPNpzDataset
ds=COWPNpzDataset(sys.argv[1])
row=ds.load(0, {"cowp/candidates/proposal_source"})
raise SystemExit(0 if "cowp/candidates/proposal_source" in row else 2)
PYPROV
then
  "$PYTHON_BIN" -u -m cowp.scripts.50_ablate_proposal_sources \
    --cache-dir "$VAL_CACHE" --output "$OUT_DIR/proposal_sources_full.json"
  "$PYTHON_BIN" -u -m cowp.scripts.50_ablate_proposal_sources \
    --cache-dir "$VAL_CACHE" --subset-modulo 2 --subset-remainder 0 \
    --output "$OUT_DIR/proposal_sources_calibration.json"
  "$PYTHON_BIN" -u -m cowp.scripts.50_ablate_proposal_sources \
    --cache-dir "$VAL_CACHE" --subset-modulo 2 --subset-remainder 1 \
    --output "$OUT_DIR/proposal_sources_heldout.json"
else
  echo "[proposal-source ablation] SKIP: $VAL_CACHE has no fresh proposal provenance; old-bank source attribution would be invalid."
  "$PYTHON_BIN" - "$OUT_DIR/proposal_sources_skipped.json" "$VAL_CACHE" <<'PYSKIP'
import json,sys
json.dump({"skipped":True,"reason":"missing proposal_source provenance; stale v16.8 bank cannot support source ablation","cache":sys.argv[2]},open(sys.argv[1],'w'),indent=2)
PYSKIP
fi

echo "Wrote ablations under $OUT_DIR"
