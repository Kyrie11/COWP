#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"

[[ "${ALLOW_LEGACY_V16_8_BANK:-0}" == "1" ]] || {
  echo "This is a DEVELOPMENT-ONLY old-bank diagnostic. Set ALLOW_LEGACY_V16_8_BANK=1 explicitly." >&2
  exit 2
}

DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export COWP_ROOT="$DATA_ROOT"
export RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-$DATA_ROOT/tensor_cache_train_waymax}"
export RAW_VAL_CACHE="${RAW_VAL_CACHE:-$DATA_ROOT/tensor_cache_val_waymax}"
export TRAIN_CACHE="${TRAIN_CACHE:-$DATA_ROOT/tensor_cache_train_waymax_transport_v16_8}"
export VAL_CACHE="${VAL_CACHE:-$DATA_ROOT/tensor_cache_val_waymax_transport_v16_8}"
export DATA_PROTOCOL="v16_8_root_conditioned_overlay"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_6_legacy_bank_diagnostic_seed2026}"
export USE_WAYMAX_OUTCOME_LABELS="${USE_WAYMAX_OUTCOME_LABELS:-1}"
mkdir -p "$OUT_ROOT/configs"
python - "$OUT_ROOT/configs/legacy_bank_diagnostic_manifest.json" "$TRAIN_CACHE" "$VAL_CACHE" <<'PY'
import json,sys
json.dump({
  'schema_version':'cowp_v16_8_6_legacy_bank_diagnostic_v1',
  'paper_grade_fresh_proposal':False,
  'proposal_source_ablation_allowed':False,
  'closed_loop_cowp_paper_claim_allowed':False,
  'train_cache':sys.argv[2], 'val_cache':sys.argv[3],
  'purpose':'development-only certificate/planner saturation diagnostic on the audited stale v16.8 proposal bank',
},open(sys.argv[1],'w'),indent=2)
PY

echo "WARNING: this run cannot validate v16.8.6 proposal improvements or paper-grade COWP Waymax claims."
exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_8_2_MECHANISM_CN.sh"
