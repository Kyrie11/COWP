#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit}"
export COWP_ROOT="${COWP_ROOT:-$DATA_ROOT}"
export RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-$DATA_ROOT/tensor_cache_train}"
export RAW_VAL_CACHE="${RAW_VAL_CACHE:-$DATA_ROOT/tensor_cache_val}"
export TRAIN_CACHE="${TRAIN_CACHE:-$DATA_ROOT/tensor_cache_train}"
export VAL_CACHE="${VAL_CACHE:-$DATA_ROOT/tensor_cache_val}"
export USE_WAYMAX_OUTCOME_LABELS="${USE_WAYMAX_OUTCOME_LABELS:-0}"
export OFFLINE_OUTCOME_RISK_PENALTY="${OFFLINE_OUTCOME_RISK_PENALTY:-0.0}"
export DATA_PROTOCOL="${DATA_PROTOCOL:-v16_8_9_causal_audit_fresh}"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_9_causal_audit_seed2026}"
mkdir -p "$OUT_ROOT/configs" "$OUT_ROOT/logs"
"$PYTHON_BIN" -u -m cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol \
  --cowp-root "$COWP_ROOT" --raw-train "$RAW_TRAIN_CACHE" --raw-val "$RAW_VAL_CACHE" \
  --transport-train "$TRAIN_CACHE" --transport-val "$VAL_CACHE" \
  --sample-scenes "${FRESH_CACHE_GATE_SAMPLE_SCENES:-256}" \
  --output "$OUT_ROOT/configs/fresh_cache_protocol_gate_v16_8_9.json" \
  | tee "$OUT_ROOT/logs/fresh_cache_protocol_gate_v16_8_9.log"

# The natural root semantics are unchanged, but fixed-critical/audit data changes
# the evaluated agent distribution. Revalidate the transferred natural decoder
# on the actual fresh v16.8.9 validation cache before spending transport/planner
# GPU-hours. This is stronger provenance than blindly trusting an old-data gate.
if [[ "${FRESH_NATURAL_REVALIDATE:-1}" == "1" ]]; then
  SOURCE_NATURAL_ROOT="${SOURCE_NATURAL_ROOT:-outputs/cowp_v16_6_natural_recovery_v9labels_seed2026}"
  MAIN_REPORT="$SOURCE_NATURAL_ROOT/eval/learned_offline/learned_natural_effectiveness.json"
  [[ -s "$MAIN_REPORT" ]] || { echo "missing old validated natural report: $MAIN_REPORT" >&2; exit 2; }
  TARGET_EPOCH="$($PYTHON_BIN - "$MAIN_REPORT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8')); e=x.get('checkpoint_epoch')
assert isinstance(e,int) and e>=0, e
print(e)
PY
)"
  printf -v TARGET_TAG '%03d' "$TARGET_EPOCH"
  FRESH_NATURAL_CKPT="$SOURCE_NATURAL_ROOT/checkpoints/natural/cowp_natural_epoch${TARGET_TAG}.pt"
  [[ -s "$FRESH_NATURAL_CKPT" ]] || FRESH_NATURAL_CKPT="$SOURCE_NATURAL_ROOT/checkpoints/natural/cowp_natural_best.pt"
  [[ -s "$FRESH_NATURAL_CKPT" ]] || { echo "missing validated natural checkpoint under $SOURCE_NATURAL_ROOT" >&2; exit 2; }
  "$PYTHON_BIN" -u -m cowp.scripts.39_diagnose_learned_natural \
    --cache-dir "$RAW_VAL_CACHE" --checkpoint "$FRESH_NATURAL_CKPT" \
    --data-config configs/data.yaml --model-config configs/model_cowp_v16_8.yaml \
    --label-config configs/label_cowp_v16_8.yaml --train-config configs/train_cowp_v16_8.yaml \
    --max-scenes "${FRESH_NATURAL_MAX_SCENES:-2000}" --batch-size "${FRESH_NATURAL_BATCH:-16}" \
    --num-workers "${FRESH_NATURAL_WORKERS:-4}" --device cuda --no-progress \
    --output "$OUT_ROOT/eval/fresh_v16_8_9_natural_effectiveness.json" \
    | tee "$OUT_ROOT/logs/fresh_v16_8_9_natural_effectiveness.log"
  "$PYTHON_BIN" -u -m cowp.scripts.40_gate_natural_effectiveness \
    --report "$OUT_ROOT/eval/fresh_v16_8_9_natural_effectiveness.json" \
    --output "$OUT_ROOT/eval/fresh_v16_8_9_natural_effectiveness_gate.json" \
    | tee "$OUT_ROOT/logs/fresh_v16_8_9_natural_effectiveness_gate.log"
fi

exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_8_2_MECHANISM_CN.sh"
