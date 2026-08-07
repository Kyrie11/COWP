#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_5_bcs_rmr_fast_seed2026}"
VAL_CACHE="${VAL_CACHE:-/data0/senzeyu2/dataset/COWP/formal_v16_8_5_bcs_rmr_bcte/tensor_cache_val_transport_v16_8_4}"
CKPT="${CKPT:-$OUT_ROOT/checkpoints/planner/cowp_planner_best.pt}"
LABEL_CFG="${LABEL_CFG:-configs/label_cowp_v16_8.yaml}"
EVAL_CFG="${EVAL_CFG:-configs/eval_cowp_v16_8.yaml}"
PAIR_WITNESS_THRESHOLD="${PAIR_WITNESS_THRESHOLD:-0.5}"
NCF_GATE_MODE="${NCF_GATE_MODE:-priority}"
CALIBRATION="$OUT_ROOT/eval/learned_offline/bcot_calibration.json"
OUT="$OUT_ROOT/eval/learned_offline/selection_ablation_shared_pass_v16_8_5.json"
[[ -s "$CKPT" ]] || { echo "missing CKPT=$CKPT" >&2; exit 2; }
[[ -s "$CALIBRATION" ]] || { echo "missing calibration=$CALIBRATION; run mechanism stage first" >&2; exit 2; }
BUDGET="$($PYTHON_BIN - "$CALIBRATION" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert bool(x.get('feasible', x.get('calibration_feasible', True))), x
print(x['bcot_risk_budget'])
PY
)"
mkdir -p "$(dirname "$OUT")"

# These are true shared-forward selection/certificate baselines.  Do NOT put
# cowp_wo_neutral_branch / cowp_wo_dual_edge here: those require label-level
# diagnostics or retraining and are now rejected rather than silently becoming no-ops.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
  --mode learned_offline \
  --methods planner_score_only,idm_lattice,conventional_safety,soft_burden_cost_only,universal_ncf,cowp \
  --checkpoint "$CKPT" --cache-dir "$VAL_CACHE" \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" --eval-config "$EVAL_CFG" \
  --device cuda --batch-size "${EVAL_BATCH_SIZE:-20}" --num-workers "${EVAL_WORKERS:-4}" --prefetch-factor 1 \
  --learned-subset-modulo 2 --learned-subset-remainder 1 \
  --witness-threshold "$PAIR_WITNESS_THRESHOLD" --bcot-risk-budget "$BUDGET" \
  --ncf-gate-mode "$NCF_GATE_MODE" --offline-fallback stop_like \
  --outcome-risk-penalty 0.0 \
  --output "$OUT" --no-progress

echo "Wrote $OUT"
