#!/usr/bin/env bash
set -euo pipefail

# v16.8.25: NO dataset/label/cache reconstruction.
# All learned-offline probes reuse formal_v16_8_24_compact_full_5k caches.
# Strict Waymax reads the existing held-out IDs directly from WOMD validation.

MODE="${1:-help}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
BASE_RUN="${BASE_RUN:-outputs/v16_8_24_compact5k_all}"
BASE_CKPT="${BASE_CKPT:-$BASE_RUN/cowp_all_best.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/v16_8_25_ctu_probe}"
REPAIR_RUN="${REPAIR_RUN:-outputs/v16_8_25_planner_repair}"
BCOT_BUDGET="${BCOT_BUDGET:-0.50}"
PROBE_N="${PROBE_N:-200}"
mkdir -p "$OUT_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

common_eval=(
  --data-config configs/data.yaml
  --label-config configs/label_cowp_v16_8.yaml
  --eval-config configs/eval_cowp_v16_8.yaml
  --checkpoint "$BASE_CKPT"
  --ncf-gate-mode priority
  --witness-threshold 0.70
  --batch-size 8
  --device cuda:0
  --num-workers 2
  --prefetch-factor 1
  --outcome-risk-penalty 0
)

case "$MODE" in
  val_sweep)
    # Question: does removing the second BCOT ranking change the operating curve?
    python -m cowp.scripts.04_eval_closed_loop \
      "${common_eval[@]}" \
      --mode learned_offline \
      --cache-dir "$COWP_ROOT/tensor_cache_val_waymax" \
      --method cowp_cert_utility \
      --bcot-risk-budget-sweep 0.30,0.35,0.40,0.45,0.50 \
      --bcot-risk-budget "$BCOT_BUDGET" \
      --output "$OUT_ROOT/val_ctu_budget_sweep.json"
    ;;

  val_compare)
    # One shared model pass. COWP and CTU MUST have identical certificate metrics.
    python -m cowp.scripts.04_eval_closed_loop \
      "${common_eval[@]}" \
      --mode learned_offline \
      --cache-dir "$COWP_ROOT/tensor_cache_val_waymax" \
      --methods cowp,cowp_cert_utility,conventional_safety,planner_score_only,soft_burden_cost_only,universal_ncf \
      --bcot-risk-budget "$BCOT_BUDGET" \
      --output "$OUT_ROOT/val_ctu_compare.json"
    python -m cowp.scripts.71_compare_ctu_probe \
      --input "$OUT_ROOT/val_ctu_compare.json" \
      --output "$OUT_ROOT/val_ctu_screen.json"
    ;;

  heldout_compare)
    # Run only after inspecting val_ctu_screen.json. This is dev evidence now, not a blind final test.
    python -m cowp.scripts.04_eval_closed_loop \
      "${common_eval[@]}" \
      --mode learned_offline \
      --cache-dir "$COWP_ROOT/tensor_cache_heldout_test_waymax" \
      --methods cowp,cowp_cert_utility,conventional_safety,planner_score_only,soft_burden_cost_only,universal_ncf \
      --bcot-risk-budget "$BCOT_BUDGET" \
      --output "$OUT_ROOT/heldout_ctu_compare.json"
    python -m cowp.scripts.71_compare_ctu_probe \
      --input "$OUT_ROOT/heldout_ctu_compare.json" \
      --output "$OUT_ROOT/heldout_ctu_screen.json"
    ;;

  make_waymax_probe_ids)
    src="$COWP_ROOT/heldout_test_scene_ids.txt"
    dst="$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt"
    python - "$src" "$dst" "$PROBE_N" <<'PY'
import hashlib, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ids=[]
for line in open(src, encoding='utf-8'):
    sid=line.strip()
    if sid and sid not in ids:
        ids.append(sid)
ranked=sorted(ids, key=lambda s: hashlib.sha256(("v16.8.25-waymax-probe|"+s).encode()).hexdigest())
sel=ranked[:n]
if len(sel) < n:
    raise SystemExit(f"requested {n} ids but only {len(sel)} unique ids exist")
open(dst,'w',encoding='utf-8').write('\n'.join(sel)+'\n')
print(f"wrote {len(sel)} paired exact IDs -> {dst}")
print("sha256", hashlib.sha256(('\n'.join(sel)+'\n').encode()).hexdigest())
PY
    ;;

  waymax_probe)
    ids="$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt"
    [[ -s "$ids" ]] || { echo "Missing $ids; run: $0 make_waymax_probe_ids" >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    for method in cowp cowp_cert_utility; do
      python -m cowp.scripts.04_eval_closed_loop \
        --data-config configs/data.yaml \
        --label-config configs/label_cowp_v16_8.yaml \
        --eval-config configs/eval_cowp_v16_8.yaml \
        --mode waymax \
        --method "$method" \
        --checkpoint "$BASE_CKPT" \
        --device cuda:0 \
        --waymax-device gpu \
        --jax-visible-devices 1 \
        --jax-preallocate false \
        --waymax-split validation \
        --scenario-ids-file "$ids" \
        --rollout-horizon-steps 80 \
        --waymax-action-mode absolute_xy_yaw \
        --ncf-gate-mode priority \
        --witness-threshold 0.70 \
        --bcot-risk-budget "$BCOT_BUDGET" \
        --outcome-risk-penalty 0 \
        --waymax-standard-metrics \
        --waymax-standard-metric-names OverlapMetric,OffroadMetric,ProgressionMetric,KinematicsInfeasibilityMetric \
        --output "$OUT_ROOT/waymax_${PROBE_N}_${method}.json"
    done
    ;;

  waymax_full)
    # Publication-strength paired exact-ID developer test after the 200-scene probe passes.
    ids="$COWP_ROOT/heldout_test_scene_ids.txt"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    for method in cowp cowp_cert_utility; do
      python -m cowp.scripts.04_eval_closed_loop \
        --data-config configs/data.yaml \
        --label-config configs/label_cowp_v16_8.yaml \
        --eval-config configs/eval_cowp_v16_8.yaml \
        --mode waymax --method "$method" --checkpoint "$BASE_CKPT" \
        --device cuda:0 --waymax-device gpu --jax-visible-devices 1 --jax-preallocate false \
        --waymax-split validation --scenario-ids-file "$ids" \
        --rollout-horizon-steps 80 --waymax-action-mode absolute_xy_yaw \
        --ncf-gate-mode priority --witness-threshold 0.70 --bcot-risk-budget "$BCOT_BUDGET" \
        --outcome-risk-penalty 0 --waymax-standard-metrics \
        --waymax-standard-metric-names OverlapMetric,OffroadMetric,ProgressionMetric,KinematicsInfeasibilityMetric \
        --output "$OUT_ROOT/waymax_full_${method}.json"
    done
    ;;

  planner_repair)
    # Run only if CTU/diagnostics show remaining post-certificate planner head room.
    # Cross-stage warm start: do NOT add --resume-training.
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    torchrun --standalone --nproc_per_node=2 -m cowp.scripts.03_train \
      --data-config configs/data.yaml \
      --model-config configs/model_cowp_v16_8.yaml \
      --label-config configs/label_cowp_v16_8.yaml \
      --train-config configs/train_cowp_v16_8_25_planner_repair.yaml \
      --cache-dir "$COWP_ROOT/tensor_cache_train_waymax" \
      --val-cache-dir "$COWP_ROOT/tensor_cache_val_waymax" \
      --stage planner \
      --resume "$BASE_CKPT" \
      --with-waymax-outcome-labels \
      --epochs 6 \
      --lr 1e-5 \
      --batch-size 4 \
      --amp --amp-dtype bfloat16 \
      --num-workers 4 --val-num-workers 2 \
      --prefetch-factor 1 --val-prefetch-factor 1 \
      --sharing-strategy file_system --fused-adamw \
      --eval-before-train \
      --output-dir "$REPAIR_RUN"
    ;;

  eval_repair_val)
    repair_ckpt="$REPAIR_RUN/cowp_planner_best.pt"
    [[ -f "$repair_ckpt" ]] || { echo "Missing $repair_ckpt; run planner_repair first" >&2; exit 2; }
    python -m cowp.scripts.04_eval_closed_loop \
      --data-config configs/data.yaml \
      --label-config configs/label_cowp_v16_8.yaml \
      --eval-config configs/eval_cowp_v16_8.yaml \
      --mode learned_offline \
      --cache-dir "$COWP_ROOT/tensor_cache_val_waymax" \
      --checkpoint "$repair_ckpt" \
      --methods cowp,cowp_cert_utility \
      --ncf-gate-mode priority --witness-threshold 0.70 \
      --bcot-risk-budget "$BCOT_BUDGET" \
      --outcome-risk-penalty 0 \
      --batch-size 8 --device cuda:0 --num-workers 2 --prefetch-factor 1 \
      --output "$REPAIR_RUN/val_repair_compare.json"
    python -m cowp.scripts.71_compare_ctu_probe \
      --input "$REPAIR_RUN/val_repair_compare.json" \
      --output "$REPAIR_RUN/val_repair_ctu_screen.json"
    ;;

  help|*)
    cat <<EOF
Usage: $0 MODE

Recommended order (no dataset rebuild):
  val_sweep
  val_compare
  heldout_compare          # only if val is non-inferior
  make_waymax_probe_ids
  waymax_probe             # paired exact-ID 200-scene strict Waymax

Only if the probes show planner-head room:
  planner_repair
  eval_repair_val

After the 200-scene strict probe passes and the algorithm is locked:
  waymax_full
EOF
    ;;
esac
