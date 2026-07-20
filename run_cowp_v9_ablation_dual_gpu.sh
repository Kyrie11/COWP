#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_CACHE="${TRAIN_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax_v9}"
VAL_CACHE="${VAL_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax_v9}"
INIT_CKPT="${INIT_CKPT:-outputs/cowp_v8_probe100_seed2026/checkpoints/planner/cowp_planner_best.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v9_ablation_seed2026}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
EPOCHS="${EPOCHS:-18}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-3e-5}"
SEED="${SEED:-2026}"

mkdir -p "$OUT_ROOT"

run_variant() {
  local name="$1" gpu="$2" cfg_src="$3"
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"/{logs,configs,checkpoints,eval}
  cp "$cfg_src" "$out/configs/train.yaml"
  cp configs/label_cowp_v9.yaml "$out/configs/label.yaml"
  sed -i -E "0,/^  seed:/{s/^  seed:.*/  seed: ${SEED}/}" "$out/configs/train.yaml"

  echo "[$name] training on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u -m cowp.scripts.03_train \
    --data-config configs/data.yaml --model-config configs/model.yaml \
    --label-config "$out/configs/label.yaml" --train-config "$out/configs/train.yaml" \
    --cache-dir "$TRAIN_CACHE" --val-cache-dir "$VAL_CACHE" \
    --stage planner --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" \
    --num-workers "$NUM_WORKERS" --prefetch-factor 1 --device cuda \
    --output-dir "$out/checkpoints" --with-waymax-outcome-labels \
    --freeze-backbone-epochs 2 --early-stop-patience 6 --early-stop-min-delta 1e-4 \
    --lr-scheduler plateau --min-lr 2e-6 --save-every 3 \
    --no-positive-oversampling --no-response-traj --no-response-components \
    --amp --fused-adamw --resume "$INIT_CKPT" \
    > >(tee "$out/logs/train.log") 2> >(tee -a "$out/logs/train.log" >&2)

  local ckpt="$out/checkpoints/cowp_planner_best.pt"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
    --mode learned_offline --method cowp --checkpoint "$ckpt" \
    --cache-dir "$VAL_CACHE" --data-config configs/data.yaml \
    --label-config "$out/configs/label.yaml" --eval-config configs/eval.yaml \
    --device cuda --batch-size 24 --num-workers 4 --prefetch-factor 1 \
    --witness-threshold 0.30 \
    --witness-threshold-sweep 0.20,0.24,0.27,0.30,0.33,0.36,0.40,0.45,0.50 \
    --ncf-gate-mode priority --offline-fallback stop_like \
    --outcome-risk-penalty 1.0 --outcome-risk-threshold 0.65 \
    --output "$out/eval/cowp.json" --no-progress \
    > >(tee "$out/logs/eval.log") 2> >(tee -a "$out/logs/eval.log" >&2)
}

# GPU0: no per-mode supervision. GPU1: per-mode labels but nearest-neighbour
# assignment. The main v9 run is balanced Sinkhorn OT and is compared against both.
run_variant aggregate_only "$GPU0" configs/train_cowp_v9_aggregate_ablation.yaml & p0=$!
run_variant nearest_match "$GPU1" configs/train_cowp_v9_nearest_ablation.yaml & p1=$!
wait "$p0"; wait "$p1"

echo "[cowp_v9_ablation] complete: $OUT_ROOT"
