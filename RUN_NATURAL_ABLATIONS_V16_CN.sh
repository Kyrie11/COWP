#!/usr/bin/env bash
set -euo pipefail

ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax}"
RAW_VAL_CACHE="${RAW_VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax}"
INIT_CKPT="${INIT_CKPT:-outputs/cowp_v10_gct_probe100_seed2026/checkpoints/planner/cowp_planner_best.pt}"
MAIN_OUT_ROOT="${MAIN_OUT_ROOT:-outputs/cowp_v16_cnob_dynamics_v9labels_seed2026}"
ABL_ROOT="${ABL_ROOT:-outputs/cowp_v16_natural_ablations_v9labels_seed2026}"
TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-0,1}"
TRAIN_NPROC="${TRAIN_NPROC:-2}"
EPOCHS="${NATURAL_EPOCHS:-20}"
BATCH="${BATCH_PER_GPU:-5}"
WORKERS="${NUM_WORKERS:-4}"
LR="${NATURAL_LR:-3.0e-5}"
mkdir -p "$ABL_ROOT"

[[ -s "$INIT_CKPT" ]] || { echo "missing INIT_CKPT=$INIT_CKPT" >&2; exit 2; }
MAIN_REPORT="$MAIN_OUT_ROOT/eval/learned_offline/learned_natural_effectiveness.json"
[[ -s "$MAIN_REPORT" ]] || { echo "run the main natural stage first; missing $MAIN_REPORT" >&2; exit 2; }

run_ablation() {
  local name="$1" model_cfg="$2" train_cfg="$3"
  local out="$ABL_ROOT/$name"
  mkdir -p "$out/checkpoints" "$out/eval" "$out/logs" "$out/configs"
  cp "$model_cfg" "$out/configs/"
  cp "$train_cfg" "$out/configs/"
  if [[ "${FORCE_TRAIN:-0}" == "1" || ! -s "$out/checkpoints/cowp_natural_best.pt" || ! -s "$out/checkpoints/history_natural.json" ]]; then
    env CUDA_VISIBLE_DEVICES="$TRAIN_VISIBLE_DEVICES" \
      "$TORCHRUN_BIN" --standalone --nproc_per_node="$TRAIN_NPROC" -m cowp.scripts.03_train \
      --data-config configs/data.yaml --model-config "$model_cfg" \
      --label-config configs/label_cowp_v16.yaml --train-config "$train_cfg" \
      --cache-dir "$RAW_TRAIN_CACHE" --val-cache-dir "$RAW_VAL_CACHE" \
      --stage natural --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
      --num-workers "$WORKERS" --prefetch-factor 1 --device cuda \
      --output-dir "$out/checkpoints" --early-stop-patience 8 --early-stop-min-delta 1e-4 \
      --lr-scheduler plateau --min-lr 2e-6 --save-every 2 --no-positive-oversampling \
      --eval-before-train --reset-checkpoint-prefix natural_decoder \
      --natural-graph-warmup-epochs 2 --grad-clip 1.0 --amp --fused-adamw \
      --resume "$INIT_CKPT" 2>&1 | tee "$out/logs/train.log"
  fi
  "$PYTHON_BIN" -u -m cowp.scripts.39_diagnose_learned_natural \
    --data-config configs/data.yaml --model-config "$model_cfg" \
    --label-config configs/label_cowp_v16.yaml --train-config "$train_cfg" \
    --cache-dir "$RAW_VAL_CACHE" --checkpoint "$out/checkpoints/cowp_natural_best.pt" \
    --max-scenes "${LEARNED_NATURAL_DIAG_SCENES:-2000}" --batch-size 8 \
    --num-workers "${DIAG_WORKERS:-8}" --device "${LEARNED_NATURAL_DIAG_DEVICE:-cuda}" \
    --output "$out/eval/learned_natural_effectiveness.json" \
    2>&1 | tee "$out/logs/diagnose.log"
}

# Same architecture/capacity; remove only the new effectiveness/preservation/physical losses.
run_ablation no_effectiveness_loss configs/model_cowp_v16.yaml configs/train_cowp_v16_no_effectiveness_loss.yaml
# Same losses/architecture; set only the OBS capacity boost to zero.
run_ablation no_obs_capacity_boost configs/model_cowp_v16_no_obs_capacity.yaml configs/train_cowp_v16.yaml

"$PYTHON_BIN" -u -m cowp.scripts.41_compare_natural_ablations \
  --main "$MAIN_REPORT" \
  --no-effectiveness-loss "$ABL_ROOT/no_effectiveness_loss/eval/learned_natural_effectiveness.json" \
  --no-obs-capacity "$ABL_ROOT/no_obs_capacity_boost/eval/learned_natural_effectiveness.json" \
  --output "$ABL_ROOT/natural_component_attribution_gate.json" \
  2>&1 | tee "$ABL_ROOT/natural_component_attribution_gate.log"
