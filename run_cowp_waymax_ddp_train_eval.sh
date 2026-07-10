#!/usr/bin/env bash
set -euo pipefail

# COWP full training + learned-offline + online Waymax closed-loop evaluation.
# Run from the patched COWP repository root, or set REPO_ROOT=/path/to/COWP.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
cd "$REPO_ROOT"

# -------------------------- Paths --------------------------
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export TFEXAMPLE_VAL="${TFEXAMPLE_VAL:-$WOMD_ROOT/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"

TRAIN_CACHE="${TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax}"
VAL_CACHE="${VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax}"
OUT_ROOT="${OUT_ROOT:-outputs/cowp_waymax_full}"

# -------------------------- Runtime --------------------------
NUM_GPUS="${NUM_GPUS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

PER_GPU_BATCH="${PER_GPU_BATCH:-32}"       # effective global batch = NUM_GPUS * PER_GPU_BATCH
EVAL_BATCH="${EVAL_BATCH:-64}"
TRAIN_WORKERS="${TRAIN_WORKERS:-6}"       # per DDP process; reduce to 4 if CPU/RAM is tight
PREFETCH="${PREFETCH:-1}"
VAL_EVERY="${VAL_EVERY:-1}"

# Outcome-risk penalty should be >0 only when planner is trained with attached Waymax labels.
OUTCOME_RISK_PENALTY="${OUTCOME_RISK_PENALTY:-0.5}"
OUTCOME_RISK_THRESHOLD="${OUTCOME_RISK_THRESHOLD:-1.10}"
THRESH_SWEEP="${THRESH_SWEEP:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/checkpoints" "$OUT_ROOT/eval/learned_offline" "$OUT_ROOT/eval/waymax"

if ! grep -q "DistributedDataParallel" cowp/scripts/03_train.py; then
  echo "ERROR: DDP patch not detected in cowp/scripts/03_train.py. Apply the supplied DDP patch or use the patched COWP zip first." >&2
  exit 2
fi
if ! python -m cowp.scripts.04_eval_closed_loop --help 2>/dev/null | grep -q -- "--waymax-split"; then
  echo "ERROR: Waymax split/shard patch not detected in 04_eval_closed_loop.py. Apply the supplied eval patch or use the patched COWP zip first." >&2
  exit 2
fi

if [[ ! -d "$TRAIN_CACHE" || ! -d "$VAL_CACHE" ]]; then
  echo "ERROR: attached Waymax tensor caches were not found:" >&2
  echo "  TRAIN_CACHE=$TRAIN_CACHE" >&2
  echo "  VAL_CACHE=$VAL_CACHE" >&2
  exit 2
fi

#echo "[0/6] Verify attached Waymax tensor caches"
#python -m cowp.scripts.14_verify_waymax_cache --cache-dir "$TRAIN_CACHE"
#python -m cowp.scripts.14_verify_waymax_cache --cache-dir "$VAL_CACHE"
#
train_ddp () {
  local stage="$1"; shift
  torchrun --standalone --nproc_per_node="$NUM_GPUS" \
    -m cowp.scripts.03_train \
    --data-config configs/data.yaml \
    --model-config configs/model.yaml \
    --train-config configs/train.yaml \
    --cache-dir "$TRAIN_CACHE" \
    --val-cache-dir "$VAL_CACHE" \
    --stage "$stage" \
    --batch-size "$PER_GPU_BATCH" \
    --num-workers "$TRAIN_WORKERS" \
    --prefetch-factor "$PREFETCH" \
    --amp \
    --fused-adamw \
    --pin-memory \
    --val-every "$VAL_EVERY" \
    "$@"
}

echo "[1/6] Train response encoder/head on attached cache, without loading dense response trajectory labels"
train_ddp response \
  --epochs "${RESPONSE_EPOCHS:-14}" \
  --lr "${RESPONSE_LR:-3e-4}" \
  --no-response-traj \
  --no-response-components \
  --output-dir "$OUT_ROOT/checkpoints/response"

echo "[2/6] Train witness stage from best response checkpoint"
train_ddp witness \
  --epochs "${WITNESS_EPOCHS:-24}" \
  --lr "${WITNESS_LR:-3e-4}" \
  --resume "$OUT_ROOT/checkpoints/response/cowp_response_best.pt" \
  --output-dir "$OUT_ROOT/checkpoints/witness"

echo "[3/6] Train planner with attached Waymax candidate outcome labels"
train_ddp planner \
  --epochs "${PLANNER_EPOCHS:-28}" \
  --lr "${PLANNER_LR:-2e-4}" \
  --resume "$OUT_ROOT/checkpoints/witness/cowp_witness_best.pt" \
  --with-waymax-outcome-labels \
  --output-dir "$OUT_ROOT/checkpoints/planner_waymax"

CKPT="$OUT_ROOT/checkpoints/planner_waymax/cowp_planner_best.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: planner checkpoint not found: $CKPT" >&2
  exit 2
fi

wait_group () {
  local status=0
  for pid in "$@"; do
    wait "$pid" || status=$?
  done
  return "$status"
}

echo "[4/6] Learned-offline evaluation on attached validation cache, methods run in parallel across GPUs"
LEARNED_METHODS=(idm_lattice conventional_safety planner_score_only soft_burden_cost_only universal_ncf cowp outcome_oracle)
pids=()
i=0
for METHOD in "${LEARNED_METHODS[@]}"; do
  GPU=$(( i % NUM_GPUS ))
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    python -m cowp.scripts.04_eval_closed_loop \
      --data-config configs/data.yaml \
      --label-config configs/label.yaml \
      --eval-config configs/eval.yaml \
      --cache-dir "$VAL_CACHE" \
      --mode learned_offline \
      --method "$METHOD" \
      --checkpoint "$CKPT" \
      --batch-size "$EVAL_BATCH" \
      --device cuda \
      --witness-threshold 0.5 \
      --witness-threshold-sweep "$THRESH_SWEEP" \
      --ncf-gate-mode priority \
      --priority-hard-threshold 0.55 \
      --offline-fallback stop_like \
      --adaptive-frontier-margin 0.20 \
      --outcome-risk-penalty "$OUTCOME_RISK_PENALTY" \
      --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
      --output "$OUT_ROOT/eval/learned_offline/${METHOD}.json"
  ) &
  pids+=("$!")
  i=$(( i + 1 ))
  if (( i % NUM_GPUS == 0 )); then
    wait_group "${pids[@]}"
    pids=()
  fi
done
if (( ${#pids[@]} > 0 )); then
  wait_group "${pids[@]}"
fi

echo "[5/6] Online Waymax closed-loop evaluation on validation split, sharded across GPUs"
ONLINE_METHODS=(planner_score_only conventional_safety soft_burden_cost_only universal_ncf cowp)
TOTAL_ONLINE_SCENARIOS="${TOTAL_ONLINE_SCENARIOS:-400}"
PER_SHARD_SCENARIOS=$(( (TOTAL_ONLINE_SCENARIOS + NUM_GPUS - 1) / NUM_GPUS ))

for METHOD in "${ONLINE_METHODS[@]}"; do
  pids=()
  for SHARD in $(seq 0 $(( NUM_GPUS - 1 ))); do
    GPU="$SHARD"
    (
      export CUDA_VISIBLE_DEVICES="$GPU"
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      python -m cowp.scripts.04_eval_closed_loop \
        --data-config configs/data.yaml \
        --label-config configs/label.yaml \
        --eval-config configs/eval.yaml \
        --mode waymax \
        --waymax-split validation \
        --tfexample-glob "$TFEXAMPLE_VAL" \
        --method "$METHOD" \
        --checkpoint "$CKPT" \
        --num-scenarios "$PER_SHARD_SCENARIOS" \
        --num-shards "$NUM_GPUS" \
        --shard-index "$SHARD" \
        --rollout-horizon-steps 80 \
        --waymax-standard-metrics \
        --waymax-device gpu \
        --jax-visible-devices 0 \
        --jax-preallocate false \
        --waymax-action-mode absolute_xy_yaw \
        --device cuda \
        --witness-threshold 0.5 \
        --ncf-gate-mode priority \
        --priority-hard-threshold 0.55 \
        --adaptive-frontier-margin 0.20 \
        --outcome-risk-penalty "$OUTCOME_RISK_PENALTY" \
        --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
        --clear-accelerator-cache \
        --output "$OUT_ROOT/eval/waymax/${METHOD}_shard${SHARD}.json"
    ) &
    pids+=("$!")
  done
  wait_group "${pids[@]}"

  python - "$OUT_ROOT/eval/waymax/${METHOD}_merged.json" "$OUT_ROOT/eval/waymax/${METHOD}_shard"*.json <<'PY'
import json, sys
from pathlib import Path
from cowp.waymax_eval.metrics_cowp import policy_diagnostic_episode_summary, policy_diagnostic_summary
from cowp.waymax_eval.metrics_standard import aggregate_waymax_standard_metrics
out = Path(sys.argv[1])
paths = [Path(p) for p in sys.argv[2:]]
payloads = [json.loads(p.read_text()) for p in paths if p.exists()]
rollouts = []
for p in payloads:
    rollouts.extend(p.get("rollouts", []))
merged = dict(payloads[0]) if payloads else {}
merged["merged_from"] = [str(p) for p in paths]
merged["num_rollouts"] = len(rollouts)
merged["steps"] = [int(x.get("steps", 0)) for x in rollouts]
merged["policy_diagnostic_summary"] = policy_diagnostic_summary(rollouts)
merged["closed_loop_cowp_metric_summary"] = policy_diagnostic_episode_summary(rollouts)
if any("standard_metric_summary" in p for p in payloads):
    merged["standard_metric_summary"] = aggregate_waymax_standard_metrics(rollouts)
merged.pop("standard_metrics", None)
out.write_text(json.dumps(merged, indent=2, default=str))
print(json.dumps({"merged_output": str(out), "num_rollouts": len(rollouts)}, indent=2))
PY
done

echo "[6/6] Done"
echo "Best planner checkpoint: $CKPT"
echo "Learned-offline outputs: $OUT_ROOT/eval/learned_offline"
echo "Waymax closed-loop outputs: $OUT_ROOT/eval/waymax"
