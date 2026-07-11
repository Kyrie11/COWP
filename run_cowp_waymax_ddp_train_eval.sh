#!/usr/bin/env bash
set -euo pipefail

# COWP full training + learned-offline + online Waymax closed-loop evaluation.
# Resume-safe version: every completed stage/method/shard can be skipped.
# Run from the patched COWP repository root, or set REPO_ROOT=/path/to/COWP.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
cd "$REPO_ROOT"

# -------------------------- Paths --------------------------
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"

# Raw tf.Example globs are kept for cache-building/debug tools that use the lightweight parser.
# Online Waymax evaluation should use Waymax/TensorFlow sharded path syntax: *_tfexample.tfrecord@N.
export TFEXAMPLE_TRAIN_GLOB="${TFEXAMPLE_TRAIN_GLOB:-$WOMD_ROOT/uncompressed/tf_example/training/*.tfrecord*}"
export TFEXAMPLE_VAL_GLOB="${TFEXAMPLE_VAL_GLOB:-$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*}"
export TFEXAMPLE_TEST_GLOB="${TFEXAMPLE_TEST_GLOB:-$WOMD_ROOT/uncompressed/tf_example/testing/*.tfrecord*}"

export WAYMAX_TFEXAMPLE_TRAIN="${WAYMAX_TFEXAMPLE_TRAIN:-$WOMD_ROOT/uncompressed/tf_example/training/training_tfexample.tfrecord@1000}"
export WAYMAX_TFEXAMPLE_VAL="${WAYMAX_TFEXAMPLE_VAL:-$WOMD_ROOT/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
export WAYMAX_TFEXAMPLE_TEST="${WAYMAX_TFEXAMPLE_TEST:-$WOMD_ROOT/uncompressed/tf_example/testing/testing_tfexample.tfrecord@150}"

# Backward-compatible variable name.  If you override TFEXAMPLE_VAL with a glob,
# this script converts it to the validation @150 path before online Waymax eval.
export TFEXAMPLE_VAL="${TFEXAMPLE_VAL:-$WAYMAX_TFEXAMPLE_VAL}"

export TRAIN_CACHE="${TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax}"
export VAL_CACHE="${VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax}"
OUT_ROOT="${OUT_ROOT:-outputs/cowp_waymax_full}"

# -------------------------- Runtime --------------------------
NUM_GPUS="${NUM_GPUS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
# Do NOT enable expandable_segments here. On the observed PyTorch/CUDA stack it can
# crash inside CUDACachingAllocator with: !block->expandable_segment_.
# max_split_size_mb is stable and helps fragmentation without entering that code path.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

PER_GPU_BATCH="${PER_GPU_BATCH:-32}"       # effective global batch = NUM_GPUS * PER_GPU_BATCH
EVAL_BATCH="${EVAL_BATCH:-64}"
TRAIN_WORKERS="${TRAIN_WORKERS:-6}"       # per DDP process; reduce to 4 if CPU/RAM is tight
PREFETCH="${PREFETCH:-1}"
VAL_EVERY="${VAL_EVERY:-1}"

# Resume/skip behavior.
# SKIP_EXISTING_STAGES=1: skip any completed train/eval stage with valid expected outputs.
# FORCE_RERUN_LEARNED=1: rerun learned-offline even if JSON exists.
# FORCE_RERUN_ONLINE=1: rerun online shards even if JSON exists.
# FORCE_REMERGE_ONLINE=1: recompute merged JSON even if it exists.
SKIP_EXISTING_STAGES="${SKIP_EXISTING_STAGES:-1}"
FORCE_RERUN_LEARNED="${FORCE_RERUN_LEARNED:-0}"
FORCE_RERUN_ONLINE="${FORCE_RERUN_ONLINE:-0}"
FORCE_REMERGE_ONLINE="${FORCE_REMERGE_ONLINE:-0}"

# Online split and scale.
ONLINE_SPLIT="${ONLINE_SPLIT:-validation}"   # training | validation | testing
TOTAL_ONLINE_SCENARIOS="${TOTAL_ONLINE_SCENARIOS:-400}"

# Outcome-risk penalty should be >0 only when planner is trained with attached Waymax labels.
OUTCOME_RISK_PENALTY="${OUTCOME_RISK_PENALTY:-0.5}"
OUTCOME_RISK_THRESHOLD="${OUTCOME_RISK_THRESHOLD:-1.10}"
THRESH_SWEEP="${THRESH_SWEEP:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/checkpoints" "$OUT_ROOT/eval/learned_offline" "$OUT_ROOT/eval/waymax"

# -------------------------- Helpers --------------------------
json_ok () {
  local f="$1"
  [[ -s "$f" ]] || return 1
  python - "$f" <<'PY' >/dev/null 2>&1
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    json.load(fh)
PY
}

should_skip_existing () {
  [[ "$SKIP_EXISTING_STAGES" == "1" ]]
}

wait_group () {
  local status=0
  for pid in "$@"; do
    wait "$pid" || status=$?
  done
  return "$status"
}

waymax_path_for_split () {
  local split="$1"
  case "$split" in
    training|train) echo "$WAYMAX_TFEXAMPLE_TRAIN" ;;
    validation|val) echo "$WAYMAX_TFEXAMPLE_VAL" ;;
    testing|test) echo "$WAYMAX_TFEXAMPLE_TEST" ;;
    *)
      echo "ERROR: ONLINE_SPLIT must be training, validation, or testing; got: $split" >&2
      exit 2
      ;;
  esac
}

normalize_waymax_path () {
  local split="$1"
  local path="${2:-}"
  if [[ -z "$path" ]]; then
    waymax_path_for_split "$split"
    return 0
  fi
  # Waymax dataloader should not receive shell glob patterns.  It should receive
  # a single sharded path such as validation_tfexample.tfrecord@150.
  if [[ "$path" == *"*"* ]]; then
    echo "WARN: online Waymax path contains a wildcard and will be replaced with @shard syntax: $path" >&2
    waymax_path_for_split "$split"
    return 0
  fi
  echo "$path"
}

print_waymax_path_hint () {
  local path="$1"
  echo "Resolved online Waymax split: $ONLINE_SPLIT"
  echo "Resolved online Waymax tf.Example path: $path"
  if [[ "$path" =~ ^(.+)@([0-9]+)$ ]]; then
    local prefix="${BASH_REMATCH[1]}"
    local shards="${BASH_REMATCH[2]}"
    local matched=0
    # This is diagnostic only.  Some filesystems/GCS-like paths may not be visible to bash globbing.
    shopt -s nullglob
    local files=("$prefix"-*-of-*)
    shopt -u nullglob
    matched="${#files[@]}"
    echo "Expected shard prefix: $prefix ; declared shards: $shards ; bash-visible matching files: $matched"
    if [[ "$matched" == "0" ]]; then
      echo "WARN: no local files matched the shard prefix by bash glob. If the files exist and TensorFlow can read @shards, this is harmless; otherwise check WOMD_ROOT." >&2
    fi
  fi
}

all_learned_outputs_ok () {
  local methods=("$@")
  local m
  for m in "${methods[@]}"; do
    json_ok "$OUT_ROOT/eval/learned_offline/${m}.json" || return 1
  done
  return 0
}

all_online_shards_ok () {
  local method="$1"
  local shard
  for shard in $(seq 0 $(( NUM_GPUS - 1 ))); do
    json_ok "$OUT_ROOT/eval/waymax/${method}_shard${shard}.json" || return 1
  done
  return 0
}

online_method_complete () {
  local method="$1"
  json_ok "$OUT_ROOT/eval/waymax/${method}_merged.json" || return 1
  all_online_shards_ok "$method" || return 1
  return 0
}

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

USER_PROVIDED_WAYMAX_TFEXAMPLE_PATH="${WAYMAX_TFEXAMPLE_PATH+x}"
if [[ -n "$USER_PROVIDED_WAYMAX_TFEXAMPLE_PATH" ]]; then
  WAYMAX_TFEXAMPLE_PATH="$(normalize_waymax_path "$ONLINE_SPLIT" "$WAYMAX_TFEXAMPLE_PATH")"
elif [[ "$ONLINE_SPLIT" == "validation" || "$ONLINE_SPLIT" == "val" ]]; then
  # Keep compatibility with older runs that overrode TFEXAMPLE_VAL, but convert globs to @150.
  WAYMAX_TFEXAMPLE_PATH="$(normalize_waymax_path "$ONLINE_SPLIT" "$TFEXAMPLE_VAL")"
else
  WAYMAX_TFEXAMPLE_PATH="$(waymax_path_for_split "$ONLINE_SPLIT")"
fi
print_waymax_path_hint "$WAYMAX_TFEXAMPLE_PATH"

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
if should_skip_existing && [[ -f "$OUT_ROOT/checkpoints/response/cowp_response_best.pt" ]]; then
  echo "  skip response: found $OUT_ROOT/checkpoints/response/cowp_response_best.pt"
else
  train_ddp response \
    --epochs "${RESPONSE_EPOCHS:-14}" \
    --lr "${RESPONSE_LR:-3e-4}" \
    --no-response-traj \
    --no-response-components \
    --output-dir "$OUT_ROOT/checkpoints/response"
fi

echo "[2/6] Train witness stage from best response checkpoint"
if should_skip_existing && [[ -f "$OUT_ROOT/checkpoints/witness/cowp_witness_best.pt" ]]; then
  echo "  skip witness: found $OUT_ROOT/checkpoints/witness/cowp_witness_best.pt"
else
  train_ddp witness \
    --epochs "${WITNESS_EPOCHS:-24}" \
    --lr "${WITNESS_LR:-3e-4}" \
    --resume "$OUT_ROOT/checkpoints/response/cowp_response_best.pt" \
    --output-dir "$OUT_ROOT/checkpoints/witness"
fi

echo "[3/6] Train planner with attached Waymax candidate outcome labels"
if should_skip_existing && [[ -f "$OUT_ROOT/checkpoints/planner_waymax/cowp_planner_best.pt" ]]; then
  echo "  skip planner: found $OUT_ROOT/checkpoints/planner_waymax/cowp_planner_best.pt"
else
  train_ddp planner \
    --epochs "${PLANNER_EPOCHS:-28}" \
    --lr "${PLANNER_LR:-2e-4}" \
    --resume "$OUT_ROOT/checkpoints/witness/cowp_witness_best.pt" \
    --with-waymax-outcome-labels \
    --output-dir "$OUT_ROOT/checkpoints/planner_waymax"
fi

CKPT="$OUT_ROOT/checkpoints/planner_waymax/cowp_planner_best.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: planner checkpoint not found: $CKPT" >&2
  exit 2
fi

echo "[4/6] Learned-offline evaluation on attached validation cache, methods run in parallel across GPUs"
LEARNED_METHODS=(idm_lattice conventional_safety planner_score_only soft_burden_cost_only universal_ncf cowp outcome_oracle)
if should_skip_existing && [[ "$FORCE_RERUN_LEARNED" != "1" ]] && all_learned_outputs_ok "${LEARNED_METHODS[@]}"; then
  echo "  skip learned-offline: all method JSON outputs exist and are valid"
else
  pids=()
  i=0
  for METHOD in "${LEARNED_METHODS[@]}"; do
    OUT_JSON="$OUT_ROOT/eval/learned_offline/${METHOD}.json"
    if should_skip_existing && [[ "$FORCE_RERUN_LEARNED" != "1" ]] && json_ok "$OUT_JSON"; then
      echo "  skip learned-offline/$METHOD: found valid $OUT_JSON"
      continue
    fi
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
        --output "$OUT_JSON"
    ) &
    pids+=("$!")
    i=$(( i + 1 ))
    if (( ${#pids[@]} >= NUM_GPUS )); then
      wait_group "${pids[@]}"
      pids=()
    fi
  done
  if (( ${#pids[@]} > 0 )); then
    wait_group "${pids[@]}"
  fi
fi

echo "[5/6] Online Waymax closed-loop evaluation on $ONLINE_SPLIT split, sharded across GPUs"
ONLINE_METHODS=(planner_score_only conventional_safety soft_burden_cost_only universal_ncf cowp)
PER_SHARD_SCENARIOS=$(( (TOTAL_ONLINE_SCENARIOS + NUM_GPUS - 1) / NUM_GPUS ))

for METHOD in "${ONLINE_METHODS[@]}"; do
  if should_skip_existing && [[ "$FORCE_RERUN_ONLINE" != "1" ]] && [[ "$FORCE_REMERGE_ONLINE" != "1" ]] && online_method_complete "$METHOD"; then
    echo "  skip online/$METHOD: all shard JSONs and merged JSON are valid"
    continue
  fi

  pids=()
  for SHARD in $(seq 0 $(( NUM_GPUS - 1 ))); do
    SHARD_JSON="$OUT_ROOT/eval/waymax/${METHOD}_shard${SHARD}.json"
    if should_skip_existing && [[ "$FORCE_RERUN_ONLINE" != "1" ]] && json_ok "$SHARD_JSON"; then
      echo "  skip online/$METHOD shard $SHARD: found valid $SHARD_JSON"
      continue
    fi
    GPU="$SHARD"
    (
      export CUDA_VISIBLE_DEVICES="$GPU"
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      python -m cowp.scripts.04_eval_closed_loop \
        --data-config configs/data.yaml \
        --label-config configs/label.yaml \
        --eval-config configs/eval.yaml \
        --mode waymax \
        --waymax-split "$ONLINE_SPLIT" \
        --tfexample-glob "$WAYMAX_TFEXAMPLE_PATH" \
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
        --output "$SHARD_JSON"
    ) &
    pids+=("$!")
  done
  if (( ${#pids[@]} > 0 )); then
    wait_group "${pids[@]}"
  fi

  if ! all_online_shards_ok "$METHOD"; then
    echo "ERROR: not all online shard outputs are present/valid for method=$METHOD; refusing to write merged metrics." >&2
    exit 2
  fi

  MERGED_JSON="$OUT_ROOT/eval/waymax/${METHOD}_merged.json"
  if should_skip_existing && [[ "$FORCE_REMERGE_ONLINE" != "1" ]] && json_ok "$MERGED_JSON" && [[ "$FORCE_RERUN_ONLINE" != "1" ]]; then
    echo "  skip merge online/$METHOD: found valid $MERGED_JSON"
    continue
  fi

  SHARD_PATHS=()
  for SHARD in $(seq 0 $(( NUM_GPUS - 1 ))); do
    SHARD_PATHS+=("$OUT_ROOT/eval/waymax/${METHOD}_shard${SHARD}.json")
  done

  python - "$MERGED_JSON" "${SHARD_PATHS[@]}" <<'PY'
import json, sys
from pathlib import Path
from cowp.waymax_eval.metrics_cowp import policy_diagnostic_episode_summary, policy_diagnostic_summary
from cowp.waymax_eval.metrics_standard import aggregate_waymax_standard_metrics
out = Path(sys.argv[1])
paths = [Path(p) for p in sys.argv[2:]]
payloads = []
for p in paths:
    if not p.exists():
        continue
    with p.open("r", encoding="utf-8") as f:
        payloads.append(json.load(f))
rollouts = []
for payload in payloads:
    rollouts.extend(payload.get("rollouts", []))
merged = dict(payloads[0]) if payloads else {}
merged["merged_from"] = [str(p) for p in paths]
merged["num_rollouts"] = len(rollouts)
merged["steps"] = [int(x.get("steps", 0)) for x in rollouts]
merged["policy_diagnostic_summary"] = policy_diagnostic_summary(rollouts)
merged["closed_loop_cowp_metric_summary"] = policy_diagnostic_episode_summary(rollouts)
if any("standard_metric_summary" in p for p in payloads):
    merged["standard_metric_summary"] = aggregate_waymax_standard_metrics(rollouts)
merged.pop("standard_metrics", None)
out.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
print(json.dumps({"merged_output": str(out), "num_rollouts": len(rollouts)}, indent=2))
PY
done

echo "[6/6] Done"
echo "Best planner checkpoint: $CKPT"
echo "Learned-offline outputs: $OUT_ROOT/eval/learned_offline"
echo "Waymax closed-loop outputs: $OUT_ROOT/eval/waymax"
