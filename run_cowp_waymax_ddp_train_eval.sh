#!/usr/bin/env bash
set -euo pipefail

# RC-NCF/COWP end-to-end pipeline: outcome replay -> staged DDP training ->
# held-out calibration -> learned-offline and Waymax closed-loop evaluation.
# Run from repository root. Every expensive stage is resume-safe.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------- User paths -------------------------
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
BASE_TRAIN_CACHE="${BASE_TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train}"
BASE_VAL_CACHE="${BASE_VAL_CACHE:-$COWP_ROOT/tensor_cache_val}"
TRAIN_CACHE="${TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax_rc24}"
VAL_CACHE="${VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax_rc24}"
WAYMAX_VAL="${WAYMAX_VAL:-$WOMD_ROOT/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
OUT_ROOT="${OUT_ROOT:-outputs/rc_ncf_full}"
REPLAY_ROOT="${REPLAY_ROOT:-$OUT_ROOT/outcome_replay}"

# ------------------------- Runtime knobs -------------------------
NUM_GPUS=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

RUN_TESTS="${RUN_TESTS:-1}"
RUN_OUTCOME_REPLAY="${RUN_OUTCOME_REPLAY:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_REPLAY="${FORCE_REPLAY:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

# 24/64 candidate coverage is a practical first pass. For the final paper run,
# set OUTCOME_CANDIDATES=0 to replay all candidates if compute permits.
OUTCOME_CANDIDATES="${OUTCOME_CANDIDATES:-24}"
TOTAL_ONLINE_SCENARIOS="${TOTAL_ONLINE_SCENARIOS:-1000}"
PER_GPU_BATCH_NATURAL="${PER_GPU_BATCH_NATURAL:-24}"
PER_GPU_BATCH_RESPONSE="${PER_GPU_BATCH_RESPONSE:-12}"
PER_GPU_BATCH_WITNESS="${PER_GPU_BATCH_WITNESS:-20}"
PER_GPU_BATCH_PLANNER="${PER_GPU_BATCH_PLANNER:-16}"
EVAL_BATCH="${EVAL_BATCH:-64}"
TRAIN_WORKERS="${TRAIN_WORKERS:-6}"
PREFETCH="${PREFETCH:-1}"

mkdir -p "$OUT_ROOT/checkpoints" "$OUT_ROOT/eval/learned_offline" "$OUT_ROOT/eval/waymax" "$REPLAY_ROOT"

json_ok() {
  local f="$1"
  [[ -s "$f" ]] && python - "$f" <<'PY' >/dev/null 2>&1
import json, sys
json.load(open(sys.argv[1], encoding="utf-8"))
PY
}

wait_all() {
  local status=0 pid
  for pid in "$@"; do wait "$pid" || status=$?; done
  return "$status"
}

run_two_gpu_replay() {
  local split="$1" cache="$2" prefix="$3"
  local pids=()
  for shard in 0 1; do
    local out="${prefix}_shard${shard}.jsonl"
    if [[ "$FORCE_REPLAY" != 1 && -s "$out" ]]; then
      echo "[replay/$split] keep existing shard $shard: $out"
      continue
    fi
    (
      export CUDA_VISIBLE_DEVICES="$shard"
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      python -u -m cowp.scripts.13_replay_waymax_candidates \
        --data-config configs/data.yaml \
        --label-config configs/label.yaml \
        --eval-config configs/eval.yaml \
        --cache-dir "$cache" \
        --state-source cache \
        --split "$split" \
        --outcomes-jsonl "$out" \
        --candidate-selection balanced \
        --max-candidates-per-scene "$OUTCOME_CANDIDATES" \
        --rollout-horizon-steps 80 \
        --waymax-device gpu \
        --waymax-action-mode absolute_xy_yaw \
        --metric-set safety_logdiv \
        --metric-eval-mode adaptive \
        --metric-eval-interval 2 \
        --num-shards 2 \
        --shard-index "$shard" \
        --retry-failed-existing
    ) & pids+=("$!")
  done
  ((${#pids[@]} == 0)) || wait_all "${pids[@]}"
}

attach_outcomes() {
  local base_cache="$1" output_cache="$2" prefix="$3"
  python -m cowp.scripts.12_attach_waymax_candidate_outcomes \
    --cache-dir "$base_cache" \
    --output-dir "$output_cache" \
    --outcomes-jsonl "${prefix}_shard0.jsonl" "${prefix}_shard1.jsonl" \
    --repair-outcomes-jsonl \
    --skip-existing
  python -m cowp.scripts.14_verify_waymax_cache --cache-dir "$output_cache"
}

train_stage() {
  local stage="$1" epochs="$2" lr="$3" batch="$4" output="$5" resume="${6:-}"
  local best="$output/cowp_${stage}_best.pt"
  if [[ "$FORCE_TRAIN" != 1 && -s "$best" ]]; then
    echo "[train/$stage] keep existing: $best"
    return
  fi
  local args=(
    torchrun --standalone --nproc_per_node=2 -m cowp.scripts.03_train
    --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml
    --cache-dir "$TRAIN_CACHE" --val-cache-dir "$VAL_CACHE"
    --stage "$stage" --epochs "$epochs" --lr "$lr" --batch-size "$batch"
    --num-workers "$TRAIN_WORKERS" --prefetch-factor "$PREFETCH"
    --amp --fused-adamw --pin-memory --val-every 1 --output-dir "$output"
  )
  [[ -z "$resume" ]] || args+=(--resume "$resume")
  [[ "$stage" != response ]] || args+=(--no-response-traj)
  [[ "$stage" != planner ]] || args+=(--with-waymax-outcome-labels)
  "${args[@]}"
}

run_learned_eval() {
  local method="$1" gpu="$2" threshold="$3" output="$4"
  if [[ "$FORCE_EVAL" != 1 ]] && json_ok "$output"; then
    echo "[learned/$method] keep existing: $output"
    return
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -m cowp.scripts.04_eval_closed_loop \
      --data-config configs/data.yaml --label-config configs/label.yaml --eval-config configs/eval.yaml \
      --cache-dir "$VAL_CACHE" --mode learned_offline --method "$method" \
      --checkpoint "$CKPT" --batch-size "$EVAL_BATCH" --device cuda \
      --witness-threshold "$threshold" --ncf-gate-mode priority \
      --priority-hard-threshold 0.55 --secondary-witness-threshold 0.85 \
      --secondary-opr-alpha 0.10 --soft-ncf-penalty 1.5 \
      --offline-fallback stop_like --adaptive-frontier-margin 0.15 \
      --outcome-risk-penalty 0.5 --outcome-risk-threshold 1.10 \
      --output "$output"
  )
}

run_waymax_method() {
  local method="$1" threshold="$2"
  local pids=() shard gpu out
  local per_shard=$(( (TOTAL_ONLINE_SCENARIOS + NUM_GPUS - 1) / NUM_GPUS ))
  for shard in 0 1; do
    gpu="$shard"
    out="$OUT_ROOT/eval/waymax/${method}_shard${shard}.json"
    if [[ "$FORCE_EVAL" != 1 ]] && json_ok "$out"; then
      echo "[waymax/$method] keep existing shard $shard"
      continue
    fi
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      python -m cowp.scripts.04_eval_closed_loop \
        --data-config configs/data.yaml --label-config configs/label.yaml --eval-config configs/eval.yaml \
        --mode waymax --waymax-split validation --tfexample-glob "$WAYMAX_VAL" \
        --method "$method" --checkpoint "$CKPT" \
        --num-scenarios "$per_shard" --num-shards 2 --shard-index "$shard" \
        --rollout-horizon-steps 80 --waymax-standard-metrics \
        --waymax-device gpu --jax-visible-devices 0 --jax-preallocate false \
        --waymax-action-mode absolute_xy_yaw --device cuda \
        --witness-threshold "$threshold" --ncf-gate-mode priority \
        --priority-hard-threshold 0.55 --secondary-witness-threshold 0.85 \
        --secondary-opr-alpha 0.10 --soft-ncf-penalty 1.5 \
        --adaptive-frontier-margin 0.15 \
        --outcome-risk-penalty 0.5 --outcome-risk-threshold 1.10 \
        --clear-accelerator-cache --output "$out"
    ) & pids+=("$!")
  done
  ((${#pids[@]} == 0)) || wait_all "${pids[@]}"
  python -m cowp.scripts.17_merge_waymax_shards \
    --output "$OUT_ROOT/eval/waymax/${method}_merged.json" \
    "$OUT_ROOT/eval/waymax/${method}_shard0.json" \
    "$OUT_ROOT/eval/waymax/${method}_shard1.json"
}

# ------------------------- Pipeline -------------------------
if [[ "$RUN_TESTS" == 1 ]]; then
  echo "[0/6] Regression tests"
  pytest --rootdir="$REPO_ROOT" -q
fi

if [[ "$RUN_OUTCOME_REPLAY" == 1 ]]; then
  echo "[1/6] Replay 24 balanced candidates with safety + log-divergence labels"
  [[ -d "$BASE_TRAIN_CACHE" && -d "$BASE_VAL_CACHE" ]] || { echo "Base caches not found" >&2; exit 2; }
  run_two_gpu_replay training "$BASE_TRAIN_CACHE" "$REPLAY_ROOT/train_rc24"
  attach_outcomes "$BASE_TRAIN_CACHE" "$TRAIN_CACHE" "$REPLAY_ROOT/train_rc24"
  run_two_gpu_replay validation "$BASE_VAL_CACHE" "$REPLAY_ROOT/val_rc24"
  attach_outcomes "$BASE_VAL_CACHE" "$VAL_CACHE" "$REPLAY_ROOT/val_rc24"
fi

[[ -d "$TRAIN_CACHE" && -d "$VAL_CACHE" ]] || { echo "Attached RC caches not found: $TRAIN_CACHE / $VAL_CACHE" >&2; exit 2; }

if [[ "$RUN_TRAIN" == 1 ]]; then
  echo "[2/6] Cross-world staged training on two GPUs"
  train_stage natural 20 3e-4 "$PER_GPU_BATCH_NATURAL" "$OUT_ROOT/checkpoints/natural"
  NAT="$OUT_ROOT/checkpoints/natural/cowp_natural_best.pt"
  train_stage response 16 2.5e-4 "$PER_GPU_BATCH_RESPONSE" "$OUT_ROOT/checkpoints/response" "$NAT"
  RESP="$OUT_ROOT/checkpoints/response/cowp_response_best.pt"
  train_stage witness 28 2e-4 "$PER_GPU_BATCH_WITNESS" "$OUT_ROOT/checkpoints/witness" "$RESP"
  WIT="$OUT_ROOT/checkpoints/witness/cowp_witness_best.pt"
  train_stage planner 32 1.5e-4 "$PER_GPU_BATCH_PLANNER" "$OUT_ROOT/checkpoints/planner" "$WIT"
fi

CKPT="$OUT_ROOT/checkpoints/planner/cowp_planner_best.pt"
[[ -s "$CKPT" ]] || { echo "Planner checkpoint not found: $CKPT" >&2; exit 2; }

if [[ "$RUN_EVAL" == 1 ]]; then
  echo "[3/6] Validation threshold calibration"
  CAL_JSON="$OUT_ROOT/eval/learned_offline/cowp_threshold_sweep.json"
  if [[ "$FORCE_EVAL" == 1 ]] || ! json_ok "$CAL_JSON"; then
    export CUDA_VISIBLE_DEVICES=0
    python -m cowp.scripts.04_eval_closed_loop \
      --data-config configs/data.yaml --label-config configs/label.yaml --eval-config configs/eval.yaml \
      --cache-dir "$VAL_CACHE" --mode learned_offline --method cowp --checkpoint "$CKPT" \
      --batch-size "$EVAL_BATCH" --device cuda --witness-threshold 0.5 \
      --witness-threshold-sweep 0.20,0.30,0.40,0.50,0.60,0.70,0.80 \
      --ncf-gate-mode priority --priority-hard-threshold 0.55 \
      --offline-fallback stop_like --adaptive-frontier-margin 0.15 \
      --outcome-risk-penalty 0.5 --outcome-risk-threshold 1.10 --output "$CAL_JSON"
  fi
  python -m cowp.scripts.18_calibrate_witness_threshold \
    --input "$CAL_JSON" --output "$OUT_ROOT/eval/witness_calibration.json" \
    --min-ncf-recall 0.90 --max-fallback 0.25
  WITNESS_THRESHOLD="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["witness_threshold"])' "$OUT_ROOT/eval/witness_calibration.json")"
  echo "Calibrated witness threshold: $WITNESS_THRESHOLD"

  echo "[4/6] Learned-offline baselines in two-GPU waves"
  METHODS=(idm_lattice conventional_safety planner_score_only soft_burden_cost_only universal_ncf cowp outcome_oracle)
  for ((i=0; i<${#METHODS[@]}; i+=2)); do
    pids=()
    for offset in 0 1; do
      j=$((i+offset)); ((j < ${#METHODS[@]})) || continue
      m="${METHODS[$j]}"
      run_learned_eval "$m" "$offset" "$WITNESS_THRESHOLD" "$OUT_ROOT/eval/learned_offline/${m}.json" & pids+=("$!")
    done
    wait_all "${pids[@]}"
  done

  echo "[5/6] Waymax closed-loop methods, each sharded across both GPUs"
  ONLINE_METHODS=(planner_score_only conventional_safety soft_burden_cost_only universal_ncf cowp)
  for m in "${ONLINE_METHODS[@]}"; do run_waymax_method "$m" "$WITNESS_THRESHOLD"; done
fi

echo "[6/6] Complete"
echo "Checkpoint: $CKPT"
echo "Learned-offline: $OUT_ROOT/eval/learned_offline"
echo "Waymax: $OUT_ROOT/eval/waymax"
