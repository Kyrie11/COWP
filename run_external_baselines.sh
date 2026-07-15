#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
TRAIN_CACHE="${TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax}"
VAL_CACHE="${VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax}"
WAYMAX_VAL="${WAYMAX_VAL:-$WOMD_ROOT/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
OUT_ROOT="${OUT_ROOT:-outputs/external_baselines}"
MODE="${MODE:-smoke}"        # smoke | full
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_OFFLINE_EVAL="${RUN_OFFLINE_EVAL:-1}"
RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-1}"
BASELINES="${BASELINES:-gameformer dtpp idm_lattice frenet_optimal state_lattice}"
DEVICE="${DEVICE:-auto}"
NO_PROGRESS="${NO_PROGRESS:-1}"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/logs" "$OUT_ROOT/checkpoints" "$OUT_ROOT/eval/learned_offline" "$OUT_ROOT/eval/waymax"

case "$MODE" in
  smoke)
    EPOCHS_GAMEFORMER="${EPOCHS_GAMEFORMER:-2}"
    EPOCHS_DTPP="${EPOCHS_DTPP:-2}"
    BATCH_GAMEFORMER="${BATCH_GAMEFORMER:-4}"
    BATCH_DTPP="${BATCH_DTPP:-4}"
    ONLINE_SCENARIOS="${ONLINE_SCENARIOS:-50}"
    ;;
  full)
    EPOCHS_GAMEFORMER="${EPOCHS_GAMEFORMER:-24}"
    EPOCHS_DTPP="${EPOCHS_DTPP:-24}"
    BATCH_GAMEFORMER="${BATCH_GAMEFORMER:-8}"
    BATCH_DTPP="${BATCH_DTPP:-8}"
    ONLINE_SCENARIOS="${ONLINE_SCENARIOS:-1000}"
    ;;
  *) echo "MODE must be smoke or full" >&2; exit 2 ;;
esac

RULE_BASELINES=" idm_lattice frenet_optimal state_lattice "
LEARNED_BASELINES=" gameformer dtpp "

is_rule_baseline() {
  [[ "$RULE_BASELINES" == *" $1 "* ]]
}

is_learned_baseline() {
  [[ "$LEARNED_BASELINES" == *" $1 "* ]]
}

progress_flag() {
  if [[ "$NO_PROGRESS" == "1" ]]; then
    echo "--no-progress"
  fi
}

run_logged() {
  local name="$1"; shift
  local log="$OUT_ROOT/logs/${name}.log"
  echo "[$name] -> $log"
  if ! "$@" > "$log" 2>&1; then
    local rc=$?
    echo "[$name] failed with exit code $rc. Last 120 log lines:" >&2
    tail -120 "$log" >&2 || true
    exit "$rc"
  fi
}

train_one() {
  local b="$1"
  local epochs batch
  if [[ "$b" == "gameformer" ]]; then
    epochs="$EPOCHS_GAMEFORMER"; batch="$BATCH_GAMEFORMER"
  elif [[ "$b" == "dtpp" ]]; then
    epochs="$EPOCHS_DTPP"; batch="$BATCH_DTPP"
  else
    echo "[$b] rule-based baseline: skip training"
    return 0
  fi
  run_logged "train_${b}" "$PYTHON_BIN" -m cowp.scripts.20_train_external_baseline \
    --baseline "$b" \
    --data-config configs/data.yaml \
    --label-config configs/label.yaml \
    --train-config configs/train.yaml \
    --cache-dir "$TRAIN_CACHE" \
    --val-cache-dir "$VAL_CACHE" \
    --output-dir "$OUT_ROOT/checkpoints/$b" \
    --epochs "$epochs" \
    --batch-size "$batch" \
    --num-workers "${NUM_WORKERS:-4}" \
    --device "$DEVICE" \
    $(progress_flag)
}

eval_one_offline_learned() {
  local b="$1"
  local ckpt="$OUT_ROOT/checkpoints/$b/external_${b}_best.pt"
  if [[ ! -s "$ckpt" ]]; then
    echo "Missing learned-baseline checkpoint: $ckpt" >&2
    exit 2
  fi
  run_logged "eval_offline_${b}" "$PYTHON_BIN" -m cowp.scripts.21_eval_external_baseline \
    --mode learned_offline \
    --data-config configs/data.yaml \
    --label-config configs/label.yaml \
    --eval-config configs/eval.yaml \
    --checkpoint "$ckpt" \
    --cache-dir "$VAL_CACHE" \
    --batch-size "${EVAL_BATCH:-16}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --device "$DEVICE" \
    --output "$OUT_ROOT/eval/learned_offline/${b}.json" \
    $(progress_flag)
}

eval_one_waymax_learned() {
  local b="$1"
  local ckpt="$OUT_ROOT/checkpoints/$b/external_${b}_best.pt"
  if [[ ! -s "$ckpt" ]]; then
    echo "Missing learned-baseline checkpoint: $ckpt" >&2
    exit 2
  fi
  run_logged "eval_waymax_${b}" "$PYTHON_BIN" -m cowp.scripts.21_eval_external_baseline \
    --mode waymax \
    --data-config configs/data.yaml \
    --label-config configs/label.yaml \
    --eval-config configs/eval.yaml \
    --checkpoint "$ckpt" \
    --waymax-split validation \
    --tfexample-glob "$WAYMAX_VAL" \
    --num-scenarios "$ONLINE_SCENARIOS" \
    --rollout-horizon-steps "${ROLLOUT_HORIZON_STEPS:-80}" \
    --waymax-action-mode "${WAYMAX_ACTION_MODE:-absolute_xy_yaw}" \
    --waymax-device "${WAYMAX_DEVICE:-gpu}" \
    --jax-preallocate false \
    --waymax-standard-metrics \
    --device "$DEVICE" \
    --output "$OUT_ROOT/eval/waymax/${b}.json" \
    $(progress_flag)
}

eval_one_offline_rule() {
  local b="$1"
  run_logged "eval_offline_${b}" "$PYTHON_BIN" -m cowp.scripts.22_eval_rule_baseline \
    --mode learned_offline \
    --baseline "$b" \
    --data-config configs/data.yaml \
    --label-config configs/label.yaml \
    --eval-config configs/eval.yaml \
    --cache-dir "$VAL_CACHE" \
    --batch-size "${EVAL_BATCH:-64}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --output "$OUT_ROOT/eval/learned_offline/${b}.json" \
    $(progress_flag)
}

eval_one_waymax_rule() {
  local b="$1"
  run_logged "eval_waymax_${b}" "$PYTHON_BIN" -m cowp.scripts.22_eval_rule_baseline \
    --mode waymax \
    --baseline "$b" \
    --data-config configs/data.yaml \
    --label-config configs/label.yaml \
    --eval-config configs/eval.yaml \
    --waymax-split validation \
    --tfexample-glob "$WAYMAX_VAL" \
    --num-scenarios "$ONLINE_SCENARIOS" \
    --rollout-horizon-steps "${ROLLOUT_HORIZON_STEPS:-80}" \
    --waymax-action-mode "${WAYMAX_ACTION_MODE:-absolute_xy_yaw}" \
    --waymax-device "${WAYMAX_DEVICE:-gpu}" \
    --jax-preallocate false \
    --waymax-standard-metrics \
    --output "$OUT_ROOT/eval/waymax/${b}.json" \
    $(progress_flag)
}

for b in $BASELINES; do
  if is_learned_baseline "$b"; then
    if [[ "$RUN_TRAIN" == "1" ]]; then train_one "$b"; fi
    if [[ "$RUN_OFFLINE_EVAL" == "1" ]]; then eval_one_offline_learned "$b"; fi
    if [[ "$RUN_ONLINE_EVAL" == "1" ]]; then eval_one_waymax_learned "$b"; fi
  elif is_rule_baseline "$b"; then
    if [[ "$RUN_TRAIN" == "1" ]]; then train_one "$b"; fi
    if [[ "$RUN_OFFLINE_EVAL" == "1" ]]; then eval_one_offline_rule "$b"; fi
    if [[ "$RUN_ONLINE_EVAL" == "1" ]]; then eval_one_waymax_rule "$b"; fi
  else
    echo "Unknown baseline: $b" >&2
    echo "Supported baselines: gameformer dtpp idm_lattice frenet_optimal state_lattice" >&2
    exit 2
  fi
  echo "[$b] done"
done
