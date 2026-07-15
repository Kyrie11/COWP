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
BASELINES="${BASELINES:-gameformer dtpp}"
DEVICE="${DEVICE:-auto}"

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

run_logged() {
  local name="$1"; shift
  local log="$OUT_ROOT/logs/${name}.log"
  echo "[$name] -> $log"
  "$@" > "$log" 2>&1
}

train_one() {
  local b="$1"
  local epochs batch
  if [[ "$b" == "gameformer" ]]; then
    epochs="$EPOCHS_GAMEFORMER"; batch="$BATCH_GAMEFORMER"
  else
    epochs="$EPOCHS_DTPP"; batch="$BATCH_DTPP"
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
    --device "$DEVICE"
}

eval_one_offline() {
  local b="$1"
  local ckpt="$OUT_ROOT/checkpoints/$b/external_${b}_best.pt"
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
    --output "$OUT_ROOT/eval/learned_offline/${b}.json"
}

eval_one_waymax() {
  local b="$1"
  local ckpt="$OUT_ROOT/checkpoints/$b/external_${b}_best.pt"
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
    --output "$OUT_ROOT/eval/waymax/${b}.json"
}

for b in $BASELINES; do
  if [[ "$RUN_TRAIN" == "1" ]]; then train_one "$b"; fi
  if [[ "$RUN_OFFLINE_EVAL" == "1" ]]; then eval_one_offline "$b"; fi
  if [[ "$RUN_ONLINE_EVAL" == "1" ]]; then eval_one_waymax "$b"; fi
  echo "[$b] done"
done
