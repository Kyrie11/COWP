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

# Progress/logging defaults: show tqdm bars in terminal and also keep log files.
NO_PROGRESS="${NO_PROGRESS:-0}"
LOG_EVERY="${LOG_EVERY:-0}"
RUN_STREAMED_LOGS="${RUN_STREAMED_LOGS:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# Speed/resource defaults for learned baselines.
AMP="${AMP:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DEVICE="${DEVICE:-auto}"
CUDA_SPLIT_LEARNED="${CUDA_SPLIT_LEARNED:-1}"
PARALLEL_LEARNED_TRAIN="${PARALLEL_LEARNED_TRAIN:-1}"
GAMEFORMER_GPU="${GAMEFORMER_GPU:-0}"
DTPP_GPU="${DTPP_GPU:-1}"
WAYMAX_GPU="${WAYMAX_GPU:-0}"
MAX_NEIGHBORS="${MAX_NEIGHBORS:-10}"
MAX_CANDIDATES_GAMEFORMER="${MAX_CANDIDATES_GAMEFORMER:-30}"
MAX_CANDIDATES_DTPP="${MAX_CANDIDATES_DTPP:-30}"

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

is_rule_baseline() { [[ "$RULE_BASELINES" == *" $1 "* ]]; }
is_learned_baseline() { [[ "$LEARNED_BASELINES" == *" $1 "* ]]; }

contains_baseline() {
  local target="$1"
  for b in $BASELINES; do
    [[ "$b" == "$target" ]] && return 0
  done
  return 1
}

progress_args() {
  if [[ "$NO_PROGRESS" == "1" ]]; then
    printf '%s\n' "--no-progress"
  fi
}

amp_args() {
  if [[ "$AMP" == "1" ]]; then
    printf '%s\n' "--amp"
  fi
}

baseline_gpu() {
  case "$1" in
    gameformer) printf '%s\n' "$GAMEFORMER_GPU" ;;
    dtpp) printf '%s\n' "$DTPP_GPU" ;;
    *) printf '%s\n' "$WAYMAX_GPU" ;;
  esac
}

baseline_position() {
  case "$1" in
    gameformer) printf '%s\n' "0" ;;
    dtpp) printf '%s\n' "1" ;;
    *) printf '%s\n' "2" ;;
  esac
}

baseline_device() {
  local b="$1"
  if is_learned_baseline "$b" && [[ "$CUDA_SPLIT_LEARNED" == "1" && "$DEVICE" == "auto" ]]; then
    printf '%s\n' "cuda:0"
  else
    printf '%s\n' "$DEVICE"
  fi
}

baseline_max_candidates() {
  case "$1" in
    gameformer) printf '%s\n' "$MAX_CANDIDATES_GAMEFORMER" ;;
    dtpp) printf '%s\n' "$MAX_CANDIDATES_DTPP" ;;
    *) printf '%s\n' "30" ;;
  esac
}

json_valid() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" <<'PY' >/dev/null 2>&1
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    json.load(f)
PY
}

train_done() {
  local b="$1"
  local epochs="$2"
  local dir="$OUT_ROOT/checkpoints/$b"
  local best="$dir/external_${b}_best.pt"
  local hist="$dir/external_${b}_history.json"
  local final_epoch="$dir/external_${b}_epoch${epochs}.pt"
  [[ "$SKIP_COMPLETED" == "1" && "$FORCE_RERUN" != "1" ]] || return 1
  [[ -s "$best" && -s "$hist" && -s "$final_epoch" ]] || return 1
  "$PYTHON_BIN" - "$hist" "$epochs" <<'PY'
import json, sys
hist_path, target = sys.argv[1], int(sys.argv[2])
try:
    with open(hist_path, 'r', encoding='utf-8') as f:
        hist = json.load(f)
    max_epoch = max((int(r.get('epoch', 0)) for r in hist), default=0)
except Exception:
    sys.exit(1)
sys.exit(0 if max_epoch >= target else 1)
PY
}

eval_done() {
  local output="$1"
  local mode="$2"
  local b="$3"
  local expected_rollouts="${4:-}"
  [[ "$SKIP_COMPLETED" == "1" && "$FORCE_RERUN" != "1" ]] || return 1
  [[ -s "$output" ]] || return 1
  "$PYTHON_BIN" - "$output" "$mode" "$b" "$expected_rollouts" <<'PY'
import json, sys
path, mode, baseline, expected = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
except Exception:
    sys.exit(1)
if payload.get('mode') != mode:
    sys.exit(1)
if mode == 'learned_offline':
    metrics = payload.get(baseline)
    if not isinstance(metrics, dict) or int(metrics.get('num_scenes', 0) or 0) <= 0:
        sys.exit(1)
elif mode == 'waymax':
    if payload.get('baseline') != baseline:
        sys.exit(1)
    n = int(payload.get('num_rollouts', 0) or 0)
    if n <= 0:
        sys.exit(1)
    if expected and expected not in ('None', '0') and n < int(expected):
        sys.exit(1)
sys.exit(0)
PY
}

run_logged_env() {
  local name="$1"; shift
  local gpu="$1"; shift
  local pos="$1"; shift
  local log="$OUT_ROOT/logs/${name}.log"
  echo "[$name] -> $log"
  local rc=0
  if [[ "$RUN_STREAMED_LOGS" == "1" ]]; then
    if [[ -n "$gpu" && "$CUDA_SPLIT_LEARNED" == "1" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" COWP_TQDM_POSITION="$pos" PYTHONUNBUFFERED=1 "$@" > >(tee "$log") 2> >(tee -a "$log" >&2) || rc=$?
    else
      COWP_TQDM_POSITION="$pos" PYTHONUNBUFFERED=1 "$@" > >(tee "$log") 2> >(tee -a "$log" >&2) || rc=$?
    fi
  else
    if [[ -n "$gpu" && "$CUDA_SPLIT_LEARNED" == "1" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" COWP_TQDM_POSITION="$pos" PYTHONUNBUFFERED=1 "$@" > "$log" 2>&1 || rc=$?
    else
      COWP_TQDM_POSITION="$pos" PYTHONUNBUFFERED=1 "$@" > "$log" 2>&1 || rc=$?
    fi
  fi
  if [[ "$rc" -ne 0 ]]; then
    echo "[$name] failed with exit code $rc. Last 120 log lines:" >&2
    tail -120 "$log" >&2 || true
    exit "$rc"
  fi
}

wait_all() {
  local rc=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  return "$rc"
}

train_one() {
  local b="$1"
  local epochs batch gpu pos dev max_candidates
  if [[ "$b" == "gameformer" ]]; then
    epochs="$EPOCHS_GAMEFORMER"; batch="$BATCH_GAMEFORMER"
  elif [[ "$b" == "dtpp" ]]; then
    epochs="$EPOCHS_DTPP"; batch="$BATCH_DTPP"
  else
    echo "[$b] rule-based baseline: skip training"
    return 0
  fi
  gpu="$(baseline_gpu "$b")"
  pos="$(baseline_position "$b")"
  dev="$(baseline_device "$b")"
  max_candidates="$(baseline_max_candidates "$b")"
  if train_done "$b" "$epochs"; then
    echo "[train_${b}] skip: checkpoint/history already reached ${epochs} epochs"
    return 0
  fi
  local cmd=("$PYTHON_BIN" -u -m cowp.scripts.20_train_external_baseline
    --baseline "$b"
    --data-config configs/data.yaml
    --label-config configs/label.yaml
    --train-config configs/train.yaml
    --cache-dir "$TRAIN_CACHE"
    --val-cache-dir "$VAL_CACHE"
    --output-dir "$OUT_ROOT/checkpoints/$b"
    --epochs "$epochs"
    --batch-size "$batch"
    --num-workers "$NUM_WORKERS"
    --prefetch-factor "$PREFETCH_FACTOR"
    --device "$dev"
    --max-neighbors "$MAX_NEIGHBORS"
    --max-candidates "$max_candidates"
    --log-every "$LOG_EVERY")
  while IFS= read -r arg; do [[ -n "$arg" ]] && cmd+=("$arg"); done < <(amp_args)
  while IFS= read -r arg; do [[ -n "$arg" ]] && cmd+=("$arg"); done < <(progress_args)
  run_logged_env "train_${b}" "$gpu" "$pos" "${cmd[@]}"
}

eval_one_offline_learned() {
  local b="$1"
  local ckpt="$OUT_ROOT/checkpoints/$b/external_${b}_best.pt"
  local gpu pos dev max_candidates
  gpu="$(baseline_gpu "$b")"; pos="$(baseline_position "$b")"; dev="$(baseline_device "$b")"; max_candidates="$(baseline_max_candidates "$b")"
  if [[ ! -s "$ckpt" ]]; then
    echo "Missing learned-baseline checkpoint: $ckpt" >&2
    exit 2
  fi
  local output="$OUT_ROOT/eval/learned_offline/${b}.json"
  if eval_done "$output" "learned_offline" "$b"; then
    echo "[eval_offline_${b}] skip: valid output already exists at $output"
    return 0
  fi
  local cmd=("$PYTHON_BIN" -u -m cowp.scripts.21_eval_external_baseline
    --mode learned_offline
    --data-config configs/data.yaml
    --label-config configs/label.yaml
    --eval-config configs/eval.yaml
    --checkpoint "$ckpt"
    --cache-dir "$VAL_CACHE"
    --batch-size "${EVAL_BATCH:-16}"
    --num-workers "$NUM_WORKERS"
    --device "$dev"
    --max-neighbors "$MAX_NEIGHBORS"
    --max-candidates "$max_candidates"
    --output "$output"
    --log-every "$LOG_EVERY")
  while IFS= read -r arg; do [[ -n "$arg" ]] && cmd+=("$arg"); done < <(progress_args)
  run_logged_env "eval_offline_${b}" "$gpu" "$pos" "${cmd[@]}"
}

eval_one_waymax_learned() {
  local b="$1"
  local ckpt="$OUT_ROOT/checkpoints/$b/external_${b}_best.pt"
  local gpu pos dev
  gpu="$(baseline_gpu "$b")"; pos="$(baseline_position "$b")"; dev="$(baseline_device "$b")"
  if [[ ! -s "$ckpt" ]]; then
    echo "Missing learned-baseline checkpoint: $ckpt" >&2
    exit 2
  fi
  local output="$OUT_ROOT/eval/waymax/${b}.json"
  if eval_done "$output" "waymax" "$b" "$ONLINE_SCENARIOS"; then
    echo "[eval_waymax_${b}] skip: valid output already exists at $output"
    return 0
  fi
  local cmd=("$PYTHON_BIN" -u -m cowp.scripts.21_eval_external_baseline
    --mode waymax
    --data-config configs/data.yaml
    --label-config configs/label.yaml
    --eval-config configs/eval.yaml
    --checkpoint "$ckpt"
    --waymax-split validation
    --tfexample-glob "$WAYMAX_VAL"
    --num-scenarios "$ONLINE_SCENARIOS"
    --rollout-horizon-steps "${ROLLOUT_HORIZON_STEPS:-80}"
    --waymax-action-mode "${WAYMAX_ACTION_MODE:-absolute_xy_yaw}"
    --waymax-device "${WAYMAX_DEVICE:-gpu}"
    --jax-preallocate false
    --waymax-standard-metrics
    --device "$dev"
    --output "$output"
    --log-every "$LOG_EVERY")
  while IFS= read -r arg; do [[ -n "$arg" ]] && cmd+=("$arg"); done < <(progress_args)
  run_logged_env "eval_waymax_${b}" "$gpu" "$pos" "${cmd[@]}"
}

eval_one_offline_rule() {
  local b="$1"
  local output="$OUT_ROOT/eval/learned_offline/${b}.json"
  if eval_done "$output" "learned_offline" "$b"; then
    echo "[eval_offline_${b}] skip: valid output already exists at $output"
    return 0
  fi
  local cmd=("$PYTHON_BIN" -u -m cowp.scripts.22_eval_rule_baseline
    --mode learned_offline
    --baseline "$b"
    --data-config configs/data.yaml
    --label-config configs/label.yaml
    --eval-config configs/eval.yaml
    --cache-dir "$VAL_CACHE"
    --batch-size "${EVAL_BATCH:-64}"
    --num-workers "$NUM_WORKERS"
    --output "$output"
    --log-every "$LOG_EVERY")
  while IFS= read -r arg; do [[ -n "$arg" ]] && cmd+=("$arg"); done < <(progress_args)
  run_logged_env "eval_offline_${b}" "" "2" "${cmd[@]}"
}

eval_one_waymax_rule() {
  local b="$1"
  local output="$OUT_ROOT/eval/waymax/${b}.json"
  if eval_done "$output" "waymax" "$b" "$ONLINE_SCENARIOS"; then
    echo "[eval_waymax_${b}] skip: valid output already exists at $output"
    return 0
  fi
  local cmd=("$PYTHON_BIN" -u -m cowp.scripts.22_eval_rule_baseline
    --mode waymax
    --baseline "$b"
    --data-config configs/data.yaml
    --label-config configs/label.yaml
    --eval-config configs/eval.yaml
    --waymax-split validation
    --tfexample-glob "$WAYMAX_VAL"
    --num-scenarios "$ONLINE_SCENARIOS"
    --rollout-horizon-steps "${ROLLOUT_HORIZON_STEPS:-80}"
    --waymax-action-mode "${WAYMAX_ACTION_MODE:-absolute_xy_yaw}"
    --waymax-device "${WAYMAX_DEVICE:-gpu}"
    --jax-preallocate false
    --waymax-standard-metrics
    --output "$output"
    --log-every "$LOG_EVERY")
  while IFS= read -r arg; do [[ -n "$arg" ]] && cmd+=("$arg"); done < <(progress_args)
  run_logged_env "eval_waymax_${b}" "$WAYMAX_GPU" "2" "${cmd[@]}"
}

LEARNED_TRAIN_ALREADY=0
if [[ "$RUN_TRAIN" == "1" && "$PARALLEL_LEARNED_TRAIN" == "1" ]]; then
  pids=()
  for b in gameformer dtpp; do
    if contains_baseline "$b"; then
      train_one "$b" &
      pids+=("$!")
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    wait_all "${pids[@]}"
    LEARNED_TRAIN_ALREADY=1
  fi
fi

for b in $BASELINES; do
  if is_learned_baseline "$b"; then
    if [[ "$RUN_TRAIN" == "1" && "$LEARNED_TRAIN_ALREADY" != "1" ]]; then train_one "$b"; fi
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
