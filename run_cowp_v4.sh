#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v4}"
TRAIN_CACHE="${TRAIN_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax}"
VAL_CACHE="${VAL_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax}"
WAYMAX_VAL="${WAYMAX_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"

EPOCH_PLANNER="${EPOCH_PLANNER:-32}"
BATCH_PLANNER="${BATCH_PLANNER:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_LEARNED_EVAL="${RUN_LEARNED_EVAL:-1}"
RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-1}"
LEARNED_METHODS="${LEARNED_METHODS:-planner_score_only conventional_safety soft_burden_cost_only universal_ncf cowp}"
ONLINE_METHODS="${ONLINE_METHODS:-planner_score_only conventional_safety cowp}"
LEARNED_EVAL_DEVICE="${LEARNED_EVAL_DEVICE:-cpu}"
LEARNED_EVAL_BATCH="${LEARNED_EVAL_BATCH:-1}"
TOTAL_ONLINE_SCENARIOS="${TOTAL_ONLINE_SCENARIOS:-300}"
NUM_WAYMAX_SHARDS="${NUM_WAYMAX_SHARDS:-2}"
WAYMAX_GPUS="${WAYMAX_GPUS:-0 1}"
WAYMAX_DEVICE="${WAYMAX_DEVICE:-gpu}"
POLICY_DEVICE="${POLICY_DEVICE:-auto}"
WITNESS_THRESHOLD="${WITNESS_THRESHOLD:-0.05}"
NCF_GATE_MODE="${NCF_GATE_MODE:-priority}"
OUTCOME_RISK_PENALTY="${OUTCOME_RISK_PENALTY:-1.0}"
OUTCOME_RISK_THRESHOLD="${OUTCOME_RISK_THRESHOLD:-1.10}"
DETACH="${DETACH:-0}"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/logs" "$OUT_ROOT/checkpoints/planner" "$OUT_ROOT/eval/learned_offline" "$OUT_ROOT/eval/waymax" "$OUT_ROOT/configs"

if [[ "$DETACH" == "1" && "${COWP_V4_DETACHED:-0}" != "1" ]]; then
  export COWP_V4_DETACHED=1
  nohup bash "$0" > "$OUT_ROOT/logs/driver.nohup.log" 2>&1 &
  echo "[cowp_v4] detached pid=$! log=$OUT_ROOT/logs/driver.nohup.log"
  exit 0
fi

LABEL_RUNTIME="$OUT_ROOT/configs/label_cowp_v4.yaml"
TRAIN_RUNTIME="$OUT_ROOT/configs/train_cowp_v4.yaml"
cp configs/label.yaml "$LABEL_RUNTIME"
cp configs/train.yaml "$TRAIN_RUNTIME"
"$PYTHON_BIN" - "$LABEL_RUNTIME" "$TRAIN_RUNTIME" <<'PY'
import sys, yaml
label_path, train_path = sys.argv[1], sys.argv[2]
with open(label_path, 'r', encoding='utf-8') as f:
    label = yaml.safe_load(f)
p = label.setdefault('planning', {})
p.update({
    'candidate_frontier_keep_fraction': 0.22,
    'candidate_frontier_min_keep': 2,
    'candidate_frontier_max_keep': 5,
    'candidate_frontier_tie_eps': 0.002,
    'candidate_frontier_min_progress_ratio': 0.12,
    'candidate_frontier_min_progress_m': 1.0,
    'candidate_frontier_score_slack': 0.85,
    'candidate_frontier_max_action_risk': 0.90,
    'candidate_certificate_penalty': 3.0,
    'candidate_pair_risk_mix': 0.10,
    'candidate_pressure_prior_mix': 0.30,
    'candidate_rule_risk_mix': 2.0,
    'candidate_action_risk_mix': 3.0,
    'candidate_outcome_risk_mix': 1.25,
    'candidate_min_ncf_prob': 0.05,
    'candidate_max_false_safe_prob': 0.95,
    'decision_risk_min_std': 0.001,
    'decision_risk_min_spread': 0.001,
})
with open(label_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(label, f, sort_keys=False, allow_unicode=True)
with open(train_path, 'r', encoding='utf-8') as f:
    train = yaml.safe_load(f)
w = train.setdefault('loss_weights', {})
w.update({
    'candidate_certificate': 4.0,
    'candidate_certificate_false_safe': 3.5,
    'candidate_certificate_ncf': 2.0,
    'candidate_certificate_quality': 1.5,
    'candidate_certificate_rank': 3.0,
    'candidate_certificate_risk_bce': 4.0,
    'candidate_certificate_risk_rank': 4.0,
    'candidate_certificate_spread': 1.0,
    'candidate_cert_min_logit_spread': 0.65,
})
with open(train_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(train, f, sort_keys=False, allow_unicode=True)
PY

logrun() {
  local name="$1"; shift
  local log="$OUT_ROOT/logs/${name}.log"
  echo "[$name] -> $log"
  "$@" > >(tee "$log") 2> >(tee -a "$log" >&2)
}

latest_ckpt() {
  ls -1 "$OUT_ROOT"/checkpoints/planner/cowp_planner_epoch*.pt 2>/dev/null | sort -V | tail -1 || true
}

ckpt="${CKPT:-$(latest_ckpt)}"
if [[ "$RUN_TRAIN" == "1" ]]; then
  if [[ -n "$ckpt" && "$FORCE_TRAIN" != "1" ]]; then
    echo "[train] skip existing checkpoint: $ckpt"
  else
    resume_args=()
    if [[ -n "$ckpt" ]]; then
      resume_args=(--resume "$ckpt" --resume-training)
    fi
    logrun train_planner "$PYTHON_BIN" -u -m cowp.scripts.03_train \
      --data-config configs/data.yaml \
      --model-config configs/model.yaml \
      --label-config "$LABEL_RUNTIME" \
      --train-config "$TRAIN_RUNTIME" \
      --cache-dir "$TRAIN_CACHE" \
      --val-cache-dir "$VAL_CACHE" \
      --stage planner \
      --epochs "$EPOCH_PLANNER" \
      --batch-size "$BATCH_PLANNER" \
      --num-workers "$NUM_WORKERS" \
      --prefetch-factor "$PREFETCH_FACTOR" \
      --device auto \
      --output-dir "$OUT_ROOT/checkpoints/planner" \
      --with-waymax-outcome-labels \
      --amp \
      "${resume_args[@]}"
  fi
fi
ckpt="${CKPT:-$(latest_ckpt)}"
if [[ -z "$ckpt" ]]; then
  echo "No planner checkpoint found. Set CKPT=/path/to/cowp_planner_epochXXX.pt or enable training." >&2
  exit 2
fi

eval_done() {
  local path="$1" method="$2" mode="$3"
  [[ "$FORCE_EVAL" != "1" && -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" "$method" "$mode" <<'PY' >/dev/null 2>&1
import json, sys
path, method, mode = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, 'r', encoding='utf-8') as f: d=json.load(f)
assert d.get('mode') == mode
if mode == 'learned_offline':
    assert method in d and d[method].get('mode') == 'learned_offline' and int(d[method].get('num_scenes',0)) > 0
elif mode == 'waymax':
    assert d.get('method') == method and int(d.get('num_rollouts',0)) > 0
PY
}

if [[ "$RUN_LEARNED_EVAL" == "1" ]]; then
  for method in $LEARNED_METHODS; do
    out="$OUT_ROOT/eval/learned_offline/${method}.json"
    if eval_done "$out" "$method" learned_offline; then
      echo "[learned/$method] keep existing $out"
      continue
    fi
    # Run learned-offline sequentially and default to CPU to avoid CUDA OOM from
    # concurrent Waymax/JAX/PyTorch processes.  This is slower but makes the
    # mechanism diagnostics complete and reproducible.
    logrun "learned_${method}" "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
      --mode learned_offline \
      --method "$method" \
      --checkpoint "$ckpt" \
      --cache-dir "$VAL_CACHE" \
      --data-config configs/data.yaml \
      --label-config "$LABEL_RUNTIME" \
      --eval-config configs/eval.yaml \
      --device "$LEARNED_EVAL_DEVICE" \
      --batch-size "$LEARNED_EVAL_BATCH" \
      --witness-threshold "$WITNESS_THRESHOLD" \
      --ncf-gate-mode "$NCF_GATE_MODE" \
      --outcome-risk-penalty "$OUTCOME_RISK_PENALTY" \
      --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
      --output "$out" \
      --no-progress
  done
fi

if [[ "$RUN_ONLINE_EVAL" == "1" ]]; then
  scenarios_per_shard=$(( (TOTAL_ONLINE_SCENARIOS + NUM_WAYMAX_SHARDS - 1) / NUM_WAYMAX_SHARDS ))
  read -r -a gpu_arr <<< "$WAYMAX_GPUS"
  for method in $ONLINE_METHODS; do
    merged="$OUT_ROOT/eval/waymax/${method}_merged.json"
    if eval_done "$merged" "$method" waymax; then
      echo "[waymax/$method] keep existing $merged"
      continue
    fi
    pids=()
    for ((shard=0; shard<NUM_WAYMAX_SHARDS; shard++)); do
      gpu="${gpu_arr[$((shard % ${#gpu_arr[@]}))]}"
      out="$OUT_ROOT/eval/waymax/${method}_shard${shard}.json"
      log="$OUT_ROOT/logs/waymax_${method}_shard${shard}.log"
      echo "[waymax/$method shard=$shard gpu=$gpu] -> $log"
      (
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
          --mode waymax \
          --method "$method" \
          --checkpoint "$ckpt" \
          --cache-dir "$VAL_CACHE" \
          --data-config configs/data.yaml \
          --label-config "$LABEL_RUNTIME" \
          --eval-config configs/eval.yaml \
          --tfexample-glob "$WAYMAX_VAL" \
          --num-shards "$NUM_WAYMAX_SHARDS" \
          --shard-index "$shard" \
          --num-scenarios "$scenarios_per_shard" \
          --device "$POLICY_DEVICE" \
          --waymax-device "$WAYMAX_DEVICE" \
          --jax-visible-devices 0 \
          --jax-preallocate false \
          --clear-accelerator-cache \
          --waymax-standard-metrics \
          --witness-threshold "$WITNESS_THRESHOLD" \
          --ncf-gate-mode "$NCF_GATE_MODE" \
          --outcome-risk-penalty "$OUTCOME_RISK_PENALTY" \
          --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
          --output "$out" \
          --no-progress
      ) > >(tee "$log") 2> >(tee -a "$log" >&2) &
      pids+=("$!")
    done
    rc=0
    for pid in "${pids[@]}"; do
      wait "$pid" || rc=1
    done
    if [[ "$rc" != "0" ]]; then
      echo "[waymax/$method] shard failed; see logs" >&2
      exit 1
    fi
    shards=()
    for ((shard=0; shard<NUM_WAYMAX_SHARDS; shard++)); do shards+=("$OUT_ROOT/eval/waymax/${method}_shard${shard}.json"); done
    logrun "merge_waymax_${method}" "$PYTHON_BIN" -u -m cowp.scripts.17_merge_waymax_shards --output "$merged" --inputs "${shards[@]}"
  done
fi

echo "[cowp_v4] done. Results under $OUT_ROOT"
