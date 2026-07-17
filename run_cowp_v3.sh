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
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v3}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR" "$OUT_ROOT/eval/learned_offline" "$OUT_ROOT/eval/waymax" "$OUT_ROOT/checkpoints/planner" "$OUT_ROOT/configs"

if [[ "${DETACH:-0}" == "1" && -z "${COWP_ALREADY_DETACHED:-}" ]]; then
  export COWP_ALREADY_DETACHED=1
  nohup bash "$0" "$@" > "$LOG_DIR/driver.nohup.log" 2>&1 &
  echo "Detached PID: $!"
  echo "Driver log: $LOG_DIR/driver.nohup.log"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

TOTAL_ONLINE_SCENARIOS="${TOTAL_ONLINE_SCENARIOS:-300}"
ROLLOUT_HORIZON_STEPS="${ROLLOUT_HORIZON_STEPS:-80}"
EPOCH_PLANNER="${EPOCH_PLANNER:-32}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
EVAL_BATCH="${EVAL_BATCH:-64}"
WITNESS_THRESHOLD="${WITNESS_THRESHOLD:-0.05}"
OUTCOME_RISK_PENALTY="${OUTCOME_RISK_PENALTY:-0.75}"
OUTCOME_RISK_THRESHOLD="${OUTCOME_RISK_THRESHOLD:-1.10}"
ONLINE_METHODS="${ONLINE_METHODS:-planner_score_only conventional_safety cowp}"

logrun() { local name="$1"; shift; echo "[$name] -> $LOG_DIR/$name.log"; "$@" > "$LOG_DIR/$name.log" 2>&1; }
json_valid() { [[ -s "$1" ]] && "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import json,sys
json.load(open(sys.argv[1]))
PY
}
ckpt_epoch() { "$PYTHON_BIN" - "$1" <<'PY'
import re,sys,torch
p=sys.argv[1]
try:
    o=torch.load(p,map_location='cpu',weights_only=False)
    print(int(o.get('epoch',-1)) if isinstance(o,dict) else -1)
except Exception:
    m=re.search(r'_epoch(\d+)\.pt$',p); print(int(m.group(1)) if m else -1)
PY
}
latest_ckpt() { local d="$1"; ls "$d"/cowp_planner_epoch*.pt 2>/dev/null | sed -E 's/.*epoch([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-; }

TRAIN_CFG="$OUT_ROOT/configs/train_cowp_v3.yaml"
LABEL_CFG="$OUT_ROOT/configs/label_cowp_v3.yaml"
logrun make_runtime_configs "$PYTHON_BIN" - "configs/train.yaml" "$TRAIN_CFG" "configs/label.yaml" "$LABEL_CFG" <<'PY'
import sys,yaml
train_src,train_dst,label_src,label_dst=sys.argv[1:5]
train=yaml.safe_load(open(train_src,encoding='utf-8'))
w=train.setdefault('loss_weights',{})
w.update({
 'planner_witness_scale':0.005,
 'candidate_certificate':4.0,
 'candidate_certificate_ncf':2.0,
 'candidate_certificate_false_safe':3.5,
 'candidate_certificate_quality':1.5,
 'candidate_certificate_prior':1.0,
 'candidate_certificate_rank':3.0,
 'candidate_certificate_risk_bce':4.0,
 'candidate_certificate_risk_rank':4.0,
 'candidate_certificate_spread':1.0,
 'candidate_cert_min_logit_spread':0.65,
 'closed_loop':2.0,
 'outcome_cls':2.0,
 'outcome_pair_rank':2.0,
 'outcome_expected_cost':1.0,
 'outcome_logdiv':0.0,
 'outcome_logdiv_unsafe_threshold':1e9,
 'candidate_ncf_cls':2.0,
 'candidate_false_safe_cls':1.5,
 'ranking':1.0,
 'imitation':0.15,
})
yaml.safe_dump(train,open(train_dst,'w',encoding='utf-8'),sort_keys=False,allow_unicode=True)
label=yaml.safe_load(open(label_src,encoding='utf-8'))
pcfg=label.setdefault('planning',{})
pcfg.update({
 'witness_probability_source':'logit',
 'evidential_probability_mix':0.0,
 'evidential_ucb_scale':0.0,
 'candidate_certificate_penalty':3.0,
 'candidate_pair_risk_mix':0.10,
 'candidate_frontier_keep_fraction':0.25,
 'candidate_frontier_min_keep':2,
 'candidate_frontier_max_keep':4,
 'candidate_frontier_tie_eps':0.002,
 'candidate_cert_fallback_min_std':0.002,
 'candidate_cert_flat_fallback_mix':0.95,
 'candidate_cert_hybrid_fallback_mix':0.30,
 'candidate_cert_fallback_outcome_mix':0.55,
 'candidate_cert_fallback_action_mix':0.25,
 'candidate_cert_fallback_rule_mix':0.15,
 'candidate_cert_fallback_pressure_mix':0.05,
 'candidate_pressure_prior_mix':0.35,
 'candidate_rule_risk_mix':2.0,
 'candidate_action_risk_mix':3.0,
 'candidate_outcome_risk_mix':1.25,
 'candidate_action_risk_penalty':3.0,
 'candidate_rule_risk_penalty':2.5,
 'decision_risk_min_std':1e-3,
 'decision_risk_min_spread':1e-3,
 'target_online_candidates':32,
})
yaml.safe_dump(label,open(label_dst,'w',encoding='utf-8'),sort_keys=False,allow_unicode=True)
PY

PLANNER_DIR="$OUT_ROOT/checkpoints/planner"
CKPT="$(latest_ckpt "$PLANNER_DIR" || true)"
CUR_EPOCH="-1"; [[ -n "$CKPT" ]] && CUR_EPOCH="$(ckpt_epoch "$CKPT")"
if [[ "$FORCE_TRAIN" == "1" || "$CUR_EPOCH" -lt $((EPOCH_PLANNER-1)) ]]; then
  RESUME_ARGS=()
  [[ -n "$CKPT" ]] && RESUME_ARGS+=(--resume "$CKPT" --resume-training)
  logrun train_planner "$PYTHON_BIN" -u -m cowp.scripts.03_train \
    --data-config configs/data.yaml --model-config configs/model.yaml --train-config "$TRAIN_CFG" --label-config "$LABEL_CFG" \
    --cache-dir "$TRAIN_CACHE" --val-cache-dir "$VAL_CACHE" --stage planner --epochs "$EPOCH_PLANNER" \
    --batch-size 16 --num-workers 6 --prefetch-factor 1 --amp --fused-adamw --with-waymax-outcome-labels \
    --val-every 1 --output-dir "$PLANNER_DIR" "${RESUME_ARGS[@]}"
fi
CKPT="$(latest_ckpt "$PLANNER_DIR")"
echo "Using checkpoint: $CKPT"

run_offline() {
  local m="$1"; local out="$OUT_ROOT/eval/learned_offline/${m}.json"
  if [[ "$FORCE_EVAL" != "1" ]] && json_valid "$out"; then echo "[offline/$m] keep existing"; return; fi
  logrun "learned_${m}" "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
    --data-config configs/data.yaml --label-config "$LABEL_CFG" --eval-config configs/eval.yaml \
    --cache-dir "$VAL_CACHE" --mode learned_offline --method "$m" --checkpoint "$CKPT" --batch-size "$EVAL_BATCH" \
    --witness-threshold "$WITNESS_THRESHOLD" --ncf-gate-mode priority --offline-fallback stop_like \
    --outcome-risk-penalty "$OUTCOME_RISK_PENALTY" --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
    --output "$out" --no-progress
}
for m in planner_score_only conventional_safety soft_burden_cost_only universal_ncf cowp; do run_offline "$m" & done
wait

run_waymax_method() {
  local m="$1"
  local pids=()
  for shard in 0 1; do
    local out="$OUT_ROOT/eval/waymax/${m}_shard${shard}.json"
    if [[ "$FORCE_EVAL" != "1" ]] && json_valid "$out"; then echo "[waymax/$m/$shard] keep existing"; else
      CUDA_VISIBLE_DEVICES="$shard" logrun "waymax_${m}_shard${shard}" "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
        --data-config configs/data.yaml --label-config "$LABEL_CFG" --eval-config configs/eval.yaml \
        --mode waymax --method "$m" --checkpoint "$CKPT" --num-scenarios "$TOTAL_ONLINE_SCENARIOS" \
        --num-shards 2 --shard-index "$shard" --tfexample-glob "$WAYMAX_VAL" \
        --rollout-horizon-steps "$ROLLOUT_HORIZON_STEPS" --waymax-standard-metrics \
        --witness-threshold "$WITNESS_THRESHOLD" --ncf-gate-mode priority \
        --outcome-risk-penalty "$OUTCOME_RISK_PENALTY" --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
        --waymax-device gpu --jax-visible-devices 0 --waymax-action-mode absolute_xy_yaw \
        --status-every 10 --output "$out" --no-progress &
      pids+=("$!")
    fi
  done
  [[ ${#pids[@]} -gt 0 ]] && wait "${pids[@]}"
  logrun "merge_waymax_${m}" "$PYTHON_BIN" -m cowp.scripts.17_merge_waymax_shards \
    --inputs "$OUT_ROOT/eval/waymax/${m}_shard0.json,$OUT_ROOT/eval/waymax/${m}_shard1.json" \
    --output "$OUT_ROOT/eval/waymax/${m}_merged.json"
}
for m in $ONLINE_METHODS; do run_waymax_method "$m"; done

echo "Done. Inspect $OUT_ROOT/eval/learned_offline/cowp.json and $OUT_ROOT/eval/waymax/cowp_merged.json"
