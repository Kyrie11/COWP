#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v6}"
TRAIN_CACHE="${TRAIN_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax}"
VAL_CACHE="${VAL_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax}"
WAYMAX_VAL="${WAYMAX_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

# The supplied cache analysis was performed on exactly 14,640 / 5,013 files.
# A v5 training log loaded 20,441 train files, so this assertion is deliberately
# strict. Set EXPECTED_TRAIN_FILES/EXPECTED_VAL_FILES only after regenerating a new
# cache-sufficiency report for a different immutable snapshot.
EXPECTED_TRAIN_FILES="${EXPECTED_TRAIN_FILES:-14640}"
EXPECTED_VAL_FILES="${EXPECTED_VAL_FILES:-5013}"
PREFLIGHT_SAMPLE="${PREFLIGHT_SAMPLE:-1024}"

RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_OFFLINE="${RUN_OFFLINE:-1}"
RUN_PROBE="${RUN_PROBE:-1}"
RUN_FULL="${RUN_FULL:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
DETACH="${DETACH:-0}"

EPOCHS="${EPOCHS:-24}"
BATCH_PER_GPU="${BATCH_PER_GPU:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-3}"
LR="${LR:-5e-5}"

PROBE_SCENARIOS="${PROBE_SCENARIOS:-100}"
FULL_SCENARIOS="${FULL_SCENARIOS:-1000}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-80}"
WITNESS_THRESHOLD="${WITNESS_THRESHOLD:-0.50}"
NCF_GATE_MODE="${NCF_GATE_MODE:-priority}"
OUTCOME_RISK_PENALTY="${OUTCOME_RISK_PENALTY:-1.0}"
OUTCOME_RISK_THRESHOLD="${OUTCOME_RISK_THRESHOLD:-0.65}"
WAYMAX_ACTION_MODE="${WAYMAX_ACTION_MODE:-absolute_xy_yaw}"

INIT_CKPT="${INIT_CKPT:-outputs/cowp_v5/checkpoints/planner/cowp_planner_best.pt}"
CKPT="${CKPT:-}"

mkdir -p "$OUT_ROOT"/{logs,configs,checkpoints/planner,eval/learned_offline,eval/probe,eval/waymax,jax_cache}
cp configs/label_cowp_v6.yaml "$OUT_ROOT/configs/label_cowp_v6.yaml"
cp configs/train_cowp_v6.yaml "$OUT_ROOT/configs/train_cowp_v6.yaml"
LABEL_CFG="$OUT_ROOT/configs/label_cowp_v6.yaml"
TRAIN_CFG="$OUT_ROOT/configs/train_cowp_v6.yaml"

if [[ "$DETACH" == "1" && "${COWP_V6_DETACHED:-0}" != "1" ]]; then
  export COWP_V6_DETACHED=1
  nohup bash "$0" > "$OUT_ROOT/logs/driver.nohup.log" 2>&1 &
  echo "[cowp_v6] detached pid=$! log=$OUT_ROOT/logs/driver.nohup.log"
  exit 0
fi

logrun() {
  local name="$1"; shift
  local log="$OUT_ROOT/logs/${name}.log"
  echo "[$name] -> $log"
  "$@" > >(tee "$log") 2> >(tee -a "$log" >&2)
}

best_ckpt() {
  local best="$OUT_ROOT/checkpoints/planner/cowp_planner_best.pt"
  [[ -s "$best" ]] && { echo "$best"; return; }
  return 1
}

json_valid() {
  local p="$1"
  [[ "$FORCE_EVAL" != "1" && -s "$p" ]] || return 1
  "$PYTHON_BIN" - "$p" <<'PY' >/dev/null
import json, sys
json.load(open(sys.argv[1], encoding='utf-8'))
PY
}

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  logrun preflight "$PYTHON_BIN" -u -m cowp.scripts.20_assert_cache_snapshot \
    --train-cache "$TRAIN_CACHE" \
    --val-cache "$VAL_CACHE" \
    --expected-train-files "$EXPECTED_TRAIN_FILES" \
    --expected-val-files "$EXPECTED_VAL_FILES" \
    --sample "$PREFLIGHT_SAMPLE" \
    --output "$OUT_ROOT/preflight_cache_snapshot.json"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  if best_ckpt >/dev/null && [[ "$FORCE_TRAIN" != "1" ]]; then
    echo "[train] keep existing $(best_ckpt)"
  else
    resume_args=()
    if [[ -s "$INIT_CKPT" ]]; then
      # Warm start only. The v6 structured certificate head has a different input
      # shape and is intentionally reinitialized by the robust checkpoint loader.
      resume_args=(--resume "$INIT_CKPT")
    else
      echo "[train] warning: INIT_CKPT not found: $INIT_CKPT; training from scratch" >&2
    fi
    logrun train_planner_ddp env CUDA_VISIBLE_DEVICES="$GPU0,$GPU1" \
      "$TORCHRUN_BIN" --standalone --nproc_per_node=2 -m cowp.scripts.03_train \
      --data-config configs/data.yaml \
      --model-config configs/model.yaml \
      --label-config "$LABEL_CFG" \
      --train-config "$TRAIN_CFG" \
      --cache-dir "$TRAIN_CACHE" \
      --val-cache-dir "$VAL_CACHE" \
      --stage planner \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_PER_GPU" \
      --lr "$LR" \
      --num-workers "$NUM_WORKERS" \
      --prefetch-factor "$PREFETCH_FACTOR" \
      --device cuda \
      --output-dir "$OUT_ROOT/checkpoints/planner" \
      --with-waymax-outcome-labels \
      --freeze-backbone-epochs "$FREEZE_BACKBONE_EPOCHS" \
      --no-positive-oversampling \
      --amp \
      --fused-adamw \
      "${resume_args[@]}"
  fi
fi

if [[ -z "$CKPT" ]]; then
  CKPT="$(best_ckpt || true)"
fi
if [[ -z "$CKPT" || ! -s "$CKPT" ]]; then
  echo "No v6 best checkpoint. Set CKPT=... or enable training." >&2
  exit 2
fi
echo "[checkpoint] $CKPT"

if [[ "$RUN_OFFLINE" == "1" ]]; then
  combined="$OUT_ROOT/eval/learned_offline/_shared_model_pass.json"
  if ! json_valid "$combined"; then
    logrun learned_offline env CUDA_VISIBLE_DEVICES="$GPU0" \
      "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
      --mode learned_offline \
      --methods planner_score_only,conventional_safety,soft_burden_cost_only,universal_ncf,cowp \
      --checkpoint "$CKPT" \
      --cache-dir "$VAL_CACHE" \
      --data-config configs/data.yaml \
      --label-config "$LABEL_CFG" \
      --eval-config configs/eval.yaml \
      --device cuda \
      --batch-size 24 \
      --num-workers 4 \
      --prefetch-factor 1 \
      --witness-threshold "$WITNESS_THRESHOLD" \
      --ncf-gate-mode "$NCF_GATE_MODE" \
      --offline-fallback stop_like \
      --outcome-risk-penalty "$OUTCOME_RISK_PENALTY" \
      --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
      --output "$combined" \
      --no-progress
  fi
  "$PYTHON_BIN" - "$combined" "$OUT_ROOT/eval/learned_offline" <<'PY'
import json, pathlib, sys
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); d=json.load(src.open())
for m in ("planner_score_only","conventional_safety","soft_burden_cost_only","universal_ncf","cowp"):
    payload={m:d[m], "mode":d.get("mode"), "checkpoint":d.get("checkpoint"),
             "ncf_gate_mode":d.get("ncf_gate_mode"), "shared_model_pass":True}
    (out/f"{m}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
PY
fi

run_online_one() {
  local method="$1" gpu="$2" scenarios="$3" out="$4" log="$5" shard_count="${6:-1}" shard_index="${7:-0}"
  local cache="$OUT_ROOT/jax_cache/${method}_g${gpu}_s${shard_index}"
  mkdir -p "$cache"
  echo "[waymax $method gpu=$gpu shard=$shard_index/$shard_count] -> $log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" JAX_COMPILATION_CACHE_DIR="$cache" \
    "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
      --mode waymax \
      --method "$method" \
      --checkpoint "$CKPT" \
      --cache-dir "$VAL_CACHE" \
      --data-config configs/data.yaml \
      --label-config "$LABEL_CFG" \
      --eval-config configs/eval.yaml \
      --tfexample-glob "$WAYMAX_VAL" \
      --num-shards "$shard_count" \
      --shard-index "$shard_index" \
      --num-scenarios "$scenarios" \
      --rollout-horizon-steps "$ROLLOUT_HORIZON" \
      --device auto \
      --waymax-device gpu \
      --waymax-action-mode "$WAYMAX_ACTION_MODE" \
      --jax-visible-devices 0 \
      --jax-preallocate false \
      --waymax-standard-metrics \
      --witness-threshold "$WITNESS_THRESHOLD" \
      --ncf-gate-mode "$NCF_GATE_MODE" \
      --outcome-risk-penalty "$OUTCOME_RISK_PENALTY" \
      --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
      --output "$out" \
      --no-progress
  ) > >(tee "$log") 2> >(tee -a "$log" >&2)
}

if [[ "$RUN_PROBE" == "1" ]]; then
  probe_cowp="$OUT_ROOT/eval/probe/cowp_${PROBE_SCENARIOS}.json"
  probe_conv="$OUT_ROOT/eval/probe/conventional_safety_${PROBE_SCENARIOS}.json"
  if ! json_valid "$probe_cowp" || ! json_valid "$probe_conv"; then
    run_online_one cowp "$GPU0" "$PROBE_SCENARIOS" "$probe_cowp" "$OUT_ROOT/logs/probe_cowp.log" & p0=$!
    run_online_one conventional_safety "$GPU1" "$PROBE_SCENARIOS" "$probe_conv" "$OUT_ROOT/logs/probe_conventional.log" & p1=$!
    wait "$p0"; wait "$p1"
  fi
  logrun gate_probe "$PYTHON_BIN" -u -m cowp.scripts.21_gate_v6_results \
    --cowp-offline "$OUT_ROOT/eval/learned_offline/cowp.json" \
    --conventional-offline "$OUT_ROOT/eval/learned_offline/conventional_safety.json" \
    --cowp-online "$probe_cowp" \
    --conventional-online "$probe_conv" \
    --probe
fi

if [[ "$RUN_FULL" == "1" ]]; then
  # Phase 1: independent baselines run concurrently, one per GPU.
  planner_out="$OUT_ROOT/eval/waymax/planner_score_only_merged.json"
  conv_out="$OUT_ROOT/eval/waymax/conventional_safety_merged.json"
  if ! json_valid "$planner_out" || ! json_valid "$conv_out"; then
    run_online_one planner_score_only "$GPU0" "$FULL_SCENARIOS" "$planner_out" "$OUT_ROOT/logs/full_planner_score_only.log" & p0=$!
    run_online_one conventional_safety "$GPU1" "$FULL_SCENARIOS" "$conv_out" "$OUT_ROOT/logs/full_conventional_safety.log" & p1=$!
    wait "$p0"; wait "$p1"
  fi

  # Phase 2: COWP is split across both GPUs and merged exactly.
  per_shard=$(( (FULL_SCENARIOS + 1) / 2 ))
  c0="$OUT_ROOT/eval/waymax/cowp_shard0.json"
  c1="$OUT_ROOT/eval/waymax/cowp_shard1.json"
  run_online_one cowp "$GPU0" "$per_shard" "$c0" "$OUT_ROOT/logs/full_cowp_shard0.log" 2 0 & p0=$!
  run_online_one cowp "$GPU1" "$per_shard" "$c1" "$OUT_ROOT/logs/full_cowp_shard1.log" 2 1 & p1=$!
  wait "$p0"; wait "$p1"
  logrun merge_cowp "$PYTHON_BIN" -u -m cowp.scripts.17_merge_waymax_shards \
    --output "$OUT_ROOT/eval/waymax/cowp_merged.json" --inputs "$c0" "$c1"

  logrun gate_full "$PYTHON_BIN" -u -m cowp.scripts.21_gate_v6_results \
    --cowp-offline "$OUT_ROOT/eval/learned_offline/cowp.json" \
    --conventional-offline "$OUT_ROOT/eval/learned_offline/conventional_safety.json" \
    --cowp-online "$OUT_ROOT/eval/waymax/cowp_merged.json" \
    --conventional-online "$conv_out"
fi

echo "[cowp_v6] complete: $OUT_ROOT"
