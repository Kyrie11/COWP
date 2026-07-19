#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v8}"
TRAIN_CACHE="${TRAIN_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax}"
VAL_CACHE="${VAL_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax}"
WAYMAX_VAL="${WAYMAX_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_OFFLINE="${RUN_OFFLINE:-1}"
RUN_PROBE="${RUN_PROBE:-1}"
RUN_PARETO_ABLATION="${RUN_PARETO_ABLATION:-1}"
RUN_FULL="${RUN_FULL:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
DETACH="${DETACH:-0}"

EPOCHS="${EPOCHS:-18}"
BATCH_PER_GPU="${BATCH_PER_GPU:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
# The new Set-Transport and compact response heads are fine-tuned on the v7 representation.
# Keep the shared graph/candidate/natural/proxy-witness stack frozen to prevent representation pollution.
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-999}"
LR="${LR:-5e-5}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-6}"
TRAIN_SEED="${TRAIN_SEED:-2026}"

PROBE_SCENARIOS="${PROBE_SCENARIOS:-100}"
FULL_SCENARIOS="${FULL_SCENARIOS:-1000}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-80}"
DEFAULT_WITNESS_THRESHOLD="${WITNESS_THRESHOLD:-0.30}"
WITNESS_SWEEP="${WITNESS_SWEEP:-0.20,0.24,0.27,0.30,0.33,0.36,0.40,0.45,0.50}"
NCF_GATE_MODE="${NCF_GATE_MODE:-priority}"
OUTCOME_RISK_PENALTY="${OUTCOME_RISK_PENALTY:-1.0}"
OUTCOME_RISK_THRESHOLD="${OUTCOME_RISK_THRESHOLD:-0.65}"
WAYMAX_ACTION_MODE="${WAYMAX_ACTION_MODE:-absolute_xy_yaw}"

INIT_CKPT="${INIT_CKPT:-outputs/cowp_v7/checkpoints/planner/cowp_planner_best.pt}"
CKPT="${CKPT:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_ROOT"/{logs,configs,checkpoints/planner,eval/learned_offline,eval/probe,eval/waymax,jax_cache}
cp configs/label_cowp_v8.yaml "$OUT_ROOT/configs/label_cowp_v8.yaml"
cp configs/label_cowp_v8_pareto_ablation.yaml "$OUT_ROOT/configs/label_cowp_v8_pareto_ablation.yaml"
cp configs/train_cowp_v8.yaml "$OUT_ROOT/configs/train_cowp_v8.yaml"
sed -i -E "0,/^  seed:/{s/^  seed:.*/  seed: ${TRAIN_SEED}/}" "$OUT_ROOT/configs/train_cowp_v8.yaml"
LABEL_CFG="$OUT_ROOT/configs/label_cowp_v8.yaml"
PARETO_CFG="$OUT_ROOT/configs/label_cowp_v8_pareto_ablation.yaml"
TRAIN_CFG="$OUT_ROOT/configs/train_cowp_v8.yaml"

if [[ "$DETACH" == "1" && "${COWP_V8_DETACHED:-0}" != "1" ]]; then
  export COWP_V8_DETACHED=1
  nohup bash "$0" > "$OUT_ROOT/logs/driver.nohup.log" 2>&1 &
  echo "[cowp_v8] detached pid=$! log=$OUT_ROOT/logs/driver.nohup.log"
  exit 0
fi

logrun() {
  local name="$1"; shift
  local log="$OUT_ROOT/logs/${name}.log"
  echo "[$name] -> $log"
  "$@" > >(tee "$log") 2> >(tee -a "$log" >&2)
}

json_valid() {
  local p="$1"
  [[ "$FORCE_EVAL" != "1" && -s "$p" ]] || return 1
  "$PYTHON_BIN" - "$p" <<'PY' >/dev/null
import json, sys
json.load(open(sys.argv[1], encoding="utf-8"))
PY
}

best_ckpt() {
  local best="$OUT_ROOT/checkpoints/planner/cowp_planner_best.pt"
  [[ -s "$best" ]] && { echo "$best"; return; }
  return 1
}

if [[ "$RUN_TRAIN" == "1" ]]; then
  if best_ckpt >/dev/null && [[ "$FORCE_TRAIN" != "1" ]]; then
    echo "[train] keep existing $(best_ckpt)"
  else
    resume_args=()
    if [[ -s "$INIT_CKPT" ]]; then
      resume_args=(--resume "$INIT_CKPT")
    else
      echo "[train] warning: INIT_CKPT not found: $INIT_CKPT; training from scratch is not recommended for this set-transport v8 stage" >&2
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
      --early-stop-patience "$EARLY_STOP_PATIENCE" \
      --early-stop-min-delta 1e-4 \
      --lr-scheduler plateau \
      --min-lr 2e-6 \
      --save-every 3 \
      --no-positive-oversampling \
      --no-response-traj \
      --no-response-components \
      --amp \
      --fused-adamw \
      "${resume_args[@]}"
  fi
fi

if [[ -z "$CKPT" ]]; then CKPT="$(best_ckpt || true)"; fi
if [[ -z "$CKPT" || ! -s "$CKPT" ]]; then
  echo "No v8 best checkpoint. Set CKPT=... or enable training." >&2
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
      --witness-threshold "$DEFAULT_WITNESS_THRESHOLD" \
      --witness-threshold-sweep "$WITNESS_SWEEP" \
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
  logrun calibrate_witness "$PYTHON_BIN" -u -m cowp.scripts.18_calibrate_witness_threshold \
    --input "$combined" \
    --method cowp \
    --output "$OUT_ROOT/eval/learned_offline/witness_calibration.json" \
    --min-ncf-recall 0.30 \
    --max-fallback 0.25
  logrun verify_mechanism "$PYTHON_BIN" -u -m cowp.scripts.25_verify_mechanism_effect \
    --input "$combined" --method cowp \
    --min-unique-selection-points 2 --min-ncf-recall 0.20 \
    --output "$OUT_ROOT/eval/learned_offline/mechanism_verification.json"
fi

CALIBRATION_JSON="$OUT_ROOT/eval/learned_offline/witness_calibration.json"
if [[ -s "$CALIBRATION_JSON" ]]; then
  ONLINE_WITNESS_THRESHOLD="$($PYTHON_BIN - "$CALIBRATION_JSON" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["witness_threshold"])
PY
)"
else
  ONLINE_WITNESS_THRESHOLD="$DEFAULT_WITNESS_THRESHOLD"
fi
echo "[witness threshold] $ONLINE_WITNESS_THRESHOLD"

run_online_one() {
  local method="$1" gpu="$2" scenarios="$3" out="$4" log="$5" label_cfg="$6" shard_count="${7:-1}" shard_index="${8:-0}"
  local cfg_tag
  cfg_tag="$(basename "$label_cfg" .yaml)"
  local cache="$OUT_ROOT/jax_cache/${method}_${cfg_tag}_g${gpu}_s${shard_index}"
  mkdir -p "$cache"
  echo "[waymax $method cfg=$cfg_tag gpu=$gpu shard=$shard_index/$shard_count] -> $log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" JAX_COMPILATION_CACHE_DIR="$cache" \
    "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
      --mode waymax \
      --method "$method" \
      --checkpoint "$CKPT" \
      --cache-dir "$VAL_CACHE" \
      --data-config configs/data.yaml \
      --label-config "$label_cfg" \
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
      --reuse-waymax-env \
      --prefilter-waymax-shards \
      --jit-waymax-env \
      --jit-waymax-metrics \
      --witness-threshold "$ONLINE_WITNESS_THRESHOLD" \
      --ncf-gate-mode "$NCF_GATE_MODE" \
      --outcome-risk-penalty "$OUTCOME_RISK_PENALTY" \
      --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
      --output "$out" \
      --no-progress
  ) > >(tee "$log") 2> >(tee -a "$log" >&2)
}

if [[ "$RUN_PROBE" == "1" ]]; then
  probe_cowp="$OUT_ROOT/eval/probe/cowp_set_transport_${PROBE_SCENARIOS}.json"
  probe_conv="$OUT_ROOT/eval/probe/conventional_safety_${PROBE_SCENARIOS}.json"
  if ! json_valid "$probe_cowp" || ! json_valid "$probe_conv"; then
    run_online_one cowp "$GPU0" "$PROBE_SCENARIOS" "$probe_cowp" "$OUT_ROOT/logs/probe_cowp_set_transport.log" "$LABEL_CFG" & p0=$!
    run_online_one conventional_safety "$GPU1" "$PROBE_SCENARIOS" "$probe_conv" "$OUT_ROOT/logs/probe_conventional.log" "$LABEL_CFG" & p1=$!
    wait "$p0"; wait "$p1"
  fi
  logrun summarize_probe "$PYTHON_BIN" -u -m cowp.scripts.24_summarize_planner_delta \
    --reference "$probe_conv" --candidate "$probe_cowp" --output "$OUT_ROOT/eval/probe/delta_conventional_vs_set_transport.json"

  if [[ "$RUN_PARETO_ABLATION" == "1" ]]; then
    probe_pareto="$OUT_ROOT/eval/probe/cowp_pareto_${PROBE_SCENARIOS}.json"
    if ! json_valid "$probe_pareto"; then
      run_online_one cowp "$GPU1" "$PROBE_SCENARIOS" "$probe_pareto" "$OUT_ROOT/logs/probe_cowp_pareto.log" "$PARETO_CFG"
    fi
    logrun summarize_frontier_ablation "$PYTHON_BIN" -u -m cowp.scripts.24_summarize_planner_delta \
      --reference "$probe_pareto" --candidate "$probe_cowp" --output "$OUT_ROOT/eval/probe/delta_pareto_vs_set_transport.json"
  fi
fi

if [[ "$RUN_FULL" == "1" ]]; then
  conv_out="$OUT_ROOT/eval/waymax/conventional_safety_merged.json"
  planner_out="$OUT_ROOT/eval/waymax/planner_score_only_merged.json"
  if ! json_valid "$conv_out" || ! json_valid "$planner_out"; then
    run_online_one conventional_safety "$GPU0" "$FULL_SCENARIOS" "$conv_out" "$OUT_ROOT/logs/full_conventional.log" "$LABEL_CFG" & p0=$!
    run_online_one planner_score_only "$GPU1" "$FULL_SCENARIOS" "$planner_out" "$OUT_ROOT/logs/full_planner.log" "$LABEL_CFG" & p1=$!
    wait "$p0"; wait "$p1"
  fi
  per_shard=$(( (FULL_SCENARIOS + 1) / 2 ))
  c0="$OUT_ROOT/eval/waymax/cowp_set_transport_shard0.json"
  c1="$OUT_ROOT/eval/waymax/cowp_set_transport_shard1.json"
  run_online_one cowp "$GPU0" "$per_shard" "$c0" "$OUT_ROOT/logs/full_cowp_shard0.log" "$LABEL_CFG" 2 0 & p0=$!
  run_online_one cowp "$GPU1" "$per_shard" "$c1" "$OUT_ROOT/logs/full_cowp_shard1.log" "$LABEL_CFG" 2 1 & p1=$!
  wait "$p0"; wait "$p1"
  logrun merge_cowp "$PYTHON_BIN" -u -m cowp.scripts.17_merge_waymax_shards \
    --output "$OUT_ROOT/eval/waymax/cowp_set_transport_merged.json" --inputs "$c0" "$c1"
fi

echo "[cowp_v8] complete: $OUT_ROOT"
