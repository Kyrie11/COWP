#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v9_probe100_seed2026}"
RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax}"
RAW_VAL_CACHE="${RAW_VAL_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax}"
TRAIN_CACHE="${TRAIN_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax_transport_v9}"
VAL_CACHE="${VAL_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax_transport_v9}"
WAYMAX_VAL="${WAYMAX_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

RUN_AUGMENT="${RUN_AUGMENT:-1}"
RUN_DIAGNOSE="${RUN_DIAGNOSE:-1}"
RUN_TRANSPORT="${RUN_TRANSPORT:-1}"
RUN_PLANNER="${RUN_PLANNER:-1}"
RUN_OFFLINE="${RUN_OFFLINE:-1}"
RUN_PROBE="${RUN_PROBE:-1}"
RUN_PARETO_ABLATION="${RUN_PARETO_ABLATION:-1}"
RUN_SEMANTIC_PROBE="${RUN_SEMANTIC_PROBE:-1}"
RUN_FULL="${RUN_FULL:-0}"
FORCE_AUGMENT="${FORCE_AUGMENT:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
DETACH="${DETACH:-0}"

AUG_TRAIN_WORKERS="${AUG_TRAIN_WORKERS:-12}"
AUG_VAL_WORKERS="${AUG_VAL_WORKERS:-6}"
AUG_CHUNKSIZE="${AUG_CHUNKSIZE:-2}"
AUG_STORAGE_MODE="${AUG_STORAGE_MODE:-overlay}"
AUG_SIDECAR_SUBDIR="${AUG_SIDECAR_SUBDIR:-.transport_v9}"
TRANSPORT_EPOCHS="${TRANSPORT_EPOCHS:-10}"
PLANNER_EPOCHS="${PLANNER_EPOCHS:-12}"
BATCH_PER_GPU="${BATCH_PER_GPU:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-999}"
TRANSPORT_LR="${TRANSPORT_LR:-2e-5}"
PLANNER_LR="${PLANNER_LR:-1e-5}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
TRAIN_SEED="${TRAIN_SEED:-2026}"

PROBE_SCENARIOS="${PROBE_SCENARIOS:-100}"
FULL_SCENARIOS="${FULL_SCENARIOS:-1000}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-80}"
DEFAULT_WITNESS_THRESHOLD="${WITNESS_THRESHOLD:-0.45}"
WITNESS_SWEEP="${WITNESS_SWEEP:-0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70}"
NCF_GATE_MODE="${NCF_GATE_MODE:-priority}"
OFFLINE_OUTCOME_RISK_PENALTY="${OFFLINE_OUTCOME_RISK_PENALTY:-0.0}"
ONLINE_OUTCOME_RISK_PENALTY="${ONLINE_OUTCOME_RISK_PENALTY:-0.75}"
OUTCOME_RISK_THRESHOLD="${OUTCOME_RISK_THRESHOLD:-0.65}"
WAYMAX_ACTION_MODE="${WAYMAX_ACTION_MODE:-absolute_xy_yaw}"

INIT_CKPT="${INIT_CKPT:-outputs/cowp_v8_probe100_seed2026/checkpoints/planner/cowp_planner_best.pt}"
TRANSPORT_CKPT="${TRANSPORT_CKPT:-}"
CKPT="${CKPT:-}"
REQUIRE_INIT_CKPT="${REQUIRE_INIT_CKPT:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_ROOT"/{logs,configs,checkpoints/transport,checkpoints/planner,eval/learned_offline,eval/probe,eval/waymax,jax_cache}
cp configs/label_cowp_v9.yaml "$OUT_ROOT/configs/label_cowp_v9.yaml"
cp configs/label_cowp_v9_pareto_ablation.yaml "$OUT_ROOT/configs/label_cowp_v9_pareto_ablation.yaml"
cp configs/train_cowp_v9.yaml "$OUT_ROOT/configs/train_cowp_v9.yaml"
cp configs/eval_cowp_v9.yaml "$OUT_ROOT/configs/eval_cowp_v9.yaml"
sed -i -E "0,/^  seed:/{s/^  seed:.*/  seed: ${TRAIN_SEED}/}" "$OUT_ROOT/configs/train_cowp_v9.yaml"
LABEL_CFG="$OUT_ROOT/configs/label_cowp_v9.yaml"
PARETO_CFG="$OUT_ROOT/configs/label_cowp_v9_pareto_ablation.yaml"
TRAIN_CFG="$OUT_ROOT/configs/train_cowp_v9.yaml"
EVAL_CFG="$OUT_ROOT/configs/eval_cowp_v9.yaml"

if [[ "$DETACH" == "1" && "${COWP_V9_DETACHED:-0}" != "1" ]]; then
  export COWP_V9_DETACHED=1
  nohup bash "$0" > "$OUT_ROOT/logs/driver.nohup.log" 2>&1 &
  echo "[cowp_v9] detached pid=$! log=$OUT_ROOT/logs/driver.nohup.log"
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

cache_ready() {
  local d="$1"
  [[ -s "$d/transport_augmentation_summary.json" ]] || return 1
  "$PYTHON_BIN" - "$d/transport_augmentation_summary.json" <<'PY' >/dev/null
import json,sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
total=int(x.get("files_total",0))
if "files_completed" in x:
    done=int(x.get("files_completed",0))
else:
    done=int(x.get("files_written",0))+int(x.get("files_skipped",0))+int(x.get("files_skipped_sidecar",0))+int(x.get("files_skipped_materialized",0))
assert total > 0 and done == total and int(x.get("error_count",0)) == 0
assert bool(x.get("complete", True))
PY
}

if [[ "$RUN_AUGMENT" == "1" ]]; then
  pids=()
  force_aug_args=()
  [[ "$FORCE_AUGMENT" == "1" ]] && force_aug_args=(--force)
  if ! cache_ready "$TRAIN_CACHE" || [[ "$FORCE_AUGMENT" == "1" ]]; then
    mkdir -p "$TRAIN_CACHE"
    (
      env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$PYTHON_BIN" -u -m cowp.scripts.26_augment_transport_labels \
        --data-config configs/data.yaml --label-config "$LABEL_CFG" \
        --input-dir "$RAW_TRAIN_CACHE" --output-dir "$TRAIN_CACHE" \
        --num-workers "$AUG_TRAIN_WORKERS" --chunksize "$AUG_CHUNKSIZE" \
        --storage-mode "$AUG_STORAGE_MODE" --sidecar-subdir "$AUG_SIDECAR_SUBDIR" \
        "${force_aug_args[@]}"
    ) > >(tee "$OUT_ROOT/logs/augment_train.log") 2> >(tee -a "$OUT_ROOT/logs/augment_train.log" >&2) &
    pids+=("$!")
  fi
  if ! cache_ready "$VAL_CACHE" || [[ "$FORCE_AUGMENT" == "1" ]]; then
    mkdir -p "$VAL_CACHE"
    (
      env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$PYTHON_BIN" -u -m cowp.scripts.26_augment_transport_labels \
        --data-config configs/data.yaml --label-config "$LABEL_CFG" \
        --input-dir "$RAW_VAL_CACHE" --output-dir "$VAL_CACHE" \
        --num-workers "$AUG_VAL_WORKERS" --chunksize "$AUG_CHUNKSIZE" \
        --storage-mode "$AUG_STORAGE_MODE" --sidecar-subdir "$AUG_SIDECAR_SUBDIR" \
        "${force_aug_args[@]}"
    ) > >(tee "$OUT_ROOT/logs/augment_val.log") 2> >(tee -a "$OUT_ROOT/logs/augment_val.log" >&2) &
    pids+=("$!")
  fi
  for pid in "${pids[@]:-}"; do [[ -n "$pid" ]] && wait "$pid"; done
fi
cache_ready "$TRAIN_CACHE" || { echo "Transport train cache is not ready: $TRAIN_CACHE" >&2; exit 2; }
cache_ready "$VAL_CACHE" || { echo "Transport val cache is not ready: $VAL_CACHE" >&2; exit 2; }

if [[ "$RUN_DIAGNOSE" == "1" ]]; then
  logrun diagnose_transport_train "$PYTHON_BIN" -u -m cowp.scripts.27_diagnose_transport_labels \
    --cache-dir "$TRAIN_CACHE" --output "$OUT_ROOT/eval/transport_train_diagnostics.json" & d0=$!
  logrun diagnose_transport_val "$PYTHON_BIN" -u -m cowp.scripts.27_diagnose_transport_labels \
    --cache-dir "$VAL_CACHE" --output "$OUT_ROOT/eval/transport_val_diagnostics.json" & d1=$!
  wait "$d0"; wait "$d1"
fi

best_transport() {
  local p="$OUT_ROOT/checkpoints/transport/cowp_witness_best.pt"
  [[ -s "$p" ]] && { echo "$p"; return; }
  return 1
}
best_planner() {
  local p="$OUT_ROOT/checkpoints/planner/cowp_planner_best.pt"
  [[ -s "$p" ]] && { echo "$p"; return; }
  return 1
}

# Stage 1: learn explicit mode conflict/retention and same-root response recovery.
if [[ "$RUN_TRANSPORT" == "1" && "$REQUIRE_INIT_CKPT" == "1" && ! -s "$INIT_CKPT" ]]; then
  echo "Required v8 initialization checkpoint is missing: $INIT_CKPT" >&2
  echo "Set INIT_CKPT to the actual v8 planner checkpoint, or REQUIRE_INIT_CKPT=0 to train from scratch." >&2
  exit 2
fi
if [[ "$RUN_TRANSPORT" == "1" ]]; then
  if best_transport >/dev/null && [[ "$FORCE_TRAIN" != "1" ]]; then
    echo "[transport] keep existing $(best_transport)"
  else
    resume_args=()
    [[ -s "$INIT_CKPT" ]] && resume_args=(--resume "$INIT_CKPT")
    logrun train_transport_ddp env CUDA_VISIBLE_DEVICES="$GPU0,$GPU1" \
      "$TORCHRUN_BIN" --standalone --nproc_per_node=2 -m cowp.scripts.03_train \
      --data-config configs/data.yaml --model-config configs/model.yaml \
      --label-config "$LABEL_CFG" --train-config "$TRAIN_CFG" \
      --cache-dir "$TRAIN_CACHE" --val-cache-dir "$VAL_CACHE" \
      --stage witness --epochs "$TRANSPORT_EPOCHS" --batch-size "$BATCH_PER_GPU" \
      --lr "$TRANSPORT_LR" --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" \
      --device cuda --output-dir "$OUT_ROOT/checkpoints/transport" \
      --freeze-backbone-epochs "$FREEZE_BACKBONE_EPOCHS" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" --early-stop-min-delta 1e-4 \
      --lr-scheduler plateau --min-lr 2e-6 --save-every 2 \
      --no-positive-oversampling --no-response-traj --no-response-components \
      --amp --fused-adamw "${resume_args[@]}"
  fi
fi
if [[ -z "$TRANSPORT_CKPT" ]]; then TRANSPORT_CKPT="$(best_transport || true)"; fi
[[ -s "$TRANSPORT_CKPT" ]] || { echo "No transport checkpoint. Set TRANSPORT_CKPT or enable RUN_TRANSPORT." >&2; exit 2; }

# Stage 2: train planner/candidate shields without rewriting the transport backbone.
if [[ "$RUN_PLANNER" == "1" ]]; then
  if best_planner >/dev/null && [[ "$FORCE_TRAIN" != "1" ]]; then
    echo "[planner] keep existing $(best_planner)"
  else
    logrun train_planner_ddp env CUDA_VISIBLE_DEVICES="$GPU0,$GPU1" \
      "$TORCHRUN_BIN" --standalone --nproc_per_node=2 -m cowp.scripts.03_train \
      --data-config configs/data.yaml --model-config configs/model.yaml \
      --label-config "$LABEL_CFG" --train-config "$TRAIN_CFG" \
      --cache-dir "$TRAIN_CACHE" --val-cache-dir "$VAL_CACHE" \
      --stage planner --epochs "$PLANNER_EPOCHS" --batch-size "$BATCH_PER_GPU" \
      --lr "$PLANNER_LR" --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" \
      --device cuda --output-dir "$OUT_ROOT/checkpoints/planner" \
      --with-waymax-outcome-labels --freeze-backbone-epochs "$FREEZE_BACKBONE_EPOCHS" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" --early-stop-min-delta 1e-4 \
      --lr-scheduler plateau --min-lr 1e-6 --save-every 2 \
      --no-positive-oversampling --no-response-traj --no-response-components \
      --amp --fused-adamw --resume "$TRANSPORT_CKPT"
  fi
fi
if [[ -z "$CKPT" ]]; then CKPT="$(best_planner || true)"; fi
[[ -s "$CKPT" ]] || { echo "No v9 planner checkpoint. Set CKPT or enable RUN_PLANNER." >&2; exit 2; }
echo "[checkpoint] $CKPT"

if [[ "$RUN_OFFLINE" == "1" ]]; then
  combined="$OUT_ROOT/eval/learned_offline/_shared_model_pass.json"
  if ! json_valid "$combined"; then
    logrun learned_offline env CUDA_VISIBLE_DEVICES="$GPU0" \
      "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
      --mode learned_offline \
      --methods planner_score_only,conventional_safety,soft_burden_cost_only,universal_ncf,cowp \
      --checkpoint "$CKPT" --cache-dir "$VAL_CACHE" \
      --data-config configs/data.yaml --label-config "$LABEL_CFG" --eval-config "$EVAL_CFG" \
      --device cuda --batch-size 20 --num-workers 4 --prefetch-factor 1 \
      --witness-threshold "$DEFAULT_WITNESS_THRESHOLD" --witness-threshold-sweep "$WITNESS_SWEEP" \
      --ncf-gate-mode "$NCF_GATE_MODE" --offline-fallback stop_like \
      --outcome-risk-penalty "$OFFLINE_OUTCOME_RISK_PENALTY" --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
      --output "$combined" --no-progress
  fi
  logrun calibrate_witness "$PYTHON_BIN" -u -m cowp.scripts.18_calibrate_witness_threshold \
    --input "$combined" --method cowp \
    --output "$OUT_ROOT/eval/learned_offline/witness_calibration.json" \
    --min-ncf-recall 0.25 --max-fallback 0.30
  # This command exits non-zero and stops the script before Waymax if the core
  # mechanism remains uninformative or collapses the feasible set.
  logrun verify_mechanism "$PYTHON_BIN" -u -m cowp.scripts.25_verify_mechanism_effect \
    --input "$combined" --method cowp \
    --min-unique-selection-points 3 --min-ncf-recall 0.25 \
    --min-witness-auprc 0.50 --min-accepted-rate 0.08 --max-fallback 0.30 \
    --min-false-safe-improvement 0.03 \
    --output "$OUT_ROOT/eval/learned_offline/mechanism_verification.json"
fi

CALIBRATION_JSON="$OUT_ROOT/eval/learned_offline/witness_calibration.json"
if [[ -s "$CALIBRATION_JSON" ]]; then
  ONLINE_WITNESS_THRESHOLD="$("$PYTHON_BIN" - "$CALIBRATION_JSON" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["witness_threshold"])
PY
)"
else
  ONLINE_WITNESS_THRESHOLD="$DEFAULT_WITNESS_THRESHOLD"
fi
echo "[witness threshold] $ONLINE_WITNESS_THRESHOLD"

run_online_one() {
  local method="$1" gpu="$2" scenarios="$3" out="$4" log="$5" label_cfg="$6" shard_count="${7:-1}" shard_index="${8:-0}" outcome_penalty="${9:-$ONLINE_OUTCOME_RISK_PENALTY}"
  local cfg_tag cache
  cfg_tag="$(basename "$label_cfg" .yaml)"
  cache="$OUT_ROOT/jax_cache/${method}_${cfg_tag}_g${gpu}_s${shard_index}"
  mkdir -p "$cache"
  (
    CUDA_VISIBLE_DEVICES="$gpu" JAX_COMPILATION_CACHE_DIR="$cache" \
    "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
      --mode waymax --method "$method" --checkpoint "$CKPT" --cache-dir "$VAL_CACHE" \
      --data-config configs/data.yaml --label-config "$label_cfg" --eval-config "$EVAL_CFG" \
      --tfexample-glob "$WAYMAX_VAL" --num-shards "$shard_count" --shard-index "$shard_index" \
      --num-scenarios "$scenarios" --rollout-horizon-steps "$ROLLOUT_HORIZON" \
      --device auto --waymax-device gpu --waymax-action-mode "$WAYMAX_ACTION_MODE" \
      --jax-visible-devices 0 --jax-preallocate false --waymax-standard-metrics \
      --reuse-waymax-env --prefilter-waymax-shards --jit-waymax-env --jit-waymax-metrics \
      --witness-threshold "$ONLINE_WITNESS_THRESHOLD" --ncf-gate-mode "$NCF_GATE_MODE" \
      --outcome-risk-penalty "$outcome_penalty" --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
      --output "$out" --no-progress
  ) > >(tee "$log") 2> >(tee -a "$log" >&2)
}

if [[ "$RUN_PROBE" == "1" ]]; then
  probe_cowp="$OUT_ROOT/eval/probe/cowp_root_transport_${PROBE_SCENARIOS}.json"
  probe_conv="$OUT_ROOT/eval/probe/conventional_safety_${PROBE_SCENARIOS}.json"
  if ! json_valid "$probe_cowp" || ! json_valid "$probe_conv"; then
    run_online_one cowp "$GPU0" "$PROBE_SCENARIOS" "$probe_cowp" "$OUT_ROOT/logs/probe_cowp_root_transport.log" "$LABEL_CFG" & p0=$!
    run_online_one conventional_safety "$GPU1" "$PROBE_SCENARIOS" "$probe_conv" "$OUT_ROOT/logs/probe_conventional.log" "$LABEL_CFG" & p1=$!
    wait "$p0"; wait "$p1"
  fi
  logrun summarize_probe "$PYTHON_BIN" -u -m cowp.scripts.24_summarize_planner_delta \
    --reference "$probe_conv" --candidate "$probe_cowp" \
    --output "$OUT_ROOT/eval/probe/delta_conventional_vs_root_transport.json"

  # A second two-GPU wave isolates the semantic certificate while testing the
  # Pareto selector ablation.  These runs use the same checkpoint and can execute
  # independently, so keeping them parallel avoids leaving either GPU idle.
  second_wave_pids=()
  if [[ "$RUN_SEMANTIC_PROBE" == "1" ]]; then
    probe_semantic="$OUT_ROOT/eval/probe/cowp_semantic_only_${PROBE_SCENARIOS}.json"
    if ! json_valid "$probe_semantic"; then
      run_online_one cowp "$GPU0" "$PROBE_SCENARIOS" "$probe_semantic" "$OUT_ROOT/logs/probe_cowp_semantic_only.log" "$LABEL_CFG" 1 0 0.0 &
      second_wave_pids+=("$!")
    fi
  fi
  if [[ "$RUN_PARETO_ABLATION" == "1" ]]; then
    probe_pareto="$OUT_ROOT/eval/probe/cowp_pareto_${PROBE_SCENARIOS}.json"
    if ! json_valid "$probe_pareto"; then
      run_online_one cowp "$GPU1" "$PROBE_SCENARIOS" "$probe_pareto" "$OUT_ROOT/logs/probe_cowp_pareto.log" "$PARETO_CFG" &
      second_wave_pids+=("$!")
    fi
  fi
  for pid in "${second_wave_pids[@]:-}"; do [[ -n "$pid" ]] && wait "$pid"; done

  if [[ "$RUN_SEMANTIC_PROBE" == "1" ]]; then
    logrun summarize_semantic_probe "$PYTHON_BIN" -u -m cowp.scripts.24_summarize_planner_delta \
      --reference "$probe_conv" --candidate "$probe_semantic" \
      --output "$OUT_ROOT/eval/probe/delta_conventional_vs_semantic_only.json"
  fi
  if [[ "$RUN_PARETO_ABLATION" == "1" ]]; then
    logrun summarize_frontier_ablation "$PYTHON_BIN" -u -m cowp.scripts.24_summarize_planner_delta \
      --reference "$probe_pareto" --candidate "$probe_cowp" \
      --output "$OUT_ROOT/eval/probe/delta_pareto_vs_root_transport.json"
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
  c0="$OUT_ROOT/eval/waymax/cowp_root_transport_shard0.json"
  c1="$OUT_ROOT/eval/waymax/cowp_root_transport_shard1.json"
  run_online_one cowp "$GPU0" "$per_shard" "$c0" "$OUT_ROOT/logs/full_cowp_shard0.log" "$LABEL_CFG" 2 0 & p0=$!
  run_online_one cowp "$GPU1" "$per_shard" "$c1" "$OUT_ROOT/logs/full_cowp_shard1.log" "$LABEL_CFG" 2 1 & p1=$!
  wait "$p0"; wait "$p1"
  logrun merge_cowp "$PYTHON_BIN" -u -m cowp.scripts.17_merge_waymax_shards \
    --output "$OUT_ROOT/eval/waymax/cowp_root_transport_merged.json" --inputs "$c0" "$c1"
fi

echo "[cowp_v9] complete: $OUT_ROOT"
