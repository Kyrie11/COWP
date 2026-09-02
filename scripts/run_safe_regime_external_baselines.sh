#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# shellcheck source=scripts/lib/v50_runtime.sh
source scripts/lib/v50_runtime.sh

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${TRAIN_SAFE:=$OCRAP_ROOT/train_safe}"
: "${VAL_SAFE:=$OCRAP_ROOT/val_safe}"
: "${TEST_SAFE:=$OCRAP_ROOT/test_safe}"
: "${RUN:=runs/safe_external_baselines}"
: "${WOMD_VAL:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
: "${CL_WOMD:=$WOMD_VAL}"
: "${WOMD_NUM_SHARDS:=150}"
CL_WOMD="$(v50_normalize_womd_spec "$CL_WOMD" "$WOMD_NUM_SHARDS")"
: "${CL_MAX_SCENARIOS:=50}"
: "${CL_BUCKET_DATASET:=$TEST_SAFE}"
: "${CL_BUCKET_SPLIT:=test}"
: "${CL_MAX_TARGETS_PER_SCENE:=1}"
: "${CL_TARGET_KEYS_FILE:=}"
: "${CL_RENDER_TRACE:=false}"
: "${CL_RENDER_MAX_AGENTS:=48}"
: "${CL_PREFLIGHT:=true}"
: "${CL_MAX_STEPS:=40}"
: "${CL_REPLAN_INTERVAL_STEPS:=1}"
: "${CL_NUM_CANDIDATES:=24}"
: "${CL_LABEL_MODE:=fast}"
: "${CL_AUDIT_EVERY_N_STEPS:=0}"
: "${CL_SAVE_PARTIAL:=true}"
: "${CL_PROFILE_TIMING:=true}"
: "${CL_RESUME_FORCE:=false}"
: "${CL_PARTIAL_WRITE_EVERY_SCENES:=32}"
: "${CL_PROGRESS_EVERY_STEPS:=10}"
: "${SKIP_COMPLETE_METHODS:=true}"
: "${DO_TRAIN:=true}"                 # permit training missing/invalid checkpoints
: "${FORCE_RETRAIN_SAFE:=false}"      # explicit architecture/data retraining
: "${DO_OFFLINE:=true}"
: "${DO_CLOSED_LOOP:=true}"
: "${CUDA_DEVICES:=0,1}"
: "${OCRAP_SDPA_BACKEND:=safe}"  # safe keeps Flash/MEM-efficient/math and disables only cuDNN SDPA
: "${OCRAP_AMP_DTYPE:=auto}"    # BF16 on supported GPUs, otherwise FP16
: "${CHECKPOINT_ROOT:=$RUN}"
: "${WAYFORMER_CHECKPOINT:=$CHECKPOINT_ROOT/wayformer_bc/best.pt}"
: "${GAMEFORMER_SAFE_CHECKPOINT:=$CHECKPOINT_ROOT/gameformer_lite/best.pt}"
: "${BETOPNET_CHECKPOINT:=$CHECKPOINT_ROOT/betopnet_lite/best.pt}"

IFS=',' read -r -a GPU_LIST <<< "$CUDA_DEVICES"
((${#GPU_LIST[@]})) || GPU_LIST=(0 1)
: "${MAX_PARALLEL:=${#GPU_LIST[@]}}"
((MAX_PARALLEL <= ${#GPU_LIST[@]})) || MAX_PARALLEL="${#GPU_LIST[@]}"
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)"
: "${THREADS_PER_JOB:=$(( CPU_COUNT / MAX_PARALLEL ))}"
((THREADS_PER_JOB >= 1)) || THREADS_PER_JOB=1
((THREADS_PER_JOB <= 8)) || THREADS_PER_JOB=8
: "${JAX_CACHE_DIR:=$RUN/.jax_compilation_cache}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
export RUN CL_WOMD
mkdir -p "$RUN" "$JAX_CACHE_DIR"

if v50_bool_true "$DO_CLOSED_LOOP" && v50_bool_true "$CL_PREFLIGHT"; then
  preflight_target_args=()
  if [[ -n "$CL_TARGET_KEYS_FILE" ]]; then
    preflight_target_args=(--target-keys-file "$CL_TARGET_KEYS_FILE" --require-target-keys)
  fi
  python tools/check_closed_loop_dataset_support.py \
    --dataset "$CL_BUCKET_DATASET" --split "$CL_BUCKET_SPLIT" --womd-pattern "$CL_WOMD" \
    --expected-source-role auto "${preflight_target_args[@]}" \
    --output "$RUN/closed_loop_dataset_support.json"
fi

# method|config|checkpoint
SPECS=(
  "wayformer_bc|configs/external_baselines/wayformer_bc.yaml|$WAYFORMER_CHECKPOINT"
  "gameformer_lite|configs/external_baselines/gameformer_lite.yaml|$GAMEFORMER_SAFE_CHECKPOINT"
  "betopnet_lite|configs/external_baselines/betopnet_lite.yaml|$BETOPNET_CHECKPOINT"
)
CLOSED_LOOP_SPECS=(
  "nominal_replay|configs/external_baselines/nominal_log_replay.yaml|"
  "${SPECS[@]}"
)

run_env() {
  local gpu="$1"; shift
  local gpu_cache="$JAX_CACHE_DIR/gpu_${gpu//[^[:alnum:]_.-]/_}"
  mkdir -p "$gpu_cache"
  env CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    TF_NUM_INTRAOP_THREADS="$THREADS_PER_JOB" TF_NUM_INTEROP_THREADS=2 MALLOC_ARENA_MAX=4 \
    XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \
    TF_FORCE_GPU_ALLOW_GROWTH=true JAX_ENABLE_X64=0 \
    JAX_COMPILATION_CACHE_DIR="$gpu_cache" JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
    PYTHONUNBUFFERED=1 "$@"
}

checkpoint_valid() {
  local ckpt="$1"
  [[ -f "$ckpt" ]] && python tools/validate_external_checkpoint.py \
    --checkpoint "$ckpt" --require-deployable-contract >/dev/null 2>&1
}

run_batches() {
  local runner="$1"; shift
  local items=("$@") base j idx failed
  for ((base=0; base<${#items[@]}; base+=MAX_PARALLEL)); do
    local pids=() names=(); failed=0
    for ((j=0; j<MAX_PARALLEL && base+j<${#items[@]}; j++)); do
      idx=$((base+j)); "$runner" "${items[$idx]}" "$idx" &
      pids+=("$!"); names+=("${items[$idx]%%|*}")
    done
    for j in "${!pids[@]}"; do
      if ! wait "${pids[$j]}"; then echo "[ERROR] ${names[$j]} failed" >&2; failed=1; fi
    done
    [[ "$failed" -eq 0 ]] || return 1
  done
}

run_train_eval_job() {
  local spec="$1" idx="$2" method config ckpt gpu train_dir
  IFS='|' read -r method config ckpt <<< "$spec"
  gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  train_dir="$(dirname "$ckpt")"
  if ! v50_bool_true "$DO_OFFLINE" && v50_bool_true "$DO_CLOSED_LOOP" && v50_bool_true "$SKIP_COMPLETE_METHODS" \
      && python tools/check_closed_loop_artifact.py --output "$RUN/closed_loop_${method}.json" --quiet; then
    echo "[REUSE] safe method=$method already has a complete closed-loop artifact; checkpoint preparation skipped"
    return 0
  fi
  echo "[START] safe train/eval baseline=$method gpu=$gpu checkpoint=$ckpt"
  if v50_bool_true "$FORCE_RETRAIN_SAFE" || ! checkpoint_valid "$ckpt"; then
    if ! v50_bool_true "$DO_TRAIN"; then
      echo "Missing/invalid checkpoint and training disabled: $ckpt" >&2; return 2
    fi
    mkdir -p "$train_dir"
    run_env "$gpu" python -u -m ocrap.cli train-baseline \
      --config "$config" --dataset "$TRAIN_SAFE" --val-dataset "$VAL_SAFE" \
      --baseline "$method" --output "$train_dir" \
      --set external_baselines.training.tqdm=false \
      --set "external_baselines.training.sdpa_backend=$OCRAP_SDPA_BACKEND" \
      --set "external_baselines.training.amp_dtype=$OCRAP_AMP_DTYPE" \
      2>&1 | tee "$RUN/train_${method}.log"
    checkpoint_valid "$ckpt" || { echo "Training produced an invalid checkpoint: $ckpt" >&2; return 2; }
  else
    echo "[REUSE] validated checkpoint $ckpt"
  fi
  if v50_bool_true "$DO_OFFLINE"; then
    run_env "$gpu" python -u -m ocrap.cli evaluate-baseline \
      --config "$config" --dataset "$TEST_SAFE" --checkpoint "$ckpt" \
      --split test --output "$RUN/eval_safe_${method}.json" --baselines "$method" \
      2>&1 | tee "$RUN/eval_safe_${method}.log"
  fi
  echo "[DONE] safe train/eval baseline=$method gpu=$gpu"
}

if v50_bool_true "$DO_OFFLINE"; then
  env CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate-baseline \
    --config configs/external_baselines/nominal_log_replay.yaml \
    --dataset "$TEST_SAFE" --split test \
    --output "$RUN/eval_safe_nominal_log_replay.json" \
    --baselines nominal_replay,log_replay \
    2>&1 | tee "$RUN/eval_safe_nominal_log_replay.log"
fi

if v50_bool_true "$DO_TRAIN" || v50_bool_true "$DO_OFFLINE" || v50_bool_true "$DO_CLOSED_LOOP"; then
  run_batches run_train_eval_job "${SPECS[@]}"
fi

run_closed_loop_method() {
  local spec="$1" idx="$2" method runtime_method config ckpt gpu
  IFS='|' read -r method config ckpt <<< "$spec"
  # ``nominal_replay`` is the reporting alias used in tables. The core
  # evaluator's deployable closed-loop method is named ``nominal``.
  runtime_method="$method"
  [[ "$method" == nominal_replay ]] && runtime_method=nominal
  gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  local output="$RUN/closed_loop_${method}.json"
  if v50_bool_true "$SKIP_COMPLETE_METHODS" && python tools/check_closed_loop_artifact.py --output "$output" --quiet; then
    echo "[REUSE] safe closed-loop method=$method is already complete: $output"
    return 0
  fi
  local checkpoint_args=() target_args=()
  if [[ "$method" != nominal_replay ]]; then
    checkpoint_valid "$ckpt" || { echo "Missing/invalid checkpoint: $ckpt" >&2; return 2; }
    checkpoint_args=(--checkpoint "$ckpt")
  fi
  if [[ -n "$CL_TARGET_KEYS_FILE" ]]; then
    target_args=(--set "closed_loop.target_keys_file=$CL_TARGET_KEYS_FILE" --set closed_loop.require_target_keys=true)
  fi
  echo "[START] safe closed-loop method=$method gpu=$gpu"
  run_env "$gpu" python -u -m ocrap.cli closed-loop \
    --config "$config" --dataset "$CL_WOMD" "${checkpoint_args[@]}" \
    --output "$output" \
    --set "closed_loop.method=$runtime_method" \
    --set "closed_loop.max_scenarios=$CL_MAX_SCENARIOS" \
    --set "closed_loop.max_bucket_targets=$CL_MAX_SCENARIOS" \
    --set "closed_loop.bucket_dataset=$CL_BUCKET_DATASET" \
    --set "closed_loop.bucket_split=$CL_BUCKET_SPLIT" \
    --set closed_loop.require_bucket_targets=true \
    --set "closed_loop.max_targets_per_scene=$CL_MAX_TARGETS_PER_SCENE" \
    --set "closed_loop.render_trace=$CL_RENDER_TRACE" \
    --set "closed_loop.render_max_agents=$CL_RENDER_MAX_AGENTS" \
    --set "closed_loop.max_steps=$CL_MAX_STEPS" \
    --set "closed_loop.replan_interval_steps=$CL_REPLAN_INTERVAL_STEPS" \
    --set "closed_loop.label_mode=$CL_LABEL_MODE" \
    --set closed_loop.force_teacher_baselines=false \
    --set "closed_loop.num_candidate_prefixes=$CL_NUM_CANDIDATES" \
    --set "closed_loop.audit_every_n_steps=$CL_AUDIT_EVERY_N_STEPS" \
    --set "closed_loop.save_partial=$CL_SAVE_PARTIAL" \
    --set "closed_loop.resume_force=$CL_RESUME_FORCE" \
    --set "closed_loop.partial_write_every_scenes=$CL_PARTIAL_WRITE_EVERY_SCENES" \
    --set "closed_loop.progress_every_steps=$CL_PROGRESS_EVERY_STEPS" \
    --set closed_loop.result_scene_detail=metrics \
    --set closed_loop.scene_journal_detail=metrics \
    --set closed_loop.memory_scene_detail=metrics \
    --set closed_loop.include_scenes_in_result=false \
    --set closed_loop.include_scenes_in_partial=false \
    --set "closed_loop.profile_timing=$CL_PROFILE_TIMING" \
    --set waymax.dataloader_include_sdc_paths=false \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.use_jit_scan_rollouts=true \
    "${target_args[@]}" \
    2>&1 | tee "$RUN/closed_loop_${method}.log"
  echo "[DONE] safe closed-loop method=$method gpu=$gpu"
}

if v50_bool_true "$DO_CLOSED_LOOP"; then
  run_batches run_closed_loop_method "${CLOSED_LOOP_SPECS[@]}"
fi

python - <<'PY'
import glob, json, os
run=os.environ['RUN']
offline=[]; closed=[]
for p in sorted(glob.glob(os.path.join(run,'eval_safe_*.json'))):
    try: d=json.load(open(p))
    except Exception: continue
    for method, values in (d.get('methods') or {}).items(): offline.append({'method':method, **values})
for p in sorted(glob.glob(os.path.join(run,'closed_loop_*.json'))):
    if p.endswith(('.progress.json','.partial')): continue
    try: d=json.load(open(p))
    except Exception: continue
    closed.append({k:d.get(k) for k in ['method','source','label_mode','num_scenes','num_decisions','collision_scene_rate','offroad_scene_rate','closed_loop_bounded_NUP','intervention_rate','timing']})
out=os.path.join(run,'safe_external_baselines_summary.json')
json.dump({'offline':offline,'closed_loop':closed,'womd_spec':os.environ.get('CL_WOMD')}, open(out,'w'), indent=2)
print({'event':'safe_external_baselines_summary','output':out,'offline_methods':len(offline),'closed_loop_methods':len(closed)})
PY
