#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# shellcheck source=scripts/lib/v50_runtime.sh
source scripts/lib/v50_runtime.sh

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${TRAIN_NEAR:=$OCRAP_ROOT/train_near_contact}"
: "${VAL_NEAR:=$OCRAP_ROOT/val_near_contact}"
: "${TEST_NEAR:=$OCRAP_ROOT/test_near_contact}"
: "${RUN:=runs/near_contact_external_baselines_optimized}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord@150}"
: "${CL_WOMD:=$WOMD_VAL_INTERACTIVE}"
: "${WOMD_NUM_SHARDS:=150}"
CL_WOMD="$(v50_normalize_womd_spec "$CL_WOMD" "$WOMD_NUM_SHARDS")"
: "${CL_MAX_SCENARIOS:=50}"
: "${CL_BUCKET_DATASET:=$TEST_NEAR}"
: "${CL_BUCKET_SPLIT:=test}"
: "${CL_MAX_TARGETS_PER_SCENE:=1}"
: "${CL_TARGET_KEYS_FILE:=}"
: "${CL_RENDER_TRACE:=false}"
: "${CL_RENDER_MAX_AGENTS:=48}"
: "${CL_PREFLIGHT:=true}"
: "${CL_ORACLE_MAX_SCENARIOS:=20}"
: "${RUN_ORACLE_CLOSED_LOOP:=false}"  # teacher-only diagnostic, excluded from deployable comparison
: "${CL_MAX_STEPS:=40}"
: "${CL_REPLAN_INTERVAL_STEPS:=1}"
: "${CL_NUM_CANDIDATES:=24}"
: "${CL_NUM_RECOVERY_OPTIONS:=12}"
: "${CL_LABEL_MODE:=fast}"
: "${CL_AUDIT_EVERY_N_STEPS:=0}"
: "${CL_SAVE_PARTIAL:=true}"
: "${CL_PROFILE_TIMING:=true}"
: "${CL_RESUME_FORCE:=false}"
: "${CL_PARTIAL_WRITE_EVERY_SCENES:=32}"
: "${CL_PROGRESS_EVERY_STEPS:=10}"
: "${SKIP_COMPLETE_METHODS:=true}"
: "${USE_DYNAMIC_SCHEDULER:=auto}"
: "${DO_OFFLINE:=true}"
: "${DO_CLOSED_LOOP:=true}"
: "${TRAIN_GAMEFORMER_IF_MISSING:=true}"
: "${FORCE_RETRAIN_GAMEFORMER:=false}"
: "${CHECKPOINT_ROOT:=$RUN}"
: "${GAMEFORMER_CHECKPOINT:=$CHECKPOINT_ROOT/gameformer_lite/best.pt}"
: "${CUDA_DEVICES:=0,1}"
: "${GAMEFORMER_GLOBAL_BATCH_SIZE:=64}"
: "${GAMEFORMER_NUM_WORKERS_TOTAL:=8}"
: "${GAMEFORMER_TRAIN_GPUS:=2}"
: "${OCRAP_SDPA_BACKEND:=safe}"
: "${OCRAP_AMP_DTYPE:=auto}"

IFS=',' read -r -a GPU_LIST <<< "$CUDA_DEVICES"
((${#GPU_LIST[@]})) || GPU_LIST=(0 1)
: "${MAX_PARALLEL:=${#GPU_LIST[@]}}"
((MAX_PARALLEL <= ${#GPU_LIST[@]})) || MAX_PARALLEL="${#GPU_LIST[@]}"
((GAMEFORMER_TRAIN_GPUS <= ${#GPU_LIST[@]})) || GAMEFORMER_TRAIN_GPUS="${#GPU_LIST[@]}"
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)"
: "${THREADS_PER_JOB:=$(( CPU_COUNT / (2 * MAX_PARALLEL) ))}"
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

# Deployable methods only by default.  The oracle can be enabled for a separate,
# small teacher-only upper-bound audit without delaying the full experiment.
CLOSED_LOOP_METHODS=(gameformer_lite marc_lite racp_lite predictive_safety_filter dro_cvar_filter cvar_risk_filter expected_risk_filter)
if v50_bool_true "$RUN_ORACLE_CLOSED_LOOP"; then CLOSED_LOOP_METHODS=(oracle_recovery_filter "${CLOSED_LOOP_METHODS[@]}"); fi
NONLEARNED=(marc_lite racp_lite expected_risk_filter cvar_risk_filter dro_cvar_filter predictive_safety_filter oracle_recovery_filter)
NONLEARNED_CSV="$(IFS=,; echo "${NONLEARNED[*]}")"

common_env=(
  OMP_NUM_THREADS="$THREADS_PER_JOB"
  MKL_NUM_THREADS="$THREADS_PER_JOB"
  OPENBLAS_NUM_THREADS="$THREADS_PER_JOB"
  NUMEXPR_NUM_THREADS="$THREADS_PER_JOB"
  TF_NUM_INTRAOP_THREADS="$THREADS_PER_JOB"
  TF_NUM_INTEROP_THREADS=2
  MALLOC_ARENA_MAX=4
  XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE"
  TF_FORCE_GPU_ALLOW_GROWTH=true
  JAX_ENABLE_X64=0
  JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  PYTHONUNBUFFERED=1
)
run_env_gpu() {
  local gpu="$1"; shift
  local cache="$JAX_CACHE_DIR/gpu_${gpu//[^[:alnum:]_.-]/_}"
  mkdir -p "$cache"
  env CUDA_VISIBLE_DEVICES="$gpu" JAX_COMPILATION_CACHE_DIR="$cache" "${common_env[@]}" "$@"
}
run_env_cpu() {
  local cache="$JAX_CACHE_DIR/cpu"; mkdir -p "$cache"
  env CUDA_VISIBLE_DEVICES="" JAX_COMPILATION_CACHE_DIR="$cache" "${common_env[@]}" "$@"
}
join_first_gpus() { local count="$1" out="" i; for ((i=0;i<count;i++)); do [[ -n "$out" ]] && out+=","; out+="${GPU_LIST[$i]}"; done; printf '%s' "$out"; }

checkpoint_valid() {
  [[ -f "$GAMEFORMER_CHECKPOINT" ]] && python tools/validate_external_checkpoint.py \
    --checkpoint "$GAMEFORMER_CHECKPOINT" --require-deployable-contract >/dev/null 2>&1
}

train_gameformer() {
  local visible train_dir
  visible="$(join_first_gpus "$GAMEFORMER_TRAIN_GPUS")"
  train_dir="$(dirname "$GAMEFORMER_CHECKPOINT")"
  mkdir -p "$train_dir"
  local command=(python -u -m ocrap.cli train-baseline)
  if ((GAMEFORMER_TRAIN_GPUS > 1)); then
    command=(torchrun --standalone --nproc_per_node="$GAMEFORMER_TRAIN_GPUS" -m ocrap.cli train-baseline)
  fi
  local train_cache="$JAX_CACHE_DIR/train"; mkdir -p "$train_cache"
  env CUDA_VISIBLE_DEVICES="$visible" JAX_COMPILATION_CACHE_DIR="$train_cache" "${common_env[@]}" "${command[@]}" \
    --config configs/external_baselines/near_contact_gameformer_lite.yaml \
    --dataset "$TRAIN_NEAR" --val-dataset "$VAL_NEAR" \
    --baseline gameformer_lite --output "$train_dir" \
    --set "external_baselines.training.global_batch_size=$GAMEFORMER_GLOBAL_BATCH_SIZE" \
    --set "external_baselines.training.num_workers_total=$GAMEFORMER_NUM_WORKERS_TOTAL" \
    --set external_baselines.training.tqdm=false \
    --set "external_baselines.training.sdpa_backend=$OCRAP_SDPA_BACKEND" \
    --set "external_baselines.training.amp_dtype=$OCRAP_AMP_DTYPE" \
    2>&1 | tee "$RUN/train_gameformer_lite.log"
}

GAMEFORMER_ARTIFACT_COMPLETE=false
if v50_bool_true "$SKIP_COMPLETE_METHODS" && python tools/check_closed_loop_artifact.py \
    --output "$RUN/closed_loop_gameformer_lite.json" --quiet; then
  GAMEFORMER_ARTIFACT_COMPLETE=true
fi
NEED_GAMEFORMER_CHECKPOINT=true
if ! v50_bool_true "$DO_OFFLINE" && [[ "$GAMEFORMER_ARTIFACT_COMPLETE" == true ]]; then
  NEED_GAMEFORMER_CHECKPOINT=false
fi
if [[ "$NEED_GAMEFORMER_CHECKPOINT" == true ]]; then
  if v50_bool_true "$FORCE_RETRAIN_GAMEFORMER" || ! checkpoint_valid; then
    if ! v50_bool_true "$TRAIN_GAMEFORMER_IF_MISSING"; then
      echo "Missing/invalid GameFormer checkpoint and training disabled: $GAMEFORMER_CHECKPOINT" >&2; exit 2
    fi
    train_gameformer
  fi
  checkpoint_valid || { echo "Invalid GameFormer checkpoint after training: $GAMEFORMER_CHECKPOINT" >&2; exit 2; }
  echo "[REUSE/READY] GameFormer checkpoint $GAMEFORMER_CHECKPOINT"
else
  echo "[REUSE] complete GameFormer closed-loop artifact; checkpoint preparation skipped"
fi

if v50_bool_true "$DO_OFFLINE"; then
  run_env_cpu python -u -m ocrap.cli evaluate-baseline \
    --config configs/external_baselines/near_contact_external_baselines.yaml \
    --dataset "$TEST_NEAR" --split test \
    --output "$RUN/eval_near_contact_nonlearned.json" --baselines "$NONLEARNED_CSV" \
    2>&1 | tee "$RUN/eval_near_contact_nonlearned.log" & p_cpu=$!
  run_env_gpu "${GPU_LIST[0]}" python -u -m ocrap.cli evaluate-baseline \
    --config configs/external_baselines/near_contact_gameformer_lite.yaml \
    --dataset "$TEST_NEAR" --checkpoint "$GAMEFORMER_CHECKPOINT" --split test \
    --output "$RUN/eval_near_contact_gameformer_lite.json" --baselines gameformer_lite \
    2>&1 | tee "$RUN/eval_near_contact_gameformer_lite.log" & p_gpu=$!
  failed=0; wait "$p_cpu" || failed=1; wait "$p_gpu" || failed=1; ((failed == 0)) || exit 1
fi

run_closed_loop_method() {
  local method="$1" gpu="$2"
  local output="$RUN/closed_loop_${method}.json"
  if v50_bool_true "$SKIP_COMPLETE_METHODS" && python tools/check_closed_loop_artifact.py --output "$output" --quiet; then
    echo "[REUSE] near closed-loop method=$method is already complete: $output"
    return 0
  fi
  local config=configs/external_baselines/near_contact_external_baselines.yaml
  local label_mode="$CL_LABEL_MODE" max_scenes="$CL_MAX_SCENARIOS" exhaustive=false sparse=true
  local ckpt=() target_args=()
  if [[ "$method" == gameformer_lite ]]; then
    config=configs/external_baselines/near_contact_gameformer_lite.yaml
    ckpt=(--checkpoint "$GAMEFORMER_CHECKPOINT")
  elif [[ "$method" == oracle_recovery_filter ]]; then
    label_mode=all; exhaustive=true; sparse=false; max_scenes="$CL_ORACLE_MAX_SCENARIOS"
  fi
  if [[ -n "$CL_TARGET_KEYS_FILE" ]]; then
    target_args=(--set "closed_loop.target_keys_file=$CL_TARGET_KEYS_FILE" --set closed_loop.require_target_keys=true)
  fi
  echo "[START] near method=$method gpu=$gpu label_mode=$label_mode max_scenes=$max_scenes"
  run_env_gpu "$gpu" python -u -m ocrap.cli closed-loop \
    --config "$config" --dataset "$CL_WOMD" "${ckpt[@]}" \
    --output "$output" \
    --set "closed_loop.method=$method" \
    --set "closed_loop.max_scenarios=$max_scenes" \
    --set "closed_loop.max_bucket_targets=$max_scenes" \
    --set "closed_loop.bucket_dataset=$CL_BUCKET_DATASET" \
    --set "closed_loop.bucket_split=$CL_BUCKET_SPLIT" \
    --set closed_loop.require_bucket_targets=true \
    --set "closed_loop.max_targets_per_scene=$CL_MAX_TARGETS_PER_SCENE" \
    --set "closed_loop.render_trace=$CL_RENDER_TRACE" \
    --set "closed_loop.render_max_agents=$CL_RENDER_MAX_AGENTS" \
    --set "closed_loop.max_steps=$CL_MAX_STEPS" \
    --set "closed_loop.replan_interval_steps=$CL_REPLAN_INTERVAL_STEPS" \
    --set "closed_loop.label_mode=$label_mode" \
    --set closed_loop.force_teacher_baselines=false \
    --set "closed_loop.external_sparse_labels=$sparse" \
    --set "closed_loop.exhaustive_teacher_labels=$exhaustive" \
    --set "closed_loop.num_candidate_prefixes=$CL_NUM_CANDIDATES" \
    --set "closed_loop.num_recovery_options=$CL_NUM_RECOVERY_OPTIONS" \
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
    --set "closed_loop.audit_every_n_steps=$CL_AUDIT_EVERY_N_STEPS" \
    --set waymax.dataloader_include_sdc_paths=false \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.use_jit_scan_rollouts=true \
    "${target_args[@]}" \
    2>&1 | tee "$RUN/closed_loop_${method}.log"
  echo "[DONE] near method=$method gpu=$gpu"
}

run_closed_loop_dynamic() {
  local -a methods=("$@")
  local next=0 active=0 failed=0 gpu method done_pid status i
  declare -A PID_GPU=() PID_METHOD=()
  launch_one() { local m="$1" g="$2"; run_closed_loop_method "$m" "$g" & local p=$!; PID_GPU[$p]="$g"; PID_METHOD[$p]="$m"; active=$((active+1)); }
  for ((i=0; i<MAX_PARALLEL && next<${#methods[@]}; i++)); do launch_one "${methods[$next]}" "${GPU_LIST[$i]}"; next=$((next+1)); done
  while ((active>0)); do
    done_pid=""; if wait -n -p done_pid; then status=0; else status=$?; fi
    gpu="${PID_GPU[$done_pid]}"; method="${PID_METHOD[$done_pid]}"; unset 'PID_GPU[$done_pid]' 'PID_METHOD[$done_pid]'; active=$((active-1))
    if ((status!=0)); then echo "[ERROR] $method failed on GPU $gpu (status=$status)" >&2; failed=1; fi
    if ((next<${#methods[@]})); then launch_one "${methods[$next]}" "$gpu"; next=$((next+1)); fi
  done
  return "$failed"
}

run_closed_loop_fallback() {
  local -a methods=("$@") pids=() names=(); local base j idx failed=0
  for ((base=0; base<${#methods[@]}; base+=MAX_PARALLEL)); do
    pids=(); names=()
    for ((j=0; j<MAX_PARALLEL && base+j<${#methods[@]}; j++)); do idx=$((base+j)); run_closed_loop_method "${methods[$idx]}" "${GPU_LIST[$j]}" & pids+=("$!"); names+=("${methods[$idx]}"); done
    for j in "${!pids[@]}"; do wait "${pids[$j]}" || { echo "[ERROR] ${names[$j]} failed" >&2; failed=1; }; done
  done
  return "$failed"
}

supports_wait_pid_capture() {
  # Bash 5.0 has ``wait -n`` but not ``wait -p``. Version-only checks are
  # therefore incorrect on common enterprise distributions.
  help wait 2>/dev/null | grep -Eq -- '(^|[[:space:]])-p([[:space:]]|[[:punct:]])'
}

if v50_bool_true "$DO_CLOSED_LOOP"; then
  use_dynamic=false
  case "${USE_DYNAMIC_SCHEDULER,,}" in
    1|true|yes|on)
      supports_wait_pid_capture || { echo "USE_DYNAMIC_SCHEDULER requested but this Bash lacks wait -p" >&2; exit 2; }
      use_dynamic=true
      ;;
    auto|'')
      supports_wait_pid_capture && use_dynamic=true
      ;;
    0|false|no|off) use_dynamic=false ;;
    *) echo "Invalid USE_DYNAMIC_SCHEDULER=$USE_DYNAMIC_SCHEDULER" >&2; exit 2 ;;
  esac
  if [[ "$use_dynamic" == true ]]; then
    echo "[SCHEDULER] dynamic wait -n/-p"
    run_closed_loop_dynamic "${CLOSED_LOOP_METHODS[@]}"
  else
    echo "[SCHEDULER] portable fixed batches (wait -p unavailable or disabled)"
    run_closed_loop_fallback "${CLOSED_LOOP_METHODS[@]}"
  fi
fi

python - <<'PY'
import glob,json,os
run=os.environ['RUN']; rows=[]
for p in sorted(glob.glob(os.path.join(run,'closed_loop_*.json'))):
    if p.endswith(('.progress.json','.partial')): continue
    try:d=json.load(open(p))
    except Exception:continue
    rows.append({k:d.get(k) for k in ['method','source','label_mode','num_scenes','num_decisions','collision_scene_rate','offroad_scene_rate','scene_min_clearance_m_p05','scene_ttc_s_p05','critical_ttc_exposure_duration_s','closed_loop_bounded_NUP','intervention_rate','timing']})
out=os.path.join(run,'closed_loop_summary.json')
with open(out,'w') as f: json.dump({'womd_spec':os.environ.get('CL_WOMD'),'methods':rows},f,indent=2)
print({'event':'near_contact_closed_loop_summary','output':out,'num_methods':len(rows)})
PY
