#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# shellcheck source=scripts/lib/v50_runtime.sh
source scripts/lib/v50_runtime.sh

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${TEST_CONTACT:=$OCRAP_ROOT/test_contact}"
: "${RUN:=runs/contact_external_baselines}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord@150}"
: "${CL_WOMD:=$WOMD_VAL_INTERACTIVE}"
: "${WOMD_NUM_SHARDS:=150}"
CL_WOMD="$(v50_normalize_womd_spec "$CL_WOMD" "$WOMD_NUM_SHARDS")"
: "${CL_MAX_SCENARIOS:=50}"
: "${CL_BUCKET_DATASET:=$TEST_CONTACT}"
: "${CL_BUCKET_SPLIT:=test}"
: "${CL_MAX_TARGETS_PER_SCENE:=1}"
: "${CL_TARGET_KEYS_FILE:=}"
: "${CL_RENDER_TRACE:=false}"
: "${CL_RENDER_MAX_AGENTS:=48}"
: "${CL_PREFLIGHT:=true}"
: "${CL_MAX_STEPS:=40}"
: "${CL_REPLAN_INTERVAL_STEPS:=1}"
: "${CL_NUM_CANDIDATES:=24}"
: "${CL_NUM_RECOVERY_OPTIONS:=12}"
: "${CL_LABEL_MODE:=fast}"            # Contact main table uses physical recovery metrics only.
: "${CL_AUDIT_EVERY_N_STEPS:=0}"
: "${CL_SAVE_PARTIAL:=true}"
: "${CL_PROFILE_TIMING:=true}"
: "${CL_RESUME_FORCE:=false}"
: "${CL_PARTIAL_WRITE_EVERY_SCENES:=32}"
: "${CL_PROGRESS_EVERY_STEPS:=10}"
: "${SKIP_COMPLETE_METHODS:=true}"
: "${DO_OFFLINE:=true}"
: "${DO_CLOSED_LOOP:=true}"
: "${CUDA_DEVICES:=0,1}"

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

METHODS=(postimpact_mpc_lite post_crash_braking post_collision_restoration severity_minimization)
METHODS_CSV="$(IFS=,; echo "${METHODS[*]}")"

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

run_batches() {
  local runner="$1"; shift; local items=("$@") base j idx failed
  for ((base=0; base<${#items[@]}; base+=MAX_PARALLEL)); do
    local pids=() names=(); failed=0
    for ((j=0; j<MAX_PARALLEL && base+j<${#items[@]}; j++)); do
      idx=$((base+j)); "$runner" "${items[$idx]}" "$idx" & pids+=("$!"); names+=("${items[$idx]}")
    done
    for j in "${!pids[@]}"; do if ! wait "${pids[$j]}"; then echo "[ERROR] ${names[$j]} failed" >&2; failed=1; fi; done
    [[ "$failed" -eq 0 ]] || return 1
  done
}

if v50_bool_true "$DO_OFFLINE"; then
  run_env "${GPU_LIST[0]}" python -u -m ocrap.cli evaluate-baseline \
    --config configs/external_baselines/contact_external_baselines.yaml \
    --dataset "$TEST_CONTACT" --split test \
    --output "$RUN/eval_contact_external_baselines.json" --baselines "$METHODS_CSV" \
    2>&1 | tee "$RUN/eval_contact_external_baselines.log"
fi

run_closed_loop_method() {
  local method="$1" idx="$2" gpu target_args=()
  gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  local output="$RUN/closed_loop_${method}.json"
  if v50_bool_true "$SKIP_COMPLETE_METHODS" && python tools/check_closed_loop_artifact.py --output "$output" --quiet; then
    echo "[REUSE] contact closed-loop method=$method is already complete: $output"
    return 0
  fi
  if [[ -n "$CL_TARGET_KEYS_FILE" ]]; then target_args=(--set "closed_loop.target_keys_file=$CL_TARGET_KEYS_FILE" --set closed_loop.require_target_keys=true); fi
  echo "[START] contact method=$method gpu=$gpu"
  run_env "$gpu" python -u -m ocrap.cli closed-loop \
    --config configs/external_baselines/contact_external_baselines.yaml \
    --dataset "$CL_WOMD" --output "$output" \
    --set "closed_loop.method=$method" \
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
    --set closed_loop.external_sparse_labels=true \
    --set closed_loop.exhaustive_teacher_labels=false \
    --set "closed_loop.num_candidate_prefixes=$CL_NUM_CANDIDATES" \
    --set "closed_loop.num_recovery_options=$CL_NUM_RECOVERY_OPTIONS" \
    --set "closed_loop.save_partial=$CL_SAVE_PARTIAL" \
    --set "closed_loop.partial_write_every_scenes=$CL_PARTIAL_WRITE_EVERY_SCENES" \
    --set "closed_loop.progress_every_steps=$CL_PROGRESS_EVERY_STEPS" \
    --set closed_loop.result_scene_detail=metrics \
    --set closed_loop.scene_journal_detail=metrics \
    --set closed_loop.memory_scene_detail=metrics \
    --set closed_loop.include_scenes_in_result=false \
    --set closed_loop.include_scenes_in_partial=false \
    --set "closed_loop.profile_timing=$CL_PROFILE_TIMING" \
    --set "closed_loop.audit_every_n_steps=$CL_AUDIT_EVERY_N_STEPS" \
    --set "closed_loop.resume_force=$CL_RESUME_FORCE" \
    --set waymax.dataloader_include_sdc_paths=false \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.use_jit_scan_rollouts=true \
    "${target_args[@]}" \
    2>&1 | tee "$RUN/closed_loop_${method}.log"
  echo "[DONE] contact method=$method gpu=$gpu"
}

if v50_bool_true "$DO_CLOSED_LOOP"; then run_batches run_closed_loop_method "${METHODS[@]}"; fi

python - <<'PY'
import glob,json,os
run=os.environ['RUN']; rows=[]
for p in sorted(glob.glob(os.path.join(run,'closed_loop_*.json'))):
    if p.endswith(('.progress.json','.partial')): continue
    try:d=json.load(open(p))
    except Exception:continue
    rows.append({k:d.get(k) for k in ['method','source','label_mode','num_scenes','num_decisions','post_contact_terminal_clearance_m','post_contact_free_space_auc_normalized_m','post_contact_clearance_gain_m','post_contact_escape_scene_rate','recontact_scene_rate','secondary_overlap_scene_rate','new_stable_stop_quality_scene_rate','offroad_scene_rate','post_contact_overlap_duration_s','timing']})
out=os.path.join(run,'closed_loop_summary.json'); json.dump({'womd_spec':os.environ.get('CL_WOMD'),'methods':rows},open(out,'w'),indent=2)
print({'event':'contact_closed_loop_summary','output':out,'num_methods':len(rows)})
PY
