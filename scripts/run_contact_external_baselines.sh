#!/usr/bin/env bash
set -euo pipefail

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${TEST_CONTACT:=$OCRAP_ROOT/test_contact}"
: "${RUN:=runs/contact_external_baselines}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}"
: "${CL_WOMD:=${WOMD_VAL_INTERACTIVE}@150}"
: "${CL_MAX_SCENARIOS:=50}"
: "${CL_MAX_STEPS:=40}"
: "${CL_REPLAN_INTERVAL_STEPS:=1}"
: "${CL_NUM_CANDIDATES:=24}"
: "${CL_AUDIT_EVERY_N_STEPS:=1}"
: "${CL_NUM_RECOVERY_OPTIONS:=12}"
: "${CL_SAVE_PARTIAL:=false}"
: "${CL_PROFILE_TIMING:=true}"
: "${DO_OFFLINE:=true}"
: "${DO_CLOSED_LOOP:=true}"
: "${CUDA_DEVICES:=0,1}"

IFS=',' read -r -a GPU_LIST <<< "$CUDA_DEVICES"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then GPU_LIST=(0 1); fi
: "${MAX_PARALLEL:=${#GPU_LIST[@]}}"
if [ "$MAX_PARALLEL" -gt "${#GPU_LIST[@]}" ]; then MAX_PARALLEL="${#GPU_LIST[@]}"; fi
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)"
: "${THREADS_PER_JOB:=$(( CPU_COUNT / MAX_PARALLEL ))}"
if [ "$THREADS_PER_JOB" -lt 1 ]; then THREADS_PER_JOB=1; fi
: "${JAX_CACHE_DIR:=$RUN/.jax_compilation_cache}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
export RUN
mkdir -p "$RUN" "$JAX_CACHE_DIR"

METHODS=(postimpact_mpc_lite post_crash_braking post_collision_restoration severity_minimization)
METHODS_CSV="$(IFS=,; echo "${METHODS[*]}")"

run_env() {
  local gpu="$1"; shift
  env CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \
    TF_FORCE_GPU_ALLOW_GROWTH=true JAX_ENABLE_X64=0 \
    JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR" JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
    PYTHONUNBUFFERED=1 "$@"
}

run_batches() {
  local runner="$1"; shift; local items=("$@") base j idx failed
  for ((base=0; base<${#items[@]}; base+=MAX_PARALLEL)); do
    local pids=() names=(); failed=0
    for ((j=0; j<MAX_PARALLEL && base+j<${#items[@]}; j++)); do
      idx=$((base+j)); "$runner" "${items[$idx]}" "$idx" &
      pids+=("$!"); names+=("${items[$idx]}")
    done
    for j in "${!pids[@]}"; do
      if ! wait "${pids[$j]}"; then echo "[ERROR] ${names[$j]} failed" >&2; failed=1; fi
    done
    [ "$failed" -eq 0 ] || return 1
  done
}

if [ "$DO_OFFLINE" = true ]; then
  run_env "${GPU_LIST[0]}" python -u -m ocrap.cli evaluate-baseline \
    --config configs/external_baselines/contact_external_baselines.yaml \
    --dataset "$TEST_CONTACT" --split test \
    --output "$RUN/eval_contact_external_baselines.json" --baselines "$METHODS_CSV" \
    2>&1 | tee "$RUN/eval_contact_external_baselines.log"
fi

run_closed_loop_method() {
  local method="$1"
  local idx="$2"
  local gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  echo "[START] contact method=$method gpu=$gpu"
  run_env "$gpu" python -u -m ocrap.cli closed-loop \
    --config configs/external_baselines/contact_external_baselines.yaml \
    --dataset "$CL_WOMD" --output "$RUN/closed_loop_${method}.json" \
    --set closed_loop.method="$method" \
    --set closed_loop.max_scenarios="$CL_MAX_SCENARIOS" \
    --set closed_loop.max_steps="$CL_MAX_STEPS" \
    --set closed_loop.replan_interval_steps="$CL_REPLAN_INTERVAL_STEPS" \
    --set closed_loop.label_mode=selected \
    --set closed_loop.force_teacher_baselines=false \
    --set closed_loop.external_sparse_labels=false \
    --set closed_loop.exhaustive_teacher_labels=false \
    --set closed_loop.num_candidate_prefixes="$CL_NUM_CANDIDATES" \
    --set closed_loop.num_recovery_options="$CL_NUM_RECOVERY_OPTIONS" \
    --set closed_loop.save_partial="$CL_SAVE_PARTIAL" \
    --set closed_loop.profile_timing="$CL_PROFILE_TIMING" \
    --set closed_loop.audit_every_n_steps="$CL_AUDIT_EVERY_N_STEPS" \
    --set closed_loop.resume_force=true \
    --set waymax.dataloader_include_sdc_paths=false \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.use_jit_scan_rollouts=true \
    2>&1 | tee "$RUN/closed_loop_${method}.log"
  echo "[DONE] contact method=$method gpu=$gpu"
}

if [ "$DO_CLOSED_LOOP" = true ]; then run_batches run_closed_loop_method "${METHODS[@]}"; fi

python - <<'PY'
import glob,json,os
run=os.environ['RUN']; rows=[]
for p in sorted(glob.glob(os.path.join(run,'closed_loop_*.json'))):
    try:d=json.load(open(p))
    except Exception:continue
    rows.append({k:d.get(k) for k in ['method','source','label_mode','num_scenes','num_decisions','closed_loop_FRA_exec','closed_loop_FRA_cand','closed_loop_DRS','closed_loop_ODG','closed_loop_post_contact_deployability','closed_loop_bounded_NUP','intervention_rate','timing']})
out=os.path.join(run,'closed_loop_summary.json'); json.dump(rows,open(out,'w'),indent=2)
print({'event':'contact_closed_loop_summary','output':out,'num_methods':len(rows)})
PY
