#!/usr/bin/env bash
set -euo pipefail

# Queue all four ablation groups immediately, but keep at most one process per
# A30. With two GPUs the safe concurrency is two training jobs, not four jobs
# sharing devices. Shared teacher-PCD and proxy splits remove repeated CPU work.

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_9_ablations}"
ASSET_ROOT="${ASSET_ROOT:-runs/ocrap_v48_8_shared_assets_4801}"
INIT_CKPT="${INIT_CKPT:?Set INIT_CKPT to a completed inherited checkpoint, preferably v48.8 precision}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"
[[ -f "$ASSET_ROOT/SHARED_ASSETS_COMPLETE.json" ]] || {
  echo "missing shared assets: $ASSET_ROOT/SHARED_ASSETS_COMPLETE.json" >&2; exit 2;
}

COMMON=(
  "TRAIN_OCRAP_ROOT=${TRAIN_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "EVAL_OCRAP_ROOT=${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "INIT_CKPT=$INIT_CKPT" "CALIBRATION_MODE=proxy_val_split"
  "CALIBRATION_FRACTION=${CALIBRATION_FRACTION:-0.50}"
  "CALIBRATION_SEED=${CALIBRATION_SEED:-4801}"
  "BUILD_TRAIN=0" "BUILD_CALIBRATION=0" "STRICT_TRAIN_DATA_GATE=0"
  "REUSE_TEACHER_INDEX=1" "AUTO_ENSURE_MANIFESTS=0"
  "PREBUILT_SPLIT_ROOT=$ASSET_ROOT/dataset_splits" "REUSE_PREBUILT_SPLITS=1"
  "SHARED_GROUP_INDEX=$ASSET_ROOT/teacher_pcd_train_index.jsonl"
  "SHARED_GROUP_SUMMARY=$ASSET_ROOT/teacher_pcd_train_index_summary.json"
  "BATCH_SIZE=${BATCH_SIZE:-72}" "NUM_WORKERS=${NUM_WORKERS:-6}"
  "PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}" "FOREGROUND=1"
  "EXACT_TEACHER_PCD=true" "GROUP_DRO_WEIGHT=0"
)

run_task() {
  local group="$1" variant="$2" gpu="$3"; shift 3
  local out="$ROOT/tasks/${group}_${variant}"
  mkdir -p "$out"
  echo "[$(date -Is)] START $group $variant GPU=$gpu" | tee -a "$ROOT/logs/scheduler.log"
  set +e
  env "${COMMON[@]}" OUTPUTDIR="$out" VARIANTS="$variant" GPU0="$gpu" GPU1="$gpu" "$@" \
    bash run_v48_two_gpu_fast_commands.txt >"$ROOT/logs/${group}_${variant}.log" 2>&1
  local rc=$?
  set -e
  echo "$rc" > "$out/controller.exit_code"
  if [[ "$rc" != 0 && "$rc" != 20 ]]; then
    echo "[$(date -Is)] FAIL $group $variant rc=$rc" | tee -a "$ROOT/logs/scheduler.log"
    return "$rc"
  fi
  echo "[$(date -Is)] END $group $variant rc=$rc" | tee -a "$ROOT/logs/scheduler.log"
}

# Four causal groups isolate the two v48.9 contributions. All use the same
# exact teacher, staged optimizer, proxy split, and direct-delta risk source.
A=(TRAIN_SCRIPT=scripts/train_ocrap_v48_9_pacer.sh RISK_SOURCE=direct_delta
   PREFERENCE_SET_MASS_LOSS=false PREFERENCE_NOOP_NOMINAL_ONLY=false
   CERTIFICATE_ALL_CANDIDATE_WEIGHT=1.0 CERTIFICATE_POLICY_TOP1_WEIGHT=0
   CERTIFICATE_POLICY_TOP1_SIGN_WEIGHT=0)
B=(TRAIN_SCRIPT=scripts/train_ocrap_v48_9_pacer.sh RISK_SOURCE=direct_delta
   PREFERENCE_SET_MASS_LOSS=true PREFERENCE_NOOP_NOMINAL_ONLY=true
   CERTIFICATE_ALL_CANDIDATE_WEIGHT=1.0 CERTIFICATE_POLICY_TOP1_WEIGHT=0
   CERTIFICATE_POLICY_TOP1_SIGN_WEIGHT=0)
C=(TRAIN_SCRIPT=scripts/train_ocrap_v48_9_pacer.sh RISK_SOURCE=direct_delta
   PREFERENCE_SET_MASS_LOSS=false PREFERENCE_NOOP_NOMINAL_ONLY=false
   CERTIFICATE_ALL_CANDIDATE_WEIGHT=0.20 CERTIFICATE_POLICY_TOP1_WEIGHT=2.0
   CERTIFICATE_POLICY_TOP1_SIGN_WEIGHT=1.5)
D=(TRAIN_SCRIPT=scripts/train_ocrap_v48_9_pacer.sh RISK_SOURCE=direct_delta
   PREFERENCE_SET_MASS_LOSS=true PREFERENCE_NOOP_NOMINAL_ONLY=true
   CERTIFICATE_ALL_CANDIDATE_WEIGHT=0.20 CERTIFICATE_POLICY_TOP1_WEIGHT=2.0
   CERTIFICATE_POLICY_TOP1_SIGN_WEIGHT=1.5)

# Four waves = eight variant jobs / two GPUs. Groups are interleaved so two
# different ablations are active concurrently while no GPU is oversubscribed.
run_task A_staged_uniform_allcandidate balanced "$GPU0" "${A[@]}" & P0=$!
run_task B_intervention_set_preference balanced "$GPU1" "${B[@]}" & P1=$!
wait "$P0"; wait "$P1"
run_task C_policy_aligned_certificate balanced "$GPU0" "${C[@]}" & P0=$!
run_task D_full_pacer balanced "$GPU1" "${D[@]}" & P1=$!
wait "$P0"; wait "$P1"
run_task A_staged_uniform_allcandidate precision "$GPU0" "${A[@]}" & P0=$!
run_task B_intervention_set_preference precision "$GPU1" "${B[@]}" & P1=$!
wait "$P0"; wait "$P1"
run_task C_policy_aligned_certificate precision "$GPU0" "${C[@]}" & P0=$!
run_task D_full_pacer precision "$GPU1" "${D[@]}" & P1=$!
wait "$P0"; wait "$P1"

python tools/summarize_v48_8_parallel_ablations.py --root "$ROOT" \
  --output "$ROOT/ablation_summary_v48_9.json"
printf '{"complete":true,"experiments":4,"variants":2,"max_parallel_gpu_jobs":2,"version":"v48.9"}\n' \
  > "$ROOT/ABLATIONS_COMPLETE.json"
