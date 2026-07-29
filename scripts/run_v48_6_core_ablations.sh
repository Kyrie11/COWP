#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

ROOT="${ABLATION_ROOT:-runs/ocrap_v48_6_ablations}"
mkdir -p "$ROOT"

common=(
  "TRAIN_OCRAP_ROOT=${TRAIN_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "EVAL_OCRAP_ROOT=${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "INIT_CKPT=${INIT_CKPT:?Set INIT_CKPT to the selected v48.5 checkpoint}"
  "CALIBRATION_MODE=proxy_val_split"
  "CALIBRATION_FRACTION=${CALIBRATION_FRACTION:-0.50}"
  "CALIBRATION_SEED=${CALIBRATION_SEED:-4801}"
  "BUILD_TRAIN=0" "BUILD_CALIBRATION=0" "STRICT_TRAIN_DATA_GATE=0"
  "REUSE_TEACHER_INDEX=0"
  "GPU0=${GPU0:-0}" "GPU1=${GPU1:-1}"
  "BATCH_SIZE=${BATCH_SIZE:-72}" "NUM_WORKERS=${NUM_WORKERS:-6}"
  "PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}"
  "EPOCHS=${EPOCHS:-12}" "PATIENCE=${PATIENCE:-4}"
  "FOREGROUND=1" "EXACT_TEACHER_PCD=true"
  "SET_CONTEXT_ENABLED=false"
  "PREFERENCE_HEAD_ENABLED=true"
  "PREFERENCE_WEIGHT=${PREFERENCE_WEIGHT:-1.50}"
  "PREFERENCE_REGRET_WEIGHT=${PREFERENCE_REGRET_WEIGHT:-0.50}"
  "PREFERENCE_LISTWISE_WEIGHT=${PREFERENCE_LISTWISE_WEIGHT:-0.75}"
  "PREFERENCE_GAP_WEIGHT=${PREFERENCE_GAP_WEIGHT:-0.25}"
  "GROUP_DRO_WEIGHT=0" "POLICY_DISTILL_WEIGHT=0" "POLICY_REGRET_WEIGHT=0"
)

run_one() {
  local name="$1"; shift
  local out="$ROOT/$name"
  echo "START $name"
  set +e
  env "${common[@]}" "OUTPUTDIR=$out" "$@" \
    bash run_v48_two_gpu_fast_commands.txt \
    >"$ROOT/${name}.controller.log" 2>&1
  rc=$?
  set -e
  mkdir -p "$out"
  echo "$rc" > "$out/controller.exit_code"
  python tools/audit_v48_6_completion.py \
    --root "$out" --require-calibration \
    --output "$out/completion_audit.json" || true
  echo "END $name rc=$rc"
}

# A reproduces the useful v48.5 ECPR path without legacy NASC.
run_one A_v485_ecpr_reference \
  PREFERENCE_CONTEXT_ENABLED=false DELTA_HEAD_ENABLED=false \
  DELTA_NLL_WEIGHT=0 RISK_SOURCE=delta_distribution

# B isolates preference-only relative set context.
run_one B_preference_context_only \
  PREFERENCE_CONTEXT_ENABLED=true DELTA_HEAD_ENABLED=false \
  DELTA_NLL_WEIGHT=0 RISK_SOURCE=delta_distribution

# C isolates the direct candidate-vs-nominal gain distribution.
run_one C_direct_delta_only \
  PREFERENCE_CONTEXT_ENABLED=false DELTA_HEAD_ENABLED=true \
  DELTA_NLL_WEIGHT=1.0 RISK_SOURCE=direct_delta

# D is the complete v48.6 RPGC model.
run_one D_full_rpgc \
  PREFERENCE_CONTEXT_ENABLED=true DELTA_HEAD_ENABLED=true \
  DELTA_NLL_WEIGHT=1.0 RISK_SOURCE=direct_delta

python tools/summarize_v48_6_ablations.py \
  --root "$ROOT" --output "$ROOT/ablation_summary_v48_6.json"
printf '{"complete":true,"experiments":4}\n' > "$ROOT/ABLATIONS_COMPLETE.json"
