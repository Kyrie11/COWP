#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_7_ablations}"
mkdir -p "$ROOT"

common=(
  "TRAIN_OCRAP_ROOT=${TRAIN_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "EVAL_OCRAP_ROOT=${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "INIT_CKPT=${INIT_CKPT:?Set INIT_CKPT to a completed v48.5/v48.6 preference checkpoint}"
  "CALIBRATION_MODE=proxy_val_split" "CALIBRATION_FRACTION=${CALIBRATION_FRACTION:-0.50}"
  "CALIBRATION_SEED=${CALIBRATION_SEED:-4801}"
  "BUILD_TRAIN=0" "BUILD_CALIBRATION=0" "STRICT_TRAIN_DATA_GATE=0" "REUSE_TEACHER_INDEX=0"
  "GPU0=${GPU0:-0}" "GPU1=${GPU1:-1}" "BATCH_SIZE=${BATCH_SIZE:-72}"
  "NUM_WORKERS=${NUM_WORKERS:-6}" "PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}"
  "FOREGROUND=1" "EXACT_TEACHER_PCD=true" "RISK_SOURCE=direct_delta"
  "GROUP_DRO_WEIGHT=0" "POLICY_DISTILL_WEIGHT=0" "POLICY_REGRET_WEIGHT=0"
  "OPPORTUNITY_ADMISSION_WEIGHT=0" "HARM_ADMISSION_WEIGHT=0"
  "SELECTIVE_RISK_WEIGHT=0" "SELECTIVE_COVERAGE_WEIGHT=0"
)

run_one() {
  local name="$1"; shift
  local out="$ROOT/$name"
  echo "START $name"
  set +e
  env "${common[@]}" "OUTPUTDIR=$out" "$@" bash run_v48_two_gpu_fast_commands.txt \
    >"$ROOT/${name}.controller.log" 2>&1
  rc=$?
  set -e
  mkdir -p "$out"; echo "$rc" > "$out/controller.exit_code"
  python tools/audit_v48_7_completion.py --root "$out" --require-calibration \
    --output "$out/completion_audit.json" || true
  # rc=20 means training/calibration completed but Natural gate rejected it.
  [[ "$rc" == 0 || "$rc" == 20 ]] || { echo "$name failed before comparable completion: rc=$rc" >&2; exit "$rc"; }
  echo "END $name rc=$rc"
}

# A: reproduce the v48.6 joint objective.
run_one A_joint_singlewinner \
  TRAIN_SCRIPT=scripts/train_ocrap_v48_trac_sr.sh \
  EPOCHS="${EPOCHS:-12}" PATIENCE="${PATIENCE:-4}" \
  PREFERENCE_CONTEXT_ENABLED=true DELTA_HEAD_ENABLED=true \
  PREFERENCE_SET_WEIGHT=0 DELTA_NLL_WEIGHT=1.0

# B: isolate gradient decoupling while retaining the old single-winner target.
run_one B_staged_singlewinner \
  TRAIN_SCRIPT=scripts/train_ocrap_v48_7_spire.sh \
  PREFERENCE_SET_WEIGHT=0

# C: ambiguity-aware target without gradient isolation.
run_one C_joint_setvalued \
  TRAIN_SCRIPT=scripts/train_ocrap_v48_trac_sr.sh \
  EPOCHS="${EPOCHS:-12}" PATIENCE="${PATIENCE:-4}" \
  PREFERENCE_CONTEXT_ENABLED=true DELTA_HEAD_ENABLED=true \
  PREFERENCE_LISTWISE_WEIGHT=0 PREFERENCE_SET_WEIGHT="${PREFERENCE_SET_WEIGHT:-1.25}" \
  PREFERENCE_TIE_EPS_NEAR="${PREFERENCE_TIE_EPS_NEAR:-0.025}" \
  PREFERENCE_TIE_EPS_CONTACT="${PREFERENCE_TIE_EPS_CONTACT:-0.010}" \
  DELTA_NLL_WEIGHT=1.0

# D: full SPIRE = set-valued preference plus isolated certificate stage.
run_one D_full_spire \
  TRAIN_SCRIPT=scripts/train_ocrap_v48_7_spire.sh \
  PREFERENCE_SET_WEIGHT="${PREFERENCE_SET_WEIGHT:-1.25}" \
  PREFERENCE_TIE_EPS_NEAR="${PREFERENCE_TIE_EPS_NEAR:-0.025}" \
  PREFERENCE_TIE_EPS_CONTACT="${PREFERENCE_TIE_EPS_CONTACT:-0.010}"

python tools/summarize_v48_7_ablations.py --root "$ROOT" --output "$ROOT/ablation_summary_v48_7.json"
printf '{"complete":true,"experiments":4,"version":"v48.7"}\n' > "$ROOT/ABLATIONS_COMPLETE.json"
