#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

ABLATION_ROOT="${ABLATION_ROOT:-runs/ocrap_v48_4_ablations}"
TRAIN_OCRAP_ROOT="${TRAIN_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
EVAL_OCRAP_ROOT="${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
INIT_CKPT="${INIT_CKPT:-runs/ocrap_v48_1_existing_data_screening/candidates/precision/model_v48_trac_sr/best.pt}"
CALIBRATION_SEED="${CALIBRATION_SEED:-4801}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$ABLATION_ROOT"

common=(
  "TRAIN_OCRAP_ROOT=$TRAIN_OCRAP_ROOT" "EVAL_OCRAP_ROOT=$EVAL_OCRAP_ROOT"
  "INIT_CKPT=$INIT_CKPT" "CALIBRATION_MODE=proxy_val_split"
  "CALIBRATION_FRACTION=${CALIBRATION_FRACTION:-0.50}" "CALIBRATION_SEED=$CALIBRATION_SEED"
  "BUILD_TRAIN=0" "BUILD_CALIBRATION=0" "STRICT_TRAIN_DATA_GATE=0"
  "REUSE_TEACHER_INDEX=0" "GPU0=$GPU0" "GPU1=$GPU1"
  "BATCH_SIZE=${BATCH_SIZE:-72}" "NUM_WORKERS=${NUM_WORKERS:-6}"
  "PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}" "EPOCHS=${EPOCHS:-8}"
  "PATIENCE=${PATIENCE:-2}" "FOREGROUND=1"
)

run_one() {
  local name="$1"; shift
  local out="$ABLATION_ROOT/$name"
  echo "===== START $name -> $out ====="
  set +e
  env "${common[@]}" "OUTPUTDIR=$out" "$@" bash run_v48_two_gpu_fast_commands.txt
  local status=$?
  set -e
  echo "$status" > "$out/controller.exit_code"
  echo "===== END $name status=$status ====="
  # A non-zero status is expected when Natural gate rejects all candidates.
  # screening_status.json and calibration diagnostics are still complete.
}

# A. SRC reference: no set context, no regret distillation, no drift defense.
run_one A_src_reference \
  SET_CONTEXT_ENABLED=false POLICY_DISTILL_WEIGHT=0 POLICY_REGRET_WEIGHT=0 \
  POLICY_ADMISSION_DISTILL_WEIGHT=0 GROUP_DRO_WEIGHT=0 \
  OPPORTUNITY_SOFT_LABEL_TEMPERATURE=0 HARM_SOFT_LABEL_TEMPERATURE=0

# B. ZI-NASC only: isolates the zero-initialized nominal-anchored set adapter.
run_one B_zi_nasc_only \
  SET_CONTEXT_ENABLED=true POLICY_DISTILL_WEIGHT=0 POLICY_REGRET_WEIGHT=0 \
  POLICY_ADMISSION_DISTILL_WEIGHT=0 GROUP_DRO_WEIGHT=0 \
  OPPORTUNITY_SOFT_LABEL_TEMPERATURE=0 HARM_SOFT_LABEL_TEMPERATURE=0

# C. DRA-RCD only: value-only ranking distillation, without set adapter or GroupDRO.
run_one C_dra_rcd_only \
  SET_CONTEXT_ENABLED=false POLICY_DISTILL_WEIGHT=1 POLICY_REGRET_WEIGHT=1 \
  POLICY_DECOUPLE_ADMISSION=true POLICY_ADMISSION_DISTILL_WEIGHT=0.15 \
  GROUP_DRO_WEIGHT=0 OPPORTUNITY_SOFT_LABEL_TEMPERATURE=0 \
  HARM_SOFT_LABEL_TEMPERATURE=0

# D. Full v48.4: ZI-NASC + DRA-RCD + soft risk labels + pseudo-environment GroupDRO.
run_one D_full_srgr \
  SET_CONTEXT_ENABLED=true POLICY_DISTILL_WEIGHT=1 POLICY_REGRET_WEIGHT=1 \
  POLICY_DECOUPLE_ADMISSION=true POLICY_ADMISSION_DISTILL_WEIGHT=0.15 \
  GROUP_DRO_WEIGHT=0.35 GROUP_DRO_TEMPERATURE=0.35 \
  OPPORTUNITY_SOFT_LABEL_TEMPERATURE=0.02 HARM_SOFT_LABEL_TEMPERATURE=0.02

python tools/summarize_v48_4_ablations.py --root "$ABLATION_ROOT" \
  --output "$ABLATION_ROOT/ablation_summary.json"
echo "Ablation summary: $ABLATION_ROOT/ablation_summary.json"
