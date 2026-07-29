#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_5_ablations}"; mkdir -p "$ROOT"
common=("TRAIN_OCRAP_ROOT=${TRAIN_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}" "EVAL_OCRAP_ROOT=${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}" "INIT_CKPT=${INIT_CKPT:-runs/ocrap_v48_1_existing_data_screening/candidates/precision/model_v48_trac_sr/best.pt}" "CALIBRATION_MODE=proxy_val_split" "CALIBRATION_FRACTION=${CALIBRATION_FRACTION:-0.50}" "CALIBRATION_SEED=${CALIBRATION_SEED:-4801}" "BUILD_TRAIN=0" "BUILD_CALIBRATION=0" "STRICT_TRAIN_DATA_GATE=0" "REUSE_TEACHER_INDEX=0" "GPU0=${GPU0:-0}" "GPU1=${GPU1:-1}" "BATCH_SIZE=${BATCH_SIZE:-72}" "NUM_WORKERS=${NUM_WORKERS:-6}" "PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}" "EPOCHS=${EPOCHS:-12}" "PATIENCE=${PATIENCE:-4}" "FOREGROUND=1" "EXACT_TEACHER_PCD=true" "RISK_SOURCE=delta_distribution" "GROUP_DRO_WEIGHT=0" "POLICY_DISTILL_WEIGHT=0" "POLICY_REGRET_WEIGHT=0")
run(){ local name="$1"; shift; local out="$ROOT/$name"; echo "START $name"; set +e; env "${common[@]}" "OUTPUTDIR=$out" "$@" bash run_v48_two_gpu_fast_commands.txt >"$ROOT/${name}.controller.log" 2>&1; rc=$?; set -e; mkdir -p "$out"; echo "$rc" >"$out/controller.exit_code"; python tools/audit_v48_5_completion.py --root "$out" --require-calibration --output "$out/completion_audit.json" || true; echo "END $name rc=$rc"; }
# All groups share exact teacher-PCD and distributional calibration.
run A_exact_pointwise SET_CONTEXT_ENABLED=false PREFERENCE_HEAD_ENABLED=false PREFERENCE_WEIGHT=0 PREFERENCE_REGRET_WEIGHT=0 DELTA_NLL_WEIGHT=0
run B_exact_zi_nasc SET_CONTEXT_ENABLED=true PREFERENCE_HEAD_ENABLED=false PREFERENCE_WEIGHT=0 PREFERENCE_REGRET_WEIGHT=0 DELTA_NLL_WEIGHT=0
run C_exact_ecpr SET_CONTEXT_ENABLED=false PREFERENCE_HEAD_ENABLED=true PREFERENCE_WEIGHT=1.5 PREFERENCE_REGRET_WEIGHT=0.5 DELTA_NLL_WEIGHT=0
run D_full_ecpr SET_CONTEXT_ENABLED=true PREFERENCE_HEAD_ENABLED=true PREFERENCE_WEIGHT=1.5 PREFERENCE_REGRET_WEIGHT=0.5 DELTA_NLL_WEIGHT=0.5
python tools/summarize_v48_5_ablations.py --root "$ROOT" --output "$ROOT/ablation_summary_v48_5.json"
