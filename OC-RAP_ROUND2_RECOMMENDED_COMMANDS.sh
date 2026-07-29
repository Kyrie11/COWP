#!/usr/bin/env bash
set -euo pipefail

export OCRAP_REPO=${OCRAP_REPO:-/path/to/OC-RAP_round2}
export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export WOMD_ROOT=${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}
cd "$OCRAP_REPO"
export PYTHONPATH="$OCRAP_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# 1) Clean direct replacement of train_safe (no _v2 output).
rm -rf "$OCRAP_ROOT/train_safe" "$OCRAP_ROOT/.train_safe_shards"
RESUME=1 GPU0=0 GPU1=1 RAW_PER_WORKER=6000 MIN_SAMPLES=15000 MAX_SAMPLES=20000 \
  REQUIRE_JAX_GPU=1 bash scripts/rebuild_ocrap_train_safe_two_gpu.sh

# 2) Delete the known-bad val_safe, but adopt/resume legacy partial near/contact/test outputs.
rm -rf "$OCRAP_ROOT/val_safe"
RESUME=1 ADOPT_LEGACY_RESUME=1 GPU0=0 GPU1=1 REQUIRE_JAX_GPU=1 \
  NEAR_TEACHER_TOP_K=0 CONTACT_TEACHER_TOP_K=0 \
  STRESS_COMPUTE_FUTURE_METRICS=true \
  bash scripts/rebuild_ocrap_val_test_regimes.sh

# 3) Subsequent resume: no legacy-adoption flag needed. Increase RAW budgets safely.
# RESUME=1 ADOPT_LEGACY_RESUME=0 GPU0=0 GPU1=1 \
#   VAL_NEAR_RAW=1200 TEST_NEAR_RAW=1600 VAL_CONTACT_RAW=1200 TEST_CONTACT_RAW=1600 \
#   NEAR_TEACHER_TOP_K=0 CONTACT_TEACHER_TOP_K=0 \
#   bash scripts/rebuild_ocrap_val_test_regimes.sh

# 4) Regenerate diagnostics with the corrected nominal-only Safe contract.
mkdir -p "$OCRAP_ROOT/reports"
for d in train_safe val_safe test_safe; do
  python -m ocrap.cli diagnose --dataset "$OCRAP_ROOT/$d" \
    --set dataset_quality.nominal_regime_dataset=true \
    --set 'dataset_quality.require_nominal_regimes=[normal]' \
    --output "$OCRAP_ROOT/reports/diagnose_${d}.json"
done
for d in val_near_contact test_near_contact val_contact test_contact; do
  python -m ocrap.cli diagnose --dataset "$OCRAP_ROOT/$d" \
    --output "$OCRAP_ROOT/reports/diagnose_${d}.json"
done

# 5) Development v45 run. The v45 training script now uses soft observation-conditioned experts.
export DATASET_DIAGNOSTICS_DIR="$OCRAP_ROOT/reports"
FINAL_RUN=0 RETRAIN_CLEAN_BASE=0 bash run_v45_two_gpu_fast_commands.txt

# Final paper run should refresh the base on clean train_safe:
# FINAL_RUN=1 RETRAIN_CLEAN_BASE=1 TRAIN_SAFE_DATA="$OCRAP_ROOT/train_safe" \
#   bash run_v45_two_gpu_fast_commands.txt
