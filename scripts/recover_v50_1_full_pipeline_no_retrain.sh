#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${CUDA_DEVICES:=0,1}"
: "${EXTERNAL_RESULTS_ROOT:=runs/all_regime_external_baselines_v50_1_full}"
: "${OCRAP_RESULTS_ROOT:=runs/ocrap_three_regime_closed_loop_v50_1_full}"
: "${COMPARISON_OUT:=runs/external_comparison_v50_3_recovered}"
: "${EXTERNAL_CHECKPOINT_ROOT:=/data0/senzeyu2/checkpoints/ocrap_external_baselines_v49}"
: "${OCRAP_MODEL_RUN:=runs/ocrap_v48_34_barrier_crossfit_dedicated_4834}"
: "${MODEL_VARIANT:=balanced}"
: "${ALLOW_DIAGNOSTIC_RC20:=1}"
: "${MAX_SCENARIOS:=0}"
: "${MAX_STEPS:=40}"
: "${SAFE_EXPECTED_COUNT:=175}"
: "${NEAR_EXPECTED_COUNT:=250}"
: "${CONTACT_EXPECTED_COUNT:=209}"
: "${RECOVER_EXTERNAL:=true}"
: "${RECOVER_OCRAP:=true}"
: "${BUILD_COMPARISON:=true}"
: "${BUILD_VIDEOS:=false}"
: "${FPS:=10}"

bool_true() { case "${1,,}" in 1|true|yes|on) return 0;; *) return 1;; esac; }

if bool_true "$RECOVER_EXTERNAL"; then
  echo "[RECOVER] external baselines: reuse checkpoints/results; run only missing Safe/Near methods"
  env \
    OCRAP_ROOT="$OCRAP_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
    OUT="$EXTERNAL_RESULTS_ROOT" \
    EXTERNAL_CHECKPOINT_ROOT="$EXTERNAL_CHECKPOINT_ROOT" \
    SAFE_CHECKPOINT_ROOT="$EXTERNAL_CHECKPOINT_ROOT/safe" \
    NEAR_CHECKPOINT_ROOT="$EXTERNAL_CHECKPOINT_ROOT/near" \
    RUN_SAFE=1 RUN_NEAR=1 RUN_CONTACT=0 \
    DO_TRAIN_SAFE=false DO_TRAIN_NEAR=false FORCE_RETRAIN_ALL=false \
    DO_OFFLINE=false DO_CLOSED_LOOP=true RENDER_TRACES=false \
    SKIP_COMPLETE_METHODS=true USE_DYNAMIC_SCHEDULER=false \
    MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" \
    bash scripts/run_all_regime_external_baselines_optimized.sh
fi

if bool_true "$RECOVER_OCRAP"; then
  echo "[RECOVER] OC-RAP: finalize complete journals first, then resume only missing targets"
  mkdir -p "$OCRAP_RESULTS_ROOT/safe" "$OCRAP_RESULTS_ROOT/near" "$OCRAP_RESULTS_ROOT/contact"
  python tools/finalize_closed_loop_from_journal.py \
    --output "$OCRAP_RESULTS_ROOT/safe/closed_loop_ocrap.json" \
    --expected-count "$SAFE_EXPECTED_COUNT" || true
  python tools/finalize_closed_loop_from_journal.py \
    --output "$OCRAP_RESULTS_ROOT/near/closed_loop_ocrap.json" \
    --expected-count "$NEAR_EXPECTED_COUNT" || true
  python tools/finalize_closed_loop_from_journal.py \
    --output "$OCRAP_RESULTS_ROOT/contact/closed_loop_ocrap.json" \
    --expected-count "$CONTACT_EXPECTED_COUNT" || true

  env \
    OCRAP_ROOT="$OCRAP_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
    OUT="$OCRAP_RESULTS_ROOT" MODEL_RUN="$OCRAP_MODEL_RUN" MODEL_VARIANT="$MODEL_VARIANT" \
    ALLOW_DIAGNOSTIC_RC20="$ALLOW_DIAGNOSTIC_RC20" \
    MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" \
    RUN_SAFE=1 RUN_NEAR=1 RUN_CONTACT=1 \
    SKIP_COMPLETE_REGIMES=true FINALIZE_COMPLETE_JOURNALS=true \
    RENDER_SAFE=false RENDER_NEAR=false RENDER_CONTACT=false \
    RESUME_FORCE=false \
    bash scripts/run_ocrap_three_regime_closed_loop.sh
fi

if bool_true "$BUILD_COMPARISON"; then
  if bool_true "$BUILD_VIDEOS"; then echo "[BUILD] paired regime tables and selective 10-scene videos"; else echo "[BUILD] paired regime tables"; fi
  env \
    OCRAP_ROOT="$OCRAP_ROOT" CUDA_DEVICES="$CUDA_DEVICES" \
    OCRAP_RESULTS_ROOT="$OCRAP_RESULTS_ROOT" \
    EXTERNAL_RESULTS_ROOT="$EXTERNAL_RESULTS_ROOT" \
    OUT="$COMPARISON_OUT" \
    BUILD_VIDEOS="$BUILD_VIDEOS" FPS="$FPS" \
    VIDEO_SELECTION_FALLBACK=true \
    EXTERNAL_CHECKPOINT_ROOT="$EXTERNAL_CHECKPOINT_ROOT" \
    OCRAP_MODEL_RUN="$OCRAP_MODEL_RUN" MODEL_VARIANT="$MODEL_VARIANT" \
    ALLOW_DIAGNOSTIC_RC20="$ALLOW_DIAGNOSTIC_RC20" \
    bash scripts/build_external_comparison_artifacts.sh
fi

echo "[DONE] recovery pipeline"
echo "  external index: $EXTERNAL_RESULTS_ROOT/EXTERNAL_BASELINE_RUN_INDEX.json"
echo "  OC-RAP index:   $OCRAP_RESULTS_ROOT/OCRAP_THREE_REGIME_RUN_INDEX.json"
echo "  comparison:     $COMPARISON_OUT/COMPARISON_INDEX.json"
