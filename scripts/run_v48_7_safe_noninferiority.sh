#!/usr/bin/env bash
set -euo pipefail

# Safe-only probe is permitted before the stress Natural gate because Safe is
# nominal-locked and RUN_DIRECT_VALUE is forced off. It does not authorize Near
# or Contact execution.
BASE_RUN="${BASE_RUN:?set BASE_RUN to a completed candidate run}"
RUN="${RUN:-runs/ocrap_v48_7_safe_noninferiority}"
SAFE_TEST="${SAFE_TEST:-/data0/senzeyu2/dataset/OCRAP/calibration_safe}"

SAFE_NOMINAL_ONLY=1 BASE_RUN="$BASE_RUN" RUN="$RUN" SAFE_TEST="$SAFE_TEST" \
RUN_OFFLINE_EVAL=0 RUN_AUDITS=0 RUN_SAFE_CLOSED_LOOP=1 RUN_SCALAR_BASELINES=0 RUN_SAFE_PAIRED_SCALAR="${RUN_SAFE_PAIRED_SCALAR:-1}" \
RUN_DIRECT_VALUE=false \
GPU_SAFE_BASELINE="${GPU_SAFE_BASELINE:-0}" GPU_SAFE="${GPU_SAFE:-1}" SAFE_MAX_TARGETS="${SAFE_MAX_TARGETS:-80}" \
SAFE_MAX_ROLLOUTS="${SAFE_MAX_ROLLOUTS:-32}" SAFE_MAX_STEPS="${SAFE_MAX_STEPS:-40}" \
  bash scripts/run_ocrap_v48_trac_sr.sh
