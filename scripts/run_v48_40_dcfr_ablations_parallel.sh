#!/usr/bin/env bash
set -Eeuo pipefail
# Run A/B/C concurrently. D is the main run and is intentionally not repeated.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
BASE_OUT="${BASE_OUT:-$REPO/runs}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OMP_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-2}" PREFETCH_FACTOR="${ABLATION_PREFETCH_FACTOR:-2}"
pids=(); arms=(A B C)
for arm in "${arms[@]}"; do
  out="$BASE_OUT/ocrap_v48_40_dcfr_ablation_${arm}"
  (OUTPUTDIR="$out" GPU0="$GPU0" GPU1="$GPU1" bash scripts/run_v48_40_dcfr_ablation_arm.sh "$arm") >"$BASE_OUT/v48_40_${arm}.launcher.log" 2>&1 &
  pids+=("$!")
done
rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "arm ${arms[$i]} failed" >&2; rc=1; fi
done
exit "$rc"
