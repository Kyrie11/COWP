#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
BASE_OUT="${BASE_OUT:-runs}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
export OMP_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${ABLATION_PREFETCH_FACTOR:-2}"

# D is the main experiment. Run only A/B/C here, concurrently. Each controller
# itself uses GPU0 for Balanced and GPU1 for Precision. The user confirmed RAM/
# VRAM capacity for this concurrency pattern.
pids=(); arms=(A B C)
for arm in "${arms[@]}"; do
  log="$BASE_OUT/ocrap_v48_41_fcfr_ablation_${arm}.launcher.log"
  mkdir -p "$BASE_OUT"
  ( BASE_OUT="$BASE_OUT" GPU0="$GPU0" GPU1="$GPU1" \
      bash scripts/run_v48_41_fcfr_ablation_arm.sh "$arm" >"$log" 2>&1 ) &
  pids+=("$!")
done
rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ablation ${arms[$i]} failed; inspect launcher log" >&2
    rc=1
  fi
done
exit "$rc"
