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
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-1}"
export PREFETCH_FACTOR="${ABLATION_PREFETCH_FACTOR:-2}"
mkdir -p "$BASE_OUT"

# All four arms are launched together for matched wall-clock conditions.  Each
# arm's existing controller maps Balanced to GPU0 and Precision to GPU1.  This
# therefore creates up to four training processes per GPU; set MAX_PARALLEL_ARMS=2
# on smaller GPUs to run two matched waves without changing experiment semantics.
MAX_PARALLEL_ARMS="${MAX_PARALLEL_ARMS:-4}"
if ! [[ "$MAX_PARALLEL_ARMS" =~ ^[1-4]$ ]]; then
  echo "MAX_PARALLEL_ARMS must be 1..4" >&2; exit 2
fi
arms=(A B C D)
run_arm() {
  local arm="$1" log
  log="$BASE_OUT/ocrap_v48_44_roct_${arm}.launcher.log"
  BASE_OUT="$BASE_OUT" GPU0="$GPU0" GPU1="$GPU1" \
    bash scripts/run_v48_44_roct_ablation_arm.sh "$arm" >"$log" 2>&1
}

pids=(); names=(); rc=0
for arm in "${arms[@]}"; do
  while (( ${#pids[@]} >= MAX_PARALLEL_ARMS )); do
    pid="${pids[0]}"; name="${names[0]}"
    if ! wait "$pid"; then echo "arm $name failed; inspect launcher log" >&2; rc=1; fi
    pids=("${pids[@]:1}"); names=("${names[@]:1}")
  done
  run_arm "$arm" &
  pids+=("$!"); names+=("$arm")
done
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "arm ${names[$i]} failed; inspect launcher log" >&2; rc=1; fi
done
exit "$rc"
