#!/usr/bin/env bash
set -euo pipefail

# Exact OC-RAP closed-loop evaluation with the optimized hot path.
# Required: WOMD_VAL and CHECKPOINT. CALIBRATION is optional when GAMMA_REC is set.

: "${WOMD_VAL:?Set WOMD_VAL to the WOMD validation TFRecord pattern or shard prefix}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained OC-RAP checkpoint}"

RUN_DIR="${RUN_DIR:-runs/ocrap_closed_loop_optimized}"
OUTPUT="${OUTPUT:-${RUN_DIR}/closed_loop_ocrap.json}"
CALIBRATION="${CALIBRATION:-}"
GAMMA_REC="${GAMMA_REC:-}"
DELTA="${DELTA:-0.05}"
GPU="${GPU:-0}"
WOMD_LIMIT="${WOMD_LIMIT:-150}"
MAX_SCENARIOS="${MAX_SCENARIOS:-50}"
MAX_STEPS="${MAX_STEPS:-40}"
REPLAN_INTERVAL="${REPLAN_INTERVAL:-1}"
LABEL_MODE="${LABEL_MODE:-fast}"
NUM_CANDIDATES="${NUM_CANDIDATES:-}"
NUM_RECOVERY_OPTIONS="${NUM_RECOVERY_OPTIONS:-}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-${RUN_DIR}/.jax_compilation_cache}"

mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT")" "$JAX_CACHE_DIR"

if [[ -z "$GAMMA_REC" ]]; then
  if [[ -z "$CALIBRATION" || ! -f "$CALIBRATION" ]]; then
    echo "Set GAMMA_REC directly or provide an existing CALIBRATION JSON." >&2
    exit 2
  fi
  GAMMA_REC="$(python - "$CALIBRATION" "$DELTA" <<'PY'
import json, sys
path, delta = sys.argv[1], sys.argv[2]
cal = json.load(open(path))
thresholds = cal.get("thresholds", {})
print(thresholds.get(delta, cal.get("gamma_rec", 0.0)))
PY
)"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

ARGS=(
  python -u -m ocrap.cli closed-loop
  --dataset "${WOMD_VAL}@${WOMD_LIMIT}"
  --checkpoint "$CHECKPOINT"
  --output "$OUTPUT"
  --set "closed_loop.max_scenarios=${MAX_SCENARIOS}"
  --set "closed_loop.max_steps=${MAX_STEPS}"
  --set "closed_loop.method=ocrap"
  --set "closed_loop.replan_interval_steps=${REPLAN_INTERVAL}"
  --set "closed_loop.label_mode=${LABEL_MODE}"
  --set "closed_loop.fast_waymax_history=true"
  --set "closed_loop.profile_timing=true"
  --set "selection.gamma_rec=${GAMMA_REC}"
  --set "waymax.jax_compilation_cache_dir=${JAX_CACHE_DIR}"
)

if [[ -n "$NUM_CANDIDATES" ]]; then
  ARGS+=(--set "closed_loop.num_candidate_prefixes=${NUM_CANDIDATES}")
fi
if [[ -n "$NUM_RECOVERY_OPTIONS" ]]; then
  ARGS+=(--set "closed_loop.num_recovery_options=${NUM_RECOVERY_OPTIONS}")
fi

printf 'Running:'
printf ' %q' "${ARGS[@]}"
printf '\n'
"${ARGS[@]}" 2>&1 | tee -a "${OUTPUT%.json}.log"
