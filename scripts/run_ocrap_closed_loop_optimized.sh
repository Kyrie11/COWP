#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=scripts/lib/v50_runtime.sh
source scripts/lib/v50_runtime.sh

: "${WOMD_VAL:?Set WOMD_VAL to a WOMD TFRecord path/spec, preferably ...tfrecord@150}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained OC-RAP checkpoint}"
[[ -f "$CHECKPOINT" ]] || { echo "Missing OC-RAP checkpoint: $CHECKPOINT" >&2; exit 2; }

RUN_DIR="${RUN_DIR:-runs/ocrap_closed_loop_optimized}"
OUTPUT="${OUTPUT:-${RUN_DIR}/closed_loop_ocrap.json}"
CALIBRATION="${CALIBRATION:-}"
GAMMA_REC="${GAMMA_REC:-}"
DELTA="${DELTA:-0.05}"
GPU="${GPU:-0}"
if [[ -n "${WOMD_LIMIT+x}" && -z "${WOMD_NUM_SHARDS+x}" ]]; then
  echo "[WARN] WOMD_LIMIT is deprecated; it represented TFRecord shard count, not scenario count. Use WOMD_NUM_SHARDS." >&2
  WOMD_NUM_SHARDS="$WOMD_LIMIT"
fi
WOMD_NUM_SHARDS="${WOMD_NUM_SHARDS:-150}"
MAX_SCENARIOS="${MAX_SCENARIOS:-50}"
RAW_MAX_SCENARIOS="${RAW_MAX_SCENARIOS:-}"
MAX_STEPS="${MAX_STEPS:-40}"
REPLAN_INTERVAL="${REPLAN_INTERVAL:-1}"
LABEL_MODE="${LABEL_MODE:-fast}"
AUDIT_EVERY_N_STEPS="${AUDIT_EVERY_N_STEPS:-0}"
NUM_CANDIDATES="${NUM_CANDIDATES:-}"
NUM_RECOVERY_OPTIONS="${NUM_RECOVERY_OPTIONS:-}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-${RUN_DIR}/.jax_compilation_cache}"
CONFIG="${CONFIG:-}"
BUCKET_DATASET="${BUCKET_DATASET:-}"
BUCKET_SPLIT="${BUCKET_SPLIT:-test}"
MAX_TARGETS_PER_SCENE="${MAX_TARGETS_PER_SCENE:-1}"
TARGET_KEYS_FILE="${TARGET_KEYS_FILE:-}"
REQUIRE_TARGET_KEYS="${REQUIRE_TARGET_KEYS:-true}"
RENDER_TRACE="${RENDER_TRACE:-false}"
RENDER_MAX_AGENTS="${RENDER_MAX_AGENTS:-48}"
SAVE_PARTIAL="${SAVE_PARTIAL:-true}"
RESUME_FORCE="${RESUME_FORCE:-false}"
PROFILE_TIMING="${PROFILE_TIMING:-true}"
PREFLIGHT="${PREFLIGHT:-true}"
PARTIAL_WRITE_EVERY_SCENES="${PARTIAL_WRITE_EVERY_SCENES:-32}"
PROGRESS_EVERY_STEPS="${PROGRESS_EVERY_STEPS:-10}"
# Full render traces are needed only for the ten selected qualitative reruns.
# Population metric runs store one compact scene-summary row per target.
if v50_bool_true "$RENDER_TRACE"; then
  RESULT_SCENE_DETAIL="${RESULT_SCENE_DETAIL:-full}"
  SCENE_JOURNAL_DETAIL="${SCENE_JOURNAL_DETAIL:-full}"
  MEMORY_SCENE_DETAIL="${MEMORY_SCENE_DETAIL:-full}"
else
  RESULT_SCENE_DETAIL="${RESULT_SCENE_DETAIL:-metrics}"
  SCENE_JOURNAL_DETAIL="${SCENE_JOURNAL_DETAIL:-metrics}"
  MEMORY_SCENE_DETAIL="${MEMORY_SCENE_DETAIL:-metrics}"
fi
INCLUDE_SCENES_IN_RESULT="${INCLUDE_SCENES_IN_RESULT:-false}"
INCLUDE_SCENES_IN_PARTIAL="${INCLUDE_SCENES_IN_PARTIAL:-false}"

mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT")" "$JAX_CACHE_DIR"

if [[ -z "$GAMMA_REC" ]]; then
  if [[ -z "$CALIBRATION" || ! -f "$CALIBRATION" ]]; then
    echo "Set GAMMA_REC directly or provide an existing CALIBRATION JSON." >&2; exit 2
  fi
  GAMMA_REC="$(python - "$CALIBRATION" "$DELTA" <<'PY'
import json, sys
cal=json.load(open(sys.argv[1])); delta=sys.argv[2]
print((cal.get('thresholds') or {}).get(delta, cal.get('gamma_rec', 0.0)))
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

DATASET_SPEC="$(v50_normalize_womd_spec "$WOMD_VAL" "$WOMD_NUM_SHARDS")"
if [[ -n "$BUCKET_DATASET" && "$PREFLIGHT" == true ]]; then
  preflight_target_args=()
  if [[ -n "$TARGET_KEYS_FILE" ]]; then
    preflight_target_args=(--target-keys-file "$TARGET_KEYS_FILE")
    v50_bool_true "$REQUIRE_TARGET_KEYS" && preflight_target_args+=(--require-target-keys)
  fi
  python tools/check_closed_loop_dataset_support.py --dataset "$BUCKET_DATASET" --split "$BUCKET_SPLIT" \
    --womd-pattern "$DATASET_SPEC" --expected-source-role auto "${preflight_target_args[@]}" \
    --output "$RUN_DIR/closed_loop_dataset_support.json"
else
  python tools/validate_womd_spec.py --spec "$DATASET_SPEC" --output "$RUN_DIR/womd_spec_validation.json"
fi

ARGS=(
  python -u -m ocrap.cli closed-loop
  --dataset "$DATASET_SPEC"
  --checkpoint "$CHECKPOINT"
  --output "$OUTPUT"
  --set "closed_loop.max_scenarios=$MAX_SCENARIOS"
  --set "closed_loop.max_bucket_targets=$MAX_SCENARIOS"
  --set "closed_loop.max_targets_per_scene=$MAX_TARGETS_PER_SCENE"
  --set "closed_loop.max_steps=$MAX_STEPS"
  --set closed_loop.method=ocrap
  --set "closed_loop.replan_interval_steps=$REPLAN_INTERVAL"
  --set "closed_loop.label_mode=$LABEL_MODE"
  --set "closed_loop.audit_every_n_steps=$AUDIT_EVERY_N_STEPS"
  --set closed_loop.fast_waymax_history=true
  --set "closed_loop.profile_timing=$PROFILE_TIMING"
  --set "closed_loop.render_trace=$RENDER_TRACE"
  --set "closed_loop.render_max_agents=$RENDER_MAX_AGENTS"
  --set "closed_loop.save_partial=$SAVE_PARTIAL"
  --set "closed_loop.resume_force=$RESUME_FORCE"
  --set "closed_loop.partial_write_every_scenes=$PARTIAL_WRITE_EVERY_SCENES"
  --set "closed_loop.progress_every_steps=$PROGRESS_EVERY_STEPS"
  --set "closed_loop.result_scene_detail=$RESULT_SCENE_DETAIL"
  --set "closed_loop.scene_journal_detail=$SCENE_JOURNAL_DETAIL"
  --set "closed_loop.memory_scene_detail=$MEMORY_SCENE_DETAIL"
  --set "closed_loop.include_scenes_in_result=$INCLUDE_SCENES_IN_RESULT"
  --set "closed_loop.include_scenes_in_partial=$INCLUDE_SCENES_IN_PARTIAL"
  --set "selection.gamma_rec=$GAMMA_REC"
  --set "waymax.jax_compilation_cache_dir=$JAX_CACHE_DIR"
  --set waymax.dataloader_include_sdc_paths=false
  --set waymax.compute_future_metrics=false
  --set waymax.teacher_metrics_stride=0
  --set waymax.use_jit_scan_rollouts=true
)
[[ -n "$CONFIG" ]] && ARGS+=(--config "$CONFIG")
if [[ -n "$BUCKET_DATASET" ]]; then
  ARGS+=(--set "closed_loop.bucket_dataset=$BUCKET_DATASET" --set "closed_loop.bucket_split=$BUCKET_SPLIT" --set closed_loop.require_bucket_targets=true)
fi
if [[ -n "$TARGET_KEYS_FILE" ]]; then
  ARGS+=(--set "closed_loop.target_keys_file=$TARGET_KEYS_FILE" --set "closed_loop.require_target_keys=$REQUIRE_TARGET_KEYS")
fi
[[ -n "$RAW_MAX_SCENARIOS" ]] && ARGS+=(--set "closed_loop.raw_max_scenarios=$RAW_MAX_SCENARIOS")
[[ -n "$NUM_CANDIDATES" ]] && ARGS+=(--set "closed_loop.num_candidate_prefixes=$NUM_CANDIDATES")
[[ -n "$NUM_RECOVERY_OPTIONS" ]] && ARGS+=(--set "closed_loop.num_recovery_options=$NUM_RECOVERY_OPTIONS")

printf 'Running:'; printf ' %q' "${ARGS[@]}"; printf '\n'
"${ARGS[@]}" 2>&1 | tee -a "${OUTPUT%.json}.log"
