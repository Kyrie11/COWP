#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=scripts/lib/v50_runtime.sh
source scripts/lib/v50_runtime.sh

: "${SELECTION_ROOT:?Set SELECTION_ROOT to comparison/selection containing near/contact selections and best-external JSONs}"
: "${OCRAP_MODEL_RUN:=runs/ocrap_v48_34_barrier_crossfit_dedicated_4834}"
: "${MODEL_VARIANT:=balanced}"
if [[ -z "${MODEL_CANDIDATE_ROOT:-}" ]]; then
  MODEL_CANDIDATE_ROOT="$OCRAP_MODEL_RUN/candidates/$MODEL_VARIANT"
  if [[ ! -f "$MODEL_CANDIDATE_ROOT/model_v48_trac_sr/best.pt" && -f "$OCRAP_MODEL_RUN/dedicated_candidates/$MODEL_VARIANT/model_v48_trac_sr/best.pt" ]]; then
    MODEL_CANDIDATE_ROOT="$OCRAP_MODEL_RUN/dedicated_candidates/$MODEL_VARIANT"
  fi
fi
: "${OCRAP_CHECKPOINT:=$MODEL_CANDIDATE_ROOT/model_v48_trac_sr/best.pt}"
: "${GAMMA_REC_JSON:=$MODEL_CANDIDATE_ROOT/calibration/gamma_rec_by_bucket_v48.json}"
: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${EXTERNAL_CHECKPOINT_ROOT:?Set EXTERNAL_CHECKPOINT_ROOT used by the full external-baseline run}"
: "${OUT:=runs/external_comparison_v50/selective_traces}"
: "${WOMD_NUM_SHARDS:=150}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord@150}"
: "${NEAR_WOMD:=$WOMD_VAL_INTERACTIVE}"
: "${CONTACT_WOMD:=$WOMD_VAL_INTERACTIVE}"
: "${NEAR_BUCKET:=$OCRAP_ROOT/test_near_contact}"
: "${CONTACT_BUCKET:=$OCRAP_ROOT/test_contact}"
: "${MAX_STEPS:=40}"
: "${NUM_CANDIDATES:=24}"
: "${NUM_RECOVERY_OPTIONS:=12}"
: "${CUDA_DEVICES:=0,1}"
: "${FPS:=10}"
: "${ALLOW_DIAGNOSTIC_RC20:=0}"
[[ "$ALLOW_DIAGNOSTIC_RC20" == 1 ]] || { echo "Set ALLOW_DIAGNOSTIC_RC20=1 for the current RC=20 checkpoint." >&2; exit 2; }
[[ -f "$OCRAP_CHECKPOINT" ]] || { echo "Missing OC-RAP checkpoint: $OCRAP_CHECKPOINT" >&2; exit 2; }
[[ -f "$GAMMA_REC_JSON" ]] || { echo "Missing gamma JSON: $GAMMA_REC_JSON" >&2; exit 2; }

NEAR_WOMD="$(v50_normalize_womd_spec "$NEAR_WOMD" "$WOMD_NUM_SHARDS")"
CONTACT_WOMD="$(v50_normalize_womd_spec "$CONTACT_WOMD" "$WOMD_NUM_SHARDS")"
NEAR_SOURCE_KEYS="$SELECTION_ROOT/near_target_keys.json"
CONTACT_SOURCE_KEYS="$SELECTION_ROOT/contact_target_keys.json"
NEAR_SOURCE_SELECTION="$SELECTION_ROOT/near_selection.json"
CONTACT_SOURCE_SELECTION="$SELECTION_ROOT/contact_selection.json"
NEAR_BEST="$SELECTION_ROOT/near_best_external.json"
CONTACT_BEST="$SELECTION_ROOT/contact_best_external.json"
for p in "$NEAR_SOURCE_KEYS" "$CONTACT_SOURCE_KEYS" "$NEAR_SOURCE_SELECTION" "$CONTACT_SOURCE_SELECTION" "$NEAR_BEST" "$CONTACT_BEST"; do [[ -f "$p" ]] || { echo "Missing selection artifact: $p" >&2; exit 2; }; done

read -r NEAR_METHOD CONTACT_METHOD NEAR_GAMMA CONTACT_GAMMA < <(python - "$NEAR_BEST" "$CONTACT_BEST" "$GAMMA_REC_JSON" <<'PY'
import json,sys
near=json.load(open(sys.argv[1]))['best']['method']; contact=json.load(open(sys.argv[2]))['best']['method']
g=json.load(open(sys.argv[3]))['gamma_rec_by_bucket']
print(near,contact,g['test_near_contact'],g['test_contact'])
PY
)
IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"; ((${#GPUS[@]})) || GPUS=(0)
GPU0="${GPUS[0]}"; GPU1="${GPUS[1]:-${GPUS[0]}}"
mkdir -p "$OUT/near/ocrap" "$OUT/near/baseline" "$OUT/contact/ocrap" "$OUT/contact/baseline" "$OUT/videos"

# Historical full-run journals can carry legacy waymax_<hash> target keys even
# after the bucket was rebuilt with new canonical scene ids. Resolve them once,
# explicitly and uniquely, using exact aliases or the stable __wx source index.
NEAR_KEYS="$OUT/near/resolved_target_keys.json"
CONTACT_KEYS="$OUT/contact/resolved_target_keys.json"
NEAR_SELECTION="$OUT/near/resolved_selection.json"
CONTACT_SELECTION="$OUT/contact/resolved_selection.json"
python tools/resolve_selected_targets_v50.py \
  --dataset "$NEAR_BUCKET" --split test --selection "$NEAR_SOURCE_SELECTION" --category positive_toy_example --max-items 5 \
  --target-keys-output "$NEAR_KEYS" --selection-output "$NEAR_SELECTION" \
  --report-output "$OUT/near/target_resolution.json"
python tools/resolve_selected_targets_v50.py \
  --dataset "$CONTACT_BUCKET" --split test --selection "$CONTACT_SOURCE_SELECTION" --category positive_toy_example --max-items 5 \
  --target-keys-output "$CONTACT_KEYS" --selection-output "$CONTACT_SELECTION" \
  --report-output "$OUT/contact/target_resolution.json"

# Fail before launching two workers if either WOMD shard set or resolved target set is invalid.
python tools/check_closed_loop_dataset_support.py --dataset "$NEAR_BUCKET" --split test --womd "$NEAR_WOMD" --target-keys-file "$NEAR_KEYS" --require-target-keys --output "$OUT/near/preflight.json"
python tools/check_closed_loop_dataset_support.py --dataset "$CONTACT_BUCKET" --split test --womd "$CONTACT_WOMD" --target-keys-file "$CONTACT_KEYS" --require-target-keys --output "$OUT/contact/preflight.json"

run_ocrap_trace() {
  local regime="$1" womd="$2" bucket="$3" keys="$4" gamma="$5" gpu="$6"
  env RUN_DIR="$OUT/$regime/ocrap" OUTPUT="$OUT/$regime/ocrap/closed_loop_ocrap.json" \
    WOMD_VAL="$womd" WOMD_NUM_SHARDS="$WOMD_NUM_SHARDS" CHECKPOINT="$OCRAP_CHECKPOINT" GAMMA_REC="$gamma" GPU="$gpu" \
    MAX_SCENARIOS=0 MAX_STEPS="$MAX_STEPS" LABEL_MODE=fast AUDIT_EVERY_N_STEPS=0 \
    NUM_CANDIDATES="$NUM_CANDIDATES" NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
    BUCKET_DATASET="$bucket" BUCKET_SPLIT=test MAX_TARGETS_PER_SCENE=1 TARGET_KEYS_FILE="$keys" REQUIRE_TARGET_KEYS=true \
    RENDER_TRACE=true RENDER_MAX_AGENTS=48 SAVE_PARTIAL=true RESUME_FORCE=false \
    bash scripts/run_ocrap_closed_loop_optimized.sh
}

run_external_trace() {
  local regime="$1" method="$2" womd="$3" bucket="$4" keys="$5" gpu="$6"
  local config checkpoint_args=()
  if [[ "$regime" == near ]]; then
    if [[ "$method" == gameformer_lite ]]; then
      config=configs/external_baselines/near_contact_gameformer_lite.yaml
      local ckpt="$EXTERNAL_CHECKPOINT_ROOT/near/gameformer_lite/best.pt"
      python tools/validate_external_checkpoint.py --checkpoint "$ckpt" --require-deployable-contract >/dev/null
      checkpoint_args=(--checkpoint "$ckpt")
    else
      config=configs/external_baselines/near_contact_external_baselines.yaml
    fi
  else
    config=configs/external_baselines/contact_external_baselines.yaml
  fi
  local run_dir="$OUT/$regime/baseline" output="$OUT/$regime/baseline/closed_loop_${method}.json"
  mkdir -p "$run_dir"
  env CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_COMPILATION_CACHE_DIR="$run_dir/.jax_compilation_cache" JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
    python -u -m ocrap.cli closed-loop --config "$config" --dataset "$womd" "${checkpoint_args[@]}" --output "$output" \
      --set "closed_loop.method=$method" --set closed_loop.max_scenarios=0 --set closed_loop.max_bucket_targets=0 \
      --set "closed_loop.bucket_dataset=$bucket" --set closed_loop.bucket_split=test --set closed_loop.require_bucket_targets=true \
      --set closed_loop.max_targets_per_scene=1 --set "closed_loop.target_keys_file=$keys" --set closed_loop.require_target_keys=true \
      --set closed_loop.render_trace=true --set closed_loop.render_max_agents=48 --set "closed_loop.max_steps=$MAX_STEPS" \
      --set closed_loop.replan_interval_steps=1 --set closed_loop.label_mode=fast --set closed_loop.audit_every_n_steps=0 \
      --set closed_loop.force_teacher_baselines=false --set closed_loop.external_sparse_labels=true \
      --set "closed_loop.num_candidate_prefixes=$NUM_CANDIDATES" --set "closed_loop.num_recovery_options=$NUM_RECOVERY_OPTIONS" \
      --set closed_loop.save_partial=true --set closed_loop.resume_force=false --set closed_loop.profile_timing=true \
      --set closed_loop.partial_write_every_scenes=5 --set closed_loop.progress_every_steps=10 \
      --set closed_loop.result_scene_detail=full --set closed_loop.scene_journal_detail=full --set closed_loop.memory_scene_detail=full \
      --set closed_loop.include_scenes_in_result=false --set closed_loop.include_scenes_in_partial=false \
      --set waymax.dataloader_include_sdc_paths=false --set waymax.compute_future_metrics=false --set waymax.teacher_metrics_stride=0 --set waymax.use_jit_scan_rollouts=true \
      2>&1 | tee "$run_dir/closed_loop_${method}.log"
}

run_pair() {
  local regime="$1" method="$2" womd="$3" bucket="$4" keys="$5" gamma="$6"
  if [[ "$GPU0" == "$GPU1" ]]; then
    # One visible GPU: serialize the two trace reruns to avoid OOM and compilation-cache thrashing.
    run_ocrap_trace "$regime" "$womd" "$bucket" "$keys" "$gamma" "$GPU0"
    run_external_trace "$regime" "$method" "$womd" "$bucket" "$keys" "$GPU0"
  else
    run_ocrap_trace "$regime" "$womd" "$bucket" "$keys" "$gamma" "$GPU0" & local p1=$!
    run_external_trace "$regime" "$method" "$womd" "$bucket" "$keys" "$GPU1" & local p2=$!
    local failed=0; wait "$p1" || failed=1; wait "$p2" || failed=1; ((failed==0))
  fi
}

run_pair near "$NEAR_METHOD" "$NEAR_WOMD" "$NEAR_BUCKET" "$NEAR_KEYS" "$NEAR_GAMMA"
run_pair contact "$CONTACT_METHOD" "$CONTACT_WOMD" "$CONTACT_BUCKET" "$CONTACT_KEYS" "$CONTACT_GAMMA"

NEAR_OCRAP_SCENES="$OUT/near/ocrap/closed_loop_ocrap.json.scenes.jsonl" \
NEAR_BASELINE_SCENES="$OUT/near/baseline/closed_loop_${NEAR_METHOD}.json.scenes.jsonl" \
CONTACT_OCRAP_SCENES="$OUT/contact/ocrap/closed_loop_ocrap.json.scenes.jsonl" \
CONTACT_BASELINE_SCENES="$OUT/contact/baseline/closed_loop_${CONTACT_METHOD}.json.scenes.jsonl" \
NEAR_SELECTION="$NEAR_SELECTION" CONTACT_SELECTION="$CONTACT_SELECTION" \
NEAR_BASELINE_NAME="$NEAR_METHOD" CONTACT_BASELINE_NAME="$CONTACT_METHOD" \
FPS="$FPS" OUT="$OUT/videos" bash scripts/build_top10_recovery_videos.sh

python - "$OUT" "$NEAR_METHOD" "$CONTACT_METHOD" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); doc={'event':'selective_recovery_trace_run_v50','near_baseline':sys.argv[2],'contact_baseline':sys.argv[3],'trace_scope':'5 selected near + 5 selected contact targets only','video_index':str(out/'videos/TOP10_VIDEO_INDEX.json')}
(out/'SELECTIVE_TRACE_INDEX.json').write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps(doc))
PY
