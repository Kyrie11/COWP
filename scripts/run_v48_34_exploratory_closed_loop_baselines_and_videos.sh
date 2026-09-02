#!/usr/bin/env bash
set -Eeuo pipefail

# Exploratory-only paired closed loop after a completed RC=0/20 certificate.
# Results are suitable for progress discussion, not deployment or paper claims
# when the Natural gate has not passed.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

: "${MODEL_RUN:?set MODEL_RUN to the completed v48.34/v48.34.1 main-run directory}"
: "${OUT:=runs/ocrap_v48_34_1_exploratory_closed_loop}"
: "${ALLOW_DIAGNOSTIC_RC20:=0}"
: "${EXPLORATORY_DATA_SCOPE:=adaptation_dev}" # adaptation_dev | heldout_test
: "${ALLOW_HELDOUT_TEST_DIAGNOSTIC:=0}"
: "${VARIANTS:=balanced,precision}"
: "${VIDEO_VARIANTS:=precision}"
: "${CUDA_DEVICES:=0,1}"
: "${MAX_TARGETS:=50}"
: "${MAX_STEPS:=40}"
: "${NUM_CANDIDATES:=24}"
: "${NUM_RECOVERY_OPTIONS:=12}"
: "${NUM_POSITIVE_VIDEOS:=3}"
: "${NUM_FAILURE_VIDEOS:=2}"
: "${VIDEO_FPS:=10}"
: "${REPORT_BOOTSTRAP:=2000}"
: "${RUN_SAFE:=1}"
: "${RUN_NEAR:=1}"
: "${RUN_CONTACT:=1}"
: "${RUN_EXTERNAL_BASELINES:=1}"
: "${RUN_VIDEOS:=1}"
: "${REQUIRE_ALL_BASELINES:=1}"

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${PROTOCOL_ROOT:=$OCRAP_ROOT/calibration_v48_14_prism_4814}"
: "${WOMD_VAL:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}"

if [[ "$EXPLORATORY_DATA_SCOPE" == adaptation_dev ]]; then
  : "${SAFE_TARGET_DATASET:=${SAFE_ADAPTATION_DEV_DATASET:-}}"
  : "${NEAR_TARGET_DATASET:=$PROTOCOL_ROOT/evidence_adapt_dev_near_contact}"
  : "${CONTACT_TARGET_DATASET:=$PROTOCOL_ROOT/evidence_adapt_dev_contact}"
  : "${SAFE_BUCKET_SPLIT:=${SAFE_ADAPTATION_DEV_SPLIT:-evidence_adapt_dev}}"
  : "${NEAR_BUCKET_SPLIT:=evidence_adapt_dev}"
  : "${CONTACT_BUCKET_SPLIT:=evidence_adapt_dev}"
  : "${SAFE_WOMD_SOURCE:=$WOMD_VAL}"
  : "${NEAR_WOMD_SOURCE:=$WOMD_VAL}"
  # v48.28/v48.29 established that adaptation-dev Contact was built from the
  # standard validation source, not validation_interactive.
  : "${CONTACT_WOMD_SOURCE:=$WOMD_VAL}"
elif [[ "$EXPLORATORY_DATA_SCOPE" == heldout_test ]]; then
  : "${SAFE_TARGET_DATASET:=$OCRAP_ROOT/test_safe}"
  : "${NEAR_TARGET_DATASET:=$OCRAP_ROOT/test_near_contact}"
  : "${CONTACT_TARGET_DATASET:=$OCRAP_ROOT/test_contact}"
  : "${SAFE_BUCKET_SPLIT:=test}"
  : "${NEAR_BUCKET_SPLIT:=test}"
  : "${CONTACT_BUCKET_SPLIT:=test}"
  : "${SAFE_WOMD_SOURCE:=$WOMD_VAL}"
  : "${NEAR_WOMD_SOURCE:=$WOMD_VAL}"
  : "${CONTACT_WOMD_SOURCE:=${CONTACT_HELDOUT_WOMD_SOURCE:-$WOMD_VAL_INTERACTIVE}}"
else
  echo "EXPLORATORY_DATA_SCOPE must be adaptation_dev or heldout_test" >&2
  exit 2
fi

: "${SAFE_METHODS:=wayformer_bc,gameformer_lite,betopnet_lite}"
: "${NEAR_METHODS:=gameformer_lite,marc_lite,racp_lite,predictive_safety_filter,cvar_risk_filter}"
: "${CONTACT_METHODS:=postimpact_mpc_lite,post_crash_braking,post_collision_restoration,severity_minimization}"

[[ "$ALLOW_DIAGNOSTIC_RC20" == 1 ]] || { echo "Set ALLOW_DIAGNOSTIC_RC20=1 explicitly." >&2; exit 2; }
if [[ "$EXPLORATORY_DATA_SCOPE" == heldout_test && "$ALLOW_HELDOUT_TEST_DIAGNOSTIC" != 1 ]]; then
  echo "Held-out inspection contaminates future model selection. Set ALLOW_HELDOUT_TEST_DIAGNOSTIC=1 explicitly." >&2
  exit 2
fi
if [[ "$RUN_SAFE" == 1 && -z "$SAFE_TARGET_DATASET" ]]; then
  echo "Set SAFE_ADAPTATION_DEV_DATASET or RUN_SAFE=0 for adaptation-dev exploration." >&2
  exit 2
fi

mkdir -p "$OUT" "$OUT/logs" "$OUT/reports" "$OUT/baselines" "$OUT/models" "$OUT/videos" "$OUT/contracts"
IFS=',' read -r -a GPU_LIST <<< "$CUDA_DEVICES"
[[ ${#GPU_LIST[@]} -ge 1 ]] || GPU_LIST=(0)
GPU0="${GPU_LIST[0]}"; GPU1="${GPU_LIST[1]:-${GPU_LIST[0]}}"

# Select the newest status by created_unix, never lexicographically by filename.
python - "$MODEL_RUN" "$OUT" "$EXPLORATORY_DATA_SCOPE" <<'PY_AUTH'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); scope=sys.argv[3]
files=list(root.glob('V48_*_COMPLETE.json'))
if not files: raise SystemExit(f'no completion status under {root}')
docs=[]
for p in files:
    try: docs.append((float(json.load(open(p)).get('created_unix',0)),p,json.load(open(p))))
    except Exception: pass
if not docs: raise SystemExit('no readable completion status')
_,status_path,d=max(docs,key=lambda x:x[0])
rc=int(d.get('pipeline_exit_code',-1))
if rc not in (0,20) or not d.get('certificate_executed') or not d.get('gate_evaluated') or not d.get('pipeline_valid'):
    raise SystemExit(f'exploratory run requires completed valid RC=0/20, got {d}')
notice={'event':'v48_34_1_exploratory_closed_loop_authorization','created_unix':time.time(),
 'source_completion':str(status_path),'source_pipeline_exit_code':rc,
 'natural_gate_passed':bool(d.get('gate_passed')),'data_scope':scope,
 'exploratory_only':True,'paper_claim_allowed':False,'deployment_allowed':False,
 'may_not_be_used_for_checkpoint_threshold_or_algorithm_selection':True,
 'heldout_test_contaminated_for_future_model_selection':scope=='heldout_test',
 'required_reporting':'all paired scenes, positive examples and failure cases; no video cherry-picking'}
(out/'EXPLORATORY_ONLY_DISCLOSURE.json').write_text(json.dumps(notice,indent=2)+'\n')
print(notice)
PY_AUTH

preflight_regime() {
  local regime="$1" enabled="$2" target="$3" split="$4" womd="$5"
  [[ "$enabled" == 1 ]] || return 0
  [[ -d "$target" ]] || { echo "missing $regime target dataset: $target" >&2; return 3; }
  python tools/check_closed_loop_dataset_support.py \
    --dataset "$target" --split "$split" --womd-pattern "$womd" \
    --expected-source-role auto --output "$OUT/contracts/${regime}_closed_loop_support.json"
}
preflight_regime safe "$RUN_SAFE" "$SAFE_TARGET_DATASET" "$SAFE_BUCKET_SPLIT" "$SAFE_WOMD_SOURCE"
preflight_regime near "$RUN_NEAR" "$NEAR_TARGET_DATASET" "$NEAR_BUCKET_SPLIT" "$NEAR_WOMD_SOURCE"
preflight_regime contact "$RUN_CONTACT" "$CONTACT_TARGET_DATASET" "$CONTACT_BUCKET_SPLIT" "$CONTACT_WOMD_SOURCE"

python - "$OUT" "$EXPLORATORY_DATA_SCOPE" \
  "$RUN_SAFE" "$SAFE_TARGET_DATASET" "$SAFE_BUCKET_SPLIT" "$SAFE_WOMD_SOURCE" \
  "$RUN_NEAR" "$NEAR_TARGET_DATASET" "$NEAR_BUCKET_SPLIT" "$NEAR_WOMD_SOURCE" \
  "$RUN_CONTACT" "$CONTACT_TARGET_DATASET" "$CONTACT_BUCKET_SPLIT" "$CONTACT_WOMD_SOURCE" <<'PY_SCOPE'
import hashlib,json,pathlib,sys,time

def hash_path(text):
    p=pathlib.Path(text)
    if not p.exists(): return None
    h=hashlib.sha256(); h.update(str(p.resolve()).encode())
    manifest=p/'manifest.csv' if p.is_dir() else None
    if manifest and manifest.is_file(): h.update(manifest.read_bytes())
    return h.hexdigest()
out=pathlib.Path(sys.argv[1]); scope=sys.argv[2]; raw=sys.argv[3:]; regimes={}
for i,name in enumerate(('safe','near','contact')):
    enabled=raw[i*4]=='1'; target=raw[i*4+1]; split=raw[i*4+2]; womd=raw[i*4+3]
    regimes[name]={'enabled':enabled,'target_dataset':str(pathlib.Path(target).resolve()) if target else '',
                   'target_contract_sha256':hash_path(target) if enabled else None,
                   'bucket_split':split,'womd_source':womd}
doc={'event':'v48_34_1_exploratory_data_scope_contract','created_unix':time.time(),'data_scope':scope,
     'heldout_test_contaminated_for_future_model_selection':scope=='heldout_test','regimes':regimes}
(out/'EXPLORATORY_DATA_SCOPE_CONTRACT.json').write_text(json.dumps(doc,indent=2)+'\n'); print(doc)
PY_SCOPE

require_file() { [[ -f "$1" ]] || { echo "missing required file: $1" >&2; return 3; }; }
model_ckpt() { printf '%s/candidates/%s/model_v48_trac_sr/best.pt' "$MODEL_RUN" "$1"; }
model_base() { printf '%s/dedicated_candidates/%s' "$MODEL_RUN" "$1"; }

run_ocrap_regime() {
  local variant="$1" regime="$2" target="$3" split="$4" womd="$5" run_dir="$6" gpu="$7" render="${8:-false}" max_per_scene="${9:-1}"
  local ckpt base; ckpt="$(model_ckpt "$variant")"; base="$(model_base "$variant")"
  require_file "$ckpt"; require_file "$base/calibration/direct_value_risk_near_v48.json"; require_file "$base/calibration/direct_value_risk_contact_v48.json"
  mkdir -p "$run_dir"
  local common=(env RUN="$run_dir" BASE_RUN="$base" CKPT="$ckpt" DEV_SHADOW_DIAGNOSTIC=1
    RUN_OFFLINE_EVAL=0 RUN_SAFE_CLOSED_LOOP=0 RUN_AUDITS=1 RUN_SCALAR_BASELINES=1 PAIR_BUCKETS_ON_TWO_GPUS=0
    AUDIT_TARGETS="$MAX_TARGETS" AUDIT_MAX_ROLLOUTS="$MAX_TARGETS" AUDIT_MAX_STEPS="$MAX_STEPS"
    AUDIT_NUM_CANDIDATES="$NUM_CANDIDATES" AUDIT_NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS"
    AUDIT_LABELS=0 AUDIT_LABEL_MODE=fast AUDIT_EVERY_N_STEPS=8 AUDIT_TOP_K=5 AUDIT_MAX_EXTRA_CANDIDATES=2
    AUDIT_MAX_TARGETS_PER_SCENE="$max_per_scene" CL_RENDER_TRACE="$render" CL_RENDER_MAX_AGENTS=64
    CL_RESUME="${CL_RESUME:-true}" BUCKET_SPLIT="$split" DEV_SHADOW_WOMD_SOURCE="$womd"
    GPU_NEAR="$gpu" GPU_CONTACT="$gpu")
  if [[ "$regime" == near ]]; then
    "${common[@]}" NEAR_TEST="$target" CONTACT_TEST="$CONTACT_TARGET_DATASET" RUN_NEAR_AUDIT=1 RUN_CONTACT_AUDIT=0 \
      bash scripts/run_ocrap_v48_trac_sr.sh > >(tee "$run_dir/controller.log") 2>&1
  else
    "${common[@]}" NEAR_TEST="$NEAR_TARGET_DATASET" CONTACT_TEST="$target" RUN_NEAR_AUDIT=0 RUN_CONTACT_AUDIT=1 \
      bash scripts/run_ocrap_v48_trac_sr.sh > >(tee "$run_dir/controller.log") 2>&1
  fi
}

run_ocrap_safe() {
  local variant="$1" target="$2" split="$3" womd="$4" run_dir="$5" render="${6:-false}"
  local ckpt; ckpt="$(model_ckpt "$variant")"; require_file "$ckpt"; mkdir -p "$run_dir"
  env RUN="$run_dir" CKPT="$ckpt" SAFE_NOMINAL_ONLY=1 RUN_OFFLINE_EVAL=0 RUN_AUDITS=0 \
    RUN_SAFE_CLOSED_LOOP=1 RUN_SAFE_PAIRED_SCALAR=1 SAFE_TEST="$target" SAFE_BUCKET_SPLIT="$split" SAFE_WOMD_SOURCE="$womd" \
    SAFE_MAX_TARGETS="$MAX_TARGETS" SAFE_MAX_ROLLOUTS="$MAX_TARGETS" SAFE_MAX_STEPS="$MAX_STEPS" \
    SAFE_NUM_CANDIDATES="$NUM_CANDIDATES" SAFE_NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
    CL_RENDER_TRACE="$render" CL_RENDER_MAX_AGENTS=64 CL_RESUME="${CL_RESUME:-true}" \
    GPU_SAFE_BASELINE="$GPU0" GPU_SAFE="$GPU1" bash scripts/run_ocrap_v48_trac_sr.sh \
    > >(tee "$run_dir/controller.log") 2>&1
}

external_checkpoint() {
  case "$1" in
    wayformer_bc) printf '%s' "${SAFE_WAYFORMER_CHECKPOINT:-}" ;;
    gameformer_lite) [[ "$2" == safe ]] && printf '%s' "${SAFE_GAMEFORMER_CHECKPOINT:-}" || printf '%s' "${NEAR_GAMEFORMER_CHECKPOINT:-}" ;;
    betopnet_lite) printf '%s' "${SAFE_BETOPNET_CHECKPOINT:-}" ;;
    *) printf '' ;;
  esac
}
external_config() {
  case "$1:$2" in
    safe:wayformer_bc) echo configs/external_baselines/wayformer_bc.yaml ;;
    safe:gameformer_lite) echo configs/external_baselines/gameformer_lite.yaml ;;
    safe:betopnet_lite) echo configs/external_baselines/betopnet_lite.yaml ;;
    near:gameformer_lite) echo configs/external_baselines/near_contact_gameformer_lite.yaml ;;
    near:*) echo configs/external_baselines/near_contact_external_baselines.yaml ;;
    contact:*) echo configs/external_baselines/contact_external_baselines.yaml ;;
    *) return 2 ;;
  esac
}
run_external_one() {
  local regime="$1" method="$2" target="$3" split="$4" womd="$5" gpu="$6"
  local cfg ckpt out_dir out_file; cfg="$(external_config "$regime" "$method")"; ckpt="$(external_checkpoint "$method" "$regime")"
  if [[ "$method" =~ ^(wayformer_bc|gameformer_lite|betopnet_lite)$ && ( -z "$ckpt" || ! -f "$ckpt" ) ]]; then
    [[ "$REQUIRE_ALL_BASELINES" == 1 ]] && { echo "missing $regime baseline checkpoint for $method: ${ckpt:-unset}" >&2; return 3; }
    echo "[SKIP] missing $regime baseline checkpoint for $method" >&2; return 0
  fi
  out_dir="$OUT/baselines/$regime"; mkdir -p "$out_dir"; out_file="$out_dir/closed_loop_${method}.json"
  local ckpt_args=(); [[ -z "$ckpt" ]] || ckpt_args=(--checkpoint "$ckpt")
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m ocrap.cli closed-loop --config "$cfg" --dataset "$womd" "${ckpt_args[@]}" --output "$out_file" \
    --set closed_loop.method="$method" --set closed_loop.bucket_dataset="$target" --set closed_loop.bucket_split="$split" \
    --set closed_loop.require_bucket_targets=true --set closed_loop.allow_legacy_source_index_targets=true \
    --set closed_loop.max_bucket_targets="$MAX_TARGETS" --set closed_loop.max_targets_per_scene=1 \
    --set closed_loop.max_rollouts="$MAX_TARGETS" --set closed_loop.raw_max_scenarios=0 --set closed_loop.max_steps="$MAX_STEPS" \
    --set closed_loop.replan_interval_steps=1 --set closed_loop.label_mode=fast --set closed_loop.audit_max_labels=0 \
    --set closed_loop.force_teacher_baselines=false --set closed_loop.external_sparse_labels=false \
    --set closed_loop.exhaustive_teacher_labels=false --set closed_loop.num_candidate_prefixes="$NUM_CANDIDATES" \
    --set closed_loop.num_recovery_options="$NUM_RECOVERY_OPTIONS" --set closed_loop.save_partial=true --set closed_loop.resume=true \
    --set closed_loop.render_trace=false --set closed_loop.profile_timing=true \
    --set waymax.retain_official_scenario_id=true --set waymax.dataloader_include_sdc_paths=false \
    --set waymax.compute_future_metrics=false --set waymax.teacher_metrics_stride=0 --set waymax.use_jit_scan_rollouts=true \
    > >(tee "$out_dir/closed_loop_${method}.log") 2>&1
}
run_method_list() {
  local regime="$1" csv="$2" target="$3" split="$4" womd="$5"
  IFS=',' read -r -a methods <<< "$csv"; local i=0 pids=() names=()
  for method in "${methods[@]}"; do
    method="${method//[[:space:]]/}"; [[ -n "$method" ]] || continue
    local gpu="$GPU0"; (( i % 2 == 0 )) || gpu="$GPU1"
    run_external_one "$regime" "$method" "$target" "$split" "$womd" "$gpu" & pids+=("$!"); names+=("$regime:$method"); i=$((i+1))
    if (( ${#pids[@]} == 2 )); then
      local failed=0; for j in "${!pids[@]}"; do wait "${pids[$j]}" || { echo "[ERROR] ${names[$j]}" >&2; failed=1; }; done
      (( failed == 0 )) || return 1; pids=(); names=()
    fi
  done
  local failed=0; for j in "${!pids[@]}"; do wait "${pids[$j]}" || { echo "[ERROR] ${names[$j]}" >&2; failed=1; }; done
  (( failed == 0 ))
}

IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"
# Safe paired control uses both GPUs and therefore runs variants sequentially.
if [[ "$RUN_SAFE" == 1 ]]; then
  for variant in "${VARIANT_LIST[@]}"; do variant="${variant//[[:space:]]/}"; [[ -n "$variant" ]] || continue; run_ocrap_safe "$variant" "$SAFE_TARGET_DATASET" "$SAFE_BUCKET_SPLIT" "$SAFE_WOMD_SOURCE" "$OUT/models/$variant/safe" false; done
fi
# Near and Contact use one GPU per variant. Balanced and Precision run together.
run_model_regime_parallel() {
  local regime="$1" enabled="$2" target="$3" split="$4" womd="$5"
  [[ "$enabled" == 1 ]] || return 0
  local pids=() names=() i=0
  for variant in "${VARIANT_LIST[@]}"; do
    variant="${variant//[[:space:]]/}"; [[ -n "$variant" ]] || continue
    local gpu="$GPU0"; (( i % 2 == 0 )) || gpu="$GPU1"
    run_ocrap_regime "$variant" "$regime" "$target" "$split" "$womd" "$OUT/models/$variant/$regime" "$gpu" false 1 &
    pids+=("$!"); names+=("$variant:$regime"); i=$((i+1))
  done
  local failed=0; for j in "${!pids[@]}"; do wait "${pids[$j]}" || { echo "[ERROR] ${names[$j]}" >&2; failed=1; }; done
  (( failed == 0 ))
}
run_model_regime_parallel near "$RUN_NEAR" "$NEAR_TARGET_DATASET" "$NEAR_BUCKET_SPLIT" "$NEAR_WOMD_SOURCE"
run_model_regime_parallel contact "$RUN_CONTACT" "$CONTACT_TARGET_DATASET" "$CONTACT_BUCKET_SPLIT" "$CONTACT_WOMD_SOURCE"

if [[ "$RUN_EXTERNAL_BASELINES" == 1 ]]; then
  [[ "$RUN_SAFE" != 1 ]] || run_method_list safe "$SAFE_METHODS" "$SAFE_TARGET_DATASET" "$SAFE_BUCKET_SPLIT" "$SAFE_WOMD_SOURCE"
  [[ "$RUN_NEAR" != 1 ]] || run_method_list near "$NEAR_METHODS" "$NEAR_TARGET_DATASET" "$NEAR_BUCKET_SPLIT" "$NEAR_WOMD_SOURCE"
  [[ "$RUN_CONTACT" != 1 ]] || run_method_list contact "$CONTACT_METHODS" "$CONTACT_TARGET_DATASET" "$CONTACT_BUCKET_SPLIT" "$CONTACT_WOMD_SOURCE"
fi

make_report() {
  local variant="$1" regime="$2" reference="$3" model="$4" methods_csv="$5"
  local args=(--regime "$regime" --reference "scalar=$reference" --method "ocrap_${variant}=$model")
  IFS=',' read -r -a methods <<< "$methods_csv"
  for method in "${methods[@]}"; do method="${method//[[:space:]]/}"; [[ -n "$method" ]] || continue; local p="$OUT/baselines/$regime/closed_loop_${method}.json"; [[ -s "$p.scenes.jsonl" ]] && args+=(--method "$method=$p"); done
  python tools/build_v48_34_paired_baseline_report.py "${args[@]}" --bootstrap "$REPORT_BOOTSTRAP" --require-core-metrics \
    --output-json "$OUT/reports/${variant}_${regime}_paired.json" --output-csv "$OUT/reports/${variant}_${regime}_paired.csv"
}
for variant in "${VARIANT_LIST[@]}"; do
  variant="${variant//[[:space:]]/}"; [[ -n "$variant" ]] || continue
  [[ "$RUN_SAFE" != 1 ]] || make_report "$variant" safe "$OUT/models/$variant/safe/closed_loop_safe_fast_v48_scalar.json" "$OUT/models/$variant/safe/closed_loop_safe_fast_v48.json" "$SAFE_METHODS"
  [[ "$RUN_NEAR" != 1 ]] || make_report "$variant" near "$OUT/models/$variant/near/audit_near_contact_selected_topk_v48_scalar.json" "$OUT/models/$variant/near/audit_near_contact_selected_topk_v48_v48.json" "$NEAR_METHODS"
  [[ "$RUN_CONTACT" != 1 ]] || make_report "$variant" contact "$OUT/models/$variant/contact/audit_contact_selected_topk_v48_scalar.json" "$OUT/models/$variant/contact/audit_contact_selected_topk_v48_v48.json" "$CONTACT_METHODS"
done

enabled_regimes=""; [[ "$RUN_SAFE" == 1 ]] && enabled_regimes=safe; [[ "$RUN_NEAR" == 1 ]] && enabled_regimes="${enabled_regimes:+$enabled_regimes,}near"; [[ "$RUN_CONTACT" == 1 ]] && enabled_regimes="${enabled_regimes:+$enabled_regimes,}contact"
python tools/build_v48_34_1_progress_tables.py --reports-dir "$OUT/reports" --output-dir "$OUT/reports/display" --variants "$VARIANTS" --enabled-regimes "$enabled_regimes"

run_video_regime() {
  local variant="$1" regime="$2" target="$3" split="$4" womd="$5" gpu="$6"
  local full="$OUT/models/$variant/$regime" model_file control_file
  if [[ "$regime" == near ]]; then model_file="$full/audit_near_contact_selected_topk_v48_v48.json"; control_file="$full/audit_near_contact_selected_topk_v48_scalar.json"; else model_file="$full/audit_contact_selected_topk_v48_v48.json"; control_file="$full/audit_contact_selected_topk_v48_scalar.json"; fi
  local video_root selection subset
  video_root="$OUT/videos/$variant/$regime"
  selection="$video_root/critical_scene_selection.json"
  subset="$video_root/target_subset"
  mkdir -p "$video_root"
  python tools/select_critical_scenes_v48_34.py --method-scenes "$model_file" --control-scenes "$control_file" --regime "$regime" --num-positive "$NUM_POSITIVE_VIDEOS" --num-failure "$NUM_FAILURE_VIDEOS" --output "$selection"
  python tools/subset_dataset_targets_v48_34.py --input "$target" --selection "$selection" --output "$subset" --overwrite
  local n; n="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("selected",[])))' "$selection")"
  [[ "$n" != 0 ]] || { echo "[WARN] no selected scenes for $variant/$regime"; return 0; }
  local trace_run="$video_root/trace_run"
  MAX_TARGETS="$n" CL_RESUME=false run_ocrap_regime "$variant" "$regime" "$subset" "$split" "$womd" "$trace_run" "$gpu" true "$n"
  local trace_model trace_control
  if [[ "$regime" == near ]]; then trace_model="$trace_run/audit_near_contact_selected_topk_v48_v48.json"; trace_control="$trace_run/audit_near_contact_selected_topk_v48_scalar.json"; else trace_model="$trace_run/audit_contact_selected_topk_v48_v48.json"; trace_control="$trace_run/audit_contact_selected_topk_v48_scalar.json"; fi
  python tools/render_critical_scenes_v48_34.py --method-scenes "$trace_model" --control-scenes "$trace_control" --selection "$selection" --output-dir "$video_root/rendered" --fps "$VIDEO_FPS"
}
if [[ "$RUN_VIDEOS" == 1 ]]; then
  IFS=',' read -r -a VIDEO_LIST <<< "$VIDEO_VARIANTS"; local_i=0
  for variant in "${VIDEO_LIST[@]}"; do
    variant="${variant//[[:space:]]/}"; [[ -n "$variant" ]] || continue
    gpu="$GPU0"; (( local_i % 2 == 0 )) || gpu="$GPU1"; local_i=$((local_i+1))
    [[ "$RUN_NEAR" != 1 ]] || run_video_regime "$variant" near "$NEAR_TARGET_DATASET" "$NEAR_BUCKET_SPLIT" "$NEAR_WOMD_SOURCE" "$gpu"
    [[ "$RUN_CONTACT" != 1 ]] || run_video_regime "$variant" contact "$CONTACT_TARGET_DATASET" "$CONTACT_BUCKET_SPLIT" "$CONTACT_WOMD_SOURCE" "$gpu"
  done
fi

python - "$OUT" <<'PY_DONE'
import json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); report_index=out/'reports'/'display'/'ALL_REGIMES_REPORT_INDEX.json'
index=json.load(open(report_index))
d={'event':'v48_34_1_exploratory_closed_loop_complete','created_unix':time.time(),
   'exploratory_only':True,'paper_claim_allowed':False,'deployment_allowed':False,
   'report_index':str(report_index),'reports_complete':bool(index.get('valid')),
   'paired_reports':sorted(str(p) for p in (out/'reports').glob('*_paired.json')),
   'display_tables':sorted(str(p) for p in (out/'reports'/'display').glob('*')),
   'video_indices':sorted(str(p) for p in (out/'videos').rglob('VIDEO_INDEX.json'))}
(out/'EXPLORATORY_CLOSED_LOOP_COMPLETE.json').write_text(json.dumps(d,indent=2)+'\n'); print(d)
if not d['reports_complete']: raise SystemExit(4)
PY_DONE
