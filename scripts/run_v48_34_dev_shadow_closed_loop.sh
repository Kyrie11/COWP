#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

OUT="${OUT:?OUT is required; point it to the v48.34 dedicated run}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
DEV_NEAR="${DEV_NEAR:-$PROTOCOL_ROOT/evidence_adapt_dev_near_contact}"
DEV_CONTACT="${DEV_CONTACT:-$PROTOCOL_ROOT/evidence_adapt_dev_contact}"
ROOT="$OUT/dev_shadow_closed_loop${SHADOW_ROOT_SUFFIX:-}"
mkdir -p "$ROOT"

# Shadow is diagnostic-only, but it still needs a completed calibration.  It may
# run after RC=20 to diagnose transfer, never after an RC=30 pipeline failure.
set +e
python - "$OUT" "$ROOT" <<'PY_PREFLIGHT'
import json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2])
status_path=out/'V48_34_COMPLETE.json'
reason=None; status={}
try: status=json.load(open(status_path))
except Exception as e: reason=f'missing_or_invalid_main_status:{e}'
if reason is None and not bool(status.get('pipeline_valid')):
    reason='main_pipeline_invalid'
if reason is None and not bool(status.get('gate_evaluated')):
    reason='certificate_not_evaluated'
if reason is None and status.get('certificate_exit_code') not in (0,20):
    reason=f'unexpected_certificate_exit:{status.get("certificate_exit_code")}'
for variant in ('balanced','precision'):
    if reason is not None: break
    base=out/'candidates'/variant
    if not (base/'model_v48_trac_sr'/'best.pt').is_file(): reason=f'missing_checkpoint:{variant}'; break
    gamma=base/'calibration'/'gamma_rec_by_bucket_v48.json'
    if not gamma.is_file(): reason=f'missing_gamma:{variant}'; break
    try:
        g=json.load(open(gamma))
        vals=[float(x) for x in (g.get('gamma_rec_by_bucket') or g.get('gamma_by_bucket') or {}).values()]
        if not vals or max(vals) <= 0: reason=f'nonpositive_gamma:{variant}'
    except Exception as e: reason=f'invalid_gamma:{variant}:{e}'
if reason is not None:
    doc={'event':'v48_34_dev_shadow_blocked','created_unix':time.time(),'reason':reason,
         'main_status':status,'uses_test_or_stress':False,'paper_result':False}
    (root/'SHADOW_BLOCKED.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    raise SystemExit(30)
(root/'SHADOW_PREFLIGHT.json').write_text(json.dumps({
  'event':'v48_34_dev_shadow_preflight','created_unix':time.time(),
  'valid':True,'certificate_exit_code':status.get('certificate_exit_code'),
  'gate_passed':bool(status.get('gate_passed')),'paper_result':False,
},ensure_ascii=False,indent=2)+'\n')
PY_PREFLIGHT
preflight_rc=$?
set -e
[[ "$preflight_rc" == 0 ]] || exit 30

# Physical paired metrics do not require online OC-MERO teacher relabeling.
# v48.28 spent >98% of wall time in selected_topk audit labels.  Fast mode
# preserves policy execution and Waymax physics while reducing multi-hour runs
# to the actual simulator/model cost.  A separate sparse diagnostic can be
# requested explicitly without overwriting the physical result directory.
SHADOW_LABEL_MODE="${SHADOW_LABEL_MODE:-fast}"
SHADOW_AUDIT_LABELS="${SHADOW_AUDIT_LABELS:-0}"
SHADOW_AUDIT_EVERY_N_STEPS="${SHADOW_AUDIT_EVERY_N_STEPS:-8}"
SHADOW_AUDIT_TOP_K="${SHADOW_AUDIT_TOP_K:-5}"
SHADOW_AUDIT_MAX_EXTRA_CANDIDATES="${SHADOW_AUDIT_MAX_EXTRA_CANDIDATES:-2}"

SHADOW_SOURCE="${DEV_SHADOW_WOMD_SOURCE:-${WOMD_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}}"
python tools/audit_v48_32_shadow_provenance.py \
  --targets "$DEV_NEAR,$DEV_CONTACT" --womd-source "$SHADOW_SOURCE" \
  --expected-source-role validation --output "$ROOT/SHADOW_PROVENANCE_AUDIT.json"
export DEV_SHADOW_WOMD_SOURCE="$SHADOW_SOURCE"
export DEV_SHADOW_RAW_MAX_SCENARIOS="${DEV_SHADOW_RAW_MAX_SCENARIOS:-0}"

python - "$ROOT" "$DEV_NEAR" "$DEV_CONTACT" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=True)
(root/'DIAGNOSTIC_ONLY_NO_PAPER.json').write_text(json.dumps({
 'event':'v48_34_dev_shadow_closed_loop','created_unix':time.time(),
 'near_dataset':sys.argv[2],'contact_dataset':sys.argv[3],
 'uses_adaptation_dev_only':True,'uses_certificate_pool':False,
 'uses_test_or_stress':False,'paper_result':False,
 'purpose':'diagnose offline-label/closed-loop transfer after Natural-gate rejection'
},indent=2)+'\n')
PY

run_variant() {
  local variant="$1" gpu="$2"
  local base="$OUT/candidates/$variant"
  [[ -f "$base/model_v48_trac_sr/best.pt" ]] || { echo "missing $variant checkpoint" >&2; return 30; }
  [[ -d "$base/calibration" ]] || { echo "missing $variant calibration" >&2; return 30; }
  local run="$ROOT/$variant"
  mkdir -p "$run"
  # Run Near then Contact sequentially on one A30. Balanced and Precision run
  # concurrently on separate cards, so the diagnostic never oversubscribes VRAM.
  DEV_SHADOW_DIAGNOSTIC=1 BASE_RUN="$base" RUN="$run" OCRAP_ROOT="$OCRAP_ROOT" \
    DEV_SHADOW_WOMD_SOURCE="${DEV_SHADOW_WOMD_SOURCE:-}" DEV_SHADOW_RAW_MAX_SCENARIOS="${DEV_SHADOW_RAW_MAX_SCENARIOS:-0}" \
    NEAR_TEST="$DEV_NEAR" CONTACT_TEST="$DEV_CONTACT" BUCKET_SPLIT=evidence_adapt_dev \
    RUN_OFFLINE_EVAL=0 RUN_AUDITS=1 RUN_SAFE_CLOSED_LOOP=0 RUN_SCALAR_BASELINES=1 \
    RUN_NEAR_AUDIT=1 RUN_CONTACT_AUDIT=0 GPU_NEAR="$gpu" GPU_CONTACT="$gpu" \
    AUDIT_TARGETS="${AUDIT_TARGETS:-16}" AUDIT_LABELS="$SHADOW_AUDIT_LABELS" \
    AUDIT_LABEL_MODE="$SHADOW_LABEL_MODE" AUDIT_EVERY_N_STEPS="$SHADOW_AUDIT_EVERY_N_STEPS" \
    AUDIT_TOP_K="$SHADOW_AUDIT_TOP_K" AUDIT_MAX_EXTRA_CANDIDATES="$SHADOW_AUDIT_MAX_EXTRA_CANDIDATES" \
    AUDIT_MAX_ROLLOUTS="${AUDIT_MAX_ROLLOUTS:-8}" AUDIT_MAX_STEPS="${AUDIT_MAX_STEPS:-24}" \
    CL_RESUME="${CL_RESUME:-0}" bash scripts/run_ocrap_v48_trac_sr.sh \
    >"$run/near_controller.log" 2>&1
  DEV_SHADOW_DIAGNOSTIC=1 BASE_RUN="$base" RUN="$run" OCRAP_ROOT="$OCRAP_ROOT" \
    DEV_SHADOW_WOMD_SOURCE="${DEV_SHADOW_WOMD_SOURCE:-}" DEV_SHADOW_RAW_MAX_SCENARIOS="${DEV_SHADOW_RAW_MAX_SCENARIOS:-0}" \
    NEAR_TEST="$DEV_NEAR" CONTACT_TEST="$DEV_CONTACT" BUCKET_SPLIT=evidence_adapt_dev \
    RUN_OFFLINE_EVAL=0 RUN_AUDITS=1 RUN_SAFE_CLOSED_LOOP=0 RUN_SCALAR_BASELINES=1 \
    RUN_NEAR_AUDIT=0 RUN_CONTACT_AUDIT=1 GPU_NEAR="$gpu" GPU_CONTACT="$gpu" \
    AUDIT_TARGETS="${AUDIT_TARGETS:-16}" AUDIT_LABELS="$SHADOW_AUDIT_LABELS" \
    AUDIT_LABEL_MODE="$SHADOW_LABEL_MODE" AUDIT_EVERY_N_STEPS="$SHADOW_AUDIT_EVERY_N_STEPS" \
    AUDIT_TOP_K="$SHADOW_AUDIT_TOP_K" AUDIT_MAX_EXTRA_CANDIDATES="$SHADOW_AUDIT_MAX_EXTRA_CANDIDATES" \
    AUDIT_MAX_ROLLOUTS="${AUDIT_MAX_ROLLOUTS:-8}" AUDIT_MAX_STEPS="${AUDIT_MAX_STEPS:-24}" \
    CL_RESUME="${CL_RESUME:-0}" bash scripts/run_ocrap_v48_trac_sr.sh \
    >"$run/contact_controller.log" 2>&1
  python tools/compare_paired_closed_loop.py \
    "$run/audit_near_contact_selected_topk_v48_scalar.json" \
    "$run/audit_near_contact_selected_topk_v48_v48.json" \
    --output "$run/near_dev_shadow_paired.json" --bootstrap "${SHADOW_BOOTSTRAP:-2000}"
  python tools/compare_paired_closed_loop.py \
    "$run/audit_contact_selected_topk_v48_scalar.json" \
    "$run/audit_contact_selected_topk_v48_v48.json" \
    --output "$run/contact_dev_shadow_paired.json" --bootstrap "${SHADOW_BOOTSTRAP:-2000}"
  python tools/check_v48_32_shadow_runtime_contract.py \
    --near "$run/audit_near_contact_selected_topk_v48_v48.json" \
    --contact "$run/audit_contact_selected_topk_v48_v48.json" \
    --output "$run/SHADOW_RUNTIME_CONTRACT.json" --require-positive-gamma
  python tools/check_v48_32_physical_target_support.py \
    --near "$run/near_dev_shadow_paired.json" \
    --contact "$run/contact_dev_shadow_paired.json" \
    --output "$run/PHYSICAL_TARGET_SUPPORT.json"
  set +e
  python tools/check_v48_32_regime_targets.py "$run" \
    --near-paired "$run/near_dev_shadow_paired.json" \
    --contact-paired "$run/contact_dev_shadow_paired.json" \
    >"$run/dev_shadow_target_check.log" 2>&1
  local target_rc=$?
  set -e
  printf '%s\n' "$target_rc" > "$run/dev_shadow_target_check.rc"
  return 0
}

run_variant balanced "$GPU0" & p0=$!
run_variant precision "$GPU1" & p1=$!
set +e
wait "$p0"; r0=$?
wait "$p1"; r1=$?
set -e
python - "$ROOT" "$r0" "$r1" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); r0=int(sys.argv[2]); r1=int(sys.argv[3])
doc={'event':'v48_34_dev_shadow_complete','created_unix':time.time(),
     'balanced_exit':r0,'precision_exit':r1,'complete':r0==0 and r1==0,
     'uses_test_or_stress':False,'paper_result':False}
(root/'DEV_SHADOW_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
if not doc['complete']: raise SystemExit(30)
PY
