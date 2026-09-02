#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_35_continuous_frontier_dedicated_4835}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
ALLOW_PARTIAL_VARIANTS="${ALLOW_PARTIAL_VARIANTS:-0}"
RESUME_AFTER_ADAPTATION="${RESUME_AFTER_ADAPTATION:-0}"
PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-5}"
EVIDENCE_CONTEXT_SOURCE="${EVIDENCE_CALIBRATOR_CONTEXT_SOURCE:-physical_relative}"
ADMISSION_PRIOR_MODE="${EVIDENCE_ADMISSION_PRIOR_MODE:-frontier_capped_slack}"

ALPHA="${ALPHA:-0.2}"
BETA="${BETA:-0.2}"
TOP_M="${TOP_M:-8}"
POSITIVE_GAIN="${POSITIVE_GAIN:-0.015}"
DEPLOYABLE_MACRO_IDS="${DEPLOYABLE_MACRO_IDS:-2,3,5,6,7}"
COMPONENT_HARM_DRS_TOLERANCE="${COMPONENT_HARM_DRS_TOLERANCE:-0.05}"
COMPONENT_HARM_DEP_TOLERANCE="${COMPONENT_HARM_DEP_TOLERANCE:-0.05}"
COMPONENT_HARM_GAP_TOLERANCE="${COMPONENT_HARM_GAP_TOLERANCE:-0.05}"
COMPONENT_HARM_HARD_TOLERANCE="${COMPONENT_HARM_HARD_TOLERANCE:-0.05}"
COMPONENT_HARM_PROXY_TOLERANCE="${COMPONENT_HARM_PROXY_TOLERANCE:-0.05}"

mkdir -p "$OUTPUTDIR/logs"
ATTEMPT_ID="${V4835_ATTEMPT_ID:-$(python - <<'PY_ATTEMPT'
import time,uuid
print(f"v48352-{time.time_ns()}-{uuid.uuid4().hex[:12]}")
PY_ATTEMPT
)}"
export V4835_ATTEMPT_ID="$ATTEMPT_ID"
python - "$OUTPUTDIR/ATTEMPT_STARTED.json" "$ATTEMPT_ID" "$SOURCE_RUN" "$PROTOCOL_ROOT" "$RESUME_AFTER_ADAPTATION" <<'PY_ATTEMPT_STATUS'
import json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
doc={'event':'v48_35_attempt_started','version':'v48.35.2-ENGINEERING-INTEGRITY',
     'created_unix':time.time(),'attempt_id':sys.argv[2],'source_run':sys.argv[3],
     'protocol_root':sys.argv[4],'resume_after_adaptation':sys.argv[5]=='1','test_roots_read':False}
tmp=p.with_name(f'.{p.name}.tmp.{os.getpid()}.{time.time_ns()}')
with tmp.open('w',encoding='utf-8') as f:
    json.dump(doc,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp,p)
PY_ATTEMPT_STATUS
# A no-retraining resume must inspect the original failure state before any
# status cleanup. The authorization is narrow and validates checkpoint bytes,
# checkpoint config, source/protocol identity, and absence of certificate access.
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"
TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"
DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
GROUP_INDEX="$OUTPUTDIR/evidence_adapt_teacher_pcd_index.jsonl"
GROUP_SUMMARY="$OUTPUTDIR/evidence_adapt_teacher_pcd_index_summary.json"
VAL_GROUP_INDEX="$OUTPUTDIR/evidence_adapt_dev_teacher_pcd_index.jsonl"
VAL_GROUP_SUMMARY="$OUTPUTDIR/evidence_adapt_dev_teacher_pcd_index_summary.json"

write_pipeline_failure() {
  local stage="$1" raw_rc="$2" detail="${3:-}" balanced_rc="${4:-}" precision_rc="${5:-}"
  python - "$OUTPUTDIR" "$PROTOCOL_ROOT" "$SOURCE_RUN" "$stage" "$raw_rc" "$detail" "$balanced_rc" "$precision_rc" "$ATTEMPT_ID" <<'PY_FAILURE'
import hashlib,json,os,pathlib,shutil,sys,time
root=pathlib.Path(sys.argv[1]); protocol=pathlib.Path(sys.argv[2]); source=pathlib.Path(sys.argv[3])
stage=sys.argv[4]; raw_rc=int(sys.argv[5]); detail=sys.argv[6]; attempt_id=sys.argv[9]
def atomic(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.tmp.{os.getpid()}.{time.time_ns()}')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(doc,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def maybe_int(x):
    try: return int(x)
    except Exception: return None
balanced_rc=maybe_int(sys.argv[7]); precision_rc=maybe_int(sys.argv[8])
variants={}
for name in ('balanced','precision'):
    p=root/'candidates'/name/'model_v48_trac_sr'/'best.pt'
    if p.is_file():
        variants[name]={'checkpoint':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
next_status={}
try: next_status=json.load(open(root/'NEXT_COMMANDS_STATUS.json'))
except Exception: pass
certificate_executed=stage in {'certificate','post_certificate_diagnostics','completion_contract','terminal_state_contract'}
gate_evaluated=bool(next_status.get('gate_evaluated',False)) if certificate_executed else False
# A natural-gate marker cannot remain active after a later engineering failure.
gate_marker=root/'GATE_FAILED.json'
if gate_marker.exists():
    history=root/'status_history'/f'overridden-by-{attempt_id}-{time.time_ns()}'
    history.mkdir(parents=True,exist_ok=True)
    shutil.move(str(gate_marker),str(history/gate_marker.name))
try: (root/'NEXT_COMMANDS.txt').unlink()
except FileNotFoundError: pass
failed={'event':'v48_35_pipeline_failed','version':'v48.35.2-ENGINEERING-INTEGRITY','created_unix':time.time(),'attempt_id':attempt_id,'stage':stage,
        'raw_exit_code':raw_rc,'normalized_exit_code':30,'pipeline_exit_code':30,'detail':detail,
        'adaptation_exit_codes':{'balanced':balanced_rc,'precision':precision_rc},
        'certificate_executed':certificate_executed,'gate_evaluated':gate_evaluated,
        'pipeline_valid':False,'test_roots_read':False}
atomic(root/'PIPELINE_FAILED.json',failed)
doc={'event':'v48_35_continuous_frontier_controller_complete','version':'v48.35.2-ENGINEERING-INTEGRITY','created_unix':time.time(),'attempt_id':attempt_id,
     'source_run':str(source),'protocol_root':str(protocol),'variants':variants,
     'raw_certificate_exit_code':raw_rc if certificate_executed else None,
     'certificate_exit_code':30 if certificate_executed else None,'pipeline_exit_code':30,
     'certificate_executed':certificate_executed,'gate_evaluated':gate_evaluated,
     'gate_passed':False,'next_commands_generated':False,
     'pipeline_valid':False,'failure_stage':stage,'test_roots_read':False}
atomic(root/'V48_35_COMPLETE.json',doc)
blocked={'event':'v48_35_next_commands_blocked','created_unix':time.time(),
         'generated':False,'reason':'pipeline_failure','failure_stage':stage,
         'pipeline_exit_code':30,'certificate_executed':certificate_executed,
         'gate_evaluated':gate_evaluated,'test_roots_read':False}
blocked['attempt_id']=attempt_id
atomic(root/'NEXT_COMMANDS_BLOCKED.json',blocked)
atomic(root/'NEXT_COMMANDS_STATUS.json',blocked)
PY_FAILURE
  set +e
  python tools/audit_v48_35_run_state.py --run "$OUTPUTDIR" \
    --output "$OUTPUTDIR/AUTHORITATIVE_RUN_STATUS.json" \
    --expect-exit-code 30 --expect-attempt-id "$ATTEMPT_ID" \
    >"$OUTPUTDIR/logs/authoritative_failure_state.log" 2>&1
  local audit_rc=$?
  set -e
  if [[ "$audit_rc" != 0 ]]; then
    printf 'authoritative failure-state audit failed: rc=%s\n' "$audit_rc" \
      >"$OUTPUTDIR/TERMINAL_STATE_AUDIT_FAILED.txt"
  fi
}

if [[ "$RESUME_AFTER_ADAPTATION" == 1 ]]; then
  set +e
  python tools/check_v48_35_resume_contract.py \
    --run "$OUTPUTDIR" --output "$OUTPUTDIR/V48_35_RESUME_CONTRACT.json" \
    --expect-source-run "$SOURCE_RUN" --expect-protocol-root "$PROTOCOL_ROOT" \
    >"$OUTPUTDIR/logs/resume_contract.log" 2>&1
  resume_contract_rc=$?
  set -e
  if [[ "$resume_contract_rc" != 0 ]]; then
    # A refused resume is itself the terminal state of this new attempt. Preserve
    # the old terminal evidence before publishing an attempt-scoped RC=30.
    python - "$OUTPUTDIR" "$ATTEMPT_ID" <<'PY_ARCHIVE_REFUSED_RESUME'
import pathlib,shutil,sys,time
root=pathlib.Path(sys.argv[1]); attempt=sys.argv[2]
names=('PIPELINE_FAILED.json','V48_35_COMPLETE.json','NEXT_COMMANDS_STATUS.json',
       'NEXT_COMMANDS_BLOCKED.json','GATE_FAILED.json','CALIBRATION_FAILED.json',
       'AUTHORITATIVE_RUN_STATUS.json')
present=[root/name for name in names if (root/name).exists()]
if present:
    dst=root/'status_history'/f'resume-refused-{attempt}-{time.time_ns()}'
    dst.mkdir(parents=True,exist_ok=True)
    for src in present: shutil.move(str(src),str(dst/src.name))
PY_ARCHIVE_REFUSED_RESUME
    write_pipeline_failure "resume_authorization" "$resume_contract_rc" \
      "resume contract rejected; inspect V48_35_RESUME_CONTRACT.json"
    echo "v48.35.2 resume refused; inspect $OUTPUTDIR/V48_35_RESUME_CONTRACT.json" >&2
    exit 30
  fi
fi

# Preserve previous active status as history, then clear the active namespace.
# The current attempt is identified by ATTEMPT_ID; downstream readers never infer
# state from the mere presence of an older marker.
python - "$OUTPUTDIR" "$ATTEMPT_ID" <<'PY_ARCHIVE_STATUS'
import pathlib,shutil,sys,time
root=pathlib.Path(sys.argv[1]); attempt=sys.argv[2]
names=('PIPELINE_FAILED.json','V48_35_COMPLETE.json','NEXT_COMMANDS_STATUS.json',
       'NEXT_COMMANDS_BLOCKED.json','GATE_FAILED.json','CALIBRATION_FAILED.json',
       'AUTHORITATIVE_RUN_STATUS.json','GATE_FAILURE_DECOMPOSITION.json',
       'learning_gates_v48_35.json','GATE_SPEC.json','dedicated_recalibration_status.json')
present=[root/name for name in names if (root/name).exists()]
if present:
    dst=root/'status_history'/f'pre-{attempt}-{time.time_ns()}'
    dst.mkdir(parents=True,exist_ok=True)
    for src in present: shutil.move(str(src),str(dst/src.name))
PY_ARCHIVE_STATUS
rm -f "$OUTPUTDIR"/ADAPTATION_FAILED_*.json "$OUTPUTDIR"/FAILURE_SIGNATURE_*.json \
      "$OUTPUTDIR/NEXT_COMMANDS.txt" "$OUTPUTDIR/chosen_base_run_dedicated.txt"

set +e
python tools/audit_dedicated_protocol_v48_16.py \
  --protocol-root "$PROTOCOL_ROOT" \
  --output "$OUTPUTDIR/dedicated_protocol_audit.json" \
  >"$OUTPUTDIR/logs/dedicated_protocol_audit.log" 2>&1
protocol_rc=$?
set -e
if [[ "$protocol_rc" != 0 ]]; then
  write_pipeline_failure protocol_audit "$protocol_rc" "$OUTPUTDIR/logs/dedicated_protocol_audit.log"
  exit 30
fi


# Exercise the exact v48.34 eligible-set policy loss before any expensive
# index construction or GPU training.  The preflight requires finite gradients
# through admission, opportunity and harm heads on multiple scene-time groups.
set +e
python tools/check_v48_34_multigroup_loss_contract.py \
  --output "$OUTPUTDIR/MULTIGROUP_LOSS_CONTRACT.json" \
  >"$OUTPUTDIR/logs/multigroup_loss_contract.log" 2>&1
loss_contract_rc=$?
set -e
if [[ "$loss_contract_rc" != 0 ]]; then
  write_pipeline_failure loss_contract_preflight "$loss_contract_rc" "$OUTPUTDIR/MULTIGROUP_LOSS_CONTRACT.json"
  exit 30
fi
set +e
python tools/check_v48_35_frontier_contract.py \
  --output "$OUTPUTDIR/CONTINUOUS_FRONTIER_CONTRACT.json" \
  >"$OUTPUTDIR/logs/continuous_frontier_contract.log" 2>&1
frontier_contract_rc=$?
set -e
if [[ "$frontier_contract_rc" != 0 ]]; then
  write_pipeline_failure frontier_contract_preflight "$frontier_contract_rc" "$OUTPUTDIR/CONTINUOUS_FRONTIER_CONTRACT.json"
  exit 30
fi

index_contract_args=(
  --summary "$GROUP_SUMMARY"
  --expected-dataset "$TRAIN_NEAR,$TRAIN_CONTACT"
  --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M"
  --positive-gain "$POSITIVE_GAIN"
  --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS"
  --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE"
  --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE"
  --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE"
  --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE"
  --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE"
)
val_index_contract_args=(
  --summary "$VAL_GROUP_SUMMARY"
  --expected-dataset "$DEV_NEAR,$DEV_CONTACT"
  --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M"
  --positive-gain "$POSITIVE_GAIN"
  --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS"
  --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE"
  --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE"
  --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE"
  --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE"
  --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE"
)

rebuild_index="${REBUILD_ADAPT_INDEX:-0}"
if [[ "$rebuild_index" != 1 && -f "$GROUP_INDEX" && -f "$GROUP_SUMMARY" ]]; then
  set +e
  python tools/check_v48_19_target_support.py "${index_contract_args[@]}" \
    --mode contract --output "$OUTPUTDIR/SUPPORT_INDEX_CONTRACT.json" \
    >"$OUTPUTDIR/logs/check_teacher_index_contract.log" 2>&1
  contract_rc=$?
  set -e
  if [[ "$contract_rc" != 0 ]]; then
    if [[ "$RESUME_AFTER_ADAPTATION" == 1 ]]; then
      write_pipeline_failure resume_training_index_contract "$contract_rc" "$OUTPUTDIR/SUPPORT_INDEX_CONTRACT.json" 0 0
      exit 30
    fi
    echo "teacher-index contract changed; rebuilding the index" | tee -a "$OUTPUTDIR/logs/check_teacher_index_contract.log"
    rebuild_index=1
  fi
else
  if [[ "$RESUME_AFTER_ADAPTATION" == 1 ]]; then
    write_pipeline_failure resume_training_index_missing 30 "resume requires the byte-identical adaptation training index" 0 0
    exit 30
  fi
  rebuild_index=1
fi

if [[ "$rebuild_index" == 1 ]]; then
  rm -f "$GROUP_INDEX" "$GROUP_SUMMARY"
  set +e
  python tools/build_teacher_pcd_index_v48.py \
    --dataset "$TRAIN_NEAR,$TRAIN_CONTACT" \
    --output "$GROUP_INDEX" --summary-output "$GROUP_SUMMARY" \
    --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" \
    --positive-gain "$POSITIVE_GAIN" --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" \
    --quality-mode warn \
    --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE" \
    --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE" \
    --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE" \
    --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE" \
    --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE" \
    >"$OUTPUTDIR/logs/build_adapt_teacher_index.log" 2>&1
  index_rc=$?
  set -e
  if [[ "$index_rc" != 0 ]]; then
    write_pipeline_failure teacher_index_build "$index_rc" "$OUTPUTDIR/logs/build_adapt_teacher_index.log"
    exit 30
  fi
fi

# Re-audit after a rebuild so SUPPORT_INDEX_CONTRACT.json always describes the
# index that will actually be used, never the stale index that triggered it.
set +e
python tools/check_v48_19_target_support.py "${index_contract_args[@]}" \
  --mode contract --output "$OUTPUTDIR/SUPPORT_INDEX_CONTRACT.json" \
  >"$OUTPUTDIR/logs/check_teacher_index_contract.log" 2>&1
contract_rc=$?
set -e
if [[ "$contract_rc" != 0 ]]; then
  write_pipeline_failure teacher_index_contract "$contract_rc" "$OUTPUTDIR/SUPPORT_INDEX_CONTRACT.json"
  exit 30
fi

set +e
python tools/check_v48_19_target_support.py "${index_contract_args[@]}" \
  --mode all --output "$OUTPUTDIR/SUPPORT_TARGET_SUPPORT.json" \
  >"$OUTPUTDIR/logs/check_support_target_support.log" 2>&1
target_rc=$?
set -e
if [[ "$target_rc" != 0 ]]; then
  write_pipeline_failure target_support "$target_rc" "$OUTPUTDIR/SUPPORT_TARGET_SUPPORT.json"
  exit 30
fi

# Validation batching must use labels computed from adaptation-dev itself.
# Reuse is allowed only after the same exact dataset/label contract audit used
# for the training index; otherwise rebuild fail-closed.
rebuild_val_index="${REBUILD_ADAPT_DEV_INDEX:-0}"
if [[ "$rebuild_val_index" != 1 && -f "$VAL_GROUP_INDEX" && -f "$VAL_GROUP_SUMMARY" ]]; then
  set +e
  python tools/check_v48_19_target_support.py "${val_index_contract_args[@]}" \
    --mode contract --output "$OUTPUTDIR/VAL_SUPPORT_INDEX_CONTRACT.json" \
    >"$OUTPUTDIR/logs/check_dev_teacher_index_contract.log" 2>&1
  val_contract_rc=$?
  set -e
  if [[ "$val_contract_rc" != 0 ]]; then
    if [[ "$RESUME_AFTER_ADAPTATION" == 1 ]]; then
      write_pipeline_failure resume_validation_index_contract "$val_contract_rc" "$OUTPUTDIR/VAL_SUPPORT_INDEX_CONTRACT.json" 0 0
      exit 30
    fi
    rebuild_val_index=1
  fi
else
  if [[ "$RESUME_AFTER_ADAPTATION" == 1 ]]; then
    write_pipeline_failure resume_validation_index_missing 30 "resume requires the byte-identical adaptation-dev index" 0 0
    exit 30
  fi
  rebuild_val_index=1
fi
if [[ "$rebuild_val_index" == 1 ]]; then
  rm -f "$VAL_GROUP_INDEX" "$VAL_GROUP_SUMMARY"
  set +e
  python tools/build_teacher_pcd_index_v48.py \
    --dataset "$DEV_NEAR,$DEV_CONTACT" \
    --output "$VAL_GROUP_INDEX" --summary-output "$VAL_GROUP_SUMMARY" \
    --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" \
    --positive-gain "$POSITIVE_GAIN" --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" \
    --quality-mode warn \
    --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE" \
    --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE" \
    --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE" \
    --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE" \
    --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE" \
    >"$OUTPUTDIR/logs/build_adapt_dev_teacher_index.log" 2>&1
  val_index_rc=$?
  set -e
  if [[ "$val_index_rc" != 0 ]]; then
    write_pipeline_failure validation_teacher_index_build "$val_index_rc" "$OUTPUTDIR/logs/build_adapt_dev_teacher_index.log"
    exit 30
  fi
fi
set +e
python tools/check_v48_19_target_support.py "${val_index_contract_args[@]}" \
  --mode contract --output "$OUTPUTDIR/VAL_SUPPORT_INDEX_CONTRACT.json" \
  >"$OUTPUTDIR/logs/check_dev_teacher_index_contract.log" 2>&1
val_contract_rc=$?
set -e
if [[ "$val_contract_rc" != 0 ]]; then
  write_pipeline_failure validation_teacher_index_contract "$val_contract_rc" "$OUTPUTDIR/VAL_SUPPORT_INDEX_CONTRACT.json"
  exit 30
fi

run_variant() {
  local variant="$1" gpu="$2"
  local source="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  local run="$OUTPUTDIR/candidates/$variant"
  local factor_cache=""
  case "$variant" in
    balanced) factor_cache="${V4835_FACTOR_CACHE_BALANCED:-}" ;;
    precision) factor_cache="${V4835_FACTOR_CACHE_PRECISION:-}" ;;
  esac
  [[ -f "$source" ]] || { echo "missing source checkpoint $source" >&2; return 30; }
  if [[ -n "$factor_cache" && ! -d "$factor_cache" ]]; then
    echo "configured factor cache does not exist: $factor_cache" >&2
    return 30
  fi
  set +e
  RUN="$run" INIT_CKPT="$source" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="$VAL_GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
  NUM_WORKERS="${NUM_WORKERS:-3}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-3}" BATCH_SIZE="${BATCH_SIZE:-96}" \
  FACTOR_EPOCHS="${FACTOR_EPOCHS:-20}" FACTOR_PATIENCE="${FACTOR_PATIENCE:-6}" \
  IDENTITY_EPOCHS="${IDENTITY_EPOCHS:-24}" IDENTITY_PATIENCE="${IDENTITY_PATIENCE:-6}" \
  FINAL_EPOCHS="${FINAL_EPOCHS:-8}" FINAL_PATIENCE="${FINAL_PATIENCE:-3}" \
  IDENTITY_LR="${IDENTITY_LR:-0.00004}" FINAL_LR="${FINAL_LR:-0.00003}" \
  V4835_IDENTITY_TRAIN_ALL=1 V4835_COUPLE_ADMISSION_PRIOR=1 \
  V4835_ADAPTIVE_IDENTITY_MARGIN=0 V4835_ENABLE_FINAL_CALIBRATION=0 \
  V4835_FACTOR_CACHE_RUN="$factor_cache" \
  PROPOSAL_TOP_K="$PROPOSAL_TOP_K" \
  EVIDENCE_CALIBRATOR_CONTEXT=true EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="$EVIDENCE_CONTEXT_SOURCE" \
  EVIDENCE_COMPONENT_COUNT=5 EVIDENCE_COMPONENT_SCALE="${EVIDENCE_COMPONENT_SCALE:-6.0}" \
  EVIDENCE_ADMISSION_PRIOR_MODE="$ADMISSION_PRIOR_MODE" EVIDENCE_ADMISSION_SCALE="${EVIDENCE_ADMISSION_SCALE:-2.0}" \
  EVIDENCE_SLACK_TEMPERATURE="${EVIDENCE_SLACK_TEMPERATURE:-0.025}" EVIDENCE_SLACK_PENALTY="${EVIDENCE_SLACK_PENALTY:-1.0}" EVIDENCE_FRONTIER_CAP_TEMPERATURE="${EVIDENCE_FRONTIER_CAP_TEMPERATURE:-0.10}" \
  IDENTITY_ELIGIBILITY_BOUNDARY_WEIGHT="${IDENTITY_ELIGIBILITY_BOUNDARY_WEIGHT:-1.0}" \
  IDENTITY_ELIGIBILITY_BOUNDARY_MARGIN="${IDENTITY_ELIGIBILITY_BOUNDARY_MARGIN:-0.20}" \
  IDENTITY_POSITIVE_MACRO_BALANCE_POWER="${IDENTITY_POSITIVE_MACRO_BALANCE_POWER:-0.50}" \
  IDENTITY_SCENE_BALANCE_POWER="${IDENTITY_SCENE_BALANCE_POWER:-0.50}" \
  bash scripts/adapt_ocrap_v48_35_continuous_frontier_variant.sh >"$OUTPUTDIR/logs/adapt_${variant}.log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" != 0 ]]; then
    local signature="$OUTPUTDIR/FAILURE_SIGNATURE_${variant}.json"
    local stage="unknown"
    if [[ -f "$run/VARIANT_STAGE_FAILED.json" ]]; then
      stage="$(python - "$run/VARIANT_STAGE_FAILED.json" <<'PY_STAGE'
import json,sys
try: print(json.load(open(sys.argv[1])).get('stage','unknown'))
except Exception: print('unknown')
PY_STAGE
)"
    fi
    set +e
    python tools/extract_v48_34_failure_signature.py \
      --log "$OUTPUTDIR/logs/adapt_${variant}.log" --output "$signature" \
      --stage "$stage" --exit-code "$rc" >"$OUTPUTDIR/logs/failure_signature_${variant}.log" 2>&1
    local signature_rc=$?
    set -e
    python - "$OUTPUTDIR" "$variant" "$rc" "$OUTPUTDIR/logs/adapt_${variant}.log" "$stage" "$signature" "$signature_rc" "$ATTEMPT_ID" <<'PY_ADAPT_FAIL'
import json,os,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); variant=sys.argv[2]; rc=int(sys.argv[3]); log=pathlib.Path(sys.argv[4]); stage=sys.argv[5]; signature=pathlib.Path(sys.argv[6])
signature_rc=int(sys.argv[7]); attempt_id=sys.argv[8]
tail='\n'.join(log.read_text(errors='replace').splitlines()[-100:]) if log.exists() else ''
sig={}
if signature_rc == 0:
    try: sig=json.load(open(signature))
    except Exception as exc: sig={'read_error':repr(exc)}
doc={
    'event':'v48_35_adaptation_failed','version':'v48.35.2-ENGINEERING-INTEGRITY',
    'attempt_id':attempt_id,'variant':variant,'stage':stage,
    'exit_code':rc,'log':str(log),'failure_signature':sig,
    'failure_signature_exit_code':signature_rc,
    'log_tail':tail,'created_unix':time.time(),'test_roots_read':False}
out=root/f'ADAPTATION_FAILED_{variant}.json'; tmp=out.with_name(f'.{out.name}.tmp.{os.getpid()}.{time.time_ns()}')
with tmp.open('w',encoding='utf-8') as f:
    json.dump(doc,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY_ADAPT_FAIL
  else
    rm -f "$OUTPUTDIR/ADAPTATION_FAILED_${variant}.json"
  fi
  return "$rc"
}

if [[ "$RESUME_AFTER_ADAPTATION" == 1 ]]; then
  s0=0; s1=0
  printf 'balanced=0 precision=0 resume_after_adaptation=1 retraining=0\n' | tee "$OUTPUTDIR/logs/adaptation_status.log"
else
  run_variant balanced "$GPU0" & p0=$!
  run_variant precision "$GPU1" & p1=$!
  set +e
  wait "$p0"; s0=$?
  wait "$p1"; s1=$?
  set -e
  printf 'balanced=%s precision=%s resume_after_adaptation=0 retraining=1\n' "$s0" "$s1" | tee "$OUTPUTDIR/logs/adaptation_status.log"
fi

if [[ "$s0" != 0 && "$s1" != 0 ]]; then
  write_pipeline_failure adaptation 30 "both variants failed; inspect FAILURE_SIGNATURE_balanced.json and FAILURE_SIGNATURE_precision.json" "$s0" "$s1"
  exit 30
fi
if [[ "$ALLOW_PARTIAL_VARIANTS" != 1 && ( "$s0" != 0 || "$s1" != 0 ) ]]; then
  write_pipeline_failure adaptation 30 "one variant failed; set ALLOW_PARTIAL_VARIANTS=1 only for explicit debugging" "$s0" "$s1"
  exit 30
fi

# v48.34 fail-closed preflight: training and downstream inference must
# construct the same frontier/prior/admission model.
for variant in balanced precision; do
  [[ "$variant" == balanced && "$s0" != 0 ]] && continue
  [[ "$variant" == precision && "$s1" != 0 ]] && continue
  ckpt="$OUTPUTDIR/candidates/$variant/model_v48_trac_sr/best.pt"
  set +e
  python tools/check_v48_35_model_contract.py \
    --checkpoint "$ckpt" \
    --support-contract "$OUTPUTDIR/candidates/$variant/FACTOR_SUPPORT_CONTRACT.json" \
    --output "$OUTPUTDIR/candidates/$variant/MODEL_INFERENCE_CONTRACT.json" \
    --expect-frontier true --expect-admission-bounded false --expect-context-enabled true --expect-context-source "$EVIDENCE_CONTEXT_SOURCE" --expect-frontier-cap-temperature "${EVIDENCE_FRONTIER_CAP_TEMPERATURE:-0.10}" \
    --expect-component-prior-logit -2.0 --expect-component-count 5 --expect-component-scale "${EVIDENCE_COMPONENT_SCALE:-6.0}" \
    --expect-admission-prior-detach any \
    --expect-admission-prior-mode "$ADMISSION_PRIOR_MODE" --expect-slack-temperature "${EVIDENCE_SLACK_TEMPERATURE:-0.025}" --expect-slack-penalty "${EVIDENCE_SLACK_PENALTY:-1.0}" \
    >"$OUTPUTDIR/logs/model_contract_${variant}.log" 2>&1
  contract_model_rc=$?
  set -e
  if [[ "$contract_model_rc" != 0 ]]; then
    write_pipeline_failure model_inference_contract "$contract_model_rc" "$OUTPUTDIR/candidates/$variant/MODEL_INFERENCE_CONTRACT.json" "$s0" "$s1"
    exit 30
  fi
  set +e
  python tools/check_v48_35_training_contract.py \
    --run "$OUTPUTDIR/candidates/$variant" \
    --output "$OUTPUTDIR/candidates/$variant/TRAINING_CONTRACT.json" \
    --expect-identity-all true --expect-prior-coupled true \
    --expect-adaptive-margin false --expect-final-enabled false \
    --expect-eligible-policy true --expect-prior-mode "$ADMISSION_PRIOR_MODE" --expect-context-source "$EVIDENCE_CONTEXT_SOURCE" --expect-proposal-top-k "$PROPOSAL_TOP_K" \
    >"$OUTPUTDIR/logs/training_contract_${variant}.log" 2>&1
  training_contract_rc=$?
  set -e
  if [[ "$training_contract_rc" != 0 ]]; then
    write_pipeline_failure training_contract "$training_contract_rc" "$OUTPUTDIR/candidates/$variant/TRAINING_CONTRACT.json" "$s0" "$s1"
    exit 30
  fi
done

variants=""
[[ "$s0" == 0 ]] && variants="balanced"
[[ "$s1" == 0 ]] && variants="${variants:+$variants,}precision"
set +e
OUTPUTDIR="$OUTPUTDIR" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
GPU0="$GPU0" GPU1="$GPU1" VARIANTS="$variants" PROPOSAL_TOP_K="$PROPOSAL_TOP_K" \
  OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit bash scripts/calibrate_v48_35_shared_certificate_pool.sh >"$OUTPUTDIR/logs/certificate_controller.log" 2>&1
raw_cert_rc=$?
set -e
case "$raw_cert_rc" in
  0|20) cert_rc="$raw_cert_rc" ;;
  *) cert_rc=30 ;;
esac

if [[ "$cert_rc" == 30 ]]; then
  write_pipeline_failure certificate "$raw_cert_rc" "$OUTPUTDIR/logs/certificate_controller.log" "$s0" "$s1"
  exit 30
fi

set +e
python tools/check_v48_16_learning_gates.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/learning_gates_v48_35.json" --version v48.35.2-ENGINEERING-INTEGRITY \
  >"$OUTPUTDIR/logs/learning_gates.log" 2>&1
learning_gates_rc=$?
python tools/summarize_v48_34_gate_failure.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/GATE_FAILURE_DECOMPOSITION.json" \
  >"$OUTPUTDIR/logs/gate_failure_decomposition.log" 2>&1
gate_decomposition_rc=$?
set -e
if [[ "$learning_gates_rc" != 0 || "$gate_decomposition_rc" != 0 ]]; then
  write_pipeline_failure post_certificate_diagnostics 4 "learning_gates_rc=$learning_gates_rc gate_decomposition_rc=$gate_decomposition_rc" "$s0" "$s1"
  exit 30
fi

set +e
python - "$OUTPUTDIR" "$PROTOCOL_ROOT" "$SOURCE_RUN" "$raw_cert_rc" "$cert_rc" "$RESUME_AFTER_ADAPTATION" "$ATTEMPT_ID" <<'PY'
import hashlib,json,os,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); protocol=pathlib.Path(sys.argv[2]); source=pathlib.Path(sys.argv[3])
raw_rc=int(sys.argv[4]); rc=int(sys.argv[5]); resumed=sys.argv[6] == '1'; attempt_id=sys.argv[7]; variants={}
for name in ('balanced','precision'):
    p=root/'candidates'/name/'model_v48_trac_sr'/'best.pt'
    if p.is_file(): variants[name]={'checkpoint':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
next_exists=(root/'NEXT_COMMANDS.txt').is_file()
blocked_exists=(root/'NEXT_COMMANDS_BLOCKED.json').is_file()
consistent=(rc==0 and next_exists and not blocked_exists) or (rc==20 and (not next_exists) and blocked_exists)
if not consistent:
    raise SystemExit(f'certificate/NEXT_COMMANDS contract mismatch: rc={rc} next={next_exists} blocked={blocked_exists}')
doc={'event':'v48_35_continuous_frontier_controller_complete','version':'v48.35.2-ENGINEERING-INTEGRITY','created_unix':time.time(),'attempt_id':attempt_id,
     'source_run':str(source),'protocol_root':str(protocol),'variants':variants,
     'raw_certificate_exit_code':raw_rc,'certificate_exit_code':rc,'pipeline_exit_code':rc,
     'certificate_executed':True,'gate_evaluated':True,'gate_passed':(rc==0 and next_exists),
     'next_commands_generated':next_exists,'pipeline_valid':True,
     'adaptation_reused_without_retraining':resumed,'resume_contract':str(root/'V48_35_RESUME_CONTRACT.json') if resumed else None,
     'test_roots_read':False}
tmp=root/f'.V48_35_COMPLETE.json.tmp.{os.getpid()}.{time.time_ns()}'
with tmp.open('w',encoding='utf-8') as f:
    json.dump(doc,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp,root/'V48_35_COMPLETE.json')
PY
completion_rc=$?
set -e
if [[ "$completion_rc" != 0 ]]; then
  write_pipeline_failure completion_contract 30 "RC/NEXT_COMMANDS mismatch" "$s0" "$s1"
  exit 30
fi
set +e
python tools/audit_v48_35_run_state.py --run "$OUTPUTDIR" \
  --output "$OUTPUTDIR/AUTHORITATIVE_RUN_STATUS.json" \
  --expect-exit-code "$cert_rc" --expect-attempt-id "$ATTEMPT_ID" --archive-stale-markers \
  >"$OUTPUTDIR/logs/authoritative_run_state.log" 2>&1
state_contract_rc=$?
set -e
if [[ "$state_contract_rc" != 0 ]]; then
  write_pipeline_failure terminal_state_contract "$state_contract_rc" "$OUTPUTDIR/AUTHORITATIVE_RUN_STATUS.json" "$s0" "$s1"
  exit 30
fi
exit "$cert_rc"
