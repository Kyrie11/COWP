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

OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_36_ocaf_dedicated_4836}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
ALLOW_PARTIAL_VARIANTS="${ALLOW_PARTIAL_VARIANTS:-0}"
RESUME_AFTER_ADAPTATION="${RESUME_AFTER_ADAPTATION:-0}"
PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-5}"
EVIDENCE_CONTEXT_SOURCE="${EVIDENCE_CALIBRATOR_CONTEXT_SOURCE:-physical_interaction}"
ADMISSION_PRIOR_MODE="${EVIDENCE_ADMISSION_PRIOR_MODE:-frontier_capped_slack}"
EVIDENCE_INTERACTION_HIDDEN="${EVIDENCE_INTERACTION_HIDDEN:-64}"
EVIDENCE_INTERACTION_DROPOUT="${EVIDENCE_INTERACTION_DROPOUT:-0.05}"
EVIDENCE_CONSENSUS_PRIOR_SCALE="${EVIDENCE_CONSENSUS_PRIOR_SCALE:-0.50}"
OPTION_EXECUTION_SEMANTICS="${OPTION_EXECUTION_SEMANTICS:-global}"
# v48.46 can deliberately vary training supervision while keeping the
# paper-consistent evaluation/certificate definition fixed across every arm.
TRAIN_OPTION_EXECUTION_SEMANTICS="${TRAIN_OPTION_EXECUTION_SEMANTICS:-$OPTION_EXECUTION_SEMANTICS}"
EVAL_OPTION_EXECUTION_SEMANTICS="${EVAL_OPTION_EXECUTION_SEMANTICS:-$OPTION_EXECUTION_SEMANTICS}"
SERIAL_VARIANTS_ON_ONE_GPU="${SERIAL_VARIANTS_ON_ONE_GPU:-0}"
export OPTION_EXECUTION_SEMANTICS TRAIN_OPTION_EXECUTION_SEMANTICS EVAL_OPTION_EXECUTION_SEMANTICS
IMPLEMENTATION_VERSION="${OCRAP_IMPLEMENTATION_VERSION:-v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX}"
export OCRAP_IMPLEMENTATION_VERSION="$IMPLEMENTATION_VERSION"

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
# Independent teacher rows are deterministic.  A small process pool removes
# Python/ZIP decompression serialization while preserving executor.map order.
V4856_TEACHER_INDEX_WORKERS="${V4856_TEACHER_INDEX_WORKERS:-4}"
V4856_TEACHER_INDEX_CHUNKSIZE="${V4856_TEACHER_INDEX_CHUNKSIZE:-16}"

mkdir -p "$OUTPUTDIR/logs"
# v48.56 engineering-only stage timing.  This is append-only diagnostic output
# and is never consumed by training, calibration or certificate decisions.
V4856_RUNTIME_TIMING_LOG="${V4856_RUNTIME_TIMING_LOG:-$OUTPUTDIR/logs/runtime_stage_timing.jsonl}"
: > "$V4856_RUNTIME_TIMING_LOG"
v4856_timing_event() {
  local event="$1" stage="$2" start="${3:-}" rc="${4:-}"
  local now duration_json rc_json
  now="$(date +%s.%N)"
  if [[ -n "$start" ]]; then duration_json="$(awk -v a="$start" -v b="$now" 'BEGIN{printf "%.6f", b-a}')"; else duration_json="null"; fi
  if [[ -n "$rc" ]]; then rc_json="$rc"; else rc_json="null"; fi
  printf '{"unix":%s,"event":"%s","stage":"%s","duration_seconds":%s,"rc":%s}\n' \
    "$now" "$event" "$stage" "$duration_json" "$rc_json" >> "$V4856_RUNTIME_TIMING_LOG"
}
V4856_PIPELINE_T0="$(date +%s.%N)"; v4856_timing_event start pipeline "$V4856_PIPELINE_T0"
# Re-entry is audited before ATTEMPT_STARTED.json or any active terminal marker is
# changed.  Repeated invocations therefore return an existing active RC=0/20 result, and
# an exact archived RC=20 resume-refusal clobber is restored safely.
REENTRY_MODE=fresh
[[ "$RESUME_AFTER_ADAPTATION" == 1 ]] && REENTRY_MODE=resume
reentry_args=(--run "$OUTPUTDIR" --mode "$REENTRY_MODE" --output "$OUTPUTDIR/V48_36_REENTRY_CONTRACT.json")
[[ "${ALLOW_COMPLETED_RUN_OVERWRITE:-0}" == 1 ]] && reentry_args+=(--allow-completed-overwrite)
set +e
python tools/check_v48_36_reentry_contract.py "${reentry_args[@]}"   >"$OUTPUTDIR/logs/reentry_contract.log" 2>&1
reentry_rc=$?
set -e
if [[ "$reentry_rc" != 0 ]]; then
  echo "v48.36 re-entry contract failed; existing terminal state was not modified" >&2
  exit 30
fi
read -r reentry_action reentry_exit_code < <(python - "$OUTPUTDIR/V48_36_REENTRY_CONTRACT.json" <<'PY_REENTRY_READ'
import json,sys
x=json.load(open(sys.argv[1]))
print(x.get('action','refuse'), x.get('existing_exit_code',''))
PY_REENTRY_READ
)
case "$reentry_action" in
  return_existing_terminal)
    echo "v48.36 output already has a valid authoritative terminal state; returning RC=${reentry_exit_code} without mutation"
    exit "$reentry_exit_code"
    ;;
  restore_archived_terminal)
    set +e
    python tools/restore_v48_36_terminal_state_after_refused_resume.py       --run "$OUTPUTDIR" --repo "$REPO" --output "$OUTPUTDIR/V48_36_4_REENTRY_RESTORE.json"       >"$OUTPUTDIR/logs/reentry_restore.log" 2>&1
    restore_rc=$?
    set -e
    if [[ "$restore_rc" != 0 ]]; then
      echo "v48.36 archived terminal restore failed; repair rolled back" >&2
      exit 30
    fi
    restored_exit_code="$(python - "$OUTPUTDIR/V48_36_4_REENTRY_RESTORE.json" <<'PY_RESTORED_RC'
import json,sys
print(int(json.load(open(sys.argv[1]))['authoritative_exit_code']))
PY_RESTORED_RC
)"
    echo "v48.36 restored the previous authoritative terminal state; returning RC=${restored_exit_code}"
    exit "$restored_exit_code"
    ;;
  refuse_preserve_current|refuse)
    echo "v48.36 re-entry refused; current active state was preserved" >&2
    exit 30
    ;;
  proceed) ;;
  *)
    echo "unknown v48.36 re-entry action: $reentry_action" >&2
    exit 30
    ;;
esac

# A no-retraining resume must inspect the original failure state before any
# attempt creation or status cleanup. The authorization is narrow and validates
# checkpoint bytes, checkpoint config, source/protocol identity, and absence of
# certificate access.
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
certificate_executed=stage in {'certificate','post_certificate_diagnostics','certificate_status_contract','completion_contract','terminal_state_contract'}
gate_evaluated=bool(next_status.get('gate_evaluated',False)) if certificate_executed else False
# Preserve the pre-failure terminal evidence before publishing a later engineering
# failure.  This makes a terminal-state contract defect diagnosable without relying
# on a log tail after V48_36_COMPLETE.json is replaced.
gate_marker=root/'GATE_FAILED.json'
history=None
if stage in {'post_certificate_diagnostics','certificate_status_contract','completion_contract','terminal_state_contract'}:
    history=root/'status_history'/f'pre-{stage}-{attempt_id}-{time.time_ns()}'
    history.mkdir(parents=True,exist_ok=True)
    for name in ('V48_36_COMPLETE.json','AUTHORITATIVE_RUN_STATUS.json',
                 'NEXT_COMMANDS_STATUS.json','NEXT_COMMANDS_BLOCKED.json',
                 'GATE_SPEC.json','dedicated_recalibration_status.json',
                 'GATE_FAILURE_DECOMPOSITION.json','learning_gates_v48_36.json'):
        source_path=root/name
        if source_path.is_file():
            shutil.copy2(source_path,history/name)
if gate_marker.exists():
    if history is None:
        history=root/'status_history'/f'overridden-by-{attempt_id}-{time.time_ns()}'
        history.mkdir(parents=True,exist_ok=True)
    shutil.move(str(gate_marker),str(history/gate_marker.name))
try: (root/'NEXT_COMMANDS.txt').unlink()
except FileNotFoundError: pass
failed={'event':'v48_36_pipeline_failed','version':'v48.36-OCAF','implementation_version':os.environ.get('OCRAP_IMPLEMENTATION_VERSION','v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX'),'created_unix':time.time(),'attempt_id':attempt_id,'stage':stage,
        'raw_exit_code':raw_rc,'normalized_exit_code':30,'pipeline_exit_code':30,'detail':detail,
        'adaptation_exit_codes':{'balanced':balanced_rc,'precision':precision_rc},
        'certificate_executed':certificate_executed,'gate_evaluated':gate_evaluated,
        'pipeline_valid':False,'test_roots_read':False}
atomic(root/'PIPELINE_FAILED.json',failed)
doc={'event':'v48_36_ocaf_controller_complete','version':'v48.36-OCAF','implementation_version':os.environ.get('OCRAP_IMPLEMENTATION_VERSION','v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX'),'created_unix':time.time(),'attempt_id':attempt_id,
     'source_run':str(source),'protocol_root':str(protocol),'variants':variants,
     'raw_certificate_exit_code':raw_rc if certificate_executed else None,
     'certificate_exit_code':30 if certificate_executed else None,'pipeline_exit_code':30,
     'certificate_executed':certificate_executed,'gate_evaluated':gate_evaluated,
     'gate_passed':False,'next_commands_generated':False,
     'pipeline_valid':False,'failure_stage':stage,'test_roots_read':False}
atomic(root/'V48_36_COMPLETE.json',doc)
blocked={'event':'v48_36_next_commands_blocked','created_unix':time.time(),
         'generated':False,'reason':'pipeline_failure','failure_stage':stage,
         'pipeline_exit_code':30,'certificate_executed':certificate_executed,
         'gate_evaluated':gate_evaluated,'test_roots_read':False}
blocked['attempt_id']=attempt_id
atomic(root/'NEXT_COMMANDS_BLOCKED.json',blocked)
atomic(root/'NEXT_COMMANDS_STATUS.json',blocked)
PY_FAILURE
  set +e
  python tools/resolve_v48_36_authoritative_result.py --run "$OUTPUTDIR" \
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
  python tools/check_v48_36_resume_contract.py \
    --run "$OUTPUTDIR" --output "$OUTPUTDIR/V48_36_RESUME_CONTRACT.json" \
    --expect-source-run "$SOURCE_RUN" --expect-protocol-root "$PROTOCOL_ROOT" \
    >"$OUTPUTDIR/logs/resume_contract.log" 2>&1
  resume_contract_rc=$?
  set -e
  if [[ "$resume_contract_rc" != 0 ]]; then
    # Resume authorization is a pre-attempt decision.  Do not create a new active
    # RC=30 or move/overwrite any existing terminal state merely because reuse was
    # refused.  The sidecar records the command failure for operator diagnosis.
    python - "$OUTPUTDIR/V48_36_RESUME_REFUSED.json" "$resume_contract_rc" <<'PY_RESUME_REFUSED'
import json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); rc=int(sys.argv[2])
doc={'event':'v48_36_resume_refused','version':'v48.36-OCAF',
     'implementation_version':os.environ.get('OCRAP_IMPLEMENTATION_VERSION','v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX'),
     'created_unix':time.time(),'resume_contract_exit_code':rc,
     'active_terminal_state_preserved':True,'attempt_created':False,
     'algorithm_changed':False,'test_roots_read':False}
tmp=p.with_name(f'.{p.name}.tmp.{os.getpid()}.{time.time_ns()}')
with tmp.open('w',encoding='utf-8') as f:
    json.dump(doc,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp,p)
PY_RESUME_REFUSED
    echo "v48.36 resume refused before attempt creation; active terminal state was preserved" >&2
    exit 30
  fi
fi

ATTEMPT_ID="${V4836_ATTEMPT_ID:-$(python - <<'PY_ATTEMPT'
import time,uuid
print(f"v4836-{time.time_ns()}-{uuid.uuid4().hex[:12]}")
PY_ATTEMPT
)}"
if [[ -z "$ATTEMPT_ID" || "$ATTEMPT_ID" == "legacy-untracked" ]]; then
  echo "invalid V4836_ATTEMPT_ID: attempt-scoped non-legacy ID required" >&2
  exit 30
fi
export V4836_ATTEMPT_ID="$ATTEMPT_ID"
python - "$OUTPUTDIR/ATTEMPT_STARTED.json" "$ATTEMPT_ID" "$SOURCE_RUN" "$PROTOCOL_ROOT" "$RESUME_AFTER_ADAPTATION" <<'PY_ATTEMPT_STATUS'
import json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
doc={'event':'v48_36_attempt_started','version':'v48.36-OCAF','implementation_version':os.environ.get('OCRAP_IMPLEMENTATION_VERSION','v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX'),
     'created_unix':time.time(),'attempt_id':sys.argv[2],'source_run':sys.argv[3],
     'protocol_root':sys.argv[4],'resume_after_adaptation':sys.argv[5]=='1',
     'protocol_seal_sha256':os.environ.get('V4845_PROTOCOL_SEAL_SHA256'),'test_roots_read':False}
tmp=p.with_name(f'.{p.name}.tmp.{os.getpid()}.{time.time_ns()}')
with tmp.open('w',encoding='utf-8') as f:
    json.dump(doc,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp,p)
PY_ATTEMPT_STATUS

# Preserve previous active status as history, then clear the active namespace.
# The current attempt is identified by ATTEMPT_ID; downstream readers never infer
# state from the mere presence of an older marker.
python - "$OUTPUTDIR" "$ATTEMPT_ID" <<'PY_ARCHIVE_STATUS'
import pathlib,shutil,sys,time
root=pathlib.Path(sys.argv[1]); attempt=sys.argv[2]
names=('PIPELINE_FAILED.json','V48_36_COMPLETE.json','NEXT_COMMANDS_STATUS.json',
       'NEXT_COMMANDS_BLOCKED.json','GATE_FAILED.json','CALIBRATION_FAILED.json',
       'AUTHORITATIVE_RUN_STATUS.json','GATE_FAILURE_DECOMPOSITION.json',
       'learning_gates_v48_36.json','GATE_SPEC.json','dedicated_recalibration_status.json')
present=[root/name for name in names if (root/name).exists()]
if present:
    dst=root/'status_history'/f'pre-{attempt}-{time.time_ns()}'
    dst.mkdir(parents=True,exist_ok=True)
    for src in present: shutil.move(str(src),str(dst/src.name))
PY_ARCHIVE_STATUS
rm -f "$OUTPUTDIR"/ADAPTATION_FAILED_*.json "$OUTPUTDIR"/FAILURE_SIGNATURE_*.json \
      "$OUTPUTDIR/NEXT_COMMANDS.txt" "$OUTPUTDIR/chosen_base_run_dedicated.txt"

# Fail fast on the immutable source checkpoints before spending time rebuilding
# teacher indexes or starting GPU jobs.  Older code checked these only inside
# each background variant, which could collapse two missing-source errors into
# the misleading message "both variants failed" and leave no in-run log.
set +e
python tools/check_v48_36_source_checkpoint_contract.py \
  --source-run "$SOURCE_RUN" --output "$OUTPUTDIR/SOURCE_CHECKPOINT_CONTRACT.json" \
  >"$OUTPUTDIR/logs/source_checkpoint_contract.log" 2>&1
source_checkpoint_rc=$?
set -e
if [[ "$source_checkpoint_rc" != 0 ]]; then
  write_pipeline_failure source_checkpoint_contract "$source_checkpoint_rc" "$OUTPUTDIR/SOURCE_CHECKPOINT_CONTRACT.json"
  exit 30
fi

# Fail closed on the exact canonical dataset roots. Legacy aliases (for example
# traincontact versus train_contact) must never silently change the experiment.
set +e
python tools/check_v48_36_dataset_root_contract.py \
  --protocol-root "$PROTOCOL_ROOT" --safe-root "$CAL_SAFE" \
  --train-near "$TRAIN_NEAR" --train-contact "$TRAIN_CONTACT" \
  --dev-near "$DEV_NEAR" --dev-contact "$DEV_CONTACT" \
  --cert-near "$CERT_NEAR" --cert-contact "$CERT_CONTACT" \
  --output "$OUTPUTDIR/DATASET_ROOT_CONTRACT.json" \
  >"$OUTPUTDIR/logs/dataset_root_contract.log" 2>&1
dataset_root_rc=$?
set -e
if [[ "$dataset_root_rc" != 0 ]]; then
  write_pipeline_failure dataset_root_contract "$dataset_root_rc" "$OUTPUTDIR/DATASET_ROOT_CONTRACT.json"
  exit 30
fi


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
python tools/check_v48_36_ocaf_bridge_contract.py \
  --output "$OUTPUTDIR/OCAF_BRIDGE_CONTRACT.json" \
  >"$OUTPUTDIR/logs/ocaf_bridge_contract.log" 2>&1
frontier_contract_rc=$?
set -e
if [[ "$frontier_contract_rc" != 0 ]]; then
  write_pipeline_failure ocaf_bridge_contract_preflight "$frontier_contract_rc" "$OUTPUTDIR/OCAF_BRIDGE_CONTRACT.json"
  exit 30
fi

# v48.36 passed its CPU bridge contract but failed on the first A30 batch in a
# CUDA index_put_ broadcast.  Exercise the exact 141-D action / 529-D nominal
# observation geometry, including backward, on every configured training GPU
# before building/reusing indexes or starting the parallel adaptation jobs.
for gpu_spec in "gpu0:$GPU0" "gpu1:$GPU1"; do
  gpu_label="${gpu_spec%%:*}"
  gpu_id="${gpu_spec#*:}"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu_id" python tools/check_v48_36_cuda_group_broadcast_contract.py \
    --device cuda:0 --batch-size 96 --group-size 8 \
    --interaction-hidden "$EVIDENCE_INTERACTION_HIDDEN" \
    $([[ "${EVIDENCE_DUAL_INTERACTION_BRIDGE:-false}" == true ]] && echo --dual-interaction-bridge) \
    $([[ "${EVIDENCE_FACTORIZED_HARM_INTERACTION:-false}" == true ]] && echo --factorized-harm-interaction) \
    $([[ "${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL:-false}" == true ]] && echo --partial-pool-harm-residual) \
    $([[ "${EVIDENCE_RANK_BENEFIT_SKIP:-false}" == true ]] && echo --rank-benefit-skip) \
    $([[ "${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT:-false}" == true ]] && echo --postprefix-obs-transport-benefit) \
    $([[ "${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM:-false}" == true ]] && echo --postprefix-obs-transport-harm) \
    --output "$OUTPUTDIR/OCAF_CUDA_GROUP_BROADCAST_CONTRACT_${gpu_label}.json" \
    >"$OUTPUTDIR/logs/ocaf_cuda_group_broadcast_contract_${gpu_label}.log" 2>&1
  cuda_group_contract_rc=$?
  set -e
  if [[ "$cuda_group_contract_rc" != 0 ]]; then
    write_pipeline_failure "ocaf_cuda_group_broadcast_preflight_${gpu_label}" \
      "$cuda_group_contract_rc" "$OUTPUTDIR/OCAF_CUDA_GROUP_BROADCAST_CONTRACT_${gpu_label}.json"
    exit 30
  fi
done

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
if [[ "${EVIDENCE_DEP_BOUNDARY_ALIGNED:-false}" == "true" ]]; then index_contract_args+=(--dep-boundary-aligned); fi
if [[ "${EVIDENCE_GAP_ORDINAL_ONLY:-false}" == "true" ]]; then index_contract_args+=(--gap-ordinal-only); fi
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
if [[ "${EVIDENCE_DEP_BOUNDARY_ALIGNED:-false}" == "true" ]]; then val_index_contract_args+=(--dep-boundary-aligned); fi
if [[ "${EVIDENCE_GAP_ORDINAL_ONLY:-false}" == "true" ]]; then val_index_contract_args+=(--gap-ordinal-only); fi

teacher_train_t0="$(date +%s.%N)"; v4856_timing_event start teacher_index_train "$teacher_train_t0"
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

V4856_TEACHER_ROLE_ARGS=()
if [[ "${EVIDENCE_DEP_BOUNDARY_ALIGNED:-false}" == "true" ]]; then V4856_TEACHER_ROLE_ARGS+=(--dep-boundary-aligned); fi
if [[ "${EVIDENCE_GAP_ORDINAL_ONLY:-false}" == "true" ]]; then V4856_TEACHER_ROLE_ARGS+=(--gap-ordinal-only); fi

V4856_RAW_REUSE_TRAIN_ARGS=()
if [[ -n "${V4856_RAW_TEACHER_INDEX:-}" && -n "${V4856_RAW_TEACHER_SUMMARY:-}" &&       -f "${V4856_RAW_TEACHER_INDEX}" && -f "${V4856_RAW_TEACHER_SUMMARY}" ]]; then
  V4856_RAW_REUSE_TRAIN_ARGS=(--reuse-raw-index "$V4856_RAW_TEACHER_INDEX" --reuse-raw-summary "$V4856_RAW_TEACHER_SUMMARY" --reuse-raw-fallback-to-build)
fi
if [[ "$rebuild_index" == 1 ]]; then
  rm -f "$GROUP_INDEX" "$GROUP_SUMMARY"
  set +e
  python tools/build_teacher_pcd_index_v48.py \
    --dataset "$TRAIN_NEAR,$TRAIN_CONTACT" \
    --output "$GROUP_INDEX" --summary-output "$GROUP_SUMMARY" \
    --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" \
    --option-execution-semantics "$TRAIN_OPTION_EXECUTION_SEMANTICS" \
    --positive-gain "$POSITIVE_GAIN" --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" \
    --quality-mode warn \
    --workers "$V4856_TEACHER_INDEX_WORKERS" --worker-chunksize "$V4856_TEACHER_INDEX_CHUNKSIZE" \
    "${V4856_RAW_REUSE_TRAIN_ARGS[@]}" \
    --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE" \
    --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE" \
    --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE" \
    --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE" \
    --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE" \
    "${V4856_TEACHER_ROLE_ARGS[@]}" \
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
v4856_timing_event end teacher_index_train "$teacher_train_t0" 0

# v48.55 TCBC optional train-only, regime-free coordinate canonicalization.
# Scales are computed from the pooled adaptation-training teacher index only;
# dev/certificate/test labels never enter this transform.  DRS remains an
# identity transform under the Y factor, while DEP/GAP receive linear RMS
# normalization that preserves zero crossings and within-component ordering.
if [[ "${V4855_COMPONENT_CANONICALIZATION:-0}" == 1 ]]; then
  COMPONENT_SCALE_FILE="$OUTPUTDIR/V48_55_COMPONENT_BOUNDARY_SCALES.json"
  set +e
  python tools/compute_v48_55_component_boundary_scales.py \
    --index "$GROUP_INDEX" --output "$COMPONENT_SCALE_FILE" \
    --target-scale "${FACTOR_COMPONENT_MARGIN_TARGET_SCALE:-0.10}" \
    --drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE" \
    --dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE" \
    --gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE" \
    --hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE" \
    --proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE" \
    >"$OUTPUTDIR/logs/v48_55_component_boundary_scales.log" 2>&1
  scale_rc=$?
  set -e
  if [[ "$scale_rc" != 0 ]]; then
    write_pipeline_failure v48_55_component_scale "$scale_rc" "$COMPONENT_SCALE_FILE"
    exit 30
  fi
  FACTOR_COMPONENT_MARGIN_CANONICAL_SCALES="$(python - "$COMPONENT_SCALE_FILE" <<'PY_V4855_SCALES'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('strategy_regime_conditioning') or d.get('test_roots_read'):
    raise SystemExit(4)
s=str(d.get('canonical_scales_csv','')).strip()
if not s: raise SystemExit(4)
print(s)
PY_V4855_SCALES
)"
  export FACTOR_COMPONENT_MARGIN_CANONICAL_SCALES
fi

# Validation batching must use labels computed from adaptation-dev itself.
# Reuse is allowed only after the same exact dataset/label contract audit used
# for the training index; otherwise rebuild fail-closed.
teacher_dev_t0="$(date +%s.%N)"; v4856_timing_event start teacher_index_dev "$teacher_dev_t0"
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
V4856_RAW_REUSE_DEV_ARGS=()
if [[ -n "${V4856_RAW_DEV_TEACHER_INDEX:-}" && -n "${V4856_RAW_DEV_TEACHER_SUMMARY:-}" &&       -f "${V4856_RAW_DEV_TEACHER_INDEX}" && -f "${V4856_RAW_DEV_TEACHER_SUMMARY}" ]]; then
  V4856_RAW_REUSE_DEV_ARGS=(--reuse-raw-index "$V4856_RAW_DEV_TEACHER_INDEX" --reuse-raw-summary "$V4856_RAW_DEV_TEACHER_SUMMARY" --reuse-raw-fallback-to-build)
fi
if [[ "$rebuild_val_index" == 1 ]]; then
  rm -f "$VAL_GROUP_INDEX" "$VAL_GROUP_SUMMARY"
  set +e
  python tools/build_teacher_pcd_index_v48.py \
    --dataset "$DEV_NEAR,$DEV_CONTACT" \
    --output "$VAL_GROUP_INDEX" --summary-output "$VAL_GROUP_SUMMARY" \
    --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" \
    --option-execution-semantics "$TRAIN_OPTION_EXECUTION_SEMANTICS" \
    --positive-gain "$POSITIVE_GAIN" --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" \
    --quality-mode warn \
    --workers "$V4856_TEACHER_INDEX_WORKERS" --worker-chunksize "$V4856_TEACHER_INDEX_CHUNKSIZE" \
    "${V4856_RAW_REUSE_DEV_ARGS[@]}" \
    --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE" \
    --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE" \
    --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE" \
    --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE" \
    --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE" \
    "${V4856_TEACHER_ROLE_ARGS[@]}" \
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
v4856_timing_event end teacher_index_dev "$teacher_dev_t0" 0

run_variant() {
  local variant="$1" gpu="$2"
  local variant_t0="$(date +%s.%N)"; v4856_timing_event start "adapt_${variant}" "$variant_t0"
  local source="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  local run="$OUTPUTDIR/candidates/$variant"
  local factor_cache=""
  case "$variant" in
    balanced) factor_cache="${V4836_FACTOR_CACHE_BALANCED:-}" ;;
    precision) factor_cache="${V4836_FACTOR_CACHE_PRECISION:-}" ;;
  esac
  local rc=0
  if [[ ! -f "$source" ]]; then
    mkdir -p "$run" "$OUTPUTDIR/logs"
    printf 'missing source checkpoint: %s\n' "$source" >"$OUTPUTDIR/logs/adapt_${variant}.log"
    python - "$run/VARIANT_STAGE_FAILED.json" "$variant" "$source" <<'PY_SOURCE_MISSING'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({'event':'v48_36_variant_stage_failed','created_unix':time.time(),
 'variant':sys.argv[2],'stage':'source_checkpoint_preflight','exit_code':30,
 'missing_source_checkpoint':sys.argv[3],'test_roots_read':False},indent=2)+'\n')
PY_SOURCE_MISSING
    rc=30
  elif [[ -n "$factor_cache" && ! -d "$factor_cache" ]]; then
    mkdir -p "$run" "$OUTPUTDIR/logs"
    printf 'configured factor cache does not exist: %s\n' "$factor_cache" >"$OUTPUTDIR/logs/adapt_${variant}.log"
    python - "$run/VARIANT_STAGE_FAILED.json" "$variant" "$factor_cache" <<'PY_CACHE_MISSING'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({'event':'v48_36_variant_stage_failed','created_unix':time.time(),
 'variant':sys.argv[2],'stage':'factor_cache_preflight','exit_code':30,
 'missing_factor_cache':sys.argv[3],'test_roots_read':False},indent=2)+'\n')
PY_CACHE_MISSING
    rc=30
  else
    set +e
    RUN="$run" INIT_CKPT="$source" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="$VAL_GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
  NUM_WORKERS="${NUM_WORKERS:-3}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-3}" BATCH_SIZE="${BATCH_SIZE:-96}" \
  FACTOR_EPOCHS="${FACTOR_EPOCHS:-20}" FACTOR_PATIENCE="${FACTOR_PATIENCE:-6}" \
  IDENTITY_EPOCHS="${IDENTITY_EPOCHS:-24}" IDENTITY_PATIENCE="${IDENTITY_PATIENCE:-6}" \
  FINAL_EPOCHS="${FINAL_EPOCHS:-8}" FINAL_PATIENCE="${FINAL_PATIENCE:-3}" \
  IDENTITY_LR="${IDENTITY_LR:-0.00004}" FINAL_LR="${FINAL_LR:-0.00003}" \
  V4836_IDENTITY_TRAIN_ALL="${V4836_IDENTITY_TRAIN_ALL:-1}" V4836_COUPLE_ADMISSION_PRIOR="${V4836_COUPLE_ADMISSION_PRIOR:-1}" \
  V4837_FACTOR_PRESERVING_IDENTITY="${V4837_FACTOR_PRESERVING_IDENTITY:-0}" \
  V4838_RFR_RESERVE_ONLY="${V4838_RFR_RESERVE_ONLY:-0}" V4838_FACTOR_ALGORITHM_FAMILY="${V4838_FACTOR_ALGORITHM_FAMILY:-${OCRAP_ALGORITHM_VERSION:-v48.36-OCAF}}" \
  V4836_ADAPTIVE_IDENTITY_MARGIN="${V4836_ADAPTIVE_IDENTITY_MARGIN:-0}" V4836_ENABLE_FINAL_CALIBRATION="${V4836_ENABLE_FINAL_CALIBRATION:-0}" \
  V4845_SOWR_MARGIN_WITNESS="${V4845_SOWR_MARGIN_WITNESS:-0}" V4845_SOWR_OBS_KERNEL="${V4845_SOWR_OBS_KERNEL:-0}" \
  SOWR_EPOCHS="${SOWR_EPOCHS:-8}" SOWR_PATIENCE="${SOWR_PATIENCE:-3}" SOWR_LR="${SOWR_LR:-0.00005}" SOWR_BATCH_SIZE="${SOWR_BATCH_SIZE:-72}" \
  V4846_SEQUENTIAL_WITNESS="${V4846_SEQUENTIAL_WITNESS:-0}" V4846_WITNESS_OBS_EPOCHS="${V4846_WITNESS_OBS_EPOCHS:-5}" V4846_WITNESS_MARGIN_EPOCHS="${V4846_WITNESS_MARGIN_EPOCHS:-5}" \
  V4846_WITNESS_PATIENCE="${V4846_WITNESS_PATIENCE:-2}" V4846_WITNESS_LR="${V4846_WITNESS_LR:-0.00004}" OPTION_EXECUTION_SEMANTICS="$TRAIN_OPTION_EXECUTION_SEMANTICS" \
  FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT="${FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT:-0.0}" FACTOR_BENEFIT_MARGIN_TEMPERATURE="${FACTOR_BENEFIT_MARGIN_TEMPERATURE:-0.025}" \
  FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT="${FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT:-0.0}" FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT="${FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT:-0.0}" \
  FACTOR_COMPONENT_MARGIN_TARGET_MODE="${FACTOR_COMPONENT_MARGIN_TARGET_MODE:-raw}" FACTOR_COMPONENT_MARGIN_TARGET_SCALE="${FACTOR_COMPONENT_MARGIN_TARGET_SCALE:-0.10}" \
  FACTOR_COMPONENT_MARGIN_CANONICAL_SCALES="${FACTOR_COMPONENT_MARGIN_CANONICAL_SCALES:-}" FACTOR_COMPONENT_MARGIN_REGRESSION_RELIABILITY="${FACTOR_COMPONENT_MARGIN_REGRESSION_RELIABILITY:-}" \
  FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT="${FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT:-0.0}" FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT="${FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT:-0.0}" FACTOR_JOINT_RESERVE_BOUNDARY_WIDTH="${FACTOR_JOINT_RESERVE_BOUNDARY_WIDTH:-0.05}" \
  EVIDENCE_JOINT_RESERVE_TEMPERATURE="${EVIDENCE_JOINT_RESERVE_TEMPERATURE:-0.025}" \
  V4836_FACTOR_CACHE_RUN="$factor_cache" \
  PROPOSAL_TOP_K="$PROPOSAL_TOP_K" \
  EVIDENCE_CALIBRATOR_CONTEXT=true EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="$EVIDENCE_CONTEXT_SOURCE" \
  EVIDENCE_INTERACTION_HIDDEN="$EVIDENCE_INTERACTION_HIDDEN" EVIDENCE_INTERACTION_DROPOUT="$EVIDENCE_INTERACTION_DROPOUT" EVIDENCE_CONSENSUS_PRIOR_SCALE="$EVIDENCE_CONSENSUS_PRIOR_SCALE" \
  EVIDENCE_COMPONENT_COUNT=5 EVIDENCE_COMPONENT_SCALE="${EVIDENCE_COMPONENT_SCALE:-6.0}" \
  EVIDENCE_ADMISSION_PRIOR_MODE="$ADMISSION_PRIOR_MODE" EVIDENCE_ADMISSION_SCALE="${EVIDENCE_ADMISSION_SCALE:-2.0}" \
  EVIDENCE_SLACK_TEMPERATURE="${EVIDENCE_SLACK_TEMPERATURE:-0.025}" EVIDENCE_SLACK_PENALTY="${EVIDENCE_SLACK_PENALTY:-1.0}" EVIDENCE_FRONTIER_CAP_TEMPERATURE="${EVIDENCE_FRONTIER_CAP_TEMPERATURE:-0.10}" \
  IDENTITY_ELIGIBILITY_BOUNDARY_WEIGHT="${IDENTITY_ELIGIBILITY_BOUNDARY_WEIGHT:-1.0}" \
  IDENTITY_ELIGIBILITY_BOUNDARY_MARGIN="${IDENTITY_ELIGIBILITY_BOUNDARY_MARGIN:-0.20}" \
  IDENTITY_POSITIVE_MACRO_BALANCE_POWER="${IDENTITY_POSITIVE_MACRO_BALANCE_POWER:-0.50}" \
  IDENTITY_SCENE_BALANCE_POWER="${IDENTITY_SCENE_BALANCE_POWER:-0.50}" \
    bash scripts/adapt_ocrap_v48_36_ocaf_variant.sh >"$OUTPUTDIR/logs/adapt_${variant}.log" 2>&1
    rc=$?
    set -e
  fi
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
    python tools/extract_v48_36_failure_signature.py \
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
    'event':'v48_36_adaptation_failed','version':'v48.36-OCAF','implementation_version':os.environ.get('OCRAP_IMPLEMENTATION_VERSION','v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX'),
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
  v4856_timing_event end "adapt_${variant}" "$variant_t0" "$rc"
  return "$rc"
}

adapt_all_t0="$(date +%s.%N)"; v4856_timing_event start adaptation_all "$adapt_all_t0"
if [[ "$RESUME_AFTER_ADAPTATION" == 1 ]]; then
  s0=0; s1=0
  printf 'balanced=0 precision=0 resume_after_adaptation=1 retraining=0\n' | tee "$OUTPUTDIR/logs/adaptation_status.log"
else
  if [[ "$SERIAL_VARIANTS_ON_ONE_GPU" == 1 ]]; then
    # run_variant also toggles errexit internally while collecting failure
    # provenance. Keep that shell-option state inside a subshell so a failed
    # Balanced adaptation cannot abort the serial controller before Precision
    # is attempted and both exit codes are recorded.
    set +e
    ( run_variant balanced "$GPU0" ); s0=$?
    ( run_variant precision "$GPU1" ); s1=$?
    set -e
  else
    run_variant balanced "$GPU0" & p0=$!
    run_variant precision "$GPU1" & p1=$!
    set +e
    wait "$p0"; s0=$?
    wait "$p1"; s1=$?
    set -e
  fi
  printf 'balanced=%s precision=%s resume_after_adaptation=0 retraining=1 serial_variants=%s\n' "$s0" "$s1" "$SERIAL_VARIANTS_ON_ONE_GPU" | tee "$OUTPUTDIR/logs/adaptation_status.log"
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
  python tools/check_v48_36_ocaf_model_contract.py \
    --checkpoint "$ckpt" \
    --support-contract "$OUTPUTDIR/candidates/$variant/FACTOR_SUPPORT_CONTRACT.json" \
    --output "$OUTPUTDIR/candidates/$variant/MODEL_INFERENCE_CONTRACT.json" \
    --expect-frontier true --expect-value-regime-conditioning false --expect-admission-bounded false --expect-context-enabled true --expect-context-source "$EVIDENCE_CONTEXT_SOURCE" --expect-interaction-hidden "$EVIDENCE_INTERACTION_HIDDEN" --expect-dual-interaction-bridge "${EVIDENCE_DUAL_INTERACTION_BRIDGE:-false}" --expect-consensus-prior-scale "$EVIDENCE_CONSENSUS_PRIOR_SCALE" --expect-frontier-cap-temperature "${EVIDENCE_FRONTIER_CAP_TEMPERATURE:-0.10}" \
    --expect-factorized-harm-interaction "${EVIDENCE_FACTORIZED_HARM_INTERACTION:-false}" --expect-partial-pool-harm-residual "${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL:-false}" --expect-partial-pool-harm-residual-scale "${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL_SCALE:-0.50}" --expect-rank-benefit-skip "${EVIDENCE_RANK_BENEFIT_SKIP:-false}" --expect-rank-benefit-gain-init "${EVIDENCE_RANK_BENEFIT_GAIN_INIT:-1.0}" --expect-postprefix-obs-transport-benefit "${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT:-false}" --expect-postprefix-obs-transport-harm "${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM:-false}" --expect-postprefix-obs-transport-scale "${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_SCALE:-1.0}" \
    --expect-roct-benefit "${EVIDENCE_ROCT_BENEFIT:-false}" --expect-roct-deployability "${EVIDENCE_ROCT_DEPLOYABILITY:-false}" --expect-roct-scale "${EVIDENCE_ROCT_SCALE:-1.0}" --expect-roct-alpha "${EVIDENCE_ROCT_ALPHA:-0.20}" --expect-roct-beta "${EVIDENCE_ROCT_BETA:-0.20}" --expect-roct-top-m "${EVIDENCE_ROCT_TOP_M:-8}" --expect-roct-option-temperature "${EVIDENCE_ROCT_OPTION_TEMPERATURE:-0.35}" --expect-common-measure-root-mass "${EVIDENCE_COMMON_MEASURE_ROOT_MASS:-false}" \
    --expect-native-certificate-preservation "${EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION:-false}" --expect-native-drs-tolerance "${EVIDENCE_NATIVE_DRS_TOLERANCE:-0.05}" --expect-native-deployability-tolerance "${EVIDENCE_NATIVE_DEPLOYABILITY_TOLERANCE:-0.05}" \
    --expect-native-margin-complete-preservation "${EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION:-false}" --expect-native-advantage-preservation "${EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION:-false}" --expect-native-exact-advantage-preservation "${EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION:-false}" --expect-native-boundary-complete-advantage-preservation "${EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION:-false}" --expect-native-physical-student-drs "${EVIDENCE_PHYSICAL_STUDENT_DRS:-false}" --expect-native-gap-tolerance "${EVIDENCE_NATIVE_GAP_TOLERANCE:-0.05}" --expect-native-positive-gain "${EVIDENCE_NATIVE_POSITIVE_GAIN:-${FACTOR_RECOVERY_ADVANTAGE_POSITIVE_GAIN:-0.015}}" \
    --expect-admission-head "$([[ "${V4838_RFR_RESERVE_ONLY:-0}" == 1 ]] && echo false || echo true)" \
    --expect-benefit-margin-temperature "${FACTOR_BENEFIT_MARGIN_TEMPERATURE:-0.025}" --expect-joint-reserve-temperature "${EVIDENCE_JOINT_RESERVE_TEMPERATURE:-0.025}" \
    --expect-component-prior-logit -2.0 --expect-component-count 5 --expect-component-scale "${EVIDENCE_COMPONENT_SCALE:-6.0}" \
    --expect-benefit-residual-scale "${EVIDENCE_BENEFIT_RESIDUAL_SCALE:-1.0}" \
    --expect-unbounded-benefit-factor "${EVIDENCE_UNBOUNDED_BENEFIT_FACTOR:-false}" --expect-unbounded-harm-factors "${EVIDENCE_UNBOUNDED_HARM_FACTORS:-false}" \
    --expect-reserve-factor-alignment "${EVIDENCE_RESERVE_FACTOR_ALIGNMENT:-false}" \
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
  python tools/check_v48_36_ocaf_training_contract.py \
    --run "$OUTPUTDIR/candidates/$variant" \
    --output "$OUTPUTDIR/candidates/$variant/TRAINING_CONTRACT.json" \
    --expect-identity-all "$([[ "${V4836_IDENTITY_TRAIN_ALL:-1}" == 1 ]] && echo true || echo false)" \
    --expect-prior-coupled "$([[ "${V4836_COUPLE_ADMISSION_PRIOR:-1}" == 1 ]] && echo true || echo false)" \
    --expect-factor-preserving "$([[ "${V4837_FACTOR_PRESERVING_IDENTITY:-0}" == 1 ]] && echo true || echo false)" \
    --expect-reserve-only "$([[ "${V4838_RFR_RESERVE_ONLY:-0}" == 1 ]] && echo true || echo false)" \
    --expect-benefit-margin-regression "${FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT:-0.0}" \
    --expect-benefit-margin-temperature "${FACTOR_BENEFIT_MARGIN_TEMPERATURE:-0.025}" \
    --expect-component-underestimation "${FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT:-0.0}" \
    --expect-safe-positive-component-overestimation "${FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT:-0.0}" \
    --expect-joint-reserve-regression "${FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT:-0.0}" \
    --expect-benefit-residual-scale "${EVIDENCE_BENEFIT_RESIDUAL_SCALE:-1.0}" \
    --expect-unbounded-benefit-factor "${EVIDENCE_UNBOUNDED_BENEFIT_FACTOR:-false}" --expect-unbounded-harm-factors "${EVIDENCE_UNBOUNDED_HARM_FACTORS:-false}" \
    --expect-reserve-factor-alignment "${EVIDENCE_RESERVE_FACTOR_ALIGNMENT:-false}" \
    --expect-dual-interaction-bridge "${EVIDENCE_DUAL_INTERACTION_BRIDGE:-false}" \
    --expect-factorized-harm-interaction "${EVIDENCE_FACTORIZED_HARM_INTERACTION:-false}" --expect-partial-pool-harm-residual "${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL:-false}" --expect-partial-pool-harm-residual-scale "${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL_SCALE:-0.50}" --expect-rank-benefit-skip "${EVIDENCE_RANK_BENEFIT_SKIP:-false}" --expect-rank-benefit-gain-init "${EVIDENCE_RANK_BENEFIT_GAIN_INIT:-1.0}" --expect-postprefix-obs-transport-benefit "${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT:-false}" --expect-postprefix-obs-transport-harm "${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM:-false}" --expect-postprefix-obs-transport-scale "${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_SCALE:-1.0}" \
    --expect-roct-benefit "${EVIDENCE_ROCT_BENEFIT:-false}" --expect-roct-deployability "${EVIDENCE_ROCT_DEPLOYABILITY:-false}" --expect-roct-scale "${EVIDENCE_ROCT_SCALE:-1.0}" --expect-roct-alpha "${EVIDENCE_ROCT_ALPHA:-0.20}" --expect-roct-beta "${EVIDENCE_ROCT_BETA:-0.20}" --expect-roct-top-m "${EVIDENCE_ROCT_TOP_M:-8}" --expect-roct-option-temperature "${EVIDENCE_ROCT_OPTION_TEMPERATURE:-0.35}" \
    --expect-component-margin-target-mode "${FACTOR_COMPONENT_MARGIN_TARGET_MODE:-raw}" \
    --expect-component-margin-target-scale "${FACTOR_COMPONENT_MARGIN_TARGET_SCALE:-0.10}" \
    --expect-algorithm-variant "${OCRAP_ALGORITHM_VERSION:-v48.36-OCAF}" \
    --expect-adaptive-margin "$([[ "${V4836_ADAPTIVE_IDENTITY_MARGIN:-0}" == 1 ]] && echo true || echo false)" --expect-final-enabled "$([[ "${V4836_ENABLE_FINAL_CALIBRATION:-0}" == 1 ]] && echo true || echo false)" \
    --expect-eligible-policy "$([[ "${V4838_RFR_RESERVE_ONLY:-0}" == 1 ]] && echo false || echo true)" \
    --expect-boundary "$([[ "${V4838_RFR_RESERVE_ONLY:-0}" == 1 ]] && echo false || echo true)" \
    --expect-prior-mode "$ADMISSION_PRIOR_MODE" --expect-context-source "$EVIDENCE_CONTEXT_SOURCE" --expect-proposal-top-k "$PROPOSAL_TOP_K" \
    >"$OUTPUTDIR/logs/training_contract_${variant}.log" 2>&1
  training_contract_rc=$?
  set -e
  if [[ "$training_contract_rc" != 0 ]]; then
    write_pipeline_failure training_contract "$training_contract_rc" "$OUTPUTDIR/candidates/$variant/TRAINING_CONTRACT.json" "$s0" "$s1"
    exit 30
  fi
done

# v48.52 optional fail-closed semantic preflight.  PSA changes only the
# boundary-complete witness teacher-sign target, so verify the nested witness
# checkpoint/stage contract before any calibration/certificate population is
# touched.  Historical versions are unaffected unless the explicit flag is set.
if [[ "${V4852_REQUIRE_PSA_CONTRACT:-0}" == 1 ]]; then
  for variant in balanced precision; do
    [[ "$variant" == balanced && "$s0" != 0 ]] && continue
    [[ "$variant" == precision && "$s1" != 0 ]] && continue
    set +e
    python tools/check_v48_52_psa_contract.py \
      --run "$OUTPUTDIR/candidates/$variant" \
      --expect-physical "${V4852_PHYSICAL_TEACHER_SIGN_ALIGNMENT:-false}" \
      --output "$OUTPUTDIR/candidates/$variant/V48_52_PSA_CONTRACT.json" \
      >"$OUTPUTDIR/logs/v48_52_psa_contract_${variant}.log" 2>&1
    psa_contract_rc=$?
    set -e
    if [[ "$psa_contract_rc" != 0 ]]; then
      write_pipeline_failure v48_52_psa_contract "$psa_contract_rc" "$OUTPUTDIR/candidates/$variant/V48_52_PSA_CONTRACT.json" "$s0" "$s1"
      exit 30
    fi
  done
fi

# v48.53 fail-closed Certificate Structural Equivalence preflight.  Unlike
# v48.52 teacher-only PSA, CSE can alter both the witness sign supervision and
# the deployed/native hard DRS coordinate.  Verify both sides explicitly before
# calibration so an asymmetric or stale checkpoint cannot be attributed as an
# algorithm result.
if [[ "${V4853_REQUIRE_CSE_CONTRACT:-0}" == 1 ]]; then
  for variant in balanced precision; do
    [[ "$variant" == balanced && "$s0" != 0 ]] && continue
    [[ "$variant" == precision && "$s1" != 0 ]] && continue
    set +e
    python tools/check_v48_53_cse_contract.py \
      --run "$OUTPUTDIR/candidates/$variant" \
      --expect-teacher-physical "${V4852_PHYSICAL_TEACHER_SIGN_ALIGNMENT:-false}" \
      --expect-student-physical "${V4853_PHYSICAL_STUDENT_SIGN_ALIGNMENT:-false}" \
      --output "$OUTPUTDIR/candidates/$variant/V48_53_CSE_CONTRACT.json" \
      >"$OUTPUTDIR/logs/v48_53_cse_contract_${variant}.log" 2>&1
    cse_contract_rc=$?
    set -e
    if [[ "$cse_contract_rc" != 0 ]]; then
      write_pipeline_failure v48_53_cse_contract "$cse_contract_rc" "$OUTPUTDIR/candidates/$variant/V48_53_CSE_CONTRACT.json" "$s0" "$s1"
      exit 30
    fi
  done
fi

# v48.54 fail-closed Invariant-Preserving Boundary Distillation preflight.
# IPBD must be training-only: teacher/student/deployment hard sign coordinates
# remain the validated q-hard BC-FC reference while the privileged selected-
# option physical zero boundary is optionally distilled into predicted margins.
if [[ "${V4854_REQUIRE_IPBD_CONTRACT:-0}" == 1 ]]; then
  for variant in balanced precision; do
    [[ "$variant" == balanced && "$s0" != 0 ]] && continue
    [[ "$variant" == precision && "$s1" != 0 ]] && continue
    set +e
    python tools/check_v48_54_ipbd_contract.py \
      --run "$OUTPUTDIR/candidates/$variant" \
      --expect-ipbd "${V4854_INVARIANT_PHYSICAL_BOUNDARY_DISTILLATION:-false}" \
      --output "$OUTPUTDIR/candidates/$variant/V48_54_IPBD_CONTRACT.json" \
      >"$OUTPUTDIR/logs/v48_54_ipbd_contract_${variant}.log" 2>&1
    ipbd_contract_rc=$?
    set -e
    if [[ "$ipbd_contract_rc" != 0 ]]; then
      write_pipeline_failure v48_54_ipbd_contract "$ipbd_contract_rc" "$OUTPUTDIR/candidates/$variant/V48_54_IPBD_CONTRACT.json" "$s0" "$s1"
      exit 30
    fi
  done
fi


# v48.55 fail-closed Coordinate-Typed Component Boundary Calibration preflight.
# This contract verifies that the DRS sign-only factor changes only continuous
# magnitude regression support, while the DEP/GAP factor uses the pooled
# train-only linear scales published at the run root.  Hard-veto and q-hard
# deployment semantics remain unchanged.
if [[ "${V4855_REQUIRE_TCBC_CONTRACT:-0}" == 1 ]]; then
  for variant in balanced precision; do
    [[ "$variant" == balanced && "$s0" != 0 ]] && continue
    [[ "$variant" == precision && "$s1" != 0 ]] && continue
    tcbc_scale_args=()
    if [[ "${V4855_COMPONENT_CANONICALIZATION:-0}" == 1 ]]; then
      tcbc_scale_args=(--scale-file "$OUTPUTDIR/V48_55_COMPONENT_BOUNDARY_SCALES.json")
    fi
    set +e
    python tools/check_v48_55_tcbc_contract.py \
      --run "$OUTPUTDIR/candidates/$variant" \
      --expect-drs-sign-only "${V4855_DRS_SIGN_ONLY:-false}" \
      --expect-continuous-canonicalization "${V4855_COMPONENT_CANONICALIZATION:-false}" \
      "${tcbc_scale_args[@]}" \
      --output "$OUTPUTDIR/candidates/$variant/V48_55_TCBC_CONTRACT.json" \
      >"$OUTPUTDIR/logs/v48_55_tcbc_contract_${variant}.log" 2>&1
    tcbc_contract_rc=$?
    set -e
    if [[ "$tcbc_contract_rc" != 0 ]]; then
      write_pipeline_failure v48_55_tcbc_contract "$tcbc_contract_rc" "$OUTPUTDIR/candidates/$variant/V48_55_TCBC_CONTRACT.json" "$s0" "$s1"
      exit 30
    fi
  done
fi

# v48.56 fail-closed Decision-Role Aligned Certificate preflight.
if [[ "${V4856_REQUIRE_DRAC_CONTRACT:-0}" == 1 ]]; then
  for variant in balanced precision; do
    [[ "$variant" == balanced && "$s0" != 0 ]] && continue
    [[ "$variant" == precision && "$s1" != 0 ]] && continue
    set +e
    python tools/check_v48_56_drac_contract.py \
      --run "$OUTPUTDIR/candidates/$variant" \
      --expect-dep-boundary-aligned "${EVIDENCE_DEP_BOUNDARY_ALIGNED:-false}" \
      --expect-gap-ordinal-only "${EVIDENCE_GAP_ORDINAL_ONLY:-false}" \
      --output "$OUTPUTDIR/candidates/$variant/V48_56_DRAC_CONTRACT.json" \
      >"$OUTPUTDIR/logs/v48_56_drac_contract_${variant}.log" 2>&1
    drac_contract_rc=$?
    set -e
    if [[ "$drac_contract_rc" != 0 ]]; then
      write_pipeline_failure v48_56_drac_contract "$drac_contract_rc" "$OUTPUTDIR/candidates/$variant/V48_56_DRAC_CONTRACT.json" "$s0" "$s1"
      exit 30
    fi
  done
fi

variants=""
[[ "$s0" == 0 ]] && variants="balanced"
[[ "$s1" == 0 ]] && variants="${variants:+$variants,}precision"
certificate_t0="$(date +%s.%N)"; v4856_timing_event start certificate_controller "$certificate_t0"
set +e
OUTPUTDIR="$OUTPUTDIR" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
GPU0="$GPU0" GPU1="$GPU1" VARIANTS="$variants" PROPOSAL_TOP_K="$PROPOSAL_TOP_K" V4836_ATTEMPT_ID="$ATTEMPT_ID" \
  OPTION_EXECUTION_SEMANTICS="$EVAL_OPTION_EXECUTION_SEMANTICS" \
  OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$OUTPUTDIR/logs/certificate_controller.log" 2>&1
raw_cert_rc=$?
set -e
case "$raw_cert_rc" in
  0|20) cert_rc="$raw_cert_rc" ;;
  *) cert_rc=30 ;;
esac
v4856_timing_event end certificate_controller "$certificate_t0" "$cert_rc"

if [[ "$cert_rc" == 30 ]]; then
  write_pipeline_failure certificate "$raw_cert_rc" "$OUTPUTDIR/logs/certificate_controller.log" "$s0" "$s1"
  exit 30
fi

set +e
python tools/check_v48_16_learning_gates.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/learning_gates_v48_36.json" --version v48.36-OCAF \
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
python tools/check_v48_36_certificate_status_contract.py \
  --run "$OUTPUTDIR" --expected-attempt-id "$ATTEMPT_ID" \
  --output "$OUTPUTDIR/V48_36_CERTIFICATE_STATUS_CONTRACT.json" \
  >"$OUTPUTDIR/logs/certificate_status_contract.log" 2>&1
certificate_status_contract_rc=$?
set -e
if [[ "$certificate_status_contract_rc" != 0 ]]; then
  write_pipeline_failure certificate_status_contract "$certificate_status_contract_rc" "$OUTPUTDIR/V48_36_CERTIFICATE_STATUS_CONTRACT.json" "$s0" "$s1"
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
doc={'event':'v48_36_ocaf_controller_complete','version':'v48.36-OCAF','implementation_version':os.environ.get('OCRAP_IMPLEMENTATION_VERSION','v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX'),'created_unix':time.time(),'attempt_id':attempt_id,
     'source_run':str(source),'protocol_root':str(protocol),'variants':variants,
     'raw_certificate_exit_code':raw_rc,'certificate_exit_code':rc,'pipeline_exit_code':rc,
     'certificate_executed':True,'gate_evaluated':True,'gate_passed':(rc==0 and next_exists),
     'next_commands_generated':next_exists,'pipeline_valid':True,
     'adaptation_reused_without_retraining':resumed,'resume_contract':str(root/'V48_36_RESUME_CONTRACT.json') if resumed else None,
     'test_roots_read':False}
tmp=root/f'.V48_36_COMPLETE.json.tmp.{os.getpid()}.{time.time_ns()}'
with tmp.open('w',encoding='utf-8') as f:
    json.dump(doc,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp,root/'V48_36_COMPLETE.json')
PY
completion_rc=$?
set -e
if [[ "$completion_rc" != 0 ]]; then
  write_pipeline_failure completion_contract 30 "RC/NEXT_COMMANDS mismatch" "$s0" "$s1"
  exit 30
fi
set +e
python tools/resolve_v48_36_authoritative_result.py --run "$OUTPUTDIR" \
  --output "$OUTPUTDIR/AUTHORITATIVE_RUN_STATUS.json" \
  --expect-exit-code "$cert_rc" --expect-attempt-id "$ATTEMPT_ID" --archive-stale-markers \
  >"$OUTPUTDIR/logs/authoritative_run_state.log" 2>&1
state_contract_rc=$?
set -e
if [[ "$state_contract_rc" != 0 ]]; then
  write_pipeline_failure terminal_state_contract "$state_contract_rc" "$OUTPUTDIR/AUTHORITATIVE_RUN_STATUS.json" "$s0" "$s1"
  exit 30
fi
# Refresh the diagnostics now that the authoritative completion exists.  Older
# ordering left learning_gates_v48_36.json claiming authoritative_state=false
# even for a valid RC=20 run, which could mislead later algorithm analysis.
set +e
python tools/check_v48_16_learning_gates.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/learning_gates_v48_36.json" --version v48.36-OCAF \
  >"$OUTPUTDIR/logs/learning_gates_post_terminal.log" 2>&1
post_learning_rc=$?
python - "$OUTPUTDIR/learning_gates_v48_36.json" "$cert_rc" "$ATTEMPT_ID" <<'PY_POST_LEARNING'
import json,sys
doc=json.load(open(sys.argv[1],encoding='utf-8')); expected=int(sys.argv[2]); attempt=sys.argv[3]
auth=doc.get('authoritative_state') or {}
if not (auth.get('valid') is True and auth.get('pipeline_valid') is True and
        int(auth.get('exit_code')) == expected and auth.get('attempt_id') == attempt):
    raise SystemExit(f'post-terminal learning-gate authoritative snapshot mismatch: {auth}')
PY_POST_LEARNING
post_learning_contract_rc=$?
set -e
if [[ "$post_learning_rc" != 0 || "$post_learning_contract_rc" != 0 ]]; then
  write_pipeline_failure post_terminal_diagnostics 4 "learning_gates_rc=$post_learning_rc snapshot_contract_rc=$post_learning_contract_rc" "$s0" "$s1"
  exit 30
fi
v4856_timing_event end pipeline "$V4856_PIPELINE_T0" "$cert_rc"
exit "$cert_rc"
