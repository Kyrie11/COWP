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

OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_33_eligible_set_policy_dedicated_4833}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
ALLOW_PARTIAL_VARIANTS="${ALLOW_PARTIAL_VARIANTS:-0}"
PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-5}"

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
# Status files are run-local state, not cache. Clear them before every controller
# execution so a resumed run cannot inherit a stale failure or authorization.
rm -f "$OUTPUTDIR"/ADAPTATION_FAILED_*.json "$OUTPUTDIR"/FAILURE_SIGNATURE_*.json "$OUTPUTDIR/PIPELINE_FAILED.json" \
      "$OUTPUTDIR/V48_33_COMPLETE.json" "$OUTPUTDIR/NEXT_COMMANDS.txt" \
      "$OUTPUTDIR/NEXT_COMMANDS_STATUS.json" "$OUTPUTDIR/NEXT_COMMANDS_BLOCKED.json" \
      "$OUTPUTDIR/GATE_FAILED.json" "$OUTPUTDIR/CALIBRATION_FAILED.json" \
      "$OUTPUTDIR/chosen_base_run_dedicated.txt"
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
  python - "$OUTPUTDIR" "$PROTOCOL_ROOT" "$SOURCE_RUN" "$stage" "$raw_rc" "$detail" "$balanced_rc" "$precision_rc" <<'PY_FAILURE'
import hashlib,json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); protocol=pathlib.Path(sys.argv[2]); source=pathlib.Path(sys.argv[3])
stage=sys.argv[4]; raw_rc=int(sys.argv[5]); detail=sys.argv[6]
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
certificate_executed=stage in {'certificate','completion_contract'}
gate_evaluated=bool(next_status.get('gate_evaluated',False)) if certificate_executed else False
failed={'event':'v48_33_pipeline_failed','created_unix':time.time(),'stage':stage,
        'raw_exit_code':raw_rc,'normalized_exit_code':30,'pipeline_exit_code':30,'detail':detail,
        'adaptation_exit_codes':{'balanced':balanced_rc,'precision':precision_rc},
        'certificate_executed':certificate_executed,'gate_evaluated':gate_evaluated,
        'pipeline_valid':False,'test_roots_read':False}
(root/'PIPELINE_FAILED.json').write_text(json.dumps(failed,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
doc={'event':'v48_33_eligible_set_policy_controller_complete','created_unix':time.time(),
     'source_run':str(source),'protocol_root':str(protocol),'variants':variants,
     'raw_certificate_exit_code':raw_rc if certificate_executed else None,
     'certificate_exit_code':30 if certificate_executed else None,'pipeline_exit_code':30,
     'certificate_executed':certificate_executed,'gate_evaluated':gate_evaluated,
     'gate_passed':False,'next_commands_generated':False,
     'pipeline_valid':False,'failure_stage':stage,'test_roots_read':False}
(root/'V48_33_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
blocked={'event':'v48_33_next_commands_blocked','created_unix':time.time(),
         'generated':False,'reason':'pipeline_failure','failure_stage':stage,
         'pipeline_exit_code':30,'certificate_executed':certificate_executed,
         'gate_evaluated':gate_evaluated,'test_roots_read':False}
(root/'NEXT_COMMANDS_BLOCKED.json').write_text(json.dumps(blocked,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(root/'NEXT_COMMANDS_STATUS.json').write_text(json.dumps(blocked,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY_FAILURE
}
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


# Exercise the exact v48.33 eligible-set policy loss before any expensive
# index construction or GPU training.  The preflight requires finite gradients
# through admission, opportunity and harm heads on multiple scene-time groups.
set +e
python tools/check_v48_33_multigroup_loss_contract.py \
  --output "$OUTPUTDIR/MULTIGROUP_LOSS_CONTRACT.json" \
  >"$OUTPUTDIR/logs/multigroup_loss_contract.log" 2>&1
loss_contract_rc=$?
set -e
if [[ "$loss_contract_rc" != 0 ]]; then
  write_pipeline_failure loss_contract_preflight "$loss_contract_rc" "$OUTPUTDIR/MULTIGROUP_LOSS_CONTRACT.json"
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
    echo "teacher-index contract changed; rebuilding the index" | tee -a "$OUTPUTDIR/logs/check_teacher_index_contract.log"
    rebuild_index=1
  fi
else
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
  [[ "$val_contract_rc" == 0 ]] || rebuild_val_index=1
else
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
    balanced) factor_cache="${V4833_FACTOR_CACHE_BALANCED:-}" ;;
    precision) factor_cache="${V4833_FACTOR_CACHE_PRECISION:-}" ;;
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
  IDENTITY_LR="${IDENTITY_LR:-0.00006}" FINAL_LR="${FINAL_LR:-0.00003}" \
  V4833_IDENTITY_TRAIN_ALL=1 V4833_COUPLE_ADMISSION_PRIOR=1 \
  V4833_ADAPTIVE_IDENTITY_MARGIN=0 V4833_ENABLE_FINAL_CALIBRATION=0 \
  V4833_FACTOR_CACHE_RUN="$factor_cache" \
  PROPOSAL_TOP_K="$PROPOSAL_TOP_K" \
  EVIDENCE_COMPONENT_COUNT=5 EVIDENCE_COMPONENT_SCALE="${EVIDENCE_COMPONENT_SCALE:-6.0}" \
  EVIDENCE_ADMISSION_PRIOR_MODE=safety_slack EVIDENCE_ADMISSION_SCALE="${EVIDENCE_ADMISSION_SCALE:-2.0}" \
  EVIDENCE_SLACK_TEMPERATURE="${EVIDENCE_SLACK_TEMPERATURE:-0.025}" EVIDENCE_SLACK_PENALTY="${EVIDENCE_SLACK_PENALTY:-1.0}" \
  bash scripts/adapt_ocrap_v48_33_eligible_set_policy_variant.sh >"$OUTPUTDIR/logs/adapt_${variant}.log" 2>&1
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
    python tools/extract_v48_33_failure_signature.py \
      --log "$OUTPUTDIR/logs/adapt_${variant}.log" --output "$signature" \
      --stage "$stage" --exit-code "$rc" >/dev/null || true
    python - "$OUTPUTDIR" "$variant" "$rc" "$OUTPUTDIR/logs/adapt_${variant}.log" "$stage" "$signature" <<'PY_ADAPT_FAIL'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); variant=sys.argv[2]; rc=int(sys.argv[3]); log=pathlib.Path(sys.argv[4]); stage=sys.argv[5]; signature=pathlib.Path(sys.argv[6])
tail='\n'.join(log.read_text(errors='replace').splitlines()[-100:]) if log.exists() else ''
sig={}
try: sig=json.load(open(signature))
except Exception: pass
(root/f'ADAPTATION_FAILED_{variant}.json').write_text(json.dumps({
    'event':'v48_33_adaptation_failed','variant':variant,'stage':stage,
    'exit_code':rc,'log':str(log),'failure_signature':sig,
    'log_tail':tail,'created_unix':time.time(),'test_roots_read':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY_ADAPT_FAIL
  else
    rm -f "$OUTPUTDIR/ADAPTATION_FAILED_${variant}.json"
  fi
  return "$rc"
}

run_variant balanced "$GPU0" & p0=$!
run_variant precision "$GPU1" & p1=$!
set +e
wait "$p0"; s0=$?
wait "$p1"; s1=$?
set -e
printf 'balanced=%s precision=%s\n' "$s0" "$s1" | tee "$OUTPUTDIR/logs/adaptation_status.log"

if [[ "$s0" != 0 && "$s1" != 0 ]]; then
  write_pipeline_failure adaptation 30 "both variants failed; inspect FAILURE_SIGNATURE_balanced.json and FAILURE_SIGNATURE_precision.json" "$s0" "$s1"
  python tools/check_v48_16_learning_gates.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/learning_gates_v48_33.json" --version v48.33-ELIGIBLE-SET-POLICY || true
  exit 30
fi
if [[ "$ALLOW_PARTIAL_VARIANTS" != 1 && ( "$s0" != 0 || "$s1" != 0 ) ]]; then
  write_pipeline_failure adaptation 30 "one variant failed; set ALLOW_PARTIAL_VARIANTS=1 only for explicit debugging" "$s0" "$s1"
  python tools/check_v48_16_learning_gates.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/learning_gates_v48_33.json" --version v48.33-ELIGIBLE-SET-POLICY || true
  exit 30
fi

# v48.33 fail-closed preflight: training and downstream inference must
# construct the same frontier/prior/admission model.
for variant in balanced precision; do
  [[ "$variant" == balanced && "$s0" != 0 ]] && continue
  [[ "$variant" == precision && "$s1" != 0 ]] && continue
  ckpt="$OUTPUTDIR/candidates/$variant/model_v48_trac_sr/best.pt"
  set +e
  python tools/check_v48_32_model_contract.py \
    --checkpoint "$ckpt" \
    --support-contract "$OUTPUTDIR/candidates/$variant/FACTOR_SUPPORT_CONTRACT.json" \
    --output "$OUTPUTDIR/candidates/$variant/MODEL_INFERENCE_CONTRACT.json" \
    --expect-frontier true --expect-admission-bounded true \
    --expect-component-prior-logit -2.0 --expect-component-count 5 --expect-component-scale "${EVIDENCE_COMPONENT_SCALE:-6.0}" \
    --expect-admission-prior-detach any \
    --expect-admission-prior-mode safety_slack --expect-slack-temperature "${EVIDENCE_SLACK_TEMPERATURE:-0.025}" --expect-slack-penalty "${EVIDENCE_SLACK_PENALTY:-1.0}" \
    >"$OUTPUTDIR/logs/model_contract_${variant}.log" 2>&1
  contract_model_rc=$?
  set -e
  if [[ "$contract_model_rc" != 0 ]]; then
    write_pipeline_failure model_inference_contract "$contract_model_rc" "$OUTPUTDIR/candidates/$variant/MODEL_INFERENCE_CONTRACT.json" "$s0" "$s1"
    exit 30
  fi
  set +e
  python tools/check_v48_33_training_contract.py \
    --run "$OUTPUTDIR/candidates/$variant" \
    --output "$OUTPUTDIR/candidates/$variant/TRAINING_CONTRACT.json" \
    --expect-identity-all true --expect-prior-coupled true \
    --expect-adaptive-margin false --expect-final-enabled false \
    --expect-eligible-policy true --expect-proposal-top-k "$PROPOSAL_TOP_K" \
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
  OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit bash scripts/calibrate_v48_33_certificate_pool.sh >"$OUTPUTDIR/logs/certificate_controller.log" 2>&1
raw_cert_rc=$?
set -e
case "$raw_cert_rc" in
  0|20) cert_rc="$raw_cert_rc" ;;
  *) cert_rc=30 ;;
esac

python tools/check_v48_16_learning_gates.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/learning_gates_v48_33.json" --version v48.33-ELIGIBLE-SET-POLICY || true
python tools/summarize_v48_30_gate_failure.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/GATE_FAILURE_DECOMPOSITION.json" || true
if [[ "$cert_rc" == 30 ]]; then
  write_pipeline_failure certificate "$raw_cert_rc" "$OUTPUTDIR/logs/certificate_controller.log" "$s0" "$s1"
  exit 30
fi

set +e
python - "$OUTPUTDIR" "$PROTOCOL_ROOT" "$SOURCE_RUN" "$raw_cert_rc" "$cert_rc" <<'PY'
import hashlib,json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); protocol=pathlib.Path(sys.argv[2]); source=pathlib.Path(sys.argv[3])
raw_rc=int(sys.argv[4]); rc=int(sys.argv[5]); variants={}
for name in ('balanced','precision'):
    p=root/'candidates'/name/'model_v48_trac_sr'/'best.pt'
    if p.is_file(): variants[name]={'checkpoint':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
next_exists=(root/'NEXT_COMMANDS.txt').is_file()
blocked_exists=(root/'NEXT_COMMANDS_BLOCKED.json').is_file()
consistent=(rc==0 and next_exists and not blocked_exists) or (rc==20 and (not next_exists) and blocked_exists)
if not consistent:
    raise SystemExit(f'certificate/NEXT_COMMANDS contract mismatch: rc={rc} next={next_exists} blocked={blocked_exists}')
doc={'event':'v48_33_eligible_set_policy_controller_complete','created_unix':time.time(),
     'source_run':str(source),'protocol_root':str(protocol),'variants':variants,
     'raw_certificate_exit_code':raw_rc,'certificate_exit_code':rc,'pipeline_exit_code':rc,
     'certificate_executed':True,'gate_evaluated':True,'gate_passed':(rc==0 and next_exists),
     'next_commands_generated':next_exists,'pipeline_valid':True,'test_roots_read':False}
(root/'V48_33_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY
completion_rc=$?
set -e
if [[ "$completion_rc" != 0 ]]; then
  write_pipeline_failure completion_contract 30 "RC/NEXT_COMMANDS mismatch" "$s0" "$s1"
  exit 30
fi
exit "$cert_rc"
