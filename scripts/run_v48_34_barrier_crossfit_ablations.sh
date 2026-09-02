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

MAIN_RUN="${MAIN_RUN:-runs/ocrap_v48_34_barrier_crossfit_dedicated_4834}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_34_barrier_crossfit_ablations_4834}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
PROPOSAL_TOP_K=5

TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"
TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"
DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
GROUP_INDEX="$MAIN_RUN/evidence_adapt_teacher_pcd_index.jsonl"
VAL_GROUP_INDEX="$MAIN_RUN/evidence_adapt_dev_teacher_pcd_index.jsonl"

mkdir -p "$ROOT/tasks" "$ROOT/logs"
rm -f "$ROOT/ABLATIONS_COMPLETE.json" "$ROOT/ABLATIONS_STATUS.json" "$ROOT/ABLATIONS_FAILED.json"

python - "$MAIN_RUN/V48_34_COMPLETE.json" <<'PY_AUTH'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f'missing main-run completion contract: {p}')
d=json.load(open(p))
if not bool(d.get('pipeline_valid', False)) or not bool(d.get('certificate_executed', False)) or not bool(d.get('gate_evaluated', False)):
    raise SystemExit('ablations require a valid, evaluated v48.34 main pipeline')
rc=int(d.get('certificate_exit_code', d.get('pipeline_exit_code', -1)) or -1)
if rc != 20:
    raise SystemExit(f'ablations are authorized only after main RC=20; observed {rc}')
if bool(d.get('test_roots_read', True)):
    raise SystemExit('main completion contract reports test-root access')
PY_AUTH

for p in "$GROUP_INDEX" "$VAL_GROUP_INDEX"; do
  [[ -s "$p" ]] || { echo "missing main-run index $p" >&2; exit 30; }
done

write_task_failed() {
  local out="$1" stage="$2" rc="$3" log="${4:-}" detail="${5:-}"
  python - "$out" "$stage" "$rc" "$log" "$detail" <<'PY_FAIL'
import json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
log=pathlib.Path(sys.argv[4]) if sys.argv[4] else None
tail='\n'.join(log.read_text(errors='replace').splitlines()[-120:]) if log and log.is_file() else ''
doc={'complete':False,'event':'v48_34_ablation_task_failed','stage':sys.argv[2],
     'raw_exit_code':int(sys.argv[3]),'normalized_exit_code':30,'log':str(log) if log else None,
     'detail':sys.argv[5],'log_tail':tail,'created_unix':time.time(),'test_roots_read':False}
(out/'TASK_FAILED.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
PY_FAIL
}

run_task() {
  local group="$1" variant="$2" gpu="$3" prior_mode="$4" boundary_weight="$5" best_metric="$6"
  local out="$ROOT/tasks/${group}_${variant}"
  local run="$out/candidates/$variant"
  local source="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  local factor_cache="$MAIN_RUN/candidates/$variant/factor_stage"
  rm -rf "$out"; mkdir -p "$out/logs"
  [[ -f "$source" ]] || { write_task_failed "$out" source 30 "" "missing source checkpoint $source"; return 30; }
  [[ -f "$factor_cache/model_v48_trac_sr/best.pt" ]] || {
    write_task_failed "$out" factor_cache 30 "" "missing main-run factor cache $factor_cache"; return 30;
  }

  set +e
  RUN="$run" INIT_CKPT="$source" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" \
  GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="$VAL_GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
  NUM_WORKERS="${NUM_WORKERS:-4}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" BATCH_SIZE="${BATCH_SIZE:-96}" \
  V4834_ENABLE_SUPPORT_RELIABILITY=1 V4834_IDENTITY_TRAIN_ALL=1 \
  V4834_COUPLE_ADMISSION_PRIOR=1 V4834_ADAPTIVE_IDENTITY_MARGIN=0 \
  V4834_ENABLE_FINAL_CALIBRATION=0 V4834_FACTOR_CACHE_RUN="$factor_cache" \
  EVIDENCE_ADMISSION_PRIOR_MODE="$prior_mode" EVIDENCE_SLACK_PENALTY="${EVIDENCE_SLACK_PENALTY:-1.5}" \
  IDENTITY_ELIGIBLE_POLICY_WEIGHT=1.25 \
  IDENTITY_ELIGIBILITY_BOUNDARY_WEIGHT="$boundary_weight" \
  IDENTITY_ELIGIBILITY_BOUNDARY_MARGIN="${IDENTITY_ELIGIBILITY_BOUNDARY_MARGIN:-0.20}" \
  IDENTITY_BEST_METRIC="$best_metric" \
  IDENTITY_EPOCHS="${IDENTITY_EPOCHS:-24}" IDENTITY_PATIENCE="${IDENTITY_PATIENCE:-6}" \
  IDENTITY_LR="${IDENTITY_LR:-0.00004}" \
  IDENTITY_POSITIVE_MACRO_BALANCE_POWER="${IDENTITY_POSITIVE_MACRO_BALANCE_POWER:-0.50}" \
  IDENTITY_SCENE_BALANCE_POWER="${IDENTITY_SCENE_BALANCE_POWER:-0.50}" \
  PROPOSAL_TOP_K=5 EVIDENCE_COMPONENT_COUNT=5 EVIDENCE_COMPONENT_SCALE="${EVIDENCE_COMPONENT_SCALE:-6.0}" \
    bash scripts/adapt_ocrap_v48_34_barrier_crossfit_variant.sh >"$out/logs/adapt.log" 2>&1
  local adapt_rc=$?
  set -e
  if [[ "$adapt_rc" != 0 ]]; then
    write_task_failed "$out" adaptation "$adapt_rc" "$out/logs/adapt.log" "v48.34 ablation training failed"
    return 30
  fi

  local boundary_expected=false
  awk "BEGIN{exit !($boundary_weight > 0)}" && boundary_expected=true || true
  set +e
  python tools/check_v48_34_training_contract.py --run "$run" \
    --output "$run/TRAINING_CONTRACT.json" \
    --expect-identity-all true --expect-prior-coupled true \
    --expect-adaptive-margin false --expect-final-enabled false \
    --expect-eligible-policy true --expect-boundary "$boundary_expected" \
    --expect-prior-mode "$prior_mode" --expect-context-source relative \
    --expect-best-metric "$best_metric" --expect-proposal-top-k 5 \
    >"$out/logs/training_contract.log" 2>&1
  local train_rc=$?
  set -e
  if [[ "$train_rc" != 0 ]]; then
    write_task_failed "$out" training_contract "$train_rc" "$out/logs/training_contract.log" "$run/TRAINING_CONTRACT.json"
    return 30
  fi

  set +e
  python tools/check_v48_34_model_contract.py \
    --checkpoint "$run/model_v48_trac_sr/best.pt" --support-contract "$run/FACTOR_SUPPORT_CONTRACT.json" \
    --output "$run/MODEL_INFERENCE_CONTRACT.json" --expect-frontier true --expect-admission-bounded true \
    --expect-component-count 5 --expect-component-scale "${EVIDENCE_COMPONENT_SCALE:-6.0}" \
    --expect-admission-prior-detach any --expect-admission-prior-mode "$prior_mode" \
    --expect-slack-temperature 0.025 --expect-slack-penalty "${EVIDENCE_SLACK_PENALTY:-1.5}" \
    >"$out/logs/model_contract.log" 2>&1
  local model_rc=$?
  set -e
  if [[ "$model_rc" != 0 ]]; then
    write_task_failed "$out" model_contract "$model_rc" "$out/logs/model_contract.log" "$run/MODEL_INFERENCE_CONTRACT.json"
    return 30
  fi

  set +e
  OUTPUTDIR="$out" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" \
  DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" GPU0="$gpu" GPU1="$gpu" VARIANTS="$variant" \
  PROPOSAL_TOP_K=5 OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit \
    bash scripts/calibrate_v48_34_certificate_pool.sh >"$out/logs/certificate.log" 2>&1
  local cert_rc=$?
  set -e
  if [[ "$cert_rc" != 0 && "$cert_rc" != 20 ]]; then
    write_task_failed "$out" certificate "$cert_rc" "$out/logs/certificate.log" "certificate artifact/protocol failure"
    return 30
  fi

  python - "$out" "$group" "$variant" "$cert_rc" "$prior_mode" "$boundary_weight" "$best_metric" <<'PY_COMPLETE'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); group=sys.argv[2]; variant=sys.argv[3]; rc=int(sys.argv[4])
prior=sys.argv[5]; boundary=float(sys.argv[6]); best_metric=sys.argv[7]
base=out/'candidates'/variant; ckpt=base/'model_v48_trac_sr'/'best.pt'; cal=base/'calibration'
regimes={}
for regime in ('near','contact'):
    d=json.load(open(cal/f'direct_value_risk_{regime}_v48.json'))
    regimes[regime]={
      'valid':bool(d.get('valid_for_deployment',False)), 'rejection_kind':d.get('rejection_kind'),
      'candidate_positive_auc':d.get('candidate_positive_auc'),
      'candidate_safe_positive_auc':d.get('candidate_safe_positive_auc'),
      'legacy_evidence_only_top1_correlation':d.get('legacy_evidence_only_top1_correlation'),
      'proposal_exact_eligible_top1_correlation':d.get('proposal_exact_eligible_top1_correlation'),
      'proposal_exact_eligible_top1_safe_positive_auc':d.get('proposal_exact_eligible_top1_safe_positive_auc'),
      'proposal_exact_eligible_harmful_switch_rate':d.get('proposal_exact_eligible_harmful_switch_rate'),
      'verify':d.get('verify'),'oracle':d.get('proposal_constrained_oracle_gate')}
doc={'complete':True,'version':'v48.34-BARRIER-CROSSFIT','group':group,'variant':variant,
     'proposal_top_k':5,'identity_all_heads':True,'safe_utility_gradient_coupled':True,
     'admission_prior_mode':prior,'eligibility_boundary_weight':boundary,
     'best_metric':best_metric,'eligible_set_policy_weight':1.25,
     'adaptive_teacher_gap_margin':False,'final_admission_calibration':False,
     'factor_cache_source':'main_v48_34','certificate_exit':rc,'gate_passed':rc==0,
     'regimes':regimes,'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
     'created_unix':time.time(),'test_roots_read':False}
(out/'TASK_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
(out/'TASK_FAILED.json').unlink(missing_ok=True)
PY_COMPLETE
}

groups=(
  A_reference_soft_eligible
  B_add_barrier_gate
  C_add_hard_boundary_lexicographic
  D_full_barrier_boundary
)
prior_modes=(safety_slack barrier_gated_slack safety_slack barrier_gated_slack)
boundary_weights=(0.0 0.0 1.0 1.0)
best_metrics=(direct_contract_safe_rank_risk direct_contract_safe_rank_risk direct_contract_lexicographic direct_contract_lexicographic)

: >"$ROOT/TASK_GPU_ASSIGNMENT.txt"
failures=0
for i in "${!groups[@]}"; do
  group="${groups[$i]}"
  echo "${group}_balanced:gpu${GPU0}" >>"$ROOT/TASK_GPU_ASSIGNMENT.txt"
  echo "${group}_precision:gpu${GPU1}" >>"$ROOT/TASK_GPU_ASSIGNMENT.txt"
  run_task "$group" balanced "$GPU0" "${prior_modes[$i]}" "${boundary_weights[$i]}" "${best_metrics[$i]}" & p0=$!
  run_task "$group" precision "$GPU1" "${prior_modes[$i]}" "${boundary_weights[$i]}" "${best_metrics[$i]}" & p1=$!
  set +e
  wait "$p0"; r0=$?
  wait "$p1"; r1=$?
  set -e
  printf '%s_balanced=%s %s_precision=%s\n' "$group" "$r0" "$group" "$r1" | tee -a "$ROOT/logs/task_wait_status.log"
  [[ "$r0" == 0 ]] || failures=$((failures+1))
  [[ "$r1" == 0 ]] || failures=$((failures+1))
done

python - "$ROOT" "$failures" <<'PY_STATUS'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); failures=int(sys.argv[2])
groups=['A_reference_soft_eligible','B_add_barrier_gate',
        'C_add_hard_boundary_lexicographic','D_full_barrier_boundary']
expected=[f'{g}_{v}' for g in groups for v in ('balanced','precision')]
missing=[x for x in expected if not (root/'tasks'/x/'TASK_COMPLETE.json').is_file()]
failed={x:json.load(open(root/'tasks'/x/'TASK_FAILED.json')) for x in expected if (root/'tasks'/x/'TASK_FAILED.json').is_file()}
doc={'complete':not missing and failures==0,'version':'v48.34-BARRIER-CROSSFIT',
     'authorization':'main v48.34 pipeline valid and certificate RC=20',
     'design':'top5 and factor checkpoint fixed; isolate barrier gate and hard-boundary/lexicographic continuation; no adaptive margin; no stage3',
     'max_concurrent_tasks':2,'expected_tasks':expected,'missing_tasks':missing,
     'failed_waits':failures,'failures':failed,'created_unix':time.time(),'test_roots_read':False}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
if doc['complete']:
    (root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
else:
    raise SystemExit(30)
PY_STATUS
