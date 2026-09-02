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

MAIN_RUN="${MAIN_RUN:-runs/ocrap_v48_33_eligible_set_policy_dedicated_4833}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_33_eligible_set_policy_ablations_4833}"
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

python - "$MAIN_RUN/V48_33_COMPLETE.json" <<'PY_AUTH'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f'missing main-run completion contract: {p}')
d=json.load(open(p))
if not bool(d.get('pipeline_valid', False)) or not bool(d.get('certificate_executed', False)) or not bool(d.get('gate_evaluated', False)):
    raise SystemExit('ablations require a valid, evaluated v48.33 main pipeline')
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
doc={'complete':False,'event':'v48_33_ablation_task_failed','stage':sys.argv[2],
     'raw_exit_code':int(sys.argv[3]),'normalized_exit_code':30,'log':str(log) if log else None,
     'detail':sys.argv[5],'log_tail':tail,'created_unix':time.time(),'test_roots_read':False}
(out/'TASK_FAILED.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
PY_FAIL
}

run_task() {
  local group="$1" variant="$2" gpu="$3" identity_all="$4" coupled="$5" eligible_weight="$6"
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
  NUM_WORKERS="${NUM_WORKERS:-4}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" BATCH_SIZE="${BATCH_SIZE:-72}" \
  V4833_ENABLE_SUPPORT_RELIABILITY=1 V4833_IDENTITY_TRAIN_ALL="$identity_all" \
  V4833_COUPLE_ADMISSION_PRIOR="$coupled" V4833_ADAPTIVE_IDENTITY_MARGIN=0 \
  V4833_ENABLE_FINAL_CALIBRATION=0 V4833_FACTOR_CACHE_RUN="$factor_cache" \
  IDENTITY_ELIGIBLE_POLICY_WEIGHT="$eligible_weight" \
  IDENTITY_EPOCHS="${IDENTITY_EPOCHS:-24}" IDENTITY_PATIENCE="${IDENTITY_PATIENCE:-6}" \
  IDENTITY_LR="${IDENTITY_LR:-0.00006}" \
  PROPOSAL_TOP_K=5 EVIDENCE_COMPONENT_COUNT=5 EVIDENCE_COMPONENT_SCALE="${EVIDENCE_COMPONENT_SCALE:-6.0}" \
    bash scripts/adapt_ocrap_v48_33_eligible_set_policy_variant.sh >"$out/logs/adapt.log" 2>&1
  local adapt_rc=$?
  set -e
  if [[ "$adapt_rc" != 0 ]]; then
    write_task_failed "$out" adaptation "$adapt_rc" "$out/logs/adapt.log" "v48.33 ablation training failed"
    return 30
  fi

  local eligible_expected=false
  python - <<PY_BOOL >/dev/null
assert float("$eligible_weight") >= 0.0
PY_BOOL
  awk "BEGIN{exit !($eligible_weight > 0)}" && eligible_expected=true || true
  set +e
  python tools/check_v48_33_training_contract.py --run "$run" \
    --output "$run/TRAINING_CONTRACT.json" \
    --expect-identity-all "$([[ "$identity_all" == 1 ]] && echo true || echo false)" \
    --expect-prior-coupled "$([[ "$coupled" == 1 ]] && echo true || echo false)" \
    --expect-adaptive-margin false --expect-final-enabled false \
    --expect-eligible-policy "$eligible_expected" --expect-proposal-top-k 5 \
    >"$out/logs/training_contract.log" 2>&1
  local train_rc=$?
  set -e
  if [[ "$train_rc" != 0 ]]; then
    write_task_failed "$out" training_contract "$train_rc" "$out/logs/training_contract.log" "$run/TRAINING_CONTRACT.json"
    return 30
  fi

  set +e
  python tools/check_v48_32_model_contract.py \
    --checkpoint "$run/model_v48_trac_sr/best.pt" --support-contract "$run/FACTOR_SUPPORT_CONTRACT.json" \
    --output "$run/MODEL_INFERENCE_CONTRACT.json" --expect-frontier true --expect-admission-bounded true \
    --expect-component-count 5 --expect-component-scale "${EVIDENCE_COMPONENT_SCALE:-6.0}" \
    --expect-admission-prior-detach any --expect-admission-prior-mode safety_slack \
    --expect-slack-temperature 0.025 --expect-slack-penalty 1.0 \
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
    bash scripts/calibrate_v48_33_certificate_pool.sh >"$out/logs/certificate.log" 2>&1
  local cert_rc=$?
  set -e
  if [[ "$cert_rc" != 0 && "$cert_rc" != 20 ]]; then
    write_task_failed "$out" certificate "$cert_rc" "$out/logs/certificate.log" "certificate artifact/protocol failure"
    return 30
  fi

  python - "$out" "$group" "$variant" "$cert_rc" "$identity_all" "$coupled" "$eligible_weight" <<'PY_COMPLETE'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); group=sys.argv[2]; variant=sys.argv[3]; rc=int(sys.argv[4])
identity_all=bool(int(sys.argv[5])); coupled=bool(int(sys.argv[6])); eligible=float(sys.argv[7])
base=out/'candidates'/variant; ckpt=base/'model_v48_trac_sr'/'best.pt'; cal=base/'calibration'
regimes={}
for regime in ('near','contact'):
    d=json.load(open(cal/f'direct_value_risk_{regime}_v48.json'))
    regimes[regime]={
      'valid':bool(d.get('valid_for_deployment',False)), 'rejection_kind':d.get('rejection_kind'),
      'candidate_positive_auc':d.get('candidate_positive_auc'),
      'candidate_safe_positive_auc':d.get('candidate_safe_positive_auc'),
      'proposal_evidence_top1_correlation':d.get('proposal_evidence_top1_correlation'),
      'proposal_evidence_top1_safe_positive_auc':d.get('proposal_evidence_top1_safe_positive_auc'),
      'proposal_evidence_top1_harm_auc':d.get('proposal_evidence_top1_harm_auc'),
      'verify':d.get('verify'),'oracle':d.get('proposal_constrained_oracle_gate')}
doc={'complete':True,'version':'v48.33-ELIGIBLE-SET-POLICY','group':group,'variant':variant,
     'proposal_top_k':5,'identity_all_heads':identity_all,'safe_utility_gradient_coupled':coupled,
     'eligible_set_policy_weight':eligible,'adaptive_teacher_gap_margin':False,
     'final_admission_calibration':False,'factor_cache_source':'main_v48_33',
     'certificate_exit':rc,'gate_passed':rc==0,'regimes':regimes,
     'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
     'created_unix':time.time(),'test_roots_read':False}
(out/'TASK_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
(out/'TASK_FAILED.json').unlink(missing_ok=True)
PY_COMPLETE
}

groups=(
  A_top5_admission_only_no_eligible_policy
  B_top5_joint_coupled_no_eligible_policy
  C_top5_admission_only_eligible_policy
  D_full_top5_eligible_set_policy
)
identity_flags=(0 1 0 1)
coupled_flags=(0 1 0 1)
eligible_weights=(0.0 0.0 1.25 1.25)

: >"$ROOT/TASK_GPU_ASSIGNMENT.txt"
failures=0
for i in "${!groups[@]}"; do
  group="${groups[$i]}"
  echo "${group}_balanced:gpu${GPU0}" >>"$ROOT/TASK_GPU_ASSIGNMENT.txt"
  echo "${group}_precision:gpu${GPU1}" >>"$ROOT/TASK_GPU_ASSIGNMENT.txt"
  run_task "$group" balanced "$GPU0" "${identity_flags[$i]}" "${coupled_flags[$i]}" "${eligible_weights[$i]}" & p0=$!
  run_task "$group" precision "$GPU1" "${identity_flags[$i]}" "${coupled_flags[$i]}" "${eligible_weights[$i]}" & p1=$!
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
groups=['A_top5_admission_only_no_eligible_policy','B_top5_joint_coupled_no_eligible_policy',
        'C_top5_admission_only_eligible_policy','D_full_top5_eligible_set_policy']
expected=[f'{g}_{v}' for g in groups for v in ('balanced','precision')]
missing=[x for x in expected if not (root/'tasks'/x/'TASK_COMPLETE.json').is_file()]
failed={x:json.load(open(root/'tasks'/x/'TASK_FAILED.json')) for x in expected if (root/'tasks'/x/'TASK_FAILED.json').is_file()}
doc={'complete':not missing and failures==0,'version':'v48.33-ELIGIBLE-SET-POLICY',
     'authorization':'main v48.33 pipeline valid and certificate RC=20',
     'design':'top5 fixed; isolate exact eligible-set loss and multi-head coupling; no adaptive margin; no stage3',
     'max_concurrent_tasks':2,'expected_tasks':expected,'missing_tasks':missing,
     'failed_waits':failures,'failures':failed,'created_unix':time.time(),'test_roots_read':False}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
if doc['complete']:
    (root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
else:
    raise SystemExit(30)
PY_STATUS
