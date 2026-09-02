#!/usr/bin/env bash
set -Eeuo pipefail

# Resume a v48.34 run that completed both adaptation variants but stopped at
# model_inference_contract because the old checker rejected the new
# barrier_gated_slack enum.  This script never retrains and refuses any other
# RC=30 signature.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

OUTPUTDIR="${OUTPUTDIR:?set OUTPUTDIR to the failed v48.34 dedicated run}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-5}"
STATUS="$OUTPUTDIR/V48_34_COMPLETE.json"
[[ -f "$STATUS" ]] || { echo "missing $STATUS" >&2; exit 30; }
mkdir -p "$OUTPUTDIR/logs"

# Establish that this is exactly the known parser failure and that all training
# products are complete and hash-consistent before certificate access.
python - "$OUTPUTDIR" <<'PY_PREFLIGHT'
import hashlib,json,pathlib,shutil,sys,time
root=pathlib.Path(sys.argv[1]); status_path=root/'V48_34_COMPLETE.json'
status=json.load(open(status_path))
fail=json.load(open(root/'PIPELINE_FAILED.json')) if (root/'PIPELINE_FAILED.json').is_file() else {}
reasons=[]
if int(status.get('pipeline_exit_code',-1)) != 30: reasons.append('pipeline_exit_code_not_30')
if status.get('failure_stage') != 'model_inference_contract': reasons.append('unexpected_failure_stage')
if fail.get('stage') != 'model_inference_contract': reasons.append('missing_matching_pipeline_failure')
if (fail.get('adaptation_exit_codes') or {}).get('balanced') != 0: reasons.append('balanced_adaptation_not_complete')
if (fail.get('adaptation_exit_codes') or {}).get('precision') != 0: reasons.append('precision_adaptation_not_complete')
variants={}
for name in ('balanced','precision'):
    run=root/'candidates'/name
    ckpt=run/'model_v48_trac_sr'/'best.pt'
    required=[run/'TRAINING_COMPLETE.json',run/'THREE_STAGE_TRAINING_COMPLETE.json',run/'STAGE_TRANSFER_INTEGRITY.json',run/'FACTOR_SUPPORT_CONTRACT.json',run/'POLICY_CONTRACT.env']
    missing=[str(p) for p in [ckpt,*required] if not p.is_file()]
    if missing:
        reasons.append(f'{name}_missing_artifacts:{missing}')
        continue
    actual=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    docs=[]
    for p in (run/'TRAINING_COMPLETE.json',run/'EVIDENCE_CORRECTION_COMPLETE.json'):
        if p.is_file(): docs.append((p,json.load(open(p))))
    for p,d in docs:
        expected=d.get('checkpoint_sha256')
        if expected and expected != actual:
            reasons.append(f'{name}_checkpoint_hash_mismatch:{p.name}')
    integrity=json.load(open(run/'STAGE_TRANSFER_INTEGRITY.json'))
    if not integrity.get('valid',False): reasons.append(f'{name}_stage_transfer_invalid')
    variants[name]={'checkpoint':str(ckpt),'sha256':actual}
if reasons:
    doc={'event':'v48_34_1_rc30_repair_preflight','created_unix':time.time(),'valid':False,'reasons':reasons,'source_status':status}
    (root/'V48_34_1_REPAIR_PREFLIGHT.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    raise SystemExit(30)
if (root/'PIPELINE_FAILED.json').is_file() and not (root/'PIPELINE_FAILED.v48_34_original.json').exists():
    shutil.copy2(root/'PIPELINE_FAILED.json',root/'PIPELINE_FAILED.v48_34_original.json')
doc={'event':'v48_34_1_rc30_repair_preflight','created_unix':time.time(),'valid':True,'known_failure':'old_checker_argparse_rejected_barrier_gated_slack','variants':variants,'source_status':status}
(root/'V48_34_1_REPAIR_PREFLIGHT.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(doc,ensure_ascii=False))
PY_PREFLIGHT

for variant in balanced precision; do
  run="$OUTPUTDIR/candidates/$variant"
  ckpt="$run/model_v48_trac_sr/best.pt"
  python tools/check_v48_34_model_contract.py \
    --checkpoint "$ckpt" --support-contract "$run/FACTOR_SUPPORT_CONTRACT.json" \
    --output "$run/MODEL_INFERENCE_CONTRACT.json" \
    --expect-frontier true --expect-admission-bounded true \
    --expect-component-prior-logit -2.0 --expect-component-count 5 \
    --expect-component-scale "${EVIDENCE_COMPONENT_SCALE:-6.0}" \
    --expect-admission-prior-detach any \
    --expect-admission-prior-mode barrier_gated_slack \
    --expect-slack-temperature "${EVIDENCE_SLACK_TEMPERATURE:-0.025}" \
    --expect-slack-penalty "${EVIDENCE_SLACK_PENALTY:-1.5}" \
    >"$OUTPUTDIR/logs/model_contract_${variant}.v48_34_1.log" 2>&1
  python tools/check_v48_34_training_contract.py \
    --run "$run" --output "$run/TRAINING_CONTRACT.json" \
    --expect-identity-all true --expect-prior-coupled true \
    --expect-adaptive-margin false --expect-final-enabled false \
    --expect-eligible-policy true --expect-boundary true \
    --expect-prior-mode barrier_gated_slack --expect-context-source relative \
    --expect-best-metric direct_contract_lexicographic \
    --expect-proposal-top-k "$PROPOSAL_TOP_K" \
    >"$OUTPUTDIR/logs/training_contract_${variant}.v48_34_1.log" 2>&1
done

# Resolve the original data contract. Environment overrides remain possible,
# but the default protocol root comes from the failed run status.
eval "$(python - "$STATUS" "$OCRAP_ROOT" <<'PY_ENV'
import json,shlex,sys
s=json.load(open(sys.argv[1])); root=s.get('protocol_root') or f'{sys.argv[2]}/calibration_v48_14_prism_4814'
source=s.get('source_run') or 'runs/ocrap_v48_13_terra_proxy_4801'
for k,v in [('PROTOCOL_ROOT',root),('SOURCE_RUN',source)]: print(f'{k}={shlex.quote(str(v))}')
PY_ENV
)"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
DEV_NEAR="${DEV_NEAR:-$PROTOCOL_ROOT/evidence_adapt_dev_near_contact}"
DEV_CONTACT="${DEV_CONTACT:-$PROTOCOL_ROOT/evidence_adapt_dev_contact}"
CERT_NEAR="${CERT_NEAR:-$PROTOCOL_ROOT/certificate_pool_near_contact}"
CERT_CONTACT="${CERT_CONTACT:-$PROTOCOL_ROOT/certificate_pool_contact}"

rm -f "$OUTPUTDIR/PIPELINE_FAILED.json" "$OUTPUTDIR/NEXT_COMMANDS.txt" \
      "$OUTPUTDIR/NEXT_COMMANDS_STATUS.json" "$OUTPUTDIR/NEXT_COMMANDS_BLOCKED.json" \
      "$OUTPUTDIR/GATE_FAILED.json" "$OUTPUTDIR/CALIBRATION_FAILED.json"

set +e
OUTPUTDIR="$OUTPUTDIR" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" \
DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" GPU0="$GPU0" GPU1="$GPU1" \
VARIANTS=balanced,precision PROPOSAL_TOP_K="$PROPOSAL_TOP_K" \
OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit \
  bash scripts/calibrate_v48_34_certificate_pool.sh \
  >"$OUTPUTDIR/logs/certificate_controller.v48_34_1.log" 2>&1
raw_cert_rc=$?
set -e
case "$raw_cert_rc" in 0|20) cert_rc="$raw_cert_rc" ;; *) cert_rc=30 ;; esac

python tools/check_v48_16_learning_gates.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/learning_gates_v48_34.json" --version v48.34.1-RC30-MODEL-CONTRACT-HOTFIX || true
python tools/summarize_v48_34_gate_failure.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/GATE_FAILURE_DECOMPOSITION.json" || true

python - "$OUTPUTDIR" "$PROTOCOL_ROOT" "$SOURCE_RUN" "$raw_cert_rc" "$cert_rc" <<'PY_COMPLETE'
import hashlib,json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); protocol=pathlib.Path(sys.argv[2]); source=pathlib.Path(sys.argv[3])
raw_rc=int(sys.argv[4]); rc=int(sys.argv[5]); variants={}
for name in ('balanced','precision'):
    p=root/'candidates'/name/'model_v48_trac_sr'/'best.pt'
    if p.is_file(): variants[name]={'checkpoint':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
next_exists=(root/'NEXT_COMMANDS.txt').is_file(); blocked_exists=(root/'NEXT_COMMANDS_BLOCKED.json').is_file()
consistent=(rc==0 and next_exists and not blocked_exists) or (rc==20 and not next_exists and blocked_exists)
if rc==30: consistent=blocked_exists or not next_exists
if not consistent: raise SystemExit(f'certificate/NEXT contract mismatch rc={rc} next={next_exists} blocked={blocked_exists}')
doc={'event':'v48_34_1_model_contract_hotfix_complete','created_unix':time.time(),
     'algorithm_version':'v48.34-BARRIER-CROSSFIT','engineering_release':'v48.34.1-RC30-MODEL-CONTRACT-HOTFIX',
     'source_run':str(source),'protocol_root':str(protocol),'variants':variants,
     'raw_certificate_exit_code':raw_rc if rc != 30 else raw_rc,
     'certificate_exit_code':rc if rc in (0,20) else None,'pipeline_exit_code':rc,
     'certificate_controller_invoked':True,'certificate_executed':rc in (0,20),'gate_evaluated':rc in (0,20),
     'gate_passed':rc==0 and next_exists,'next_commands_generated':next_exists,
     'pipeline_valid':rc in (0,20),'failure_stage':None if rc in (0,20) else 'certificate',
     'adaptation_reused_without_retraining':True,'test_roots_read':False}
for name in ('V48_34_1_COMPLETE.json','V48_34_COMPLETE.json'):
    (root/name).write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
repair={'event':'v48_34_1_repair_provenance','created_unix':time.time(),
        'known_failure_repaired':'argparse_prior_mode_enum','adaptation_reused':True,
        'certificate_controller_invoked':True,'certificate_raw_exit_code':raw_rc,'pipeline_exit_code':rc,'test_roots_read':False}
(root/'RC30_REPAIR_PROVENANCE.json').write_text(json.dumps(repair,ensure_ascii=False,indent=2)+'\n')
if rc==30:
    failed={'event':'v48_34_1_pipeline_failed','created_unix':time.time(),'stage':'certificate','raw_exit_code':raw_rc,
            'normalized_exit_code':30,'pipeline_exit_code':30,'certificate_controller_invoked':True,'certificate_executed':False,'gate_evaluated':False,
            'pipeline_valid':False,'test_roots_read':False}
    (root/'PIPELINE_FAILED.json').write_text(json.dumps(failed,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(doc,ensure_ascii=False))
PY_COMPLETE

exit "$cert_rc"
