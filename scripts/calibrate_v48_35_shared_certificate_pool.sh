#!/usr/bin/env bash
set -Eeuo pipefail

# Finalise a v48.35 CONTINUOUS-FRONTIER run on the scene-disjoint dedicated
# certificate pool. Thresholds are fitted only on adaptation-dev and the exact
# preregistered fit/verify contract is checked before certificate access.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

OUTPUTDIR="${OUTPUTDIR:?OUTPUTDIR is required}"
CAL_SAFE="${CAL_SAFE:?CAL_SAFE is required}"
CERT_NEAR="${CERT_NEAR:?CERT_NEAR is required}"
CERT_CONTACT="${CERT_CONTACT:?CERT_CONTACT is required}"
DEV_NEAR="${DEV_NEAR:?DEV_NEAR is required}"
DEV_CONTACT="${DEV_CONTACT:?DEV_CONTACT is required}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
ATTEMPT_ID="${V4835_ATTEMPT_ID:-legacy-untracked}"
SAFE_WOMD_SOURCE="${SAFE_WOMD_SOURCE:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
export V4835_ATTEMPT_ID="$ATTEMPT_ID" SAFE_WOMD_SOURCE
for d in "$CAL_SAFE" "$CERT_NEAR" "$CERT_CONTACT" "$DEV_NEAR" "$DEV_CONTACT"; do
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing certificate dataset $d" >&2; exit 2; }
done
mkdir -p "$OUTPUTDIR/logs"
rm -f "$OUTPUTDIR/NEXT_COMMANDS.txt" "$OUTPUTDIR/NEXT_COMMANDS_STATUS.json" \
      "$OUTPUTDIR/NEXT_COMMANDS_BLOCKED.json" "$OUTPUTDIR/GATE_FAILED.json" \
      "$OUTPUTDIR/CALIBRATION_FAILED.json" "$OUTPUTDIR/chosen_base_run_dedicated.txt"
: > "$OUTPUTDIR/logs/v48_35_certificate_status.log"

# Freeze the statistical protocol before reading certificate scores. Rerunning
# the same output directory with changed bounds/counts is rejected unless the
# operator explicitly starts a new run directory.
export CERTIFICATE_CONFIDENCE_LEVEL="${CERTIFICATE_CONFIDENCE_LEVEL:-0.90}"
export CERTIFICATE_BOUND_TYPE="${CERTIFICATE_BOUND_TYPE:-one_sided}"
export PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-5}"
export POSITIVE_GAIN="${POSITIVE_GAIN:-0.015}"
export NEGATIVE_GAIN="${NEGATIVE_GAIN:-0.010}"
export HARM_LABEL_MODE="${HARM_LABEL_MODE:-component_veto}"
export OPPORTUNITY_LABEL_MODE="${OPPORTUNITY_LABEL_MODE:-raw_benefit}"
export GATE_POSITIVE_MODE="${GATE_POSITIVE_MODE:-safe_benefit}"
export COMPONENT_HARM_DRS_TOLERANCE="${COMPONENT_HARM_DRS_TOLERANCE:-0.05}"
export COMPONENT_HARM_DEP_TOLERANCE="${COMPONENT_HARM_DEP_TOLERANCE:-0.05}"
export COMPONENT_HARM_GAP_TOLERANCE="${COMPONENT_HARM_GAP_TOLERANCE:-0.05}"
export COMPONENT_HARM_HARD_TOLERANCE="${COMPONENT_HARM_HARD_TOLERANCE:-0.05}"
export COMPONENT_HARM_PROXY_TOLERANCE="${COMPONENT_HARM_PROXY_TOLERANCE:-0.05}"
export NEAR_MIN_FIT_SELECTED="${NEAR_MIN_FIT_SELECTED:-10}"
export NEAR_MIN_VERIFY_SELECTED="${NEAR_MIN_VERIFY_SELECTED:-8}"
export NEAR_MIN_FIT_PRECISION_LCB="${NEAR_MIN_FIT_PRECISION_LCB:-0.50}"
export NEAR_MIN_VERIFY_PRECISION_LCB="${NEAR_MIN_VERIFY_PRECISION_LCB:-0.40}"
export NEAR_MAX_FIT_HARM_UCB="${NEAR_MAX_FIT_HARM_UCB:-0.12}"
export NEAR_MAX_VERIFY_HARM_UCB="${NEAR_MAX_VERIFY_HARM_UCB:-0.14}"
export NEAR_MAX_FIT_SELECTED_HARM_UCB="${NEAR_MAX_FIT_SELECTED_HARM_UCB:-0.22}"
export NEAR_MAX_VERIFY_SELECTED_HARM_UCB="${NEAR_MAX_VERIFY_SELECTED_HARM_UCB:-0.25}"
export CONTACT_MIN_FIT_SELECTED="${CONTACT_MIN_FIT_SELECTED:-16}"
export CONTACT_MIN_VERIFY_SELECTED="${CONTACT_MIN_VERIFY_SELECTED:-10}"
export CONTACT_MIN_FIT_PRECISION_LCB="${CONTACT_MIN_FIT_PRECISION_LCB:-0.50}"
export CONTACT_MIN_VERIFY_PRECISION_LCB="${CONTACT_MIN_VERIFY_PRECISION_LCB:-0.40}"
export CONTACT_MAX_FIT_HARM_UCB="${CONTACT_MAX_FIT_HARM_UCB:-0.14}"
export CONTACT_MAX_VERIFY_HARM_UCB="${CONTACT_MAX_VERIFY_HARM_UCB:-0.16}"
export CONTACT_MAX_FIT_SELECTED_HARM_UCB="${CONTACT_MAX_FIT_SELECTED_HARM_UCB:-0.22}"
export CONTACT_MAX_VERIFY_SELECTED_HARM_UCB="${CONTACT_MAX_VERIFY_SELECTED_HARM_UCB:-0.25}"
python - "$OUTPUTDIR/GATE_SPEC.json" "$CAL_SAFE" "$CERT_NEAR" "$CERT_CONTACT" "$DEV_NEAR" "$DEV_CONTACT" <<'PY_GATE'
import hashlib,json,os,pathlib,sys,time
def atomic(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.tmp.{os.getpid()}.{time.time_ns()}')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(doc,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def f(name): return float(os.environ[name])
def i(name): return int(os.environ[name])
def dataset_record(role, value):
    root=pathlib.Path(value).resolve(); manifest=root/'manifest.csv'
    return {'role':role,'root':str(root),'manifest':str(manifest),
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None}
def bucket(prefix):
    return {
      'fit': {'min_selected':i(f'{prefix}_MIN_FIT_SELECTED'),
              'min_precision_lcb':f(f'{prefix}_MIN_FIT_PRECISION_LCB'),
              'max_harmful_group_ucb':f(f'{prefix}_MAX_FIT_HARM_UCB'),
              'max_harmful_selected_ucb':f(f'{prefix}_MAX_FIT_SELECTED_HARM_UCB')},
      'verify': {'min_selected':i(f'{prefix}_MIN_VERIFY_SELECTED'),
                 'min_precision_lcb':f(f'{prefix}_MIN_VERIFY_PRECISION_LCB'),
                 'max_harmful_group_ucb':f(f'{prefix}_MAX_VERIFY_HARM_UCB'),
                 'max_harmful_selected_ucb':f(f'{prefix}_MAX_VERIFY_SELECTED_HARM_UCB')},
    }
protocol={
 'version':'v48.35-CONTINUOUS-FRONTIER',
 'confidence':{'level':f('CERTIFICATE_CONFIDENCE_LEVEL'),'bound_type':os.environ['CERTIFICATE_BOUND_TYPE']},
 'benefit':{'positive_gain':f('POSITIVE_GAIN'),'opportunity_label_mode':os.environ['OPPORTUNITY_LABEL_MODE'],
            'gate_positive_mode':os.environ.get('GATE_POSITIVE_MODE','safe_benefit')},
 'policy':{'proposal_top_k':i('PROPOSAL_TOP_K'),'selection_semantics':'rank_topk_then_filter_then_evidence_rerank'},
 'harm':{'negative_gain_legacy':f('NEGATIVE_GAIN'),'label_mode':os.environ['HARM_LABEL_MODE'],
         'component_tolerances':{
           'drs':f('COMPONENT_HARM_DRS_TOLERANCE'),
           'deployability_gate':f('COMPONENT_HARM_DEP_TOLERANCE'),
           'gap_discount':f('COMPONENT_HARM_GAP_TOLERANCE'),
           'hard_violation':f('COMPONENT_HARM_HARD_TOLERANCE'),
           'harm_proxy':f('COMPONENT_HARM_PROXY_TOLERANCE')}},
 'near':bucket('NEAR'),'contact':bucket('CONTACT'),
 'datasets':[dataset_record('safe_calibration',sys.argv[2]),
             dataset_record('near_certificate',sys.argv[3]),
             dataset_record('contact_certificate',sys.argv[4]),
             dataset_record('near_threshold_fit_dev',sys.argv[5]),
             dataset_record('contact_threshold_fit_dev',sys.argv[6])],
 'threshold_source':'pooled_evidence_adapt_dev_shared_rule','shared_deployment_rule':True,'strategy_regime_conditioning':False,
 'certificate_mode':'external_rule_full_verification',
 'certificate_labels_used_for_threshold_fit':False,
 'fit_verify_scene_disjoint':True,'test_roots_read':False,
}
canonical=json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()
doc={'event':'v48_35_gate_protocol_preregistered','version':'v48.35.2-ENGINEERING-INTEGRITY','created_unix':time.time(),
     'attempt_id':os.environ.get('V4835_ATTEMPT_ID'),'protocol_sha256':hashlib.sha256(canonical).hexdigest(),'protocol':protocol}
p=pathlib.Path(sys.argv[1])
if p.exists():
    old=json.load(open(p))
    if old.get('protocol') != protocol:
        raise SystemExit('GATE_SPEC.json already exists with a different protocol; use a new OUTPUTDIR')
atomic(p,doc)
PY_GATE

calibrate_variant() {
  local variant="$1"
  local gpu="$2"
  local run="$OUTPUTDIR/candidates/$variant"
  local ckpt="$run/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "skip missing variant $variant"; return 0; }
  local contract="$run/POLICY_CONTRACT.env"
  [[ -f "$contract" ]] || { echo "missing policy contract: $contract" >&2; return 3; }
  set -a
  # shellcheck disable=SC1090
  source "$contract"
  set +a
  local tmp="$run/calibration.v48_14.tmp.$$" final="$run/calibration"
  rm -rf "$tmp"; mkdir -p "$tmp" "$run/logs"
  local datasets=("$CERT_NEAR,$CERT_CONTACT" "$CAL_SAFE" "$CERT_NEAR" "$CERT_CONTACT")
  local buckets=(mix safe near contact)
  local mins=(180 80 100 140)
  local allowed=(certificate_pool calibration certificate_pool certificate_pool)
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m ocrap.cli calibrate \
      --dataset "${datasets[$i]}" --checkpoint "$ckpt" \
      --output "$tmp/calibration_${buckets[$i]}_v48.json" \
      --set calibration.required_min_for_delta="${mins[$i]}" \
      --set calibration.allowed_split_ids="${allowed[$i]}" \
      --set calibration.exact_split_ids=true \
      --set calibration.allow_validation_fallback=false \
      2>&1 | tee "$run/logs/v48_35_calibrate_${buckets[$i]}.log"
  done
  python tools/write_gamma_by_bucket.py \
    --safe "$tmp/calibration_safe_v48.json" \
    --near "$tmp/calibration_near_v48.json" \
    --contact "$tmp/calibration_contact_v48.json" \
    --delta 0.05 --output "$tmp/gamma_rec_by_bucket_v48.json" \
    2>&1 | tee "$run/logs/v48_35_gamma.log"

  local common=(
    --checkpoint "$ckpt" --method-version=v48_35_continuous_frontier_dev_frozen_policy_risk_certificate --risk-source="${RISK_SOURCE:-ordinal_evidence}"
    --conditional-recovery-ranking --proposal-top-k "${PROPOSAL_TOP_K:-5}" --evidence-rerank-top-k
    --macro-constraint-mode="${MACRO_CONSTRAINT_MODE:-opportunity_normalized}"
    --positive-gain="$POSITIVE_GAIN" --negative-gain="$NEGATIVE_GAIN"
    --max-selected-macro-share="${MAX_SELECTED_MACRO_SHARE:-0.95}"
    --max-macro-excess-share="${MAX_MACRO_EXCESS_SHARE:-0.15}"
    --harm-label-mode="$HARM_LABEL_MODE"
    --opportunity-label-mode="$OPPORTUNITY_LABEL_MODE"
    --gate-positive-mode="$GATE_POSITIVE_MODE"
    --certificate-confidence-level="$CERTIFICATE_CONFIDENCE_LEVEL"
    --certificate-bound-type="$CERTIFICATE_BOUND_TYPE"
    --grid-size="${CERTIFICATE_GRID_SIZE:-31}"
    --component-harm-drs-tolerance="$COMPONENT_HARM_DRS_TOLERANCE"
    --component-harm-dep-tolerance="$COMPONENT_HARM_DEP_TOLERANCE"
    --component-harm-gap-tolerance="$COMPONENT_HARM_GAP_TOLERANCE"
    --component-harm-hard-tolerance="$COMPONENT_HARM_HARD_TOLERANCE"
    --component-harm-proxy-tolerance="$COMPONENT_HARM_PROXY_TOLERANCE"
    --max-hard="${POLICY_METRIC_MAX_HARD:-1.0}"
    --min-nominal-deviation="${POLICY_METRIC_MIN_NOMINAL_DEVIATION:-0.002}"
  )
  set +e
  # Extract the exact top-k proposal rows separately for audit only.  These
  # per-stratum jobs do not authorize deployment thresholds.
  CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
    --dataset "$DEV_NEAR" --allowed-splits=evidence_adapt_dev --bucket near "${common[@]}" \
    --development-fit-only --output "$tmp/dev_diagnostic_near_v48.json" --proposal-rows-output "$tmp/dev_diagnostic_near_v48.proposal_rows.jsonl" \
    --required-min-groups=1 --required-min-scenes=1 \
    --min-fit-selected=1 --min-fit-precision-lcb=0 \
    --max-fit-harmful-group-ucb=1 --max-fit-harmful-selected-ucb=1 \
    2>&1 | tee "$run/logs/v48_35_dev_diagnostic_near.log"; local dn=${PIPESTATUS[0]}
  CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
    --dataset "$DEV_CONTACT" --allowed-splits=evidence_adapt_dev --bucket contact "${common[@]}" \
    --development-fit-only --output "$tmp/dev_diagnostic_contact_v48.json" --proposal-rows-output "$tmp/dev_diagnostic_contact_v48.proposal_rows.jsonl" \
    --required-min-groups=1 --required-min-scenes=1 \
    --min-fit-selected=1 --min-fit-precision-lcb=0 \
    --max-fit-harmful-group-ucb=1 --max-fit-harmful-selected-ucb=1 \
    2>&1 | tee "$run/logs/v48_35_dev_diagnostic_contact.log"; local dc=${PIPESTATUS[0]}
  if [[ ( "$dn" != 0 && "$dn" != 3 ) || ( "$dc" != 0 && "$dc" != 3 ) ]]; then
    set -e
    echo "adaptation-dev proposal extraction failed: near=$dn contact=$dc" >&2
    return 30
  fi

  # One rule is fitted over the pooled adaptation-dev population. Near/Contact
  # names are audit strata only and can constrain the worst stratum, but cannot
  # select different policy parameters. RC=3 here is a valid algorithm rejection;
  # the diagnostic shared rule is still frozen for a non-authorizing audit.
  python -u tools/calibrate_shared_continuous_rule_v48_35.py \
    --stratum "near=$tmp/dev_diagnostic_near_v48.proposal_rows.jsonl" \
    --stratum "contact=$tmp/dev_diagnostic_contact_v48.proposal_rows.jsonl" \
    --output "$tmp/dev_frozen_shared_rule_v48.json" \
    --grid-size="${SHARED_RULE_GRID_SIZE:-15}" \
    --positive-gain="$POSITIVE_GAIN" --negative-gain="$NEGATIVE_GAIN" \
    --confidence-level="$CERTIFICATE_CONFIDENCE_LEVEL" \
    --min-selected="near=$NEAR_MIN_FIT_SELECTED,contact=$CONTACT_MIN_FIT_SELECTED" \
    --min-precision-lcb="near=$NEAR_MIN_FIT_PRECISION_LCB,contact=$CONTACT_MIN_FIT_PRECISION_LCB" \
    --max-harmful-group-ucb="near=$NEAR_MAX_FIT_HARM_UCB,contact=$CONTACT_MAX_FIT_HARM_UCB" \
    --max-harmful-selected-ucb="near=$NEAR_MAX_FIT_SELECTED_HARM_UCB,contact=$CONTACT_MAX_FIT_SELECTED_HARM_UCB" \
    --max-macro-share="${MAX_SELECTED_MACRO_SHARE:-0.95}" \
    2>&1 | tee "$run/logs/v48_35_shared_dev_rule.log"; local ds=${PIPESTATUS[0]}
  if [[ "$ds" != 0 && "$ds" != 3 ]]; then
    set -e
    echo "shared development-rule fitter failed as an artifact error: rc=$ds" >&2
    return 30
  fi
  set -e

  # Fail closed before certificate access if training, proposal extraction and
  # the preregistered one-rule protocol do not describe the same population.
  python tools/check_v48_35_metric_calibration_contract.py \
    --train-summary "$run/model_v48_trac_sr/train_summary.json" \
    --near-dev "$tmp/dev_diagnostic_near_v48.json" \
    --contact-dev "$tmp/dev_diagnostic_contact_v48.json" \
    --shared-rule "$tmp/dev_frozen_shared_rule_v48.json" \
    --gate-spec "$OUTPUTDIR/GATE_SPEC.json" \
    --policy-contract "$run/POLICY_CONTRACT.env" \
    --output "$tmp/METRIC_CALIBRATION_CONTRACT.json"

  set +e
  # The complete dedicated certificate is verification-only. Both workers read
  # the byte-identical shared rule. No certificate label can alter it.
  CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
    --dataset "$CERT_NEAR" --allowed-splits=certificate_pool --bucket near "${common[@]}" \
    --verification-only --frozen-rule-json "$tmp/dev_frozen_shared_rule_v48.json" \
    --output "$tmp/direct_value_risk_near_v48.json" --rows-output "$tmp/direct_value_risk_near_v48.rows.jsonl" --proposal-rows-output "$tmp/direct_value_risk_near_v48.proposal_rows.jsonl" \
    --required-min-groups="${NEAR_MIN_GROUPS:-120}" --required-min-scenes="${NEAR_MIN_SCENES:-60}" \
    --min-verify-selected="$NEAR_MIN_VERIFY_SELECTED" --min-verify-precision-lcb="$NEAR_MIN_VERIFY_PRECISION_LCB" \
    --max-verify-harmful-group-ucb="$NEAR_MAX_VERIFY_HARM_UCB" \
    --max-verify-harmful-selected-ucb="$NEAR_MAX_VERIFY_SELECTED_HARM_UCB" \
    2>&1 | tee "$run/logs/v48_35_policy_near.log"; local sn=${PIPESTATUS[0]}
  CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
    --dataset "$CERT_CONTACT" --allowed-splits=certificate_pool --bucket contact "${common[@]}" \
    --verification-only --frozen-rule-json "$tmp/dev_frozen_shared_rule_v48.json" \
    --output "$tmp/direct_value_risk_contact_v48.json" --rows-output "$tmp/direct_value_risk_contact_v48.rows.jsonl" --proposal-rows-output "$tmp/direct_value_risk_contact_v48.proposal_rows.jsonl" \
    --required-min-groups="${CONTACT_MIN_GROUPS:-180}" --required-min-scenes="${CONTACT_MIN_SCENES:-80}" \
    --min-verify-selected="$CONTACT_MIN_VERIFY_SELECTED" --min-verify-precision-lcb="$CONTACT_MIN_VERIFY_PRECISION_LCB" \
    --max-verify-harmful-group-ucb="$CONTACT_MAX_VERIFY_HARM_UCB" \
    --max-verify-harmful-selected-ucb="$CONTACT_MAX_VERIFY_SELECTED_HARM_UCB" \
    2>&1 | tee "$run/logs/v48_35_policy_contact.log"; local sc=${PIPESTATUS[0]}
  set -e

  python - "$tmp" "$variant" "$ckpt" "$sn" "$sc" <<'PY'
import hashlib,json,os,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); variant=sys.argv[2]; ckpt=pathlib.Path(sys.argv[3])
near_rc,contact_rc=int(sys.argv[4]),int(sys.argv[5]); attempt_id=os.environ.get('V4835_ATTEMPT_ID')
def atomic(path,doc):
    tmp=path.with_name(f'.{path.name}.tmp.{os.getpid()}.{time.time_ns()}')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(doc,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
required=[
 'calibration_mix_v48.json','calibration_safe_v48.json','calibration_near_v48.json','calibration_contact_v48.json',
 'gamma_rec_by_bucket_v48.json','dev_diagnostic_near_v48.json','dev_diagnostic_contact_v48.json','dev_frozen_shared_rule_v48.json',
 'METRIC_CALIBRATION_CONTRACT.json',
 'direct_value_risk_near_v48.json','direct_value_risk_contact_v48.json',
]
missing=[x for x in required if not (root/x).is_file()]
if missing: raise SystemExit('missing final calibration artifacts: '+','.join(missing))
for name in ('mix','safe','near','contact'):
    d=json.load(open(root/f'calibration_{name}_v48.json'))
    if int(d.get('num_samples',0) or 0) <= 0:
        raise SystemExit(f'empty standard calibration dataset for {name}; splits={d.get("splits")} allowed={d.get("allowed_split_ids")}')
for name,rc in (('near',near_rc),('contact',contact_rc)):
    d=json.load(open(root/f'direct_value_risk_{name}_v48.json'))
    if int(d.get('num_groups',0) or 0) <= 0 or int(d.get('num_scenes',0) or 0) <= 0:
        raise SystemExit(f'empty policy certificate for {name}; rc={rc}; kept={d.get("kept_split_counts")} allowed={d.get("allowed_split_ids")}')
    if d.get('certificate_mode') != 'external_rule_full_verification':
        raise SystemExit(f'unexpected certificate mode for {name}: {d.get("certificate_mode")}')
    if int(d.get('fit_groups',0) or 0) != 0 or int(d.get('verify_groups',0) or 0) <= 0:
        raise SystemExit(f'certificate must be full verification with no internal fit for {name}')
    source=d.get('frozen_rule_source') or {}
    if not source.get('sha256'):
        raise SystemExit(f'missing frozen adaptation-dev rule provenance for {name}')
    shared_sha=hashlib.sha256((root/'dev_frozen_shared_rule_v48.json').read_bytes()).hexdigest()
    if source.get('sha256') != shared_sha:
        raise SystemExit(f'{name} did not use the byte-identical shared rule')
    if rc not in (0,3,4):
        raise SystemExit(f'policy certificate process failed for {name}: rc={rc}')
    feasibility=d.get('certificate_support_feasibility') or {}
    if rc == 4 and bool(feasibility.get('overall', True)):
        raise SystemExit(f'policy certificate artifact/protocol failure for {name}: rc={rc}')
safe=json.load(open(root/'calibration_safe_v48.json'))
safe_status={
 'event':'v48_35_safe_regime_status','version':'v48.35.2-ENGINEERING-INTEGRITY','created_unix':time.time(),'attempt_id':attempt_id,'variant':variant,
 'standard_calibration_valid':int(safe.get('num_samples',0) or 0)>0,
 'num_samples':int(safe.get('num_samples',0) or 0),'num_negative':int(safe.get('num_negative',0) or 0),
 'gamma_rec':safe.get('gamma_rec'),'policy_natural_gate_evaluated':False,
 'reason':'no dedicated scene-disjoint Safe policy certificate population is registered; Safe is checked by calibrated recovery threshold plus paired non-inferiority closed loop',
 'test_roots_read':False}
atomic(root/'SAFE_REGIME_STATUS.json',safe_status)
certificate_data_valid=(near_rc in (0,3) and contact_rc in (0,3))
doc={'event':'v48_35_certificate_pool_calibration_complete','version':'v48.35.2-ENGINEERING-INTEGRITY','created_unix':time.time(),'attempt_id':attempt_id,
 'variant':variant,'checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
 'near_exit_code':near_rc,'contact_exit_code':contact_rc,
 'certificate_executed':True,'gate_evaluated':certificate_data_valid,
 'safe_policy_gate_evaluated':False,
 'certificate_data_valid':certificate_data_valid,'allowed_split_id':'certificate_pool','test_roots_read':False,
 'scene_roles_disjoint':True,'threshold_source':'pooled_evidence_adapt_dev_shared_rule','shared_deployment_rule':True,'strategy_regime_conditioning':False,
 'certificate_mode':'external_rule_full_verification',
 'opportunity_label_mode':os.environ.get('OPPORTUNITY_LABEL_MODE','raw_benefit'),
 'gate_positive_mode':os.environ.get('GATE_POSITIVE_MODE','safe_benefit')}
atomic(root/'CERTIFICATE_CALIBRATION_COMPLETE.json',doc)
PY
  rm -rf "$final.old"
  [[ ! -e "$final" ]] || mv "$final" "$final.old"
  mv "$tmp" "$final"
  rm -rf "$final.old"
  echo "$variant near=$sn contact=$sc" >> "$OUTPUTDIR/logs/v48_35_certificate_status.log"

  local view="$OUTPUTDIR/dedicated_candidates/$variant"
  rm -rf "$view"; mkdir -p "$view"
  ln -s "$(realpath --relative-to="$view" "$run/model_v48_trac_sr")" "$view/model_v48_trac_sr"
  ln -s "$(realpath --relative-to="$view" "$final")" "$view/calibration"
  ln -s "$(realpath --relative-to="$view" "$contract")" "$view/POLICY_CONTRACT.env"
  if [[ "$sn" == 0 && "$sc" == 0 ]]; then return 0; fi
  if [[ "$sn" == 4 || "$sc" == 4 ]]; then return 30; fi
  if [[ ( "$sn" == 0 || "$sn" == 3 ) && ( "$sc" == 0 || "$sc" == 3 ) ]]; then return 20; fi
  return 30
}

VARIANTS="${VARIANTS:-balanced,precision}"
S0=0; S1=0; P0=""; P1=""
if [[ ",$VARIANTS," == *,balanced,* ]]; then calibrate_variant balanced "$GPU0" & P0=$!; fi
if [[ ",$VARIANTS," == *,precision,* ]]; then calibrate_variant precision "$GPU1" & P1=$!; fi
set +e
if [[ -n "$P0" ]]; then wait "$P0"; S0=$?; fi
if [[ -n "$P1" ]]; then wait "$P1"; S1=$?; fi
set -e

python - "$OUTPUTDIR" "$S0" "$S1" "$VARIANTS" <<'PY'
import json,os,pathlib,shlex,sys,time
root=pathlib.Path(sys.argv[1]); report={}; valid=[]; attempt_id=os.environ.get('V4835_ATTEMPT_ID')
def atomic_json(path,doc):
    tmp=path.with_name(f'.{path.name}.tmp.{os.getpid()}.{time.time_ns()}')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(doc,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def atomic_text(path,text):
    tmp=path.with_name(f'.{path.name}.tmp.{os.getpid()}.{time.time_ns()}')
    with tmp.open('w',encoding='utf-8') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
requested={x.strip() for x in sys.argv[4].split(',') if x.strip()}
for name in ('balanced','precision'):
    if name not in requested: continue
    run=root/'candidates'/name; docs=[]; ok=True; report[name]={}
    for bucket in ('near','contact'):
        p=run/'calibration'/f'direct_value_risk_{bucket}_v48.json'
        try: d=json.load(open(p))
        except Exception as e:
            report[name][bucket]={'missing':str(e)}; ok=False; continue
        if int(d.get('num_groups',0) or 0) <= 0 or int(d.get('num_scenes',0) or 0) <= 0:
            report[name][bucket]={'artifact_error':'empty certificate data','num_groups':d.get('num_groups'),'num_scenes':d.get('num_scenes')}; ok=False; continue
        docs.append(d); ok &= bool(d.get('valid_for_deployment',False))
        report[name][bucket]={
          'valid':d.get('valid_for_deployment'),'candidate_auc':d.get('candidate_positive_auc'),
          'proposal_rank_top1_corr':d.get('unconstrained_group_top1_correlation'),
          'learned_evidence_top1_corr':d.get('proposal_evidence_top1_correlation'),
          'learned_evidence_positive_auc':d.get('proposal_evidence_top1_positive_auc'),
          'learned_evidence_harm_auc':d.get('proposal_evidence_top1_harm_auc'),
          'learned_evidence_false_switch_rate':d.get('proposal_evidence_nonpositive_false_switch_rate'),
          'learned_evidence_harmful_switch_rate':d.get('proposal_evidence_harmful_switch_rate'),
          'proposal_constrained_oracle_gate':d.get('proposal_constrained_oracle_gate'),
          'proposal_support_curve':d.get('proposal_support_curve'),
          'verify':d.get('verify'),
          'warnings':d.get('warnings'),
        }
    if ok:
        harm=sum(float((d.get('verify') or {}).get('harmful_selected_rate') or 0) for d in docs)
        adv=sum(float(((d.get('verify') or {}).get('teacher_advantage_mean', (d.get('verify') or {}).get('teacher_advantage_selected_mean'))) or 0) for d in docs)
        recall=sum(float((d.get('verify') or {}).get('positive_recall') or 0) for d in docs)
        valid.append((harm,-adv,-recall,name,run))
requested_codes={'balanced':int(sys.argv[2]),'precision':int(sys.argv[3])}
requested_codes={k:v for k,v in requested_codes.items() if k in requested}
all_requested_gates_evaluated=bool(requested_codes) and all(v in (0,20) for v in requested_codes.values())
status={'event':'v48_35_certificate_candidate_selection','version':'v48.35.2-ENGINEERING-INTEGRITY','created_unix':time.time(),'attempt_id':attempt_id,
        'controller_exit_codes':{'balanced':int(sys.argv[2]),'precision':int(sys.argv[3])},
        'certificate_executed':True,'gate_evaluated':all_requested_gates_evaluated,
        'valid_candidates':[x[3] for x in valid],'candidates':report,'test_roots_read':False}
atomic_json(root/'dedicated_recalibration_status.json',status)
requested_code_values=list(requested_codes.values())
artifact_failure=any(code not in (0,20) for code in requested_code_values) or any(
    any(('missing' in bucket_doc or 'artifact_error' in bucket_doc) for bucket_doc in variant_doc.values())
    for variant_doc in report.values()
)
def write_blocked(reason, exit_code):
    doc={
      'event':'v48_35_next_commands_blocked','version':'v48.35.2-ENGINEERING-INTEGRITY','created_unix':time.time(),'attempt_id':attempt_id,
      'reason':reason,'exit_code':int(exit_code),'certificate_executed':True,
      'gate_evaluated':bool(status.get('gate_evaluated',False)) if reason=='natural_gate_failed' else False,
      'test_roots_read':False,'controller_exit_codes':status['controller_exit_codes'],
    }
    atomic_json(root/'NEXT_COMMANDS_BLOCKED.json',doc)
    atomic_json(root/'NEXT_COMMANDS_STATUS.json',{**doc,'generated':False})
if artifact_failure:
    atomic_json(root/'CALIBRATION_FAILED.json',status)
    write_blocked('certificate_artifact_or_protocol_failure',30)
    print(json.dumps(status,ensure_ascii=False,indent=2))
    raise SystemExit(30)
if not valid:
    atomic_json(root/'GATE_FAILED.json',status)
    write_blocked('natural_gate_failed',20)
    print(json.dumps(status,ensure_ascii=False,indent=2))
    raise SystemExit(20)
valid.sort(); chosen=valid[0][4]
atomic_text(root/'chosen_base_run_dedicated.txt',str(chosen)+'\n')
safe_source=os.environ['SAFE_WOMD_SOURCE']
commands=(
 '# Natural gate passed on the scene-disjoint dedicated certificate pool.\n'
 f'BASE_RUN={shlex.quote(str(chosen))} RUN={shlex.quote(str(root / "safe_paired"))} SAFE_NOMINAL_ONLY=1 RUN_SAFE_PAIRED_SCALAR=1 SAFE_WOMD_SOURCE={shlex.quote(safe_source)} SAFE_RAW_MAX_SCENARIOS=0 bash scripts/run_v48_35_safe_noninferiority.sh\n'
 f'OUT={shlex.quote(str(root))} bash scripts/run_v48_35_stress_if_authorized.sh\n')
atomic_text(root/'NEXT_COMMANDS.txt',commands)
atomic_json(root/'NEXT_COMMANDS_STATUS.json',{
  'event':'v48_35_next_commands_generated','version':'v48.35.2-ENGINEERING-INTEGRITY','created_unix':time.time(),'attempt_id':attempt_id,
  'generated':True,'certificate_executed':True,'gate_evaluated':True,'gate_passed':True,'chosen_base_run':str(chosen),
  'test_roots_read':False,
})
print(json.dumps(status,ensure_ascii=False,indent=2))
print('chosen',chosen)
PY
