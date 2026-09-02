#!/usr/bin/env bash
set -euo pipefail

# Finalise a v48.19 FACET-BRIDGE run using a scene-disjoint dedicated certificate pool.
# Evidence-adaptation train/dev scenes are never read here and test_* is sealed.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

OUTPUTDIR="${OUTPUTDIR:?OUTPUTDIR is required}"
CAL_SAFE="${CAL_SAFE:?CAL_SAFE is required}"
CERT_NEAR="${CERT_NEAR:?CERT_NEAR is required}"
CERT_CONTACT="${CERT_CONTACT:?CERT_CONTACT is required}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
for d in "$CAL_SAFE" "$CERT_NEAR" "$CERT_CONTACT"; do
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing certificate dataset $d" >&2; exit 2; }
done
mkdir -p "$OUTPUTDIR/logs"
: > "$OUTPUTDIR/logs/v48_19_certificate_status.log"

# Freeze the statistical protocol before reading certificate scores. Rerunning
# the same output directory with changed bounds/counts is rejected unless the
# operator explicitly starts a new run directory.
export CERTIFICATE_CONFIDENCE_LEVEL="${CERTIFICATE_CONFIDENCE_LEVEL:-0.90}"
export CERTIFICATE_BOUND_TYPE="${CERTIFICATE_BOUND_TYPE:-one_sided}"
export POSITIVE_GAIN="${POSITIVE_GAIN:-0.015}"
export NEGATIVE_GAIN="${NEGATIVE_GAIN:-0.010}"
export HARM_LABEL_MODE="${HARM_LABEL_MODE:-component_veto}"
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
python - "$OUTPUTDIR/GATE_SPEC.json" "$CAL_SAFE" "$CERT_NEAR" "$CERT_CONTACT" <<'PY_GATE'
import hashlib,json,os,pathlib,sys,time
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
 'version':'v48.19-FACET-BRIDGE',
 'confidence':{'level':f('CERTIFICATE_CONFIDENCE_LEVEL'),'bound_type':os.environ['CERTIFICATE_BOUND_TYPE']},
 'benefit':{'positive_gain':f('POSITIVE_GAIN')},
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
             dataset_record('contact_certificate',sys.argv[4])],
 'fit_verify_scene_disjoint':True,'test_roots_read':False,
}
canonical=json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()
doc={'event':'v48_19_gate_protocol_preregistered','created_unix':time.time(),
     'protocol_sha256':hashlib.sha256(canonical).hexdigest(),'protocol':protocol}
p=pathlib.Path(sys.argv[1])
if p.exists():
    old=json.load(open(p))
    if old.get('protocol') != protocol:
        raise SystemExit('GATE_SPEC.json already exists with a different protocol; use a new OUTPUTDIR')
else:
    p.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
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
      --set calibration.allow_validation_fallback=false \
      2>&1 | tee "$run/logs/v48_19_calibrate_${buckets[$i]}.log"
  done
  python tools/write_gamma_by_bucket.py \
    --safe "$tmp/calibration_safe_v48.json" \
    --near "$tmp/calibration_near_v48.json" \
    --contact "$tmp/calibration_contact_v48.json" \
    --delta 0.05 --output "$tmp/gamma_rec_by_bucket_v48.json" \
    2>&1 | tee "$run/logs/v48_19_gamma.log"

  local common=(
    --checkpoint "$ckpt" --risk-source="${RISK_SOURCE:-ordinal_evidence}" --allowed-splits=certificate_pool
    --conditional-recovery-ranking --proposal-top-k "${PROPOSAL_TOP_K:-3}" --evidence-rerank-top-k
    --macro-constraint-mode="${MACRO_CONSTRAINT_MODE:-opportunity_normalized}"
    --positive-gain="$POSITIVE_GAIN" --negative-gain="$NEGATIVE_GAIN"
    --max-selected-macro-share="${MAX_SELECTED_MACRO_SHARE:-0.95}"
    --max-macro-excess-share="${MAX_MACRO_EXCESS_SHARE:-0.15}"
    --harm-label-mode="$HARM_LABEL_MODE"
    --certificate-confidence-level="$CERTIFICATE_CONFIDENCE_LEVEL"
    --certificate-bound-type="$CERTIFICATE_BOUND_TYPE"
    --component-harm-drs-tolerance="$COMPONENT_HARM_DRS_TOLERANCE"
    --component-harm-dep-tolerance="$COMPONENT_HARM_DEP_TOLERANCE"
    --component-harm-gap-tolerance="$COMPONENT_HARM_GAP_TOLERANCE"
    --component-harm-hard-tolerance="$COMPONENT_HARM_HARD_TOLERANCE"
    --component-harm-proxy-tolerance="$COMPONENT_HARM_PROXY_TOLERANCE"
  )
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
    --dataset "$CERT_NEAR" --bucket near "${common[@]}" \
    --output "$tmp/direct_value_risk_near_v48.json" --rows-output "$tmp/direct_value_risk_near_v48.rows.jsonl" \
    --required-min-groups="${NEAR_MIN_GROUPS:-120}" --required-min-scenes="${NEAR_MIN_SCENES:-60}" \
    --min-fit-selected="$NEAR_MIN_FIT_SELECTED" --min-verify-selected="$NEAR_MIN_VERIFY_SELECTED" \
    --min-fit-precision-lcb="$NEAR_MIN_FIT_PRECISION_LCB" --min-verify-precision-lcb="$NEAR_MIN_VERIFY_PRECISION_LCB" \
    --max-fit-harmful-group-ucb="$NEAR_MAX_FIT_HARM_UCB" \
    --max-verify-harmful-group-ucb="$NEAR_MAX_VERIFY_HARM_UCB" \
    --max-fit-harmful-selected-ucb="$NEAR_MAX_FIT_SELECTED_HARM_UCB" \
    --max-verify-harmful-selected-ucb="$NEAR_MAX_VERIFY_SELECTED_HARM_UCB" \
    2>&1 | tee "$run/logs/v48_14_policy_near.log"; local sn=${PIPESTATUS[0]}
  CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
    --dataset "$CERT_CONTACT" --bucket contact "${common[@]}" \
    --output "$tmp/direct_value_risk_contact_v48.json" --rows-output "$tmp/direct_value_risk_contact_v48.rows.jsonl" \
    --required-min-groups="${CONTACT_MIN_GROUPS:-180}" --required-min-scenes="${CONTACT_MIN_SCENES:-80}" \
    --min-fit-selected="$CONTACT_MIN_FIT_SELECTED" --min-verify-selected="$CONTACT_MIN_VERIFY_SELECTED" \
    --min-fit-precision-lcb="$CONTACT_MIN_FIT_PRECISION_LCB" --min-verify-precision-lcb="$CONTACT_MIN_VERIFY_PRECISION_LCB" \
    --max-fit-harmful-group-ucb="$CONTACT_MAX_FIT_HARM_UCB" \
    --max-verify-harmful-group-ucb="$CONTACT_MAX_VERIFY_HARM_UCB" \
    --max-fit-harmful-selected-ucb="$CONTACT_MAX_FIT_SELECTED_HARM_UCB" \
    --max-verify-harmful-selected-ucb="$CONTACT_MAX_VERIFY_SELECTED_HARM_UCB" \
    2>&1 | tee "$run/logs/v48_14_policy_contact.log"; local sc=${PIPESTATUS[0]}
  set -e

  python - "$tmp" "$variant" "$ckpt" "$sn" "$sc" <<'PY'
import hashlib,json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); variant=sys.argv[2]; ckpt=pathlib.Path(sys.argv[3])
near_rc,contact_rc=int(sys.argv[4]),int(sys.argv[5])
required=[
 'calibration_mix_v48.json','calibration_safe_v48.json','calibration_near_v48.json','calibration_contact_v48.json',
 'gamma_rec_by_bucket_v48.json','direct_value_risk_near_v48.json','direct_value_risk_contact_v48.json',
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
    if int(d.get('fit_groups',0) or 0) <= 0 or int(d.get('verify_groups',0) or 0) <= 0:
        raise SystemExit(f'non-disjoint/empty certificate folds for {name}')
    if rc not in (0,3,4):
        raise SystemExit(f'policy certificate process failed for {name}: rc={rc}')
    feasibility=d.get('certificate_support_feasibility') or {}
    if rc == 4 and bool(feasibility.get('overall', True)):
        raise SystemExit(f'policy certificate artifact/protocol failure for {name}: rc={rc}')
doc={'event':'v48_19_certificate_pool_calibration_complete','created_unix':time.time(),
 'variant':variant,'checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
 'near_exit_code':near_rc,'contact_exit_code':contact_rc,'gate_evaluated':True,
 'certificate_data_valid':True,'allowed_split_id':'certificate_pool','test_roots_read':False,
 'scene_roles_disjoint':True}
(root/'CERTIFICATE_CALIBRATION_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
  rm -rf "$final.old"
  [[ ! -e "$final" ]] || mv "$final" "$final.old"
  mv "$tmp" "$final"
  rm -rf "$final.old"
  echo "$variant near=$sn contact=$sc" >> "$OUTPUTDIR/logs/v48_19_certificate_status.log"

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
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); report={}; valid=[]
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
          'top1_corr':d.get('unconstrained_group_top1_correlation'),'verify':d.get('verify'),
          'warnings':d.get('warnings'),
        }
    if ok:
        harm=sum(float((d.get('verify') or {}).get('harmful_selected_rate') or 0) for d in docs)
        adv=sum(float(((d.get('verify') or {}).get('teacher_advantage_mean', (d.get('verify') or {}).get('teacher_advantage_selected_mean'))) or 0) for d in docs)
        recall=sum(float((d.get('verify') or {}).get('positive_recall') or 0) for d in docs)
        valid.append((harm,-adv,-recall,name,run))
status={'event':'v48_19_certificate_candidate_selection','created_unix':time.time(),
        'controller_exit_codes':{'balanced':int(sys.argv[2]),'precision':int(sys.argv[3])},
        'valid_candidates':[x[3] for x in valid],'candidates':report,'test_roots_read':False}
(root/'dedicated_recalibration_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
requested_codes=[int(sys.argv[2]) if 'balanced' in requested else 0, int(sys.argv[3]) if 'precision' in requested else 0]
artifact_failure=any(code not in (0,20) for code in requested_codes) or any(
    any(('missing' in bucket_doc or 'artifact_error' in bucket_doc) for bucket_doc in variant_doc.values())
    for variant_doc in report.values()
)
if artifact_failure:
    (root/'CALIBRATION_FAILED.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(status,ensure_ascii=False,indent=2))
    raise SystemExit(30)
if not valid:
    (root/'GATE_FAILED.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(status,ensure_ascii=False,indent=2))
    raise SystemExit(20)
valid.sort(); chosen=valid[0][4]
(root/'chosen_base_run_dedicated.txt').write_text(str(chosen)+'\n')
(root/'NEXT_COMMANDS.txt').write_text(
 f'''# Natural gate passed on the scene-disjoint dedicated certificate pool.\n'''
 f'''BASE_RUN={chosen} RUN={root}/safe_paired SAFE_NOMINAL_ONLY=1 RUN_SAFE_PAIRED_SCALAR=1 SAFE_WOMD_SOURCE=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150 SAFE_RAW_MAX_SCENARIOS=0 bash scripts/run_v48_7_safe_noninferiority.sh\n'''
 f'''OUT={root} bash scripts/run_v48_19_stress_if_authorized.sh\n''')
print(json.dumps(status,ensure_ascii=False,indent=2))
print('chosen',chosen)
PY
