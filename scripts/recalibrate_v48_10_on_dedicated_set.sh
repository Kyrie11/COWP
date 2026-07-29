#!/usr/bin/env bash
set -euo pipefail

# Replace proxy-val calibration with a newly built dedicated calibration set
# without retraining either model and without reading test_*.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
OUTPUTDIR="${OUTPUTDIR:?set OUTPUTDIR to the completed proxy screening run}"
CALIBRATION_OCRAP_ROOT="${CALIBRATION_OCRAP_ROOT:?set CALIBRATION_OCRAP_ROOT}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
CAL_SAFE="$CALIBRATION_OCRAP_ROOT/calibration_safe"
CAL_NEAR="$CALIBRATION_OCRAP_ROOT/calibration_near_contact"
CAL_CONTACT="$CALIBRATION_OCRAP_ROOT/calibration_contact"
for d in "$CAL_SAFE" "$CAL_NEAR" "$CAL_CONTACT"; do [[ -d "$d" ]] || { echo "missing $d" >&2; exit 2; }; done
mkdir -p "$OUTPUTDIR/logs"

recalibrate_variant(){
  local variant="$1" gpu="$2" src="$OUTPUTDIR/candidates/$variant"
  local ckpt="$src/model_v48_trac_sr/best.pt" cal="$src/calibration_dedicated"
  [[ -f "$ckpt" ]] || return 0
  rm -rf "$cal"; mkdir -p "$cal" "$src/logs"
  for bucket in mix safe near contact; do
    case "$bucket" in
      mix) data="$CAL_NEAR,$CAL_CONTACT"; min=100 ;;
      safe) data="$CAL_SAFE"; min=50 ;;
      near) data="$CAL_NEAR"; min=50 ;;
      contact) data="$CAL_CONTACT"; min=50 ;;
    esac
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m ocrap.cli calibrate \
      --dataset "$data" --checkpoint "$ckpt" --output "$cal/calibration_${bucket}_v48.json" \
      --set calibration.required_min_for_delta="$min" \
      2>&1 | tee "$src/logs/dedicated_calibrate_${bucket}.log"
  done
  python tools/write_gamma_by_bucket.py \
    --safe "$cal/calibration_safe_v48.json" --near "$cal/calibration_near_v48.json" \
    --contact "$cal/calibration_contact_v48.json" --delta 0.05 \
    --output "$cal/gamma_rec_by_bucket_v48.json" \
    2>&1 | tee "$src/logs/dedicated_gamma.log"

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
    --dataset "$CAL_NEAR" --checkpoint "$ckpt" --bucket near --risk-source "${RISK_SOURCE:-ordinal_evidence}" --conditional-recovery-ranking \
    --output "$cal/direct_value_risk_near_v48.json" --rows-output "$cal/direct_value_risk_near_v48.rows.jsonl" \
    --required-min-groups="${DEDICATED_NEAR_MIN_GROUPS:-120}" --required-min-scenes="${DEDICATED_NEAR_MIN_SCENES:-60}" \
    --min-fit-selected="${DEDICATED_NEAR_MIN_FIT_SELECTED:-12}" --min-verify-selected="${DEDICATED_NEAR_MIN_VERIFY_SELECTED:-8}" \
    --max-fit-harmful-group-ucb="${DEDICATED_NEAR_MAX_FIT_HARM_UCB:-0.12}" \
    --max-verify-harmful-group-ucb="${DEDICATED_NEAR_MAX_VERIFY_HARM_UCB:-0.14}" \
    --max-fit-harmful-selected-ucb="${DEDICATED_NEAR_MAX_FIT_SELECTED_HARM_UCB:-0.25}" \
    --max-verify-harmful-selected-ucb="${DEDICATED_NEAR_MAX_VERIFY_SELECTED_HARM_UCB:-0.30}" \
    --max-selected-macro-share="${MAX_SELECTED_MACRO_SHARE:-0.85}" \
    2>&1 | tee "$src/logs/dedicated_policy_near.log"; sn=${PIPESTATUS[0]}
  CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
    --dataset "$CAL_CONTACT" --checkpoint "$ckpt" --bucket contact --risk-source "${RISK_SOURCE:-ordinal_evidence}" --conditional-recovery-ranking \
    --output "$cal/direct_value_risk_contact_v48.json" --rows-output "$cal/direct_value_risk_contact_v48.rows.jsonl" \
    --required-min-groups="${DEDICATED_CONTACT_MIN_GROUPS:-150}" --required-min-scenes="${DEDICATED_CONTACT_MIN_SCENES:-60}" \
    --min-fit-selected="${DEDICATED_CONTACT_MIN_FIT_SELECTED:-12}" --min-verify-selected="${DEDICATED_CONTACT_MIN_VERIFY_SELECTED:-8}" \
    --max-fit-harmful-group-ucb="${DEDICATED_CONTACT_MAX_FIT_HARM_UCB:-0.14}" \
    --max-verify-harmful-group-ucb="${DEDICATED_CONTACT_MAX_VERIFY_HARM_UCB:-0.16}" \
    --max-fit-harmful-selected-ucb="${DEDICATED_CONTACT_MAX_FIT_SELECTED_HARM_UCB:-0.25}" \
    --max-verify-harmful-selected-ucb="${DEDICATED_CONTACT_MAX_VERIFY_SELECTED_HARM_UCB:-0.30}" \
    --max-selected-macro-share="${MAX_SELECTED_MACRO_SHARE:-0.85}" \
    2>&1 | tee "$src/logs/dedicated_policy_contact.log"; sc=${PIPESTATUS[0]}
  set -e
  echo "$variant near=$sn contact=$sc" >> "$OUTPUTDIR/logs/dedicated_recalibration_status.log"

  local view="$OUTPUTDIR/dedicated_candidates/$variant"
  rm -rf "$view"; mkdir -p "$view"
  ln -s "$(realpath --relative-to="$view" "$src/model_v48_trac_sr")" "$view/model_v48_trac_sr"
  ln -s "$(realpath --relative-to="$view" "$cal")" "$view/calibration"
}

recalibrate_variant balanced "$GPU0" & P0=$!
recalibrate_variant precision "$GPU1" & P1=$!
wait "$P0" || true; wait "$P1" || true

python - "$OUTPUTDIR" <<'PY' | tee "$OUTPUTDIR/logs/dedicated_candidate_selection.log"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); valid=[]; report={}
for name in ('balanced','precision'):
    r=root/'dedicated_candidates'/name; docs=[]; ok=True; report[name]={}
    for bucket in ('near','contact'):
        p=r/'calibration'/f'direct_value_risk_{bucket}_v48.json'
        try:d=json.load(open(p))
        except Exception as e: report[name][bucket]={'missing':str(e)}; ok=False; continue
        docs.append(d); ok &= bool(d.get('valid_for_deployment',False))
        report[name][bucket]={'valid':d.get('valid_for_deployment'),'verify':d.get('verify'),
          'candidate_auc':d.get('candidate_positive_auc'),'top1_corr':d.get('unconstrained_group_top1_correlation'),
          'warnings':d.get('warnings')}
    if ok:
        harm=sum(float((d.get('verify') or {}).get('harmful_selected_rate') or 0.0) for d in docs)
        corr=sum(float(d.get('unconstrained_group_top1_correlation') or -1.0) for d in docs)
        valid.append((harm,-corr,name,r))
(root/'dedicated_recalibration_status.json').write_text(json.dumps({'valid_candidates':[x[2] for x in valid],'candidates':report},ensure_ascii=False,indent=2)+'\n')
print(report)
if not valid: raise SystemExit('no candidate passed dedicated calibration; proxy result is not promoted and test remains sealed')
valid.sort(); chosen=valid[0][3]
(root/'chosen_base_run_dedicated.txt').write_text(str(chosen)+'\n')
print('chosen dedicated',chosen)
PY

echo "v48.10 dedicated recalibration complete. No model was retrained and no test root was read."
