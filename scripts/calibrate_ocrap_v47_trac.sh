#!/usr/bin/env bash
set -euo pipefail
export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export EVAL_OCRAP_ROOT=${EVAL_OCRAP_ROOT:-$OCRAP_ROOT}
export BASE_RUN=${BASE_RUN:?set BASE_RUN to a trained v47 OC-TRAC run}
export CKPT=${CKPT:-$BASE_RUN/model_v47_trac/best.pt}
export CAL_DIR=${CAL_DIR:-$BASE_RUN/calibration}
export CAL_GPU=${CAL_GPU:-0}
export TRAC_CONTRACT_MODE=${TRAC_CONTRACT_MODE:-development}
mkdir -p "$CAL_DIR"
[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 2; }

case "$TRAC_CONTRACT_MODE" in
  development)
    : "${TRAC_MIN_GROUPS:=60}"; : "${TRAC_MIN_SCENES:=20}"
    : "${TRAC_MIN_VERIFY_SELECTED:=3}"; : "${TRAC_MIN_VERIFY_PRECISION:=0.50}"
    : "${TRAC_MIN_VERIFY_PRECISION_LCB:=0.15}"; : "${TRAC_MIN_VERIFY_ADV_MEAN:=0.0}"
    : "${TRAC_MAX_VERIFY_HARMFUL_UCB:=0.12}"
    : "${TRAC_MAX_VERIFY_HARMFUL_SELECTED_UCB:=0.75}"
    : "${TRAC_MIN_PRED_TEACHER_CORRELATION:=0.10}"
    : "${TRAC_MIN_MACRO_POSITIVE_COUNT:=2}"; : "${TRAC_MIN_MACRO_POSITIVE_RATE:=0.02}"
    ;;
  final)
    # Publication contract: scene-independent calibration evidence and a
    # conditional false-admission bound among actions that would execute.
    : "${TRAC_MIN_GROUPS:=200}"; : "${TRAC_MIN_SCENES:=100}"
    : "${TRAC_MIN_VERIFY_SELECTED:=25}"; : "${TRAC_MIN_VERIFY_PRECISION:=0.80}"
    : "${TRAC_MIN_VERIFY_PRECISION_LCB:=0.65}"; : "${TRAC_MIN_VERIFY_ADV_MEAN:=0.01}"
    : "${TRAC_MAX_VERIFY_HARMFUL_UCB:=0.06}"
    : "${TRAC_MAX_VERIFY_HARMFUL_SELECTED_UCB:=0.10}"
    : "${TRAC_MIN_PRED_TEACHER_CORRELATION:=0.15}"
    : "${TRAC_MIN_MACRO_POSITIVE_COUNT:=5}"; : "${TRAC_MIN_MACRO_POSITIVE_RATE:=0.05}"
    [[ -n "${TRAC_CAL_NEAR_DATA:-}" && -n "${TRAC_CAL_CONTACT_DATA:-}" ]] || {
      echo "FINAL contract requires dedicated TRAC_CAL_NEAR_DATA and TRAC_CAL_CONTACT_DATA roots" >&2; exit 2; }
    [[ "${TRAC_CAL_NEAR_DATA%/}" != "${EVAL_OCRAP_ROOT%/}/val_near_contact" ]] || {
      echo "FINAL contract forbids reusing val_near_contact for calibration" >&2; exit 2; }
    [[ "${TRAC_CAL_CONTACT_DATA%/}" != "${EVAL_OCRAP_ROOT%/}/val_contact" ]] || {
      echo "FINAL contract forbids reusing val_contact for calibration" >&2; exit 2; }
    ;;
  *) echo "unknown TRAC_CONTRACT_MODE=$TRAC_CONTRACT_MODE" >&2; exit 2 ;;
esac
export TRAC_MIN_GROUPS TRAC_MIN_SCENES TRAC_MIN_VERIFY_SELECTED TRAC_MIN_VERIFY_PRECISION
export TRAC_MIN_VERIFY_PRECISION_LCB TRAC_MIN_VERIFY_ADV_MEAN
export TRAC_MAX_VERIFY_HARMFUL_UCB TRAC_MAX_VERIFY_HARMFUL_SELECTED_UCB
export TRAC_MIN_PRED_TEACHER_CORRELATION TRAC_MIN_MACRO_POSITIVE_COUNT TRAC_MIN_MACRO_POSITIVE_RATE

echo "OC-TRAC calibration contract=$TRAC_CONTRACT_MODE groups>=$TRAC_MIN_GROUPS scenes>=$TRAC_MIN_SCENES selected>=$TRAC_MIN_VERIFY_SELECTED precision>=$TRAC_MIN_VERIFY_PRECISION precision_lcb>=$TRAC_MIN_VERIFY_PRECISION_LCB selected_adv_mean>=$TRAC_MIN_VERIFY_ADV_MEAN conditional_harm_ucb<=$TRAC_MAX_VERIFY_HARMFUL_SELECTED_UCB"

GLOBAL_CORR_ARGS=()
if [[ "${TRAC_REQUIRE_GLOBAL_CORRELATION:-0}" == "1" ]]; then
  GLOBAL_CORR_ARGS+=(--require-global-correlation)
fi

for bucket in near contact; do
  if [[ "$bucket" == near ]]; then
    data=${TRAC_CAL_NEAR_DATA:-$EVAL_OCRAP_ROOT/val_near_contact}; harm=${NEAR_MAX_HARM:-0.55}; hard=${NEAR_MAX_HARD:-0.0}
  else
    data=${TRAC_CAL_CONTACT_DATA:-$EVAL_OCRAP_ROOT/val_contact}; harm=${CONTACT_MAX_HARM:-0.70}; hard=${CONTACT_MAX_HARD:-1.0}
  fi
  CUDA_VISIBLE_DEVICES="$CAL_GPU" PYTHONUNBUFFERED=1 python -u tools/calibrate_direct_value_risk_v47.py \
    --dataset "$data" --checkpoint "$CKPT" --bucket "$bucket" \
    --output "$CAL_DIR/direct_value_risk_${bucket}_v47.json" \
    --rows-output "$CAL_DIR/direct_value_risk_${bucket}_v47.rows.jsonl" \
    --contract-mode "$TRAC_CONTRACT_MODE" \
    --required-min-groups "$TRAC_MIN_GROUPS" \
    --required-min-scenes "$TRAC_MIN_SCENES" \
    --fold-unit ${TRAC_FOLD_UNIT:-scene} \
    --macro-ids ${TRAC_DIRECT_MACRO_IDS:-2,3,5,7} \
    --min-macro-fit-positive-count "$TRAC_MIN_MACRO_POSITIVE_COUNT" \
    --min-macro-fit-positive-rate "$TRAC_MIN_MACRO_POSITIVE_RATE" \
    --min-nominal-deviation ${TRAC_MIN_DEVIATION:-0.002} \
    --max-hard "$hard" --max-harm "$harm" \
    --positive-gain ${TRAC_POSITIVE_GAIN:-0.015} \
    --negative-gain ${TRAC_NEGATIVE_GAIN:-0.010} \
    --min-opportunity ${TRAC_MIN_OPPORTUNITY:-0.0} \
    --min-score-advantage ${TRAC_MIN_SCORE_ADV:--0.10} \
    --min-fit-selected ${TRAC_MIN_FIT_SELECTED:-4} \
    --min-fit-precision ${TRAC_MIN_FIT_PRECISION:-0.55} \
    --min-fit-precision-lcb ${TRAC_MIN_FIT_PRECISION_LCB:-0.20} \
    --min-fit-teacher-advantage-mean ${TRAC_MIN_FIT_ADV_MEAN:-0.0} \
    --max-fit-harmful-selected-rate ${TRAC_MAX_FIT_HARMFUL_RATE:-0.20} \
    --max-fit-harmful-selected-ucb ${TRAC_MAX_FIT_HARMFUL_UCB:-0.75} \
    --max-predicted-harm ${TRAC_MAX_PREDICTED_HARM:-0.95} \
    --min-verify-selected "$TRAC_MIN_VERIFY_SELECTED" \
    --min-verify-precision "$TRAC_MIN_VERIFY_PRECISION" \
    --min-verify-precision-lcb "$TRAC_MIN_VERIFY_PRECISION_LCB" \
    --min-verify-teacher-advantage-mean "$TRAC_MIN_VERIFY_ADV_MEAN" \
    --max-verify-harmful-group-ucb "$TRAC_MAX_VERIFY_HARMFUL_UCB" \
    --max-verify-harmful-selected-ucb "$TRAC_MAX_VERIFY_HARMFUL_SELECTED_UCB" \
    --min-pred-teacher-correlation "$TRAC_MIN_PRED_TEACHER_CORRELATION" \
    "${GLOBAL_CORR_ARGS[@]}" \
    2>&1 | tee "$CAL_DIR/direct_value_risk_${bucket}_v47.log"
done

python - "$CAL_DIR" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); ok=True
for bucket in ('near','contact'):
    d=json.load(open(root/f'direct_value_risk_{bucket}_v47.json'))
    verify=d.get('verify') or {}
    print(bucket, {
      'contract': d.get('contract_mode'),
      'active_valid': d.get('valid_for_active_contract'),
      'development_valid': d.get('valid_for_development'),
      'deployment_valid': d.get('valid_for_deployment'),
      'opportunity_threshold': d.get('direct_value_opportunity_threshold'),
      'harm_threshold': d.get('direct_value_harm_threshold'),
      'score_threshold': d.get('direct_value_threshold'),
      'verify_selected': verify.get('num_selected'),
      'verify_precision': verify.get('challenge_precision'),
      'verify_precision_lcb90': verify.get('challenge_precision_lcb90'),
      'verify_selected_advantage_mean': verify.get('selected_teacher_advantage_mean'),
      'verify_harm_selected_ucb90': verify.get('harmful_selected_ucb90'),
      'verify_harm_group_ucb90': verify.get('harmful_group_exposure_ucb90'),
      'correlation': d.get('pred_teacher_advantage_correlation'),
      'supported_macro_ids': d.get('supported_macro_ids'),
      'warnings': d.get('warnings'),
    })
    ok &= bool(d.get('valid_for_active_contract', False))
raise SystemExit(0 if ok else 3)
PY
