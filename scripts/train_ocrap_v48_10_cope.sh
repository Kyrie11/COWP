#!/usr/bin/env bash
set -euo pipefail

# v48.10 OC-TRAC-COPE
# Stage P: Conditional Option Preference ranks recovery options only.
# Stage E: Monotone Ordinal Evidence decides whether the frozen top-1 recovery
#          is beneficial, dead-zone, or harmful relative to nominal.

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

RUN="${RUN:?RUN is required}"
INIT_CKPT="${INIT_CKPT:?INIT_CKPT is required}"
VARIANT="${VARIANT:-balanced}"
PREF_HIDDEN="${PREFERENCE_CONTEXT_HIDDEN:-32}"
COPE_CONDITIONAL_PREFERENCE="${COPE_CONDITIONAL_PREFERENCE:-true}"
COPE_ORDINAL_EVIDENCE="${COPE_ORDINAL_EVIDENCE:-true}"
case "$COPE_CONDITIONAL_PREFERENCE" in
  1|true|TRUE|yes|YES)
    PREFERENCE_CONDITIONAL_MODE_VALUE=true
    STAGE_P_CONDITIONAL_WEIGHT="${PREFERENCE_CONDITIONAL_SET_WEIGHT:-1.50}"
    STAGE_P_ALL_GROUP_WEIGHT=0
    STAGE_P_SET_MASS=false
    STAGE_P_NOOP=false
    ;;
  *)
    PREFERENCE_CONDITIONAL_MODE_VALUE=false
    STAGE_P_CONDITIONAL_WEIGHT=0
    STAGE_P_ALL_GROUP_WEIGHT="${PREFERENCE_ALL_GROUP_SET_WEIGHT:-1.50}"
    STAGE_P_SET_MASS="${PREFERENCE_SET_MASS_LOSS:-true}"
    STAGE_P_NOOP="${PREFERENCE_NOOP_NOMINAL_ONLY:-true}"
    ;;
esac
case "$COPE_ORDINAL_EVIDENCE" in
  1|true|TRUE|yes|YES)
    STAGE_E_DELTA_MODE=ordinal_evidence
    STAGE_E_METRIC_SOURCE=ordinal_evidence
    STAGE_E_CENTERED_WEIGHT=0
    STAGE_E_POLICY_REG=0
    STAGE_E_POLICY_SIGN=0
    STAGE_E_ORDINAL_TOP1="${ORDINAL_EVIDENCE_POLICY_TOP1_WEIGHT:-2.50}"
    STAGE_E_ORDINAL_ALL="${ORDINAL_EVIDENCE_ALL_CANDIDATE_WEIGHT:-0.25}"
    ;;
  *)
    STAGE_E_DELTA_MODE=gaussian
    STAGE_E_METRIC_SOURCE=gaussian_delta
    STAGE_E_CENTERED_WEIGHT="${CERTIFICATE_ALL_CANDIDATE_WEIGHT:-0.20}"
    STAGE_E_POLICY_REG="${CERTIFICATE_POLICY_TOP1_WEIGHT:-2.00}"
    STAGE_E_POLICY_SIGN="${CERTIFICATE_POLICY_TOP1_SIGN_WEIGHT:-1.50}"
    STAGE_E_ORDINAL_TOP1=0
    STAGE_E_ORDINAL_ALL=0
    ;;
esac
PREF_RUN="$RUN/stages/conditional_preference"
PREF_MODEL="$PREF_RUN/model_preference"
PREF_CAL="$PREF_RUN/preference_audit"
FINAL_MODEL="${MODEL_DIR:-$RUN/model_v48_trac_sr}"
FINAL_CAL="${CAL_DIR:-$RUN/calibration}"
mkdir -p "$PREF_RUN/logs" "$PREF_MODEL" "$PREF_CAL" "$FINAL_MODEL" "$FINAL_CAL"

COMMON_ENV=(
  TRAIN_OCRAP_ROOT="$TRAIN_OCRAP_ROOT" EVAL_OCRAP_ROOT="$EVAL_OCRAP_ROOT"
  TRAIN_MIX="$TRAIN_MIX" VAL_MIX="$VAL_MIX" CAL_MIX="$CAL_MIX"
  VAL_SAFE="${VAL_SAFE:-}" VAL_NEAR="${VAL_NEAR:-}" VAL_CONTACT="${VAL_CONTACT:-}"
  TRAIN_GPU="$TRAIN_GPU" VARIANT="$VARIANT" GROUP_INDEX="$GROUP_INDEX"
  NUM_WORKERS="${NUM_WORKERS:-6}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
  BATCH_SIZE="${BATCH_SIZE:-96}" EXACT_TEACHER_PCD=true
  SET_CONTEXT_ENABLED=false PREFERENCE_HEAD_ENABLED=true PREFERENCE_CONTEXT_ENABLED=true
  PREFERENCE_CONTEXT_HIDDEN="$PREF_HIDDEN"
  RELATIVE_INCLUDE_ABSOLUTE=false
  GROUP_DRO_WEIGHT=0 POLICY_DISTILL_WEIGHT=0 POLICY_REGRET_WEIGHT=0
  POLICY_ADMISSION_DISTILL_WEIGHT=0 OPPORTUNITY_ADMISSION_WEIGHT=0 HARM_ADMISSION_WEIGHT=0
  SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0
  OPPORTUNITY_AUX_WEIGHT=0 HARM_W=0 EXPERT_SPECIALIZATION_WEIGHT=0
  POSITIVE_MACRO_BALANCE_POWER="${POSITIVE_MACRO_BALANCE_POWER:-0.50}"
  POLICY_METRIC_MIN_FOLD_POSITIVE="${POLICY_METRIC_MIN_FOLD_POSITIVE:-6}"
  POLICY_METRIC_ROBUST_TOP_K="${POLICY_METRIC_ROBUST_TOP_K:-2}"
  BEST_METRIC_MIN_DELTA="${BEST_METRIC_MIN_DELTA:-0.00001}"
  PREFERENCE_CONDITIONAL_MODE="$PREFERENCE_CONDITIONAL_MODE_VALUE"
  CONDITIONAL_RECOVERY_RANKING="$PREFERENCE_CONDITIONAL_MODE_VALUE"
)

cat > "$RUN/STAGE_ARCHITECTURE.json" <<JSON
{
  "version": "v48.10-COPE",
  "preference_context_hidden": $PREF_HIDDEN,
  "relative_include_absolute": false,
  "stage_p_trainable": ["direct_preference_context_adapter"],
  "stage_e_trainable": ["direct_delta_adapter"],
  "delta_mode": "$STAGE_E_DELTA_MODE",
  "conditional_preference": "$PREFERENCE_CONDITIONAL_MODE_VALUE"
}
JSON

# Stage P — conditional recovery ranking only.  Nominal is absent from this
# objective; no-op and harmful groups still teach the least-bad recovery at a
# lower weight, while admission is deferred to Stage E.
env "${COMMON_ENV[@]}" \
  RUN="$PREF_RUN" MODEL_DIR="$PREF_MODEL" CAL_DIR="$PREF_CAL" \
  INIT_CKPT="$INIT_CKPT" \
  EPOCHS="${PREFERENCE_EPOCHS:-12}" PATIENCE="${PREFERENCE_PATIENCE:-4}" \
  LR="${PREFERENCE_LR:-0.00010}" ENCODER_LR_SCALE=0 ENCODER_ANCHOR_WEIGHT=0 \
  PREFERENCE_DROPOUT="${PREFERENCE_DROPOUT:-0.00}" \
  TRAINABLE_PARAM_PREFIXES='direct_preference_context_adapter' \
  BEST_METRIC=direct_preference_risk_fold_robust SKIP_POST_TRAIN_CALIBRATION=1 \
  DELTA_HEAD_ENABLED=false DELTA_MODE=gaussian \
  POINT_WEIGHT=0 VALUE_LISTWISE_WEIGHT=0 CENTERED_WEIGHT=0 ADVANTAGE_WEIGHT=0 \
  SETWISE_W=0 PREFERENCE_WEIGHT=0 PREFERENCE_REGRET_WEIGHT=0 \
  PREFERENCE_LISTWISE_WEIGHT=0 PREFERENCE_GAP_WEIGHT=0 PREFERENCE_SET_WEIGHT=0 \
  PREFERENCE_ALL_GROUP_SET_WEIGHT="$STAGE_P_ALL_GROUP_WEIGHT" PREFERENCE_SET_REPLACE_SINGLEWINNER=true \
  PREFERENCE_SET_MASS_LOSS="$STAGE_P_SET_MASS" PREFERENCE_NOOP_NOMINAL_ONLY="$STAGE_P_NOOP" \
  PREFERENCE_CONDITIONAL_SET_WEIGHT="$STAGE_P_CONDITIONAL_WEIGHT" \
  PREFERENCE_CONDITIONAL_NOOP_WEIGHT="${PREFERENCE_CONDITIONAL_NOOP_WEIGHT:-0.30}" \
  PREFERENCE_CONDITIONAL_REGRET_WEIGHT="${PREFERENCE_CONDITIONAL_REGRET_WEIGHT:-0.60}" \
  PREFERENCE_SET_MARGIN="${PREFERENCE_SET_MARGIN:-0.018}" \
  PREFERENCE_TIE_EPS_NEAR="${PREFERENCE_TIE_EPS_NEAR:-0.025}" \
  PREFERENCE_TIE_EPS_CONTACT="${PREFERENCE_TIE_EPS_CONTACT:-0.012}" \
  DELTA_NLL_WEIGHT=0 DELTA_SIGN_WEIGHT=0 \
  ORDINAL_EVIDENCE_POLICY_TOP1_WEIGHT=0 ORDINAL_EVIDENCE_ALL_CANDIDATE_WEIGHT=0 \
  bash scripts/train_ocrap_v48_trac_sr.sh

PREF_CKPT="$PREF_MODEL/best.pt"
[[ -f "$PREF_CKPT" ]] || { echo "missing conditional-preference checkpoint: $PREF_CKPT" >&2; exit 3; }

# Rank-only audit. It does not authorize deployment; it records conditional
# recovery top-1 and regret before evidence training.
set +e
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u tools/calibrate_policy_risk_v48.py \
  --dataset "${VAL_NEAR:-$CAL_MIX}" --checkpoint "$PREF_CKPT" --bucket near \
  --risk-source heads --conditional-recovery-ranking \
  --output "$PREF_CAL/preference_near.json" --rows-output "$PREF_CAL/preference_near.rows.jsonl" \
  --required-min-groups=1 --required-min-scenes=1 --min-fit-selected=1 --min-verify-selected=1 \
  >"$PREF_RUN/logs/preference_audit_near.log" 2>&1
PREF_NEAR_RC=$?
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u tools/calibrate_policy_risk_v48.py \
  --dataset "${VAL_CONTACT:-$CAL_MIX}" --checkpoint "$PREF_CKPT" --bucket contact \
  --risk-source heads --conditional-recovery-ranking \
  --output "$PREF_CAL/preference_contact.json" --rows-output "$PREF_CAL/preference_contact.rows.jsonl" \
  --required-min-groups=1 --required-min-scenes=1 --min-fit-selected=1 --min-verify-selected=1 \
  >"$PREF_RUN/logs/preference_audit_contact.log" 2>&1
PREF_CONTACT_RC=$?
set -e
printf '{"complete":true,"near_exit":%s,"contact_exit":%s}\n' "$PREF_NEAR_RC" "$PREF_CONTACT_RC" \
  > "$PREF_CAL/PREFERENCE_AUDIT_COMPLETE.json"

# Stage E — monotone ordinal evidence.  The preference adapter is frozen and
# verified byte-for-geometry through strict_init_prefixes.  Ordered logits model
# benefit / dead-zone / harm instead of regressing the tri-modal delta to zero.
env "${COMMON_ENV[@]}" \
  RUN="$RUN" MODEL_DIR="$FINAL_MODEL" CAL_DIR="$FINAL_CAL" \
  INIT_CKPT="$PREF_CKPT" STRICT_INIT_PREFIXES='direct_preference_context_adapter' \
  EPOCHS="${EVIDENCE_EPOCHS:-10}" PATIENCE="${EVIDENCE_PATIENCE:-4}" \
  LR="${EVIDENCE_LR:-0.00012}" ENCODER_LR_SCALE=0 ENCODER_ANCHOR_WEIGHT=0 \
  DELTA_HEAD_ENABLED=true DELTA_MODE="$STAGE_E_DELTA_MODE" \
  DELTA_HIDDEN="${EVIDENCE_HIDDEN:-48}" DELTA_DROPOUT="${EVIDENCE_DROPOUT:-0.00}" \
  DELTA_INITIAL_LOGVAR="${EVIDENCE_INITIAL_WIDTH_RAW:--2.0}" \
  TRAINABLE_PARAM_PREFIXES='direct_delta_adapter' \
  BEST_METRIC=direct_certificate_risk_fold_robust SKIP_POST_TRAIN_CALIBRATION=0 \
  POLICY_METRIC_RISK_SOURCE="$STAGE_E_METRIC_SOURCE" \
  POINT_WEIGHT=0 VALUE_LISTWISE_WEIGHT=0 CENTERED_WEIGHT="$STAGE_E_CENTERED_WEIGHT" ADVANTAGE_WEIGHT=0 \
  SETWISE_W=0 PREFERENCE_WEIGHT=0 PREFERENCE_REGRET_WEIGHT=0 \
  PREFERENCE_LISTWISE_WEIGHT=0 PREFERENCE_GAP_WEIGHT=0 PREFERENCE_SET_WEIGHT=0 \
  PREFERENCE_ALL_GROUP_SET_WEIGHT=0 PREFERENCE_CONDITIONAL_SET_WEIGHT=0 \
  DELTA_NLL_WEIGHT=0 DELTA_SIGN_WEIGHT=0 \
  CERTIFICATE_POLICY_TOP1_WEIGHT="$STAGE_E_POLICY_REG" CERTIFICATE_POLICY_TOP1_SIGN_WEIGHT="$STAGE_E_POLICY_SIGN" \
  ORDINAL_EVIDENCE_POLICY_TOP1_WEIGHT="$STAGE_E_ORDINAL_TOP1" \
  ORDINAL_EVIDENCE_ALL_CANDIDATE_WEIGHT="$STAGE_E_ORDINAL_ALL" \
  ORDINAL_EVIDENCE_FOCAL_GAMMA="${ORDINAL_EVIDENCE_FOCAL_GAMMA:-1.5}" \
  bash scripts/train_ocrap_v48_trac_sr.sh

python - "$RUN" "$PREF_CKPT" "$FINAL_MODEL/best.pt" <<'PY'
import hashlib,json,pathlib,sys,time
run,pref,final=map(pathlib.Path,sys.argv[1:])
for p in (pref,final):
    if not p.is_file(): raise SystemExit(f'missing staged checkpoint: {p}')
doc={
  'event':'v48_10_cope_staged_training_complete','created_unix':time.time(),
  'preference_checkpoint':str(pref),'preference_sha256':hashlib.sha256(pref.read_bytes()).hexdigest(),
  'evidence_checkpoint':str(final),'evidence_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
  'architecture_contract':str(run/'STAGE_ARCHITECTURE.json'),
}
(run/'STAGED_TRAINING_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
