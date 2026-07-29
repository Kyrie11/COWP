#!/usr/bin/env bash
set -euo pipefail

# v48.9 OC-TRAC-PACER staged optimization.
# Stage P learns an intervention-aware partial-label set preference: material
# opportunities admit a teacher-equivalent recovery set, while no-opportunity
# groups explicitly prefer nominal. Stage C freezes ranking and learns relative
# gain on the candidate distribution induced by Stage P. Deployment uses the
# empirical policy-risk gate; split-conformal remains an optional ablation.

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

RUN="${RUN:?RUN is required}"
INIT_CKPT="${INIT_CKPT:?INIT_CKPT is required}"
VARIANT="${VARIANT:-balanced}"
PREF_RUN="$RUN/stages/preference"
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
  BATCH_SIZE="${BATCH_SIZE:-72}" EXACT_TEACHER_PCD=true
  SET_CONTEXT_ENABLED=false PREFERENCE_HEAD_ENABLED=true PREFERENCE_CONTEXT_ENABLED=true
  RELATIVE_INCLUDE_ABSOLUTE=false
  GROUP_DRO_WEIGHT=0 POLICY_DISTILL_WEIGHT=0 POLICY_REGRET_WEIGHT=0
  POLICY_ADMISSION_DISTILL_WEIGHT=0 OPPORTUNITY_ADMISSION_WEIGHT=0 HARM_ADMISSION_WEIGHT=0
  SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0
  OPPORTUNITY_AUX_WEIGHT=0 HARM_W=0 EXPERT_SPECIALIZATION_WEIGHT=0
  POSITIVE_MACRO_BALANCE_POWER="${POSITIVE_MACRO_BALANCE_POWER:-0.50}"
  POLICY_METRIC_RANK_HARM_WEIGHT="${POLICY_METRIC_RANK_HARM_WEIGHT:-0.25}"
  POLICY_METRIC_RANK_FALSE_WEIGHT="${POLICY_METRIC_RANK_FALSE_WEIGHT:-0.30}"
  POLICY_METRIC_MIN_FOLD_POSITIVE="${POLICY_METRIC_MIN_FOLD_POSITIVE:-6}"
  POLICY_METRIC_ROBUST_TOP_K="${POLICY_METRIC_ROBUST_TOP_K:-2}"
  BEST_METRIC_MIN_DELTA="${BEST_METRIC_MIN_DELTA:-0.00001}"
)

# Stage P: only the small invariant context residual is trainable. The inherited
# pointwise preference is retained as a frozen base, avoiding the 808k-parameter
# earlier high-capacity preference fit on only a few hundred informative groups.
env "${COMMON_ENV[@]}" \
  RUN="$PREF_RUN" MODEL_DIR="$PREF_MODEL" CAL_DIR="$PREF_CAL" \
  INIT_CKPT="$INIT_CKPT" \
  EPOCHS="${PREFERENCE_EPOCHS:-12}" PATIENCE="${PREFERENCE_PATIENCE:-4}" \
  LR="${PREFERENCE_LR:-0.00010}" ENCODER_LR_SCALE=0 ENCODER_ANCHOR_WEIGHT=0 \
  PREFERENCE_CONTEXT_HIDDEN="${PREFERENCE_CONTEXT_HIDDEN:-32}" \
  PREFERENCE_DROPOUT="${PREFERENCE_DROPOUT:-0.00}" \
  TRAINABLE_PARAM_PREFIXES='direct_preference_context_adapter' \
  BEST_METRIC=direct_preference_risk_fold_robust SKIP_POST_TRAIN_CALIBRATION=1 \
  DELTA_HEAD_ENABLED=false POINT_WEIGHT=0 VALUE_LISTWISE_WEIGHT=0 CENTERED_WEIGHT=0 ADVANTAGE_WEIGHT=0 \
  SETWISE_W=0 PREFERENCE_WEIGHT=0 PREFERENCE_REGRET_WEIGHT=0 \
  PREFERENCE_LISTWISE_WEIGHT=0 PREFERENCE_GAP_WEIGHT=0 PREFERENCE_SET_WEIGHT=0 \
  PREFERENCE_ALL_GROUP_SET_WEIGHT="${PREFERENCE_ALL_GROUP_SET_WEIGHT:-1.50}" \
  PREFERENCE_SET_REPLACE_SINGLEWINNER=true \
  PREFERENCE_SET_MASS_LOSS="${PREFERENCE_SET_MASS_LOSS:-true}" \
  PREFERENCE_NOOP_NOMINAL_ONLY="${PREFERENCE_NOOP_NOMINAL_ONLY:-true}" \
  PREFERENCE_DEADZONE_MARGIN="${PREFERENCE_DEADZONE_MARGIN:-0.008}" \
  PREFERENCE_SET_MARGIN="${PREFERENCE_SET_MARGIN:-0.018}" \
  PREFERENCE_NOMINAL_MARGIN="${PREFERENCE_NOMINAL_MARGIN:-0.025}" \
  PREFERENCE_HARM_MARGIN="${PREFERENCE_HARM_MARGIN:-0.035}" \
  PREFERENCE_TIE_EPS_NEAR="${PREFERENCE_TIE_EPS_NEAR:-0.025}" \
  PREFERENCE_TIE_EPS_CONTACT="${PREFERENCE_TIE_EPS_CONTACT:-0.012}" \
  DELTA_NLL_WEIGHT=0 DELTA_SIGN_WEIGHT=0 \
  bash scripts/train_ocrap_v48_trac_sr.sh

PREF_CKPT="$PREF_MODEL/best.pt"
[[ -f "$PREF_CKPT" ]] || { echo "missing preference checkpoint: $PREF_CKPT" >&2; exit 3; }

# Produce a rank-only audit before certificate training. It intentionally uses
# frozen legacy heads merely to populate admission fields; all preference
# metrics are independent of whether this audit passes Natural gate.
set +e
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u tools/calibrate_policy_risk_v48.py \
  --dataset "${VAL_NEAR:-$CAL_MIX}" --checkpoint "$PREF_CKPT" --bucket near \
  --risk-source heads --output "$PREF_CAL/preference_near.json" \
  --rows-output "$PREF_CAL/preference_near.rows.jsonl" \
  --required-min-groups=1 --required-min-scenes=1 --min-fit-selected=1 --min-verify-selected=1 \
  >"$PREF_RUN/logs/preference_audit_near.log" 2>&1
PREF_NEAR_RC=$?
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u tools/calibrate_policy_risk_v48.py \
  --dataset "${VAL_CONTACT:-$CAL_MIX}" --checkpoint "$PREF_CKPT" --bucket contact \
  --risk-source heads --output "$PREF_CAL/preference_contact.json" \
  --rows-output "$PREF_CAL/preference_contact.rows.jsonl" \
  --required-min-groups=1 --required-min-scenes=1 --min-fit-selected=1 --min-verify-selected=1 \
  >"$PREF_RUN/logs/preference_audit_contact.log" 2>&1
PREF_CONTACT_RC=$?
set -e
printf '{"complete":true,"near_exit":%s,"contact_exit":%s}\n' "$PREF_NEAR_RC" "$PREF_CONTACT_RC" \
  > "$PREF_CAL/PREFERENCE_AUDIT_COMPLETE.json"

# Stage C: ranking is immutable. Train a relative-only robust delta mean. The
# log-variance row remains at its fixed initializer because NLL is disabled.
# Policy risk is controlled on held-out scenes by empirical precision/harm
# bounds, not by a self-reported variance or an all-candidate residual radius.
env "${COMMON_ENV[@]}" \
  RUN="$RUN" MODEL_DIR="$FINAL_MODEL" CAL_DIR="$FINAL_CAL" \
  INIT_CKPT="$PREF_CKPT" \
  EPOCHS="${CERTIFICATE_EPOCHS:-10}" PATIENCE="${CERTIFICATE_PATIENCE:-4}" \
  LR="${CERTIFICATE_LR:-0.00012}" ENCODER_LR_SCALE=0 ENCODER_ANCHOR_WEIGHT=0 \
  DELTA_HIDDEN="${DELTA_HIDDEN:-64}" DELTA_DROPOUT="${DELTA_DROPOUT:-0.00}" \
  DELTA_INITIAL_LOGVAR="${DELTA_INITIAL_LOGVAR:--4.605170186}" \
  TRAINABLE_PARAM_PREFIXES='direct_delta_adapter' \
  BEST_METRIC=direct_certificate_risk_fold_robust SKIP_POST_TRAIN_CALIBRATION=0 \
  DELTA_HEAD_ENABLED=true POINT_WEIGHT=0 VALUE_LISTWISE_WEIGHT=0 CENTERED_WEIGHT="${CERTIFICATE_ALL_CANDIDATE_WEIGHT:-0.20}" ADVANTAGE_WEIGHT=0 \
  SETWISE_W=0 PREFERENCE_WEIGHT=0 PREFERENCE_REGRET_WEIGHT=0 \
  PREFERENCE_LISTWISE_WEIGHT=0 PREFERENCE_GAP_WEIGHT=0 PREFERENCE_SET_WEIGHT=0 \
  PREFERENCE_ALL_GROUP_SET_WEIGHT=0 PREFERENCE_SET_REPLACE_SINGLEWINNER=false \
  DELTA_NLL_WEIGHT="${CERTIFICATE_DELTA_NLL_WEIGHT:-0}" \
  DELTA_SIGN_WEIGHT="${CERTIFICATE_DELTA_SIGN_WEIGHT:-0.20}" \
  DELTA_SIGN_TEMPERATURE="${DELTA_SIGN_TEMPERATURE:-0.040}" \
  CERTIFICATE_POLICY_TOP1_WEIGHT="${CERTIFICATE_POLICY_TOP1_WEIGHT:-2.00}" \
  CERTIFICATE_POLICY_TOP1_SIGN_WEIGHT="${CERTIFICATE_POLICY_TOP1_SIGN_WEIGHT:-1.50}" \
  CERTIFICATE_POLICY_TOP1_TEMPERATURE="${CERTIFICATE_POLICY_TOP1_TEMPERATURE:-0.035}" \
  bash scripts/train_ocrap_v48_trac_sr.sh

python - "$RUN" "$PREF_CKPT" "$FINAL_MODEL/best.pt" <<'PY'
import hashlib,json,pathlib,sys,time
run,pref,final=map(pathlib.Path,sys.argv[1:])
for p in (pref,final):
    if not p.is_file(): raise SystemExit(f'missing staged checkpoint: {p}')
doc={
  'event':'v48_9_pacer_staged_training_complete','created_unix':time.time(),
  'preference_checkpoint':str(pref),'preference_sha256':hashlib.sha256(pref.read_bytes()).hexdigest(),
  'certificate_checkpoint':str(final),'certificate_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
  'preference_audit':str(run/'stages'/'preference'/'preference_audit'),
}
(run/'STAGED_TRAINING_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
