#!/usr/bin/env bash
set -euo pipefail

# v48.7 OC-TRAC-SPIRE staged training.
# Stage P learns only the independent preference residuals with exact-PCD,
# ambiguity-aware set supervision. Stage C freezes ranking and trains only the
# candidate-minus-nominal delta certificate. This prevents the v48.6 direct
# delta objective from rewriting the preference representation.

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

RUN="${RUN:?RUN is required}"
INIT_CKPT="${INIT_CKPT:?INIT_CKPT is required}"
VARIANT="${VARIANT:-balanced}"
PREF_RUN="$RUN/stages/preference"
PREF_MODEL="$PREF_RUN/model_preference"
PREF_CAL="$PREF_RUN/calibration"
FINAL_MODEL="${MODEL_DIR:-$RUN/model_v48_trac_sr}"
FINAL_CAL="${CAL_DIR:-$RUN/calibration}"
mkdir -p "$PREF_RUN/logs" "$FINAL_MODEL" "$FINAL_CAL"

COMMON_ENV=(
  TRAIN_OCRAP_ROOT="$TRAIN_OCRAP_ROOT" EVAL_OCRAP_ROOT="$EVAL_OCRAP_ROOT"
  TRAIN_MIX="$TRAIN_MIX" VAL_MIX="$VAL_MIX" CAL_MIX="$CAL_MIX"
  VAL_SAFE="${VAL_SAFE:-}" VAL_NEAR="${VAL_NEAR:-}" VAL_CONTACT="${VAL_CONTACT:-}"
  TRAIN_GPU="$TRAIN_GPU" VARIANT="$VARIANT" GROUP_INDEX="$GROUP_INDEX"
  NUM_WORKERS="${NUM_WORKERS:-6}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
  BATCH_SIZE="${BATCH_SIZE:-72}" EXACT_TEACHER_PCD=true
  SET_CONTEXT_ENABLED=false PREFERENCE_HEAD_ENABLED=true PREFERENCE_CONTEXT_ENABLED=true
  GROUP_DRO_WEIGHT=0 POLICY_DISTILL_WEIGHT=0 POLICY_REGRET_WEIGHT=0
  POLICY_ADMISSION_DISTILL_WEIGHT=0 OPPORTUNITY_ADMISSION_WEIGHT=0 HARM_ADMISSION_WEIGHT=0
  SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0
  OPPORTUNITY_AUX_WEIGHT=0 HARM_W=0 EXPERT_SPECIALIZATION_WEIGHT=0
  POSITIVE_MACRO_BALANCE_POWER="${POSITIVE_MACRO_BALANCE_POWER:-0.25}"
  POLICY_METRIC_OPP_THRESHOLD="${POLICY_METRIC_OPP_THRESHOLD:-0.65}"
  POLICY_METRIC_HARM_THRESHOLD="${POLICY_METRIC_HARM_THRESHOLD:-0.30}"
  POLICY_METRIC_RANK_MARGIN="${POLICY_METRIC_RANK_MARGIN:-0.020}"
  POLICY_METRIC_MISS_WEIGHT="${POLICY_METRIC_MISS_WEIGHT:-0.25}"
)

# Stage P: ranking only. Keep the inherited encoder/value surface immutable.
env "${COMMON_ENV[@]}" \
  RUN="$PREF_RUN" MODEL_DIR="$PREF_MODEL" CAL_DIR="$PREF_CAL" \
  INIT_CKPT="$INIT_CKPT" \
  EPOCHS="${PREFERENCE_EPOCHS:-10}" PATIENCE="${PREFERENCE_PATIENCE:-3}" \
  LR="${PREFERENCE_LR:-0.00018}" ENCODER_LR_SCALE=0 ENCODER_ANCHOR_WEIGHT=0 \
  TRAINABLE_PARAM_PREFIXES='direct_preference_adapter,direct_preference_context_adapter' \
  BEST_METRIC=direct_preference_risk_fold_worst SKIP_POST_TRAIN_CALIBRATION=1 \
  DELTA_HEAD_ENABLED=false POINT_WEIGHT=0 VALUE_LISTWISE_WEIGHT=0 CENTERED_WEIGHT=0 ADVANTAGE_WEIGHT=0 \
  SETWISE_W=0 PREFERENCE_WEIGHT="${PREFERENCE_WEIGHT:-1.00}" \
  PREFERENCE_REGRET_WEIGHT="${PREFERENCE_REGRET_WEIGHT:-0.75}" \
  PREFERENCE_LISTWISE_WEIGHT=0 PREFERENCE_GAP_WEIGHT="${PREFERENCE_GAP_WEIGHT:-0.15}" \
  PREFERENCE_SET_WEIGHT="${PREFERENCE_SET_WEIGHT:-1.25}" \
  PREFERENCE_SET_MARGIN="${PREFERENCE_SET_MARGIN:-0.020}" \
  PREFERENCE_TIE_EPS_NEAR="${PREFERENCE_TIE_EPS_NEAR:-0.025}" \
  PREFERENCE_TIE_EPS_CONTACT="${PREFERENCE_TIE_EPS_CONTACT:-0.010}" \
  DELTA_NLL_WEIGHT=0 \
  bash scripts/train_ocrap_v48_trac_sr.sh

PREF_CKPT="$PREF_MODEL/best.pt"
[[ -f "$PREF_CKPT" ]] || { echo "missing preference checkpoint: $PREF_CKPT" >&2; exit 3; }

# Stage C: certificate only. The new delta adapter is absent from the stage-P
# checkpoint, so it starts from the model's conservative zero-mean initializer.
env "${COMMON_ENV[@]}" \
  RUN="$RUN" MODEL_DIR="$FINAL_MODEL" CAL_DIR="$FINAL_CAL" \
  INIT_CKPT="$PREF_CKPT" \
  EPOCHS="${CERTIFICATE_EPOCHS:-8}" PATIENCE="${CERTIFICATE_PATIENCE:-3}" \
  LR="${CERTIFICATE_LR:-0.00020}" ENCODER_LR_SCALE=0 ENCODER_ANCHOR_WEIGHT=0 \
  TRAINABLE_PARAM_PREFIXES='direct_delta_adapter' \
  BEST_METRIC=direct_certificate_risk_fold_worst SKIP_POST_TRAIN_CALIBRATION=0 \
  DELTA_HEAD_ENABLED=true POINT_WEIGHT=0 VALUE_LISTWISE_WEIGHT=0 CENTERED_WEIGHT=1.0 ADVANTAGE_WEIGHT=0 \
  SETWISE_W=0 PREFERENCE_WEIGHT=0 PREFERENCE_REGRET_WEIGHT=0 \
  PREFERENCE_LISTWISE_WEIGHT=0 PREFERENCE_GAP_WEIGHT=0 PREFERENCE_SET_WEIGHT=0 \
  PREFERENCE_TIE_EPS_NEAR="${PREFERENCE_TIE_EPS_NEAR:-0.025}" \
  PREFERENCE_TIE_EPS_CONTACT="${PREFERENCE_TIE_EPS_CONTACT:-0.010}" \
  DELTA_NLL_WEIGHT="${DELTA_NLL_WEIGHT:-1.00}" \
  bash scripts/train_ocrap_v48_trac_sr.sh

python - "$RUN" "$PREF_CKPT" "$FINAL_MODEL/best.pt" <<'PY'
import hashlib,json,pathlib,sys,time
run,pref,final=map(pathlib.Path,sys.argv[1:])
for p in (pref,final):
    if not p.is_file(): raise SystemExit(f'missing staged checkpoint: {p}')
doc={
  'event':'v48_7_spire_staged_training_complete','created_unix':time.time(),
  'preference_checkpoint':str(pref),'preference_sha256':hashlib.sha256(pref.read_bytes()).hexdigest(),
  'certificate_checkpoint':str(final),'certificate_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
}
(run/'STAGED_TRAINING_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
