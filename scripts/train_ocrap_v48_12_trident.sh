#!/usr/bin/env bash
set -euo pipefail

# v48.12 OC-TRAC-TRIDENT
# Stage R: recovery-only tournament learns exact gap-weighted option preference.
# Stage E: regime-specific ordered evidence adds cross-group benefit/harm ranking for the
#          frozen policy top-1 candidate. Calibration is policy-first/no-fallback.

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

RUN="${RUN:?RUN is required}"
INIT_CKPT="${INIT_CKPT:?INIT_CKPT is required}"
VARIANT="${VARIANT:-balanced}"
TOURNAMENT_HIDDEN="${SET_TOURNAMENT_HIDDEN:-48}"
TOURNAMENT_HEADS="${SET_TOURNAMENT_HEADS:-4}"
case "${CASTER_ORDERED_EVIDENCE:-true}" in
  1|true|TRUE|yes|YES)
    OLD_ORDINAL_TOP1=0; OLD_ORDINAL_ALL=0
    ORDERED_TOP1="${ORDERED_EVIDENCE_TOP1_WEIGHT:-3.00}"
    ORDERED_ALL="${ORDERED_EVIDENCE_ALL_WEIGHT:-0.20}"
    ;;
  *)
    OLD_ORDINAL_TOP1="${ORDINAL_EVIDENCE_POLICY_TOP1_WEIGHT:-2.50}"
    OLD_ORDINAL_ALL="${ORDINAL_EVIDENCE_ALL_CANDIDATE_WEIGHT:-0.25}"
    ORDERED_TOP1=0; ORDERED_ALL=0
    ;;
esac
PREF_RUN="$RUN/stages/set_tournament"
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
  SET_CONTEXT_ENABLED=false PREFERENCE_HEAD_ENABLED=false PREFERENCE_CONTEXT_ENABLED=false
  RELATIVE_INCLUDE_ABSOLUTE=false
  SET_TOURNAMENT_ENABLED=true SET_TOURNAMENT_HIDDEN="$TOURNAMENT_HIDDEN"
  SET_TOURNAMENT_HEADS="$TOURNAMENT_HEADS" SET_TOURNAMENT_DROPOUT="${SET_TOURNAMENT_DROPOUT:-0.05}"
  SET_TOURNAMENT_REPLACE_BASE=true
  GROUP_DRO_WEIGHT=0 POLICY_DISTILL_WEIGHT=0 POLICY_REGRET_WEIGHT=0
  POLICY_ADMISSION_DISTILL_WEIGHT=0 OPPORTUNITY_ADMISSION_WEIGHT=0 HARM_ADMISSION_WEIGHT=0
  SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0
  OPPORTUNITY_AUX_WEIGHT=0 HARM_W=0 EXPERT_SPECIALIZATION_WEIGHT=0
  POSITIVE_MACRO_BALANCE_POWER="${POSITIVE_MACRO_BALANCE_POWER:-0.65}"
  POLICY_METRIC_MIN_FOLD_POSITIVE="${POLICY_METRIC_MIN_FOLD_POSITIVE:-6}"
  POLICY_METRIC_ROBUST_TOP_K="${POLICY_METRIC_ROBUST_TOP_K:-2}"
  BEST_METRIC_MIN_DELTA="${BEST_METRIC_MIN_DELTA:-0.00001}"
  CONDITIONAL_RECOVERY_RANKING=true POLICY_FIRST_NO_FALLBACK=true
  PREFERENCE_CONDITIONAL_MODE=true
)

cat > "$RUN/STAGE_ARCHITECTURE.json" <<JSON
{
  "version": "v48.12-TRIDENT",
  "policy_semantics": "recovery_pair_tournament_then_cross_group_ordinal_evidence",
  "set_tournament_hidden": $TOURNAMENT_HIDDEN,
  "set_tournament_heads": $TOURNAMENT_HEADS,
  "stage_t_trainable": ["direct_preference_set_ranker"],
  "stage_e_trainable": ["direct_delta_adapters"],
  "delta_mode": "ordinal_evidence",
  "delta_regime_experts": true,
  "delta_policy_features": true
}
JSON

# Stage T: replace the inherited candidate-level rank with a recovery-only set
# tournament. Nominal does not participate in the tournament; admission is Stage E.
env "${COMMON_ENV[@]}" \
  RUN="$PREF_RUN" MODEL_DIR="$PREF_MODEL" CAL_DIR="$PREF_CAL" \
  INIT_CKPT="$INIT_CKPT" \
  EPOCHS="${PREFERENCE_EPOCHS:-14}" PATIENCE="${PREFERENCE_PATIENCE:-4}" \
  LR="${PREFERENCE_LR:-0.00012}" ENCODER_LR_SCALE=0 ENCODER_ANCHOR_WEIGHT=0 \
  TRAINABLE_PARAM_PREFIXES='direct_preference_set_ranker' \
  BEST_METRIC=direct_preference_risk_fold_robust SKIP_POST_TRAIN_CALIBRATION=1 \
  DELTA_HEAD_ENABLED=false DELTA_MODE=gaussian \
  POINT_WEIGHT=0 VALUE_LISTWISE_WEIGHT=0 CENTERED_WEIGHT=0 ADVANTAGE_WEIGHT=0 \
  SETWISE_W=0 PREFERENCE_WEIGHT=0 PREFERENCE_REGRET_WEIGHT=0 \
  PREFERENCE_LISTWISE_WEIGHT=0 PREFERENCE_GAP_WEIGHT=0 PREFERENCE_SET_WEIGHT=0 \
  PREFERENCE_ALL_GROUP_SET_WEIGHT=0 PREFERENCE_SET_REPLACE_SINGLEWINNER=true \
  PREFERENCE_CONDITIONAL_SET_WEIGHT="${PREFERENCE_CONDITIONAL_SET_WEIGHT:-1.75}" \
  PREFERENCE_CONDITIONAL_NOOP_WEIGHT="${PREFERENCE_CONDITIONAL_NOOP_WEIGHT:-0.20}" \
  PREFERENCE_CONDITIONAL_REGRET_WEIGHT="${PREFERENCE_CONDITIONAL_REGRET_WEIGHT:-0.80}" \
  PREFERENCE_CONDITIONAL_PAIRWISE_WEIGHT="${PREFERENCE_CONDITIONAL_PAIRWISE_WEIGHT:-1.00}" \
  PREFERENCE_CONDITIONAL_PAIRWISE_MIN_GAP="${PREFERENCE_CONDITIONAL_PAIRWISE_MIN_GAP:-0.012}" \
  PREFERENCE_CONDITIONAL_PAIRWISE_MARGIN="${PREFERENCE_CONDITIONAL_PAIRWISE_MARGIN:-0.018}" \
  PREFERENCE_SET_MARGIN="${PREFERENCE_SET_MARGIN:-0.020}" \
  PREFERENCE_TIE_EPS_NEAR="${PREFERENCE_TIE_EPS_NEAR:-0.025}" \
  PREFERENCE_TIE_EPS_CONTACT="${PREFERENCE_TIE_EPS_CONTACT:-0.010}" \
  DELTA_NLL_WEIGHT=0 DELTA_SIGN_WEIGHT=0 \
  ORDINAL_EVIDENCE_POLICY_TOP1_WEIGHT=0 ORDINAL_EVIDENCE_ALL_CANDIDATE_WEIGHT=0 \
  ORDINAL_EVIDENCE_ORDERED_NLL_TOP1_WEIGHT=0 ORDINAL_EVIDENCE_ORDERED_NLL_ALL_WEIGHT=0 \
  bash scripts/train_ocrap_v48_trac_sr.sh

PREF_CKPT="$PREF_MODEL/best.pt"
[[ -f "$PREF_CKPT" ]] || { echo "missing set-tournament checkpoint: $PREF_CKPT" >&2; exit 3; }

# Rank-only audits. These do not authorize deployment.
set +e
for bucket in near contact; do
  if [[ "$bucket" == near ]]; then data="${VAL_NEAR:-$CAL_MIX}"; else data="${VAL_CONTACT:-$CAL_MIX}"; fi
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u tools/calibrate_policy_risk_v48.py \
    --dataset "$data" --checkpoint "$PREF_CKPT" --bucket "$bucket" \
    --risk-source heads --conditional-recovery-ranking --policy-first-no-fallback \
    --output "$PREF_CAL/preference_${bucket}.json" \
    --rows-output "$PREF_CAL/preference_${bucket}.rows.jsonl" \
    --required-min-groups=1 --required-min-scenes=1 --min-fit-selected=1 --min-verify-selected=1 \
    >"$PREF_RUN/logs/preference_audit_${bucket}.log" 2>&1
  echo $? > "$PREF_CAL/preference_${bucket}.exit_code"
done
set -e
printf '{"complete":true}\n' > "$PREF_CAL/PREFERENCE_AUDIT_COMPLETE.json"

# Stage E: frozen set tournament, policy-conditioned regime-specific ordered
# evidence. Proper 3-class NLL emphasises harmful-vs-dead separation.
env "${COMMON_ENV[@]}" \
  RUN="$RUN" MODEL_DIR="$FINAL_MODEL" CAL_DIR="$FINAL_CAL" \
  INIT_CKPT="$PREF_CKPT" STRICT_INIT_PREFIXES='direct_preference_set_ranker' \
  EPOCHS="${EVIDENCE_EPOCHS:-12}" PATIENCE="${EVIDENCE_PATIENCE:-4}" \
  LR="${EVIDENCE_LR:-0.00012}" ENCODER_LR_SCALE=0 ENCODER_ANCHOR_WEIGHT=0 \
  DELTA_HEAD_ENABLED=true DELTA_MODE=ordinal_evidence \
  DELTA_HIDDEN="${EVIDENCE_HIDDEN:-48}" DELTA_DROPOUT="${EVIDENCE_DROPOUT:-0.02}" \
  DELTA_INITIAL_LOGVAR="${EVIDENCE_INITIAL_WIDTH_RAW:--2.0}" \
  DELTA_REGIME_EXPERTS="${DELTA_REGIME_EXPERTS:-true}" DELTA_POLICY_FEATURES="${DELTA_POLICY_FEATURES:-true}" \
  TRAINABLE_PARAM_PREFIXES='direct_delta_adapters' \
  BEST_METRIC=direct_certificate_risk_fold_robust SKIP_POST_TRAIN_CALIBRATION=0 \
  POLICY_METRIC_RISK_SOURCE=ordinal_evidence \
  POINT_WEIGHT=0 VALUE_LISTWISE_WEIGHT=0 CENTERED_WEIGHT=0 ADVANTAGE_WEIGHT=0 \
  SETWISE_W=0 PREFERENCE_WEIGHT=0 PREFERENCE_REGRET_WEIGHT=0 \
  PREFERENCE_LISTWISE_WEIGHT=0 PREFERENCE_GAP_WEIGHT=0 PREFERENCE_SET_WEIGHT=0 \
  PREFERENCE_ALL_GROUP_SET_WEIGHT=0 PREFERENCE_CONDITIONAL_SET_WEIGHT=0 \
  DELTA_NLL_WEIGHT=0 DELTA_SIGN_WEIGHT=0 \
  CERTIFICATE_POLICY_TOP1_WEIGHT=0 CERTIFICATE_POLICY_TOP1_SIGN_WEIGHT=0 \
  ORDINAL_EVIDENCE_POLICY_TOP1_WEIGHT="$OLD_ORDINAL_TOP1" ORDINAL_EVIDENCE_ALL_CANDIDATE_WEIGHT="$OLD_ORDINAL_ALL" \
  ORDINAL_EVIDENCE_ORDERED_NLL_TOP1_WEIGHT="$ORDERED_TOP1" \
  ORDINAL_EVIDENCE_ORDERED_NLL_ALL_WEIGHT="$ORDERED_ALL" \
  ORDINAL_EVIDENCE_HARM_CLASS_WEIGHT="${ORDERED_EVIDENCE_HARM_WEIGHT:-3.00}" \
  ORDINAL_EVIDENCE_DEAD_CLASS_WEIGHT="${ORDERED_EVIDENCE_DEAD_WEIGHT:-0.40}" \
  ORDINAL_EVIDENCE_BENEFIT_CLASS_WEIGHT="${ORDERED_EVIDENCE_BENEFIT_WEIGHT:-1.40}" \
  ORDINAL_EVIDENCE_PAIRWISE_BENEFIT_WEIGHT="${ORDERED_EVIDENCE_PAIRWISE_BENEFIT_WEIGHT:-0.60}" \
  ORDINAL_EVIDENCE_PAIRWISE_HARM_WEIGHT="${ORDERED_EVIDENCE_PAIRWISE_HARM_WEIGHT:-1.40}" \
  ORDINAL_EVIDENCE_PAIRWISE_MARGIN="${ORDERED_EVIDENCE_PAIRWISE_MARGIN:-0.25}" \
  bash scripts/train_ocrap_v48_trac_sr.sh

python - "$RUN" "$PREF_CKPT" "$FINAL_MODEL/best.pt" <<'PY'
import hashlib,json,pathlib,sys,time
run,pref,final=map(pathlib.Path,sys.argv[1:])
for p in (pref,final):
    if not p.is_file(): raise SystemExit(f'missing staged checkpoint: {p}')
doc={
  'event':'v48_12_trident_staged_training_complete','created_unix':time.time(),
  'preference_checkpoint':str(pref),'preference_sha256':hashlib.sha256(pref.read_bytes()).hexdigest(),
  'evidence_checkpoint':str(final),'evidence_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
  'architecture_contract':str(run/'STAGE_ARCHITECTURE.json'),
  'policy_first_no_fallback':True,
}
(run/'STAGED_TRAINING_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
