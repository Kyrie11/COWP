#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

FINAL_RUN="${RUN:?RUN is required}"
SOURCE_CKPT="${INIT_CKPT:?INIT_CKPT is required}"
GROUP_INDEX="${GROUP_INDEX:?GROUP_INDEX is required}"
FACTOR_RUN="$FINAL_RUN/factor_stage"
IDENTITY_RUN="$FINAL_RUN/identity_stage"
ENABLE_SUPPORT_RELIABILITY="${V4835_ENABLE_SUPPORT_RELIABILITY:-1}"
IDENTITY_TRAIN_ALL="${V4835_IDENTITY_TRAIN_ALL:-1}"
COUPLE_ADMISSION_PRIOR="${V4835_COUPLE_ADMISSION_PRIOR:-1}"
ADAPTIVE_IDENTITY_MARGIN="${V4835_ADAPTIVE_IDENTITY_MARGIN:-0}"
ENABLE_FINAL_CALIBRATION="${V4835_ENABLE_FINAL_CALIBRATION:-0}"
FACTOR_CACHE_RUN="${V4835_FACTOR_CACHE_RUN:-}"
PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-5}"
EVIDENCE_CONTEXT_SOURCE="${EVIDENCE_CALIBRATOR_CONTEXT_SOURCE:-physical_relative}"
export PROPOSAL_TOP_K EVIDENCE_CONTEXT_SOURCE
SUPPORT_JSON="$FINAL_RUN/FACTOR_SUPPORT_CONTRACT.json"
SUPPORT_ENV="$FINAL_RUN/FACTOR_SUPPORT_CONTRACT.env"
CURRENT_STAGE="initialization"
on_variant_error() {
  local rc=$?
  trap - ERR
  python - "$FINAL_RUN" "${VARIANT:-unknown}" "$CURRENT_STAGE" "$rc" "${BASH_LINENO[0]:-0}" "${BASH_COMMAND:-unknown}" <<'PY_STAGE_FAIL'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=True)
doc={'event':'v48_35_variant_stage_failed','created_unix':time.time(),'variant':sys.argv[2],
     'stage':sys.argv[3],'exit_code':int(sys.argv[4]),'shell_line':int(sys.argv[5] or 0),
     'shell_command':sys.argv[6],'test_roots_read':False}
(root/'VARIANT_STAGE_FAILED.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY_STAGE_FAIL
  exit "$rc"
}
trap on_variant_error ERR

rm -rf "$IDENTITY_RUN" "$FINAL_RUN/model_v48_trac_sr" "$FINAL_RUN/calibration"
mkdir -p "$FINAL_RUN" "$IDENTITY_RUN"
rm -f "$FINAL_RUN/VARIANT_STAGE_FAILED.json"

python tools/build_v48_32_factor_support_contract.py \
  --index "$GROUP_INDEX" --output "$SUPPORT_JSON" --env-output "$SUPPORT_ENV" \
  --macro-ids "${DIRECT_VALUE_MACRO_IDS:-2,3,5,6,7}" \
  --max-hard "${POLICY_METRIC_MAX_HARD:-1.0}" \
  --min-nominal-deviation "${POLICY_METRIC_MIN_NOMINAL_DEVIATION:-0.002}" \
  --min-positive "${FACTOR_SUPPORT_MIN_POSITIVE:-40}" \
  --drs-tolerance "${COMPONENT_HARM_DRS_TOLERANCE:-0.05}" \
  --dep-tolerance "${COMPONENT_HARM_DEP_TOLERANCE:-0.05}" \
  --gap-tolerance "${COMPONENT_HARM_GAP_TOLERANCE:-0.05}" \
  --hard-tolerance "${COMPONENT_HARM_HARD_TOLERANCE:-0.05}" \
  --proxy-tolerance "${COMPONENT_HARM_PROXY_TOLERANCE:-0.05}" \
  --require-readable-samples
# shellcheck disable=SC1090
source "$SUPPORT_ENV"
if [[ "$ENABLE_SUPPORT_RELIABILITY" != 1 ]]; then
  EVIDENCE_COMPONENT_RELIABILITY="1,1,1,1,1"
fi
export EVIDENCE_COMPONENT_RELIABILITY

factor_cache_contract_args=(
  --source-checkpoint "$SOURCE_CKPT"
  --group-index "$GROUP_INDEX"
  --validation-group-index "${VAL_GROUP_INDEX:?VAL_GROUP_INDEX is required for exact factor cache validation}"
  --support-contract "$SUPPORT_JSON"
  --train-mix "${TRAIN_MIX:?TRAIN_MIX is required}"
  --validation-mix "${VAL_MIX:?VAL_MIX is required}"
  --variant "${VARIANT:?VARIANT is required}"
  --setting "proposal_top_k=$PROPOSAL_TOP_K"
  --setting "component_count=${EVIDENCE_COMPONENT_COUNT:-5}"
  --setting "component_scale=${EVIDENCE_COMPONENT_SCALE:-6.0}"
  --setting "component_reliability=$EVIDENCE_COMPONENT_RELIABILITY"
  --setting "calibrator_hidden=${EVIDENCE_CALIBRATOR_HIDDEN:-32}"
  --setting "calibrator_scale=${EVIDENCE_CALIBRATOR_SCALE:-0.75}"
  --setting "calibrator_context=${EVIDENCE_CALIBRATOR_CONTEXT:-true}"
  --setting "calibrator_context_source=$EVIDENCE_CONTEXT_SOURCE"
  --setting "tournament_hidden=${SET_TOURNAMENT_HIDDEN:-48}"
  --setting "tournament_heads=${SET_TOURNAMENT_HEADS:-4}"
  --setting "tournament_dropout=${SET_TOURNAMENT_DROPOUT:-0.05}"
  --setting "benefit_listwise_weight=${FACTOR_BENEFIT_LISTWISE_WEIGHT:-0.75}"
  --setting "component_tail_weight=${FACTOR_COMPONENT_TAIL_WEIGHT:-0.50}"
  --setting "component_margin_regression_weight=${FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT:-0.75}"
  --setting "positive_group_boost=${FACTOR_POSITIVE_GROUP_BOOST:-3.0}"
  --setting "positive_macro_balance_power=${FACTOR_POSITIVE_MACRO_BALANCE_POWER:-0.50}"
  --setting "scene_balance_power=${FACTOR_SCENE_BALANCE_POWER:-0.50}"
  --setting "epochs=${FACTOR_EPOCHS:-20}"
  --setting "patience=${FACTOR_PATIENCE:-6}"
  --setting "learning_rate=${FACTOR_LR:-0.00015}"
  --setting "batch_size=${BATCH_SIZE:-72}"
  --setting "deterministic_algorithms=${DETERMINISTIC_ALGORITHMS:-true}"
)


CURRENT_STAGE="factor_stage"
# Stage 1: natural-population raw benefit + signed physical margins.
if [[ -n "$FACTOR_CACHE_RUN" ]]; then
  python tools/manage_v48_32_factor_cache.py --mode verify-reuse \
    "${factor_cache_contract_args[@]}" \
    --contract "$FACTOR_CACHE_RUN/FACTOR_CACHE_CONTRACT.json" \
    --output "$FINAL_RUN/FACTOR_CACHE_VALIDATION.json"
  python tools/materialize_v48_35_factor_cache.py \
    --source-stage "$FACTOR_CACHE_RUN" --destination-stage "$FACTOR_RUN" \
    --output "$FINAL_RUN/FACTOR_CACHE_MATERIALIZATION.json"
else
  rm -rf "$FACTOR_RUN"; mkdir -p "$FACTOR_RUN"
  RUN="$FACTOR_RUN" INIT_CKPT="$SOURCE_CKPT" \
  EVIDENCE_CALIBRATOR_CONTEXT=true EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="$EVIDENCE_CONTEXT_SOURCE" \
  EVIDENCE_ADMISSION_HEAD=false \
  EVIDENCE_COMPONENT_COUNT="${EVIDENCE_COMPONENT_COUNT:-5}" \
  EVIDENCE_ADMISSION_PRIOR_MODE=frontier_capped_slack EVIDENCE_ADMISSION_PRIOR_DETACH=true \
  ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=false \
  ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT="${FACTOR_BENEFIT_LISTWISE_WEIGHT:-0.75}" \
  ORDINAL_EVIDENCE_COMPONENT_TAIL_WEIGHT="${FACTOR_COMPONENT_TAIL_WEIGHT:-0.50}" \
  ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_WEIGHT="${FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT:-0.75}" \
  ORDINAL_EVIDENCE_SAFE_UTILITY_REGRESSION_WEIGHT=0 \
  ORDINAL_EVIDENCE_SAFE_UTILITY_LISTWISE_WEIGHT=0 \
  ORDINAL_EVIDENCE_ELIGIBLE_POLICY_WEIGHT=0 \
  ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_WEIGHT=0 \
  ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_TEACHER_SCALE=0 \
  ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_WEIGHT=0 \
  ORDINAL_EVIDENCE_ADMISSION_WEIGHT=0 SETWISE_W=0 \
  SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0 \
  OPPORTUNITY_ADMISSION_WEIGHT=0 HARM_ADMISSION_WEIGHT=0 \
  GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false \
  POSITIVE_GROUP_BOOST="${FACTOR_POSITIVE_GROUP_BOOST:-3.0}" \
  POSITIVE_MACRO_BALANCE_POWER="${FACTOR_POSITIVE_MACRO_BALANCE_POWER:-0.50}" SCENE_BALANCE_POWER="${FACTOR_SCENE_BALANCE_POWER:-0.50}" \
  BEST_METRIC=direct_factor_supervised_risk EVALUATE_INITIAL_CHECKPOINT=false \
  EVIDENCE_ADAPT_LR="${FACTOR_LR:-0.00015}" \
  EVIDENCE_ADAPT_EPOCHS="${FACTOR_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${FACTOR_PATIENCE:-6}" \
    bash scripts/adapt_ocrap_v48_35_continuous_frontier_single_stage.sh
  python tools/manage_v48_32_factor_cache.py --mode create \
    "${factor_cache_contract_args[@]}" \
    --contract "$FACTOR_RUN/FACTOR_CACHE_CONTRACT.json" \
    --output "$FINAL_RUN/FACTOR_CACHE_VALIDATION.json"
fi
FACTOR_CKPT="$FACTOR_RUN/model_v48_trac_sr/best.pt"
[[ -f "$FACTOR_CKPT" ]] || { echo "missing factor checkpoint $FACTOR_CKPT" >&2; exit 30; }

CURRENT_STAGE="identity_stage"
# Stage 2: proposal-local action identity. The deployment safe-utility prior is
# coupled to the compact benefit/component heads by default, so candidate AUC can
# influence the exact evidence-reranked top-1 instead of stopping at detached
# candidate classification.
identity_prefixes=direct_evidence_concord_admission_calibrator
if [[ "$IDENTITY_TRAIN_ALL" == 1 ]]; then
  identity_prefixes='direct_evidence_concord_benefit_calibrator,direct_evidence_concord_harm_calibrator,direct_evidence_concord_admission_calibrator'
fi
prior_detach=true
[[ "$COUPLE_ADMISSION_PRIOR" == 1 ]] && prior_detach=false
teacher_scale=0.0
[[ "$ADAPTIVE_IDENTITY_MARGIN" == 1 ]] && teacher_scale="${IDENTITY_TEACHER_GAP_SCALE:-0.75}"

RUN="$IDENTITY_RUN" INIT_CKPT="$FACTOR_CKPT" \
EVIDENCE_CALIBRATOR_CONTEXT=true EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="$EVIDENCE_CONTEXT_SOURCE" \
EVIDENCE_ADMISSION_HEAD=true EVIDENCE_COMPONENT_COUNT="${EVIDENCE_COMPONENT_COUNT:-5}" \
EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE="$identity_prefixes" \
EVIDENCE_ADMISSION_PRIOR_MODE="${EVIDENCE_ADMISSION_PRIOR_MODE:-frontier_capped_slack}" EVIDENCE_ADMISSION_PRIOR_DETACH="$prior_detach" \
EVIDENCE_SLACK_TEMPERATURE="${EVIDENCE_SLACK_TEMPERATURE:-0.025}" \
EVIDENCE_SLACK_PENALTY="${EVIDENCE_SLACK_PENALTY:-1.0}" \
EVIDENCE_FRONTIER_CAP_TEMPERATURE="${EVIDENCE_FRONTIER_CAP_TEMPERATURE:-0.10}" \
EVIDENCE_ADMISSION_SCALE="${EVIDENCE_ADMISSION_SCALE:-2.0}" EVIDENCE_ADMISSION_BOUNDED=false \
ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=false \
ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT="${IDENTITY_BENEFIT_LISTWISE_WEIGHT:-0.35}" \
ORDINAL_EVIDENCE_COMPONENT_TAIL_WEIGHT="${IDENTITY_COMPONENT_TAIL_WEIGHT:-0.30}" \
ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_WEIGHT="${IDENTITY_COMPONENT_MARGIN_REGRESSION_WEIGHT:-0.35}" \
ORDINAL_EVIDENCE_INTRAGROUP_BENEFIT_WEIGHT="${IDENTITY_INTRAGROUP_BENEFIT_WEIGHT:-0.35}" \
ORDINAL_EVIDENCE_INTRAGROUP_HARM_WEIGHT="${IDENTITY_INTRAGROUP_HARM_WEIGHT:-0.45}" \
ORDINAL_EVIDENCE_SAFE_UTILITY_REGRESSION_WEIGHT="${IDENTITY_SAFE_UTILITY_REGRESSION_WEIGHT:-0.75}" \
ORDINAL_EVIDENCE_SAFE_UTILITY_LISTWISE_WEIGHT="${IDENTITY_SAFE_UTILITY_LISTWISE_WEIGHT:-0.50}" \
ORDINAL_EVIDENCE_ELIGIBLE_POLICY_WEIGHT="${IDENTITY_ELIGIBLE_POLICY_WEIGHT:-1.25}" \
ORDINAL_EVIDENCE_ELIGIBLE_POLICY_TEMPERATURE="${IDENTITY_ELIGIBLE_POLICY_TEMPERATURE:-0.10}" \
ORDINAL_EVIDENCE_ELIGIBILITY_LOGIT_TEMPERATURE="${IDENTITY_ELIGIBILITY_LOGIT_TEMPERATURE:-0.25}" \
ORDINAL_EVIDENCE_ELIGIBLE_OPPORTUNITY_THRESHOLD="${POLICY_METRIC_OPPORTUNITY_THRESHOLD:-0.50}" \
ORDINAL_EVIDENCE_ELIGIBLE_HARM_THRESHOLD="${POLICY_METRIC_HARM_THRESHOLD:-0.50}" \
ORDINAL_EVIDENCE_ELIGIBILITY_BOUNDARY_WEIGHT="${IDENTITY_ELIGIBILITY_BOUNDARY_WEIGHT:-1.0}" \
ORDINAL_EVIDENCE_ELIGIBILITY_BOUNDARY_MARGIN="${IDENTITY_ELIGIBILITY_BOUNDARY_MARGIN:-0.20}" \
ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_WEIGHT="${IDENTITY_SAFE_HARD_NEGATIVE_WEIGHT:-2.00}" \
ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_MARGIN="${IDENTITY_SAFE_HARD_NEGATIVE_MARGIN:-0.04}" \
ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_TEACHER_SCALE="$teacher_scale" \
ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_WEIGHT="${IDENTITY_FRONTIER_PAIRWISE_WEIGHT:-0.0}" \
ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_MARGIN="${IDENTITY_FRONTIER_MARGIN:-0.04}" \
ORDINAL_EVIDENCE_ADMISSION_WEIGHT="${IDENTITY_ADMISSION_BINARY_WEIGHT:-0.25}" \
ORDINAL_EVIDENCE_ADMISSION_POS_WEIGHT="${IDENTITY_ADMISSION_POS_WEIGHT:-2.0}" \
ORDINAL_EVIDENCE_ADMISSION_HARM_NEGATIVE_WEIGHT="${IDENTITY_ADMISSION_HARM_NEGATIVE_WEIGHT:-2.5}" \
SETWISE_W="${IDENTITY_SETWISE_WEIGHT:-0.10}" \
EVIDENCE_CALIBRATOR_ANCHOR_WEIGHT="${IDENTITY_CALIBRATOR_ANCHOR_WEIGHT:-0.35}" \
SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0 \
OPPORTUNITY_ADMISSION_WEIGHT=0 HARM_ADMISSION_WEIGHT=0 \
GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false \
POSITIVE_GROUP_BOOST="${IDENTITY_POSITIVE_GROUP_BOOST:-3.0}" POSITIVE_MACRO_BALANCE_POWER="${IDENTITY_POSITIVE_MACRO_BALANCE_POWER:-0.50}" SCENE_BALANCE_POWER="${IDENTITY_SCENE_BALANCE_POWER:-0.50}" \
EVIDENCE_ADAPT_LR="${IDENTITY_LR:-0.00004}" \
BEST_METRIC="${IDENTITY_BEST_METRIC:-direct_contract_lexicographic}" EVALUATE_INITIAL_CHECKPOINT=true \
EVIDENCE_ADAPT_EPOCHS="${IDENTITY_EPOCHS:-24}" EVIDENCE_ADAPT_PATIENCE="${IDENTITY_PATIENCE:-6}" \
  bash scripts/adapt_ocrap_v48_35_continuous_frontier_single_stage.sh
IDENTITY_CKPT="$IDENTITY_RUN/model_v48_trac_sr/best.pt"
[[ -f "$IDENTITY_CKPT" ]] || { echo "missing identity checkpoint $IDENTITY_CKPT" >&2; exit 30; }

CURRENT_STAGE="final_calibration_stage"
# Stage 3: admission-only calibration. Epoch zero is a valid selected fallback;
# the stage must never fail merely because optimization found no safer update.
if [[ "$ENABLE_FINAL_CALIBRATION" == 1 ]]; then
  RUN="$FINAL_RUN" INIT_CKPT="$IDENTITY_CKPT" \
  EVIDENCE_CALIBRATOR_CONTEXT=true EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="$EVIDENCE_CONTEXT_SOURCE" \
  EVIDENCE_ADMISSION_HEAD=true EVIDENCE_COMPONENT_COUNT="${EVIDENCE_COMPONENT_COUNT:-5}" \
  EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_evidence_concord_admission_calibrator \
  EVIDENCE_ADMISSION_PRIOR_MODE=frontier_capped_slack EVIDENCE_ADMISSION_PRIOR_DETACH=true \
  EVIDENCE_SLACK_TEMPERATURE="${EVIDENCE_SLACK_TEMPERATURE:-0.025}" \
  EVIDENCE_SLACK_PENALTY="${EVIDENCE_SLACK_PENALTY:-1.0}" \
  EVIDENCE_FRONTIER_CAP_TEMPERATURE="${EVIDENCE_FRONTIER_CAP_TEMPERATURE:-0.10}" \
  EVIDENCE_ADMISSION_SCALE="${EVIDENCE_ADMISSION_SCALE:-2.0}" EVIDENCE_ADMISSION_BOUNDED=false \
  ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=false \
  ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT=0 \
  ORDINAL_EVIDENCE_COMPONENT_TAIL_WEIGHT=0 \
  ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_WEIGHT=0 \
  ORDINAL_EVIDENCE_INTRAGROUP_BENEFIT_WEIGHT=0 \
  ORDINAL_EVIDENCE_INTRAGROUP_HARM_WEIGHT=0 \
  ORDINAL_EVIDENCE_SAFE_UTILITY_REGRESSION_WEIGHT="${FINAL_SAFE_UTILITY_REGRESSION_WEIGHT:-0.75}" \
  ORDINAL_EVIDENCE_SAFE_UTILITY_LISTWISE_WEIGHT="${FINAL_SAFE_UTILITY_LISTWISE_WEIGHT:-0.35}" \
  ORDINAL_EVIDENCE_ELIGIBLE_POLICY_WEIGHT="${FINAL_ELIGIBLE_POLICY_WEIGHT:-0.75}" \
  ORDINAL_EVIDENCE_ELIGIBLE_OPPORTUNITY_THRESHOLD="${POLICY_METRIC_OPPORTUNITY_THRESHOLD:-0.50}" \
  ORDINAL_EVIDENCE_ELIGIBLE_HARM_THRESHOLD="${POLICY_METRIC_HARM_THRESHOLD:-0.50}" \
  ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_WEIGHT="${FINAL_SAFE_HARD_NEGATIVE_WEIGHT:-1.50}" \
  ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_MARGIN="${FINAL_SAFE_HARD_NEGATIVE_MARGIN:-0.04}" \
  ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_TEACHER_SCALE="$teacher_scale" \
  ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_WEIGHT=0 \
  ORDINAL_EVIDENCE_ADMISSION_WEIGHT="${FINAL_ADMISSION_BINARY_WEIGHT:-0.05}" \
  SETWISE_W="${FINAL_SETWISE_WEIGHT:-0.10}" \
  EVIDENCE_CALIBRATOR_ANCHOR_WEIGHT="${FINAL_CALIBRATOR_ANCHOR_WEIGHT:-0.05}" \
  SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0 \
  OPPORTUNITY_ADMISSION_WEIGHT=0 HARM_ADMISSION_WEIGHT=0 \
  GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false \
  POSITIVE_GROUP_BOOST="${FINAL_POSITIVE_GROUP_BOOST:-3.0}" POSITIVE_MACRO_BALANCE_POWER=0.0 \
  EVIDENCE_ADAPT_LR="${FINAL_LR:-0.00003}" \
  BEST_METRIC=direct_contract_lexicographic EVALUATE_INITIAL_CHECKPOINT=true \
  EVIDENCE_ADAPT_EPOCHS="${FINAL_EPOCHS:-10}" EVIDENCE_ADAPT_PATIENCE="${FINAL_PATIENCE:-4}" \
    bash scripts/adapt_ocrap_v48_35_continuous_frontier_single_stage.sh
else
  rm -rf "$FINAL_RUN/model_v48_trac_sr"
  for item in model_v48_trac_sr STAGE_ARCHITECTURE.json POLICY_CONTRACT.env TRAINING_COMPLETE.json EVIDENCE_CORRECTION_COMPLETE.json; do
    [[ -e "$IDENTITY_RUN/$item" ]] || { echo "missing identity artifact $IDENTITY_RUN/$item" >&2; exit 30; }
    cp -a "$IDENTITY_RUN/$item" "$FINAL_RUN/$item"
  done
fi

FINAL_CKPT="$FINAL_RUN/model_v48_trac_sr/best.pt"
transfer_extra=()
[[ "$ENABLE_FINAL_CALIBRATION" == 1 ]] || transfer_extra+=(--final-stage-disabled)
CURRENT_STAGE="stage_transfer_integrity"
python tools/check_v48_32_stage_transfer.py \
  --factor "$FACTOR_CKPT" --identity "$IDENTITY_CKPT" --final "$FINAL_CKPT" \
  --output "$FINAL_RUN/STAGE_TRANSFER_INTEGRITY.json" "${transfer_extra[@]}"

CURRENT_STAGE="completion_metadata"
python - "$FINAL_RUN" "$SOURCE_CKPT" "$FACTOR_CKPT" "$IDENTITY_CKPT" "$FINAL_CKPT" "$SUPPORT_JSON" "$ENABLE_SUPPORT_RELIABILITY" "$IDENTITY_TRAIN_ALL" "$COUPLE_ADMISSION_PRIOR" "$ADAPTIVE_IDENTITY_MARGIN" "$ENABLE_FINAL_CALIBRATION" "$EVIDENCE_CONTEXT_SOURCE" <<'PY'
import hashlib,json,pathlib,sys,time
run,source,factor,identity,final,support=map(pathlib.Path,sys.argv[1:7])
support_enabled=sys.argv[7] == '1'; identity_all=sys.argv[8] == '1'
coupled=sys.argv[9] == '1'; adaptive=sys.argv[10] == '1'; final_enabled=sys.argv[11] == '1'; context_source=sys.argv[12]
for p in (source,factor,identity,final,support,run/'STAGE_TRANSFER_INTEGRITY.json'):
    if not p.is_file(): raise SystemExit(f'missing v48.35 artifact: {p}')
transfer=json.load(open(run/'STAGE_TRANSFER_INTEGRITY.json'))
doc={
 'event':'v48_35_continuous_frontier_complete','created_unix':time.time(),
 'source_checkpoint':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'factor_checkpoint':str(factor),'factor_sha256':hashlib.sha256(factor.read_bytes()).hexdigest(),
 'identity_checkpoint':str(identity),'identity_sha256':hashlib.sha256(identity.read_bytes()).hexdigest(),
 'final_checkpoint':str(final),'final_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
 'factor_support_contract':str(support),'factor_support_sha256':hashlib.sha256(support.read_bytes()).hexdigest(),
 'stage1_population':'natural_without_replacement','stage2_population':'natural_without_replacement',
 'stage3_population':'natural_without_replacement' if final_enabled else 'disabled',
 'stage2_trainable':'all_compact_evidence_calibrators' if identity_all else 'admission_only_reference',
 'deployment_safe_utility_gradient_coupled':coupled,
 'adaptive_teacher_gap_margin':adaptive,
 'stage3_trainable':['admission_calibrator'] if final_enabled else [],
 'stage2_selected_initial_checkpoint':bool(transfer.get('identity_selected_initial_checkpoint',False)),
 'stage3_selected_initial_checkpoint':bool(transfer.get('final_selected_initial_checkpoint',False)),
 'model_regime_routing':False,'shared_deployment_rule_required':True,'audit_strata_only':['near','contact'],
 'evidence_context_source':context_source,
 'continuous_unified_semantics':'top5 proposal plus physical-relative component margins and noncompensatory frontier cap',
 'independent_measured_hard_veto':True,
 'checkpoint_metric':'direct_contract_lexicographic',
 'semantic_frontier_eligibility_metric':True,
 'final_thresholds_fit_by_single_shared_rule':True,
 'train_metric_uses_final_fitted_thresholds':False,
 'selection_semantics':'rank_topk_then_filter_then_evidence_rerank',
 'support_reliability_enabled':support_enabled,'final_calibration_enabled':final_enabled,
 'test_roots_read':False,
}
(run/'THREE_STAGE_TRAINING_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY

CURRENT_STAGE="complete"
rm -f "$FINAL_RUN/VARIANT_STAGE_FAILED.json"
