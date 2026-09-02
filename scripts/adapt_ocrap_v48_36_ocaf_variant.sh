#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

FINAL_RUN="${RUN:?RUN is required}"
SOURCE_CKPT="${INIT_CKPT:?INIT_CKPT is required}"
ORIGINAL_SOURCE_CKPT="$SOURCE_CKPT"
GROUP_INDEX="${GROUP_INDEX:?GROUP_INDEX is required}"
FACTOR_RUN="$FINAL_RUN/factor_stage"
IDENTITY_RUN="$FINAL_RUN/identity_stage"
WITNESS_RUN="$FINAL_RUN/witness_stage"
V4846_OBS_RUN="$FINAL_RUN/v48_46_witness_obs"
V4846_MARGIN_RUN="$FINAL_RUN/v48_46_witness_margin"
V4846_SEQUENTIAL_WITNESS="${V4846_SEQUENTIAL_WITNESS:-0}"
V4847_OBS_RUN="$FINAL_RUN/v48_47_decision_obs"
V4847_FRONTIER_RUN="$FINAL_RUN/v48_47_recovery_frontier"
V4847_DECISION_OBS="${V4847_DECISION_OBS:-0}"
V4847_RECOVERY_FRONTIER="${V4847_RECOVERY_FRONTIER:-0}"
OPTION_EXECUTION_SEMANTICS="${OPTION_EXECUTION_SEMANTICS:-global}"
export OPTION_EXECUTION_SEMANTICS
ENABLE_SUPPORT_RELIABILITY="${V4836_ENABLE_SUPPORT_RELIABILITY:-1}"
IDENTITY_TRAIN_ALL="${V4836_IDENTITY_TRAIN_ALL:-1}"
COUPLE_ADMISSION_PRIOR="${V4836_COUPLE_ADMISSION_PRIOR:-1}"
ADAPTIVE_IDENTITY_MARGIN="${V4836_ADAPTIVE_IDENTITY_MARGIN:-0}"
ENABLE_FINAL_CALIBRATION="${V4836_ENABLE_FINAL_CALIBRATION:-0}"
HAF_FACTOR_PRESERVE="${V4837_FACTOR_PRESERVING_IDENTITY:-0}"
RFR_RESERVE_ONLY="${V4838_RFR_RESERVE_ONLY:-0}"
FACTOR_CACHE_RUN="${V4836_FACTOR_CACHE_RUN:-}"
PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-5}"
EVIDENCE_CONTEXT_SOURCE="${EVIDENCE_CALIBRATOR_CONTEXT_SOURCE:-physical_interaction}"
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
doc={'event':'v48_36_variant_stage_failed','created_unix':time.time(),'variant':sys.argv[2],
     'stage':sys.argv[3],'exit_code':int(sys.argv[4]),'shell_line':int(sys.argv[5] or 0),
     'shell_command':sys.argv[6],'test_roots_read':False}
(root/'VARIANT_STAGE_FAILED.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY_STAGE_FAIL
  exit "$rc"
}
trap on_variant_error ERR

rm -rf "$IDENTITY_RUN" "$WITNESS_RUN" "$V4846_OBS_RUN" "$V4846_MARGIN_RUN" "$V4847_OBS_RUN" "$V4847_FRONTIER_RUN" "$FINAL_RUN/model_v48_trac_sr" "$FINAL_RUN/calibration"
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
  $([[ "${EVIDENCE_DEP_BOUNDARY_ALIGNED:-false}" == "true" ]] && printf %s "--dep-boundary-aligned") \
  $([[ "${EVIDENCE_GAP_ORDINAL_ONLY:-false}" == "true" ]] && printf %s "--gap-ordinal-only") \
  --require-readable-samples
# shellcheck disable=SC1090
source "$SUPPORT_ENV"
if [[ "$ENABLE_SUPPORT_RELIABILITY" != 1 ]]; then
  EVIDENCE_COMPONENT_RELIABILITY="1,1,1,1,1"
fi
export EVIDENCE_COMPONENT_RELIABILITY

# v48.47 DS-OFR: paper-native decision-sufficient witness calibration.
# Factor X reweights the unchanged physical observation-equivalence labels by
# recovery-decision conflict. Factor Y calibrates the candidate-relative DRS /
# deployability frontier directly through OC-MERO. Root logits and every
# downstream evidence/policy head remain frozen. No regime label is consumed.
if [[ "$V4847_DECISION_OBS" == 1 || "$V4847_RECOVERY_FRONTIER" == 1 ]]; then
  if [[ "$V4846_SEQUENTIAL_WITNESS" == 1 || "${V4845_SOWR_MARGIN_WITNESS:-0}" == 1 || "${V4845_SOWR_OBS_KERNEL:-0}" == 1 ]]; then
    echo "v48.47 witness factors cannot be combined with historical v48.45/v48.46 witness stages" >&2
    exit 2
  fi
  export OPTION_EXECUTION_SEMANTICS=observation_class
  SOURCE_CKPT="$ORIGINAL_SOURCE_CKPT"
  if [[ "$V4847_DECISION_OBS" == 1 ]]; then
    CURRENT_STAGE="v48_47_decision_weighted_observation"
    V4847_OBS_REUSE_RUN="${V4847_OBS_REUSE_RUN:-}"
    if [[ -z "$V4847_OBS_REUSE_RUN" && -n "${V4847_OBS_REUSE_BASE:-}" ]]; then
      reuse_candidate="${V4847_OBS_REUSE_BASE%/}/candidates/${VARIANT:?VARIANT is required}/v48_47_decision_obs"
      if [[ -f "$reuse_candidate/V48_47_WITNESS_COMPLETE.json" && -f "$reuse_candidate/model_v48_47_witness/best.pt" ]]; then
        V4847_OBS_REUSE_RUN="$reuse_candidate"
      fi
    fi
    if [[ -n "$V4847_OBS_REUSE_RUN" ]]; then
      python tools/reuse_v48_47_witness_stage.py \
        --source-run "$V4847_OBS_REUSE_RUN" --destination-run "$V4847_OBS_RUN" \
        --expected-source "$ORIGINAL_SOURCE_CKPT" --expected-stage decision_obs \
        --expected-train-mix "${TRAIN_MIX:?TRAIN_MIX is required}" \
        --expected-val-mix "${VAL_MIX:?VAL_MIX is required}" --expected-group-index "$GROUP_INDEX" \
        --expected-epochs "${V4847_OBS_EPOCHS:-5}" --expected-obs-loss-weight "${V4847_OBS_LOSS_WEIGHT:-1.50}" \
        --expected-conflict-scale "${V4847_OBS_CONFLICT_SCALE:-3.0}" \
        --expected-conflict-temperature "${V4847_OBS_CONFLICT_TEMPERATURE:-0.20}" \
        --expected-max-weight "${V4847_OBS_MAX_WEIGHT:-4.0}"
    else
      RUN="$V4847_OBS_RUN" INIT_CKPT="$ORIGINAL_SOURCE_CKPT" \
      TRAIN_MIX="${TRAIN_MIX:?TRAIN_MIX is required}" VAL_MIX="${VAL_MIX:?VAL_MIX is required}" \
      GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="${VAL_GROUP_INDEX:-}" TRAIN_GPU="${TRAIN_GPU:-0}" VARIANT="${VARIANT:?VARIANT is required}" \
      V4847_WITNESS_STAGE=decision_obs OPTION_EXECUTION_SEMANTICS=observation_class \
        bash scripts/adapt_ocrap_v48_47_dsofr_witness_stage.sh
    fi
    SOURCE_CKPT="$V4847_OBS_RUN/model_v48_47_witness/best.pt"
    [[ -f "$SOURCE_CKPT" ]] || { echo "missing v48.47 decision-observation checkpoint $SOURCE_CKPT" >&2; exit 30; }
  fi
  if [[ "$V4847_RECOVERY_FRONTIER" == 1 ]]; then
    CURRENT_STAGE="v48_47_recovery_frontier"
    RUN="$V4847_FRONTIER_RUN" INIT_CKPT="$SOURCE_CKPT" \
    TRAIN_MIX="${TRAIN_MIX:?TRAIN_MIX is required}" VAL_MIX="${VAL_MIX:?VAL_MIX is required}" \
    GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="${VAL_GROUP_INDEX:-}" TRAIN_GPU="${TRAIN_GPU:-0}" VARIANT="${VARIANT:?VARIANT is required}" \
    V4847_WITNESS_STAGE=frontier OPTION_EXECUTION_SEMANTICS=observation_class \
      bash scripts/adapt_ocrap_v48_47_dsofr_witness_stage.sh
    SOURCE_CKPT="$V4847_FRONTIER_RUN/model_v48_47_witness/best.pt"
    [[ -f "$SOURCE_CKPT" ]] || { echo "missing v48.47 recovery-frontier checkpoint $SOURCE_CKPT" >&2; exit 30; }
  fi
fi

# v48.46 OC-SWIC: staged witness calibration.  Observation embedding is
# calibrated first, frozen, then the signed margin head is calibrated.  Root
# logits stay frozen because v48.45.6 showed no validation support for updating
# them.  This is property-conditioned, not regime-conditioned.
if [[ "$V4846_SEQUENTIAL_WITNESS" == 1 ]]; then
  CURRENT_STAGE="v48_46_observation_witness"
  RUN="$V4846_OBS_RUN" INIT_CKPT="$ORIGINAL_SOURCE_CKPT" \
  TRAIN_MIX="${TRAIN_MIX:?TRAIN_MIX is required}" VAL_MIX="${VAL_MIX:?VAL_MIX is required}" \
  GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="${VAL_GROUP_INDEX:-}" TRAIN_GPU="${TRAIN_GPU:-0}" VARIANT="${VARIANT:?VARIANT is required}" \
  V4846_WITNESS_STAGE=obs OPTION_EXECUTION_SEMANTICS="$OPTION_EXECUTION_SEMANTICS" \
    bash scripts/adapt_ocrap_v48_46_ocswic_witness_stage.sh
  obs_ckpt="$V4846_OBS_RUN/model_v48_46_witness/best.pt"
  [[ -f "$obs_ckpt" ]] || { echo "missing v48.46 observation checkpoint $obs_ckpt" >&2; exit 30; }

  CURRENT_STAGE="v48_46_margin_witness"
  RUN="$V4846_MARGIN_RUN" INIT_CKPT="$obs_ckpt" \
  TRAIN_MIX="${TRAIN_MIX:?TRAIN_MIX is required}" VAL_MIX="${VAL_MIX:?VAL_MIX is required}" \
  GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="${VAL_GROUP_INDEX:-}" TRAIN_GPU="${TRAIN_GPU:-0}" VARIANT="${VARIANT:?VARIANT is required}" \
  V4846_WITNESS_STAGE=margin OPTION_EXECUTION_SEMANTICS="$OPTION_EXECUTION_SEMANTICS" \
    bash scripts/adapt_ocrap_v48_46_ocswic_witness_stage.sh
  SOURCE_CKPT="$V4846_MARGIN_RUN/model_v48_46_witness/best.pt"
  [[ -f "$SOURCE_CKPT" ]] || { echo "missing v48.46 margin checkpoint $SOURCE_CKPT" >&2; exit 30; }
fi

# v48.45 SOWR: optional, regime-agnostic recalibration of the paper-matched
# recovery witness before any OCAF/ROCT factor adaptation.  The shared encoder,
# root decoder, proposal policy, thresholds and gate remain unchanged.
SOWR_MARGIN_WITNESS="${V4845_SOWR_MARGIN_WITNESS:-0}"
SOWR_OBS_KERNEL="${V4845_SOWR_OBS_KERNEL:-0}"
if [[ "$V4846_SEQUENTIAL_WITNESS" != 1 && ( "$SOWR_MARGIN_WITNESS" == 1 || "$SOWR_OBS_KERNEL" == 1 ) ]]; then
  CURRENT_STAGE="shared_option_witness_recalibration"
  RUN="$WITNESS_RUN" INIT_CKPT="$ORIGINAL_SOURCE_CKPT" \
  TRAIN_MIX="${TRAIN_MIX:?TRAIN_MIX is required}" VAL_MIX="${VAL_MIX:?VAL_MIX is required}" \
  GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="${VAL_GROUP_INDEX:-}" \
  TRAIN_GPU="${TRAIN_GPU:-0}" VARIANT="${VARIANT:?VARIANT is required}" \
  V4845_SOWR_MARGIN_WITNESS="$SOWR_MARGIN_WITNESS" \
  V4845_SOWR_OBS_KERNEL="$SOWR_OBS_KERNEL" \
  SOWR_EPOCHS="${SOWR_EPOCHS:-8}" SOWR_PATIENCE="${SOWR_PATIENCE:-3}" \
  SOWR_LR="${SOWR_LR:-0.00005}" SOWR_BATCH_SIZE="${SOWR_BATCH_SIZE:-72}" \
    bash scripts/adapt_ocrap_v48_45_sowr_stage.sh
  SOURCE_CKPT="$WITNESS_RUN/model_v48_sowr/best.pt"
  [[ -f "$SOURCE_CKPT" ]] || { echo "missing SOWR checkpoint $SOURCE_CKPT" >&2; exit 30; }
fi

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
  --setting "benefit_residual_scale=${EVIDENCE_BENEFIT_RESIDUAL_SCALE:-1.0}"
  --setting "unbounded_benefit_factor=${EVIDENCE_UNBOUNDED_BENEFIT_FACTOR:-false}"
  --setting "unbounded_harm_factors=${EVIDENCE_UNBOUNDED_HARM_FACTORS:-false}"
  --setting "reserve_factor_alignment=${EVIDENCE_RESERVE_FACTOR_ALIGNMENT:-false}"
  --setting "component_reliability=$EVIDENCE_COMPONENT_RELIABILITY"
  --setting "calibrator_hidden=${EVIDENCE_CALIBRATOR_HIDDEN:-32}"
  --setting "calibrator_scale=${EVIDENCE_CALIBRATOR_SCALE:-0.75}"
  --setting "calibrator_context=${EVIDENCE_CALIBRATOR_CONTEXT:-true}"
  --setting "calibrator_context_source=$EVIDENCE_CONTEXT_SOURCE"
  --setting "interaction_hidden=${EVIDENCE_INTERACTION_HIDDEN:-64}"
  --setting "interaction_dropout=${EVIDENCE_INTERACTION_DROPOUT:-0.05}"
  --setting "dual_interaction_bridge=${EVIDENCE_DUAL_INTERACTION_BRIDGE:-false}"
  --setting "factorized_harm_interaction=${EVIDENCE_FACTORIZED_HARM_INTERACTION:-false}"
  --setting "partial_pool_harm_residual=${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL:-false}"
  --setting "partial_pool_harm_residual_scale=${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL_SCALE:-0.50}"
  --setting "rank_benefit_skip=${EVIDENCE_RANK_BENEFIT_SKIP:-false}"
  --setting "rank_benefit_gain_init=${EVIDENCE_RANK_BENEFIT_GAIN_INIT:-1.0}"
  --setting "postprefix_obs_transport_benefit=${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT:-false}"
  --setting "postprefix_obs_transport_harm=${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM:-false}"
  --setting "postprefix_obs_transport_scale=${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_SCALE:-1.0}"
  --setting "roct_benefit=${EVIDENCE_ROCT_BENEFIT:-false}"
  --setting "roct_deployability=${EVIDENCE_ROCT_DEPLOYABILITY:-false}"
  --setting "roct_scale=${EVIDENCE_ROCT_SCALE:-1.0}"
  --setting "roct_alpha=${EVIDENCE_ROCT_ALPHA:-0.2}"
  --setting "roct_beta=${EVIDENCE_ROCT_BETA:-0.2}"
  --setting "roct_top_m=${EVIDENCE_ROCT_TOP_M:-8}"
  --setting "roct_option_temperature=${EVIDENCE_ROCT_OPTION_TEMPERATURE:-0.35}"
  --setting "common_measure_root_mass=${EVIDENCE_COMMON_MEASURE_ROOT_MASS:-false}"
  --setting "native_certificate_preservation=${EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION:-false}"
  --setting "native_drs_tolerance=${EVIDENCE_NATIVE_DRS_TOLERANCE:-0.05}"
  --setting "native_deployability_tolerance=${EVIDENCE_NATIVE_DEPLOYABILITY_TOLERANCE:-0.05}"
  --setting "native_margin_complete_preservation=${EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION:-false}"
  --setting "native_advantage_preservation=${EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION:-false}"
  --setting "native_exact_advantage_preservation=${EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION:-false}"
  --setting "native_boundary_complete_advantage_preservation=${EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION:-false}"
  --setting "native_physical_student_drs=${EVIDENCE_PHYSICAL_STUDENT_DRS:-false}"
  --setting "native_gap_tolerance=${EVIDENCE_NATIVE_GAP_TOLERANCE:-0.05}"
  --setting "native_dep_boundary_aligned=${EVIDENCE_DEP_BOUNDARY_ALIGNED:-false}"
  --setting "native_positive_gain=${EVIDENCE_NATIVE_POSITIVE_GAIN:-${FACTOR_RECOVERY_ADVANTAGE_POSITIVE_GAIN:-0.015}}"
  --setting "sowr_margin_witness=${V4845_SOWR_MARGIN_WITNESS:-0}"
  --setting "sowr_obs_kernel=${V4845_SOWR_OBS_KERNEL:-0}"
  --setting "v4846_sequential_witness=$V4846_SEQUENTIAL_WITNESS"
  --setting "v4847_decision_obs=$V4847_DECISION_OBS"
  --setting "v4847_recovery_frontier=$V4847_RECOVERY_FRONTIER"
  --setting "v4847_obs_conflict_scale=${V4847_OBS_CONFLICT_SCALE:-3.0}"
  --setting "v4847_obs_conflict_temperature=${V4847_OBS_CONFLICT_TEMPERATURE:-0.20}"
  --setting "v4847_frontier_loss_weight=${V4847_FRONTIER_LOSS_WEIGHT:-2.00}"
  --setting "v4847_frontier_margin_anchor_weight=${V4847_FRONTIER_MARGIN_ANCHOR_WEIGHT:-0.25}"
  --setting "v4851_boundary_complete_frontier=${V4851_BOUNDARY_COMPLETE_FRONTIER:-false}"
  --setting "v4852_physical_teacher_sign_alignment=${V4852_PHYSICAL_TEACHER_SIGN_ALIGNMENT:-false}"
  --setting "v4853_physical_student_sign_alignment=${V4853_PHYSICAL_STUDENT_SIGN_ALIGNMENT:-false}"
  --setting "v4854_invariant_physical_boundary_distillation=${V4854_INVARIANT_PHYSICAL_BOUNDARY_DISTILLATION:-false}"
  --setting "v4856_dep_boundary_aligned=${EVIDENCE_DEP_BOUNDARY_ALIGNED:-false}"
  --setting "v4856_gap_ordinal_only=${EVIDENCE_GAP_ORDINAL_ONLY:-false}"
  --setting "training_option_execution_semantics=$OPTION_EXECUTION_SEMANTICS"
  --setting "sowr_epochs=${SOWR_EPOCHS:-8}"
  --setting "sowr_learning_rate=${SOWR_LR:-0.00005}"
  --setting "consensus_prior_scale=${EVIDENCE_CONSENSUS_PRIOR_SCALE:-0.50}"
  --setting "admission_prior_mode=${EVIDENCE_ADMISSION_PRIOR_MODE:-frontier_capped_slack}"
  --setting "tournament_hidden=${SET_TOURNAMENT_HIDDEN:-48}"
  --setting "tournament_heads=${SET_TOURNAMENT_HEADS:-4}"
  --setting "tournament_dropout=${SET_TOURNAMENT_DROPOUT:-0.05}"
  --setting "benefit_listwise_weight=${FACTOR_BENEFIT_LISTWISE_WEIGHT:-1.00}"
  --setting "component_tail_weight=${FACTOR_COMPONENT_TAIL_WEIGHT:-0.75}"
  --setting "component_margin_regression_weight=${FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT:-1.00}"
  --setting "component_margin_target_mode=${FACTOR_COMPONENT_MARGIN_TARGET_MODE:-raw}"
  --setting "component_margin_target_scale=${FACTOR_COMPONENT_MARGIN_TARGET_SCALE:-0.10}"
  --setting "component_margin_canonical_scales=${FACTOR_COMPONENT_MARGIN_CANONICAL_SCALES:-}"
  --setting "component_margin_regression_reliability=${FACTOR_COMPONENT_MARGIN_REGRESSION_RELIABILITY:-}"
  --setting "component_underestimation_weight=${FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT:-0.0}"
  --setting "safe_positive_component_overestimation_weight=${FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT:-0.0}"
  --setting "benefit_margin_regression_weight=${FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT:-0.0}"
  --setting "benefit_margin_temperature=${FACTOR_BENEFIT_MARGIN_TEMPERATURE:-0.025}"
  --setting "joint_reserve_regression_weight=${FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT:-0.0}"
  --setting "joint_reserve_boundary_weight=${FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT:-0.0}"
  --setting "joint_reserve_boundary_width=${FACTOR_JOINT_RESERVE_BOUNDARY_WIDTH:-0.05}"
  --setting "factor_algorithm_family=${V4838_FACTOR_ALGORITHM_FAMILY:-${OCRAP_ALGORITHM_VERSION:-v48.36-OCAF}}"
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
EVIDENCE_INTERACTION_HIDDEN="${EVIDENCE_INTERACTION_HIDDEN:-64}" EVIDENCE_INTERACTION_DROPOUT="${EVIDENCE_INTERACTION_DROPOUT:-0.05}" EVIDENCE_DUAL_INTERACTION_BRIDGE="${EVIDENCE_DUAL_INTERACTION_BRIDGE:-false}" EVIDENCE_FACTORIZED_HARM_INTERACTION="${EVIDENCE_FACTORIZED_HARM_INTERACTION:-false}" EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL="${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL:-false}" EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL_SCALE="${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL_SCALE:-0.50}" EVIDENCE_RANK_BENEFIT_SKIP="${EVIDENCE_RANK_BENEFIT_SKIP:-false}" EVIDENCE_RANK_BENEFIT_GAIN_INIT="${EVIDENCE_RANK_BENEFIT_GAIN_INIT:-1.0}" EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT="${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT:-false}" EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM="${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM:-false}" EVIDENCE_POSTPREFIX_OBS_TRANSPORT_SCALE="${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_SCALE:-1.0}" EVIDENCE_CONSENSUS_PRIOR_SCALE="${EVIDENCE_CONSENSUS_PRIOR_SCALE:-0.50}" \
  EVIDENCE_ADMISSION_HEAD=false \
  EVIDENCE_COMPONENT_COUNT="${EVIDENCE_COMPONENT_COUNT:-5}" \
  EVIDENCE_COMPONENT_SCALE="${EVIDENCE_COMPONENT_SCALE:-6.0}" \
  EVIDENCE_BENEFIT_RESIDUAL_SCALE="${EVIDENCE_BENEFIT_RESIDUAL_SCALE:-1.0}" \
  EVIDENCE_UNBOUNDED_BENEFIT_FACTOR="${EVIDENCE_UNBOUNDED_BENEFIT_FACTOR:-false}" \
  EVIDENCE_UNBOUNDED_HARM_FACTORS="${EVIDENCE_UNBOUNDED_HARM_FACTORS:-false}" \
  EVIDENCE_RESERVE_FACTOR_ALIGNMENT="${EVIDENCE_RESERVE_FACTOR_ALIGNMENT:-false}" \
  EVIDENCE_ADMISSION_PRIOR_MODE="${EVIDENCE_ADMISSION_PRIOR_MODE:-frontier_capped_slack}" EVIDENCE_ADMISSION_PRIOR_DETACH=true \
  EVIDENCE_BENEFIT_MARGIN_TEMPERATURE="${FACTOR_BENEFIT_MARGIN_TEMPERATURE:-0.025}" \
  EVIDENCE_JOINT_RESERVE_TEMPERATURE="${EVIDENCE_JOINT_RESERVE_TEMPERATURE:-0.025}" \
  ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=false \
  ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT="${FACTOR_BENEFIT_LISTWISE_WEIGHT:-1.00}" \
  ORDINAL_EVIDENCE_COMPONENT_TAIL_WEIGHT="${FACTOR_COMPONENT_TAIL_WEIGHT:-0.75}" \
  ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_WEIGHT="${FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT:-1.00}" \
  ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_MODE="${FACTOR_COMPONENT_MARGIN_TARGET_MODE:-raw}" \
  ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_SCALE="${FACTOR_COMPONENT_MARGIN_TARGET_SCALE:-0.10}" \
  ORDINAL_EVIDENCE_COMPONENT_MARGIN_CANONICAL_SCALES="${FACTOR_COMPONENT_MARGIN_CANONICAL_SCALES:-}" \
  ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_RELIABILITY="${FACTOR_COMPONENT_MARGIN_REGRESSION_RELIABILITY:-}" \
  ORDINAL_EVIDENCE_COMPONENT_UNDERESTIMATION_WEIGHT="${FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT:-0.0}" \
  ORDINAL_EVIDENCE_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT="${FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT:-0.0}" \
  ORDINAL_EVIDENCE_BENEFIT_MARGIN_REGRESSION_WEIGHT="${FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT:-0.0}" \
  ORDINAL_EVIDENCE_BENEFIT_MARGIN_TEMPERATURE="${FACTOR_BENEFIT_MARGIN_TEMPERATURE:-0.025}" \
  ORDINAL_EVIDENCE_JOINT_RESERVE_REGRESSION_WEIGHT="${FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT:-0.0}" \
  ORDINAL_EVIDENCE_JOINT_RESERVE_BOUNDARY_WEIGHT="${FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT:-0.0}" \
  ORDINAL_EVIDENCE_JOINT_RESERVE_BOUNDARY_WIDTH="${FACTOR_JOINT_RESERVE_BOUNDARY_WIDTH:-0.05}" \
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
    bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh
  python tools/manage_v48_32_factor_cache.py --mode create \
    "${factor_cache_contract_args[@]}" \
    --contract "$FACTOR_RUN/FACTOR_CACHE_CONTRACT.json" \
    --output "$FINAL_RUN/FACTOR_CACHE_VALIDATION.json"
fi
FACTOR_CKPT="$FACTOR_RUN/model_v48_trac_sr/best.pt"
[[ -f "$FACTOR_CKPT" ]] || { echo "missing factor checkpoint $FACTOR_CKPT" >&2; exit 30; }

CURRENT_STAGE="identity_stage"
if [[ "$RFR_RESERVE_ONLY" == 1 ]]; then
  # v48.38 RFR: v48.37 C/D repeatedly selected identity epoch zero.  Do not
  # spend compute learning a residual that degrades exact deployment metrics.
  # Materialize a byte-identical no-learning stage so all downstream provenance
  # and certificate machinery remains unchanged and auditable.
  IDENTITY_TRAIN_ALL=0
  COUPLE_ADMISSION_PRIOR=0
  HAF_FACTOR_PRESERVE=0
  ENABLE_FINAL_CALIBRATION=0
  identity_prefixes=""
  python tools/materialize_v48_38_reserve_stage.py \
    --factor-stage "$FACTOR_RUN" --destination "$IDENTITY_RUN" --role identity \
    --implementation-version "${OCRAP_IMPLEMENTATION_VERSION:-v48.38-RFR}"
  IDENTITY_CKPT="$IDENTITY_RUN/model_v48_trac_sr/best.pt"
  [[ -f "$IDENTITY_CKPT" ]] || { echo "missing reserve identity checkpoint $IDENTITY_CKPT" >&2; exit 30; }
else
# Stage 2: proposal-local action identity. The deployment safe-utility prior is
# coupled to the compact benefit/component heads by default, so candidate AUC can
# influence the exact evidence-reranked top-1 instead of stopping at detached
# candidate classification.
identity_prefixes=direct_evidence_concord_admission_calibrator
if [[ "$HAF_FACTOR_PRESERVE" == 1 ]]; then
  # v48.37 HAF: factor heads and the observation-conditioned action bridge form
  # a physically anchored coordinate system.  Stage 2 may learn only the
  # admission residual over that frozen coordinate system; otherwise sparse
  # admission gradients can rotate benefit/harm semantics after factor fitting.
  IDENTITY_TRAIN_ALL=0
  COUPLE_ADMISSION_PRIOR=0
else
  if [[ "$IDENTITY_TRAIN_ALL" == 1 ]]; then
    identity_prefixes='direct_evidence_concord_benefit_calibrator,direct_evidence_concord_harm_calibrator,direct_evidence_concord_admission_calibrator'
  fi
  if [[ "$EVIDENCE_CONTEXT_SOURCE" == physical_interaction ]]; then
    identity_prefixes+=',direct_evidence_interaction_bridge'
  fi
  if [[ "${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL:-false}" == true ]]; then
    identity_prefixes+=',direct_evidence_concord_harm_component_residuals'
  fi
  if [[ "${EVIDENCE_RANK_BENEFIT_SKIP:-false}" == true ]]; then
    identity_prefixes+=',direct_evidence_rank_benefit_log_gain'
  fi
  if [[ "${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT:-false}" == true ]]; then
    identity_prefixes+=',direct_evidence_postprefix_obs_transport_benefit'
  fi
  if [[ "${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM:-false}" == true ]]; then
    identity_prefixes+=',direct_evidence_postprefix_obs_transport_harm'
  fi
  if [[ "${EVIDENCE_ROCT_BENEFIT:-false}" == true ]]; then
    identity_prefixes+=',direct_evidence_roct_benefit'
  fi
  if [[ "${EVIDENCE_ROCT_DEPLOYABILITY:-false}" == true ]]; then
    identity_prefixes+=',direct_evidence_roct_deployability'
  fi
fi
prior_detach=true
[[ "$COUPLE_ADMISSION_PRIOR" == 1 ]] && prior_detach=false
teacher_scale=0.0
[[ "$ADAPTIVE_IDENTITY_MARGIN" == 1 ]] && teacher_scale="${IDENTITY_TEACHER_GAP_SCALE:-0.75}"
identity_benefit_listwise="${IDENTITY_BENEFIT_LISTWISE_WEIGHT:-0.35}"
identity_component_tail="${IDENTITY_COMPONENT_TAIL_WEIGHT:-0.50}"
identity_component_margin="${IDENTITY_COMPONENT_MARGIN_REGRESSION_WEIGHT:-0.75}"
identity_intragroup_benefit="${IDENTITY_INTRAGROUP_BENEFIT_WEIGHT:-0.35}"
identity_intragroup_harm="${IDENTITY_INTRAGROUP_HARM_WEIGHT:-0.45}"
if [[ "$HAF_FACTOR_PRESERVE" == 1 ]]; then
  identity_benefit_listwise=0
  identity_component_tail=0
  identity_component_margin=0
  identity_intragroup_benefit=0
  identity_intragroup_harm=0
fi

RUN="$IDENTITY_RUN" INIT_CKPT="$FACTOR_CKPT" \
EVIDENCE_CALIBRATOR_CONTEXT=true EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="$EVIDENCE_CONTEXT_SOURCE" \
EVIDENCE_INTERACTION_HIDDEN="${EVIDENCE_INTERACTION_HIDDEN:-64}" EVIDENCE_INTERACTION_DROPOUT="${EVIDENCE_INTERACTION_DROPOUT:-0.05}" EVIDENCE_DUAL_INTERACTION_BRIDGE="${EVIDENCE_DUAL_INTERACTION_BRIDGE:-false}" EVIDENCE_FACTORIZED_HARM_INTERACTION="${EVIDENCE_FACTORIZED_HARM_INTERACTION:-false}" EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL="${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL:-false}" EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL_SCALE="${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL_SCALE:-0.50}" EVIDENCE_RANK_BENEFIT_SKIP="${EVIDENCE_RANK_BENEFIT_SKIP:-false}" EVIDENCE_RANK_BENEFIT_GAIN_INIT="${EVIDENCE_RANK_BENEFIT_GAIN_INIT:-1.0}" EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT="${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT:-false}" EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM="${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM:-false}" EVIDENCE_POSTPREFIX_OBS_TRANSPORT_SCALE="${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_SCALE:-1.0}" EVIDENCE_CONSENSUS_PRIOR_SCALE="${EVIDENCE_CONSENSUS_PRIOR_SCALE:-0.50}" \
EVIDENCE_ADMISSION_HEAD=true EVIDENCE_COMPONENT_COUNT="${EVIDENCE_COMPONENT_COUNT:-5}" \
EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE="$identity_prefixes" \
EVIDENCE_COMPONENT_SCALE="${EVIDENCE_COMPONENT_SCALE:-6.0}" EVIDENCE_BENEFIT_RESIDUAL_SCALE="${EVIDENCE_BENEFIT_RESIDUAL_SCALE:-1.0}" \
EVIDENCE_UNBOUNDED_BENEFIT_FACTOR="${EVIDENCE_UNBOUNDED_BENEFIT_FACTOR:-false}" EVIDENCE_UNBOUNDED_HARM_FACTORS="${EVIDENCE_UNBOUNDED_HARM_FACTORS:-false}" \
EVIDENCE_RESERVE_FACTOR_ALIGNMENT="${EVIDENCE_RESERVE_FACTOR_ALIGNMENT:-false}" \
EVIDENCE_ADMISSION_PRIOR_MODE="${EVIDENCE_ADMISSION_PRIOR_MODE:-frontier_capped_slack}" EVIDENCE_ADMISSION_PRIOR_DETACH="$prior_detach" \
EVIDENCE_SLACK_TEMPERATURE="${EVIDENCE_SLACK_TEMPERATURE:-0.025}" \
EVIDENCE_SLACK_PENALTY="${EVIDENCE_SLACK_PENALTY:-1.0}" \
EVIDENCE_FRONTIER_CAP_TEMPERATURE="${EVIDENCE_FRONTIER_CAP_TEMPERATURE:-0.10}" \
EVIDENCE_ADMISSION_SCALE="${EVIDENCE_ADMISSION_SCALE:-2.0}" EVIDENCE_ADMISSION_BOUNDED=false \
ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=false \
ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT="$identity_benefit_listwise" \
ORDINAL_EVIDENCE_COMPONENT_TAIL_WEIGHT="$identity_component_tail" \
ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_WEIGHT="$identity_component_margin" \
ORDINAL_EVIDENCE_BENEFIT_MARGIN_REGRESSION_WEIGHT=0 \
ORDINAL_EVIDENCE_BENEFIT_MARGIN_TEMPERATURE="${FACTOR_BENEFIT_MARGIN_TEMPERATURE:-0.025}" \
ORDINAL_EVIDENCE_INTRAGROUP_BENEFIT_WEIGHT="$identity_intragroup_benefit" \
ORDINAL_EVIDENCE_INTRAGROUP_HARM_WEIGHT="$identity_intragroup_harm" \
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
  bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh
IDENTITY_CKPT="$IDENTITY_RUN/model_v48_trac_sr/best.pt"
[[ -f "$IDENTITY_CKPT" ]] || { echo "missing identity checkpoint $IDENTITY_CKPT" >&2; exit 30; }
fi

CURRENT_STAGE="final_calibration_stage"
if [[ "$RFR_RESERVE_ONLY" == 1 ]]; then
  python tools/materialize_v48_38_reserve_stage.py \
    --factor-stage "$FACTOR_RUN" --destination "$FINAL_RUN" --role final \
    --implementation-version "${OCRAP_IMPLEMENTATION_VERSION:-v48.38-RFR}"
else
# Stage 3: admission-only calibration. Epoch zero is a valid selected fallback;
# the stage must never fail merely because optimization found no safer update.
if [[ "$ENABLE_FINAL_CALIBRATION" == 1 ]]; then
  RUN="$FINAL_RUN" INIT_CKPT="$IDENTITY_CKPT" \
  EVIDENCE_CALIBRATOR_CONTEXT=true EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="$EVIDENCE_CONTEXT_SOURCE" \
EVIDENCE_INTERACTION_HIDDEN="${EVIDENCE_INTERACTION_HIDDEN:-64}" EVIDENCE_INTERACTION_DROPOUT="${EVIDENCE_INTERACTION_DROPOUT:-0.05}" EVIDENCE_CONSENSUS_PRIOR_SCALE="${EVIDENCE_CONSENSUS_PRIOR_SCALE:-0.50}" \
  EVIDENCE_ADMISSION_HEAD=true EVIDENCE_COMPONENT_COUNT="${EVIDENCE_COMPONENT_COUNT:-5}" \
  EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_evidence_concord_admission_calibrator \
  EVIDENCE_COMPONENT_SCALE="${EVIDENCE_COMPONENT_SCALE:-6.0}" EVIDENCE_BENEFIT_RESIDUAL_SCALE="${EVIDENCE_BENEFIT_RESIDUAL_SCALE:-1.0}" \
  EVIDENCE_UNBOUNDED_BENEFIT_FACTOR="${EVIDENCE_UNBOUNDED_BENEFIT_FACTOR:-false}" EVIDENCE_UNBOUNDED_HARM_FACTORS="${EVIDENCE_UNBOUNDED_HARM_FACTORS:-false}" \
  EVIDENCE_RESERVE_FACTOR_ALIGNMENT="${EVIDENCE_RESERVE_FACTOR_ALIGNMENT:-false}" \
  EVIDENCE_ADMISSION_PRIOR_MODE="${EVIDENCE_ADMISSION_PRIOR_MODE:-frontier_capped_slack}" EVIDENCE_ADMISSION_PRIOR_DETACH=true \
  EVIDENCE_SLACK_TEMPERATURE="${EVIDENCE_SLACK_TEMPERATURE:-0.025}" \
  EVIDENCE_SLACK_PENALTY="${EVIDENCE_SLACK_PENALTY:-1.0}" \
  EVIDENCE_FRONTIER_CAP_TEMPERATURE="${EVIDENCE_FRONTIER_CAP_TEMPERATURE:-0.10}" \
  EVIDENCE_ADMISSION_SCALE="${EVIDENCE_ADMISSION_SCALE:-2.0}" EVIDENCE_ADMISSION_BOUNDED=false \
  ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=false \
  ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT=0 \
  ORDINAL_EVIDENCE_COMPONENT_TAIL_WEIGHT=0 \
  ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_WEIGHT=0 \
  ORDINAL_EVIDENCE_BENEFIT_MARGIN_REGRESSION_WEIGHT=0 \
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
    bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh
else
  rm -rf "$FINAL_RUN/model_v48_trac_sr"
  for item in model_v48_trac_sr STAGE_ARCHITECTURE.json POLICY_CONTRACT.env TRAINING_COMPLETE.json EVIDENCE_CORRECTION_COMPLETE.json; do
    [[ -e "$IDENTITY_RUN/$item" ]] || { echo "missing identity artifact $IDENTITY_RUN/$item" >&2; exit 30; }
    cp -a "$IDENTITY_RUN/$item" "$FINAL_RUN/$item"
  done
fi

fi

FINAL_CKPT="$FINAL_RUN/model_v48_trac_sr/best.pt"
transfer_extra=()
final_allowed_prefixes=direct_evidence_concord_admission_calibrator
if [[ "$RFR_RESERVE_ONLY" == 1 ]]; then
  transfer_extra+=(--identity-stage-skipped --final-stage-disabled)
  final_allowed_prefixes=""
else
  [[ "$ENABLE_FINAL_CALIBRATION" == 1 ]] || transfer_extra+=(--final-stage-disabled)
fi
IMPLEMENTATION_VERSION="${OCRAP_IMPLEMENTATION_VERSION:-v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX}"
CURRENT_STAGE="stage_transfer_integrity"
python tools/check_v48_36_stage_transfer.py \
  --factor "$FACTOR_CKPT" --identity "$IDENTITY_CKPT" --final "$FINAL_CKPT" \
  --identity-architecture "$IDENTITY_RUN/STAGE_ARCHITECTURE.json" \
  --final-architecture "$FINAL_RUN/STAGE_ARCHITECTURE.json" \
  --identity-allowed-prefixes "$identity_prefixes" \
  --final-allowed-prefixes "$final_allowed_prefixes" \
  --implementation-version "$IMPLEMENTATION_VERSION" \
  --output "$FINAL_RUN/STAGE_TRANSFER_INTEGRITY.json" "${transfer_extra[@]}"

CURRENT_STAGE="completion_metadata"
python tools/finalize_v48_36_adaptation_variant.py \
  --run "$FINAL_RUN" --source "$SOURCE_CKPT" \
  --factor "$FACTOR_CKPT" --identity "$IDENTITY_CKPT" --final "$FINAL_CKPT" \
  --support "$SUPPORT_JSON" \
  --support-reliability-enabled "$ENABLE_SUPPORT_RELIABILITY" \
  --identity-train-all "$IDENTITY_TRAIN_ALL" \
  --factor-preserving-identity "$HAF_FACTOR_PRESERVE" \
  --reserve-only "$RFR_RESERVE_ONLY" \
  --prior-coupled "$COUPLE_ADMISSION_PRIOR" \
  --adaptive-margin "$ADAPTIVE_IDENTITY_MARGIN" \
  --final-enabled "$ENABLE_FINAL_CALIBRATION" \
  --context-source "$EVIDENCE_CONTEXT_SOURCE" \
  --consensus-prior-scale "${EVIDENCE_CONSENSUS_PRIOR_SCALE:-0.50}" \
  --interaction-hidden "${EVIDENCE_INTERACTION_HIDDEN:-64}" \
  --interaction-dropout "${EVIDENCE_INTERACTION_DROPOUT:-0.05}" \
  --admission-prior-mode "${EVIDENCE_ADMISSION_PRIOR_MODE:-frontier_capped_slack}" \
  --implementation-version "$IMPLEMENTATION_VERSION"
CURRENT_STAGE="complete"
rm -f "$FINAL_RUN/VARIANT_STAGE_FAILED.json"
