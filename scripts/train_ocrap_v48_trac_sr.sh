#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Two concurrent A30 jobs share a Xeon Gold 5220R. Limit intra-op pools so
# dataloader workers do not oversubscribe the CPU.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

TRAIN_OCRAP_ROOT="${TRAIN_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP_v48_train}"
EVAL_OCRAP_ROOT="${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
TRAIN_MIX="${TRAIN_MIX:-$TRAIN_OCRAP_ROOT/train_near_contact,$TRAIN_OCRAP_ROOT/train_contact}"
VAL_MIX="${VAL_MIX:-$EVAL_OCRAP_ROOT/val_near_contact,$EVAL_OCRAP_ROOT/val_contact}"
CAL_MIX="${CAL_MIX:-$VAL_MIX}"
ALLOW_SCRATCH_INIT="${ALLOW_SCRATCH_INIT:-0}"
if [[ "$ALLOW_SCRATCH_INIT" == 1 ]]; then
  # Explicit scratch mode is used only by the one-off shared-source rebuild.
  # An empty INIT_CKPT is intentional; normal adaptation remains fail-closed.
  INIT_CKPT="${INIT_CKPT-}"
  ENCODER_ANCHOR_WEIGHT="${ENCODER_ANCHOR_WEIGHT:-0}"
else
  INIT_CKPT="${INIT_CKPT:-runs/ocrap_v47_trac_balanced/model_v47_trac/best.pt}"
  ENCODER_ANCHOR_WEIGHT="${ENCODER_ANCHOR_WEIGHT:-0.02}"
fi
VARIANT="${VARIANT:-balanced}"
RUN="${RUN:-runs/ocrap_v48_trac_sr_${VARIANT}}"
MODEL_DIR="${MODEL_DIR:-$RUN/model_v48_trac_sr}"
CAL_DIR="${CAL_DIR:-$RUN/calibration}"
LOG_DIR="${LOG_DIR:-$RUN/logs}"
TRAIN_GPU="${TRAIN_GPU:-0}"
GROUP_INDEX="${GROUP_INDEX:-$RUN/teacher_pcd_train_index.jsonl}"
mkdir -p "$MODEL_DIR" "$CAL_DIR" "$LOG_DIR"
if [[ -n "$INIT_CKPT" ]]; then
  [[ -f "$INIT_CKPT" ]] || { echo "missing INIT_CKPT=$INIT_CKPT" >&2; exit 2; }
elif [[ "$ALLOW_SCRATCH_INIT" != 1 ]]; then
  echo "empty INIT_CKPT requires ALLOW_SCRATCH_INIT=1" >&2
  exit 2
fi
[[ -f "$GROUP_INDEX" ]] || { echo "missing GROUP_INDEX=$GROUP_INDEX" >&2; exit 2; }

case "$VARIANT" in
  balanced)
    LR="${LR:-0.00012}"; ENCODER_LR_SCALE="${ENCODER_LR_SCALE:-0.18}"
    POS_W="${POS_W:-6.0}"; FP_W="${FP_W:-2.2}"; HARM_W="${HARM_W:-0.10}"
    SETWISE_W="${SETWISE_W:-0.40}"; DISAGREE="${DISAGREE:-0.50}" ;;
  precision)
    LR="${LR:-0.00009}"; ENCODER_LR_SCALE="${ENCODER_LR_SCALE:-0.12}"
    POS_W="${POS_W:-5.0}"; FP_W="${FP_W:-3.5}"; HARM_W="${HARM_W:-0.15}"
    SETWISE_W="${SETWISE_W:-0.60}"; DISAGREE="${DISAGREE:-0.70}" ;;
  *) echo "unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

# No encoder freeze.  v47's candidate AUC showed learnable signal, but its
# frozen representation could not turn that signal into a correct setwise top-1.
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u -m ocrap.cli train \
  --dataset "$TRAIN_MIX" --val-dataset "$VAL_MIX" --output "$MODEL_DIR" \
  --set seed="${SEED:-7}" \
  --set training.init_checkpoint="$INIT_CKPT" \
  --set training.freeze_param_prefixes="${FREEZE_PARAM_PREFIXES:-}" \
  --set training.trainable_param_prefixes="${TRAINABLE_PARAM_PREFIXES:-}" \
  --set training.strict_init_prefixes="${STRICT_INIT_PREFIXES:-}" \
  --set training.strict_init_allowed_missing_prefixes="${STRICT_INIT_ALLOWED_MISSING_PREFIXES:-}" \
  --set training.direct_only_fast_path="${DIRECT_ONLY_FAST_PATH:-true}" \
  --set training.witness_fast_path="${WITNESS_FAST_PATH:-}" \
  --set training.frozen_modules_eval="${FROZEN_MODULES_EVAL:-false}" \
  --set training.group_index_path="$GROUP_INDEX" \
  --set training.validation_group_index_path="${VAL_GROUP_INDEX:-}" \
  --set training.epochs="${EPOCHS:-12}" --set training.early_stop_patience="${PATIENCE:-3}" \
  --set training.batch_size="${BATCH_SIZE:-96}" --set training.lr="$LR" \
  --set training.encoder_lr_scale="$ENCODER_LR_SCALE" \
  --set training.encoder_anchor_weight="$ENCODER_ANCHOR_WEIGHT" \
  --set training.weight_decay=0.00002 \
  --set training.grad_clip=3.0 --set training.require_cuda=true \
  --set training.amp=true --set training.amp_dtype=bfloat16 \
  --set training.allow_tf32=true --set training.matmul_precision=high \
  --set training.cudnn_benchmark="${CUDNN_BENCHMARK:-true}" --set training.pin_memory=true \
  --set training.deterministic_algorithms="${DETERMINISTIC_ALGORITHMS:-false}" \
  --set training.num_workers="${NUM_WORKERS:-6}" --set training.persistent_workers=true \
  --set training.prefetch_factor="${PREFETCH_FACTOR:-2}" --set training.cache_samples_in_memory="${CACHE_SAMPLES_IN_MEMORY:-false}" \
  --set training.persistent_tensor_cache="${PERSISTENT_TENSOR_CACHE:-false}" --set training.persistent_tensor_cache_dir="${PERSISTENT_TENSOR_CACHE_DIR:-}" \
  --set training.persistent_tensor_cache_build_workers="${PERSISTENT_TENSOR_CACHE_BUILD_WORKERS:-1}" \
  --set training.option_execution_semantics="${OPTION_EXECUTION_SEMANTICS:-global}" \
  --set training.decision_weighted_obs_enabled="${DECISION_WEIGHTED_OBS_ENABLED:-false}" \
  --set training.decision_weighted_obs_gamma="${DECISION_WEIGHTED_OBS_GAMMA:-0.0}" \
  --set training.decision_weighted_obs_temperature="${DECISION_WEIGHTED_OBS_TEMPERATURE:-0.20}" \
  --set training.decision_weighted_obs_conflict_scale="${DECISION_WEIGHTED_OBS_CONFLICT_SCALE:-3.0}" \
  --set training.decision_weighted_obs_max_weight="${DECISION_WEIGHTED_OBS_MAX_WEIGHT:-4.0}" \
  --set training.recovery_frontier_option_temperature="${RECOVERY_FRONTIER_OPTION_TEMPERATURE:-0.35}" \
  --set training.recovery_frontier_deployability_tolerance="${RECOVERY_FRONTIER_DEPLOYABILITY_TOLERANCE:-0.05}" \
  --set training.recovery_frontier_drs_tolerance="${RECOVERY_FRONTIER_DRS_TOLERANCE:-0.05}" \
  --set training.recovery_frontier_gap_tolerance="${RECOVERY_FRONTIER_GAP_TOLERANCE:-0.05}" \
  --set training.recovery_frontier_positive_gain="${RECOVERY_FRONTIER_POSITIVE_GAIN:-0.015}" \
  --set training.recovery_frontier_pcd_weight="${RECOVERY_FRONTIER_PCD_WEIGHT:-1.0}" \
  --set training.recovery_frontier_decision_equivalent="${RECOVERY_FRONTIER_DECISION_EQUIVALENT:-false}" \
  --set training.recovery_frontier_boundary_complete="${RECOVERY_FRONTIER_BOUNDARY_COMPLETE:-false}" \
  --set training.recovery_frontier_physical_teacher_sign_alignment="${RECOVERY_FRONTIER_PHYSICAL_TEACHER_SIGN_ALIGNMENT:-false}" \
  --set training.recovery_frontier_physical_student_sign_alignment="${RECOVERY_FRONTIER_PHYSICAL_STUDENT_SIGN_ALIGNMENT:-false}" \
  --set training.invariant_physical_boundary_distillation="${INVARIANT_PHYSICAL_BOUNDARY_DISTILLATION:-false}" \
  --set training.recovery_frontier_sign_temperature="${RECOVERY_FRONTIER_SIGN_TEMPERATURE:-0.08}" \
  --set training.recovery_frontier_regression_weight="${RECOVERY_FRONTIER_REGRESSION_WEIGHT:-1.0}" \
  --set training.recovery_frontier_sign_weight="${RECOVERY_FRONTIER_SIGN_WEIGHT:-0.50}" \
  --set training.progress=true \
  --set training.save_every_epoch="${SAVE_EVERY_EPOCH:-true}" --set training.save_latest="${SAVE_LATEST:-true}" --set training.best_metric="${BEST_METRIC:-direct_policy_risk_fold_worst}" \
  --set training.best_metric_mode=min --set training.best_metric_min_delta="${BEST_METRIC_MIN_DELTA:-0.000001}" \
  --set training.evaluate_initial_checkpoint="${EVALUATE_INITIAL_CHECKPOINT:-false}" \
  --set training.group_batching=true --set training.group_batching_replacement="${GROUP_BATCHING_REPLACEMENT:-true}" \
  --set training.group_batch_stratified="${GROUP_BATCH_STRATIFIED:-false}" \
  --set training.group_batch_positive_fraction="${GROUP_BATCH_POSITIVE_FRACTION:-0.30}" \
  --set training.group_batch_safe_positive_target="${GROUP_BATCH_SAFE_POSITIVE_TARGET:-${ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET:-false}}" \
  --set training.group_batch_harmful_fraction="${GROUP_BATCH_HARMFUL_FRACTION:-0.35}" \
  --set training.group_batch_dead_fraction="${GROUP_BATCH_DEAD_FRACTION:-0.35}" \
  --set training.group_batch_negative_advantage_gain_max="${NEGATIVE_GAIN:-0.010}" \
  --set training.group_batch_positive_advantage_boost="${POSITIVE_GROUP_BOOST:-5.0}" \
  --set training.group_batch_positive_best_macro_balance_power="${POSITIVE_MACRO_BALANCE_POWER:-0.5}" \
  --set training.group_batch_positive_advantage_gain_min="${POSITIVE_GAIN:-0.015}" \
  --set training.group_batch_positive_advantage_macro_ids=2,3,5,6,7 \
  --set training.group_batch_positive_advantage_bucket_ids=1,2 \
  --set training.group_batch_require_positive_advantage_groups=true \
  --set training.group_batch_scene_balance_power="${SCENE_BALANCE_POWER:-0.50}" \
  --set training.group_batch_hard_boost=0 \
  --set training.artifact_sampler_weight=0 --set training.negative_deployable_sampler_weight=0 \
  --set training.safe_positive_sampler_weight=0 --set training.regime_balance_power=1.0 \
  --set model.encoder_type=structured_transformer --set model.d_model=192 --set model.d_obs=64 \
  --set model.transformer_layers=2 --set model.transformer_heads=4 \
  --set model.direct_recovery_value_head=true \
  --set model.direct_recovery_value_pooling=candidate_concat_raw \
  --set model.direct_recovery_value_output=score \
  --set model.direct_recovery_value_regime_conditioning=false \
  --set model.direct_recovery_value_experts=true \
  --set model.direct_recovery_value_num_experts=2 \
  --set model.direct_recovery_value_expert_routing=robust_ensemble \
  --set model.direct_recovery_expert_disagreement_penalty="$DISAGREE" \
  --set model.direct_recovery_opportunity_head=true \
  --set model.direct_recovery_harm_head=true \
  --set model.direct_recovery_set_context="${SET_CONTEXT_ENABLED:-false}" \
  --set model.direct_recovery_set_context_hidden="${SET_CONTEXT_HIDDEN:-192}" \
  --set model.direct_recovery_set_context_dropout="${SET_CONTEXT_DROPOUT:-0.05}" \
  --set model.direct_recovery_preference_head="${PREFERENCE_HEAD_ENABLED:-true}" \
  --set model.direct_recovery_preference_hidden="${PREFERENCE_HIDDEN:-96}" \
  --set model.direct_recovery_preference_dropout="${PREFERENCE_DROPOUT:-0.05}" \
  --set model.direct_recovery_preference_context="${PREFERENCE_CONTEXT_ENABLED:-true}" \
  --set model.direct_recovery_preference_context_hidden="${PREFERENCE_CONTEXT_HIDDEN:-128}" \
  --set model.direct_recovery_relative_features_include_absolute="${RELATIVE_INCLUDE_ABSOLUTE:-true}" \
  --set model.direct_recovery_set_tournament="${SET_TOURNAMENT_ENABLED:-false}" \
  --set model.direct_recovery_set_tournament_hidden="${SET_TOURNAMENT_HIDDEN:-48}" \
  --set model.direct_recovery_set_tournament_heads="${SET_TOURNAMENT_HEADS:-4}" \
  --set model.direct_recovery_set_tournament_dropout="${SET_TOURNAMENT_DROPOUT:-0.05}" \
  --set model.direct_recovery_set_tournament_replace_base="${SET_TOURNAMENT_REPLACE_BASE:-true}" \
  --set model.direct_recovery_delta_head="${DELTA_HEAD_ENABLED:-true}" \
  --set model.direct_recovery_delta_regime_experts="${DELTA_REGIME_EXPERTS:-false}" \
  --set model.direct_recovery_delta_policy_features="${DELTA_POLICY_FEATURES:-false}" \
  --set model.direct_recovery_delta_hidden="${DELTA_HIDDEN:-128}" \
  --set model.direct_recovery_delta_dropout="${DELTA_DROPOUT:-0.05}" \
  --set model.direct_recovery_delta_initial_logvar="${DELTA_INITIAL_LOGVAR:--4.605170186}" \
  --set model.direct_recovery_delta_mode="${DELTA_MODE:-gaussian}" \
  --set model.direct_recovery_evidence_calibrator="${EVIDENCE_CALIBRATOR_ENABLED:-false}" \
  --set model.direct_recovery_evidence_calibrator_hidden="${EVIDENCE_CALIBRATOR_HIDDEN:-8}" \
  --set model.direct_recovery_evidence_calibrator_scale="${EVIDENCE_CALIBRATOR_SCALE:-0.25}" \
  --set model.direct_recovery_evidence_calibrator_mode="${EVIDENCE_CALIBRATOR_MODE:-center_width}" \
  --set model.direct_recovery_evidence_calibrator_context="${EVIDENCE_CALIBRATOR_CONTEXT:-false}" \
  --set model.direct_recovery_evidence_calibrator_context_detach="${EVIDENCE_CALIBRATOR_CONTEXT_DETACH:-true}" \
  --set model.direct_recovery_evidence_calibrator_context_source="${EVIDENCE_CALIBRATOR_CONTEXT_SOURCE:-relative}" \
  --set model.direct_recovery_evidence_interaction_hidden="${EVIDENCE_INTERACTION_HIDDEN:-64}" \
  --set model.direct_recovery_evidence_interaction_dropout="${EVIDENCE_INTERACTION_DROPOUT:-0.05}" \
  --set model.direct_recovery_evidence_dual_interaction_bridge="${EVIDENCE_DUAL_INTERACTION_BRIDGE:-false}" \
  --set model.direct_recovery_evidence_factorized_harm_interaction="${EVIDENCE_FACTORIZED_HARM_INTERACTION:-false}" \
  --set model.direct_recovery_evidence_partial_pool_harm_residual="${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL:-false}" \
  --set model.direct_recovery_evidence_partial_pool_harm_residual_scale="${EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL_SCALE:-0.50}" \
  --set model.direct_recovery_evidence_rank_benefit_skip="${EVIDENCE_RANK_BENEFIT_SKIP:-false}" \
  --set model.direct_recovery_evidence_rank_benefit_gain_init="${EVIDENCE_RANK_BENEFIT_GAIN_INIT:-1.0}" \
  --set model.direct_recovery_evidence_postprefix_obs_transport_benefit="${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT:-false}" \
  --set model.direct_recovery_evidence_postprefix_obs_transport_harm="${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM:-false}" \
  --set model.direct_recovery_evidence_postprefix_obs_transport_scale="${EVIDENCE_POSTPREFIX_OBS_TRANSPORT_SCALE:-1.0}" \
  --set model.direct_recovery_evidence_roct_benefit="${EVIDENCE_ROCT_BENEFIT:-false}" \
  --set model.direct_recovery_evidence_roct_deployability="${EVIDENCE_ROCT_DEPLOYABILITY:-false}" \
  --set model.direct_recovery_evidence_roct_scale="${EVIDENCE_ROCT_SCALE:-1.0}" \
  --set model.direct_recovery_evidence_roct_alpha="${EVIDENCE_ROCT_ALPHA:-0.2}" \
  --set model.direct_recovery_evidence_roct_beta="${EVIDENCE_ROCT_BETA:-0.2}" \
  --set model.direct_recovery_evidence_roct_top_m="${EVIDENCE_ROCT_TOP_M:-8}" \
  --set model.direct_recovery_evidence_roct_option_temperature="${EVIDENCE_ROCT_OPTION_TEMPERATURE:-0.35}" \
  --set model.direct_recovery_evidence_common_measure_root_mass="${EVIDENCE_COMMON_MEASURE_ROOT_MASS:-false}" \
  --set model.direct_recovery_absolute_feasibility_head="${ABSOLUTE_FEASIBILITY_HEAD:-false}" \
  --set model.direct_recovery_absolute_option_margin_correction="${ABSOLUTE_OPTION_MARGIN_CORRECTION:-false}" \
  --set model.direct_recovery_absolute_physical_headroom_correction="${ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION:-false}" \
  --set model.direct_recovery_absolute_executable_witness_correction="${ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION:-false}" \
  --set model.direct_recovery_absolute_common_witness_correction="${ABSOLUTE_COMMON_WITNESS_CORRECTION:-false}" \
  --set model.direct_recovery_absolute_quantifier_witness_correction="${ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION:-false}" \
  --set model.direct_recovery_absolute_semantic_witness_correction="${ABSOLUTE_SEMANTIC_WITNESS_CORRECTION:-false}" \
  --set model.direct_recovery_semantic_witness_active_set_alignment="${SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT:-true}" \
  --set model.direct_recovery_semantic_witness_path_stop_alignment="${SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT:-true}" \
  --set model.direct_recovery_semantic_witness_classlocal_transport="${SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT:-false}" \
  --set model.direct_recovery_semantic_witness_route_alignment="${SEMANTIC_WITNESS_ROUTE_ALIGNMENT:-false}" \
  --set model.direct_recovery_semantic_witness_reentry_alignment="${SEMANTIC_WITNESS_REENTRY_ALIGNMENT:-false}" \
  --set model.direct_recovery_semantic_witness_control_projection="${SEMANTIC_WITNESS_CONTROL_PROJECTION:-false}" \
  --set model.direct_recovery_semantic_witness_boundary_transport="${SEMANTIC_WITNESS_BOUNDARY_TRANSPORT:-false}" \
  --set model.direct_recovery_semantic_witness_projection_fidelity_weighting="${SEMANTIC_WITNESS_PROJECTION_FIDELITY:-false}" \
  --set model.direct_recovery_semantic_witness_active_constraint_typed_source="${SEMANTIC_WITNESS_ACTIVE_CONSTRAINT_TYPED_SOURCE:-false}" \
  --set model.direct_recovery_semantic_witness_root_tail_source="${SEMANTIC_WITNESS_ROOT_TAIL_SOURCE:-false}" \
  --set model.direct_recovery_semantic_witness_structured_tail_field="${SEMANTIC_WITNESS_STRUCTURED_TAIL_FIELD:-false}" \
  --set model.direct_recovery_semantic_witness_signed_tail_channels="${SEMANTIC_WITNESS_SIGNED_TAIL_CHANNELS:-false}" \
  --set model.direct_recovery_semantic_witness_counterfactual_tail_response="${SEMANTIC_WITNESS_COUNTERFACTUAL_TAIL_RESPONSE:-false}" \
  --set model.direct_recovery_semantic_witness_tail_localization="${SEMANTIC_WITNESS_TAIL_LOCALIZATION:-false}" \
  --set model.direct_recovery_semantic_witness_demand_normalized_fidelity="${SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY:-false}" \
  --set model.direct_recovery_semantic_witness_robust_occupancy="${SEMANTIC_WITNESS_ROBUST_OCCUPANCY:-false}" \
  --set model.direct_recovery_semantic_witness_soft_occupancy_disagreement="${SEMANTIC_WITNESS_SOFT_OCCUPANCY_DISAGREEMENT:-false}" \
  --set model.direct_recovery_semantic_witness_boundary_localized_occupancy_trust="${SEMANTIC_WITNESS_BOUNDARY_LOCALIZED_OCCUPANCY_TRUST:-false}" \
  --set model.direct_recovery_semantic_witness_history_occupancy_reachability="${SEMANTIC_WITNESS_HISTORY_OCCUPANCY_REACHABILITY:-false}" \
  --set model.direct_recovery_semantic_witness_interaction_box_support="${SEMANTIC_WITNESS_INTERACTION_BOX_SUPPORT:-false}" \
  --set model.direct_recovery_semantic_witness_interaction_hull_support="${SEMANTIC_WITNESS_INTERACTION_HULL_SUPPORT:-false}" \
  --set model.direct_recovery_semantic_witness_interaction_anchor_support="${SEMANTIC_WITNESS_INTERACTION_ANCHOR_SUPPORT:-false}" \
  --set model.direct_recovery_semantic_witness_interaction_response_support="${SEMANTIC_WITNESS_INTERACTION_RESPONSE_SUPPORT:-false}" \
  --set model.direct_recovery_evidence_native_certificate_preservation="${EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION:-false}" \
  --set model.direct_recovery_evidence_native_margin_complete_preservation="${EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION:-false}" \
  --set model.direct_recovery_evidence_native_advantage_preservation="${EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION:-false}" \
  --set model.direct_recovery_evidence_native_exact_advantage_preservation="${EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION:-false}" \
  --set model.direct_recovery_evidence_native_boundary_complete_advantage_preservation="${EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION:-false}" \
  --set model.direct_recovery_evidence_physical_student_drs="${EVIDENCE_PHYSICAL_STUDENT_DRS:-false}" \
  --set model.direct_recovery_evidence_native_drs_tolerance="${EVIDENCE_NATIVE_DRS_TOLERANCE:-0.05}" \
  --set model.direct_recovery_evidence_native_deployability_tolerance="${EVIDENCE_NATIVE_DEPLOYABILITY_TOLERANCE:-0.05}" \
  --set model.direct_recovery_evidence_native_dep_boundary_aligned="${EVIDENCE_DEP_BOUNDARY_ALIGNED:-false}" \
  --set model.direct_recovery_evidence_native_gap_tolerance="${EVIDENCE_NATIVE_GAP_TOLERANCE:-0.05}" \
  --set model.direct_recovery_evidence_native_positive_gain="${EVIDENCE_NATIVE_POSITIVE_GAIN:-${POSITIVE_GAIN:-0.015}}" \
  --set model.direct_recovery_evidence_calibrator_shared="${EVIDENCE_CALIBRATOR_SHARED:-false}" \
  --set model.direct_recovery_evidence_calibrator_regime_scale="${EVIDENCE_CALIBRATOR_REGIME_SCALE:-0.25}" \
  --set model.direct_recovery_evidence_unified_experts="${EVIDENCE_UNIFIED_EXPERTS:-false}" \
  --set model.direct_recovery_evidence_component_heads="${EVIDENCE_COMPONENT_HEADS:-false}" \
  --set model.direct_recovery_evidence_component_count="${EVIDENCE_COMPONENT_COUNT:-3}" \
  --set model.direct_recovery_evidence_component_scale="${EVIDENCE_COMPONENT_SCALE:-2.0}" \
  --set model.direct_recovery_evidence_benefit_residual_scale="${EVIDENCE_BENEFIT_RESIDUAL_SCALE:-1.0}" \
  --set model.direct_recovery_evidence_unbounded_benefit_factor="${EVIDENCE_UNBOUNDED_BENEFIT_FACTOR:-false}" \
  --set model.direct_recovery_evidence_unbounded_harm_factors="${EVIDENCE_UNBOUNDED_HARM_FACTORS:-false}" \
  --set model.direct_recovery_evidence_component_reliability="${EVIDENCE_COMPONENT_RELIABILITY:-}" \
  --set model.direct_recovery_evidence_concord="${EVIDENCE_CONCORD:-false}" \
  --set model.direct_recovery_evidence_consensus_disagreement_penalty="${EVIDENCE_CONSENSUS_DISAGREEMENT_PENALTY:-0.15}" \
  --set model.direct_recovery_evidence_consensus_prior_scale="${EVIDENCE_CONSENSUS_PRIOR_SCALE:-1.0}" \
  --set model.direct_recovery_evidence_admission_head="${EVIDENCE_ADMISSION_HEAD:-false}" \
  --set model.direct_recovery_evidence_admission_scale="${EVIDENCE_ADMISSION_SCALE:-2.0}" \
  --set model.direct_recovery_evidence_admission_bounded="${EVIDENCE_ADMISSION_BOUNDED:-true}" \
  --set model.direct_recovery_evidence_admission_prior_detach="${EVIDENCE_ADMISSION_PRIOR_DETACH:-true}" \
  --set model.direct_recovery_evidence_admission_prior_mode="${EVIDENCE_ADMISSION_PRIOR_MODE:-risk_centered}" \
  --set model.direct_recovery_evidence_slack_temperature="${EVIDENCE_SLACK_TEMPERATURE:-0.025}" \
  --set model.direct_recovery_evidence_slack_penalty="${EVIDENCE_SLACK_PENALTY:-1.0}" \
  --set model.direct_recovery_evidence_frontier_cap_temperature="${EVIDENCE_FRONTIER_CAP_TEMPERATURE:-0.10}" \
  --set model.direct_recovery_evidence_benefit_margin_temperature="${EVIDENCE_BENEFIT_MARGIN_TEMPERATURE:-${ORDINAL_EVIDENCE_BENEFIT_MARGIN_TEMPERATURE:-0.025}}" \
  --set model.direct_recovery_evidence_joint_reserve_temperature="${EVIDENCE_JOINT_RESERVE_TEMPERATURE:-0.025}" \
  --set model.direct_recovery_evidence_reserve_factor_alignment="${EVIDENCE_RESERVE_FACTOR_ALIGNMENT:-false}" \
  --set model.direct_recovery_evidence_frontier="${EVIDENCE_FRONTIER:-false}" \
  --set model.direct_recovery_evidence_component_prior_logit="${EVIDENCE_COMPONENT_PRIOR_LOGIT:--2.0}" \
  --set loss_weights.dep="${LOSS_DEP:-0}" --set loss_weights.orc="${LOSS_ORC:-0}" --set loss_weights.assign="${LOSS_ASSIGN:-0}" \
  --set loss_weights.sig="${LOSS_SIG:-0}" --set loss_weights.margin="${LOSS_MARGIN:-0}" --set loss_weights.obs="${LOSS_OBS:-0}" \
  --set loss_weights.anti_oracle="${LOSS_ANTI_ORACLE:-0}" --set loss_weights.artifact_gap="${LOSS_ARTIFACT_GAP:-0}" \
  --set loss_weights.admission="${LOSS_ADMISSION:-0}" --set loss_weights.utility="${LOSS_UTILITY:-0}" \
  --set loss_weights.option_q="${LOSS_OPTION_Q:-0}" --set loss_weights.option_admission="${LOSS_OPTION_ADMISSION:-0}" \
  --set loss_weights.option_success="${LOSS_OPTION_SUCCESS:-0}" --set loss_weights.option_success_bce="${LOSS_OPTION_SUCCESS_BCE:-0}" \
  --set loss_weights.option_best="${LOSS_OPTION_BEST:-0}" \
  --set loss_weights.option_class_success="${LOSS_OPTION_CLASS_SUCCESS:-0}" --set loss_weights.option_class_best="${LOSS_OPTION_CLASS_BEST:-0}" \
  --set loss_weights.recovery_frontier="${LOSS_RECOVERY_FRONTIER:-0}" \
  --set loss_weights.physical_boundary_distill="${LOSS_PHYSICAL_BOUNDARY_DISTILL:-0}" \
  --set loss_weights.group_ce="${LOSS_GROUP_CE:-0}" \
  --set loss_weights.group_distill="${LOSS_GROUP_DISTILL:-0}" --set loss_weights.nominal_switch="${LOSS_NOMINAL_SWITCH:-0}" \
  --set loss_weights.safe_nominal="${LOSS_SAFE_NOMINAL:-0}" --set loss_weights.protective_macro="${LOSS_PROTECTIVE_MACRO:-0}" \
  --set loss_weights.macro_drs="${LOSS_MACRO_DRS:-0}" --set loss_weights.teacher_pcd_direct="${LOSS_TEACHER_PCD_DIRECT:-0}" \
  --set loss_weights.recovery_advantage="${LOSS_RECOVERY_ADVANTAGE:-0}" --set loss_weights.direct_router_balance="${LOSS_DIRECT_ROUTER_BALANCE:-0}" \
  --set loss_weights.direct_recovery_value="${DIRECT_VALUE_WEIGHT:-10.0}" \
  --set training.direct_value_macro_ids=2,3,5,6,7 --set training.direct_value_bucket_ids=1,2 \
  --set training.direct_value_temperature="${DIRECT_TEMPERATURE:-0.10}" \
  --set training.direct_value_positive_gain="${POSITIVE_GAIN:-0.015}" \
  --set training.direct_value_negative_gain="${NEGATIVE_GAIN:-0.010}" \
  --set training.direct_value_rank_margin="${RANK_MARGIN:-0.020}" \
  --set training.direct_value_point_weight="${POINT_WEIGHT:-0.05}" \
  --set training.direct_value_listwise_weight="${VALUE_LISTWISE_WEIGHT:-0.15}" \
  --set training.direct_value_centered_weight="${CENTERED_WEIGHT:-1.0}" \
  --set training.direct_value_advantage_weight="${ADVANTAGE_WEIGHT:-1.0}" \
  --set training.direct_value_output_mode=score \
  --set training.direct_value_pairwise_weight="${LEGACY_PAIRWISE_WEIGHT:-0.0}" \
  --set training.direct_value_top_rank_weight="${LEGACY_TOP_RANK_WEIGHT:-0.0}" \
  --set training.direct_value_positive_group_weight="$POS_W" \
  --set training.direct_value_negative_group_weight=1.5 \
  --set training.direct_value_ambiguous_group_weight=0.08 \
  --set training.direct_value_near_weight=1.4 \
  --set training.direct_value_contact_weight=1.6 \
  --set training.direct_value_min_group_range=0.004 \
  --set training.direct_value_exact_teacher_pcd="${EXACT_TEACHER_PCD:-true}" \
  --set training.direct_value_preference_weight="${PREFERENCE_WEIGHT:-1.50}" \
  --set training.direct_value_preference_temperature="${PREFERENCE_TEMPERATURE:-0.06}" \
  --set training.direct_value_preference_min_gap="${PREFERENCE_MIN_GAP:-0.010}" \
  --set training.direct_value_preference_margin="${PREFERENCE_MARGIN:-0.030}" \
  --set training.direct_value_preference_confidence_scale="${PREFERENCE_CONFIDENCE_SCALE:-0.040}" \
  --set training.direct_value_preference_regret_weight="${PREFERENCE_REGRET_WEIGHT:-0.50}" \
  --set training.direct_value_preference_listwise_weight="${PREFERENCE_LISTWISE_WEIGHT:-0.75}" \
  --set training.direct_value_preference_gap_weight="${PREFERENCE_GAP_WEIGHT:-0.25}" \
  --set training.direct_value_preference_set_weight="${PREFERENCE_SET_WEIGHT:-0.0}" \
  --set training.direct_value_preference_set_margin="${PREFERENCE_SET_MARGIN:-0.020}" \
  --set training.direct_value_preference_tie_epsilon_near="${PREFERENCE_TIE_EPS_NEAR:-0.025}" \
  --set training.direct_value_preference_tie_epsilon_contact="${PREFERENCE_TIE_EPS_CONTACT:-0.010}" \
  --set training.direct_value_preference_all_group_set_weight="${PREFERENCE_ALL_GROUP_SET_WEIGHT:-0.0}" \
  --set training.direct_value_preference_set_replace_singlewinner="${PREFERENCE_SET_REPLACE_SINGLEWINNER:-false}" \
  --set training.direct_value_preference_nominal_margin="${PREFERENCE_NOMINAL_MARGIN:-0.020}" \
  --set training.direct_value_preference_harm_margin="${PREFERENCE_HARM_MARGIN:-0.030}" \
  --set training.direct_value_preference_set_mass_loss="${PREFERENCE_SET_MASS_LOSS:-false}" \
  --set training.direct_value_preference_noop_nominal_only="${PREFERENCE_NOOP_NOMINAL_ONLY:-false}" \
  --set training.direct_value_preference_deadzone_margin="${PREFERENCE_DEADZONE_MARGIN:-0.008}" \
  --set training.direct_value_preference_conditional_set_weight="${PREFERENCE_CONDITIONAL_SET_WEIGHT:-0.0}" \
  --set training.direct_value_preference_conditional_noop_weight="${PREFERENCE_CONDITIONAL_NOOP_WEIGHT:-0.35}" \
  --set training.direct_value_preference_conditional_regret_weight="${PREFERENCE_CONDITIONAL_REGRET_WEIGHT:-0.50}" \
  --set training.direct_value_preference_conditional_pairwise_weight="${PREFERENCE_CONDITIONAL_PAIRWISE_WEIGHT:-0.0}" \
  --set training.direct_value_preference_conditional_pairwise_min_gap="${PREFERENCE_CONDITIONAL_PAIRWISE_MIN_GAP:-0.01}" \
  --set training.direct_value_preference_conditional_pairwise_margin="${PREFERENCE_CONDITIONAL_PAIRWISE_MARGIN:-0.02}" \
  --set training.direct_value_preference_proposal_topk_weight="${PREFERENCE_PROPOSAL_TOPK_WEIGHT:-0.0}" \
  --set training.direct_value_preference_proposal_topk="${PREFERENCE_PROPOSAL_TOPK:-3}" \
  --set training.direct_value_preference_proposal_margin="${PREFERENCE_PROPOSAL_MARGIN:-0.02}" \
  --set training.direct_value_preference_conditional_mode="${PREFERENCE_CONDITIONAL_MODE:-false}" \
  --set training.direct_value_delta_nll_weight="${DELTA_NLL_WEIGHT:-1.00}" \
  --set training.direct_value_delta_sign_weight="${DELTA_SIGN_WEIGHT:-0.0}" \
  --set training.direct_value_delta_sign_temperature="${DELTA_SIGN_TEMPERATURE:-0.04}" \
  --set training.direct_value_certificate_policy_top1_weight="${CERTIFICATE_POLICY_TOP1_WEIGHT:-0.0}" \
  --set training.direct_value_certificate_policy_top1_sign_weight="${CERTIFICATE_POLICY_TOP1_SIGN_WEIGHT:-0.0}" \
  --set training.direct_value_certificate_policy_top1_temperature="${CERTIFICATE_POLICY_TOP1_TEMPERATURE:-0.04}" \
  --set training.direct_value_ordinal_evidence_policy_top1_weight="${ORDINAL_EVIDENCE_POLICY_TOP1_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_all_candidate_weight="${ORDINAL_EVIDENCE_ALL_CANDIDATE_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_focal_gamma="${ORDINAL_EVIDENCE_FOCAL_GAMMA:-1.5}" \
  --set training.direct_value_ordinal_evidence_ordered_nll_top1_weight="${ORDINAL_EVIDENCE_ORDERED_NLL_TOP1_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_ordered_nll_all_weight="${ORDINAL_EVIDENCE_ORDERED_NLL_ALL_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_harm_class_weight="${ORDINAL_EVIDENCE_HARM_CLASS_WEIGHT:-2.0}" \
  --set training.direct_value_ordinal_evidence_dead_class_weight="${ORDINAL_EVIDENCE_DEAD_CLASS_WEIGHT:-0.5}" \
  --set training.direct_value_ordinal_evidence_benefit_class_weight="${ORDINAL_EVIDENCE_BENEFIT_CLASS_WEIGHT:-1.25}" \
  --set training.direct_value_ordinal_evidence_hard_harm_weight="${ORDINAL_EVIDENCE_HARD_HARM_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_hard_benefit_weight="${ORDINAL_EVIDENCE_HARD_BENEFIT_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_hard_example_gamma="${ORDINAL_EVIDENCE_HARD_EXAMPLE_GAMMA:-2.0}" \
  --set training.direct_value_ordinal_evidence_class_balanced_weight="${ORDINAL_EVIDENCE_CLASS_BALANCED_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_batch_balanced="${ORDINAL_EVIDENCE_BATCH_BALANCED:-false}" \
  --set training.direct_value_ordinal_evidence_independent_tails="${ORDINAL_EVIDENCE_INDEPENDENT_TAILS:-false}" \
  --set training.direct_value_ordinal_evidence_factorized_harm="${ORDINAL_EVIDENCE_FACTORIZED_HARM:-false}" \
  --set training.direct_value_ordinal_evidence_factorized_harm_temperature="${ORDINAL_EVIDENCE_FACTORIZED_HARM_TEMPERATURE:-0.05}" \
  --set training.direct_value_ordinal_evidence_factorized_harm_drs_tolerance="${COMPONENT_HARM_DRS_TOLERANCE:-0.05}" \
  --set training.direct_value_ordinal_evidence_factorized_harm_dep_tolerance="${COMPONENT_HARM_DEP_TOLERANCE:-0.05}" \
  --set training.direct_value_ordinal_evidence_factorized_harm_gap_tolerance="${COMPONENT_HARM_GAP_TOLERANCE:-0.05}" \
  --set training.direct_value_ordinal_evidence_factorized_harm_hard_tolerance="${COMPONENT_HARM_HARD_TOLERANCE:-0.05}" \
  --set training.direct_value_ordinal_evidence_factorized_harm_proxy_tolerance="${COMPONENT_HARM_PROXY_TOLERANCE:-0.05}" \
  --set training.direct_value_ordinal_evidence_dep_boundary_aligned="${EVIDENCE_DEP_BOUNDARY_ALIGNED:-false}" \
  --set training.direct_value_ordinal_evidence_gap_ordinal_only="${EVIDENCE_GAP_ORDINAL_ONLY:-false}" \
  --set training.direct_value_ordinal_evidence_balanced_replaces_erm="${ORDINAL_EVIDENCE_BALANCED_REPLACES_ERM:-false}" \
  --set training.direct_value_ordinal_evidence_component_tail_weight="${ORDINAL_EVIDENCE_COMPONENT_TAIL_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_component_margin_regression_weight="${ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_component_margin_target_mode="${ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_MODE:-raw}" \
  --set training.direct_value_ordinal_evidence_component_margin_target_scale="${ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_SCALE:-0.10}" \
  --set training.direct_value_ordinal_evidence_component_margin_canonical_scales="${ORDINAL_EVIDENCE_COMPONENT_MARGIN_CANONICAL_SCALES:-}" \
  --set training.direct_value_ordinal_evidence_component_margin_regression_reliability="${ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_RELIABILITY:-}" \
  --set training.direct_value_ordinal_evidence_component_underestimation_weight="${ORDINAL_EVIDENCE_COMPONENT_UNDERESTIMATION_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_safe_positive_component_overestimation_weight="${ORDINAL_EVIDENCE_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_benefit_margin_regression_weight="${ORDINAL_EVIDENCE_BENEFIT_MARGIN_REGRESSION_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_benefit_margin_temperature="${ORDINAL_EVIDENCE_BENEFIT_MARGIN_TEMPERATURE:-0.025}" \
  --set training.direct_value_ordinal_evidence_joint_reserve_regression_weight="${ORDINAL_EVIDENCE_JOINT_RESERVE_REGRESSION_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_joint_reserve_boundary_weight="${ORDINAL_EVIDENCE_JOINT_RESERVE_BOUNDARY_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_joint_reserve_boundary_width="${ORDINAL_EVIDENCE_JOINT_RESERVE_BOUNDARY_WIDTH:-0.05}" \
  --set training.direct_value_ordinal_evidence_component_reliability="${EVIDENCE_COMPONENT_RELIABILITY:-}" \
  --set training.direct_value_ordinal_evidence_global_balance="${ORDINAL_EVIDENCE_GLOBAL_BALANCE:-false}" \
  --set training.direct_value_ordinal_evidence_safe_set_temperature="${ORDINAL_EVIDENCE_SAFE_SET_TEMPERATURE:-0.08}" \
  --set training.direct_value_ordinal_evidence_safe_benefit_target="${ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET:-false}" \
  --set training.direct_value_ordinal_evidence_group_opportunity_weight="${ORDINAL_EVIDENCE_GROUP_OPPORTUNITY_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_admission_weight="${ORDINAL_EVIDENCE_ADMISSION_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_admission_pos_weight="${ORDINAL_EVIDENCE_ADMISSION_POS_WEIGHT:-4.0}" \
  --set training.direct_value_ordinal_evidence_admission_harm_negative_weight="${ORDINAL_EVIDENCE_ADMISSION_HARM_NEGATIVE_WEIGHT:-2.0}" \
  --set training.direct_value_ordinal_evidence_benefit_margin_weight="${ORDINAL_EVIDENCE_BENEFIT_MARGIN_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_harm_margin_weight="${ORDINAL_EVIDENCE_HARM_MARGIN_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_target_probability="${ORDINAL_EVIDENCE_TARGET_PROBABILITY:-0.60}" \
  --set training.direct_value_evidence_calibrator_anchor_weight="${EVIDENCE_CALIBRATOR_ANCHOR_WEIGHT:-0.0}" \
  --set training.direct_value_absolute_feasibility_weight="${ABSOLUTE_FEASIBILITY_WEIGHT:-0.0}" \
  --set training.direct_value_absolute_feasibility_truth_contract="${ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT:-legacy_full}" \
  --set training.direct_value_absolute_feasibility_supervision_objective="${ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE:-binary_sign}" \
  --set training.direct_value_absolute_feasibility_truth_index="${ABSOLUTE_FEASIBILITY_TRUTH_INDEX:-}" \
  --set training.direct_value_ordinal_evidence_proposal_topk_weight="${ORDINAL_EVIDENCE_PROPOSAL_TOPK_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_proposal_topk="${ORDINAL_EVIDENCE_PROPOSAL_TOPK:-3}" \
  --set training.direct_value_ordinal_evidence_proposal_rank_decay="${ORDINAL_EVIDENCE_PROPOSAL_RANK_DECAY:-0.75}" \
  --set training.direct_value_ordinal_evidence_intragroup_benefit_weight="${ORDINAL_EVIDENCE_INTRAGROUP_BENEFIT_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_intragroup_harm_weight="${ORDINAL_EVIDENCE_INTRAGROUP_HARM_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_benefit_listwise_weight="${ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_benefit_listwise_temperature="${ORDINAL_EVIDENCE_BENEFIT_LISTWISE_TEMPERATURE:-0.08}" \
  --set training.direct_value_ordinal_evidence_safe_utility_regression_weight="${ORDINAL_EVIDENCE_SAFE_UTILITY_REGRESSION_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_safe_utility_listwise_weight="${ORDINAL_EVIDENCE_SAFE_UTILITY_LISTWISE_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_safe_utility_temperature="${ORDINAL_EVIDENCE_SAFE_UTILITY_TEMPERATURE:-0.10}" \
  --set training.direct_value_ordinal_evidence_eligible_policy_weight="${ORDINAL_EVIDENCE_ELIGIBLE_POLICY_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_eligible_policy_temperature="${ORDINAL_EVIDENCE_ELIGIBLE_POLICY_TEMPERATURE:-0.10}" \
  --set training.direct_value_ordinal_evidence_eligibility_logit_temperature="${ORDINAL_EVIDENCE_ELIGIBILITY_LOGIT_TEMPERATURE:-0.25}" \
  --set training.direct_value_ordinal_evidence_eligible_opportunity_threshold="${ORDINAL_EVIDENCE_ELIGIBLE_OPPORTUNITY_THRESHOLD:-0.65}" \
  --set training.direct_value_ordinal_evidence_eligible_harm_threshold="${ORDINAL_EVIDENCE_ELIGIBLE_HARM_THRESHOLD:-0.30}" \
  --set training.direct_value_ordinal_evidence_eligibility_boundary_weight="${ORDINAL_EVIDENCE_ELIGIBILITY_BOUNDARY_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_eligibility_boundary_margin="${ORDINAL_EVIDENCE_ELIGIBILITY_BOUNDARY_MARGIN:-0.20}" \
  --set training.direct_value_ordinal_evidence_frontier_pairwise_weight="${ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_frontier_pairwise_margin="${ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_MARGIN:-0.25}" \
  --set training.direct_value_ordinal_evidence_safe_hard_negative_weight="${ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_safe_hard_negative_margin="${ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_MARGIN:-0.05}" \
  --set training.direct_value_ordinal_evidence_safe_hard_negative_teacher_scale="${ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_TEACHER_SCALE:-0.0}" \
  --set training.direct_value_strict_shape_contract="${DIRECT_VALUE_STRICT_SHAPE_CONTRACT:-false}" \
  --set training.direct_value_ordinal_evidence_categorical_group_policy="${ORDINAL_EVIDENCE_CATEGORICAL_GROUP_POLICY:-false}" \
  --set training.direct_value_ordinal_evidence_intragroup_margin="${ORDINAL_EVIDENCE_INTRAGROUP_MARGIN:-0.25}" \
  --set training.direct_value_ordinal_evidence_pairwise_benefit_weight="${ORDINAL_EVIDENCE_PAIRWISE_BENEFIT_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_pairwise_harm_weight="${ORDINAL_EVIDENCE_PAIRWISE_HARM_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_pairwise_margin="${ORDINAL_EVIDENCE_PAIRWISE_MARGIN:-0.25}" \
  --set training.direct_policy_metric_harm_weight="${POLICY_METRIC_HARM_WEIGHT:-0.35}" \
  --set training.direct_policy_metric_false_intervention_weight="${POLICY_METRIC_FALSE_WEIGHT:-0.15}" \
  --set training.direct_policy_metric_missed_opportunity_weight="${POLICY_METRIC_MISS_WEIGHT:-0.25}" \
  --set training.direct_policy_metric_min_positive_recall="${POLICY_METRIC_MIN_POSITIVE_RECALL:-0.0}" \
  --set training.direct_policy_metric_recall_shortfall_weight="${POLICY_METRIC_RECALL_SHORTFALL_WEIGHT:-0.0}" \
  --set training.direct_policy_metric_rank_miss_weight="${POLICY_METRIC_RANK_MISS_WEIGHT:-0.10}" \
  --set training.direct_policy_metric_rank_harm_weight="${POLICY_METRIC_RANK_HARM_WEIGHT:-0.25}" \
  --set training.direct_policy_metric_rank_false_switch_weight="${POLICY_METRIC_RANK_FALSE_WEIGHT:-0.15}" \
  --set training.direct_policy_metric_min_fold_positive="${POLICY_METRIC_MIN_FOLD_POSITIVE:-6}" \
  --set training.direct_policy_metric_robust_top_k="${POLICY_METRIC_ROBUST_TOP_K:-2}" \
  --set training.direct_policy_metric_cross_regime_min_recall="${POLICY_METRIC_CROSS_MIN_RECALL:-0.25}" \
  --set training.direct_policy_metric_cross_regime_recall_weight="${POLICY_METRIC_CROSS_RECALL_WEIGHT:-2.0}" \
  --set training.direct_policy_metric_cross_regime_harm_weight="${POLICY_METRIC_CROSS_HARM_WEIGHT:-0.50}" \
  --set training.direct_policy_metric_cross_regime_false_weight="${POLICY_METRIC_CROSS_FALSE_WEIGHT:-0.20}" \
  --set training.direct_policy_metric_facet_min_recall="${POLICY_METRIC_FACET_MIN_RECALL:-0.20}" \
  --set training.direct_policy_metric_facet_harm_budget="${POLICY_METRIC_FACET_HARM_BUDGET:-0.05}" \
  --set training.direct_policy_metric_facet_false_budget="${POLICY_METRIC_FACET_FALSE_BUDGET:-0.10}" \
  --set training.direct_policy_metric_facet_base_weight="${POLICY_METRIC_FACET_BASE_WEIGHT:-0.10}" \
  --set training.direct_policy_metric_facet_recall_weight="${POLICY_METRIC_FACET_RECALL_WEIGHT:-12.0}" \
  --set training.direct_policy_metric_facet_harm_excess_weight="${POLICY_METRIC_FACET_HARM_EXCESS_WEIGHT:-10.0}" \
  --set training.direct_policy_metric_facet_false_excess_weight="${POLICY_METRIC_FACET_FALSE_EXCESS_WEIGHT:-3.0}" \
  --set training.direct_policy_metric_facet_raw_harm_weight="${POLICY_METRIC_FACET_RAW_HARM_WEIGHT:-0.25}" \
  --set training.direct_policy_metric_facet_raw_false_weight="${POLICY_METRIC_FACET_RAW_FALSE_WEIGHT:-0.10}" \
  --set training.direct_policy_metric_opportunity_threshold="${POLICY_METRIC_OPP_THRESHOLD:-0.65}" \
  --set training.direct_policy_metric_harm_threshold="${POLICY_METRIC_HARM_THRESHOLD:-0.30}" \
  --set training.direct_policy_metric_rank_margin_threshold="${POLICY_METRIC_RANK_MARGIN:-0.020}" \
  --set training.direct_policy_metric_min_delta_mean="${POLICY_METRIC_MIN_DELTA:-0.0}" \
  --set training.direct_policy_metric_risk_source="${POLICY_METRIC_RISK_SOURCE:-gaussian_delta}" \
  --set training.direct_policy_metric_proposal_top_k="${POLICY_METRIC_PROPOSAL_TOP_K:-1}" \
  --set training.direct_policy_metric_evidence_rerank_top_k="${POLICY_METRIC_EVIDENCE_RERANK_TOP_K:-false}" \
  --set training.direct_policy_metric_safe_opportunity="${POLICY_METRIC_SAFE_OPPORTUNITY:-false}" \
  --set training.direct_policy_metric_soft_temperature="${POLICY_METRIC_SOFT_TEMPERATURE:-0.10}" \
  --set training.direct_policy_metric_eligibility_logit_temperature="${POLICY_METRIC_ELIGIBILITY_LOGIT_TEMPERATURE:-0.25}" \
  --set training.direct_policy_metric_categorical_group_policy="${POLICY_METRIC_CATEGORICAL_GROUP_POLICY:-false}" \
  --set training.direct_policy_metric_exact_eligibility="${POLICY_METRIC_EXACT_ELIGIBILITY:-false}" \
  --set training.direct_policy_metric_max_hard="${POLICY_METRIC_MAX_HARD:-1.0}" \
  --set training.direct_policy_metric_min_nominal_deviation="${POLICY_METRIC_MIN_NOMINAL_DEVIATION:-0.002}" \
  --set training.direct_policy_metric_concord_miss_weight="${POLICY_METRIC_CONCORD_MISS_WEIGHT:-2.0}" \
  --set training.direct_policy_metric_concord_false_weight="${POLICY_METRIC_CONCORD_FALSE_WEIGHT:-0.75}" \
  --set training.direct_policy_metric_concord_harm_weight="${POLICY_METRIC_CONCORD_HARM_WEIGHT:-2.0}" \
  --set training.direct_policy_metric_concord_regret_weight="${POLICY_METRIC_CONCORD_REGRET_WEIGHT:-0.50}" \
  --set training.direct_policy_metric_concord_safe_mass_weight="${POLICY_METRIC_CONCORD_SAFE_MASS_WEIGHT:-1.0}" \
  --set training.direct_policy_metric_covenant_harm_weight="${POLICY_METRIC_COVENANT_HARM_WEIGHT:-1.5}" \
  --set training.direct_policy_metric_covenant_false_weight="${POLICY_METRIC_COVENANT_FALSE_WEIGHT:-0.5}" \
  --set training.direct_policy_metric_frontier_harm_weight="${POLICY_METRIC_FRONTIER_HARM_WEIGHT:-1.5}" \
  --set training.direct_policy_metric_frontier_false_weight="${POLICY_METRIC_FRONTIER_FALSE_WEIGHT:-0.5}" \
  --set training.direct_policy_metric_frontier_global_harm_tiebreak="${POLICY_METRIC_FRONTIER_GLOBAL_HARM_TIEBREAK:-0.25}" \
  --set training.direct_policy_metric_integrity_min_recall="${POLICY_METRIC_INTEGRITY_MIN_RECALL:-0.20}" \
  --set training.direct_policy_metric_integrity_recall_weight="${POLICY_METRIC_INTEGRITY_RECALL_WEIGHT:-20.0}" \
  --set training.direct_policy_metric_integrity_min_precision="${POLICY_METRIC_INTEGRITY_MIN_PRECISION:-0.60}" \
  --set training.direct_policy_metric_integrity_precision_weight="${POLICY_METRIC_INTEGRITY_PRECISION_WEIGHT:-8.0}" \
  --set training.direct_policy_metric_integrity_invalid_weight="${POLICY_METRIC_INTEGRITY_INVALID_WEIGHT:-4.0}" \
  --set training.direct_policy_metric_integrity_safe_regret_weight="${POLICY_METRIC_INTEGRITY_SAFE_REGRET_WEIGHT:-2.0}" \
  --set training.direct_policy_metric_integrity_all_abstain_weight="${POLICY_METRIC_INTEGRITY_ALL_ABSTAIN_WEIGHT:-8.0}" \
  --set training.direct_policy_metric_contract_min_safe_top1_recall="${POLICY_METRIC_CONTRACT_MIN_SAFE_TOP1_RECALL:-0.20}" \
  --set training.direct_policy_metric_contract_zero_top1_weight="${POLICY_METRIC_CONTRACT_ZERO_TOP1_WEIGHT:-100.0}" \
  --set training.direct_policy_metric_contract_top1_shortfall_weight="${POLICY_METRIC_CONTRACT_TOP1_SHORTFALL_WEIGHT:-20.0}" \
  --set training.direct_policy_metric_contract_all_abstain_weight="${POLICY_METRIC_CONTRACT_ALL_ABSTAIN_WEIGHT:-10.0}" \
  --set training.direct_policy_metric_contract_invalid_weight="${POLICY_METRIC_CONTRACT_INVALID_WEIGHT:-4.0}" \
  --set training.direct_policy_metric_contract_regret_weight="${POLICY_METRIC_CONTRACT_REGRET_WEIGHT:-2.0}" \
  --set training.direct_value_false_positive_weight="$FP_W" \
  --set training.direct_value_opportunity_weight="${OPPORTUNITY_AUX_WEIGHT:-0.15}" \
  --set training.direct_value_opportunity_pos_weight=8.0 \
  --set training.direct_value_harm_weight="$HARM_W" \
  --set training.direct_value_harm_pos_weight=7.0 \
  --set training.direct_value_setwise_admission_weight="$SETWISE_W" \
  --set training.direct_value_opportunity_admission_weight="${OPPORTUNITY_ADMISSION_WEIGHT:-0.0}" \
  --set training.direct_value_harm_admission_weight="${HARM_ADMISSION_WEIGHT:-0.0}" \
  --set training.direct_value_selective_risk_weight="${SELECTIVE_RISK_WEIGHT:-2.0}" \
  --set training.direct_value_selective_harm_budget="${SELECTIVE_HARM_BUDGET:-0.05}" \
  --set training.direct_value_selective_coverage_weight="${SELECTIVE_COVERAGE_WEIGHT:-1.0}" \
  --set training.direct_value_selective_coverage_target="${SELECTIVE_COVERAGE_TARGET:-0.65}" \
  --set training.direct_value_policy_distill_weight="${POLICY_DISTILL_WEIGHT:-0.0}" \
  --set training.direct_value_policy_teacher_temperature="${POLICY_TEACHER_TEMPERATURE:-0.06}" \
  --set training.direct_value_policy_regret_weight="${POLICY_REGRET_WEIGHT:-0.0}" \
  --set training.direct_value_policy_regret_margin="${POLICY_REGRET_MARGIN:-0.005}" \
  --set training.direct_value_policy_decouple_admission="${POLICY_DECOUPLE_ADMISSION:-true}" \
  --set training.direct_value_policy_admission_distill_weight="${POLICY_ADMISSION_DISTILL_WEIGHT:-0.05}" \
  --set training.direct_value_opportunity_soft_label_temperature="${OPPORTUNITY_SOFT_LABEL_TEMPERATURE:-0.02}" \
  --set training.direct_value_harm_soft_label_temperature="${HARM_SOFT_LABEL_TEMPERATURE:-0.02}" \
  --set training.direct_value_group_dro_weight="${GROUP_DRO_WEIGHT:-0.0}" \
  --set training.direct_value_group_dro_temperature="${GROUP_DRO_TEMPERATURE:-0.35}" \
  --set training.direct_value_group_dro_severity_thresholds="${GROUP_DRO_SEVERITY_THRESHOLDS:-0.25,0.55}" \
  --set training.direct_value_expert_specialization_weight="${EXPERT_SPECIALIZATION_WEIGHT:-0.30}" \
  2>&1 | tee "$LOG_DIR/train_v48_trac_sr.log"

# Standard OC-MERO calibration is retained for the base feasibility certificate.
if [[ "${SKIP_POST_TRAIN_CALIBRATION:-0}" != "1" ]]; then
for bucket in mix safe near contact; do
  case "$bucket" in
    mix) data="$CAL_MIX"; min=100 ;;
    safe) data="${VAL_SAFE:-$EVAL_OCRAP_ROOT/val_safe}"; min=50 ;;
    near) data="${VAL_NEAR:-$EVAL_OCRAP_ROOT/val_near_contact}"; min=50 ;;
    contact) data="${VAL_CONTACT:-$EVAL_OCRAP_ROOT/val_contact}"; min=50 ;;
  esac
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u -m ocrap.cli calibrate \
    --dataset "$data" --checkpoint "$MODEL_DIR/best.pt" \
    --output "$CAL_DIR/calibration_${bucket}_v48.json" \
    --set calibration.required_min_for_delta="$min" \
    2>&1 | tee "$LOG_DIR/calibrate_${bucket}_v48.log"
done
python tools/write_gamma_by_bucket.py \
  --safe "$CAL_DIR/calibration_safe_v48.json" \
  --near "$CAL_DIR/calibration_near_v48.json" \
  --contact "$CAL_DIR/calibration_contact_v48.json" \
  --delta 0.05 --output "$CAL_DIR/gamma_rec_by_bucket_v48.json" \
  2>&1 | tee "$LOG_DIR/write_gamma_v48.log"

fi

python - "$MODEL_DIR/best.pt" "$MODEL_DIR/train_summary.json" "$RUN/TRAINING_COMPLETE.json" <<'PYDONE'
import hashlib,json,pathlib,sys,time
ckpt,summary,out=map(pathlib.Path,sys.argv[1:])
if not ckpt.is_file() or not summary.is_file(): raise SystemExit("missing completed training artifacts")
d=json.loads(summary.read_text()); doc={"event":"variant_training_complete","created_unix":time.time(),"checkpoint":str(ckpt),"checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest(),"best_epoch":d.get("best_epoch"),"epochs_completed":d.get("epochs_completed"),"best_metric":d.get("best_metric")}
out.write_text(json.dumps(doc,indent=2)+"\n")
PYDONE
