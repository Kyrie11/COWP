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
INIT_CKPT="${INIT_CKPT:-runs/ocrap_v47_trac_balanced/model_v47_trac/best.pt}"
VARIANT="${VARIANT:-balanced}"
RUN="${RUN:-runs/ocrap_v48_trac_sr_${VARIANT}}"
MODEL_DIR="${MODEL_DIR:-$RUN/model_v48_trac_sr}"
CAL_DIR="${CAL_DIR:-$RUN/calibration}"
LOG_DIR="${LOG_DIR:-$RUN/logs}"
TRAIN_GPU="${TRAIN_GPU:-0}"
GROUP_INDEX="${GROUP_INDEX:-$RUN/teacher_pcd_train_index.jsonl}"
mkdir -p "$MODEL_DIR" "$CAL_DIR" "$LOG_DIR"
[[ -f "$INIT_CKPT" ]] || { echo "missing INIT_CKPT=$INIT_CKPT" >&2; exit 2; }
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
  --set training.direct_only_fast_path=true \
  --set training.group_index_path="$GROUP_INDEX" \
  --set training.epochs="${EPOCHS:-12}" --set training.early_stop_patience="${PATIENCE:-3}" \
  --set training.batch_size="${BATCH_SIZE:-96}" --set training.lr="$LR" \
  --set training.encoder_lr_scale="$ENCODER_LR_SCALE" \
  --set training.encoder_anchor_weight="${ENCODER_ANCHOR_WEIGHT:-0.02}" \
  --set training.weight_decay=0.00002 \
  --set training.grad_clip=3.0 --set training.require_cuda=true \
  --set training.amp=true --set training.amp_dtype=bfloat16 \
  --set training.allow_tf32=true --set training.matmul_precision=high \
  --set training.cudnn_benchmark=true --set training.pin_memory=true \
  --set training.num_workers="${NUM_WORKERS:-6}" --set training.persistent_workers=true \
  --set training.prefetch_factor="${PREFETCH_FACTOR:-2}" --set training.progress=true \
  --set training.save_every_epoch=true --set training.best_metric="${BEST_METRIC:-direct_policy_risk_fold_worst}" \
  --set training.best_metric_mode=min --set training.best_metric_min_delta="${BEST_METRIC_MIN_DELTA:-0.000001}" \
  --set training.group_batching=true --set training.group_batching_replacement=true \
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
  --set loss_weights.dep=0 --set loss_weights.orc=0 --set loss_weights.assign=0 \
  --set loss_weights.sig=0 --set loss_weights.margin=0 --set loss_weights.obs=0 \
  --set loss_weights.anti_oracle=0 --set loss_weights.artifact_gap=0 \
  --set loss_weights.admission=0 --set loss_weights.utility=0 \
  --set loss_weights.option_q=0 --set loss_weights.option_admission=0 \
  --set loss_weights.option_success=0 --set loss_weights.option_success_bce=0 \
  --set loss_weights.option_best=0 --set loss_weights.group_ce=0 \
  --set loss_weights.group_distill=0 --set loss_weights.nominal_switch=0 \
  --set loss_weights.safe_nominal=0 --set loss_weights.protective_macro=0 \
  --set loss_weights.macro_drs=0 --set loss_weights.teacher_pcd_direct=0 \
  --set loss_weights.recovery_advantage=0 --set loss_weights.direct_router_balance=0 \
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
  --set training.direct_value_ordinal_evidence_pairwise_benefit_weight="${ORDINAL_EVIDENCE_PAIRWISE_BENEFIT_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_pairwise_harm_weight="${ORDINAL_EVIDENCE_PAIRWISE_HARM_WEIGHT:-0.0}" \
  --set training.direct_value_ordinal_evidence_pairwise_margin="${ORDINAL_EVIDENCE_PAIRWISE_MARGIN:-0.25}" \
  --set training.direct_policy_metric_harm_weight="${POLICY_METRIC_HARM_WEIGHT:-0.35}" \
  --set training.direct_policy_metric_false_intervention_weight="${POLICY_METRIC_FALSE_WEIGHT:-0.15}" \
  --set training.direct_policy_metric_missed_opportunity_weight="${POLICY_METRIC_MISS_WEIGHT:-0.25}" \
  --set training.direct_policy_metric_rank_miss_weight="${POLICY_METRIC_RANK_MISS_WEIGHT:-0.10}" \
  --set training.direct_policy_metric_rank_harm_weight="${POLICY_METRIC_RANK_HARM_WEIGHT:-0.25}" \
  --set training.direct_policy_metric_rank_false_switch_weight="${POLICY_METRIC_RANK_FALSE_WEIGHT:-0.15}" \
  --set training.direct_policy_metric_min_fold_positive="${POLICY_METRIC_MIN_FOLD_POSITIVE:-6}" \
  --set training.direct_policy_metric_robust_top_k="${POLICY_METRIC_ROBUST_TOP_K:-2}" \
  --set training.direct_policy_metric_opportunity_threshold="${POLICY_METRIC_OPP_THRESHOLD:-0.65}" \
  --set training.direct_policy_metric_harm_threshold="${POLICY_METRIC_HARM_THRESHOLD:-0.30}" \
  --set training.direct_policy_metric_rank_margin_threshold="${POLICY_METRIC_RANK_MARGIN:-0.020}" \
  --set training.direct_policy_metric_min_delta_mean="${POLICY_METRIC_MIN_DELTA:-0.0}" \
  --set training.direct_policy_metric_risk_source="${POLICY_METRIC_RISK_SOURCE:-gaussian_delta}" \
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
