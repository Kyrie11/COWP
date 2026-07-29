#!/usr/bin/env bash
set -euo pipefail

export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export TRAIN_MIX=${TRAIN_MIX:-$OCRAP_ROOT/train_safe,$OCRAP_ROOT/train_contact,$OCRAP_ROOT/train_near_contact}
export VAL_MIX=${VAL_MIX:-$OCRAP_ROOT/val_safe,$OCRAP_ROOT/val_contact,$OCRAP_ROOT/val_near_contact}
export BASE_RUN=${BASE_RUN:-runs/ocrap_v33_brake_tail}
export INIT_CKPT=${INIT_CKPT:-$BASE_RUN/model_v33_brake_tail/best.pt}
export VARIANT=${VARIANT:-balanced}
export RUN=${RUN:-runs/ocrap_v39_ocrac_${VARIANT}}
export MODEL_DIR=${MODEL_DIR:-$RUN/model_v39_ocrac}
export CAL_DIR=${CAL_DIR:-$RUN/calibration}
export TRAIN_GPU=${TRAIN_GPU:-0}
export FREEZE_PREFIXES=${FREEZE_PREFIXES:-encoder,root_queries,root_cross_attn,root_self_attn,root_norm1,root_norm2,root_norm3,root_ffn}
mkdir -p "$MODEL_DIR" "$CAL_DIR"
[[ -f "$INIT_CKPT" ]] || { echo "missing INIT_CKPT=$INIT_CKPT" >&2; exit 2; }

# v39 OC-RAC: Observation-Consistent Counterfactual Recovery-Advantage
# Calibration. Unlike v38's selector-only tail bypass, this fine-tune directly
# corrects scene-time inversions where teacher-failed nominal is predicted more
# deployable than a shared recovery macro. Both near-contact and contact are
# supervised; safe nominal preservation remains explicit.
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONUNBUFFERED=1 python -u -m ocrap.cli train \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --output "$MODEL_DIR" \
  --set training.init_checkpoint="$INIT_CKPT" \
  --set training.freeze_param_prefixes="$FREEZE_PREFIXES" \
  --set training.epochs=${EPOCHS:-14} \
  --set training.early_stop_patience=${PATIENCE:-4} \
  --set training.batch_size=${BATCH_SIZE:-64} \
  --set training.lr=${LR:-0.000012} \
  --set training.weight_decay=0.00001 \
  --set training.artifact_sampler_weight=1.0 \
  --set training.negative_deployable_sampler_weight=2.0 \
  --set training.regime_balance_power=1.0 \
  --set training.group_batching=true \
  --set training.group_batching_replacement=true \
  --set training.group_batch_hard_macro_ids=2,3,5,7 \
  --set training.group_batch_hard_bucket_ids=1,2 \
  --set training.group_batch_hard_min_r_dep=${GROUP_HARD_MIN_R_DEP:-0.15} \
  --set training.group_batch_hard_boost=${GROUP_HARD_BOOST:-10.0} \
  --set training.num_workers=4 \
  --set training.progress=true \
  --set training.require_cuda=true \
  --set training.save_every_epoch=true \
  --set training.best_metric=${BEST_METRIC:-loss_recovery_advantage} \
  --set training.best_metric_mode=min \
  --set model.encoder_type=structured_transformer \
  --set model.d_model=192 \
  --set model.d_obs=64 \
  --set model.transformer_layers=2 \
  --set model.transformer_heads=4 \
  --set loss_weights.margin=1.0 \
  --set loss_weights.obs=0.8 \
  --set loss_weights.anti_oracle=0.8 \
  --set loss_weights.utility=0.05 \
  --set loss_weights.option_q=0.35 \
  --set loss_weights.option_admission=0.30 \
  --set loss_weights.option_success=0.30 \
  --set loss_weights.option_success_bce=0.30 \
  --set loss_weights.option_best=0.20 \
  --set loss_weights.group_ce=0.15 \
  --set loss_weights.group_distill=0.10 \
  --set loss_weights.nominal_switch=0.08 \
  --set loss_weights.safe_nominal=${SAFE_NOMINAL_WEIGHT:-0.55} \
  --set loss_weights.protective_macro=${PROTECTIVE_WEIGHT:-0.50} \
  --set loss_weights.macro_drs=${MACRO_DRS_WEIGHT:-0.65} \
  --set loss_weights.ddc=${DDC_WEIGHT:-3.0} \
  --set loss_weights.teacher_pcd_direct=${TEACHER_PCD_DIRECT_WEIGHT:-12.0} \
  --set loss_weights.recovery_advantage=${RECOVERY_ADVANTAGE_WEIGHT:-16.0} \
  --set training.protective_macro_ids=2,3,5,7 \
  --set training.protective_macro_bucket_ids=1,2 \
  --set training.protective_macro_min_teacher_r_dep=-0.20 \
  --set training.protective_macro_min_teacher_drs=0.50 \
  --set training.protective_macro_min_teacher_pcd_gain=0.02 \
  --set training.protective_macro_max_nominal_teacher_pcd=0.90 \
  --set training.protective_macro_margin=0.12 \
  --set training.protective_macro_target_min_pred_drs=0.72 \
  --set training.macro_drs_ids=2,3,5,7 \
  --set training.macro_drs_bucket_ids=1,2 \
  --set training.macro_drs_pos_threshold=0.80 \
  --set training.macro_drs_neg_threshold=0.05 \
  --set training.macro_drs_pos_weight=5.0 \
  --set training.macro_drs_neg_weight=1.2 \
  --set training.ddc_macro_ids=2,3,5,7 \
  --set training.ddc_bucket_ids=1,2 \
  --set training.ddc_margin=0.16 \
  --set training.ddc_min_teacher_pcd_gain=0.02 \
  --set training.ddc_min_teacher_best_pcd=0.48 \
  --set training.ddc_max_nominal_teacher_pcd=0.68 \
  --set training.ddc_target_min_pred_pcd=0.50 \
  --set training.ddc_nominal_max_pred_pcd=0.50 \
  --set training.teacher_pcd_direct_macro_ids=2,3,5,7 \
  --set training.teacher_pcd_direct_positive_macro_ids=2,3,5,7 \
  --set training.teacher_pcd_direct_bucket_ids=1,2 \
  --set training.teacher_pcd_direct_regression_weight=1.3 \
  --set training.teacher_pcd_direct_ranking_weight=3.5 \
  --set training.teacher_pcd_direct_nominal_penalty_weight=2.0 \
  --set training.teacher_pcd_direct_false_positive_weight=2.0 \
  --set training.teacher_pcd_direct_margin=0.22 \
  --set training.teacher_pcd_direct_min_teacher_pcd_gain=0.02 \
  --set training.teacher_pcd_direct_min_teacher_best_pcd=0.48 \
  --set training.teacher_pcd_direct_max_nominal_teacher_pcd=0.70 \
  --set training.teacher_pcd_direct_target_min_pred_pcd=0.55 \
  --set training.teacher_pcd_direct_nominal_max_pred_pcd=0.48 \
  --set training.teacher_pcd_direct_focus_non_nominal_weight=3.0 \
  --set training.teacher_pcd_direct_false_positive_margin=0.03 \
  --set training.teacher_pcd_direct_component_weight=1.2 \
  --set training.teacher_pcd_direct_positive_component_weight=1.5 \
  --set training.teacher_pcd_direct_nominal_cap_weight=1.8 \
  --set training.recovery_advantage_macro_ids=2,3,5,7 \
  --set training.recovery_advantage_bucket_ids=1,2 \
  --set training.recovery_advantage_positive_gain=${ADV_POS_GAIN:-0.025} \
  --set training.recovery_advantage_negative_gain=${ADV_NEG_GAIN:-0.025} \
  --set training.recovery_advantage_margin=${ADV_MARGIN:-0.12} \
  --set training.recovery_advantage_regression_weight=${ADV_REG_WEIGHT:-1.0} \
  --set training.recovery_advantage_ranking_weight=${ADV_RANK_WEIGHT:-2.0} \
  --set training.recovery_advantage_component_weight=${ADV_COMPONENT_WEIGHT:-1.0} \
  --set training.recovery_advantage_false_positive_weight=${ADV_FALSE_POS_WEIGHT:-1.5} \
  --set training.recovery_advantage_nominal_failure_pcd_max=${ADV_NOMINAL_FAILURE_MAX:-0.25} \
  --set training.recovery_advantage_target_min_pred_pcd=${ADV_TARGET_MIN_PCD:-0.54} \
  --set training.recovery_advantage_nominal_max_pred_pcd=${ADV_NOMINAL_MAX_PCD:-0.46} \
  --set training.recovery_advantage_near_weight=${ADV_NEAR_WEIGHT:-1.8} \
  --set training.recovery_advantage_contact_weight=${ADV_CONTACT_WEIGHT:-1.0} \
  2>&1 | tee "$MODEL_DIR/train_v39_ocrac.log"

for bucket in mix safe near contact; do
  case "$bucket" in
    mix) data="$VAL_MIX"; min=100 ;;
    safe) data="$OCRAP_ROOT/val_safe"; min=50 ;;
    near) data="$OCRAP_ROOT/val_near_contact"; min=50 ;;
    contact) data="$OCRAP_ROOT/val_contact"; min=50 ;;
  esac
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
    --dataset "$data" --checkpoint "$MODEL_DIR/best.pt" \
    --output "$CAL_DIR/calibration_${bucket}_v39.json" \
    --set calibration.required_min_for_delta="$min" \
    2>&1 | tee "$CAL_DIR/calibrate_${bucket}_v39.log"
done

python tools/write_gamma_by_bucket.py \
  --safe "$CAL_DIR/calibration_safe_v39.json" \
  --near "$CAL_DIR/calibration_near_v39.json" \
  --contact "$CAL_DIR/calibration_contact_v39.json" \
  --delta 0.05 --output "$CAL_DIR/gamma_rec_by_bucket_v39.json" \
  2>&1 | tee "$CAL_DIR/write_gamma_v39.log"

printf '\n[v39:%s] model=%s\ncalibration=%s\ngamma=%s\n' "$VARIANT" "$MODEL_DIR/best.pt" "$CAL_DIR/calibration_mix_v39.json" "$CAL_DIR/gamma_rec_by_bucket_v39.json"
