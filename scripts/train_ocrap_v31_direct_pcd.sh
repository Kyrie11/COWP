#!/usr/bin/env bash
set -euo pipefail

export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export TRAIN_MIX=${TRAIN_MIX:-$OCRAP_ROOT/train_safe,$OCRAP_ROOT/train_contact,$OCRAP_ROOT/train_near_contact}
export VAL_MIX=${VAL_MIX:-$OCRAP_ROOT/val_safe,$OCRAP_ROOT/val_contact,$OCRAP_ROOT/val_near_contact}
export BASE_RUN=${BASE_RUN:-runs/ocrap_v28_residual_ddc}
export INIT_CKPT=${INIT_CKPT:-$BASE_RUN/model_v28_residual_ddc/best.pt}
export RUN=${RUN:-runs/ocrap_v31_direct_pcd}
export MODEL_DIR=${MODEL_DIR:-$RUN/model_v31_direct_pcd}
export CAL_DIR=${CAL_DIR:-$RUN/calibration}
export TRAIN_GPU=${TRAIN_GPU:-0}
mkdir -p "$MODEL_DIR" "$CAL_DIR"

[[ -f "$INIT_CKPT" ]] || { echo "missing INIT_CKPT=$INIT_CKPT" >&2; exit 2; }

# v31 direct teacher-PCD: start from v28 residual-DDC, then fine-tune
# the deployability heads with absolute teacher-PCD regression and group ranking.
# This targets the v30 audit failure where learned PCD ranks nominal/merge above
# paper-best brake/yield in contact scenes.
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONUNBUFFERED=1 python -u -m ocrap.cli train \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --output "$MODEL_DIR" \
  --set training.init_checkpoint="$INIT_CKPT" \
  --set training.freeze_param_prefixes=encoder,root_queries,root_cross_attn,root_self_attn,root_norm1,root_norm2,root_norm3,root_ffn \
  --set training.epochs=${EPOCHS:-10} \
  --set training.early_stop_patience=${PATIENCE:-3} \
  --set training.batch_size=${BATCH_SIZE:-64} \
  --set training.lr=${LR:-0.00004} \
  --set training.weight_decay=0.00001 \
  --set training.artifact_sampler_weight=1.0 \
  --set training.negative_deployable_sampler_weight=1.50 \
  --set training.regime_balance_power=0.80 \
  --set training.group_batching=true \
  --set training.group_batching_replacement=true \
  --set training.num_workers=4 \
  --set training.progress=true \
  --set training.require_cuda=true \
  --set training.save_every_epoch=false \
  --set model.encoder_type=structured_transformer \
  --set model.d_model=192 \
  --set model.d_obs=64 \
  --set model.transformer_layers=2 \
  --set model.transformer_heads=4 \
  --set loss_weights.margin=1.0 \
  --set loss_weights.obs=0.7 \
  --set loss_weights.anti_oracle=0.8 \
  --set loss_weights.utility=0.05 \
  --set loss_weights.option_q=0.35 \
  --set loss_weights.option_admission=0.30 \
  --set loss_weights.option_success=0.25 \
  --set loss_weights.option_success_bce=0.25 \
  --set loss_weights.option_best=0.20 \
  --set loss_weights.group_ce=0.18 \
  --set loss_weights.group_distill=0.10 \
  --set loss_weights.nominal_switch=0.06 \
  --set loss_weights.safe_nominal=0.30 \
  --set loss_weights.protective_macro=0.45 \
  --set loss_weights.macro_drs=0.50 \
  --set loss_weights.ddc=${DDC_WEIGHT:-4.0} \
  --set loss_weights.teacher_pcd_direct=${TEACHER_PCD_DIRECT_WEIGHT:-8.0} \
  --set training.group_ce_temperature=0.35 \
  --set training.group_ce_teacher_gap_weight=0.25 \
  --set training.group_ce_pred_gap_weight=0.25 \
  --set training.group_distill_teacher_gap_weight=0.25 \
  --set training.group_distill_pred_gap_weight=0.25 \
  --set training.protective_macro_ids=2,3,7 \
  --set training.protective_macro_bucket_ids=2 \
  --set training.protective_macro_min_teacher_r_dep=0.0 \
  --set training.protective_macro_min_teacher_drs=0.50 \
  --set training.protective_macro_min_teacher_pcd_gain=0.02 \
  --set training.protective_macro_max_nominal_teacher_pcd=0.90 \
  --set training.protective_macro_margin=0.14 \
  --set training.protective_macro_pred_gap_weight=0.18 \
  --set training.protective_macro_pred_drs_weight=0.65 \
  --set training.protective_macro_teacher_gap_weight=0.10 \
  --set training.protective_macro_teacher_drs_weight=0.70 \
  --set training.protective_macro_target_min_pred_drs=0.70 \
  --set training.teacher_pcd_direct_macro_ids=${TEACHER_PCD_DIRECT_MACROS:-2,3,5,7} \
  --set training.teacher_pcd_direct_bucket_ids=${TEACHER_PCD_DIRECT_BUCKETS:-2} \
  --set training.teacher_pcd_direct_regression_weight=${TEACHER_PCD_DIRECT_REG_WEIGHT:-1.0} \
  --set training.teacher_pcd_direct_ranking_weight=${TEACHER_PCD_DIRECT_RANK_WEIGHT:-2.8} \
  --set training.teacher_pcd_direct_nominal_penalty_weight=${TEACHER_PCD_DIRECT_NOMINAL_WEIGHT:-1.0} \
  --set training.teacher_pcd_direct_false_positive_weight=${TEACHER_PCD_DIRECT_FALSE_POS_WEIGHT:-1.8} \
  --set training.teacher_pcd_direct_margin=${TEACHER_PCD_DIRECT_MARGIN:-0.20} \
  --set training.teacher_pcd_direct_min_teacher_pcd_gain=${TEACHER_PCD_DIRECT_MIN_GAIN:-0.015} \
  --set training.teacher_pcd_direct_min_teacher_best_pcd=${TEACHER_PCD_DIRECT_MIN_BEST:-0.50} \
  --set training.teacher_pcd_direct_max_nominal_teacher_pcd=${TEACHER_PCD_DIRECT_MAX_NOMINAL:-0.70} \
  --set training.teacher_pcd_direct_target_min_pred_pcd=${TEACHER_PCD_DIRECT_TARGET_MIN_PRED:-0.52} \
  --set training.teacher_pcd_direct_nominal_max_pred_pcd=${TEACHER_PCD_DIRECT_NOMINAL_MAX_PRED:-0.50} \
  --set training.teacher_pcd_direct_focus_non_nominal_weight=${TEACHER_PCD_DIRECT_FOCUS_WEIGHT:-2.5} \
  --set training.teacher_pcd_direct_false_positive_margin=${TEACHER_PCD_DIRECT_FALSE_POS_MARGIN:-0.03} \
  --set training.macro_drs_ids=2,3,5,7 \
  --set training.macro_drs_bucket_ids=2 \
  --set training.macro_drs_pos_threshold=0.80 \
  --set training.macro_drs_neg_threshold=0.05 \
  --set training.macro_drs_pos_weight=4.0 \
  --set training.macro_drs_neg_weight=1.0 \
  --set training.ddc_macro_ids=2,3,5,7 \
  --set training.ddc_bucket_ids=2 \
  --set training.ddc_margin=${DDC_MARGIN:-0.20} \
  --set training.ddc_min_teacher_pcd_gain=${DDC_MIN_TEACHER_PCD_GAIN:-0.020} \
  --set training.ddc_min_teacher_best_pcd=${DDC_MIN_TEACHER_BEST_PCD:-0.50} \
  --set training.ddc_max_nominal_teacher_pcd=${DDC_MAX_NOMINAL_TEACHER_PCD:-0.64} \
  --set training.ddc_pred_gap_weight=0.28 \
  --set training.ddc_pred_drs_weight=0.45 \
  --set training.ddc_utility_weight=0.00 \
  --set training.ddc_target_min_pred_pcd=${DDC_TARGET_MIN_PRED_PCD:-0.50} \
  --set training.ddc_nominal_max_pred_pcd=${DDC_NOMINAL_MAX_PRED_PCD:-0.52} \
  2>&1 | tee "$MODEL_DIR/train_v31_direct_pcd.log"

PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
  --dataset "$VAL_MIX" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --output "$CAL_DIR/calibration_mix_v31.json" \
  --set calibration.required_min_for_delta=100 \
  2>&1 | tee "$CAL_DIR/calibrate_mix_v31.log"

PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
  --dataset "$OCRAP_ROOT/val_safe" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --output "$CAL_DIR/calibration_safe_v31.json" \
  --set calibration.required_min_for_delta=50 \
  2>&1 | tee "$CAL_DIR/calibrate_safe_v31.log"

PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
  --dataset "$OCRAP_ROOT/val_near_contact" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --output "$CAL_DIR/calibration_near_v31.json" \
  --set calibration.required_min_for_delta=50 \
  2>&1 | tee "$CAL_DIR/calibrate_near_v31.log"

PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
  --dataset "$OCRAP_ROOT/val_contact" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --output "$CAL_DIR/calibration_contact_v31.json" \
  --set calibration.required_min_for_delta=50 \
  2>&1 | tee "$CAL_DIR/calibrate_contact_v31.log"

python tools/write_gamma_by_bucket.py \
  --safe "$CAL_DIR/calibration_safe_v31.json" \
  --near "$CAL_DIR/calibration_near_v31.json" \
  --contact "$CAL_DIR/calibration_contact_v31.json" \
  --delta 0.05 \
  --output "$CAL_DIR/gamma_rec_by_bucket_v31.json" \
  2>&1 | tee "$CAL_DIR/write_gamma_v31.log"

printf '\n[v31] model: %s\n[v31] calibration: %s\n[v31] gamma map: %s\n' "$MODEL_DIR/best.pt" "$CAL_DIR/calibration_mix_v31.json" "$CAL_DIR/gamma_rec_by_bucket_v31.json"
