#!/usr/bin/env bash
set -euo pipefail
export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
# v46 OC-RACE: two explicitly supervised Near/Contact value experts share the
# frozen OC-MERO encoder.  A router is distilled from build-regime labels during
# training, but deployment still routes only from observable scene/prefix
# features.  This prevents the unidentifiable soft-mixture collapse seen in v45.
# Safe remains protected by the frozen OC-MERO path.
export TRAIN_MIX=${TRAIN_MIX:-$OCRAP_ROOT/train_near_contact,$OCRAP_ROOT/train_contact}
export VAL_MIX=${VAL_MIX:-$OCRAP_ROOT/val_near_contact,$OCRAP_ROOT/val_contact}
export BASE_RUN=${BASE_RUN:-runs/ocrap_v39_ocrac_balanced}
export INIT_CKPT=${INIT_CKPT:-$BASE_RUN/model_v39_ocrac/best.pt}
export VARIANT=${VARIANT:-expert_balanced}
export RUN=${RUN:-runs/ocrap_v46_race_${VARIANT}}
export MODEL_DIR=${MODEL_DIR:-$RUN/model_v46_race}
export CAL_DIR=${CAL_DIR:-$RUN/calibration}
export TRAIN_GPU=${TRAIN_GPU:-0}
mkdir -p "$MODEL_DIR" "$CAL_DIR"
[[ -f "$INIT_CKPT" ]] || { echo "missing INIT_CKPT=$INIT_CKPT" >&2; exit 2; }

case "$VARIANT" in
  expert_balanced)
    POS_W=${POS_W:-7.0}; NEAR_W=${NEAR_W:-1.4}; CONTACT_W=${CONTACT_W:-1.4}
    PAIR_W=${PAIR_W:-1.5}; TOP_W=${TOP_W:-1.0}; FP_W=${FP_W:-2.0}
    OPP_W=${OPP_W:-1.25}; OPP_POS_W=${OPP_POS_W:-8.0}; LR=${LR:-0.00005} ;;
  expert_precision)
    POS_W=${POS_W:-9.0}; NEAR_W=${NEAR_W:-1.3}; CONTACT_W=${CONTACT_W:-1.7}
    PAIR_W=${PAIR_W:-2.0}; TOP_W=${TOP_W:-1.5}; FP_W=${FP_W:-3.0}
    OPP_W=${OPP_W:-1.75}; OPP_POS_W=${OPP_POS_W:-10.0}; LR=${LR:-0.00003} ;;
  *) echo "unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac
FREEZE_PREFIXES=${FREEZE_PREFIXES:-encoder,root_queries,root_cross_attn,root_self_attn,root_norm1,root_norm2,root_norm3,root_ffn,option_embeddings,option_feature_proj,root_logit_head,margin_head,obs_embed_head,utility_head,root_signature_head,root_future_signature_head}

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONUNBUFFERED=1 python -u -m ocrap.cli train \
  --dataset "$TRAIN_MIX" --val-dataset "$VAL_MIX" --output "$MODEL_DIR" \
  --set training.init_checkpoint="$INIT_CKPT" \
  --set training.freeze_param_prefixes="$FREEZE_PREFIXES" \
  --set training.epochs=${EPOCHS:-10} --set training.early_stop_patience=${PATIENCE:-3} \
  --set training.batch_size=${BATCH_SIZE:-64} --set training.lr="$LR" --set training.weight_decay=0.00005 --set training.grad_clip=${GRAD_CLIP:-1.0} \
  --set training.group_batching=true --set training.group_batching_replacement=false \
  --set training.group_batch_hard_boost=0 \
  --set training.group_batch_positive_advantage_boost=0 \
  --set training.artifact_sampler_weight=0 --set training.negative_deployable_sampler_weight=0 \
  --set training.safe_positive_sampler_weight=0 --set training.regime_balance_power=1.0 \
  --set training.num_workers=${NUM_WORKERS:-4} --set training.progress=true \
  --set training.require_cuda=true --set training.save_every_epoch=false \
  --set training.frozen_modules_eval=true \
  --set training.best_metric=loss_direct_recovery_value_worst --set training.best_metric_mode=min \
  --set model.encoder_type=structured_transformer --set model.d_model=192 --set model.d_obs=64 \
  --set model.transformer_layers=2 --set model.transformer_heads=4 \
  --set model.direct_recovery_value_head=true --set model.direct_recovery_value_pooling=candidate_concat \
  --set model.direct_recovery_value_output=score \
  --set model.direct_recovery_value_regime_conditioning=false \
  --set model.direct_recovery_value_experts=true \
  --set model.direct_recovery_value_num_experts=2 \
  --set model.direct_recovery_value_expert_routing=soft_observation \
  --set model.direct_recovery_value_router_temperature=${ROUTER_TEMPERATURE:-0.70} \
  --set model.direct_recovery_value_router_pooling=shared_raw \
  --set model.direct_recovery_opportunity_head=true \
  --set loss_weights.dep=0 --set loss_weights.orc=0 --set loss_weights.assign=0 --set loss_weights.sig=0 \
  --set loss_weights.margin=0 --set loss_weights.obs=0 --set loss_weights.anti_oracle=0 \
  --set loss_weights.artifact_gap=0 --set loss_weights.admission=0 --set loss_weights.utility=0 \
  --set loss_weights.option_q=0 --set loss_weights.option_admission=0 --set loss_weights.option_success=0 \
  --set loss_weights.option_success_bce=0 --set loss_weights.option_best=0 --set loss_weights.group_ce=0 \
  --set loss_weights.group_distill=0 --set loss_weights.nominal_switch=0 --set loss_weights.safe_nominal=0 \
  --set loss_weights.protective_macro=0 --set loss_weights.macro_drs=0 --set loss_weights.ddc=0 \
  --set loss_weights.teacher_pcd_direct=0 --set loss_weights.recovery_advantage=0 \
  --set loss_weights.direct_recovery_value=${DIRECT_VALUE_WEIGHT:-1.0} \
  --set loss_weights.direct_router_balance=${ROUTER_BALANCE_WEIGHT:-0.005} \
  --set loss_weights.direct_router_supervision=${ROUTER_SUPERVISION_WEIGHT:-0.25} \
  --set training.direct_value_macro_ids=${DIRECT_MACRO_IDS:-2,3,5,7} --set training.direct_value_bucket_ids=1,2 \
  --set training.direct_value_temperature=${DIRECT_TEMPERATURE:-0.10} \
  --set training.direct_value_positive_gain=${DIRECT_POSITIVE_GAIN:-0.015} \
  --set training.direct_value_negative_gain=${DIRECT_NEGATIVE_GAIN:-0.010} \
  --set training.direct_value_rank_margin=${DIRECT_RANK_MARGIN:-0.020} \
  --set training.direct_value_expert_supervision=true \
  --set training.direct_value_expert_bucket_ids=1,2 \
  --set training.direct_value_mixture_weight=${DIRECT_MIXTURE_WEIGHT:-0.25} \
  --set training.direct_router_supervision=true \
  --set training.direct_value_point_weight=${DIRECT_POINT_WEIGHT:-0.02} --set training.direct_value_listwise_weight=0.05 \
  --set training.direct_value_centered_weight=1.0 --set training.direct_value_advantage_weight=1.0 \
  --set training.direct_value_output_mode=score --set training.direct_value_pairwise_weight="$PAIR_W" \
  --set training.direct_value_top_rank_weight="$TOP_W" \
  --set training.direct_value_positive_group_weight="$POS_W" \
  --set training.direct_value_negative_group_weight=1.0 \
  --set training.direct_value_ambiguous_group_weight=${DIRECT_AMBIGUOUS_WEIGHT:-0.25} \
  --set training.direct_value_near_weight="$NEAR_W" --set training.direct_value_contact_weight="$CONTACT_W" \
  --set training.direct_value_min_group_range=0.005 --set training.direct_value_false_positive_weight="$FP_W" \
  --set training.direct_value_opportunity_weight="$OPP_W" \
  --set training.direct_value_opportunity_pos_weight="$OPP_POS_W" \
  2>&1 | tee "$MODEL_DIR/train_v46_race.log"

# OC-MERO heads remain frozen and in eval mode.  Each expert is directly
# supervised on its task bucket; the soft router is only the deployable
# observation-conditioned interpolation mechanism.
for bucket in mix safe near contact; do
  case "$bucket" in
    mix) data="$VAL_MIX"; min=100;; safe) data="$OCRAP_ROOT/val_safe"; min=50;;
    near) data="$OCRAP_ROOT/val_near_contact"; min=50;; contact) data="$OCRAP_ROOT/val_contact"; min=50;; esac
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u -m ocrap.cli calibrate --dataset "$data" --checkpoint "$MODEL_DIR/best.pt" \
    --output "$CAL_DIR/calibration_${bucket}_v46.json" --set calibration.required_min_for_delta="$min" \
    2>&1 | tee "$CAL_DIR/calibrate_${bucket}_v46.log"
done
python tools/write_gamma_by_bucket.py --safe "$CAL_DIR/calibration_safe_v46.json" --near "$CAL_DIR/calibration_near_v46.json" \
  --contact "$CAL_DIR/calibration_contact_v46.json" --delta 0.05 --output "$CAL_DIR/gamma_rec_by_bucket_v46.json"

BASE_RUN="$RUN" CKPT="$MODEL_DIR/best.pt" CAL_DIR="$CAL_DIR" CAL_GPU="$TRAIN_GPU" \
  bash scripts/calibrate_ocrap_v46_race.sh
