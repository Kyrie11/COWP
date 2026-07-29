#!/usr/bin/env bash
set -euo pipefail

export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export TRAIN_MIX=${TRAIN_MIX:-$OCRAP_ROOT/train_safe,$OCRAP_ROOT/train_contact,$OCRAP_ROOT/train_near_contact}
export VAL_MIX=${VAL_MIX:-$OCRAP_ROOT/val_safe,$OCRAP_ROOT/val_contact,$OCRAP_ROOT/val_near_contact}
export BASE_RUN=${BASE_RUN:-runs/ocrap_v39_ocrac_balanced}
export INIT_CKPT=${INIT_CKPT:-$BASE_RUN/model_v39_ocrac/best.pt}
export VARIANT=${VARIANT:-head_only}
export RUN=${RUN:-runs/ocrap_v40_ocuvra_${VARIANT}}
export MODEL_DIR=${MODEL_DIR:-$RUN/model_v40_ocuvra}
export CAL_DIR=${CAL_DIR:-$RUN/calibration}
export TRAIN_GPU=${TRAIN_GPU:-0}
mkdir -p "$MODEL_DIR" "$CAL_DIR"
[[ -f "$INIT_CKPT" ]] || { echo "missing INIT_CKPT=$INIT_CKPT" >&2; exit 2; }

# OC-UVRA separates the OC-MERO deployability certificate from candidate-value
# ranking.  The default head_only variant leaves every v39 certificate head
# unchanged and learns only the direct value/uncertainty head.  adapter_light is
# a secondary candidate that also permits gentle option/margin adaptation.
case "$VARIANT" in
  head_only)
    FREEZE_PREFIXES=${FREEZE_PREFIXES:-encoder,root_queries,root_cross_attn,root_self_attn,root_norm1,root_norm2,root_norm3,root_ffn,option_embeddings,option_feature_proj,root_logit_head,margin_head,obs_embed_head,utility_head,root_signature_head,root_future_signature_head}
    LEGACY_MARGIN_WEIGHT=${LEGACY_MARGIN_WEIGHT:-0.0}
    LEGACY_OBS_WEIGHT=${LEGACY_OBS_WEIGHT:-0.0}
    LEGACY_OPTION_WEIGHT=${LEGACY_OPTION_WEIGHT:-0.0}
    ;;
  adapter_light)
    FREEZE_PREFIXES=${FREEZE_PREFIXES:-encoder,root_queries,root_cross_attn,root_self_attn,root_norm1,root_norm2,root_norm3,root_ffn}
    LEGACY_MARGIN_WEIGHT=${LEGACY_MARGIN_WEIGHT:-0.20}
    LEGACY_OBS_WEIGHT=${LEGACY_OBS_WEIGHT:-0.15}
    LEGACY_OPTION_WEIGHT=${LEGACY_OPTION_WEIGHT:-0.10}
    ;;
  *) echo "unknown VARIANT=$VARIANT (head_only|adapter_light)" >&2; exit 2 ;;
esac

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONUNBUFFERED=1 python -u -m ocrap.cli train \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --output "$MODEL_DIR" \
  --set training.init_checkpoint="$INIT_CKPT" \
  --set training.freeze_param_prefixes="$FREEZE_PREFIXES" \
  --set training.epochs=${EPOCHS:-18} \
  --set training.early_stop_patience=${PATIENCE:-5} \
  --set training.batch_size=${BATCH_SIZE:-64} \
  --set training.lr=${LR:-0.00008} \
  --set training.weight_decay=${WEIGHT_DECAY:-0.00001} \
  --set training.artifact_sampler_weight=0.8 \
  --set training.negative_deployable_sampler_weight=1.2 \
  --set training.regime_balance_power=1.0 \
  --set training.group_batching=true \
  --set training.group_batching_replacement=true \
  --set training.group_batch_hard_macro_ids=2,3,5,7 \
  --set training.group_batch_hard_bucket_ids=1,2 \
  --set training.group_batch_hard_min_r_dep=${GROUP_HARD_MIN_R_DEP:-0.10} \
  --set training.group_batch_hard_boost=${GROUP_HARD_BOOST:-5.0} \
  --set training.num_workers=${NUM_WORKERS:-4} \
  --set training.progress=true \
  --set training.require_cuda=true \
  --set training.save_every_epoch=true \
  --set training.best_metric=loss_direct_recovery_value \
  --set training.best_metric_mode=min \
  --set model.encoder_type=structured_transformer \
  --set model.d_model=192 \
  --set model.d_obs=64 \
  --set model.transformer_layers=2 \
  --set model.transformer_heads=4 \
  --set model.direct_recovery_value_head=true \
  --set loss_weights.assign=0.0 \
  --set loss_weights.sig=0.0 \
  --set loss_weights.margin="$LEGACY_MARGIN_WEIGHT" \
  --set loss_weights.obs="$LEGACY_OBS_WEIGHT" \
  --set loss_weights.anti_oracle=0.0 \
  --set loss_weights.artifact_gap=0.0 \
  --set loss_weights.admission=0.0 \
  --set loss_weights.utility=0.0 \
  --set loss_weights.option_q="$LEGACY_OPTION_WEIGHT" \
  --set loss_weights.option_admission="$LEGACY_OPTION_WEIGHT" \
  --set loss_weights.option_success="$LEGACY_OPTION_WEIGHT" \
  --set loss_weights.option_success_bce="$LEGACY_OPTION_WEIGHT" \
  --set loss_weights.option_best="$LEGACY_OPTION_WEIGHT" \
  --set loss_weights.group_ce=0.0 \
  --set loss_weights.group_distill=0.0 \
  --set loss_weights.nominal_switch=0.0 \
  --set loss_weights.safe_nominal=0.0 \
  --set loss_weights.protective_macro=0.0 \
  --set loss_weights.macro_drs=0.0 \
  --set loss_weights.ddc=0.0 \
  --set loss_weights.teacher_pcd_direct=0.0 \
  --set loss_weights.recovery_advantage=0.0 \
  --set loss_weights.direct_recovery_value=${DIRECT_VALUE_WEIGHT:-10.0} \
  --set training.direct_value_macro_ids=2,3,5,7 \
  --set training.direct_value_bucket_ids=1,2 \
  --set training.direct_value_temperature=${DIRECT_TEMPERATURE:-0.10} \
  --set training.direct_value_positive_gain=${DIRECT_POSITIVE_GAIN:-0.025} \
  --set training.direct_value_negative_gain=${DIRECT_NEGATIVE_GAIN:-0.020} \
  --set training.direct_value_rank_margin=${DIRECT_RANK_MARGIN:-0.035} \
  --set training.direct_value_point_weight=${DIRECT_POINT_WEIGHT:-1.0} \
  --set training.direct_value_listwise_weight=${DIRECT_LISTWISE_WEIGHT:-1.2} \
  --set training.direct_value_advantage_weight=${DIRECT_ADVANTAGE_WEIGHT:-1.5} \
  --set training.direct_value_false_positive_weight=${DIRECT_FALSE_POSITIVE_WEIGHT:-1.5} \
  --set training.direct_value_variance_floor=${DIRECT_VARIANCE_FLOOR:-0.0025} \
  2>&1 | tee "$MODEL_DIR/train_v40_ocuvra.log"

for bucket in mix safe near contact; do
  case "$bucket" in
    mix) data="$VAL_MIX"; min=100 ;;
    safe) data="$OCRAP_ROOT/val_safe"; min=50 ;;
    near) data="$OCRAP_ROOT/val_near_contact"; min=50 ;;
    contact) data="$OCRAP_ROOT/val_contact"; min=50 ;;
  esac
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
    --dataset "$data" --checkpoint "$MODEL_DIR/best.pt" \
    --output "$CAL_DIR/calibration_${bucket}_v40.json" \
    --set calibration.required_min_for_delta="$min" \
    2>&1 | tee "$CAL_DIR/calibrate_${bucket}_v40.log"
done

python tools/write_gamma_by_bucket.py \
  --safe "$CAL_DIR/calibration_safe_v40.json" \
  --near "$CAL_DIR/calibration_near_v40.json" \
  --contact "$CAL_DIR/calibration_contact_v40.json" \
  --delta 0.05 --output "$CAL_DIR/gamma_rec_by_bucket_v40.json" \
  2>&1 | tee "$CAL_DIR/write_gamma_v40.log"

printf '\n[v40:%s] model=%s\ncalibration=%s\ngamma=%s\n' "$VARIANT" "$MODEL_DIR/best.pt" "$CAL_DIR/calibration_mix_v40.json" "$CAL_DIR/gamma_rec_by_bucket_v40.json"

# Calibrate selection-valid direct-value advantage bounds separately by regime.
# The score is max over all recovery candidates in a scene-time group, so the
# resulting lower bound remains valid after the selector chooses a candidate.
python tools/calibrate_direct_value_advantage.py \
  --dataset "$OCRAP_ROOT/val_near_contact" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --output "$CAL_DIR/direct_value_advantage_near_v40.json" \
  --delta ${DIRECT_CAL_DELTA:-0.05} \
  --required-min-groups ${DIRECT_CAL_MIN_GROUPS:-50} \
  2>&1 | tee "$CAL_DIR/direct_value_advantage_near_v40.log"

python tools/calibrate_direct_value_advantage.py \
  --dataset "$OCRAP_ROOT/val_contact" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --output "$CAL_DIR/direct_value_advantage_contact_v40.json" \
  --delta ${DIRECT_CAL_DELTA:-0.05} \
  --required-min-groups ${DIRECT_CAL_MIN_GROUPS:-50} \
  2>&1 | tee "$CAL_DIR/direct_value_advantage_contact_v40.log"
