#!/usr/bin/env bash
set -Eeuo pipefail
# v48.46 OC-SWIC witness stage: one paper-property witness at a time.
# No regime labels, regime routers, regime thresholds, or regime-specific policy.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

RUN="${RUN:?RUN is required}"
INIT_CKPT="${INIT_CKPT:?INIT_CKPT is required}"
TRAIN_MIX="${TRAIN_MIX:?TRAIN_MIX is required}"
VAL_MIX="${VAL_MIX:?VAL_MIX is required}"
GROUP_INDEX="${GROUP_INDEX:?GROUP_INDEX is required}"
TRAIN_GPU="${TRAIN_GPU:-0}"
VARIANT="${VARIANT:?VARIANT is required}"
WITNESS_STAGE="${V4846_WITNESS_STAGE:?V4846_WITNESS_STAGE=obs|margin is required}"
OPTION_SEMANTICS="${OPTION_EXECUTION_SEMANTICS:-global}"

case "$WITNESS_STAGE" in
  obs)
    prefixes="obs_embed_head"
    loss_margin=0; loss_obs=1.50
    epochs="${V4846_WITNESS_OBS_EPOCHS:-5}"
    ;;
  margin)
    prefixes="margin_head"
    loss_margin=2.00; loss_obs=0
    epochs="${V4846_WITNESS_MARGIN_EPOCHS:-5}"
    ;;
  *) echo "unknown V4846_WITNESS_STAGE=$WITNESS_STAGE" >&2; exit 2 ;;
esac
case "$OPTION_SEMANTICS" in
  global)
    loss_option_success=0.50; loss_option_success_bce=0.50; loss_option_best=1.00
    loss_option_class_success=0; loss_option_class_best=0
    ;;
  observation_class)
    loss_option_success=0; loss_option_success_bce=0; loss_option_best=0
    loss_option_class_success=0.50; loss_option_class_best=1.00
    ;;
  *) echo "unknown OPTION_EXECUTION_SEMANTICS=$OPTION_SEMANTICS" >&2; exit 2 ;;
esac

[[ -f "$INIT_CKPT" ]] || { echo "missing INIT_CKPT=$INIT_CKPT" >&2; exit 2; }
[[ -f "$GROUP_INDEX" ]] || { echo "missing GROUP_INDEX=$GROUP_INDEX" >&2; exit 2; }
mkdir -p "$RUN"
python tools/check_v48_45_sowr_source_architecture.py \
  --checkpoint "$INIT_CKPT" --output "$RUN/V48_46_SOURCE_ARCHITECTURE_CONTRACT.json"
cat > "$RUN/V48_46_WITNESS_STAGE.json" <<JSON
{
  "version":"v48.46-OC-SWIC",
  "stage":"$WITNESS_STAGE",
  "option_execution_semantics":"$OPTION_SEMANTICS",
  "trainable_param_prefixes":"$prefixes",
  "root_logit_head_frozen":true,
  "shared_encoder_frozen":true,
  "root_decoder_frozen":true,
  "direct_policy_heads_frozen":true,
  "regime_id_exposed":false,
  "test_roots_read":false
}
JSON

# Preserve the exact v48.45 rebuilt-source architecture.  Disable all downstream
# OCAF/ROCT heads locally: this stage only calibrates the recovery witness.
RUN="$RUN" MODEL_DIR="$RUN/model_v48_46_witness" CAL_DIR="$RUN/calibration" \
INIT_CKPT="$INIT_CKPT" VARIANT="$VARIANT" TRAIN_MIX="$TRAIN_MIX" VAL_MIX="$VAL_MIX" CAL_MIX="$VAL_MIX" \
GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="${VAL_GROUP_INDEX:-}" TRAIN_GPU="$TRAIN_GPU" \
TRAINABLE_PARAM_PREFIXES="$prefixes" STRICT_INIT_PREFIXES="" \
SET_CONTEXT_ENABLED=false PREFERENCE_HEAD_ENABLED=false PREFERENCE_CONTEXT_ENABLED=false RELATIVE_INCLUDE_ABSOLUTE=false \
SET_TOURNAMENT_ENABLED=true SET_TOURNAMENT_HIDDEN=48 SET_TOURNAMENT_HEADS=4 SET_TOURNAMENT_DROPOUT=0.05 SET_TOURNAMENT_REPLACE_BASE=true \
DELTA_HEAD_ENABLED=true DELTA_MODE=ordinal_evidence DELTA_HIDDEN=48 DELTA_DROPOUT=0.02 DELTA_REGIME_EXPERTS=true DELTA_POLICY_FEATURES=true \
EVIDENCE_CALIBRATOR_ENABLED=false EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=relative EVIDENCE_DUAL_INTERACTION_BRIDGE=false \
EVIDENCE_FACTORIZED_HARM_INTERACTION=false EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false EVIDENCE_RANK_BENEFIT_SKIP=false \
EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=false \
EVIDENCE_ROCT_BENEFIT=false EVIDENCE_ROCT_DEPLOYABILITY=false EVIDENCE_UNIFIED_EXPERTS=false EVIDENCE_COMPONENT_HEADS=false \
EVIDENCE_CONCORD=false EVIDENCE_ADMISSION_HEAD=false EVIDENCE_FRONTIER=false EVIDENCE_RESERVE_FACTOR_ALIGNMENT=false \
EVIDENCE_ADMISSION_PRIOR_MODE=risk_centered EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.025 EVIDENCE_COMPONENT_RELIABILITY= \
DIRECT_ONLY_FAST_PATH=false DIRECT_VALUE_WEIGHT=0 OPTION_EXECUTION_SEMANTICS="$OPTION_SEMANTICS" \
LOSS_ASSIGN=0 LOSS_MARGIN="$loss_margin" LOSS_OBS="$loss_obs" \
LOSS_DEP=0.75 LOSS_ORC=0.25 LOSS_ANTI_ORACLE=0.50 LOSS_ARTIFACT_GAP=0.25 LOSS_ADMISSION=0.50 \
LOSS_OPTION_Q=1.50 LOSS_OPTION_ADMISSION=1.00 \
LOSS_OPTION_SUCCESS="$loss_option_success" LOSS_OPTION_SUCCESS_BCE="$loss_option_success_bce" LOSS_OPTION_BEST="$loss_option_best" \
LOSS_OPTION_CLASS_SUCCESS="$loss_option_class_success" LOSS_OPTION_CLASS_BEST="$loss_option_class_best" \
LOSS_UTILITY=0 LOSS_SIG=0 LOSS_GROUP_CE=0 LOSS_GROUP_DISTILL=0 LOSS_NOMINAL_SWITCH=0 LOSS_SAFE_NOMINAL=0 \
LOSS_PROTECTIVE_MACRO=0 LOSS_MACRO_DRS=0 LOSS_TEACHER_PCD_DIRECT=0 LOSS_RECOVERY_ADVANTAGE=0 LOSS_DIRECT_ROUTER_BALANCE=0 \
EPOCHS="$epochs" PATIENCE="${V4846_WITNESS_PATIENCE:-2}" LR="${V4846_WITNESS_LR:-0.00004}" \
ENCODER_LR_SCALE=0 ENCODER_ANCHOR_WEIGHT=0 BEST_METRIC=loss BEST_METRIC_MIN_DELTA="${V4846_WITNESS_MIN_DELTA:-0.000001}" \
EVALUATE_INITIAL_CHECKPOINT=true GROUP_BATCH_STRATIFIED=true GROUP_BATCHING_REPLACEMENT=true SKIP_POST_TRAIN_CALIBRATION=1 \
NUM_WORKERS="${NUM_WORKERS:-3}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-3}" BATCH_SIZE="${V4846_WITNESS_BATCH_SIZE:-72}" \
  bash scripts/train_ocrap_v48_trac_sr.sh

CKPT="$RUN/model_v48_46_witness/best.pt"
SUMMARY="$RUN/model_v48_46_witness/train_summary.json"
[[ -f "$CKPT" && -f "$SUMMARY" ]] || { echo "v48.46 witness output missing" >&2; exit 30; }
python tools/check_v48_45_sowr_stage_isolation.py \
  --source "$INIT_CKPT" --checkpoint "$CKPT" --allowed-prefixes "$prefixes" \
  --output "$RUN/V48_46_STAGE_ISOLATION_CONTRACT.json"
python - "$RUN" "$INIT_CKPT" "$CKPT" "$SUMMARY" "$WITNESS_STAGE" "$OPTION_SEMANTICS" "$prefixes" <<'PY'
import hashlib,json,pathlib,sys,time
run,source,ckpt,summary=map(pathlib.Path,sys.argv[1:5]); stage=sys.argv[5]; sem=sys.argv[6]; prefixes=sys.argv[7].split(',')
doc=json.loads(summary.read_text()); hist=doc.get('history') or []; best_epoch=int(doc.get('best_epoch',-1))
initial=(hist[0].get('val') if hist and int(hist[0].get('epoch',-1))==0 else {}) or {}; best={}
for row in hist:
    if int(row.get('epoch',-999))==best_epoch: best=(row.get('val') or {}); break
keys=['loss','loss_root','loss_margin','loss_obs','loss_dep','loss_orc','loss_admission','loss_option_q','loss_option_admission','loss_option_success','loss_option_success_bce','loss_option_best','loss_option_class_success','loss_option_class_best']
out={'event':'v48_46_witness_stage_complete','version':'v48.46-OC-SWIC','stage':stage,'created_unix':time.time(),
     'option_execution_semantics':sem,'trainable_param_prefixes':prefixes,'regime_id_exposed':False,'test_roots_read':False,
     'source_checkpoint':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
     'checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'best_epoch':best_epoch,
     'initial_val':{k:float(initial[k]) for k in keys if k in initial},'best_val':{k:float(best[k]) for k in keys if k in best}}
out['delta_best_minus_initial']={k:out['best_val'][k]-out['initial_val'][k] for k in out['best_val'] if k in out['initial_val']}
(run/'V48_46_WITNESS_COMPLETE.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
PY
