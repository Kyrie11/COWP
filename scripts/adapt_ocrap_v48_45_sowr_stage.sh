#!/usr/bin/env bash
set -Eeuo pipefail
# v48.45 SOWR — Shared-Option Witness Recalibration.
# Recalibrate only the paper-matched recovery witness heads with the exact
# training teacher contract, then freeze them again before OCAF/ROCT fitting.
# No Safe/Near/Contact id, router, threshold, or policy is exposed to the model.
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
MARGIN_WITNESS="${V4845_SOWR_MARGIN_WITNESS:-0}"
OBS_KERNEL="${V4845_SOWR_OBS_KERNEL:-0}"

if [[ "$MARGIN_WITNESS" != 1 && "$OBS_KERNEL" != 1 ]]; then
  echo "SOWR stage requires V4845_SOWR_MARGIN_WITNESS=1 and/or V4845_SOWR_OBS_KERNEL=1" >&2
  exit 2
fi
[[ -f "$INIT_CKPT" ]] || { echo "missing INIT_CKPT=$INIT_CKPT" >&2; exit 2; }
[[ -f "$GROUP_INDEX" ]] || { echo "missing GROUP_INDEX=$GROUP_INDEX" >&2; exit 2; }
python tools/check_v48_45_sowr_source_architecture.py \
  --checkpoint "$INIT_CKPT" --output "$RUN/SOWR_SOURCE_ARCHITECTURE_CONTRACT.json"

prefixes=""
if [[ "$MARGIN_WITNESS" == 1 ]]; then
  # Root probability and root-option signed margin jointly determine the
  # option-resolved shared-recovery witness.  The shared encoder/root decoder
  # remains frozen, avoiding the broad encoder fine-tuning already tried in v48.
  prefixes="root_logit_head,margin_head"
fi
if [[ "$OBS_KERNEL" == 1 ]]; then
  [[ -n "$prefixes" ]] && prefixes+=","
  prefixes+="obs_embed_head"
fi

mkdir -p "$RUN"
cat > "$RUN/SOWR_STAGE_ARCHITECTURE.json" <<JSON
{
  "version": "v48.45-SOWR",
  "role": "shared_option_witness_recalibration",
  "margin_witness_recalibration": $([[ "$MARGIN_WITNESS" == 1 ]] && echo true || echo false),
  "observation_kernel_recalibration": $([[ "$OBS_KERNEL" == 1 ]] && echo true || echo false),
  "trainable_param_prefixes": "${prefixes}",
  "shared_encoder_frozen": true,
  "root_decoder_frozen": true,
  "direct_policy_heads_frozen": true,
  "regime_id_exposed": false,
  "teacher_contract": "m_star+c_star+root_probs -> OC-MERO shared-option q",
  "test_roots_read": false
}
JSON

# Use the full OC-MERO forward only in this short witness stage.  Direct policy
# loss is zero.  The high weights are on signed margins, observation equivalence,
# deployability, option-resolved q, shared-option admission, and best shared option.
# All supervision already exists in the training NPZs; no new labels or regime
# branches are introduced.
# SOWR is an internal witness recalibration stage.  It must not run the generic
# post-training bucket calibrator against default val_* roots; downstream v48.36
# performs the authoritative calibration after factor adaptation.  Keep this
# comment outside the backslash-continued environment-assignment command below.
# Stage-isolation contract: outer v48.44-D exports ROCT/OCAF flags for the
# downstream factor stage.  They must not leak into this source-witness model,
# whose checkpoint intentionally has no learned component-evidence heads.
RUN="$RUN" MODEL_DIR="$RUN/model_v48_sowr" CAL_DIR="$RUN/calibration" \
INIT_CKPT="$INIT_CKPT" VARIANT="$VARIANT" TRAIN_MIX="$TRAIN_MIX" VAL_MIX="$VAL_MIX" CAL_MIX="$VAL_MIX" \
GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="${VAL_GROUP_INDEX:-}" TRAIN_GPU="$TRAIN_GPU" \
TRAINABLE_PARAM_PREFIXES="$prefixes" STRICT_INIT_PREFIXES="" \
SET_CONTEXT_ENABLED=false PREFERENCE_HEAD_ENABLED=false PREFERENCE_CONTEXT_ENABLED=false RELATIVE_INCLUDE_ABSOLUTE=false \
SET_TOURNAMENT_ENABLED=true SET_TOURNAMENT_HIDDEN=48 SET_TOURNAMENT_HEADS=4 SET_TOURNAMENT_DROPOUT=0.05 SET_TOURNAMENT_REPLACE_BASE=true \
DELTA_HEAD_ENABLED=true DELTA_MODE=ordinal_evidence DELTA_HIDDEN=48 DELTA_DROPOUT=0.02 DELTA_REGIME_EXPERTS=true DELTA_POLICY_FEATURES=true \
EVIDENCE_CALIBRATOR_ENABLED=false EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=relative \
EVIDENCE_DUAL_INTERACTION_BRIDGE=false \
EVIDENCE_FACTORIZED_HARM_INTERACTION=false EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false \
EVIDENCE_RANK_BENEFIT_SKIP=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=false \
EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=false EVIDENCE_ROCT_BENEFIT=false EVIDENCE_ROCT_DEPLOYABILITY=false \
EVIDENCE_UNIFIED_EXPERTS=false EVIDENCE_COMPONENT_HEADS=false EVIDENCE_CONCORD=false \
EVIDENCE_ADMISSION_HEAD=false EVIDENCE_FRONTIER=false EVIDENCE_RESERVE_FACTOR_ALIGNMENT=false \
EVIDENCE_ADMISSION_PRIOR_MODE=risk_centered EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.025 EVIDENCE_COMPONENT_RELIABILITY= \
DIRECT_ONLY_FAST_PATH=false DIRECT_VALUE_WEIGHT=0 \
LOSS_ASSIGN="$([[ "$MARGIN_WITNESS" == 1 ]] && echo 0.35 || echo 0)" \
LOSS_MARGIN="$([[ "$MARGIN_WITNESS" == 1 ]] && echo 2.00 || echo 0)" \
LOSS_OBS="$([[ "$OBS_KERNEL" == 1 ]] && echo 1.50 || echo 0)" \
LOSS_DEP=0.75 LOSS_ORC=0.25 LOSS_ANTI_ORACLE=0.50 LOSS_ARTIFACT_GAP=0.25 LOSS_ADMISSION=0.50 \
LOSS_OPTION_Q=1.50 LOSS_OPTION_ADMISSION=1.00 LOSS_OPTION_SUCCESS=0.50 LOSS_OPTION_SUCCESS_BCE=0.50 LOSS_OPTION_BEST=1.00 \
LOSS_UTILITY=0 LOSS_SIG=0 LOSS_GROUP_CE=0 LOSS_GROUP_DISTILL=0 LOSS_NOMINAL_SWITCH=0 LOSS_SAFE_NOMINAL=0 \
LOSS_PROTECTIVE_MACRO=0 LOSS_MACRO_DRS=0 LOSS_TEACHER_PCD_DIRECT=0 LOSS_RECOVERY_ADVANTAGE=0 LOSS_DIRECT_ROUTER_BALANCE=0 \
EPOCHS="${SOWR_EPOCHS:-8}" PATIENCE="${SOWR_PATIENCE:-3}" LR="${SOWR_LR:-0.00005}" \
ENCODER_LR_SCALE=0 ENCODER_ANCHOR_WEIGHT=0 BEST_METRIC=loss BEST_METRIC_MIN_DELTA="${SOWR_BEST_METRIC_MIN_DELTA:-0.000001}" \
EVALUATE_INITIAL_CHECKPOINT=true GROUP_BATCH_STRATIFIED=true GROUP_BATCHING_REPLACEMENT=true \
SKIP_POST_TRAIN_CALIBRATION=1 \
NUM_WORKERS="${NUM_WORKERS:-2}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" BATCH_SIZE="${SOWR_BATCH_SIZE:-72}" \
  bash scripts/train_ocrap_v48_trac_sr.sh

CKPT="$RUN/model_v48_sowr/best.pt"
SUMMARY="$RUN/model_v48_sowr/train_summary.json"
[[ -f "$CKPT" && -f "$SUMMARY" ]] || { echo "SOWR output missing" >&2; exit 30; }

python tools/check_v48_45_sowr_stage_isolation.py \
  --source "$INIT_CKPT" --checkpoint "$CKPT" --allowed-prefixes "$prefixes" \
  --output "$RUN/SOWR_STAGE_ISOLATION_CONTRACT.json"

python - "$RUN" "$INIT_CKPT" "$CKPT" "$SUMMARY" "$MARGIN_WITNESS" "$OBS_KERNEL" "$prefixes" <<'PY'
import hashlib,json,pathlib,sys,time
run,source,ckpt,summary=map(pathlib.Path,sys.argv[1:5])
margin=sys.argv[5]=='1'; obs=sys.argv[6]=='1'; prefixes=sys.argv[7].split(',')
doc=json.loads(summary.read_text())
hist=doc.get('history') or []
initial=(hist[0].get('val') if hist and int(hist[0].get('epoch',-1))==0 else {}) or {}
best_epoch=int(doc.get('best_epoch',-1))
best={}
for row in hist:
    if int(row.get('epoch',-999))==best_epoch:
        best=(row.get('val') or {}); break
keys=['loss','loss_root','loss_margin','loss_obs','loss_dep','loss_orc','loss_admission',
      'loss_option_q','loss_option_admission','loss_option_success','loss_option_success_bce','loss_option_best']
out={
 'event':'v48_45_sowr_complete','version':'v48.45-SOWR','created_unix':time.time(),
 'margin_witness_recalibration':margin,'observation_kernel_recalibration':obs,
 'trainable_param_prefixes':prefixes,'regime_id_exposed':False,'test_roots_read':False,
 'source_checkpoint':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
 'best_epoch':best_epoch,
 'initial_val':{k:float(initial[k]) for k in keys if k in initial},
 'best_val':{k:float(best[k]) for k in keys if k in best},
}
for k in list(out['best_val']):
    if k in out['initial_val']:
        out.setdefault('delta_best_minus_initial',{})[k]=out['best_val'][k]-out['initial_val'][k]
p=run/'SOWR_COMPLETE.json'; p.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
PY
