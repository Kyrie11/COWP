#!/usr/bin/env bash
set -Eeuo pipefail
# v48.47 DS-OFR: Decision-Sufficient Observation / Recovery-Frontier witness stage.
# No Safe/Near/Contact identifier, router, per-regime threshold, policy, or loss.
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
STAGE="${V4847_WITNESS_STAGE:?V4847_WITNESS_STAGE=decision_obs|frontier is required}"
IPBD="${V4854_INVARIANT_PHYSICAL_BOUNDARY_DISTILLATION:-false}"
OPTION_SEMANTICS="${OPTION_EXECUTION_SEMANTICS:-observation_class}"
[[ "$OPTION_SEMANTICS" == observation_class ]] || {
  echo "v48.47 requires paper-consistent OPTION_EXECUTION_SEMANTICS=observation_class" >&2; exit 2;
}

# This script is a nested witness-only process.  Mask all downstream native
# admission transports for the whole child process so outer v48.48/v48.49 arm
# flags cannot leak into OCRAPModel construction during DRFC calibration.
export EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=false
export EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=false
export EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=false
export EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false

case "$STAGE" in
  decision_obs)
    prefixes="obs_embed_head"
    epochs="${V4847_OBS_EPOCHS:-5}"
    loss_obs="${V4847_OBS_LOSS_WEIGHT:-1.50}"
    loss_margin=0
    loss_frontier=0
    dw_obs=true
    ;;
  frontier)
    prefixes="margin_head"
    epochs="${V4847_FRONTIER_EPOCHS:-5}"
    loss_obs=0
    # A small absolute-margin anchor prevents a candidate-relative calibration
    # stage from drifting the global signed scale while the structural frontier
    # term supplies the deployment-aligned gradient.
    loss_margin="${V4847_FRONTIER_MARGIN_ANCHOR_WEIGHT:-0.25}"
    loss_frontier="${V4847_FRONTIER_LOSS_WEIGHT:-2.00}"
    dw_obs=false
    ;;
  *) echo "unknown V4847_WITNESS_STAGE=$STAGE" >&2; exit 2 ;;
esac

[[ -f "$INIT_CKPT" ]] || { echo "missing INIT_CKPT=$INIT_CKPT" >&2; exit 2; }
[[ -f "$GROUP_INDEX" ]] || { echo "missing GROUP_INDEX=$GROUP_INDEX" >&2; exit 2; }
mkdir -p "$RUN"
python tools/check_v48_45_sowr_source_architecture.py \
  --checkpoint "$INIT_CKPT" --output "$RUN/V48_47_SOURCE_ARCHITECTURE_CONTRACT.json"

python - "$RUN/V48_47_WITNESS_STAGE.json" "$STAGE" "$OPTION_SEMANTICS" "$prefixes" \
  "$TRAIN_MIX" "$VAL_MIX" "$GROUP_INDEX" "$dw_obs" "$epochs" "$loss_margin" "$loss_obs" "$loss_frontier" \
  "${V4847_OBS_CONFLICT_SCALE:-3.0}" "${V4847_OBS_CONFLICT_TEMPERATURE:-0.20}" "${V4847_OBS_MAX_WEIGHT:-4.0}" \
  "${V4847_FRONTIER_SIGN_TEMPERATURE:-0.08}" "${V4847_FRONTIER_REGRESSION_WEIGHT:-1.0}" "${V4847_FRONTIER_SIGN_WEIGHT:-0.50}" "${V4850_DECISION_EQUIVALENT_FRONTIER:-false}" "${V4851_BOUNDARY_COMPLETE_FRONTIER:-false}" "${V4850_FRONTIER_GAP_TOLERANCE:-0.05}" "${V4850_FRONTIER_POSITIVE_GAIN:-0.015}" "${V4850_FRONTIER_PCD_WEIGHT:-1.0}" "${V4852_PHYSICAL_TEACHER_SIGN_ALIGNMENT:-false}" "${V4853_PHYSICAL_STUDENT_SIGN_ALIGNMENT:-false}" "$IPBD" <<'PY_STAGE'
import hashlib,json,pathlib,sys,time
p=pathlib.Path(sys.argv[1])
stage,sem,prefixes=sys.argv[2:5]
train_mix,val_mix,group_index=sys.argv[5:8]
dw=sys.argv[8].lower()=="true"
d={
 'event':'v48_47_witness_stage_contract','version':'v48.47-DS-OFR','stage':stage,
 'option_execution_semantics':sem,'trainable_param_prefixes':prefixes.split(','),
 'decision_weighted_observation_loss':dw,'root_logit_head_frozen':True,
 'witness_fast_path':stage,'frozen_modules_eval':True,
 'shared_encoder_frozen':True,'root_decoder_frozen':True,'direct_policy_heads_frozen':True,
 # Stage-local isolation contract: v48.47 calibrates only the native witness.
 # Later deterministic admission transports (v48.48 NCP / v48.49 DCP) must be
 # disabled while this witness checkpoint is being produced.
 'native_certificate_preservation':False,
 'native_margin_complete_preservation':False,
 'native_advantage_preservation':False,'native_exact_advantage_preservation':False,'native_boundary_complete_advantage_preservation':False,
 'strategy_regime_conditioning':False,'test_roots_read':False,
 'train_mix':train_mix,'val_mix':val_mix,'group_index':group_index,
 'group_index_sha256':hashlib.sha256(pathlib.Path(group_index).read_bytes()).hexdigest(),
 'epochs':int(sys.argv[9]),'loss_margin':float(sys.argv[10]),'loss_obs':float(sys.argv[11]),
 'loss_recovery_frontier':float(sys.argv[12]),
 'obs_conflict_scale':float(sys.argv[13]),'obs_conflict_temperature':float(sys.argv[14]),'obs_max_weight':float(sys.argv[15]),
 'frontier_sign_temperature':float(sys.argv[16]),'frontier_regression_weight':float(sys.argv[17]),'frontier_sign_weight':float(sys.argv[18]),
 'decision_equivalent_frontier':sys.argv[19].lower()=='true',
 'boundary_complete_frontier':sys.argv[20].lower()=='true','frontier_gap_tolerance':float(sys.argv[21]),
 'frontier_positive_gain':float(sys.argv[22]),'frontier_pcd_weight':float(sys.argv[23]),
 'physical_teacher_sign_alignment':sys.argv[24].lower()=='true',
 'physical_student_sign_alignment':sys.argv[25].lower()=='true',
 'invariant_physical_boundary_distillation':sys.argv[26].lower()=='true',
 'physical_boundary_distillation_weight':(float(sys.argv[18]) if sys.argv[26].lower()=='true' else 0.0),
 'physical_boundary_distillation_coordinate':'teacher_q_selected_mstar_zero_to_predicted_margin',
 'teacher_sign_coordinate':('q_selected_mstar_physical_drs_exact_pcd' if sys.argv[24].lower()=='true' else 'q_hard_proxy_drs_exact_pcd'),
 'student_sign_coordinate':('q_selected_predicted_margin_physical_drs_exact_pcd' if sys.argv[25].lower()=='true' else 'hard_qbest_ge_zero_root_mass_exact_pcd'),
 'frontier_order_coordinate':'smooth_boundary_drs_smooth_pcd',
 'created_unix':time.time(),
}
p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
PY_STAGE

# Preserve the exact v48.45 rebuilt-source architecture.  All downstream
# OCAF/ROCT evidence heads are stage-locally disabled: these stages calibrate
# only the paper-native observation/margin witness.
RUN="$RUN" MODEL_DIR="$RUN/model_v48_47_witness" CAL_DIR="$RUN/calibration" \
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
EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=false EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=false EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=false EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=false \
EVIDENCE_CONCORD=false EVIDENCE_ADMISSION_HEAD=false EVIDENCE_FRONTIER=false EVIDENCE_RESERVE_FACTOR_ALIGNMENT=false \
EVIDENCE_ADMISSION_PRIOR_MODE=risk_centered EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.025 EVIDENCE_COMPONENT_RELIABILITY= \
DIRECT_ONLY_FAST_PATH=false WITNESS_FAST_PATH="$STAGE" FROZEN_MODULES_EVAL=true DIRECT_VALUE_WEIGHT=0 OPTION_EXECUTION_SEMANTICS=observation_class \
DECISION_WEIGHTED_OBS_ENABLED="$dw_obs" \
DECISION_WEIGHTED_OBS_GAMMA="${V4847_OBS_GAMMA:-0.0}" \
DECISION_WEIGHTED_OBS_TEMPERATURE="${V4847_OBS_CONFLICT_TEMPERATURE:-0.20}" \
DECISION_WEIGHTED_OBS_CONFLICT_SCALE="${V4847_OBS_CONFLICT_SCALE:-3.0}" \
DECISION_WEIGHTED_OBS_MAX_WEIGHT="${V4847_OBS_MAX_WEIGHT:-4.0}" \
RECOVERY_FRONTIER_OPTION_TEMPERATURE="${V4847_FRONTIER_OPTION_TEMPERATURE:-0.35}" \
RECOVERY_FRONTIER_DEPLOYABILITY_TOLERANCE="${V4847_FRONTIER_DEP_TOLERANCE:-0.05}" \
RECOVERY_FRONTIER_DRS_TOLERANCE="${V4847_FRONTIER_DRS_TOLERANCE:-0.05}" \
RECOVERY_FRONTIER_GAP_TOLERANCE="${V4850_FRONTIER_GAP_TOLERANCE:-0.05}" \
RECOVERY_FRONTIER_POSITIVE_GAIN="${V4850_FRONTIER_POSITIVE_GAIN:-0.015}" \
RECOVERY_FRONTIER_PCD_WEIGHT="${V4850_FRONTIER_PCD_WEIGHT:-1.0}" \
RECOVERY_FRONTIER_DECISION_EQUIVALENT="${V4850_DECISION_EQUIVALENT_FRONTIER:-false}" \
RECOVERY_FRONTIER_BOUNDARY_COMPLETE="${V4851_BOUNDARY_COMPLETE_FRONTIER:-false}" \
RECOVERY_FRONTIER_PHYSICAL_TEACHER_SIGN_ALIGNMENT="${V4852_PHYSICAL_TEACHER_SIGN_ALIGNMENT:-false}" \
RECOVERY_FRONTIER_PHYSICAL_STUDENT_SIGN_ALIGNMENT="${V4853_PHYSICAL_STUDENT_SIGN_ALIGNMENT:-false}" \
INVARIANT_PHYSICAL_BOUNDARY_DISTILLATION="$IPBD" \
EVIDENCE_PHYSICAL_STUDENT_DRS=false \
RECOVERY_FRONTIER_SIGN_TEMPERATURE="${V4847_FRONTIER_SIGN_TEMPERATURE:-0.08}" \
RECOVERY_FRONTIER_REGRESSION_WEIGHT="${V4847_FRONTIER_REGRESSION_WEIGHT:-1.0}" \
RECOVERY_FRONTIER_SIGN_WEIGHT="${V4847_FRONTIER_SIGN_WEIGHT:-0.50}" \
LOSS_ASSIGN=0 LOSS_MARGIN="$loss_margin" LOSS_OBS="$loss_obs" LOSS_RECOVERY_FRONTIER="$loss_frontier" \
LOSS_PHYSICAL_BOUNDARY_DISTILL="$( [[ "$IPBD" == true ]] && printf '%s' "${V4847_FRONTIER_SIGN_WEIGHT:-0.50}" || printf '0' )" \
LOSS_DEP=0 LOSS_ORC=0 LOSS_ANTI_ORACLE=0 LOSS_ARTIFACT_GAP=0 LOSS_ADMISSION=0 \
LOSS_OPTION_Q=0 LOSS_OPTION_ADMISSION=0 LOSS_OPTION_SUCCESS=0 LOSS_OPTION_SUCCESS_BCE=0 LOSS_OPTION_BEST=0 \
LOSS_OPTION_CLASS_SUCCESS=0 LOSS_OPTION_CLASS_BEST=0 \
LOSS_UTILITY=0 LOSS_SIG=0 LOSS_GROUP_CE=0 LOSS_GROUP_DISTILL=0 LOSS_NOMINAL_SWITCH=0 LOSS_SAFE_NOMINAL=0 \
LOSS_PROTECTIVE_MACRO=0 LOSS_MACRO_DRS=0 LOSS_TEACHER_PCD_DIRECT=0 LOSS_RECOVERY_ADVANTAGE=0 LOSS_DIRECT_ROUTER_BALANCE=0 \
EPOCHS="$epochs" PATIENCE="${V4847_WITNESS_PATIENCE:-2}" LR="${V4847_WITNESS_LR:-0.00004}" \
ENCODER_LR_SCALE=0 ENCODER_ANCHOR_WEIGHT=0 BEST_METRIC=loss BEST_METRIC_MIN_DELTA="${V4847_WITNESS_MIN_DELTA:-0.000001}" \
EVALUATE_INITIAL_CHECKPOINT=true GROUP_BATCH_STRATIFIED=true GROUP_BATCHING_REPLACEMENT=true SKIP_POST_TRAIN_CALIBRATION=1 \
NUM_WORKERS="${NUM_WORKERS:-3}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-3}" BATCH_SIZE="${V4847_WITNESS_BATCH_SIZE:-72}" \
  bash scripts/train_ocrap_v48_trac_sr.sh

CKPT="$RUN/model_v48_47_witness/best.pt"
SUMMARY="$RUN/model_v48_47_witness/train_summary.json"
[[ -f "$CKPT" && -f "$SUMMARY" ]] || { echo "v48.47 witness output missing" >&2; exit 30; }
python tools/check_v48_45_sowr_stage_isolation.py \
  --source "$INIT_CKPT" --checkpoint "$CKPT" --allowed-prefixes "$prefixes" \
  --output "$RUN/V48_47_STAGE_ISOLATION_CONTRACT.json"
python - "$RUN" "$INIT_CKPT" "$CKPT" "$SUMMARY" "$STAGE" "$prefixes" <<'PY_DONE'
import hashlib,json,pathlib,sys,time
run,source,ckpt,summary=map(pathlib.Path,sys.argv[1:5]); stage=sys.argv[5]; prefixes=sys.argv[6].split(',')
doc=json.loads(summary.read_text()); hist=doc.get('history') or []; best_epoch=int(doc.get('best_epoch',-1))
initial=(hist[0].get('val') if hist and int(hist[0].get('epoch',-1))==0 else {}) or {}; best={}
for row in hist:
    if int(row.get('epoch',-999))==best_epoch: best=(row.get('val') or {}); break
keys=['loss','loss_margin','loss_obs','loss_recovery_frontier','loss_physical_boundary_distill','loss_dep','loss_option_q']
out={'event':'v48_47_witness_stage_complete','version':'v48.47-DS-OFR','stage':stage,'created_unix':time.time(),
     'option_execution_semantics':'observation_class','trainable_param_prefixes':prefixes,
     'strategy_regime_conditioning':False,'test_roots_read':False,
     'source_checkpoint':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
     'checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'best_epoch':best_epoch,
     'initial_val':{k:float(initial[k]) for k in keys if k in initial},'best_val':{k:float(best[k]) for k in keys if k in best}}
out['delta_best_minus_initial']={k:out['best_val'][k]-out['initial_val'][k] for k in out['best_val'] if k in out['initial_val']}
contract=json.loads((run/'V48_47_WITNESS_STAGE.json').read_text()); out['stage_contract']=contract
(run/'V48_47_WITNESS_COMPLETE.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
PY_DONE
