#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
export CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}" CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}"
REFERENCE_A="${V4858_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
B_RUN="$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_native_B"
C_RUN="$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_main"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.58-reference-reuse-contract.json"
FEAS_AUDIT="$BASE_OUT/OC-RAP-v48.58-RIFA-feasibility-role-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.58-DCP-DRFC-BCDE-RIFA-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.58-PIPELINE_COMPLETE.json"
mkdir -p "$BASE_OUT"
# Reruns are fail-clean: remove stale top-level products before any new work so
# an aborted execution cannot be mistaken for a completed v48.58 experiment.
rm -f "$REF_AUDIT" "$FEAS_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" \
      "$B_RUN.zip" "$C_RUN.zip" "$BASE_OUT/OC-RAP-v48.58-RIFA-audits.zip"

# v48.58 is deliberately NOT a CMRI continuation. Stage-I is the validated
# v48.56-A semantic reference. B and C add only a lexicographic absolute
# feasibility stage after the frozen rank proposal and before relative evidence.
bash scripts/prepare_v48_45_protocol.sh
python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
for f in evidence_adapt_teacher_pcd_index.jsonl evidence_adapt_teacher_pcd_index_summary.json \
         evidence_adapt_dev_teacher_pcd_index.jsonl evidence_adapt_dev_teacher_pcd_index_summary.json; do
  [[ -s "$REFERENCE_A/$f" ]] || { echo "missing reference teacher artifact: $REFERENCE_A/$f" >&2; exit 30; }
done

CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"
DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"
TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
for d in "$CERT_NEAR" "$CERT_CONTACT" "$DEV_NEAR" "$DEV_CONTACT" "$TRAIN_NEAR" "$TRAIN_CONTACT"; do
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing v48.58 protocol root: $d" >&2; exit 30; }
done

materialize_native_arm(){
  local v src dst
  rm -rf "$B_RUN"; mkdir -p "$B_RUN/candidates" "$B_RUN/logs"
  for v in balanced precision; do
    src="$REFERENCE_A/candidates/$v"; dst="$B_RUN/candidates/$v"
    [[ -f "$src/model_v48_trac_sr/best.pt" && -f "$src/POLICY_CONTRACT.env" ]] || { echo "missing A candidate $v" >&2; exit 30; }
    mkdir -p "$dst/model_v48_trac_sr"
    cp --reflink=auto "$src/model_v48_trac_sr/best.pt" "$dst/model_v48_trac_sr/best.pt"
    [[ -f "$src/model_v48_trac_sr/train_summary.json" ]] && cp "$src/model_v48_trac_sr/train_summary.json" "$dst/model_v48_trac_sr/train_summary.json"
    cp "$src/POLICY_CONTRACT.env" "$dst/POLICY_CONTRACT.env"
    python tools/rewrite_v48_58_policy_contract.py --contract "$dst/POLICY_CONTRACT.env" --mode native --threshold 0.5
    [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]] && cp "$src/FACTOR_SUPPORT_CONTRACT.env" "$dst/FACTOR_SUPPORT_CONTRACT.env"
  done
  python - "$B_RUN/V48_58_FACTOR_CONTRACT.json" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps({
 'event':'v48_58_factor_contract','version':'v48.58-DCP-DRFC-BCDE-RIFA','arm':'B',
 'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'raw native absolute feasibility',
 'stage_ii_predicate':'sigmoid(predicted R_dep) >= 0.5','stage_ii_threshold':0.5,
 'placement':'rank_topk -> absolute feasibility -> relative opportunity/harm -> evidence rerank',
 'cmri':False,'root_logit_recalibration':False,'new_learned_parameters':False,
 'teacher_semantics_changed':False,'strategy_regime_conditioning':False,'proposal_top_k':5,
 'threshold_search':False,'test_roots_read':False,'created_unix':time.time()},indent=2,sort_keys=True)+'\n')
PY
}

# Fixed architecture semantics needed to instantiate the v48.56-A checkpoint.
# They are frozen in C; only direct_absolute_feasibility_head is trainable.
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false EVIDENCE_UNBOUNDED_HARM_FACTORS=false EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0
export EVIDENCE_COMPONENT_SCALE=6.0 EVIDENCE_COMPONENT_HEADS=true EVIDENCE_COMPONENT_COUNT=5
export EVIDENCE_DUAL_INTERACTION_BRIDGE=true EVIDENCE_FACTORIZED_HARM_INTERACTION=false EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false
export EVIDENCE_RANK_BENEFIT_SKIP=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=false
export EVIDENCE_ROCT_BENEFIT=true EVIDENCE_ROCT_DEPLOYABILITY=true EVIDENCE_ROCT_SCALE=3.0 EVIDENCE_ROCT_ALPHA=0.20 EVIDENCE_ROCT_BETA=0.20 EVIDENCE_ROCT_TOP_M=8 EVIDENCE_ROCT_OPTION_TEMPERATURE=0.35
export EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050
export EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=false EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true
export EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=false EVIDENCE_PHYSICAL_STUDENT_DRS=false
export EVIDENCE_DEP_BOUNDARY_ALIGNED=false EVIDENCE_GAP_ORDINAL_ONLY=false EVIDENCE_COMMON_MEASURE_ROOT_MASS=false
# v48.58.1 engineering hotfix: the v48.56-A Stage-I checkpoint does not contain
# the historical learned admission calibrator.  RIFA adds only the 9-parameter
# absolute-feasibility head, so fail closed against accidental re-instantiation.
export EVIDENCE_ADMISSION_HEAD=false
export EVIDENCE_NATIVE_DRS_TOLERANCE=0.05 EVIDENCE_NATIVE_DEPLOYABILITY_TOLERANCE=0.05 EVIDENCE_NATIVE_GAP_TOLERANCE=0.05 EVIDENCE_NATIVE_POSITIVE_GAIN=0.015
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction PROPOSAL_TOP_K=5
export TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class EVAL_OPTION_EXECUTION_SEMANTICS=observation_class OPTION_EXECUTION_SEMANTICS=observation_class
export ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_MODE=raw ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_SCALE=0.10 ORDINAL_EVIDENCE_COMPONENT_MARGIN_CANONICAL_SCALES=""
export ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_RELIABILITY="1,1,1,0,0"

train_afe(){
  # Do not combine these declarations: bash expands RHS values before the local
  # assignments take effect.  The old one-line declaration therefore captured
  # the global loop variable left by materialize_native_arm (v=precision), making
  # both background jobs use/write the precision variant.
  local v="$1"
  local gpu="$2"
  local src="$REFERENCE_A/candidates/$v"
  local dst="$C_RUN/candidates/$v"
  mkdir -p "$dst"
  if [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]]; then set -a; source "$src/FACTOR_SUPPORT_CONTRACT.env"; set +a; else export EVIDENCE_COMPONENT_RELIABILITY="1,1,1,0,0"; fi
  RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" \
  INIT_CKPT="$src/model_v48_trac_sr/best.pt" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" \
  GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" \
  VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" \
  EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_feasibility_head \
  STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_feasibility_head \
  ABSOLUTE_FEASIBILITY_HEAD=true ABSOLUTE_FEASIBILITY_WEIGHT=1.0 \
  GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false \
  BEST_METRIC=direct_absolute_feasibility_bce BEST_METRIC_MIN_DELTA=0.00001 \
  EVIDENCE_ADAPT_EPOCHS="${V4858_AFE_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4858_AFE_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4858_AFE_LR:-0.001}" \
  MAX_EVIDENCE_CALIBRATOR_PARAMS=9 OCRAP_ALGORITHM_VERSION="v48.58-DCP-DRFC-BCDE-RIFA-AFE" \
  bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$C_RUN/logs/afe_${v}.log" 2>&1
  grep -qx 'ABSOLUTE_FEASIBILITY_MODE=learned' "$dst/POLICY_CONTRACT.env" || { echo "AFE policy contract missing learned mode for $v" >&2; exit 30; }
  grep -qx 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' "$dst/POLICY_CONTRACT.env" || { echo "AFE policy contract missing fixed 0.5 threshold for $v" >&2; exit 30; }
  grep -qx 'SELECTION_SEMANTICS=rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank' "$dst/POLICY_CONTRACT.env" || { echo "AFE selection contract mismatch for $v" >&2; exit 30; }
  python tools/check_v48_58_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --output "$dst/V48_58_STAGE_I_STATE_ISOLATION.json"
}

materialize_native_arm
rm -rf "$C_RUN"; mkdir -p "$C_RUN/candidates" "$C_RUN/logs"
set +e
train_afe balanced "$GPU0" & p0=$!
train_afe precision "$GPU1" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?
set -e
[[ "$r0" == 0 && "$r1" == 0 ]] || { echo "v48.58 AFE training failed balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/check_v48_58_variant_isolation.py \
  --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" \
  --native-run "$B_RUN" --learned-run "$C_RUN" \
  --output "$C_RUN/V48_58_VARIANT_ISOLATION.json"
python - "$C_RUN/V48_58_FACTOR_CONTRACT.json" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps({
 'event':'v48_58_factor_contract','version':'v48.58-DCP-DRFC-BCDE-RIFA','arm':'C/Main',
 'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'learned Absolute Feasibility Evidence (AFE)',
 'afe_features':'detached [ROCT_abs(4), native_certificate_abs(4)]','afe_head':'Linear(8,1), 9 parameters',
 'afe_target':'1[R_dep_star(candidate) >= 0]','afe_scope':'candidate-only Near+Contact adaptation-train',
 'afe_threshold':0.5,'afe_threshold_search':False,'stage_i_trainable':False,
 'afe_group_batch_stratified':False,'afe_group_batching_replacement':False,
 'placement':'rank_topk -> absolute feasibility -> relative opportunity/harm -> evidence rerank',
 'cmri':False,'root_logit_recalibration':False,'regime_id_input':False,'strategy_regime_conditioning':False,
 'teacher_semantics_changed':False,'proposal_top_k':5,'test_roots_read':False,'created_unix':time.time()},indent=2,sort_keys=True)+'\n')
PY

run_calibration(){
  local run="$1" mode="$2" attempt="$3"
  set +e
  OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="$attempt" \
  OCRAP_IMPLEMENTATION_VERSION="v48.58.2-RIFA-SELECTION-CONTRACT-HOTFIX" \
  CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
  ABSOLUTE_FEASIBILITY_MODE="$mode" ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class \
  HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 \
  bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_58_certificate_controller.log" 2>&1
  local rc=$?; set -e
  case "$rc" in 0|20) echo "$mode calibration valid evidence RC=$rc";; *) echo "$mode calibration engineering failure RC=$rc" >&2; return 30;; esac
}
run_calibration "$B_RUN" native "v48.58-B-native-$(date +%s)"
run_calibration "$C_RUN" learned "v48.58-C-AFE-$(date +%s)"

python tools/audit_v48_58_feasibility_role.py --arm "A=$REFERENCE_A" --arm "B=$B_RUN" --arm "C_Main=$C_RUN" --output "$FEAS_AUDIT"
python tools/compare_v48_58_rifa.py --a "$REFERENCE_A" --b "$B_RUN" --c "$C_RUN" --feasibility-audit "$FEAS_AUDIT" --output "$COMPARE"
python tools/check_v48_58_pipeline_complete.py \
  --reference-contract "$REF_AUDIT" --native-run "$B_RUN" --learned-run "$C_RUN" \
  --feasibility-audit "$FEAS_AUDIT" --comparison "$COMPARE" --output "$PIPELINE_COMPLETE"

cd "$BASE_OUT"
for run in "$B_RUN" "$C_RUN"; do b="$(basename "$run")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"; done
zip -qj OC-RAP-v48.58-RIFA-audits.zip "$REF_AUDIT" "$FEAS_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" \
  "$C_RUN/V48_58_VARIANT_ISOLATION.json" \
  "$C_RUN/candidates/balanced/V48_58_STAGE_I_STATE_ISOLATION.json" "$C_RUN/candidates/precision/V48_58_STAGE_I_STATE_ISOLATION.json"
echo "v48.58 complete. Upload $(basename "$B_RUN").zip + $(basename "$C_RUN").zip + OC-RAP-v48.58-RIFA-audits.zip"
