#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
export CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}" CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}"
REFERENCE_A="${V4860_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
B_RUN="${V4860_NATIVE_B:-$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_native_B}"
C_RUN="${V4860_AFE_C:-$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_main}"
D_RUN="${V4860_ORFC_D:-$BASE_OUT/ocrap_v48_59_dcp_drfc_bcde_rifa_orfc_main}"
E_RUN="$BASE_OUT/ocrap_v48_60_dcp_drfc_bcde_rifa_cphr_main"
V59_COMPLETE="${V4860_V59_COMPLETE:-$BASE_OUT/OC-RAP-v48.59-PIPELINE_COMPLETE.json}"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.60-reference-reuse-contract.json"
FEAS_AUDIT="$BASE_OUT/OC-RAP-v48.60-CPHR-feasibility-role-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.60-DCP-DRFC-BCDE-RIFA-CPHR-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.60-PIPELINE_COMPLETE.json"
mkdir -p "$BASE_OUT"
rm -rf "$E_RUN"
rm -f "$REF_AUDIT" "$FEAS_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$E_RUN.zip" "$BASE_OUT/OC-RAP-v48.60-CPHR-audits.zip"

# V48.60 is a single-axis follow-up to the V48.59 cross-severity ORFC STOP.
# A/B/C/D are reused bitwise.  E/Main changes only the absolute-feasibility source:
# six zero-initialized bounded non-negative weights on deployable signed physical
# headroom computed from the COMPLETE executable prefix/current observed agents,
# carried as a Stage-II-only side channel so the frozen Stage-I flat input is unchanged, added to
# the frozen native R_dep logit.  No regime id, free bias, threshold search,
# centering, proposal expansion, root retraining, or margin-head retraining.
bash scripts/prepare_v48_45_protocol.sh
python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python - "$V59_COMPLETE" "$B_RUN" "$C_RUN" "$D_RUN" <<'PY'
import json,pathlib,sys
p,b,c,d_run=map(pathlib.Path,sys.argv[1:])
if not p.is_file(): raise SystemExit(f'missing V48.59 prerequisite sentinel: {p}')
meta=json.loads(p.read_text())
if not (meta.get('valid') and meta.get('attribution_ready') and not meta.get('test_roots_read')):
    raise SystemExit('V48.59 prerequisite is not attribution-ready')
for run in (b,c,d_run):
    if not run.is_dir(): raise SystemExit(f'missing prerequisite run: {run}')
PY
for f in evidence_adapt_teacher_pcd_index.jsonl evidence_adapt_teacher_pcd_index_summary.json \
         evidence_adapt_dev_teacher_pcd_index.jsonl evidence_adapt_dev_teacher_pcd_index_summary.json; do
  [[ -s "$REFERENCE_A/$f" ]] || { echo "missing reference teacher artifact: $REFERENCE_A/$f" >&2; exit 30; }
done

CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
for d in "$CERT_NEAR" "$CERT_CONTACT" "$DEV_NEAR" "$DEV_CONTACT" "$TRAIN_NEAR" "$TRAIN_CONTACT"; do
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing v48.60 protocol root: $d" >&2; exit 30; }
done
mkdir -p "$E_RUN/candidates" "$E_RUN/logs"

# Exact frozen V48.56-A Stage-I architecture contract (same as V48.58).
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false EVIDENCE_UNBOUNDED_HARM_FACTORS=false EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0
export EVIDENCE_COMPONENT_SCALE=6.0 EVIDENCE_COMPONENT_HEADS=true EVIDENCE_COMPONENT_COUNT=5
export EVIDENCE_DUAL_INTERACTION_BRIDGE=true EVIDENCE_FACTORIZED_HARM_INTERACTION=false EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false
export EVIDENCE_RANK_BENEFIT_SKIP=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=false
export EVIDENCE_ROCT_BENEFIT=true EVIDENCE_ROCT_DEPLOYABILITY=true EVIDENCE_ROCT_SCALE=3.0 EVIDENCE_ROCT_ALPHA=0.20 EVIDENCE_ROCT_BETA=0.20 EVIDENCE_ROCT_TOP_M=8 EVIDENCE_ROCT_OPTION_TEMPERATURE=0.35
export EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050
export EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=false EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true
export EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=false EVIDENCE_PHYSICAL_STUDENT_DRS=false
export EVIDENCE_DEP_BOUNDARY_ALIGNED=false EVIDENCE_GAP_ORDINAL_ONLY=false EVIDENCE_COMMON_MEASURE_ROOT_MASS=false
export EVIDENCE_ADMISSION_HEAD=false
export EVIDENCE_NATIVE_DRS_TOLERANCE=0.05 EVIDENCE_NATIVE_DEPLOYABILITY_TOLERANCE=0.05 EVIDENCE_NATIVE_GAP_TOLERANCE=0.05 EVIDENCE_NATIVE_POSITIVE_GAIN=0.015
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction PROPOSAL_TOP_K=5
export TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class EVAL_OPTION_EXECUTION_SEMANTICS=observation_class OPTION_EXECUTION_SEMANTICS=observation_class
export ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_MODE=raw ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_SCALE=0.10 ORDINAL_EVIDENCE_COMPONENT_MARGIN_CANONICAL_SCALES=""
export ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_RELIABILITY="1,1,1,0,0"

train_cphr(){
  local v="$1"; local gpu="$2"; local src="$REFERENCE_A/candidates/$v"; local dst="$E_RUN/candidates/$v"
  mkdir -p "$dst"
  if [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]]; then set -a; source "$src/FACTOR_SUPPORT_CONTRACT.env"; set +a; else export EVIDENCE_COMPONENT_RELIABILITY="1,1,1,0,0"; fi
  RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" \
  INIT_CKPT="$src/model_v48_trac_sr/best.pt" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" \
  GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" \
  EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_physical_headroom_weight \
  STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_physical_headroom_weight \
  ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=false ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=true ABSOLUTE_FEASIBILITY_WEIGHT=1.0 \
  GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false \
  BEST_METRIC=direct_absolute_feasibility_bce BEST_METRIC_MIN_DELTA=0.00001 \
  EVIDENCE_ADAPT_EPOCHS="${V4860_CPHR_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4860_CPHR_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4860_CPHR_LR:-0.001}" \
  MAX_EVIDENCE_CALIBRATOR_PARAMS=6 OCRAP_ALGORITHM_VERSION="v48.60-DCP-DRFC-BCDE-RIFA-CPHR" \
  bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$E_RUN/logs/cphr_${v}.log" 2>&1
  grep -qx 'ABSOLUTE_FEASIBILITY_MODE=learned' "$dst/POLICY_CONTRACT.env" || { echo "CPHR policy mode missing for $v" >&2; exit 30; }
  grep -qx 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' "$dst/POLICY_CONTRACT.env" || { echo "CPHR fixed threshold missing for $v" >&2; exit 30; }
  grep -qx 'SELECTION_SEMANTICS=rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank' "$dst/POLICY_CONTRACT.env" || { echo "CPHR selection contract mismatch for $v" >&2; exit 30; }
  python tools/check_v48_60_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --output "$dst/V48_60_STAGE_I_STATE_ISOLATION.json"
}

set +e
train_cphr balanced "$GPU0" & p0=$!
train_cphr precision "$GPU1" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?
set -e
[[ "$r0" == 0 && "$r1" == 0 ]] || { echo "v48.60 CPHR training failed balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/check_v48_60_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --cphr-run "$E_RUN" --output "$E_RUN/V48_60_VARIANT_ISOLATION.json"
python - "$E_RUN/V48_60_FACTOR_CONTRACT.json" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);p.write_text(json.dumps({
 'event':'v48_60_factor_contract','version':'v48.60-DCP-DRFC-BCDE-RIFA-CPHR','arm':'E/Main',
 'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'Contextual Physical Headroom Reserve (CPHR)',
 'source_intervention':'native R_dep logit + bounded nonnegative linear correction over six signed observable physical headroom coordinates',
 'physical_features':['min_clearance_reserve','terminal_clearance_reserve','clearance_recovery_gain','stopping_reserve','control_envelope_reserve','stability_reserve'],
 'physical_feature_schema':2,'physical_feature_source':'full_executable_prefix_side_channel','prefix_timestamp_contract':'prefix_states[i] occurs at (i+1)/sample_rate_hz',
 'trainable_state':'direct_absolute_physical_headroom_weight[6]','trainable_parameters':6,
 'initialization':'all zeros; no bias; execution-exact native B source at epoch 0','weight_constraint':'elementwise [0,2]',
 'target':'1[R_dep_star(candidate) >= 0]','scope':'candidate-only Near+Contact adaptation-train; shared regime-agnostic function at execution','threshold':0.5,'threshold_search':False,
 'afe_head':False,'orfc_option_bias':False,'centering':False,'proposal_top_k':5,'proposal_expansion':False,'root_retraining':False,'margin_head_retraining':False,
 'teacher_margin_distillation':False,'regime_id_input':False,'strategy_regime_conditioning':False,'teacher_semantics_changed':False,'test_roots_read':False,'created_unix':time.time()
},indent=2,sort_keys=True)+'\n')
PY

run_calibration(){
  set +e
  OUTPUTDIR="$E_RUN" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.60-E-CPHR-$(date +%s)" \
  OCRAP_IMPLEMENTATION_VERSION="v48.60.1-CPHR-FULLPREFIX" \
  CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
  ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class \
  HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 \
  bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$E_RUN/logs/v48_60_certificate_controller.log" 2>&1
  local rc=$?;set -e
  case "$rc" in 0|20) echo "CPHR calibration valid evidence RC=$rc";; *) echo "CPHR calibration engineering failure RC=$rc" >&2; return 30;; esac
}
run_calibration

python tools/audit_v48_60_feasibility_role.py --arm "A=$REFERENCE_A" --arm "B_native=$B_RUN" --arm "C_AFE=$C_RUN" --arm "D_ORFC=$D_RUN" --arm "E_CPHR=$E_RUN" --output "$FEAS_AUDIT"
python tools/compare_v48_60_cphr.py --a "$REFERENCE_A" --b "$B_RUN" --c "$C_RUN" --d "$D_RUN" --e "$E_RUN" --feasibility-audit "$FEAS_AUDIT" --output "$COMPARE"
python tools/check_v48_60_pipeline_complete.py --reference-contract "$REF_AUDIT" --v59-complete "$V59_COMPLETE" --cphr-run "$E_RUN" --feasibility-audit "$FEAS_AUDIT" --comparison "$COMPARE" --output "$PIPELINE_COMPLETE"

cd "$BASE_OUT"; b="$(basename "$E_RUN")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"
zip -qj OC-RAP-v48.60-CPHR-audits.zip "$REF_AUDIT" "$FEAS_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V59_COMPLETE" \
  "$E_RUN/V48_60_VARIANT_ISOLATION.json" "$E_RUN/V48_60_FACTOR_CONTRACT.json" \
  "$E_RUN/candidates/balanced/V48_60_STAGE_I_STATE_ISOLATION.json" "$E_RUN/candidates/precision/V48_60_STAGE_I_STATE_ISOLATION.json"
echo "v48.60 complete. Upload $b.zip + OC-RAP-v48.60-CPHR-audits.zip"
