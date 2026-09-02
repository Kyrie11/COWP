#!/usr/bin/env bash
# V48.75 OC-STCA: preregistered supervision truth-contract adjudication after
# V48.74 signed-viability STOP.  No new planner feature is introduced.
export OCRAP_V48_74_SIGNED_VIABILITY=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
REFERENCE_A="${V4875_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
Q67_RUN="${V4875_Q67:-$BASE_OUT/ocrap_v48_67_dcp_drfc_bcde_rifa_pbrw_control}"
T68_RUN="${V4875_T68:-$BASE_OUT/ocrap_v48_68_dcp_drfc_bcde_rifa_rtrw_fidelity}"
C_RUN="$BASE_OUT/ocrap_v48_75_dcp_drfc_bcde_rifa_stca_projection_censored"
D_RUN="$BASE_OUT/ocrap_v48_75_dcp_drfc_bcde_rifa_stca_main"
V74_COMPLETE="${V4875_V74_COMPLETE:-$BASE_OUT/OC-RAP-v48.74-PIPELINE_COMPLETE.json}"
V74_COMPARE="${V4875_V74_COMPARE:-$BASE_OUT/OC-RAP-v48.74-DCP-DRFC-BCDE-RIFA-OC-SVBW-comparison.json}"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.75-reference-reuse-contract.json"
RUNTIME_AUDIT="$BASE_OUT/OC-RAP-v48.75-runtime-code-contract.json"
TRUTH_AUDIT="$BASE_OUT/OC-RAP-v48.75-OC-STCA-truth-contract-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.75-DCP-DRFC-BCDE-RIFA-OC-STCA-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.75-PIPELINE_COMPLETE.json"
mkdir -p "$BASE_OUT"
rm -rf "$C_RUN" "$D_RUN"
rm -f "$REF_AUDIT" "$RUNTIME_AUDIT" "$TRUTH_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$C_RUN.zip" "$D_RUN.zip" "$BASE_OUT/OC-RAP-v48.75-OC-STCA-audits.zip"

python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python tools/check_v48_75_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME_AUDIT"
python - "$V74_COMPLETE" "$V74_COMPARE" "$REFERENCE_A" "$Q67_RUN" "$T68_RUN" <<'PY'
import json,pathlib,sys
complete,compare=map(pathlib.Path,sys.argv[1:3]);runs=[pathlib.Path(x) for x in sys.argv[3:]]
if not complete.is_file():raise SystemExit(f'missing V48.74 prerequisite sentinel: {complete}')
m=json.loads(complete.read_text())
if not (m.get('valid') and m.get('attribution_ready') and m.get('engineering_version')=='v48.74.2-OC-SVBW-ENGFIX' and not m.get('test_roots_read')):raise SystemExit(f'V48.74 prerequisite invalid: {m}')
if not compare.is_file():raise SystemExit(f'missing V48.74 comparison: {compare}')
p=(json.loads(compare.read_text()).get('preregistered_decision') or {})
if not (p.get('status')=='STOP' and p.get('next_branch')=='signed_viability_stop_then_supervision_truth_contract_no_parameter_sweep' and p.get('P_first_order_mechanism',{}).get('go') is False and p.get('Q_main_mechanism',{}).get('go') is False):raise SystemExit(f'V48.74 branch mismatch: {p}')
for r in runs:
 if not r.is_dir():raise SystemExit(f'missing prerequisite run: {r}')
PY
for f in evidence_adapt_teacher_pcd_index.jsonl evidence_adapt_teacher_pcd_index_summary.json evidence_adapt_dev_teacher_pcd_index.jsonl evidence_adapt_dev_teacher_pcd_index_summary.json; do
 [[ -s "$REFERENCE_A/$f" ]] || { echo "missing reference teacher artifact: $REFERENCE_A/$f" >&2; exit 30; }
done
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
for d in "$CERT_NEAR" "$CERT_CONTACT" "$DEV_NEAR" "$DEV_CONTACT" "$TRAIN_NEAR" "$TRAIN_CONTACT"; do [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing V48.75 protocol root: $d" >&2; exit 30; }; done

# Frozen V48.56-A Stage-I / relative-evidence contract.
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false EVIDENCE_UNBOUNDED_HARM_FACTORS=false EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0
export EVIDENCE_COMPONENT_SCALE=6.0 EVIDENCE_COMPONENT_HEADS=true EVIDENCE_COMPONENT_COUNT=5
export EVIDENCE_DUAL_INTERACTION_BRIDGE=true EVIDENCE_FACTORIZED_HARM_INTERACTION=false EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false
export EVIDENCE_RANK_BENEFIT_SKIP=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=false
export EVIDENCE_ROCT_BENEFIT=true EVIDENCE_ROCT_DEPLOYABILITY=true EVIDENCE_ROCT_SCALE=3.0 EVIDENCE_ROCT_ALPHA=0.20 EVIDENCE_ROCT_BETA=0.20 EVIDENCE_ROCT_TOP_M=8 EVIDENCE_ROCT_OPTION_TEMPERATURE=0.35
export EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050
export EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=false EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true
export EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=false EVIDENCE_PHYSICAL_STUDENT_DRS=false
export EVIDENCE_DEP_BOUNDARY_ALIGNED=false EVIDENCE_GAP_ORDINAL_ONLY=false EVIDENCE_COMMON_MEASURE_ROOT_MASS=false EVIDENCE_ADMISSION_HEAD=false
export EVIDENCE_NATIVE_DRS_TOLERANCE=0.05 EVIDENCE_NATIVE_DEPLOYABILITY_TOLERANCE=0.05 EVIDENCE_NATIVE_GAP_TOLERANCE=0.05 EVIDENCE_NATIVE_POSITIVE_GAIN=0.015
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction PROPOSAL_TOP_K=5
export TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class EVAL_OPTION_EXECUTION_SEMANTICS=observation_class OPTION_EXECUTION_SEMANTICS=observation_class
export ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_MODE=raw ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_SCALE=0.10 ORDINAL_EVIDENCE_COMPONENT_MARGIN_CANONICAL_SCALES=""
export ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_RELIABILITY="1,1,1,0,0"

train_stca_arm(){
 local run="$1" arm="$2" fidelity="$3";mkdir -p "$run/candidates" "$run/logs"
 train_one(){
  local v="$1"
  local gpu="$2"
  local src="$REFERENCE_A/candidates/$v"
  local dst="$run/candidates/$v"
  mkdir -p "$dst"
  if [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]];then set -a;source "$src/FACTOR_SUPPORT_CONTRACT.env";set +a;else export EVIDENCE_COMPONENT_RELIABILITY="1,1,1,0,0";fi
  RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" INIT_CKPT="$src/model_v48_trac_sr/best.pt" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" \
  EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_semantic_witness_gain STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_semantic_witness_gain \
  ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=false ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=false ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION=false ABSOLUTE_COMMON_WITNESS_CORRECTION=false ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION=false \
  ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT=true SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=false \
  SEMANTIC_WITNESS_ROUTE_ALIGNMENT=true SEMANTIC_WITNESS_REENTRY_ALIGNMENT=true SEMANTIC_WITNESS_CONTROL_PROJECTION=true SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false \
  SEMANTIC_WITNESS_PROJECTION_FIDELITY="$fidelity" SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY=false SEMANTIC_WITNESS_ROBUST_OCCUPANCY=false SEMANTIC_WITNESS_SOFT_OCCUPANCY_DISAGREEMENT=false \
  SEMANTIC_WITNESS_BOUNDARY_LOCALIZED_OCCUPANCY_TRUST=false SEMANTIC_WITNESS_HISTORY_OCCUPANCY_REACHABILITY=false SEMANTIC_WITNESS_INTERACTION_BOX_SUPPORT=false SEMANTIC_WITNESS_INTERACTION_HULL_SUPPORT=false SEMANTIC_WITNESS_INTERACTION_ANCHOR_SUPPORT=false SEMANTIC_WITNESS_INTERACTION_RESPONSE_SUPPORT=false \
  ABSOLUTE_FEASIBILITY_WEIGHT=1.0 ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=censor_exact_0p5 GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false BEST_METRIC=direct_absolute_feasibility_bce BEST_METRIC_MIN_DELTA=0.00001 \
  EVIDENCE_ADAPT_EPOCHS="${V4875_STCA_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4875_STCA_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4875_STCA_LR:-0.001}" MAX_EVIDENCE_CALIBRATOR_PARAMS=2 \
  OCRAP_ALGORITHM_VERSION="v48.75-DCP-DRFC-BCDE-RIFA-OC-STCA-${arm}" bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$run/logs/stca_${arm}_${v}.log" 2>&1
  grep -qx 'ABSOLUTE_FEASIBILITY_MODE=learned' "$dst/POLICY_CONTRACT.env" || exit 30
  grep -qx 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' "$dst/POLICY_CONTRACT.env" || exit 30
  grep -qx 'ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=censor_exact_0p5' "$dst/POLICY_CONTRACT.env" || exit 30
  python tools/check_v48_75_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --fidelity "$fidelity" --output "$dst/V48_75_STAGE_I_STATE_ISOLATION.json"
 }
 set +e;train_one balanced "$GPU0" & p0=$!;train_one precision "$GPU1" & p1=$!;wait "$p0";r0=$?;wait "$p1";r1=$?;set -e
 [[ "$r0" == 0 && "$r1" == 0 ]] || { echo "V48.75 $arm training failed balanced=$r0 precision=$r1" >&2;exit 30; }
 python tools/check_v48_75_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --stca-run "$run" --fidelity "$fidelity" --output "$run/V48_75_VARIANT_ISOLATION.json"
 python - "$run/V48_75_FACTOR_CONTRACT.json" "$arm" "$fidelity" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);arm=sys.argv[2];fid=sys.argv[3].lower()=='true'
d={'event':'v48_75_factor_contract','version':'v48.75-DCP-DRFC-BCDE-RIFA-OC-STCA','engineering_version':'v48.75.0-OC-STCA','arm':arm,'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'Observation-Consistent Structural-Truth Contract Adjudication (OC-STCA)','source_intervention':'no new planner feature; exact R_dep*=0.5 structural plateau candidates are censored from absolute-feasibility BCE/model selection only','floor_value':0.5,'floor_tolerance':1e-8,'floor_rows_relabelled':False,'teacher_files_modified':False,'dataset_reconstruction':False,'absolute_feasibility_truth_contract':'censor_exact_0p5','semantic_witness_feature_schema':4 if fid else 3,'semantic_witness_feature_source':'robust_trust_projected_recovery_witness' if fid else 'projected_boundary_common_executable_recovery_witness','active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,'route_alignment':True,'reentry_alignment':True,'control_projection':True,'boundary_transport':False,'projection_fidelity':fid,'demand_normalized_fidelity':False,'robust_occupancy':False,'soft_occupancy_disagreement':False,'boundary_localized_occupancy_trust':False,'history_occupancy_reachability':False,'interaction_box_support':False,'interaction_hull_support':False,'interaction_anchor_support':False,'interaction_response_support':False,'trainable_state':'direct_absolute_semantic_witness_gain[2]','trainable_parameters':2,'target_for_nonfloor_rows':'1[R_dep_star(candidate)>=0]','censored_rows':'exact R_dep_star=0.5 candidates are unknown for this BCE, not negative','scope':'Near+Contact adaptation-train; one shared regime-agnostic mechanism','threshold':0.5,'threshold_search':False,'proposal_top_k':5,'proposal_expansion':False,'root_retraining':False,'relative_score_intervention':False,'teacher_future_input':False,'regime_id_input':False,'teacher_semantics_changed':False,'test_roots_read':False,'created_unix':time.time()}
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
}
train_stca_arm "$C_RUN" C75_PROJ_CENSORED false
train_stca_arm "$D_RUN" D75_FIDELITY_CENSORED true

run_calibration(){ local run="$1" tag="$2";set +e;OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.75-${tag}-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.75.0-OC-STCA-${tag}" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_75_${tag}_certificate_controller.log" 2>&1;rc=$?;set -e;case "$rc" in 0|20) echo "$tag calibration valid evidence RC=$rc";;*)echo "$tag calibration engineering failure RC=$rc" >&2;return 30;;esac; }
run_calibration "$C_RUN" C75_PROJ_CENSORED
run_calibration "$D_RUN" D75_FIDELITY_CENSORED
python tools/audit_v48_75_truth_contract.py --q67 "$Q67_RUN" --t68 "$T68_RUN" --c75 "$C_RUN" --d75 "$D_RUN" --output "$TRUTH_AUDIT"
python tools/compare_v48_75_stca.py --truth-audit "$TRUTH_AUDIT" --v74-complete "$V74_COMPLETE" --v74-comparison "$V74_COMPARE" --output "$COMPARE"
python tools/check_v48_75_pipeline_complete.py --reference-contract "$REF_AUDIT" --runtime-contract "$RUNTIME_AUDIT" --v74-complete "$V74_COMPLETE" --v74-comparison "$V74_COMPARE" --censored-projection-run "$C_RUN" --censored-fidelity-run "$D_RUN" --truth-audit "$TRUTH_AUDIT" --comparison "$COMPARE" --output "$PIPELINE_COMPLETE"
cd "$BASE_OUT"
for run in "$C_RUN" "$D_RUN";do b="$(basename "$run")";rm -f "$b.zip";zip -qr "$b.zip" "$b";done
zip -qj OC-RAP-v48.75-OC-STCA-audits.zip "$REF_AUDIT" "$RUNTIME_AUDIT" "$TRUTH_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V74_COMPLETE" "$V74_COMPARE" "$C_RUN/V48_75_VARIANT_ISOLATION.json" "$C_RUN/V48_75_FACTOR_CONTRACT.json" "$D_RUN/V48_75_VARIANT_ISOLATION.json" "$D_RUN/V48_75_FACTOR_CONTRACT.json" "$C_RUN/candidates/balanced/V48_75_STAGE_I_STATE_ISOLATION.json" "$C_RUN/candidates/precision/V48_75_STAGE_I_STATE_ISOLATION.json" "$D_RUN/candidates/balanced/V48_75_STAGE_I_STATE_ISOLATION.json" "$D_RUN/candidates/precision/V48_75_STAGE_I_STATE_ISOLATION.json"
echo "v48.75 complete. Upload $(basename "$D_RUN").zip + $(basename "$C_RUN").zip + OC-RAP-v48.75-OC-STCA-audits.zip"
