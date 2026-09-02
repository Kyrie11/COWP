#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
REFERENCE_A="${V4872_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
B_RUN="${V4872_NATIVE_B:-$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_native_B}"
F_RUN="${V4872_ERWF_F:-$BASE_OUT/ocrap_v48_61_dcp_drfc_bcde_rifa_erwf_main}"
P66_RUN="${V4872_P66:-$BASE_OUT/ocrap_v48_66_dcp_drfc_bcde_rifa_acrw_main}"
E70_RUN="${V4872_E70:-$BASE_OUT/ocrap_v48_70_dcp_drfc_bcde_rifa_dotw_occsoft}"
J71_RUN="${V4872_J71:-$BASE_OUT/ocrap_v48_71_dcp_drfc_bcde_rifa_borw_history}"
L_RUN="$BASE_OUT/ocrap_v48_72_dcp_drfc_bcde_rifa_iorw_box"
M_RUN="$BASE_OUT/ocrap_v48_72_dcp_drfc_bcde_rifa_iorw_main"
V71_COMPLETE="${V4872_V71_COMPLETE:-$BASE_OUT/OC-RAP-v48.71-PIPELINE_COMPLETE.json}"
V71_COMPARE="${V4872_V71_COMPARE:-$BASE_OUT/OC-RAP-v48.71-DCP-DRFC-BCDE-RIFA-OC-BORW-comparison.json}"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.72-reference-reuse-contract.json"
RUNTIME_AUDIT="$BASE_OUT/OC-RAP-v48.72-runtime-code-contract.json"
FEAS_AUDIT="$BASE_OUT/OC-RAP-v48.72-OC-IORW-feasibility-role-audit.json"
REACH_AUDIT="$BASE_OUT/OC-RAP-v48.72-OC-IORW-interaction-reachability-audit.json"
TRUTH_AUDIT="$BASE_OUT/OC-RAP-v48.72-OC-IORW-truth-floor-strata-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.72-DCP-DRFC-BCDE-RIFA-OC-IORW-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.72-PIPELINE_COMPLETE.json"
export PERSISTENT_TENSOR_CACHE="${V4872_PERSISTENT_TENSOR_CACHE:-true}"
export PERSISTENT_TENSOR_CACHE_DIR="${V4872_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4872_iorw}"
# Engineering-only: ordered parallel decode/feature construction. It changes
# neither sample order nor tensor values; schema-8 L/M share the same cache key.
export PERSISTENT_TENSOR_CACHE_BUILD_WORKERS="${V4872_TENSOR_CACHE_BUILD_WORKERS:-8}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"
rm -rf "$L_RUN" "$M_RUN"
rm -f "$REF_AUDIT" "$RUNTIME_AUDIT" "$FEAS_AUDIT" "$REACH_AUDIT" "$TRUTH_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" \
  "$L_RUN.zip" "$M_RUN.zip" "$BASE_OUT/OC-RAP-v48.72-OC-IORW-audits.zip"

# V48.72 OC-IORW is preregistered from V48.71 STOP.
# V48.71 showed that history-derived set-valued information has a real main
# effect, while the current boundary-localization statistic is not a validated
# mechanism. The remaining causal question is geometry: does the isotropic
# circumball reject recovery because it charges tangential uncertainty and
# impossible Cartesian box corners? L uses directional component-box support;
# M uses the empirical joint acceleration hull. Both keep the signed CV
# certificate, projection, projection fidelity, route/re-entry, active-set,
# top-K, threshold and all Stage-I tensors frozen. Boundary transport remains OFF.

python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python tools/check_v48_72_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME_AUDIT"
python - "$V71_COMPLETE" "$V71_COMPARE" "$REFERENCE_A" "$B_RUN" "$F_RUN" "$P66_RUN" "$E70_RUN" "$J71_RUN" <<'PY2'
import json,pathlib,sys
complete=pathlib.Path(sys.argv[1]); compare=pathlib.Path(sys.argv[2]); runs=[pathlib.Path(x) for x in sys.argv[3:]]
if not complete.is_file(): raise SystemExit(f'missing V48.71 prerequisite sentinel: {complete}')
meta=json.loads(complete.read_text())
if not (meta.get('valid') and meta.get('attribution_ready') and meta.get('engineering_version')=='v48.71.0-OC-BORW' and not meta.get('test_roots_read')):
    raise SystemExit(f'V48.71 prerequisite is not attribution-ready: {meta}')
if not compare.is_file(): raise SystemExit(f'missing V48.71 comparison: {compare}')
pr=(json.loads(compare.read_text()).get('preregistered_decision') or {})
if not (pr.get('status')=='STOP' and pr.get('occupancy_reachability_trust_go') is False and pr.get('next_branch')=='interaction_aware_observation_only_occupancy_reachability'):
    raise SystemExit(f'V48.71 branch contract mismatch: {pr}')
for run in runs:
    if not run.is_dir(): raise SystemExit(f'missing prerequisite run: {run}')
PY2
for f in evidence_adapt_teacher_pcd_index.jsonl evidence_adapt_teacher_pcd_index_summary.json evidence_adapt_dev_teacher_pcd_index.jsonl evidence_adapt_dev_teacher_pcd_index_summary.json; do
  [[ -s "$REFERENCE_A/$f" ]] || { echo "missing reference teacher artifact: $REFERENCE_A/$f" >&2; exit 30; }
done
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
for d in "$CERT_NEAR" "$CERT_CONTACT" "$DEV_NEAR" "$DEV_CONTACT" "$TRAIN_NEAR" "$TRAIN_CONTACT"; do
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing v48.72 protocol root: $d" >&2; exit 30; }
done

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

train_iorw_arm(){
  local run="$1"; local arm="$2"; local hull="$3"
  mkdir -p "$run/candidates" "$run/logs"
  train_one(){
    local v="$1"; local gpu="$2"; local src="$REFERENCE_A/candidates/$v"; local dst="$run/candidates/$v"; mkdir -p "$dst"
    if [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]]; then set -a; source "$src/FACTOR_SUPPORT_CONTRACT.env"; set +a; else export EVIDENCE_COMPONENT_RELIABILITY="1,1,1,0,0"; fi
    RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" INIT_CKPT="$src/model_v48_trac_sr/best.pt" \
    TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" \
    EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_semantic_witness_gain STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_semantic_witness_gain \
    ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=false ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=false ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION=false ABSOLUTE_COMMON_WITNESS_CORRECTION=false ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION=false \
    ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT=true SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=false \
    SEMANTIC_WITNESS_ROUTE_ALIGNMENT=true SEMANTIC_WITNESS_REENTRY_ALIGNMENT=true SEMANTIC_WITNESS_CONTROL_PROJECTION=true SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false \
    SEMANTIC_WITNESS_PROJECTION_FIDELITY=true SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY=false SEMANTIC_WITNESS_ROBUST_OCCUPANCY=false SEMANTIC_WITNESS_SOFT_OCCUPANCY_DISAGREEMENT=false \
    SEMANTIC_WITNESS_BOUNDARY_LOCALIZED_OCCUPANCY_TRUST=false SEMANTIC_WITNESS_HISTORY_OCCUPANCY_REACHABILITY=false \
    SEMANTIC_WITNESS_INTERACTION_BOX_SUPPORT=true SEMANTIC_WITNESS_INTERACTION_HULL_SUPPORT="$hull" ABSOLUTE_FEASIBILITY_WEIGHT=1.0 \
    GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false BEST_METRIC=direct_absolute_feasibility_bce BEST_METRIC_MIN_DELTA=0.00001 \
    EVIDENCE_ADAPT_EPOCHS="${V4872_IORW_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4872_IORW_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4872_IORW_LR:-0.001}" \
    MAX_EVIDENCE_CALIBRATOR_PARAMS=2 OCRAP_ALGORITHM_VERSION="v48.72-DCP-DRFC-BCDE-RIFA-OC-IORW-${arm}" \
    bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$run/logs/iorw_${arm}_${v}.log" 2>&1
    grep -qx 'ABSOLUTE_FEASIBILITY_MODE=learned' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'SELECTION_SEMANTICS=rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank' "$dst/POLICY_CONTRACT.env" || exit 30
    python tools/check_v48_72_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --box true --hull "$hull" --output "$dst/V48_72_STAGE_I_STATE_ISOLATION.json"
  }
  set +e; train_one balanced "$GPU0" & p0=$!; train_one precision "$GPU1" & p1=$!; wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
  [[ "$r0" == 0 && "$r1" == 0 ]] || { echo "v48.72 $arm training failed balanced=$r0 precision=$r1" >&2; exit 30; }
  python tools/check_v48_72_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --iorw-run "$run" --box true --hull "$hull" --output "$run/V48_72_VARIANT_ISOLATION.json"
  python - "$run/V48_72_FACTOR_CONTRACT.json" "$arm" "$hull" <<'PY2'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);arm=sys.argv[2];hull=sys.argv[3].lower()=='true'
d={'event':'v48_72_factor_contract','version':'v48.72-DCP-DRFC-BCDE-RIFA-OC-IORW','arm':arm,'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'Observation-Consistent Interaction-Oriented Reachability Witness (OC-IORW)','source_intervention':'replace V48.71 isotropic history circumball with candidate-oriented acceleration-set support; L uses the observed component box, M uses the empirical joint acceleration hull.','semantic_witness_feature_schema':8,'semantic_witness_feature_source':'interaction_oriented_history_reachability_projected_recovery_witness','active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,'route_alignment':True,'reentry_alignment':True,'control_projection':True,'boundary_transport':False,'projection_fidelity':True,'demand_normalized_fidelity':False,'robust_occupancy':False,'soft_occupancy_disagreement':False,'boundary_localized_occupancy_trust':False,'history_occupancy_reachability':False,'interaction_box_support':True,'interaction_hull_support':hull,'correction_locus':'candidate-global same-option common support before native OC-MERO aggregation','positive_logic':'strictly-positive projection-fidelity trust multiplied by 1/(1+candidate-oriented occupancy optimism); L uses component-box support and M uses empirical-hull support','negative_logic':'frozen universal-failure correction; no per-option negative veto','physical_composition':'non-compensatory historical CV clearance / legacy stopping / observation-active stability / route / persistent re-entry; actuator feasibility enforced by construction','interaction_geometry':'for each projected recovery and CV relative line of sight n, lower separation by g(t) times the acceleration-set support h_A(n); empirical hull is conv({0,a_tau}) and excludes unobserved Cartesian corners','signed_certificate':'historical CV certificate is unchanged; interaction reachability affects confidence only and cannot change positive-certificate sign/set','boundary_transport_rule':'OFF unless physically selective trust passes both label-ordering and non-floor physical-consistency gates','trainable_state':'direct_absolute_semantic_witness_gain[2]','trainable_parameters':2,'initialization':'all zeros; execution-exact native B at epoch 0','gain_constraint':'elementwise [0,2]','target':'1[R_dep_star(candidate) >= 0]','scope':'Near+Contact adaptation-train; one shared regime-agnostic mechanism','threshold':0.5,'threshold_search':False,'afe_head':False,'proposal_top_k':5,'proposal_expansion':False,'root_retraining':False,'margin_head_retraining':False,'relative_score_intervention':False,'teacher_margin_distillation':False,'teacher_future_input':False,'regime_id_input':False,'strategy_regime_conditioning':False,'teacher_semantics_changed':False,'dataset_reconstruction':False,'test_roots_read':False,'engineering_acceleration':'schema-8 materializes both L/M diagnostics under one cache key; ordered parallel cache build only, no tensor/order change','created_unix':time.time()};p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY2
}

train_iorw_arm "$L_RUN" L72_BOX_SUPPORT false
train_iorw_arm "$M_RUN" M72_Main_OCIORW true

run_calibration(){
 local run="$1"; local tag="$2"; set +e
 OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.72-${tag}-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.72.0-OC-IORW-${tag}" \
 CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
 ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 \
 bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_72_${tag}_certificate_controller.log" 2>&1
 rc=$?;set -e;case "$rc" in 0|20) echo "$tag calibration valid evidence RC=$rc";;*) echo "$tag calibration engineering failure RC=$rc" >&2;return 30;;esac
}
run_calibration "$L_RUN" L72_BOX_SUPPORT
run_calibration "$M_RUN" M72_Main_OCIORW

python tools/audit_v48_72_feasibility_role.py --arm "B_native=$B_RUN" --arm "F_ERWF=$F_RUN" --arm "P66_OCACRW=$P66_RUN" --arm "E70_OCCSOFT=$E70_RUN" --arm "J71_HISTORY_TUBE=$J71_RUN" --arm "L72_BOX_SUPPORT=$L_RUN" --arm "M72_OCIORW=$M_RUN" --output "$FEAS_AUDIT"
python tools/audit_v48_72_interaction_reachability.py --j71 "$J71_RUN" --l72 "$L_RUN" --m72 "$M_RUN" --output "$REACH_AUDIT"
python tools/audit_v48_72_truth_strata.py --arm "B_native=$B_RUN" --arm "P66_OCACRW=$P66_RUN" --arm "E70_OCCSOFT=$E70_RUN" --arm "J71_HISTORY_TUBE=$J71_RUN" --arm "L72_BOX_SUPPORT=$L_RUN" --arm "M72_OCIORW=$M_RUN" --output "$TRUTH_AUDIT"
python tools/compare_v48_72_iorw.py --feasibility-audit "$FEAS_AUDIT" --reachability-audit "$REACH_AUDIT" --truth-strata "$TRUTH_AUDIT" --v71-complete "$V71_COMPLETE" --v71-comparison "$V71_COMPARE" --output "$COMPARE"
python tools/check_v48_72_pipeline_complete.py --reference-contract "$REF_AUDIT" --v71-complete "$V71_COMPLETE" --v71-comparison "$V71_COMPARE" --runtime-contract "$RUNTIME_AUDIT" --box-run "$L_RUN" --main-run "$M_RUN" --feasibility-audit "$FEAS_AUDIT" --reachability-audit "$REACH_AUDIT" --truth-strata "$TRUTH_AUDIT" --comparison "$COMPARE" --output "$PIPELINE_COMPLETE"

cd "$BASE_OUT"
for run in "$L_RUN" "$M_RUN"; do b="$(basename "$run")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"; done
zip -qj OC-RAP-v48.72-OC-IORW-audits.zip "$REF_AUDIT" "$RUNTIME_AUDIT" "$FEAS_AUDIT" "$REACH_AUDIT" "$TRUTH_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V71_COMPLETE" "$V71_COMPARE" \
  "$L_RUN/V48_72_VARIANT_ISOLATION.json" "$L_RUN/V48_72_FACTOR_CONTRACT.json" "$M_RUN/V48_72_VARIANT_ISOLATION.json" "$M_RUN/V48_72_FACTOR_CONTRACT.json" \
  "$L_RUN/candidates/balanced/V48_72_STAGE_I_STATE_ISOLATION.json" "$L_RUN/candidates/precision/V48_72_STAGE_I_STATE_ISOLATION.json" \
  "$M_RUN/candidates/balanced/V48_72_STAGE_I_STATE_ISOLATION.json" "$M_RUN/candidates/precision/V48_72_STAGE_I_STATE_ISOLATION.json"
echo "v48.72 complete. Upload $(basename "$M_RUN").zip + OC-RAP-v48.72-OC-IORW-audits.zip; L zip is useful for independent geometry attribution."
