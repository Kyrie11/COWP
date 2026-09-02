#!/usr/bin/env bash

# V48.74 execution-exact acceleration controls.  Both nested arms materialize
# the same 22-D tensor; the model selects coordinate 20 or 21.
export OCRAP_V48_74_SIGNED_VIABILITY=1
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
REFERENCE_A="${V4874_REFERENCE_A:-${V4873_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}}"
B_RUN="${V4874_NATIVE_B:-${V4873_NATIVE_B:-$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_native_B}}"
F_RUN="${V4874_ERWF_F:-${V4873_ERWF_F:-$BASE_OUT/ocrap_v48_61_dcp_drfc_bcde_rifa_erwf_main}}"
P66_RUN="${V4874_P66:-${V4873_P66:-$BASE_OUT/ocrap_v48_66_dcp_drfc_bcde_rifa_acrw_main}}"
T68_RUN="${V4874_T68:-$BASE_OUT/ocrap_v48_68_dcp_drfc_bcde_rifa_rtrw_fidelity}"
N_RUN="$BASE_OUT/ocrap_v48_74_dcp_drfc_bcde_rifa_svbw_anchor"
O_RUN="$BASE_OUT/ocrap_v48_74_dcp_drfc_bcde_rifa_svbw_main"
V73_COMPLETE="${V4874_V73_COMPLETE:-$BASE_OUT/OC-RAP-v48.73-PIPELINE_COMPLETE.json}"
V73_COMPARE="${V4874_V73_COMPARE:-$BASE_OUT/OC-RAP-v48.73-DCP-DRFC-BCDE-RIFA-OC-IRRW-comparison.json}"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.74-reference-reuse-contract.json"
RUNTIME_AUDIT="$BASE_OUT/OC-RAP-v48.74-runtime-code-contract.json"
FEAS_AUDIT="$BASE_OUT/OC-RAP-v48.74-OC-SVBW-feasibility-role-audit.json"
RESPONSE_AUDIT="$BASE_OUT/OC-RAP-v48.74-OC-SVBW-signed-viability-audit.json"
TRUTH_AUDIT="$BASE_OUT/OC-RAP-v48.74-OC-SVBW-truth-floor-strata-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.74-DCP-DRFC-BCDE-RIFA-OC-SVBW-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.74-PIPELINE_COMPLETE.json"
export PERSISTENT_TENSOR_CACHE="${V4874_PERSISTENT_TENSOR_CACHE:-${V4873_PERSISTENT_TENSOR_CACHE:-true}}"
export PERSISTENT_TENSOR_CACHE_DIR="${V4874_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4874_svbw}"
# Execution-exact engineering contract. Schema 10 emits both P/Q signed-viability
# debts under one canonical cache key, so the second arm reuses the first cold build.
export PERSISTENT_TENSOR_CACHE_BUILD_WORKERS="${V4874_TENSOR_CACHE_BUILD_WORKERS:-${V4873_TENSOR_CACHE_BUILD_WORKERS:-8}}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"
rm -rf "$N_RUN" "$O_RUN"
rm -f "$REF_AUDIT" "$RUNTIME_AUDIT" "$FEAS_AUDIT" "$RESPONSE_AUDIT" "$TRUTH_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$N_RUN.zip" "$O_RUN.zip" "$BASE_OUT/OC-RAP-v48.74-OC-SVBW-audits.zip"

# Preregistered branch after V48.73 STOP: stop enriching exogenous acceleration
# response sets and test finite-time signed viability of the already actuator-
# projected executable recovery. Boundary transport, regime routing, teacher
# semantics, Stage-I, top-K and the relative ranker remain frozen.
python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python tools/check_v48_74_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME_AUDIT"
python - "$V73_COMPLETE" "$V73_COMPARE" "$REFERENCE_A" "$B_RUN" "$F_RUN" "$P66_RUN" "$T68_RUN" <<'PY2'
import json,pathlib,sys
complete=pathlib.Path(sys.argv[1]); compare=pathlib.Path(sys.argv[2]); runs=[pathlib.Path(x) for x in sys.argv[3:]]
if not complete.is_file(): raise SystemExit(f'missing V48.73 prerequisite sentinel: {complete}')
meta=json.loads(complete.read_text())
if not (meta.get('valid') and meta.get('attribution_ready') and meta.get('engineering_version')=='v48.73.0-OC-IRRW' and not meta.get('test_roots_read')): raise SystemExit(f'V48.73 prerequisite is not attribution-ready: {meta}')
if not compare.is_file(): raise SystemExit(f'missing V48.73 comparison: {compare}')
pr=(json.loads(compare.read_text()).get('preregistered_decision') or {})
if not (pr.get('status')=='STOP' and pr.get('interaction_response_reachability_go') is False and pr.get('next_branch')=='interaction_response_reachability_stop_no_parameter_sweep'): raise SystemExit(f'V48.73 branch contract mismatch: {pr}')
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
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing v48.74 protocol root: $d" >&2; exit 30; }
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

train_svbw_arm(){
  local run="$1"; local arm="$2"; local response="$3"; mkdir -p "$run/candidates" "$run/logs"
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
    SEMANTIC_WITNESS_INTERACTION_BOX_SUPPORT=true SEMANTIC_WITNESS_INTERACTION_HULL_SUPPORT=true SEMANTIC_WITNESS_INTERACTION_ANCHOR_SUPPORT=true SEMANTIC_WITNESS_INTERACTION_RESPONSE_SUPPORT="$response" \
    ABSOLUTE_FEASIBILITY_WEIGHT=1.0 GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false BEST_METRIC=direct_absolute_feasibility_bce BEST_METRIC_MIN_DELTA=0.00001 \
    EVIDENCE_ADAPT_EPOCHS="${V4874_SVBW_EPOCHS:-${V4873_SVBW_EPOCHS:-20}}" EVIDENCE_ADAPT_PATIENCE="${V4874_SVBW_PATIENCE:-${V4873_SVBW_PATIENCE:-5}}" EVIDENCE_ADAPT_LR="${V4874_SVBW_LR:-${V4873_SVBW_LR:-0.001}}" MAX_EVIDENCE_CALIBRATOR_PARAMS=2 \
    OCRAP_ALGORITHM_VERSION="v48.74-DCP-DRFC-BCDE-RIFA-OC-SVBW-${arm}" bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$run/logs/svbw_${arm}_${v}.log" 2>&1
    grep -qx 'ABSOLUTE_FEASIBILITY_MODE=learned' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'SELECTION_SEMANTICS=rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank' "$dst/POLICY_CONTRACT.env" || exit 30
    python tools/check_v48_74_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --anchor true --response "$response" --output "$dst/V48_74_STAGE_I_STATE_ISOLATION.json"
  }
  set +e; train_one balanced "$GPU0" & p0=$!; train_one precision "$GPU1" & p1=$!; wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
  [[ "$r0" == 0 && "$r1" == 0 ]] || { echo "v48.74 $arm training failed balanced=$r0 precision=$r1" >&2; exit 30; }
  python tools/check_v48_74_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --svbw-run "$run" --anchor true --response "$response" --output "$run/V48_74_VARIANT_ISOLATION.json"
  python - "$run/V48_74_FACTOR_CONTRACT.json" "$arm" "$response" <<'PY2'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);arm=sys.argv[2];response=sys.argv[3].lower()=='true'
d={'event':'v48_74_factor_contract','version':'v48.74-DCP-DRFC-BCDE-RIFA-OC-SVBW','engineering_version':'v48.74.2-OC-SVBW-ENGFIX','arm':arm,'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'Observation-Consistent Signed-Viability Barrier Witness (OC-SVBW)','source_intervention':'compute first/high-order finite-time signed-clearance viability debt on the actuator-projected executable recovery against the frozen observation-only CV continuation','semantic_witness_feature_schema':10,'semantic_witness_feature_source':'signed_finite_time_viability_projected_recovery_witness','coordinate_20':'first_order_signed_finite_time_viability_debt','coordinate_21':'non_compensatory_high_order_signed_finite_time_viability_debt','active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,'route_alignment':True,'reentry_alignment':True,'control_projection':True,'boundary_transport':False,'projection_fidelity':True,'demand_normalized_fidelity':False,'robust_occupancy':False,'soft_occupancy_disagreement':False,'boundary_localized_occupancy_trust':False,'history_occupancy_reachability':False,'interaction_box_support':True,'interaction_hull_support':True,'interaction_anchor_support':True,'interaction_response_support':response,'selector_alias':('Q74_HIGH_ORDER_COORD21' if response else 'P74_FIRST_ORDER_COORD20'),'correction_locus':'candidate-global same-option common support before native OC-MERO aggregation','positive_logic':'strictly-positive projection-fidelity trust multiplied by 1/(1+d_viability); P uses d1 and Q uses max(d1,d2), so positive-certificate sign/set is unchanged','negative_logic':'frozen universal-failure correction; no per-option negative veto','signed_certificate':'h=distance-(r_ego+r_agent); B1=h+tau*h_dot; B2=h+tau*h_dot+0.5*tau^2*h_ddot; Q debt is non-compensatory max(first,second)','observation_contract':'same frozen observation-only CV agent continuation; no teacher future, test root, latent regime id or dataset reconstruction','boundary_transport_rule':'OFF unless signed-viability trust passes label-ordering, dual selective-retention and non-floor physical-consistency gates','trainable_state':'direct_absolute_semantic_witness_gain[2]','trainable_parameters':2,'initialization':'all zeros; execution-exact native B at epoch 0','gain_constraint':'elementwise [0,2]','target':'1[R_dep_star(candidate) >= 0]','scope':'Near+Contact adaptation-train; one shared regime-agnostic mechanism','threshold':0.5,'threshold_search':False,'afe_head':False,'proposal_top_k':5,'proposal_expansion':False,'root_retraining':False,'margin_head_retraining':False,'relative_score_intervention':False,'teacher_margin_distillation':False,'teacher_future_input':False,'regime_id_input':False,'strategy_regime_conditioning':False,'teacher_semantics_changed':False,'dataset_reconstruction':False,'test_roots_read':False,'engineering_acceleration':'schema-10 materializes both P/Q debts under one canonical cache key; coordinates 0-19 remain historical; second arm must reuse the first materialization','created_unix':time.time()};p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY2
}
train_svbw_arm "$N_RUN" P74_FIRST_ORDER_SVBW false
train_svbw_arm "$O_RUN" Q74_MAIN_OC_SVBW true
run_calibration(){ local run="$1"; local tag="$2"; set +e; OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.74-${tag}-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.74.2-OC-SVBW-ENGFIX-${tag}" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_74_${tag}_certificate_controller.log" 2>&1; rc=$?; set -e; case "$rc" in 0|20) echo "$tag calibration valid evidence RC=$rc";;*) echo "$tag calibration engineering failure RC=$rc" >&2; return 30;;esac; }
run_calibration "$N_RUN" P74_FIRST_ORDER_SVBW
run_calibration "$O_RUN" Q74_MAIN_OC_SVBW
python tools/audit_v48_74_feasibility_role.py --arm "B_native=$B_RUN" --arm "F_ERWF=$F_RUN" --arm "P66_OCACRW=$P66_RUN" --arm "T68_FIDELITY=$T68_RUN" --arm "P74_FIRST_ORDER_SVBW=$N_RUN" --arm "Q74_OCSVBW=$O_RUN" --output "$FEAS_AUDIT"
python tools/audit_v48_74_interaction_response.py --reference "$T68_RUN" --p74 "$N_RUN" --q74 "$O_RUN" --output "$RESPONSE_AUDIT"
python tools/audit_v48_74_truth_strata.py --arm "B_native=$B_RUN" --arm "P66_OCACRW=$P66_RUN" --arm "T68_FIDELITY=$T68_RUN" --arm "P74_FIRST_ORDER_SVBW=$N_RUN" --arm "Q74_OCSVBW=$O_RUN" --output "$TRUTH_AUDIT"
python tools/compare_v48_74_svbw.py --feasibility-audit "$FEAS_AUDIT" --response-audit "$RESPONSE_AUDIT" --truth-strata "$TRUTH_AUDIT" --v73-complete "$V73_COMPLETE" --v73-comparison "$V73_COMPARE" --output "$COMPARE"
python tools/check_v48_74_pipeline_complete.py --reference-contract "$REF_AUDIT" --v73-complete "$V73_COMPLETE" --v73-comparison "$V73_COMPARE" --runtime-contract "$RUNTIME_AUDIT" --anchor-run "$N_RUN" --main-run "$O_RUN" --feasibility-audit "$FEAS_AUDIT" --response-audit "$RESPONSE_AUDIT" --truth-strata "$TRUTH_AUDIT" --comparison "$COMPARE" --output "$PIPELINE_COMPLETE"
cd "$BASE_OUT"
for run in "$N_RUN" "$O_RUN"; do b="$(basename "$run")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"; done
zip -qj OC-RAP-v48.74-OC-SVBW-audits.zip "$REF_AUDIT" "$RUNTIME_AUDIT" "$FEAS_AUDIT" "$RESPONSE_AUDIT" "$TRUTH_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V73_COMPLETE" "$V73_COMPARE" "$N_RUN/V48_74_VARIANT_ISOLATION.json" "$N_RUN/V48_74_FACTOR_CONTRACT.json" "$O_RUN/V48_74_VARIANT_ISOLATION.json" "$O_RUN/V48_74_FACTOR_CONTRACT.json" "$N_RUN/candidates/balanced/V48_74_STAGE_I_STATE_ISOLATION.json" "$N_RUN/candidates/precision/V48_74_STAGE_I_STATE_ISOLATION.json" "$O_RUN/candidates/balanced/V48_74_STAGE_I_STATE_ISOLATION.json" "$O_RUN/candidates/precision/V48_74_STAGE_I_STATE_ISOLATION.json"
echo "v48.74 complete. Upload $(basename "$O_RUN").zip + OC-RAP-v48.74-OC-SVBW-audits.zip; $(basename "$N_RUN").zip is required for clean P74-vs-Q74 attribution."
