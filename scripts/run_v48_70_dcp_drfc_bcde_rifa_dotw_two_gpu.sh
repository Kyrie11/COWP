#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
REFERENCE_A="${V4870_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
B_RUN="${V4870_NATIVE_B:-$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_native_B}"
F_RUN="${V4870_ERWF_F:-$BASE_OUT/ocrap_v48_61_dcp_drfc_bcde_rifa_erwf_main}"
P66_RUN="${V4870_P66:-$BASE_OUT/ocrap_v48_66_dcp_drfc_bcde_rifa_acrw_main}"
Q67_RUN="${V4870_Q67:-$BASE_OUT/ocrap_v48_67_dcp_drfc_bcde_rifa_pbrw_control}"
T68_RUN="${V4870_T68:-$BASE_OUT/ocrap_v48_68_dcp_drfc_bcde_rifa_rtrw_fidelity}"
D69_RUN="${V4870_D69:-$BASE_OUT/ocrap_v48_69_dcp_drfc_bcde_rifa_dtrw_main}"
E_RUN="$BASE_OUT/ocrap_v48_70_dcp_drfc_bcde_rifa_dotw_occsoft"
G_RUN="$BASE_OUT/ocrap_v48_70_dcp_drfc_bcde_rifa_dotw_main"
V69_COMPLETE="${V4870_V69_COMPLETE:-$BASE_OUT/OC-RAP-v48.69-PIPELINE_COMPLETE.json}"
V69_COMPARE="${V4870_V69_COMPARE:-$BASE_OUT/OC-RAP-v48.69-DCP-DRFC-BCDE-RIFA-OC-DTRW-comparison.json}"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.70-reference-reuse-contract.json"
RUNTIME_AUDIT="$BASE_OUT/OC-RAP-v48.70-runtime-code-contract.json"
FEAS_AUDIT="$BASE_OUT/OC-RAP-v48.70-OC-DOTW-feasibility-role-audit.json"
TRUST_AUDIT="$BASE_OUT/OC-RAP-v48.70-OC-DOTW-soft-occupancy-trust-audit.json"
TRUTH_AUDIT="$BASE_OUT/OC-RAP-v48.70-OC-DOTW-truth-floor-strata-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.70-DCP-DRFC-BCDE-RIFA-OC-DOTW-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.70-PIPELINE_COMPLETE.json"
export PERSISTENT_TENSOR_CACHE="${V4870_PERSISTENT_TENSOR_CACHE:-true}"
export PERSISTENT_TENSOR_CACHE_DIR="${V4870_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4870_dotw}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"
rm -rf "$E_RUN" "$G_RUN"
rm -f "$REF_AUDIT" "$RUNTIME_AUDIT" "$FEAS_AUDIT" "$TRUST_AUDIT" "$TRUTH_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$G_RUN.zip" "$BASE_OUT/OC-RAP-v48.70-OC-DOTW-audits.zip"

# V48.70 OC-DOTW is preregistered from the attribution-ready V48.69 STOP.
# V48.69 falsified observation demand as a sufficient explanation for a large
# actuator projection: D relaxed safe-positive support but produced zero new
# admissions and D-T68 AUC was negative in all eight cells.  At the same time,
# current positive-certificate rows show deterministic CV clearance can rank
# teacher-infeasible witnesses as *safer* than teacher-feasible ones.
#
# V48.68 already falsified hard min(CV, current-acceleration) occupancy.  Hence
# V48.70 does NOT re-enable that hard barrier.  It adds only a strictly-positive
# epistemic trust multiplier from CV-vs-bounded-CA disagreement while retaining
# CV as the signed physical certificate.  The 2x2 is:
#   T68 historical: demand OFF, soft occupancy OFF
#   D69 historical: demand ON,  soft occupancy OFF
#   E70_OCCSOFT:    demand OFF, soft occupancy ON
#   G70/Main:       demand ON,  soft occupancy ON
# Boundary transport remains OFF by changelog until witness trust itself GOes.

python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python tools/check_v48_70_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME_AUDIT"
python - "$V69_COMPLETE" "$V69_COMPARE" "$REFERENCE_A" "$B_RUN" "$F_RUN" "$P66_RUN" "$Q67_RUN" "$T68_RUN" "$D69_RUN" <<'PY2'
import json,pathlib,sys
complete=pathlib.Path(sys.argv[1]); compare=pathlib.Path(sys.argv[2]); runs=[pathlib.Path(x) for x in sys.argv[3:]]
if not complete.is_file(): raise SystemExit(f'missing V48.69 prerequisite sentinel: {complete}')
meta=json.loads(complete.read_text())
if not (meta.get('valid') and meta.get('attribution_ready') and meta.get('engineering_version')=='v48.69.1-OC-DTRW-ENGFIX' and not meta.get('test_roots_read')):
    raise SystemExit(f'V48.69 prerequisite is not attribution-ready: {meta}')
if not compare.is_file(): raise SystemExit(f'missing V48.69 comparison: {compare}')
pr=(json.loads(compare.read_text()).get('preregistered_decision') or {})
if not (pr.get('status')=='STOP' and pr.get('demand_tempering_mechanism_gate') is False):
    raise SystemExit(f'V48.69 branch contract mismatch: {pr}')
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
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing v48.70 protocol root: $d" >&2; exit 30; }
done

# Exact frozen V48.56-A Stage-I architecture / relative-evidence contract.
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

train_dotw_arm(){
  local run="$1"; local arm="$2"; local demand="$3"
  mkdir -p "$run/candidates" "$run/logs"
  train_one(){
    local v="$1"; local gpu="$2"; local src="$REFERENCE_A/candidates/$v"; local dst="$run/candidates/$v"
    mkdir -p "$dst"
    if [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]]; then set -a; source "$src/FACTOR_SUPPORT_CONTRACT.env"; set +a; else export EVIDENCE_COMPONENT_RELIABILITY="1,1,1,0,0"; fi
    RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" INIT_CKPT="$src/model_v48_trac_sr/best.pt" \
    TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" \
    EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_semantic_witness_gain STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_semantic_witness_gain \
    ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=false ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=false ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION=false ABSOLUTE_COMMON_WITNESS_CORRECTION=false ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION=false \
    ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT=true SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=false \
    SEMANTIC_WITNESS_ROUTE_ALIGNMENT=true SEMANTIC_WITNESS_REENTRY_ALIGNMENT=true SEMANTIC_WITNESS_CONTROL_PROJECTION=true SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false \
    SEMANTIC_WITNESS_PROJECTION_FIDELITY=true SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY="$demand" SEMANTIC_WITNESS_ROBUST_OCCUPANCY=false SEMANTIC_WITNESS_SOFT_OCCUPANCY_DISAGREEMENT=true ABSOLUTE_FEASIBILITY_WEIGHT=1.0 \
    GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false BEST_METRIC=direct_absolute_feasibility_bce BEST_METRIC_MIN_DELTA=0.00001 \
    EVIDENCE_ADAPT_EPOCHS="${V4870_DOTW_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4870_DOTW_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4870_DOTW_LR:-0.001}" \
    MAX_EVIDENCE_CALIBRATOR_PARAMS=2 OCRAP_ALGORITHM_VERSION="v48.70-DCP-DRFC-BCDE-RIFA-OC-DOTW-${arm}" \
    bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$run/logs/dotw_${arm}_${v}.log" 2>&1
    grep -qx 'ABSOLUTE_FEASIBILITY_MODE=learned' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'SELECTION_SEMANTICS=rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank' "$dst/POLICY_CONTRACT.env" || exit 30
    python tools/check_v48_70_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --demand "$demand" --output "$dst/V48_70_STAGE_I_STATE_ISOLATION.json"
  }
  set +e
  train_one balanced "$GPU0" & p0=$!; train_one precision "$GPU1" & p1=$!
  wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
  [[ "$r0" == 0 && "$r1" == 0 ]] || { echo "v48.70 $arm training failed balanced=$r0 precision=$r1" >&2; exit 30; }
  python tools/check_v48_70_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --dotw-run "$run" --demand "$demand" --output "$run/V48_70_VARIANT_ISOLATION.json"
  python - "$run/V48_70_FACTOR_CONTRACT.json" "$arm" "$demand" <<'PY2'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);arm=sys.argv[2];demand=sys.argv[3].lower()=='true'
d={'event':'v48_70_factor_contract','version':'v48.70-DCP-DRFC-BCDE-RIFA-OC-DOTW','arm':arm,'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'Observation-Consistent Demand-Occupancy-Tempered Recovery Witness (OC-DOTW)',
'source_intervention':'retain actuator-projected recovery, route+reentry, active-set alignment and projection-fidelity trust; use CV-vs-bounded observed-acceleration disagreement only as a soft epistemic multiplier on common support. Historical CV remains the signed physical certificate; v48.68 hard occupancy min and v48.67 boundary transport stay OFF.',
'semantic_witness_feature_schema':6,'semantic_witness_feature_source':'demand_occupancy_tempered_projected_recovery_witness','active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,'route_alignment':True,'reentry_alignment':True,'control_projection':True,'boundary_transport':False,'projection_fidelity':True,'demand_normalized_fidelity':demand,'robust_occupancy':False,'soft_occupancy_disagreement':True,
'correction_locus':'candidate-global same-option common support before native OC-MERO aggregation','positive_logic':'projection-fidelity trust, optional v48.69 demand tempering, then w_occ=1/(1+delta_occ) where delta_occ=max ReLU(clear_CV-clear_CA)/distance_scale; all multipliers are strictly positive and cannot change certificate sign/set','negative_logic':'frozen universal-failure correction; no per-option negative veto','physical_composition':'non-compensatory historical CV clearance / legacy stopping / observation-active stability / route / persistent re-entry; actuator feasibility enforced by construction','recovery_controller':'v48.67 magnitude/rate/jerk projected deterministic recovery; raw desired-command violation is soft trust only','agent_prediction':'signed certificate remains v48.67 CV. Bounded current-observed-acceleration continuation is computed only as disagreement evidence with acceleration held for existing prefix_horizon_s; never hard-conjoined','boundary_transport_rule':'OFF until trust mechanism GO','trainable_state':'direct_absolute_semantic_witness_gain[2]','trainable_parameters':2,'initialization':'all zeros; execution-exact native B at epoch 0','gain_constraint':'elementwise [0,2]','target':'1[R_dep_star(candidate) >= 0]','scope':'Near+Contact adaptation-train; one shared regime-agnostic mechanism','threshold':0.5,'threshold_search':False,'afe_head':False,'proposal_top_k':5,'proposal_expansion':False,'root_retraining':False,'margin_head_retraining':False,'relative_score_intervention':False,'teacher_margin_distillation':False,'teacher_future_input':False,'regime_id_input':False,'strategy_regime_conditioning':False,'teacher_semantics_changed':False,'dataset_reconstruction':False,'test_roots_read':False,'created_unix':time.time()}
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY2
}

train_dotw_arm "$E_RUN" E70_OCCSOFT false
train_dotw_arm "$G_RUN" G70_Main_OCDOTW true

run_calibration(){
 local run="$1"; local tag="$2"; set +e
 OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.70-${tag}-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.70.0-OC-DOTW-${tag}" \
 CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
 ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 \
 bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_70_${tag}_certificate_controller.log" 2>&1
 rc=$?;set -e;case "$rc" in 0|20) echo "$tag calibration valid evidence RC=$rc";;*) echo "$tag calibration engineering failure RC=$rc" >&2;return 30;;esac
}
run_calibration "$E_RUN" E70_OCCSOFT
run_calibration "$G_RUN" G70_Main_OCDOTW

python tools/audit_v48_70_feasibility_role.py --arm "B_native=$B_RUN" --arm "F_ERWF=$F_RUN" --arm "P66_OCACRW=$P66_RUN" --arm "Q67_CTRLPROJ=$Q67_RUN" --arm "T68_FIDELITY=$T68_RUN" --arm "D69_DTRW=$D69_RUN" --arm "E70_OCCSOFT=$E_RUN" --arm "G70_OCDOTW=$G_RUN" --output "$FEAS_AUDIT"
python tools/audit_v48_70_soft_occupancy_trust.py --t68 "$T68_RUN" --d69 "$D69_RUN" --e70 "$E_RUN" --g70 "$G_RUN" --output "$TRUST_AUDIT"
python tools/audit_v48_70_truth_strata.py --arm "B_native=$B_RUN" --arm "P66_OCACRW=$P66_RUN" --arm "T68_FIDELITY=$T68_RUN" --arm "D69_DTRW=$D69_RUN" --arm "E70_OCCSOFT=$E_RUN" --arm "G70_OCDOTW=$G_RUN" --output "$TRUTH_AUDIT"
python tools/compare_v48_70_dotw.py --feasibility-audit "$FEAS_AUDIT" --trust-audit "$TRUST_AUDIT" --truth-strata "$TRUTH_AUDIT" --v69-complete "$V69_COMPLETE" --v69-comparison "$V69_COMPARE" --output "$COMPARE"
python tools/check_v48_70_pipeline_complete.py --reference-contract "$REF_AUDIT" --v69-complete "$V69_COMPLETE" --v69-comparison "$V69_COMPARE" --runtime-contract "$RUNTIME_AUDIT" --occsoft-run "$E_RUN" --main-run "$G_RUN" --feasibility-audit "$FEAS_AUDIT" --trust-audit "$TRUST_AUDIT" --truth-strata "$TRUTH_AUDIT" --comparison "$COMPARE" --output "$PIPELINE_COMPLETE"

cd "$BASE_OUT";b="$(basename "$G_RUN")";rm -f "$b.zip";zip -qr "$b.zip" "$b"
zip -qj OC-RAP-v48.70-OC-DOTW-audits.zip "$REF_AUDIT" "$RUNTIME_AUDIT" "$FEAS_AUDIT" "$TRUST_AUDIT" "$TRUTH_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V69_COMPLETE" "$V69_COMPARE" \
  "$E_RUN/V48_70_VARIANT_ISOLATION.json" "$E_RUN/V48_70_FACTOR_CONTRACT.json" "$G_RUN/V48_70_VARIANT_ISOLATION.json" "$G_RUN/V48_70_FACTOR_CONTRACT.json" \
  "$E_RUN/candidates/balanced/V48_70_STAGE_I_STATE_ISOLATION.json" "$E_RUN/candidates/precision/V48_70_STAGE_I_STATE_ISOLATION.json" \
  "$G_RUN/candidates/balanced/V48_70_STAGE_I_STATE_ISOLATION.json" "$G_RUN/candidates/precision/V48_70_STAGE_I_STATE_ISOLATION.json"
echo "v48.70 complete. Upload $b.zip + OC-RAP-v48.70-OC-DOTW-audits.zip"
