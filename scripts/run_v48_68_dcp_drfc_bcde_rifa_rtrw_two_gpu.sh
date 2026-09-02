#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
REFERENCE_A="${V4868_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
B_RUN="${V4868_NATIVE_B:-$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_native_B}"
F_RUN="${V4868_ERWF_F:-$BASE_OUT/ocrap_v48_61_dcp_drfc_bcde_rifa_erwf_main}"
P66_RUN="${V4868_P66:-$BASE_OUT/ocrap_v48_66_dcp_drfc_bcde_rifa_acrw_main}"
Q67_RUN="${V4868_Q67:-$BASE_OUT/ocrap_v48_67_dcp_drfc_bcde_rifa_pbrw_control}"
T_RUN="$BASE_OUT/ocrap_v48_68_dcp_drfc_bcde_rifa_rtrw_fidelity"
U_RUN="$BASE_OUT/ocrap_v48_68_dcp_drfc_bcde_rifa_rtrw_occupancy"
V_RUN="$BASE_OUT/ocrap_v48_68_dcp_drfc_bcde_rifa_rtrw_main"
V67_COMPLETE="${V4868_V67_COMPLETE:-$BASE_OUT/OC-RAP-v48.67-PIPELINE_COMPLETE.json}"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.68-reference-reuse-contract.json"
FEAS_AUDIT="$BASE_OUT/OC-RAP-v48.68-OC-RTRW-feasibility-role-audit.json"
TRUST_AUDIT="$BASE_OUT/OC-RAP-v48.68-OC-RTRW-robust-trust-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.68-DCP-DRFC-BCDE-RIFA-OC-RTRW-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.68-PIPELINE_COMPLETE.json"
export PERSISTENT_TENSOR_CACHE="${V4868_PERSISTENT_TENSOR_CACHE:-true}"
export PERSISTENT_TENSOR_CACHE_DIR="${V4868_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4868_rtrw}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"
rm -rf "$T_RUN" "$U_RUN" "$V_RUN"
rm -f "$REF_AUDIT" "$FEAS_AUDIT" "$TRUST_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V_RUN.zip" "$BASE_OUT/OC-RAP-v48.68-OC-RTRW-audits.zip"

# V48.68 OC-RTRW is preregistered from the attribution-ready V48.67.1 STOP.
# v48.67 proved that actuator projection removes the artificial control veto,
# but the newly recovered witnesses are mostly false and CV-only clearance is
# often optimistic.  Boundary transport also amplifies false admission, so it
# is frozen OFF.  We test exactly two trust factors on the same projected policy:
#   T_FIDELITY: sign-preserving soft trust from raw projection severity
#   U_OCCUPANCY: observation-only CV/current-acceleration occupancy envelope
#   V/Main: both (2x2 with historical Q67_CTRLPROJ as baseline)
# No regime input, threshold search, teacher future/component margin, option
# library expansion, Stage-I retraining, class-local correction, boundary
# transport or relative-ranker intervention is permitted.

python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python - "$V67_COMPLETE" "$REFERENCE_A" "$B_RUN" "$F_RUN" "$P66_RUN" "$Q67_RUN" <<'PY2'
import json,pathlib,sys
p,*runs=map(pathlib.Path,sys.argv[1:])
if not p.is_file(): raise SystemExit(f'missing V48.67 prerequisite sentinel: {p}')
meta=json.loads(p.read_text())
if not (meta.get('valid') and meta.get('attribution_ready') and meta.get('engineering_version')=='v48.67.1-OC-PBRW-ENGFIX' and not meta.get('test_roots_read')):
    raise SystemExit(f'V48.67 prerequisite is not attribution-ready: {meta}')
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
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing v48.68 protocol root: $d" >&2; exit 30; }
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

train_rtrw_arm(){
  local run="$1"; local arm="$2"; local fidelity="$3"; local occupancy="$4"
  mkdir -p "$run/candidates" "$run/logs"
  train_one(){
    local v="$1"; local gpu="$2"; local src="$REFERENCE_A/candidates/$v"; local dst="$run/candidates/$v"
    mkdir -p "$dst"
    if [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]]; then set -a; source "$src/FACTOR_SUPPORT_CONTRACT.env"; set +a; else export EVIDENCE_COMPONENT_RELIABILITY="1,1,1,0,0"; fi
    RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" INIT_CKPT="$src/model_v48_trac_sr/best.pt"     TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"     EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_semantic_witness_gain STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_semantic_witness_gain     ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=false ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=false ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION=false ABSOLUTE_COMMON_WITNESS_CORRECTION=false ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION=false     ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT=true SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=false     SEMANTIC_WITNESS_ROUTE_ALIGNMENT=true SEMANTIC_WITNESS_REENTRY_ALIGNMENT=true SEMANTIC_WITNESS_CONTROL_PROJECTION=true SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false SEMANTIC_WITNESS_PROJECTION_FIDELITY="$fidelity" SEMANTIC_WITNESS_ROBUST_OCCUPANCY="$occupancy" ABSOLUTE_FEASIBILITY_WEIGHT=1.0     GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false BEST_METRIC=direct_absolute_feasibility_bce BEST_METRIC_MIN_DELTA=0.00001     EVIDENCE_ADAPT_EPOCHS="${V4868_RTRW_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4868_RTRW_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4868_RTRW_LR:-0.001}"     MAX_EVIDENCE_CALIBRATOR_PARAMS=2 OCRAP_ALGORITHM_VERSION="v48.68-DCP-DRFC-BCDE-RIFA-OC-RTRW-${arm}"     bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$run/logs/rtrw_${arm}_${v}.log" 2>&1
    grep -qx 'ABSOLUTE_FEASIBILITY_MODE=learned' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'SELECTION_SEMANTICS=rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank' "$dst/POLICY_CONTRACT.env" || exit 30
    python tools/check_v48_68_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --fidelity "$fidelity" --occupancy "$occupancy" --output "$dst/V48_68_STAGE_I_STATE_ISOLATION.json"
  }
  set +e
  train_one balanced "$GPU0" & p0=$!; train_one precision "$GPU1" & p1=$!
  wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
  [[ "$r0" == 0 && "$r1" == 0 ]] || { echo "v48.68 $arm training failed balanced=$r0 precision=$r1" >&2; exit 30; }
  python tools/check_v48_68_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --rtrw-run "$run" --fidelity "$fidelity" --occupancy "$occupancy" --output "$run/V48_68_VARIANT_ISOLATION.json"
  python - "$run/V48_68_FACTOR_CONTRACT.json" "$arm" "$fidelity" "$occupancy" <<'PY2'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);arm=sys.argv[2];fid=sys.argv[3].lower()=='true';occ=sys.argv[4].lower()=='true'
d={'event':'v48_68_factor_contract','version':'v48.68-DCP-DRFC-BCDE-RIFA-OC-RTRW','arm':arm,'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'Observation-Consistent Robust-Trust Recovery Witness (OC-RTRW)',
'source_intervention':'retain v48.67 actuator-projected recovery, v48.66 route+persistent-reentry trust constraints, and historical additive transport; independently test soft projection-fidelity trust and an observation-only CV/current-acceleration occupancy envelope',
'semantic_witness_feature_schema':4,'semantic_witness_feature_source':'robust_trust_projected_recovery_witness','active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,'route_alignment':True,'reentry_alignment':True,'control_projection':True,'boundary_transport':False,'projection_fidelity':fid,'robust_occupancy':occ,
'correction_locus':'candidate-global same-option common support before native OC-MERO aggregation','positive_logic':'projected actuator-feasible witness; optional strictly-positive projection-fidelity multiplier discounts rescue according to raw desired-command violation','negative_logic':'frozen universal-failure correction; no per-option negative veto','physical_composition':'non-compensatory clearance / legacy stopping / observation-active stability / route / persistent re-entry; actuator feasibility enforced by construction','recovery_controller':'same deterministic recovery modes and params with magnitude/rate/jerk projection before state integration; optional raw-controller side computation is diagnostic trust only','agent_prediction':'observation-only CV; robust arm uses min-clearance envelope over CV and current-observed-acceleration continuation with acceleration held for existing prefix_horizon_s','boundary_transport_rule':'OFF; v48.67 bounded boundary transport is deliberately not carried into Main until witness trust recovers','trainable_state':'direct_absolute_semantic_witness_gain[2]','trainable_parameters':2,'initialization':'all zeros; execution-exact native B at epoch 0','gain_constraint':'elementwise [0,2]','target':'1[R_dep_star(candidate) >= 0]','scope':'Near+Contact adaptation-train; one shared regime-agnostic mechanism','threshold':0.5,'threshold_search':False,'afe_head':False,'proposal_top_k':5,'proposal_expansion':False,'root_retraining':False,'margin_head_retraining':False,'relative_score_intervention':False,'teacher_margin_distillation':False,'teacher_future_input':False,'regime_id_input':False,'strategy_regime_conditioning':False,'teacher_semantics_changed':False,'dataset_reconstruction':False,'test_roots_read':False,'created_unix':time.time()}
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY2
}

train_rtrw_arm "$T_RUN" T_FIDELITY true false
train_rtrw_arm "$U_RUN" U_OCCUPANCY false true
train_rtrw_arm "$V_RUN" V_Main_OCRTRW true true

run_calibration(){
 local run="$1"; local tag="$2"; set +e
 OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.68-${tag}-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.68.0-OC-RTRW-${tag}"  CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT"  ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5  bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_68_${tag}_certificate_controller.log" 2>&1
 rc=$?;set -e;case "$rc" in 0|20) echo "$tag calibration valid evidence RC=$rc";;*) echo "$tag calibration engineering failure RC=$rc" >&2;return 30;;esac
}
run_calibration "$T_RUN" T_FIDELITY
run_calibration "$U_RUN" U_OCCUPANCY
run_calibration "$V_RUN" V_Main_OCRTRW

python tools/audit_v48_68_feasibility_role.py --arm "B_native=$B_RUN" --arm "F_ERWF=$F_RUN" --arm "P66_OCACRW=$P66_RUN" --arm "Q67_CTRLPROJ=$Q67_RUN" --arm "T_FIDELITY=$T_RUN" --arm "U_OCCUPANCY=$U_RUN" --arm "V_Main_OCRTRW=$V_RUN" --output "$FEAS_AUDIT"
python tools/audit_v48_68_robust_trust.py --q67 "$Q67_RUN" --t "$T_RUN" --u "$U_RUN" --v "$V_RUN" --output "$TRUST_AUDIT"
python tools/compare_v48_68_rtrw.py --feasibility-audit "$FEAS_AUDIT" --v67-complete "$V67_COMPLETE" --output "$COMPARE"
python tools/check_v48_68_pipeline_complete.py --reference-contract "$REF_AUDIT" --v67-complete "$V67_COMPLETE" --fidelity-run "$T_RUN" --occupancy-run "$U_RUN" --main-run "$V_RUN" --feasibility-audit "$FEAS_AUDIT" --trust-audit "$TRUST_AUDIT" --comparison "$COMPARE" --output "$PIPELINE_COMPLETE"

cd "$BASE_OUT";b="$(basename "$V_RUN")";rm -f "$b.zip";zip -qr "$b.zip" "$b"
zip -qj OC-RAP-v48.68-OC-RTRW-audits.zip "$REF_AUDIT" "$FEAS_AUDIT" "$TRUST_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V67_COMPLETE" \
  "$T_RUN/V48_68_VARIANT_ISOLATION.json" "$T_RUN/V48_68_FACTOR_CONTRACT.json" "$U_RUN/V48_68_VARIANT_ISOLATION.json" "$U_RUN/V48_68_FACTOR_CONTRACT.json" "$V_RUN/V48_68_VARIANT_ISOLATION.json" "$V_RUN/V48_68_FACTOR_CONTRACT.json" \
  "$T_RUN/candidates/balanced/V48_68_STAGE_I_STATE_ISOLATION.json" "$T_RUN/candidates/precision/V48_68_STAGE_I_STATE_ISOLATION.json" \
  "$U_RUN/candidates/balanced/V48_68_STAGE_I_STATE_ISOLATION.json" "$U_RUN/candidates/precision/V48_68_STAGE_I_STATE_ISOLATION.json" \
  "$V_RUN/candidates/balanced/V48_68_STAGE_I_STATE_ISOLATION.json" "$V_RUN/candidates/precision/V48_68_STAGE_I_STATE_ISOLATION.json"
echo "v48.68 complete. Upload $b.zip + OC-RAP-v48.68-OC-RTRW-audits.zip"
