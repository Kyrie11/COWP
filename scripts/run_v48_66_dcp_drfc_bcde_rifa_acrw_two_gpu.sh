#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
REFERENCE_A="${V4866_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
B_RUN="${V4866_NATIVE_B:-$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_native_B}"
F_RUN="${V4866_ERWF_F:-$BASE_OUT/ocrap_v48_61_dcp_drfc_bcde_rifa_erwf_main}"
I_RUN="${V4866_ACTIVE_I:-$BASE_OUT/ocrap_v48_64_dcp_drfc_bcde_rifa_sarw_active}"
M65_RUN="${V4866_M65:-$BASE_OUT/ocrap_v48_65_dcp_drfc_bcde_rifa_clrw_main}"
N_RUN="$BASE_OUT/ocrap_v48_66_dcp_drfc_bcde_rifa_acrw_route"
O_RUN="$BASE_OUT/ocrap_v48_66_dcp_drfc_bcde_rifa_acrw_reentry"
P_RUN="$BASE_OUT/ocrap_v48_66_dcp_drfc_bcde_rifa_acrw_main"
V65_COMPLETE="${V4866_V65_COMPLETE:-$BASE_OUT/OC-RAP-v48.65-PIPELINE_COMPLETE.json}"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.66-reference-reuse-contract.json"
FEAS_AUDIT="$BASE_OUT/OC-RAP-v48.66-OC-ACRW-feasibility-role-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.66-DCP-DRFC-BCDE-RIFA-OC-ACRW-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.66-PIPELINE_COMPLETE.json"
export PERSISTENT_TENSOR_CACHE="${V4866_PERSISTENT_TENSOR_CACHE:-true}"
export PERSISTENT_TENSOR_CACHE_DIR="${V4866_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4866_acrw}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"
rm -rf "$N_RUN" "$O_RUN" "$P_RUN"
rm -f "$REF_AUDIT" "$FEAS_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$P_RUN.zip" "$BASE_OUT/OC-RAP-v48.66-OC-ACRW-audits.zip"

# V48.66 is the preregistered next branch after the engineering-valid V48.65
# scientific STOP.  V48.65 directly tested the class-local correction-locus
# hypothesis and regressed Near source AUC; its read-only teacher audit also
# found zero feasibility-sign dependence on class-local-vs-global option choice.
# We therefore revert the learned correction locus to the v48.64 candidate-
# global same-option common support, retain the validated active-set repair,
# keep legacy stopping, and test only two missing observation-certifiable
# active-constraint families:
#   N_ROUTE: executable-route consistency only
#   O_REENTRY: persistent post-contact re-entry only
#   P/Main: both (route x re-entry 2x2 with historical I_ACTIVESET baseline)
# No regime input, threshold search, option-library change, Stage-I retraining,
# teacher future/component margin, or relative-ranker intervention is allowed.

python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python - "$V65_COMPLETE" "$REFERENCE_A" "$B_RUN" "$F_RUN" "$I_RUN" "$M65_RUN" <<'PY2'
import json,pathlib,sys
p,*runs=map(pathlib.Path,sys.argv[1:])
if not p.is_file(): raise SystemExit(f'missing V48.65 prerequisite sentinel: {p}')
meta=json.loads(p.read_text())
if not (meta.get('valid') and meta.get('attribution_ready') and meta.get('engineering_version')=='v48.65.0-OC-CLRW' and not meta.get('test_roots_read')):
    raise SystemExit(f'V48.65 prerequisite is not attribution-ready: {meta}')
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
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing v48.66 protocol root: $d" >&2; exit 30; }
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

train_acrw_arm(){
  local run="$1"; local arm="$2"; local route="$3"; local reentry="$4"
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
    SEMANTIC_WITNESS_ROUTE_ALIGNMENT="$route" SEMANTIC_WITNESS_REENTRY_ALIGNMENT="$reentry" ABSOLUTE_FEASIBILITY_WEIGHT=1.0 \
    GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false BEST_METRIC=direct_absolute_feasibility_bce BEST_METRIC_MIN_DELTA=0.00001 \
    EVIDENCE_ADAPT_EPOCHS="${V4866_ACRW_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4866_ACRW_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4866_ACRW_LR:-0.001}" \
    MAX_EVIDENCE_CALIBRATOR_PARAMS=2 OCRAP_ALGORITHM_VERSION="v48.66-DCP-DRFC-BCDE-RIFA-OC-ACRW-${arm}" \
    bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$run/logs/acrw_${arm}_${v}.log" 2>&1
    grep -qx 'ABSOLUTE_FEASIBILITY_MODE=learned' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'SELECTION_SEMANTICS=rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank' "$dst/POLICY_CONTRACT.env" || exit 30
    python tools/check_v48_66_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --route "$route" --reentry "$reentry" --output "$dst/V48_66_STAGE_I_STATE_ISOLATION.json"
  }
  set +e
  train_one balanced "$GPU0" & p0=$!; train_one precision "$GPU1" & p1=$!
  wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
  [[ "$r0" == 0 && "$r1" == 0 ]] || { echo "v48.66 $arm training failed balanced=$r0 precision=$r1" >&2; exit 30; }
  python tools/check_v48_66_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --acrw-run "$run" --route "$route" --reentry "$reentry" --output "$run/V48_66_VARIANT_ISOLATION.json"
  python - "$run/V48_66_FACTOR_CONTRACT.json" "$arm" "$route" "$reentry" <<'PY2'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);arm=sys.argv[2];route=sys.argv[3].lower()=='true';reentry=sys.argv[4].lower()=='true'
d={'event':'v48_66_factor_contract','version':'v48.66-DCP-DRFC-BCDE-RIFA-OC-ACRW','arm':arm,'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'Observation-Consistent Active-Constraint Recovery Witness (OC-ACRW)',
'source_intervention':'retain v48.64 candidate-global common executable recovery witness and active-set alignment; add only observation-certifiable route consistency and/or persistent post-contact re-entry as non-compensatory signed barriers',
'semantic_witness_feature_schema':2,'semantic_witness_feature_source':'active_constraint_coverage_common_executable_recovery_witness','active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,'route_alignment':route,'reentry_alignment':reentry,
'correction_locus':'candidate-global same-option common support before native OC-MERO aggregation','positive_logic':'exists supported option whose enabled observation-certifiable active constraints all have positive signed viability','negative_logic':'forall supported options fail at least one enabled active constraint','physical_composition':'non-compensatory minimum of clearance / legacy stopping / control / observation-active stability plus enabled route and persistent-reentry barriers','recovery_controller':'existing deterministic rollout_recovery_controller','agent_prediction':'current observation constant velocity only',
'trainable_state':'direct_absolute_semantic_witness_gain[2]','trainable_parameters':2,'initialization':'all zeros; execution-exact native B source at epoch 0','gain_constraint':'elementwise [0,2]','target':'1[R_dep_star(candidate) >= 0]','scope':'Near+Contact adaptation-train; one shared regime-agnostic execution mechanism','threshold':0.5,'threshold_search':False,
'afe_head':False,'proposal_top_k':5,'proposal_expansion':False,'root_retraining':False,'margin_head_retraining':False,'relative_score_intervention':False,'teacher_margin_distillation':False,'teacher_future_input':False,'regime_id_input':False,'strategy_regime_conditioning':False,'teacher_semantics_changed':False,'dataset_reconstruction':False,'test_roots_read':False,'created_unix':time.time()}
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY2
}

train_acrw_arm "$N_RUN" N_ROUTE true false
train_acrw_arm "$O_RUN" O_REENTRY false true
train_acrw_arm "$P_RUN" P_Main_OCACRW true true

run_calibration(){
  local run="$1"; local tag="$2"
  set +e
  OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.66-${tag}-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.66.0-OC-ACRW-${tag}" \
  CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
  ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 \
  bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_66_${tag}_certificate_controller.log" 2>&1
  rc=$?; set -e; case "$rc" in 0|20) echo "$tag calibration valid evidence RC=$rc";; *) echo "$tag calibration engineering failure RC=$rc" >&2; return 30;; esac
}
run_calibration "$N_RUN" N_ROUTE
run_calibration "$O_RUN" O_REENTRY
run_calibration "$P_RUN" P_Main_OCACRW

python tools/audit_v48_66_feasibility_role.py \
  --arm "B_native=$B_RUN" --arm "F_ERWF=$F_RUN" --arm "I_ACTIVESET=$I_RUN" --arm "M65_OCCLRW=$M65_RUN" \
  --arm "N_ROUTE=$N_RUN" --arm "O_REENTRY=$O_RUN" --arm "P_Main_OCACRW=$P_RUN" --output "$FEAS_AUDIT"
python tools/compare_v48_66_acrw.py --b "$B_RUN" --f "$F_RUN" --i "$I_RUN" --m65 "$M65_RUN" --n "$N_RUN" --o "$O_RUN" --p "$P_RUN" --feasibility-audit "$FEAS_AUDIT" --v65-complete "$V65_COMPLETE" --output "$COMPARE"
python tools/check_v48_66_pipeline_complete.py --reference-contract "$REF_AUDIT" --v65-complete "$V65_COMPLETE" --route-run "$N_RUN" --reentry-run "$O_RUN" --main-run "$P_RUN" --feasibility-audit "$FEAS_AUDIT" --comparison "$COMPARE" --output "$PIPELINE_COMPLETE"

cd "$BASE_OUT"; b="$(basename "$P_RUN")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"
zip -qj OC-RAP-v48.66-OC-ACRW-audits.zip "$REF_AUDIT" "$FEAS_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V65_COMPLETE" \
  "$N_RUN/V48_66_VARIANT_ISOLATION.json" "$N_RUN/V48_66_FACTOR_CONTRACT.json" "$O_RUN/V48_66_VARIANT_ISOLATION.json" "$O_RUN/V48_66_FACTOR_CONTRACT.json" "$P_RUN/V48_66_VARIANT_ISOLATION.json" "$P_RUN/V48_66_FACTOR_CONTRACT.json" \
  "$N_RUN/candidates/balanced/V48_66_STAGE_I_STATE_ISOLATION.json" "$N_RUN/candidates/precision/V48_66_STAGE_I_STATE_ISOLATION.json" \
  "$O_RUN/candidates/balanced/V48_66_STAGE_I_STATE_ISOLATION.json" "$O_RUN/candidates/precision/V48_66_STAGE_I_STATE_ISOLATION.json" \
  "$P_RUN/candidates/balanced/V48_66_STAGE_I_STATE_ISOLATION.json" "$P_RUN/candidates/precision/V48_66_STAGE_I_STATE_ISOLATION.json"
echo "v48.66 complete. Upload $b.zip + OC-RAP-v48.66-OC-ACRW-audits.zip"
