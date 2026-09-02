#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
REFERENCE_A="${V4865_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
B_RUN="${V4865_NATIVE_B:-$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_native_B}"
F_RUN="${V4865_ERWF_F:-$BASE_OUT/ocrap_v48_61_dcp_drfc_bcde_rifa_erwf_main}"
H_RUN="${V4865_OCQARW_H:-$BASE_OUT/ocrap_v48_63_dcp_drfc_bcde_rifa_ocqarw_main}"
I_RUN="${V4865_ACTIVE_I:-$BASE_OUT/ocrap_v48_64_dcp_drfc_bcde_rifa_sarw_active}"
L_RUN="$BASE_OUT/ocrap_v48_65_dcp_drfc_bcde_rifa_clrw_classlocal"
M_RUN="$BASE_OUT/ocrap_v48_65_dcp_drfc_bcde_rifa_clrw_main"
V64_COMPLETE="${V4865_V64_COMPLETE:-$BASE_OUT/OC-RAP-v48.64-PIPELINE_COMPLETE.json}"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.65-reference-reuse-contract.json"
TEACHER_AUDIT="$BASE_OUT/OC-RAP-v48.65-teacher-certificate-semantics-audit.json"
FEAS_AUDIT="$BASE_OUT/OC-RAP-v48.65-OC-CLRW-feasibility-role-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.65-DCP-DRFC-BCDE-RIFA-OC-CLRW-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.65-PIPELINE_COMPLETE.json"
export PERSISTENT_TENSOR_CACHE="${V4865_PERSISTENT_TENSOR_CACHE:-true}"
export PERSISTENT_TENSOR_CACHE_DIR="${V4865_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4865_clrw}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"
rm -rf "$L_RUN" "$M_RUN"
rm -f "$REF_AUDIT" "$TEACHER_AUDIT" "$FEAS_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$M_RUN.zip" "$BASE_OUT/OC-RAP-v48.65-OC-CLRW-audits.zip"

# V48.65 is a preregistered 2x2 correction-locus experiment after the
# engineering-valid V48.64.1 scientific STOP.  V48.64 active-set alignment
# restored physical positive-witness availability but did not improve the
# feasibility pass/AUC gate; J_PATHSTOP did not supply a robust main effect.
# V48.65 therefore freezes legacy stopping and tests whether the two-gain
# correction must live at OC-MERO q[i,l] (observation-class local) instead of
# being candidate-global and broadcast across roots/options.
# Historical H: classlocal=false, active=false; historical I: false,true.
# New L: classlocal=true, active=false; new M/Main: true,true.

python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python - "$V64_COMPLETE" "$REFERENCE_A" "$B_RUN" "$F_RUN" "$H_RUN" "$I_RUN" <<'PY2'
import json,pathlib,sys
p,*runs=map(pathlib.Path,sys.argv[1:])
if not p.is_file(): raise SystemExit(f'missing V48.64.1 prerequisite sentinel: {p}')
meta=json.loads(p.read_text())
if not (meta.get('valid') and meta.get('attribution_ready') and meta.get('engineering_version')=='v48.64.1-OC-SARW-ENGFIX' and not meta.get('test_roots_read')):
    raise SystemExit(f'V48.64.1 prerequisite is not attribution-ready: {meta}')
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
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing v48.65 protocol root: $d" >&2; exit 30; }
done

# Read-only truth-contract audit: no dataset reconstruction, no test roots.
python tools/audit_v48_65_teacher_certificate_semantics.py \
  --root "$TRAIN_NEAR" --root "$TRAIN_CONTACT" --root "$DEV_NEAR" --root "$DEV_CONTACT" --root "$CERT_NEAR" --root "$CERT_CONTACT" \
  --alpha 0.20 --beta 0.20 --top-m 8 --output "$TEACHER_AUDIT"

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

train_clrw_arm(){
  local run="$1"; local arm="$2"; local active="$3"
  mkdir -p "$run/candidates" "$run/logs"
  train_one(){
    local v="$1"; local gpu="$2"; local src="$REFERENCE_A/candidates/$v"; local dst="$run/candidates/$v"
    mkdir -p "$dst"
    if [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]]; then set -a; source "$src/FACTOR_SUPPORT_CONTRACT.env"; set +a; else export EVIDENCE_COMPONENT_RELIABILITY="1,1,1,0,0"; fi
    RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" INIT_CKPT="$src/model_v48_trac_sr/best.pt" \
    TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" \
    EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_semantic_witness_gain STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_semantic_witness_gain \
    ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=false ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=false ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION=false ABSOLUTE_COMMON_WITNESS_CORRECTION=false ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION=false \
    ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT="$active" SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=true ABSOLUTE_FEASIBILITY_WEIGHT=1.0 \
    GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false BEST_METRIC=direct_absolute_feasibility_bce BEST_METRIC_MIN_DELTA=0.00001 \
    EVIDENCE_ADAPT_EPOCHS="${V4865_CLRW_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4865_CLRW_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4865_CLRW_LR:-0.001}" \
    MAX_EVIDENCE_CALIBRATOR_PARAMS=2 OCRAP_ALGORITHM_VERSION="v48.65-DCP-DRFC-BCDE-RIFA-OC-CLRW-${arm}" \
    bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$run/logs/clrw_${arm}_${v}.log" 2>&1
    grep -qx 'ABSOLUTE_FEASIBILITY_MODE=learned' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'SELECTION_SEMANTICS=rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank' "$dst/POLICY_CONTRACT.env" || exit 30
    python tools/check_v48_65_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --active-set "$active" --classlocal true --path-stop false --output "$dst/V48_65_STAGE_I_STATE_ISOLATION.json"
  }
  set +e
  train_one balanced "$GPU0" & p0=$!; train_one precision "$GPU1" & p1=$!
  wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
  [[ "$r0" == 0 && "$r1" == 0 ]] || { echo "v48.65 $arm training failed balanced=$r0 precision=$r1" >&2; exit 30; }
  python tools/check_v48_65_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --clrw-run "$run" --active-set "$active" --classlocal true --path-stop false --output "$run/V48_65_VARIANT_ISOLATION.json"
  python - "$run/V48_65_FACTOR_CONTRACT.json" "$arm" "$active" <<'PY2'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);arm=sys.argv[2];active=sys.argv[3].lower()=='true'
d={'event':'v48_65_factor_contract','version':'v48.65-DCP-DRFC-BCDE-RIFA-OC-CLRW','arm':arm,'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'Observation-Class-Local Recovery Witness (OC-CLRW)',
'source_intervention':'move the same two-gain semantics correction from candidate-global root-margin broadcast to OC-MERO q[i,l], after compatible-root aggregation and before per-observation-class option maximization',
'semantic_witness_feature_schema':1,'semantic_witness_feature_source':'semantics_aligned_common_executable_recovery_witness','active_set_alignment':active,'path_stop_alignment':False,'classlocal_transport':True,
'correction_locus':'OC-MERO q[i,l] after compatible-root aggregation and before per-observation-class max','positive_logic':'class-local supported positive physical viability','negative_logic':'per-observation-class forall-supported-option failure','physical_composition':'v48.64 non-compensatory clearance/stop/control/stability barrier; active-set factor only; legacy stopping frozen','recovery_controller':'existing deterministic rollout_recovery_controller','agent_prediction':'current observation constant velocity only',
'trainable_state':'direct_absolute_semantic_witness_gain[2]','trainable_parameters':2,'initialization':'all zeros; execution-exact native B source at epoch 0','gain_constraint':'elementwise [0,2]','target':'1[R_dep_star(candidate) >= 0]','scope':'Near+Contact adaptation-train; one shared regime-agnostic execution mechanism','threshold':0.5,'threshold_search':False,
'afe_head':False,'proposal_top_k':5,'proposal_expansion':False,'root_retraining':False,'margin_head_retraining':False,'relative_score_intervention':False,'teacher_margin_distillation':False,'teacher_future_input':False,'regime_id_input':False,'strategy_regime_conditioning':False,'teacher_semantics_changed':False,'dataset_reconstruction':False,'test_roots_read':False,'created_unix':time.time()}
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY2
}

train_clrw_arm "$L_RUN" L_CLASSLOCAL false
train_clrw_arm "$M_RUN" M_Main_OCCLRW true

run_calibration(){
  local run="$1"; local tag="$2"
  set +e
  OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.65-${tag}-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.65.0-OC-CLRW-${tag}" \
  CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
  ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 \
  bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_65_${tag}_certificate_controller.log" 2>&1
  rc=$?; set -e; case "$rc" in 0|20) echo "$tag calibration valid evidence RC=$rc";; *) echo "$tag calibration engineering failure RC=$rc" >&2; return 30;; esac
}
run_calibration "$L_RUN" L_CLASSLOCAL
run_calibration "$M_RUN" M_Main_OCCLRW

python tools/audit_v48_65_feasibility_role.py \
  --arm "B_native=$B_RUN" --arm "F_ERWF=$F_RUN" --arm "H_OCQARW=$H_RUN" --arm "I_ACTIVESET=$I_RUN" --arm "L_CLASSLOCAL=$L_RUN" --arm "M_OCCLRW=$M_RUN" --output "$FEAS_AUDIT"
python tools/compare_v48_65_clrw.py --a "$REFERENCE_A" --b "$B_RUN" --f "$F_RUN" --h "$H_RUN" --i "$I_RUN" --l "$L_RUN" --m "$M_RUN" --feasibility-audit "$FEAS_AUDIT" --teacher-semantics-audit "$TEACHER_AUDIT" --output "$COMPARE"
python tools/check_v48_65_pipeline_complete.py --reference-contract "$REF_AUDIT" --v64-complete "$V64_COMPLETE" --teacher-semantics-audit "$TEACHER_AUDIT" --classlocal-run "$L_RUN" --main-run "$M_RUN" --feasibility-audit "$FEAS_AUDIT" --comparison "$COMPARE" --output "$PIPELINE_COMPLETE"

cd "$BASE_OUT"; b="$(basename "$M_RUN")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"
zip -qj OC-RAP-v48.65-OC-CLRW-audits.zip "$REF_AUDIT" "$TEACHER_AUDIT" "$FEAS_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V64_COMPLETE" \
  "$L_RUN/V48_65_VARIANT_ISOLATION.json" "$L_RUN/V48_65_FACTOR_CONTRACT.json" "$M_RUN/V48_65_VARIANT_ISOLATION.json" "$M_RUN/V48_65_FACTOR_CONTRACT.json" \
  "$L_RUN/candidates/balanced/V48_65_STAGE_I_STATE_ISOLATION.json" "$L_RUN/candidates/precision/V48_65_STAGE_I_STATE_ISOLATION.json" \
  "$M_RUN/candidates/balanced/V48_65_STAGE_I_STATE_ISOLATION.json" "$M_RUN/candidates/precision/V48_65_STAGE_I_STATE_ISOLATION.json"
echo "v48.65 complete. Upload $b.zip + OC-RAP-v48.65-OC-CLRW-audits.zip"
