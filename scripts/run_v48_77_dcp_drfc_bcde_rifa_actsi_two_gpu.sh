#!/usr/bin/env bash
# V48.77 OC-ACTSI: structured absolute-source interface after V48.76 OC-ICSM STOP.
# Keep V48.76 truth/supervision exact; replace only the global 2-gain transport
# with a regime/option-agnostic 6x2 table indexed by the binding active constraint.
export OCRAP_V48_74_SIGNED_VIABILITY=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"

REFERENCE_A="${V4877_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
C75_RUN="${V4877_C75:-$BASE_OUT/ocrap_v48_75_dcp_drfc_bcde_rifa_stca_projection_censored}"
D75_RUN="${V4877_D75:-$BASE_OUT/ocrap_v48_75_dcp_drfc_bcde_rifa_stca_main}"
E76_RUN="${V4877_E76:-$BASE_OUT/ocrap_v48_76_dcp_drfc_bcde_rifa_icsm_projection_margin}"
F76_RUN="${V4877_F76:-$BASE_OUT/ocrap_v48_76_dcp_drfc_bcde_rifa_icsm_main}"
G_RUN="$BASE_OUT/ocrap_v48_77_dcp_drfc_bcde_rifa_actsi_typed_projection"
H_RUN="$BASE_OUT/ocrap_v48_77_dcp_drfc_bcde_rifa_actsi_main"
V76_COMPLETE="${V4877_V76_COMPLETE:-$BASE_OUT/OC-RAP-v48.76-PIPELINE_COMPLETE.json}"
V76_COMPARE="${V4877_V76_COMPARE:-$BASE_OUT/OC-RAP-v48.76-DCP-DRFC-BCDE-RIFA-OC-ICSM-comparison.json}"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.77-reference-reuse-contract.json"
RUNTIME_AUDIT="$BASE_OUT/OC-RAP-v48.77-runtime-code-contract.json"
AUDIT="$BASE_OUT/OC-RAP-v48.77-OC-ACTSI-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.77-DCP-DRFC-BCDE-RIFA-OC-ACTSI-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.77-PIPELINE_COMPLETE.json"

mkdir -p "$BASE_OUT"
rm -rf "$G_RUN" "$H_RUN"
rm -f "$REF_AUDIT" "$RUNTIME_AUDIT" "$AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" \
       "$G_RUN.zip" "$H_RUN.zip" "$BASE_OUT/OC-RAP-v48.77-OC-ACTSI-audits.zip"

python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python tools/check_v48_77_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME_AUDIT"

python - "$V76_COMPLETE" "$V76_COMPARE" "$REFERENCE_A" "$C75_RUN" "$D75_RUN" "$E76_RUN" "$F76_RUN" <<'PY'
import json,pathlib,sys
complete,compare=map(pathlib.Path,sys.argv[1:3]);runs=[pathlib.Path(x) for x in sys.argv[3:]]
if not complete.is_file(): raise SystemExit(f'missing V48.76 sentinel: {complete}')
if not compare.is_file(): raise SystemExit(f'missing V48.76 comparison: {compare}')
m=json.loads(complete.read_text());p=json.loads(compare.read_text()).get('preregistered_decision') or {}
if not (m.get('valid') and m.get('attribution_ready') and m.get('engineering_version')=='v48.76.0-OC-ICSM' and not m.get('test_roots_read')):
    raise SystemExit(f'V48.76 prerequisite invalid: {m}')
if not (p.get('status')=='STOP' and p.get('signed_margin_supervision_go') is False and p.get('next_branch')=='signed_margin_supervision_stop_two_gain_transport_representation_bottleneck_then_structured_absolute_source_interface'):
    raise SystemExit(f'V48.76 branch mismatch: {p}')
for r in runs:
    if not r.is_dir(): raise SystemExit(f'missing prerequisite run: {r}')
PY

for f in evidence_adapt_teacher_pcd_index.jsonl evidence_adapt_teacher_pcd_index_summary.json evidence_adapt_dev_teacher_pcd_index.jsonl evidence_adapt_dev_teacher_pcd_index_summary.json; do
  [[ -s "$REFERENCE_A/$f" ]] || { echo "missing reference teacher artifact: $REFERENCE_A/$f" >&2; exit 30; }
done
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"
DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"
TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
for d in "$CERT_NEAR" "$CERT_CONTACT" "$DEV_NEAR" "$DEV_CONTACT" "$TRAIN_NEAR" "$TRAIN_CONTACT"; do
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing V48.77 protocol root: $d" >&2; exit 30; }
done

# Frozen Stage-I / RIFA / evidence contract inherited from V48.76.
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false EVIDENCE_UNBOUNDED_HARM_FACTORS=false EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0 EVIDENCE_COMPONENT_SCALE=6.0 EVIDENCE_COMPONENT_HEADS=true EVIDENCE_COMPONENT_COUNT=5
export EVIDENCE_DUAL_INTERACTION_BRIDGE=true EVIDENCE_FACTORIZED_HARM_INTERACTION=false EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false EVIDENCE_RANK_BENEFIT_SKIP=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=false
export EVIDENCE_ROCT_BENEFIT=true EVIDENCE_ROCT_DEPLOYABILITY=true EVIDENCE_ROCT_SCALE=3.0 EVIDENCE_ROCT_ALPHA=0.20 EVIDENCE_ROCT_BETA=0.20 EVIDENCE_ROCT_TOP_M=8 EVIDENCE_ROCT_OPTION_TEMPERATURE=0.35 EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050
export EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=false EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=false EVIDENCE_PHYSICAL_STUDENT_DRS=false EVIDENCE_DEP_BOUNDARY_ALIGNED=false EVIDENCE_GAP_ORDINAL_ONLY=false EVIDENCE_COMMON_MEASURE_ROOT_MASS=false EVIDENCE_ADMISSION_HEAD=false
export EVIDENCE_NATIVE_DRS_TOLERANCE=0.05 EVIDENCE_NATIVE_DEPLOYABILITY_TOLERANCE=0.05 EVIDENCE_NATIVE_GAP_TOLERANCE=0.05 EVIDENCE_NATIVE_POSITIVE_GAIN=0.015 EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction PROPOSAL_TOP_K=5
export TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class EVAL_OPTION_EXECUTION_SEMANTICS=observation_class OPTION_EXECUTION_SEMANTICS=observation_class ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_MODE=raw ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_SCALE=0.10 ORDINAL_EVIDENCE_COMPONENT_MARGIN_CANONICAL_SCALES="" ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_RELIABILITY="1,1,1,0,0"

train_arm() {
  local run="$1"
  local arm="$2"
  local fidelity="$3"
  mkdir -p "$run/candidates" "$run/logs"
  train_one() {
    local v="$1"
    local gpu="$2"
    local src="$REFERENCE_A/candidates/$v"
    local dst="$run/candidates/$v"
    mkdir -p "$dst"
    if [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]]; then
      set -a; source "$src/FACTOR_SUPPORT_CONTRACT.env"; set +a
    else
      export EVIDENCE_COMPONENT_RELIABILITY="1,1,1,0,0"
    fi
    RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" \
    INIT_CKPT="$src/model_v48_trac_sr/best.pt" TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" \
    GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" \
    EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_semantic_witness_gain STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_semantic_witness_gain \
    ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=false ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=false ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION=false ABSOLUTE_COMMON_WITNESS_CORRECTION=false ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION=false ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true \
    SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT=true SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=false \
    SEMANTIC_WITNESS_ROUTE_ALIGNMENT=true SEMANTIC_WITNESS_REENTRY_ALIGNMENT=true SEMANTIC_WITNESS_CONTROL_PROJECTION=true SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false \
    SEMANTIC_WITNESS_PROJECTION_FIDELITY="$fidelity" SEMANTIC_WITNESS_ACTIVE_CONSTRAINT_TYPED_SOURCE=true \
    SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY=false SEMANTIC_WITNESS_ROBUST_OCCUPANCY=false SEMANTIC_WITNESS_SOFT_OCCUPANCY_DISAGREEMENT=false \
    SEMANTIC_WITNESS_BOUNDARY_LOCALIZED_OCCUPANCY_TRUST=false SEMANTIC_WITNESS_HISTORY_OCCUPANCY_REACHABILITY=false SEMANTIC_WITNESS_INTERACTION_BOX_SUPPORT=false SEMANTIC_WITNESS_INTERACTION_HULL_SUPPORT=false SEMANTIC_WITNESS_INTERACTION_ANCHOR_SUPPORT=false SEMANTIC_WITNESS_INTERACTION_RESPONSE_SUPPORT=false \
    ABSOLUTE_FEASIBILITY_WEIGHT=1.0 ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=censor_exact_0p5 ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE=signed_margin_huber \
    GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false BEST_METRIC=direct_absolute_signed_margin_huber BEST_METRIC_MIN_DELTA=0.00001 \
    EVIDENCE_ADAPT_EPOCHS="${V4877_ACTSI_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4877_ACTSI_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4877_ACTSI_LR:-0.001}" \
    MAX_EVIDENCE_CALIBRATOR_PARAMS=12 OCRAP_ALGORITHM_VERSION="v48.77-DCP-DRFC-BCDE-RIFA-OC-ACTSI-${arm}" \
    bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$run/logs/actsi_${arm}_${v}.log" 2>&1
    grep -qx 'ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=censor_exact_0p5' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE=signed_margin_huber' "$dst/POLICY_CONTRACT.env" || exit 30
    python tools/check_v48_77_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --fidelity "$fidelity" --output "$dst/V48_77_STAGE_I_STATE_ISOLATION.json"
  }
  set +e
  train_one balanced "$GPU0" & p0=$!
  train_one precision "$GPU1" & p1=$!
  wait "$p0"; r0=$?
  wait "$p1"; r1=$?
  set -e
  [[ "$r0" == 0 && "$r1" == 0 ]] || { echo "V48.77 $arm training failed balanced=$r0 precision=$r1" >&2; exit 30; }
  python tools/check_v48_77_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --actsi-run "$run" --fidelity "$fidelity" --output "$run/V48_77_VARIANT_ISOLATION.json"
  python - "$run/V48_77_FACTOR_CONTRACT.json" "$arm" "$fidelity" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);arm=sys.argv[2];fid=sys.argv[3].lower()=='true'
d={'event':'v48_77_factor_contract','version':'v48.77-DCP-DRFC-BCDE-RIFA-OC-ACTSI','engineering_version':'v48.77.0-OC-ACTSI','arm':arm,'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'Observation-Consistent Active-Constraint Typed Source Interface (OC-ACTSI)','source_intervention':'replace the V48.76 global positive/negative semantic slopes by a 6x2 table indexed only by the binding non-compensatory active constraint; physical certificate, features, OC-MERO and signed-margin supervision are unchanged','active_constraint_typed_source':True,'active_constraint_rows':['clearance','stopping','control','stability','route','persistent_reentry'],'gain_columns':['positive_rescue','universal_failure'],'trainable_state':'direct_absolute_semantic_witness_gain[6,2]','trainable_parameters':12,'option_id_input':False,'regime_id_input':False,'absolute_feasibility_truth_contract':'censor_exact_0p5','absolute_feasibility_supervision_objective':'signed_margin_huber','huber_beta':1.0,'floor_rows_relabelled':False,'teacher_files_modified':False,'dataset_reconstruction':False,'projection_fidelity':fid,'control_projection':True,'route_alignment':True,'reentry_alignment':True,'boundary_transport':False,'scope':'Near+Contact shared regime-agnostic absolute source','threshold':0.5,'threshold_search':False,'proposal_top_k':5,'proposal_expansion':False,'root_retraining':False,'relative_score_intervention':False,'teacher_future_input':False,'test_roots_read':False,'created_unix':time.time()}
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
}

train_arm "$G_RUN" G77_TYPED_PROJ false
train_arm "$H_RUN" H77_MAIN_ACTSI true

run_cal() {
  local run="$1"; local tag="$2"
  set +e
  OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.77-${tag}-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.77.0-OC-ACTSI-${tag}" \
  CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
  ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 \
  bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_77_${tag}_certificate_controller.log" 2>&1
  rc=$?; set -e
  case "$rc" in 0|20) echo "$tag calibration valid evidence RC=$rc";; *) echo "$tag calibration engineering failure RC=$rc" >&2; return 30;; esac
}
run_cal "$G_RUN" G77_TYPED_PROJ
run_cal "$H_RUN" H77_MAIN_ACTSI

python tools/audit_v48_77_actsi.py --c75 "$C75_RUN" --d75 "$D75_RUN" --e76 "$E76_RUN" --f76 "$F76_RUN" --g77 "$G_RUN" --h77 "$H_RUN" --output "$AUDIT"
python tools/compare_v48_77_actsi.py --audit "$AUDIT" --v76-complete "$V76_COMPLETE" --v76-comparison "$V76_COMPARE" --output "$COMPARE"
python tools/check_v48_77_pipeline_complete.py --reference-contract "$REF_AUDIT" --runtime-contract "$RUNTIME_AUDIT" --v76-complete "$V76_COMPLETE" --v76-comparison "$V76_COMPARE" --g77-run "$G_RUN" --h77-run "$H_RUN" --audit "$AUDIT" --comparison "$COMPARE" --output "$PIPELINE_COMPLETE"

cd "$BASE_OUT"
for run in "$G_RUN" "$H_RUN"; do
  b="$(basename "$run")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"
done
zip -qj OC-RAP-v48.77-OC-ACTSI-audits.zip "$REF_AUDIT" "$RUNTIME_AUDIT" "$AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V76_COMPLETE" "$V76_COMPARE" \
  "$G_RUN/V48_77_VARIANT_ISOLATION.json" "$G_RUN/V48_77_FACTOR_CONTRACT.json" "$H_RUN/V48_77_VARIANT_ISOLATION.json" "$H_RUN/V48_77_FACTOR_CONTRACT.json" \
  "$G_RUN/candidates/balanced/V48_77_STAGE_I_STATE_ISOLATION.json" "$G_RUN/candidates/precision/V48_77_STAGE_I_STATE_ISOLATION.json" \
  "$H_RUN/candidates/balanced/V48_77_STAGE_I_STATE_ISOLATION.json" "$H_RUN/candidates/precision/V48_77_STAGE_I_STATE_ISOLATION.json"
echo "v48.77 complete. Upload $(basename "$H_RUN").zip + $(basename "$G_RUN").zip + OC-RAP-v48.77-OC-ACTSI-audits.zip"
