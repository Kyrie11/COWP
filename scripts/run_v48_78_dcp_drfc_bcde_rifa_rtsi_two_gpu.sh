#!/usr/bin/env bash
# V48.78 OC-RTSI: close the V48.64-77 option-translation/gain family.
# Learn only a p-weighted zero-mean root-tail shape of the frozen executable
# recovery margins; J78 further gates that deformation by the exact nested
# OC-MERO LCVAR tail influence. No regime ID, option-ID bias or boundary transport.
export OCRAP_V48_74_SIGNED_VIABILITY=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
set -Eeuo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
REFERENCE_A="${V4878_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
C75_RUN="${V4878_C75:-$BASE_OUT/ocrap_v48_75_dcp_drfc_bcde_rifa_stca_projection_censored}"
E76_RUN="${V4878_E76:-$BASE_OUT/ocrap_v48_76_dcp_drfc_bcde_rifa_icsm_projection_margin}"
V77_COMPARE="${V4878_V77_COMPARE:-$BASE_OUT/OC-RAP-v48.77-DCP-DRFC-BCDE-RIFA-OC-ACTSI-comparison.json}"
V77_COMPLETE="${V4878_V77_COMPLETE:-$BASE_OUT/OC-RAP-v48.77-PIPELINE_COMPLETE.json}"
I_RUN="$BASE_OUT/ocrap_v48_78_dcp_drfc_bcde_rifa_rtsi_root_shape"
J_RUN="$BASE_OUT/ocrap_v48_78_dcp_drfc_bcde_rifa_rtsi_main"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.78-reference-reuse-contract.json"
RUNTIME_AUDIT="$BASE_OUT/OC-RAP-v48.78-runtime-code-contract.json"
AUDIT="$BASE_OUT/OC-RAP-v48.78-OC-RTSI-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.78-DCP-DRFC-BCDE-RIFA-OC-RTSI-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.78-PIPELINE_COMPLETE.json"
TENSOR_CACHE_DIR="${V4878_TENSOR_CACHE_DIR:-$BASE_OUT/.ocrap_v48_78_tensor_cache}"
mkdir -p "$BASE_OUT" "$TENSOR_CACHE_DIR"
rm -rf "$I_RUN" "$J_RUN"
rm -f "$REF_AUDIT" "$RUNTIME_AUDIT" "$AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$I_RUN.zip" "$J_RUN.zip" "$BASE_OUT/OC-RAP-v48.78-OC-RTSI-audits.zip"

python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python tools/check_v48_78_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME_AUDIT"
python - "$V77_COMPLETE" "$V77_COMPARE" "$REFERENCE_A" "$C75_RUN" "$E76_RUN" <<'PY'
import json,pathlib,sys
complete,compare=map(pathlib.Path,sys.argv[1:3]);runs=[pathlib.Path(x) for x in sys.argv[3:]]
for p in (complete,compare):
    if not p.is_file():raise SystemExit(f'missing V48.77 prerequisite: {p}')
m=json.loads(complete.read_text());p=json.loads(compare.read_text());d=p.get('preregistered_decision') or {}
if not (m.get('valid') and m.get('attribution_ready') and m.get('engineering_version')=='v48.77.0-OC-ACTSI' and not m.get('test_roots_read')):raise SystemExit(f'V48.77 pipeline invalid: {m}')
if not (p.get('valid') and p.get('attribution_ready') and d.get('status')=='STOP' and d.get('active_constraint_typed_source_go') is False and d.get('next_branch')=='active_typed_transport_stop_close_gain_transport_family_then_structured_ocmero_tail_source_interface_no_gain_sweep'):raise SystemExit(f'V48.77 branch mismatch: {d}')
for r in runs:
    if not r.is_dir():raise SystemExit(f'missing prerequisite run: {r}')
PY

for f in evidence_adapt_teacher_pcd_index.jsonl evidence_adapt_teacher_pcd_index_summary.json evidence_adapt_dev_teacher_pcd_index.jsonl evidence_adapt_dev_teacher_pcd_index_summary.json; do
  [[ -s "$REFERENCE_A/$f" ]] || { echo "missing reference teacher artifact: $REFERENCE_A/$f" >&2; exit 30; }
done
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
for d in "$CERT_NEAR" "$CERT_CONTACT" "$DEV_NEAR" "$DEV_CONTACT" "$TRAIN_NEAR" "$TRAIN_CONTACT"; do [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing protocol root: $d" >&2; exit 30; }; done

# Frozen Stage-I/RIFA contract. Same as V48.76/77.
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false EVIDENCE_UNBOUNDED_HARM_FACTORS=false EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0 EVIDENCE_COMPONENT_SCALE=6.0 EVIDENCE_COMPONENT_HEADS=true EVIDENCE_COMPONENT_COUNT=5
export EVIDENCE_DUAL_INTERACTION_BRIDGE=true EVIDENCE_FACTORIZED_HARM_INTERACTION=false EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false EVIDENCE_RANK_BENEFIT_SKIP=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=false
export EVIDENCE_ROCT_BENEFIT=true EVIDENCE_ROCT_DEPLOYABILITY=true EVIDENCE_ROCT_SCALE=3.0 EVIDENCE_ROCT_ALPHA=0.20 EVIDENCE_ROCT_BETA=0.20 EVIDENCE_ROCT_TOP_M=8 EVIDENCE_ROCT_OPTION_TEMPERATURE=0.35 EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050
export EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=false EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=false EVIDENCE_PHYSICAL_STUDENT_DRS=false EVIDENCE_DEP_BOUNDARY_ALIGNED=false EVIDENCE_GAP_ORDINAL_ONLY=false EVIDENCE_COMMON_MEASURE_ROOT_MASS=false EVIDENCE_ADMISSION_HEAD=false
export EVIDENCE_NATIVE_DRS_TOLERANCE=0.05 EVIDENCE_NATIVE_DEPLOYABILITY_TOLERANCE=0.05 EVIDENCE_NATIVE_GAP_TOLERANCE=0.05 EVIDENCE_NATIVE_POSITIVE_GAIN=0.015 EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction PROPOSAL_TOP_K=5
export TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class EVAL_OPTION_EXECUTION_SEMANTICS=observation_class OPTION_EXECUTION_SEMANTICS=observation_class ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_MODE=raw ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_SCALE=0.10 ORDINAL_EVIDENCE_COMPONENT_MARGIN_CANONICAL_SCALES="" ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_RELIABILITY="1,1,1,0,0"

train_arm() {
  local run="$1" arm="$2" tail="$3"
  mkdir -p "$run/candidates" "$run/logs"
  train_one() {
    local v="$1"
    local gpu="$2"
    local src="$REFERENCE_A/candidates/$v"
    local dst="$run/candidates/$v"
    mkdir -p "$dst"
    if [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]]; then set -a; source "$src/FACTOR_SUPPORT_CONTRACT.env"; set +a; else export EVIDENCE_COMPONENT_RELIABILITY="1,1,1,0,0"; fi
    RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" INIT_CKPT="$src/model_v48_trac_sr/best.pt" \
    TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" \
    EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_root_tail_source_scale STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_root_tail_source_scale \
    ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=false ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=false ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION=false ABSOLUTE_COMMON_WITNESS_CORRECTION=false ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION=false ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true \
    SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT=true SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=false SEMANTIC_WITNESS_ROUTE_ALIGNMENT=true SEMANTIC_WITNESS_REENTRY_ALIGNMENT=true SEMANTIC_WITNESS_CONTROL_PROJECTION=true SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false \
    SEMANTIC_WITNESS_PROJECTION_FIDELITY=false SEMANTIC_WITNESS_ACTIVE_CONSTRAINT_TYPED_SOURCE=false SEMANTIC_WITNESS_ROOT_TAIL_SOURCE=true SEMANTIC_WITNESS_TAIL_LOCALIZATION="$tail" \
    SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY=false SEMANTIC_WITNESS_ROBUST_OCCUPANCY=false SEMANTIC_WITNESS_SOFT_OCCUPANCY_DISAGREEMENT=false SEMANTIC_WITNESS_BOUNDARY_LOCALIZED_OCCUPANCY_TRUST=false SEMANTIC_WITNESS_HISTORY_OCCUPANCY_REACHABILITY=false SEMANTIC_WITNESS_INTERACTION_BOX_SUPPORT=false SEMANTIC_WITNESS_INTERACTION_HULL_SUPPORT=false SEMANTIC_WITNESS_INTERACTION_ANCHOR_SUPPORT=false SEMANTIC_WITNESS_INTERACTION_RESPONSE_SUPPORT=false \
    ABSOLUTE_FEASIBILITY_WEIGHT=1.0 ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=censor_exact_0p5 ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE=signed_margin_huber \
    GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false BEST_METRIC=direct_absolute_signed_margin_huber BEST_METRIC_MIN_DELTA=0.00001 \
    EVIDENCE_ADAPT_EPOCHS="${V4878_RTSI_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4878_RTSI_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4878_RTSI_LR:-0.001}" MAX_EVIDENCE_CALIBRATOR_PARAMS=1 \
    PERSISTENT_TENSOR_CACHE=true PERSISTENT_TENSOR_CACHE_DIR="$TENSOR_CACHE_DIR" PERSISTENT_TENSOR_CACHE_BUILD_WORKERS="${PERSISTENT_TENSOR_CACHE_BUILD_WORKERS:-8}" SAVE_EVERY_EPOCH=false SAVE_LATEST=false \
    OCRAP_ALGORITHM_VERSION="v48.78-DCP-DRFC-BCDE-RIFA-OC-RTSI-${arm}" \
    bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$run/logs/rtsi_${arm}_${v}.log" 2>&1
    grep -qx 'ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=censor_exact_0p5' "$dst/POLICY_CONTRACT.env" || exit 30
    grep -qx 'ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE=signed_margin_huber' "$dst/POLICY_CONTRACT.env" || exit 30
    python tools/check_v48_78_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --tail-localization "$tail" --output "$dst/V48_78_STAGE_I_STATE_ISOLATION.json"
  }
  set +e; train_one balanced "$GPU0" & p0=$!; train_one precision "$GPU1" & p1=$!; wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
  [[ "$r0" == 0 && "$r1" == 0 ]] || { echo "V48.78 $arm training failed balanced=$r0 precision=$r1" >&2; exit 30; }
  python tools/check_v48_78_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --rtsi-run "$run" --tail-localization "$tail" --output "$run/V48_78_VARIANT_ISOLATION.json"
  python - "$run/V48_78_FACTOR_CONTRACT.json" "$arm" "$tail" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);arm=sys.argv[2];tail=sys.argv[3].lower()=='true'
d={'event':'v48_78_factor_contract','version':'v48.78-DCP-DRFC-BCDE-RIFA-OC-RTSI','engineering_version':'v48.78.0-OC-RTSI','arm':arm,'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'Observation-Consistent Root-Tail Source Interface (OC-RTSI)','source_intervention':'p-weighted zero-mean within-option root-margin deformation from deterministic observation-compatible OC-MERO lower-tail influence; J78 additionally composes the exact outer deployability-tail influence and native best-option responsibility','root_tail_source':True,'tail_localization':tail,'trainable_state':'direct_absolute_root_tail_source_scale[1]','expected_trainable_parameters':1,'option_translation_zero_mean':True,'option_id_input':False,'regime_id_input':False,'classlocal_transport':False,'active_constraint_typed_source':False,'projection_fidelity':False,'boundary_transport':False,'absolute_feasibility_truth_contract':'censor_exact_0p5','absolute_feasibility_supervision_objective':'signed_margin_huber','huber_beta':1.0,'teacher_files_modified':False,'dataset_reconstruction':False,'control_projection':True,'route_alignment':True,'reentry_alignment':True,'threshold':0.5,'threshold_search':False,'proposal_top_k':5,'root_retraining':False,'relative_score_intervention':False,'teacher_future_input':False,'test_roots_read':False,'persistent_tensor_cache':True,'save_every_epoch':False,'save_latest':False,'created_unix':time.time()}
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY
}

train_arm "$I_RUN" I78_ROOT_SHAPE false
train_arm "$J_RUN" J78_MAIN_RTSI true

run_cal() {
  local run="$1" tag="$2"; set +e
  OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.78-${tag}-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.78.0-OC-RTSI-${tag}" \
  CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 \
  bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_78_${tag}_certificate_controller.log" 2>&1
  rc=$?;set -e;case "$rc" in 0|20) echo "$tag calibration valid evidence RC=$rc";;*) echo "$tag calibration engineering failure RC=$rc" >&2;return 30;;esac
}
run_cal "$I_RUN" I78_ROOT_SHAPE
run_cal "$J_RUN" J78_MAIN_RTSI
python tools/audit_v48_78_rtsi.py --c75 "$C75_RUN" --e76 "$E76_RUN" --i78 "$I_RUN" --j78 "$J_RUN" --output "$AUDIT"
python tools/compare_v48_78_rtsi.py --audit "$AUDIT" --v77-comparison "$V77_COMPARE" --output "$COMPARE"

python - "$REF_AUDIT" "$RUNTIME_AUDIT" "$I_RUN" "$J_RUN" "$AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" <<'PY'
import hashlib,json,pathlib,sys
ref,runtime,irun,jrun,audit,compare,out=map(pathlib.Path,sys.argv[1:])
errors=[]
def load(p):return json.loads(p.read_text()) if p.is_file() else {}
for p in (ref,runtime,audit,compare):
 d=load(p)
 if not (p.is_file() and d.get('valid',True)):errors.append(f'invalid/missing {p}')
for run in (irun,jrun):
 for v in ('balanced','precision'):
  for rel in (f'candidates/{v}/V48_78_STAGE_I_STATE_ISOLATION.json',f'candidates/{v}/model_v48_trac_sr/best.pt'):
   if not (run/rel).is_file():errors.append(f'missing {run/rel}')
 for rel in ('V48_78_VARIANT_ISOLATION.json','V48_78_FACTOR_CONTRACT.json','dedicated_recalibration_status.json'):
  if not (run/rel).is_file():errors.append(f'missing {run/rel}')
doc={'schema':'ocrap-v48.78-rtsi-pipeline-complete-v1','algorithm_version':'v48.78-DCP-DRFC-BCDE-RIFA-OC-RTSI','engineering_version':'v48.78.0-OC-RTSI','valid':not errors,'attribution_ready':not errors,'errors':errors,'arms':{'I78_ROOT_SHAPE':str(irun),'J78_MAIN_RTSI':str(jrun),'historical_C75':'historical','historical_E76':'historical'},'checkpoint_packaging_required':True,'persistent_tensor_cache':True,'save_every_epoch':False,'save_latest':False,'dataset_reconstruction':False,'test_roots_read':False}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
print(json.dumps({'event':'v48_78_pipeline_complete','valid':not errors,'output':str(out)}));raise SystemExit(0 if not errors else 30)
PY

cd "$BASE_OUT"
for run in "$I_RUN" "$J_RUN"; do b="$(basename "$run")";rm -f "$b.zip";zip -qr "$b.zip" "$b";done
zip -qj OC-RAP-v48.78-OC-RTSI-audits.zip "$REF_AUDIT" "$RUNTIME_AUDIT" "$AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V77_COMPLETE" "$V77_COMPARE" "$I_RUN/V48_78_VARIANT_ISOLATION.json" "$I_RUN/V48_78_FACTOR_CONTRACT.json" "$J_RUN/V48_78_VARIANT_ISOLATION.json" "$J_RUN/V48_78_FACTOR_CONTRACT.json" "$I_RUN/candidates/balanced/V48_78_STAGE_I_STATE_ISOLATION.json" "$I_RUN/candidates/precision/V48_78_STAGE_I_STATE_ISOLATION.json" "$J_RUN/candidates/balanced/V48_78_STAGE_I_STATE_ISOLATION.json" "$J_RUN/candidates/precision/V48_78_STAGE_I_STATE_ISOLATION.json"
echo "v48.78 complete. Upload $(basename "$J_RUN").zip + $(basename "$I_RUN").zip + OC-RAP-v48.78-OC-RTSI-audits.zip; keep all best.pt files in the run zips."
