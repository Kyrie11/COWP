#!/usr/bin/env bash
# V48.79 OC-PSTC: Physical-vs-Structural Truth-Contract adjudication.
# Reuse the exact J78 nested root-tail one-scalar source. The only scientific
# intervention is supervision censorship to candidates whose *teacher nested
# OC-MERO active tail* has zero conservative exposure to structural teacher
# floors/overrides/hidden-branch semantics. No label rewrite, no new source
# capacity, no regime input and no boundary transport.
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
REFERENCE_A="${V4879_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
J78_RUN="${V4879_J78:-$BASE_OUT/ocrap_v48_78_dcp_drfc_bcde_rifa_rtsi_main}"
V78_COMPARE="${V4879_V78_COMPARE:-$BASE_OUT/OC-RAP-v48.78-DCP-DRFC-BCDE-RIFA-OC-RTSI-comparison.json}"
V78_COMPLETE="${V4879_V78_COMPLETE:-$BASE_OUT/OC-RAP-v48.78-PIPELINE_COMPLETE.json}"
K_RUN="$BASE_OUT/ocrap_v48_79_dcp_drfc_bcde_rifa_pstc_main"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.79-reference-reuse-contract.json"
RUNTIME_AUDIT="$BASE_OUT/OC-RAP-v48.79-runtime-code-contract.json"
TRAIN_TRUTH_INDEX="$BASE_OUT/OC-RAP-v48.79-train-dev-physical-structural-truth-index.jsonl"
TRAIN_TRUTH_SUMMARY="$BASE_OUT/OC-RAP-v48.79-train-dev-physical-structural-truth-index-summary.json"
EVAL_TRUTH_INDEX="$BASE_OUT/OC-RAP-v48.79-dev-certificate-physical-structural-truth-index.jsonl"
EVAL_TRUTH_SUMMARY="$BASE_OUT/OC-RAP-v48.79-dev-certificate-physical-structural-truth-index-summary.json"
AUDIT="$BASE_OUT/OC-RAP-v48.79-OC-PSTC-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.79-DCP-DRFC-BCDE-RIFA-OC-PSTC-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.79-PIPELINE_COMPLETE.json"
# Deliberately reuse V48.78 tensor cache: the truth sidecar is attached after
# tensor materialization and does not change any model feature tensor.
TENSOR_CACHE_DIR="${V4879_TENSOR_CACHE_DIR:-$BASE_OUT/.ocrap_v48_78_tensor_cache}"
mkdir -p "$BASE_OUT" "$TENSOR_CACHE_DIR"
rm -rf "$K_RUN"
rm -f "$REF_AUDIT" "$RUNTIME_AUDIT" "$TRAIN_TRUTH_INDEX" "$TRAIN_TRUTH_SUMMARY" "$EVAL_TRUTH_INDEX" "$EVAL_TRUTH_SUMMARY" "$AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$K_RUN.zip" "$BASE_OUT/OC-RAP-v48.79-OC-PSTC-audits.zip"

python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python tools/check_v48_79_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME_AUDIT"
python - "$V78_COMPLETE" "$V78_COMPARE" "$REFERENCE_A" "$J78_RUN" <<'PY'
import json,pathlib,sys
complete,compare=map(pathlib.Path,sys.argv[1:3]);runs=[pathlib.Path(x) for x in sys.argv[3:]]
for p in (complete,compare):
    if not p.is_file():raise SystemExit(f'missing V48.78 prerequisite: {p}')
m=json.loads(complete.read_text());p=json.loads(compare.read_text());d=p.get('preregistered_decision') or {}
if not (m.get('valid') and m.get('attribution_ready') and m.get('engineering_version')=='v48.78.0-OC-RTSI' and not m.get('test_roots_read')):raise SystemExit(f'V48.78 pipeline invalid: {m}')
if not (p.get('valid') and p.get('attribution_ready') and d.get('status')=='STOP' and d.get('I78_root_shape_go') is False and d.get('J78_root_tail_source_go') is False and d.get('next_branch')=='root_tail_source_stop_close_low_capacity_absolute_source_adapter_family_then_teacher_truth_contract_adjudication_before_any_new_source_capacity'):raise SystemExit(f'V48.78 branch mismatch: {d}')
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

# Build only train+dev truth metadata before GPU training. Certificate metadata
# is intentionally not read until training/model selection has finished.
python tools/build_v48_79_pstc_truth_index.py \
  --root train_near="$TRAIN_NEAR" --root train_contact="$TRAIN_CONTACT" \
  --root dev_near="$DEV_NEAR" --root dev_contact="$DEV_CONTACT" \
  --output "$TRAIN_TRUTH_INDEX" --summary "$TRAIN_TRUTH_SUMMARY" \
  --alpha 0.20 --beta 0.20 --top-m 8 --workers "${V4879_TRUTH_INDEX_WORKERS:-8}"

# Frozen Stage-I/RIFA contract, execution-identical to J78 except truth policy.
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false EVIDENCE_UNBOUNDED_HARM_FACTORS=false EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0 EVIDENCE_COMPONENT_SCALE=6.0 EVIDENCE_COMPONENT_HEADS=true EVIDENCE_COMPONENT_COUNT=5
export EVIDENCE_DUAL_INTERACTION_BRIDGE=true EVIDENCE_FACTORIZED_HARM_INTERACTION=false EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false EVIDENCE_RANK_BENEFIT_SKIP=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=false
export EVIDENCE_ROCT_BENEFIT=true EVIDENCE_ROCT_DEPLOYABILITY=true EVIDENCE_ROCT_SCALE=3.0 EVIDENCE_ROCT_ALPHA=0.20 EVIDENCE_ROCT_BETA=0.20 EVIDENCE_ROCT_TOP_M=8 EVIDENCE_ROCT_OPTION_TEMPERATURE=0.35 EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050
export EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=false EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=false EVIDENCE_PHYSICAL_STUDENT_DRS=false EVIDENCE_DEP_BOUNDARY_ALIGNED=false EVIDENCE_GAP_ORDINAL_ONLY=false EVIDENCE_COMMON_MEASURE_ROOT_MASS=false EVIDENCE_ADMISSION_HEAD=false
export EVIDENCE_NATIVE_DRS_TOLERANCE=0.05 EVIDENCE_NATIVE_DEPLOYABILITY_TOLERANCE=0.05 EVIDENCE_NATIVE_GAP_TOLERANCE=0.05 EVIDENCE_NATIVE_POSITIVE_GAIN=0.015 EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction PROPOSAL_TOP_K=5
export TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class EVAL_OPTION_EXECUTION_SEMANTICS=observation_class OPTION_EXECUTION_SEMANTICS=observation_class ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_MODE=raw ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_SCALE=0.10 ORDINAL_EVIDENCE_COMPONENT_MARGIN_CANONICAL_SCALES="" ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_RELIABILITY="1,1,1,0,0"

mkdir -p "$K_RUN/candidates" "$K_RUN/logs"
train_one() {
  local v="$1"; local gpu="$2"; local src="$REFERENCE_A/candidates/$v"; local dst="$K_RUN/candidates/$v"
  mkdir -p "$dst"
  if [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]]; then set -a; source "$src/FACTOR_SUPPORT_CONTRACT.env"; set +a; else export EVIDENCE_COMPONENT_RELIABILITY="1,1,1,0,0"; fi
  RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" INIT_CKPT="$src/model_v48_trac_sr/best.pt" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" \
  EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_root_tail_source_scale STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_root_tail_source_scale \
  ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=false ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=false ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION=false ABSOLUTE_COMMON_WITNESS_CORRECTION=false ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION=false ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true \
  SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT=true SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=false SEMANTIC_WITNESS_ROUTE_ALIGNMENT=true SEMANTIC_WITNESS_REENTRY_ALIGNMENT=true SEMANTIC_WITNESS_CONTROL_PROJECTION=true SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false \
  SEMANTIC_WITNESS_PROJECTION_FIDELITY=false SEMANTIC_WITNESS_ACTIVE_CONSTRAINT_TYPED_SOURCE=false SEMANTIC_WITNESS_ROOT_TAIL_SOURCE=true SEMANTIC_WITNESS_TAIL_LOCALIZATION=true \
  SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY=false SEMANTIC_WITNESS_ROBUST_OCCUPANCY=false SEMANTIC_WITNESS_SOFT_OCCUPANCY_DISAGREEMENT=false SEMANTIC_WITNESS_BOUNDARY_LOCALIZED_OCCUPANCY_TRUST=false SEMANTIC_WITNESS_HISTORY_OCCUPANCY_REACHABILITY=false SEMANTIC_WITNESS_INTERACTION_BOX_SUPPORT=false SEMANTIC_WITNESS_INTERACTION_HULL_SUPPORT=false SEMANTIC_WITNESS_INTERACTION_ANCHOR_SUPPORT=false SEMANTIC_WITNESS_INTERACTION_RESPONSE_SUPPORT=false \
  ABSOLUTE_FEASIBILITY_WEIGHT=1.0 ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=censor_structural_tail ABSOLUTE_FEASIBILITY_TRUTH_INDEX="$TRAIN_TRUTH_INDEX" ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE=signed_margin_huber \
  GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false BEST_METRIC=direct_absolute_signed_margin_huber BEST_METRIC_MIN_DELTA=0.00001 \
  EVIDENCE_ADAPT_EPOCHS="${V4879_PSTC_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4879_PSTC_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4879_PSTC_LR:-0.001}" MAX_EVIDENCE_CALIBRATOR_PARAMS=1 \
  PERSISTENT_TENSOR_CACHE=true PERSISTENT_TENSOR_CACHE_DIR="$TENSOR_CACHE_DIR" PERSISTENT_TENSOR_CACHE_BUILD_WORKERS="${PERSISTENT_TENSOR_CACHE_BUILD_WORKERS:-8}" SAVE_EVERY_EPOCH=false SAVE_LATEST=false \
  OCRAP_ALGORITHM_VERSION="v48.79-DCP-DRFC-BCDE-RIFA-OC-PSTC-K79_PHYSICAL_TAIL_PROBE" \
  bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$K_RUN/logs/pstc_K79_${v}.log" 2>&1
  grep -qx 'ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=censor_structural_tail' "$dst/POLICY_CONTRACT.env" || exit 30
  grep -qx 'ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE=signed_margin_huber' "$dst/POLICY_CONTRACT.env" || exit 30
  grep -Fxq "ABSOLUTE_FEASIBILITY_TRUTH_INDEX=$TRAIN_TRUTH_INDEX" "$dst/POLICY_CONTRACT.env" || exit 30
  python tools/check_v48_79_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --truth-index "$TRAIN_TRUTH_INDEX" --output "$dst/V48_79_STAGE_I_STATE_ISOLATION.json"
}
set +e; train_one balanced "$GPU0" & p0=$!; train_one precision "$GPU1" & p1=$!; wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ "$r0" == 0 && "$r1" == 0 ]] || { echo "V48.79 K79 training failed balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/check_v48_79_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --pstc-run "$K_RUN" --truth-index "$TRAIN_TRUTH_INDEX" --output "$K_RUN/V48_79_VARIANT_ISOLATION.json"
python - "$K_RUN/V48_79_FACTOR_CONTRACT.json" "$TRAIN_TRUTH_SUMMARY" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); summary=json.loads(pathlib.Path(sys.argv[2]).read_text())
d={'event':'v48_79_factor_contract','version':'v48.79-DCP-DRFC-BCDE-RIFA-OC-PSTC','engineering_version':'v48.79.0-OC-PSTC','arm':'K79_PHYSICAL_TAIL_PROBE','stage_i':'bitwise v48.56-A checkpoint','stage_ii':'same J78 nested deployability-tail zero-translation root source','scientific_intervention':'supervision truth contract only: retain candidate iff exact nested teacher OC-MERO active tail has zero conservative exposure to structural floor/override/hidden-branch semantics','root_tail_source':True,'tail_localization':True,'source_capacity_changed_vs_J78':False,'trainable_state':'direct_absolute_root_tail_source_scale[1]','expected_trainable_parameters':1,'option_translation_zero_mean':True,'option_id_input':False,'regime_id_input':False,'classlocal_transport':False,'active_constraint_typed_source':False,'projection_fidelity':False,'boundary_transport':False,'absolute_feasibility_truth_contract':'censor_structural_tail','absolute_feasibility_supervision_objective':'signed_margin_huber','huber_beta':1.0,'truth_index_summary':summary,'teacher_files_modified':False,'teacher_labels_changed':False,'dataset_reconstruction':False,'control_projection':True,'route_alignment':True,'reentry_alignment':True,'threshold':0.5,'threshold_search':False,'proposal_top_k':5,'root_retraining':False,'relative_score_intervention':False,'teacher_future_input_to_model':False,'test_roots_read':False,'persistent_tensor_cache':True,'tensor_cache_reuses_v48_78_feature_key':True,'save_every_epoch':False,'save_latest':False,'created_unix':time.time()}
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY

set +e
OUTPUTDIR="$K_RUN" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.79-K79-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.79.0-OC-PSTC-K79" \
CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 \
bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$K_RUN/logs/v48_79_K79_certificate_controller.log" 2>&1
rc=$?; set -e; case "$rc" in 0|20) echo "K79 calibration valid evidence RC=$rc";;*) echo "K79 calibration engineering failure RC=$rc" >&2; exit 30;; esac

# Only now read certificate structural metadata for held-out adjudication.
python tools/build_v48_79_pstc_truth_index.py \
  --root dev_near="$DEV_NEAR" --root dev_contact="$DEV_CONTACT" \
  --root certificate_near="$CERT_NEAR" --root certificate_contact="$CERT_CONTACT" \
  --output "$EVAL_TRUTH_INDEX" --summary "$EVAL_TRUTH_SUMMARY" \
  --alpha 0.20 --beta 0.20 --top-m 8 --workers "${V4879_TRUTH_INDEX_WORKERS:-8}"

python tools/audit_v48_79_pstc.py --j78 "$J78_RUN" --k79 "$K_RUN" --truth-index "$EVAL_TRUTH_INDEX" --truth-summary "$EVAL_TRUTH_SUMMARY" --output "$AUDIT"
python tools/compare_v48_79_pstc.py --audit "$AUDIT" --v78-comparison "$V78_COMPARE" --output "$COMPARE"

python - "$REF_AUDIT" "$RUNTIME_AUDIT" "$K_RUN" "$TRAIN_TRUTH_SUMMARY" "$EVAL_TRUTH_SUMMARY" "$AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" <<'PY'
import json,pathlib,sys
ref,runtime,krun,trsum,evsum,audit,compare,out=map(pathlib.Path,sys.argv[1:]); errors=[]
def load(p): return json.loads(p.read_text()) if p.is_file() else {}
for p in (ref,runtime,trsum,evsum,audit,compare):
 d=load(p)
 if not (p.is_file() and d.get('valid',True)): errors.append(f'invalid/missing {p}')
for v in ('balanced','precision'):
 for rel in (f'candidates/{v}/V48_79_STAGE_I_STATE_ISOLATION.json',f'candidates/{v}/model_v48_trac_sr/best.pt'):
  if not (krun/rel).is_file(): errors.append(f'missing {krun/rel}')
for rel in ('V48_79_VARIANT_ISOLATION.json','V48_79_FACTOR_CONTRACT.json','dedicated_recalibration_status.json'):
 if not (krun/rel).is_file(): errors.append(f'missing {krun/rel}')
doc={'schema':'ocrap-v48.79-pstc-pipeline-complete-v1','algorithm_version':'v48.79-DCP-DRFC-BCDE-RIFA-OC-PSTC','engineering_version':'v48.79.0-OC-PSTC','valid':not errors,'attribution_ready':not errors,'errors':errors,'arms':{'K79_PHYSICAL_TAIL_PROBE':str(krun),'historical_J78':'historical'},'checkpoint_packaging_required':True,'truth_contract':'censor_structural_tail','teacher_labels_changed':False,'source_capacity_changed_vs_J78':False,'persistent_tensor_cache':True,'save_every_epoch':False,'save_latest':False,'dataset_reconstruction':False,'test_roots_read':False}
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n'); print(json.dumps({'event':'v48_79_pipeline_complete','valid':not errors,'output':str(out)})); raise SystemExit(0 if not errors else 30)
PY

cd "$BASE_OUT"
b="$(basename "$K_RUN")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"
zip -qj OC-RAP-v48.79-OC-PSTC-audits.zip "$REF_AUDIT" "$RUNTIME_AUDIT" "$TRAIN_TRUTH_INDEX" "$TRAIN_TRUTH_SUMMARY" "$EVAL_TRUTH_INDEX" "$EVAL_TRUTH_SUMMARY" "$AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V78_COMPLETE" "$V78_COMPARE" "$K_RUN/V48_79_VARIANT_ISOLATION.json" "$K_RUN/V48_79_FACTOR_CONTRACT.json" "$K_RUN/candidates/balanced/V48_79_STAGE_I_STATE_ISOLATION.json" "$K_RUN/candidates/precision/V48_79_STAGE_I_STATE_ISOLATION.json"
echo "v48.79 complete. Upload $b.zip + OC-RAP-v48.79-OC-PSTC-audits.zip; keep both best.pt files in the run zip."
