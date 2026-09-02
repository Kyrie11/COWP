#!/usr/bin/env bash
# V48.83 OC-CRTF: Observation-Consistent Counterfactual Recovery Tail Field.
# One new causal arm: the V48.82 signed nested-tail field is made action-relative
# by subtracting the unique nominal root-option interaction in each scene-time set.
export OCRAP_V48_74_SIGNED_VIABILITY=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
REFERENCE_A="${V4883_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V4883_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
O82_RUN="${V4883_O82:-$BASE_OUT/ocrap_v48_82_dcp_drfc_bcde_rifa_sntf_main}"
V82_COMPARE="${V4883_V82_COMPARE:-$BASE_OUT/OC-RAP-v48.82-DCP-DRFC-BCDE-RIFA-OC-SNTF-comparison.json}"
P_RUN="$BASE_OUT/ocrap_v48_83_dcp_drfc_bcde_rifa_crtf_main"
RUNTIME="$BASE_OUT/OC-RAP-v48.83-runtime-code-contract.json"
REF="$BASE_OUT/OC-RAP-v48.83-reference-reuse-contract.json"
AUDIT="$BASE_OUT/OC-RAP-v48.83-OC-CRTF-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.83-DCP-DRFC-BCDE-RIFA-OC-CRTF-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.83-PIPELINE_COMPLETE.json"
CACHE="${V4883_TENSOR_CACHE_DIR:-$BASE_OUT/.ocrap_v48_78_tensor_cache}"
mkdir -p "$BASE_OUT" "$CACHE"; rm -rf "$P_RUN"; rm -f "$RUNTIME" "$REF" "$AUDIT" "$COMPARE" "$COMPLETE"

python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF"
python tools/check_v48_83_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V82_COMPARE" "$L80_RUN" "$O82_RUN" <<'PY'
import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text()); d=p.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and d.get('status')=='SIGNED_NESTED_TAIL_FIELD_STOP' and d.get('signed_channel_increment_go')):
    raise SystemExit('V48.82 STOP + signed-channel-increment prerequisite missing')
for z in sys.argv[2:]:
    if not pathlib.Path(z).is_dir(): raise SystemExit(f'missing historical run: {z}')
PY

CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"

# V48.83 does not intervene on truth semantics.  Reuse byte-identical V48.80
# interval truth indices when their schema/SHA/role contract is valid; otherwise
# rebuild the exact same V48.80 truth object.  This removes ~9 minutes of repeated
# CPU work without changing supervision.
TRAIN_IDX82="$BASE_OUT/OC-RAP-v48.82-train-dev-physical-interval-truth-index.jsonl"
TRAIN_SUM82="$BASE_OUT/OC-RAP-v48.82-train-dev-physical-interval-truth-index-summary.json"
EVAL_IDX82="$BASE_OUT/OC-RAP-v48.82-dev-certificate-physical-interval-truth-index.jsonl"
EVAL_SUM82="$BASE_OUT/OC-RAP-v48.82-dev-certificate-physical-interval-truth-index-summary.json"
TRAIN_IDX="$TRAIN_IDX82"; TRAIN_SUM="$TRAIN_SUM82"; EVAL_IDX="$EVAL_IDX82"; EVAL_SUM="$EVAL_SUM82"
if ! python tools/check_v48_83_truth_index_reuse.py --index "$TRAIN_IDX82" --summary "$TRAIN_SUM82" --roles train_near,train_contact,dev_near,dev_contact >/dev/null 2>&1; then
  TRAIN_IDX="$BASE_OUT/OC-RAP-v48.83-train-dev-physical-interval-truth-index.jsonl"; TRAIN_SUM="$BASE_OUT/OC-RAP-v48.83-train-dev-physical-interval-truth-index-summary.json"
  rm -f "$TRAIN_IDX" "$TRAIN_SUM"
  python tools/build_v48_80_interval_truth_index.py --root train_near="$TRAIN_NEAR" --root train_contact="$TRAIN_CONTACT" --root dev_near="$DEV_NEAR" --root dev_contact="$DEV_CONTACT" --output "$TRAIN_IDX" --summary "$TRAIN_SUM" --workers "${V4883_TRUTH_INDEX_WORKERS:-8}"
fi
python tools/check_v48_83_truth_index_reuse.py --index "$TRAIN_IDX" --summary "$TRAIN_SUM" --roles train_near,train_contact,dev_near,dev_contact

export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false EVIDENCE_UNBOUNDED_HARM_FACTORS=false EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0 EVIDENCE_COMPONENT_SCALE=6.0 EVIDENCE_COMPONENT_HEADS=true EVIDENCE_COMPONENT_COUNT=5 EVIDENCE_DUAL_INTERACTION_BRIDGE=true EVIDENCE_ROCT_BENEFIT=true EVIDENCE_ROCT_DEPLOYABILITY=true EVIDENCE_ROCT_SCALE=3.0 EVIDENCE_ROCT_ALPHA=0.20 EVIDENCE_ROCT_BETA=0.20 EVIDENCE_ROCT_TOP_M=8 EVIDENCE_ROCT_OPTION_TEMPERATURE=0.35 EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050 EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true EVIDENCE_COMMON_MEASURE_ROOT_MASS=false EVIDENCE_ADMISSION_HEAD=false EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction PROPOSAL_TOP_K=5
export TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class EVAL_OPTION_EXECUTION_SEMANTICS=observation_class OPTION_EXECUTION_SEMANTICS=observation_class

train_one(){
  local v="$1" gpu="$2"; local src="$REFERENCE_A/candidates/$v" dst="$P_RUN/candidates/$v"; mkdir -p "$dst" "$P_RUN/logs"
  RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" INIT_CKPT="$src/model_v48_trac_sr/best.pt" TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_structured_tail_field_weight STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_structured_tail_field_weight ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT=true SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=false SEMANTIC_WITNESS_ROUTE_ALIGNMENT=true SEMANTIC_WITNESS_REENTRY_ALIGNMENT=true SEMANTIC_WITNESS_CONTROL_PROJECTION=true SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false SEMANTIC_WITNESS_PROJECTION_FIDELITY=false SEMANTIC_WITNESS_ACTIVE_CONSTRAINT_TYPED_SOURCE=false SEMANTIC_WITNESS_ROOT_TAIL_SOURCE=true SEMANTIC_WITNESS_TAIL_LOCALIZATION=true SEMANTIC_WITNESS_STRUCTURED_TAIL_FIELD=true SEMANTIC_WITNESS_SIGNED_TAIL_CHANNELS=true SEMANTIC_WITNESS_COUNTERFACTUAL_TAIL_RESPONSE=true ABSOLUTE_FEASIBILITY_WEIGHT=1.0 ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=structural_interval_bounds ABSOLUTE_FEASIBILITY_TRUTH_INDEX="$TRAIN_IDX" ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE=signed_margin_interval_huber BEST_METRIC=direct_absolute_signed_margin_interval_huber BEST_METRIC_MIN_DELTA=0.00001 EVIDENCE_ADAPT_EPOCHS="${V4883_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4883_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4883_LR:-0.001}" MAX_EVIDENCE_CALIBRATOR_PARAMS=384 PERSISTENT_TENSOR_CACHE=true PERSISTENT_TENSOR_CACHE_DIR="$CACHE" PERSISTENT_TENSOR_CACHE_BUILD_WORKERS="${PERSISTENT_TENSOR_CACHE_BUILD_WORKERS:-8}" SAVE_EVERY_EPOCH=false SAVE_LATEST=false OCRAP_ALGORITHM_VERSION="v48.83-OC-CRTF-P83" bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$P_RUN/logs/P83_${v}.log" 2>&1
  python tools/check_v48_83_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --truth-index "$TRAIN_IDX" --output "$dst/V48_83_STAGE_I_STATE_ISOLATION.json"
}
mkdir -p "$P_RUN/candidates"
# Only one new causal arm is trained.  Balanced/precision run simultaneously on
# the two A30s, cutting the V48.82 two-stage wall time roughly in half.
set +e; train_one balanced "$GPU0" & p0=$!; train_one precision "$GPU1" & p1=$!; wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || exit 30

set +e
OUTPUTDIR="$P_RUN" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.83-P83-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.83.0-OC-CRTF-P83" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$P_RUN/logs/v48_83_P83_certificate_controller.log" 2>&1
rc=$?; set -e; case $rc in 0|20);;*) exit 30;;esac

if ! python tools/check_v48_83_truth_index_reuse.py --index "$EVAL_IDX82" --summary "$EVAL_SUM82" --roles dev_near,dev_contact,certificate_near,certificate_contact >/dev/null 2>&1; then
  EVAL_IDX="$BASE_OUT/OC-RAP-v48.83-dev-certificate-physical-interval-truth-index.jsonl"; EVAL_SUM="$BASE_OUT/OC-RAP-v48.83-dev-certificate-physical-interval-truth-index-summary.json"
  rm -f "$EVAL_IDX" "$EVAL_SUM"
  python tools/build_v48_80_interval_truth_index.py --root dev_near="$DEV_NEAR" --root dev_contact="$DEV_CONTACT" --root certificate_near="$CERT_NEAR" --root certificate_contact="$CERT_CONTACT" --output "$EVAL_IDX" --summary "$EVAL_SUM" --workers "${V4883_TRUTH_INDEX_WORKERS:-8}"
fi
python tools/check_v48_83_truth_index_reuse.py --index "$EVAL_IDX" --summary "$EVAL_SUM" --roles dev_near,dev_contact,certificate_near,certificate_contact
python tools/audit_v48_83_crtf.py --l80 "$L80_RUN" --o82 "$O82_RUN" --p83 "$P_RUN" --truth-index "$EVAL_IDX" --truth-summary "$EVAL_SUM" --output "$AUDIT"
python tools/compare_v48_83_crtf.py --audit "$AUDIT" --v82-comparison "$V82_COMPARE" --output "$COMPARE"
python tools/check_v48_83_pipeline_complete.py --run "$P_RUN" --runtime "$RUNTIME" --reference "$REF" --audit "$AUDIT" --comparison "$COMPARE" --train-truth-index "$TRAIN_IDX" --eval-truth-index "$EVAL_IDX" --output "$COMPLETE"
cd "$BASE_OUT"; b=$(basename "$P_RUN"); rm -f "$b.zip"; zip -qr "$b.zip" "$b"
zip -qj OC-RAP-v48.83-OC-CRTF-audits.zip "$REF" "$RUNTIME" "$TRAIN_SUM" "$EVAL_SUM" "$AUDIT" "$COMPARE" "$COMPLETE"
echo "V48.83 complete: upload $b.zip + OC-RAP-v48.83-OC-CRTF-audits.zip"
