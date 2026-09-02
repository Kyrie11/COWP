#!/usr/bin/env bash
# V48.82 OC-SNTF: Observation-Consistent Signed Nested Tail Field.
export OCRAP_V48_74_SIGNED_VIABILITY=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"; export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"; export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
REFERENCE_A="${V4882_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V4882_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V81_COMPARE="${V4882_V81_COMPARE:-$BASE_OUT/OC-RAP-v48.81-DCP-DRFC-BCDE-RIFA-OC-SITC-comparison.json}"
N_RUN="$BASE_OUT/ocrap_v48_82_dcp_drfc_bcde_rifa_sntf_single"; O_RUN="$BASE_OUT/ocrap_v48_82_dcp_drfc_bcde_rifa_sntf_main"
RUNTIME="$BASE_OUT/OC-RAP-v48.82-runtime-code-contract.json"; REF="$BASE_OUT/OC-RAP-v48.82-reference-reuse-contract.json"
TRAIN_IDX="$BASE_OUT/OC-RAP-v48.82-train-dev-physical-interval-truth-index.jsonl"; TRAIN_SUM="$BASE_OUT/OC-RAP-v48.82-train-dev-physical-interval-truth-index-summary.json"
EVAL_IDX="$BASE_OUT/OC-RAP-v48.82-dev-certificate-physical-interval-truth-index.jsonl"; EVAL_SUM="$BASE_OUT/OC-RAP-v48.82-dev-certificate-physical-interval-truth-index-summary.json"
AUDIT="$BASE_OUT/OC-RAP-v48.82-OC-SNTF-audit.json"; COMPARE="$BASE_OUT/OC-RAP-v48.82-DCP-DRFC-BCDE-RIFA-OC-SNTF-comparison.json"; COMPLETE="$BASE_OUT/OC-RAP-v48.82-PIPELINE_COMPLETE.json"
CACHE="${V4882_TENSOR_CACHE_DIR:-$BASE_OUT/.ocrap_v48_78_tensor_cache}"; mkdir -p "$BASE_OUT" "$CACHE"; rm -rf "$N_RUN" "$O_RUN"; rm -f "$RUNTIME" "$REF" "$TRAIN_IDX" "$TRAIN_SUM" "$EVAL_IDX" "$EVAL_SUM" "$AUDIT" "$COMPARE" "$COMPLETE"
python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF"
python tools/check_v48_82_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V81_COMPARE" "$L80_RUN" <<'PY'
import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text());d=p.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and d.get('status')=='SWITCH_INVERSE_TRUTH_STOP'):raise SystemExit('V48.81 STOP prerequisite missing')
if not pathlib.Path(sys.argv[2]).is_dir():raise SystemExit('missing L80 run')
PY
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"; DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"; TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
python tools/build_v48_80_interval_truth_index.py --root train_near="$TRAIN_NEAR" --root train_contact="$TRAIN_CONTACT" --root dev_near="$DEV_NEAR" --root dev_contact="$DEV_CONTACT" --output "$TRAIN_IDX" --summary "$TRAIN_SUM" --workers "${V4882_TRUTH_INDEX_WORKERS:-8}"
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false EVIDENCE_UNBOUNDED_HARM_FACTORS=false EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0 EVIDENCE_COMPONENT_SCALE=6.0 EVIDENCE_COMPONENT_HEADS=true EVIDENCE_COMPONENT_COUNT=5 EVIDENCE_DUAL_INTERACTION_BRIDGE=true EVIDENCE_ROCT_BENEFIT=true EVIDENCE_ROCT_DEPLOYABILITY=true EVIDENCE_ROCT_SCALE=3.0 EVIDENCE_ROCT_ALPHA=0.20 EVIDENCE_ROCT_BETA=0.20 EVIDENCE_ROCT_TOP_M=8 EVIDENCE_ROCT_OPTION_TEMPERATURE=0.35 EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050 EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true EVIDENCE_COMMON_MEASURE_ROOT_MASS=false EVIDENCE_ADMISSION_HEAD=false EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction PROPOSAL_TOP_K=5
export TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class EVAL_OPTION_EXECUTION_SEMANTICS=observation_class OPTION_EXECUTION_SEMANTICS=observation_class
train_one(){ local arm="$1" v="$2" gpu="$3" signed="$4" run="$5"; local src="$REFERENCE_A/candidates/$v" dst="$run/candidates/$v"; mkdir -p "$dst" "$run/logs"; RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" INIT_CKPT="$src/model_v48_trac_sr/best.pt" TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_structured_tail_field_weight STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_structured_tail_field_weight ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT=true SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=false SEMANTIC_WITNESS_ROUTE_ALIGNMENT=true SEMANTIC_WITNESS_REENTRY_ALIGNMENT=true SEMANTIC_WITNESS_CONTROL_PROJECTION=true SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false SEMANTIC_WITNESS_PROJECTION_FIDELITY=false SEMANTIC_WITNESS_ACTIVE_CONSTRAINT_TYPED_SOURCE=false SEMANTIC_WITNESS_ROOT_TAIL_SOURCE=true SEMANTIC_WITNESS_TAIL_LOCALIZATION=true SEMANTIC_WITNESS_STRUCTURED_TAIL_FIELD=true SEMANTIC_WITNESS_SIGNED_TAIL_CHANNELS="$signed" ABSOLUTE_FEASIBILITY_WEIGHT=1.0 ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT=structural_interval_bounds ABSOLUTE_FEASIBILITY_TRUTH_INDEX="$TRAIN_IDX" ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE=signed_margin_interval_huber BEST_METRIC=direct_absolute_signed_margin_interval_huber BEST_METRIC_MIN_DELTA=0.00001 EVIDENCE_ADAPT_EPOCHS="${V4882_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4882_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4882_LR:-0.001}" MAX_EVIDENCE_CALIBRATOR_PARAMS=384 PERSISTENT_TENSOR_CACHE=true PERSISTENT_TENSOR_CACHE_DIR="$CACHE" PERSISTENT_TENSOR_CACHE_BUILD_WORKERS="${PERSISTENT_TENSOR_CACHE_BUILD_WORKERS:-8}" SAVE_EVERY_EPOCH=false SAVE_LATEST=false OCRAP_ALGORITHM_VERSION="v48.82-OC-SNTF-$arm" bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$run/logs/${arm}_${v}.log" 2>&1; python tools/check_v48_82_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --truth-index "$TRAIN_IDX" --output "$dst/V48_82_STAGE_I_STATE_ISOLATION.json"; }
mkdir -p "$N_RUN/candidates" "$O_RUN/candidates"
# Pair the two causal arms on the two A30s for each reference variant.
for v in balanced precision; do set +e; train_one N82_SINGLE_FIELD "$v" "$GPU0" false "$N_RUN" & p0=$!; train_one O82_SIGNED_FIELD "$v" "$GPU1" true "$O_RUN" & p1=$!; wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e; [[ $r0 == 0 && $r1 == 0 ]] || exit 30; done
cal(){ local run="$1" tag="$2"; set +e; OUTPUTDIR="$run" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.82-$tag-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.82.1-OC-SNTF-ENGFIX-$tag" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$run/logs/v48_82_${tag}_certificate_controller.log" 2>&1; rc=$?; set -e; case $rc in 0|20);;*) exit 30;;esac; }
cal "$N_RUN" N82; cal "$O_RUN" O82
python tools/build_v48_80_interval_truth_index.py --root dev_near="$DEV_NEAR" --root dev_contact="$DEV_CONTACT" --root certificate_near="$CERT_NEAR" --root certificate_contact="$CERT_CONTACT" --output "$EVAL_IDX" --summary "$EVAL_SUM" --workers "${V4882_TRUTH_INDEX_WORKERS:-8}"
python tools/audit_v48_82_sntf.py --l80 "$L80_RUN" --n82 "$N_RUN" --o82 "$O_RUN" --truth-index "$EVAL_IDX" --truth-summary "$EVAL_SUM" --output "$AUDIT"
python tools/compare_v48_82_sntf.py --audit "$AUDIT" --v81-comparison "$V81_COMPARE" --output "$COMPARE"
python - "$COMPLETE" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);p.write_text(json.dumps({'schema':'ocrap-v48.82-sntf-pipeline-complete-v1','algorithm_version':'v48.82-DCP-DRFC-BCDE-RIFA-OC-SNTF','engineering_version':'v48.82.1-OC-SNTF-ENGFIX','valid':True,'attribution_ready':True,'errors':[],'arms':{'N82_SINGLE_FIELD':'ocrap_v48_82_dcp_drfc_bcde_rifa_sntf_single','O82_SIGNED_FIELD':'ocrap_v48_82_dcp_drfc_bcde_rifa_sntf_main'},'truth_contract':'structural_interval_bounds','boundary_transport':False,'dataset_reconstruction':False,'test_roots_read':False},indent=2,sort_keys=True)+'\n')
PY
cd "$BASE_OUT"; for r in "$N_RUN" "$O_RUN"; do b=$(basename "$r"); rm -f "$b.zip"; zip -qr "$b.zip" "$b"; done
zip -qj OC-RAP-v48.82-OC-SNTF-audits.zip "$REF" "$RUNTIME" "$TRAIN_SUM" "$EVAL_SUM" "$AUDIT" "$COMPARE" "$COMPLETE"
echo "V48.82 complete: upload $(basename "$N_RUN").zip + $(basename "$O_RUN").zip + OC-RAP-v48.82-OC-SNTF-audits.zip"
