#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
export CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}" CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}"
REFERENCE_A="${V4862_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
B_RUN="${V4862_NATIVE_B:-$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_native_B}"
C_RUN="${V4862_AFE_C:-$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_main}"
D_RUN="${V4862_ORFC_D:-$BASE_OUT/ocrap_v48_59_dcp_drfc_bcde_rifa_orfc_main}"
E_RUN="${V4862_CPHR_E:-$BASE_OUT/ocrap_v48_60_dcp_drfc_bcde_rifa_cphr_main}"
F_RUN="${V4862_ERWF_F:-$BASE_OUT/ocrap_v48_61_dcp_drfc_bcde_rifa_erwf_main}"
G_RUN="$BASE_OUT/ocrap_v48_62_dcp_drfc_bcde_rifa_occwrf_main"
V61_COMPLETE="${V4862_V61_COMPLETE:-$BASE_OUT/OC-RAP-v48.61-PIPELINE_COMPLETE.json}"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.62-reference-reuse-contract.json"
FEAS_AUDIT="$BASE_OUT/OC-RAP-v48.62-OC-CWRF-feasibility-role-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.62-DCP-DRFC-BCDE-RIFA-OC-CWRF-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.62-PIPELINE_COMPLETE.json"
export PERSISTENT_TENSOR_CACHE="${V4862_PERSISTENT_TENSOR_CACHE:-true}"
export PERSISTENT_TENSOR_CACHE_DIR="${V4862_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4862_occwrf}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"
rm -rf "$G_RUN"
rm -f "$REF_AUDIT" "$FEAS_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$G_RUN.zip" "$BASE_OUT/OC-RAP-v48.62-OC-CWRF-audits.zip"

# V48.62 is a single-axis absolute-source experiment after the engineering-valid
# V48.61 ERWF STOP.  A--F are reused.  G replaces ERWF's compensatory six-way
# linear sum by an observation-consistent common-witness factorization:
#   (i) finite-time non-compensatory recovery barrier per candidate x option;
#  (ii) frozen common-option support across observation-aliased latent roots;
# (iii) only two shared bounded gains for positive common-witness rescue and
#       negative physical-witness veto.  No regime ID/router, option bias,
# threshold search, proposal expansion, teacher future, root/margin retraining,
# centering or relative-ranking intervention is introduced.
bash scripts/prepare_v48_45_protocol.sh
python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python - "$V61_COMPLETE" "$B_RUN" "$C_RUN" "$D_RUN" "$E_RUN" "$F_RUN" <<'PY'
import json,pathlib,sys
p,b,c,d,e,f=map(pathlib.Path,sys.argv[1:])
if not p.is_file(): raise SystemExit(f'missing V48.61 prerequisite sentinel: {p}')
meta=json.loads(p.read_text())
if not (meta.get('valid') and meta.get('attribution_ready') and meta.get('engineering_version')=='v48.61.0-ERWF' and not meta.get('test_roots_read')):
    raise SystemExit(f'V48.61 prerequisite is not attribution-ready: {meta}')
for run in (b,c,d,e,f):
    if not run.is_dir(): raise SystemExit(f'missing prerequisite run: {run}')
PY
for f in evidence_adapt_teacher_pcd_index.jsonl evidence_adapt_teacher_pcd_index_summary.json \
         evidence_adapt_dev_teacher_pcd_index.jsonl evidence_adapt_dev_teacher_pcd_index_summary.json; do
  [[ -s "$REFERENCE_A/$f" ]] || { echo "missing reference teacher artifact: $REFERENCE_A/$f" >&2; exit 30; }
done

CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
for d in "$CERT_NEAR" "$CERT_CONTACT" "$DEV_NEAR" "$DEV_CONTACT" "$TRAIN_NEAR" "$TRAIN_CONTACT"; do
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing v48.62 protocol root: $d" >&2; exit 30; }
done
mkdir -p "$G_RUN/candidates" "$G_RUN/logs"

# Exact frozen V48.56-A Stage-I architecture/relative-evidence contract.
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false EVIDENCE_UNBOUNDED_HARM_FACTORS=false EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0
export EVIDENCE_COMPONENT_SCALE=6.0 EVIDENCE_COMPONENT_HEADS=true EVIDENCE_COMPONENT_COUNT=5
export EVIDENCE_DUAL_INTERACTION_BRIDGE=true EVIDENCE_FACTORIZED_HARM_INTERACTION=false EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false
export EVIDENCE_RANK_BENEFIT_SKIP=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=false EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=false
export EVIDENCE_ROCT_BENEFIT=true EVIDENCE_ROCT_DEPLOYABILITY=true EVIDENCE_ROCT_SCALE=3.0 EVIDENCE_ROCT_ALPHA=0.20 EVIDENCE_ROCT_BETA=0.20 EVIDENCE_ROCT_TOP_M=8 EVIDENCE_ROCT_OPTION_TEMPERATURE=0.35
export EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050
export EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=false EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true
export EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=false EVIDENCE_PHYSICAL_STUDENT_DRS=false
export EVIDENCE_DEP_BOUNDARY_ALIGNED=false EVIDENCE_GAP_ORDINAL_ONLY=false EVIDENCE_COMMON_MEASURE_ROOT_MASS=false
export EVIDENCE_ADMISSION_HEAD=false
export EVIDENCE_NATIVE_DRS_TOLERANCE=0.05 EVIDENCE_NATIVE_DEPLOYABILITY_TOLERANCE=0.05 EVIDENCE_NATIVE_GAP_TOLERANCE=0.05 EVIDENCE_NATIVE_POSITIVE_GAIN=0.015
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction PROPOSAL_TOP_K=5
export TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class EVAL_OPTION_EXECUTION_SEMANTICS=observation_class OPTION_EXECUTION_SEMANTICS=observation_class
export ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_MODE=raw ORDINAL_EVIDENCE_COMPONENT_MARGIN_TARGET_SCALE=0.10 ORDINAL_EVIDENCE_COMPONENT_MARGIN_CANONICAL_SCALES=""
export ORDINAL_EVIDENCE_COMPONENT_MARGIN_REGRESSION_RELIABILITY="1,1,1,0,0"

train_occwrf(){
  local v="$1"; local gpu="$2"; local src="$REFERENCE_A/candidates/$v"; local dst="$G_RUN/candidates/$v"
  mkdir -p "$dst"
  if [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]]; then set -a; source "$src/FACTOR_SUPPORT_CONTRACT.env"; set +a; else export EVIDENCE_COMPONENT_RELIABILITY="1,1,1,0,0"; fi
  RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" \
  INIT_CKPT="$src/model_v48_trac_sr/best.pt" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" \
  GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" \
  EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_common_witness_gain \
  STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_common_witness_gain \
  ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=false ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=false \
  ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION=false ABSOLUTE_COMMON_WITNESS_CORRECTION=true ABSOLUTE_FEASIBILITY_WEIGHT=1.0 \
  GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false \
  BEST_METRIC=direct_absolute_feasibility_bce BEST_METRIC_MIN_DELTA=0.00001 \
  EVIDENCE_ADAPT_EPOCHS="${V4862_OCCWRF_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4862_OCCWRF_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4862_OCCWRF_LR:-0.001}" \
  MAX_EVIDENCE_CALIBRATOR_PARAMS=2 OCRAP_ALGORITHM_VERSION="v48.62-DCP-DRFC-BCDE-RIFA-OC-CWRF" \
  bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$G_RUN/logs/occwrf_${v}.log" 2>&1
  grep -qx 'ABSOLUTE_FEASIBILITY_MODE=learned' "$dst/POLICY_CONTRACT.env" || { echo "OC-CWRF policy mode missing for $v" >&2; exit 30; }
  grep -qx 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' "$dst/POLICY_CONTRACT.env" || { echo "OC-CWRF fixed threshold missing for $v" >&2; exit 30; }
  grep -qx 'SELECTION_SEMANTICS=rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank' "$dst/POLICY_CONTRACT.env" || { echo "OC-CWRF selection contract mismatch for $v" >&2; exit 30; }
  python tools/check_v48_62_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --output "$dst/V48_62_STAGE_I_STATE_ISOLATION.json"
}

set +e
train_occwrf balanced "$GPU0" & p0=$!
train_occwrf precision "$GPU1" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?
set -e
[[ "$r0" == 0 && "$r1" == 0 ]] || { echo "v48.62 OC-CWRF training failed balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/check_v48_62_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --occwrf-run "$G_RUN" --output "$G_RUN/V48_62_VARIANT_ISOLATION.json"
python - "$G_RUN/V48_62_FACTOR_CONTRACT.json" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);p.write_text(json.dumps({
 'event':'v48_62_factor_contract','version':'v48.62-DCP-DRFC-BCDE-RIFA-OC-CWRF','arm':'G/Main',
 'stage_i':'bitwise v48.56-A checkpoint','stage_ii':'Observation-Consistent Common-Witness Recovery Field (OC-CWRF)',
 'source_intervention':'finite-time non-compensatory recovery barrier AND observation-consistent common-option support, applied before unchanged OC-MERO',
 'common_witness_features':['recovery_min_clearance_reserve','recovery_terminal_clearance_reserve','recovery_clearance_gain','recovery_terminal_stopping_reserve','recovery_control_envelope_reserve','recovery_min_stability_reserve','recovery_terminal_stability_reserve','recovery_stability_gain','recovery_clearance_floor_gain','recovery_stability_floor_gain'],
 'common_witness_feature_schema':1,'common_witness_feature_source':'observation_consistent_option_resolved_finite_time_recovery_witness',
 'physical_composition':'min hard constraints with max(invariance, finite-time recovery) for clearance/stability',
 'root_composition':'same-option relative support aggregated over observation-aliased root pairs; no root ID exposed',
 'recovery_controller':'existing deterministic rollout_recovery_controller','agent_prediction':'current observation constant velocity only',
 'trainable_state':'direct_absolute_common_witness_gain[2]','trainable_parameters':2,
 'initialization':'all zeros; execution-exact native B source at epoch 0','gain_constraint':'elementwise [0,2]; common-positive rescue and physical-negative veto only',
 'target':'1[R_dep_star(candidate) >= 0]','scope':'Near+Contact adaptation-train, one shared regime-agnostic execution mechanism','threshold':0.5,'threshold_search':False,
 'afe_head':False,'orfc_option_bias':False,'cphr_candidate_scalar':False,'erwf_compensatory_linear_sum':False,'centering':False,'proposal_top_k':5,'proposal_expansion':False,
 'root_retraining':False,'margin_head_retraining':False,'relative_score_intervention':False,'teacher_margin_distillation':False,'teacher_future_input':False,
 'regime_id_input':False,'strategy_regime_conditioning':False,'teacher_semantics_changed':False,'test_roots_read':False,'created_unix':time.time()
},indent=2,sort_keys=True)+'\n')
PY

run_calibration(){
  set +e
  OUTPUTDIR="$G_RUN" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.62-G-OC-CWRF-$(date +%s)" \
  OCRAP_IMPLEMENTATION_VERSION="v48.62.0-OC-CWRF" \
  CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
  ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class \
  HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 \
  bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$G_RUN/logs/v48_62_certificate_controller.log" 2>&1
  local rc=$?; set -e
  case "$rc" in 0|20) echo "OC-CWRF calibration valid evidence RC=$rc";; *) echo "OC-CWRF calibration engineering failure RC=$rc" >&2; return 30;; esac
}
run_calibration

python tools/audit_v48_62_feasibility_role.py \
  --arm "A=$REFERENCE_A" --arm "B_native=$B_RUN" --arm "C_AFE=$C_RUN" --arm "D_ORFC=$D_RUN" --arm "E_CPHR=$E_RUN" --arm "F_ERWF=$F_RUN" --arm "G_OCCWRF=$G_RUN" --output "$FEAS_AUDIT"
python tools/compare_v48_62_occwrf.py --a "$REFERENCE_A" --b "$B_RUN" --c "$C_RUN" --d "$D_RUN" --e "$E_RUN" --f "$F_RUN" --g "$G_RUN" --feasibility-audit "$FEAS_AUDIT" --output "$COMPARE"
python tools/check_v48_62_pipeline_complete.py --reference-contract "$REF_AUDIT" --v61-complete "$V61_COMPLETE" --occwrf-run "$G_RUN" --feasibility-audit "$FEAS_AUDIT" --comparison "$COMPARE" --output "$PIPELINE_COMPLETE"

cd "$BASE_OUT"; b="$(basename "$G_RUN")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"
zip -qj OC-RAP-v48.62-OC-CWRF-audits.zip "$REF_AUDIT" "$FEAS_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V61_COMPLETE" \
  "$G_RUN/V48_62_VARIANT_ISOLATION.json" "$G_RUN/V48_62_FACTOR_CONTRACT.json" \
  "$G_RUN/candidates/balanced/V48_62_STAGE_I_STATE_ISOLATION.json" "$G_RUN/candidates/precision/V48_62_STAGE_I_STATE_ISOLATION.json"
echo "v48.62 complete. Upload $b.zip + OC-RAP-v48.62-OC-CWRF-audits.zip"
