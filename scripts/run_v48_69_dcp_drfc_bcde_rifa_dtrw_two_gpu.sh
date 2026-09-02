#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
REFERENCE_A="${V4869_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
B_RUN="${V4869_NATIVE_B:-$BASE_OUT/ocrap_v48_58_dcp_drfc_bcde_rifa_native_B}"
F_RUN="${V4869_ERWF_F:-$BASE_OUT/ocrap_v48_61_dcp_drfc_bcde_rifa_erwf_main}"
P66_RUN="${V4869_P66:-$BASE_OUT/ocrap_v48_66_dcp_drfc_bcde_rifa_acrw_main}"
Q67_RUN="${V4869_Q67:-$BASE_OUT/ocrap_v48_67_dcp_drfc_bcde_rifa_pbrw_control}"
T68_RUN="${V4869_T68:-$BASE_OUT/ocrap_v48_68_dcp_drfc_bcde_rifa_rtrw_fidelity}"
D_RUN="$BASE_OUT/ocrap_v48_69_dcp_drfc_bcde_rifa_dtrw_main"
V68_COMPLETE="${V4869_V68_COMPLETE:-$BASE_OUT/OC-RAP-v48.68-PIPELINE_COMPLETE.json}"
V68_COMPARE="${V4869_V68_COMPARE:-$BASE_OUT/OC-RAP-v48.68-DCP-DRFC-BCDE-RIFA-OC-RTRW-comparison.json}"
REF_AUDIT="$BASE_OUT/OC-RAP-v48.69-reference-reuse-contract.json"
FEAS_AUDIT="$BASE_OUT/OC-RAP-v48.69-OC-DTRW-feasibility-role-audit.json"
DEMAND_AUDIT="$BASE_OUT/OC-RAP-v48.69-OC-DTRW-demand-trust-audit.json"
TRUTH_AUDIT="$BASE_OUT/OC-RAP-v48.69-OC-DTRW-truth-debt-audit.json"
COMPARE="$BASE_OUT/OC-RAP-v48.69-DCP-DRFC-BCDE-RIFA-OC-DTRW-comparison.json"
PIPELINE_COMPLETE="$BASE_OUT/OC-RAP-v48.69-PIPELINE_COMPLETE.json"
export PERSISTENT_TENSOR_CACHE="${V4869_PERSISTENT_TENSOR_CACHE:-true}"
export PERSISTENT_TENSOR_CACHE_DIR="${V4869_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4869_dtrw}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"
rm -rf "$D_RUN"
rm -f "$REF_AUDIT" "$FEAS_AUDIT" "$DEMAND_AUDIT" "$TRUTH_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$D_RUN.zip" "$BASE_OUT/OC-RAP-v48.69-OC-DTRW-audits.zip"

# V48.69 is preregistered from the attribution-ready V48.68 STOP branch:
#   - projection-fidelity soft trust (T68) is retained: it improved AUC 8/8 and selectivity without changing certificate sign;
#   - the CV/current-acceleration hard occupancy min is rejected and frozen OFF;
#   - v48.67 boundary transport remains OFF until witness trust/admission is source-valid.
# D69 tests one causal hypothesis only: absolute projection severity over-penalizes urgent but valid recoveries.
# The penalty is tempered by observation-derived active recovery demand reconstructed from existing signed witness coordinates.
# No regime input/router, threshold/LR/horizon sweep, teacher future, new option library, Stage-I/relative-ranker retraining,
# class-local/path-stop transport, robust-occupancy hard min, or boundary transport is permitted.

python tools/check_v48_58_reference_reuse.py --reference-run "$REFERENCE_A" --output "$REF_AUDIT"
python - "$V68_COMPLETE" "$V68_COMPARE" "$REFERENCE_A" "$B_RUN" "$F_RUN" "$P66_RUN" "$Q67_RUN" "$T68_RUN" <<'PY2'
import json,pathlib,sys
complete=pathlib.Path(sys.argv[1]); compare=pathlib.Path(sys.argv[2]); runs=[pathlib.Path(x) for x in sys.argv[3:]]
if not complete.is_file(): raise SystemExit(f'missing V48.68 prerequisite sentinel: {complete}')
meta=json.loads(complete.read_text())
if not (meta.get('valid') and meta.get('attribution_ready') and meta.get('engineering_version')=='v48.68.0-OC-RTRW' and not meta.get('test_roots_read')):
    raise SystemExit(f'V48.68 prerequisite is not attribution-ready: {meta}')
if not compare.is_file(): raise SystemExit(f'missing V48.68 comparison: {compare}')
pr=(json.loads(compare.read_text()).get('preregistered_decision') or {})
if not (pr.get('status')=='STOP' and pr.get('projection_fidelity_mechanism_gate') is True and pr.get('robust_occupancy_mechanism_gate') is False):
    raise SystemExit(f'V48.68 branch contract mismatch: {pr}')
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
  [[ -d "$d" && -f "$d/manifest.csv" ]] || { echo "missing v48.69 protocol root: $d" >&2; exit 30; }
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

mkdir -p "$D_RUN/candidates" "$D_RUN/logs"
train_one() {
  local v="$1"; local gpu="$2"; local src="$REFERENCE_A/candidates/$v"; local dst="$D_RUN/candidates/$v"
  mkdir -p "$dst"
  if [[ -f "$src/FACTOR_SUPPORT_CONTRACT.env" ]]; then set -a; source "$src/FACTOR_SUPPORT_CONTRACT.env"; set +a; else export EVIDENCE_COMPONENT_RELIABILITY="1,1,1,0,0"; fi
  RUN="$dst" MODEL_DIR="$dst/model_v48_trac_sr" VARIANT="$v" TRAIN_GPU="$gpu" INIT_CKPT="$src/model_v48_trac_sr/best.pt" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl" VAL_GROUP_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl" \
  EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_semantic_witness_gain STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_semantic_witness_gain \
  ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=false ABSOLUTE_PHYSICAL_HEADROOM_CORRECTION=false ABSOLUTE_EXECUTABLE_WITNESS_CORRECTION=false ABSOLUTE_COMMON_WITNESS_CORRECTION=false ABSOLUTE_QUANTIFIER_WITNESS_CORRECTION=false \
  ABSOLUTE_SEMANTIC_WITNESS_CORRECTION=true SEMANTIC_WITNESS_ACTIVE_SET_ALIGNMENT=true SEMANTIC_WITNESS_PATH_STOP_ALIGNMENT=false SEMANTIC_WITNESS_CLASSLOCAL_TRANSPORT=false \
  SEMANTIC_WITNESS_ROUTE_ALIGNMENT=true SEMANTIC_WITNESS_REENTRY_ALIGNMENT=true SEMANTIC_WITNESS_CONTROL_PROJECTION=true SEMANTIC_WITNESS_BOUNDARY_TRANSPORT=false \
  SEMANTIC_WITNESS_PROJECTION_FIDELITY=true SEMANTIC_WITNESS_DEMAND_NORMALIZED_FIDELITY=true SEMANTIC_WITNESS_ROBUST_OCCUPANCY=false ABSOLUTE_FEASIBILITY_WEIGHT=1.0 \
  GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false BEST_METRIC=direct_absolute_feasibility_bce BEST_METRIC_MIN_DELTA=0.00001 \
  EVIDENCE_ADAPT_EPOCHS="${V4869_DTRW_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${V4869_DTRW_PATIENCE:-5}" EVIDENCE_ADAPT_LR="${V4869_DTRW_LR:-0.001}" \
  MAX_EVIDENCE_CALIBRATOR_PARAMS=2 OCRAP_ALGORITHM_VERSION="v48.69-DCP-DRFC-BCDE-RIFA-OC-DTRW-Main" \
  bash scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh >"$D_RUN/logs/dtrw_${v}.log" 2>&1
  grep -qx 'ABSOLUTE_FEASIBILITY_MODE=learned' "$dst/POLICY_CONTRACT.env" || exit 30
  grep -qx 'ABSOLUTE_FEASIBILITY_THRESHOLD=0.5' "$dst/POLICY_CONTRACT.env" || exit 30
  grep -qx 'SELECTION_SEMANTICS=rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank' "$dst/POLICY_CONTRACT.env" || exit 30
  python tools/check_v48_69_state_isolation.py --reference "$src/model_v48_trac_sr/best.pt" --adapted "$dst/model_v48_trac_sr/best.pt" --output "$dst/V48_69_STAGE_I_STATE_ISOLATION.json"
}
set +e
train_one balanced "$GPU0" & p0=$!; train_one precision "$GPU1" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ "$r0" == 0 && "$r1" == 0 ]] || { echo "v48.69 D69 training failed balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/check_v48_69_variant_isolation.py --reference-run "$REFERENCE_A" --reference-contract "$REF_AUDIT" --dtrw-run "$D_RUN" --output "$D_RUN/V48_69_VARIANT_ISOLATION.json"
python - "$D_RUN/V48_69_FACTOR_CONTRACT.json" <<'PY2'
import json,pathlib,time,sys
p=pathlib.Path(sys.argv[1])
d={'event':'v48_69_factor_contract','version':'v48.69-DCP-DRFC-BCDE-RIFA-OC-DTRW','arm':'D69_Main_OCDTRW','stage_i':'bitwise v48.56-A checkpoint','stage_ii':'Observation-Consistent Demand-Tempered Recovery Witness (OC-DTRW)',
'source_intervention':'retain v48.67 actuator projection and v48.68 validated projection-fidelity trust, but normalize the fidelity penalty by observation-derived active recovery demand; v48.68 robust occupancy hard-min and v48.67 boundary transport remain OFF',
'semantic_witness_feature_schema':5,'semantic_witness_feature_source':'demand_tempered_projected_recovery_witness','active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,'route_alignment':True,'reentry_alignment':True,'control_projection':True,'boundary_transport':False,'projection_fidelity':True,'demand_normalized_fidelity':True,'robust_occupancy':False,
'correction_locus':'candidate-global same-option common support before native OC-MERO aggregation','positive_logic':'actuator-projected positive witness with projection-fidelity trust f=(1+demand)/(1+demand+raw_projection_violation); demand=max(observed clearance deficit, observed active stability deficit)','negative_logic':'frozen universal-failure correction; no per-option negative veto','physical_composition':'non-compensatory CV clearance / legacy stopping / observation-active stability / route / persistent re-entry; actuator feasibility enforced by construction','recovery_controller':'v48.67 magnitude/rate/jerk projected deterministic recovery; raw desired-command violation is soft trust only','agent_prediction':'v48.67 observation-only CV; rejected v48.68 CV/current-acceleration hard min is OFF','boundary_transport_rule':'OFF until witness trust/admission source GO','trainable_state':'direct_absolute_semantic_witness_gain[2]','trainable_parameters':2,'initialization':'all zeros; execution-exact native B at epoch 0','gain_constraint':'elementwise [0,2]','target':'1[R_dep_star(candidate) >= 0]','scope':'Near+Contact adaptation-train; one shared regime-agnostic mechanism','threshold':0.5,'threshold_search':False,'afe_head':False,'proposal_top_k':5,'proposal_expansion':False,'root_retraining':False,'margin_head_retraining':False,'relative_score_intervention':False,'teacher_margin_distillation':False,'teacher_future_input':False,'regime_id_input':False,'strategy_regime_conditioning':False,'teacher_semantics_changed':False,'dataset_reconstruction':False,'test_roots_read':False,'created_unix':time.time()}
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY2

set +e
OUTPUTDIR="$D_RUN" GPU0="$GPU0" GPU1="$GPU1" V4836_ATTEMPT_ID="v48.69-D69-$(date +%s)" OCRAP_IMPLEMENTATION_VERSION="v48.69.1-OC-DTRW-ENGFIX-Main" \
CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
ABSOLUTE_FEASIBILITY_MODE=learned ABSOLUTE_FEASIBILITY_THRESHOLD=0.5 OPTION_EXECUTION_SEMANTICS=observation_class HARM_LABEL_MODE=component_veto OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit PROPOSAL_TOP_K=5 \
bash scripts/calibrate_v48_36_shared_certificate_pool.sh >"$D_RUN/logs/v48_69_D69_certificate_controller.log" 2>&1
rc=$?; set -e
case "$rc" in 0|20) echo "D69 calibration valid evidence RC=$rc";; *) echo "D69 calibration engineering failure RC=$rc" >&2; exit 30;; esac

python tools/audit_v48_69_feasibility_role.py --arm "B_native=$B_RUN" --arm "F_ERWF=$F_RUN" --arm "P66_OCACRW=$P66_RUN" --arm "Q67_CTRLPROJ=$Q67_RUN" --arm "T68_FIDELITY=$T68_RUN" --arm "D69_DTRW=$D_RUN" --output "$FEAS_AUDIT"
python tools/audit_v48_69_demand_trust.py --t68 "$T68_RUN" --d69 "$D_RUN" --output "$DEMAND_AUDIT"
python tools/audit_v48_69_truth_debt.py --run "$D_RUN" --output "$TRUTH_AUDIT"
python tools/compare_v48_69_dtrw.py --feasibility-audit "$FEAS_AUDIT" --demand-audit "$DEMAND_AUDIT" --truth-audit "$TRUTH_AUDIT" --v68-complete "$V68_COMPLETE" --v68-comparison "$V68_COMPARE" --output "$COMPARE"
python tools/check_v48_69_pipeline_complete.py --reference-contract "$REF_AUDIT" --v68-complete "$V68_COMPLETE" --v68-comparison "$V68_COMPARE" --dtrw-run "$D_RUN" --feasibility-audit "$FEAS_AUDIT" --demand-audit "$DEMAND_AUDIT" --truth-audit "$TRUTH_AUDIT" --comparison "$COMPARE" --output "$PIPELINE_COMPLETE"

cd "$BASE_OUT"; b="$(basename "$D_RUN")"; rm -f "$b.zip"; zip -qr "$b.zip" "$b"
zip -qj OC-RAP-v48.69-OC-DTRW-audits.zip "$REF_AUDIT" "$FEAS_AUDIT" "$DEMAND_AUDIT" "$TRUTH_AUDIT" "$COMPARE" "$PIPELINE_COMPLETE" "$V68_COMPLETE" "$V68_COMPARE" \
  "$D_RUN/V48_69_VARIANT_ISOLATION.json" "$D_RUN/V48_69_FACTOR_CONTRACT.json" \
  "$D_RUN/candidates/balanced/V48_69_STAGE_I_STATE_ISOLATION.json" "$D_RUN/candidates/precision/V48_69_STAGE_I_STATE_ISOLATION.json"
echo "v48.69 complete. Upload $b.zip + OC-RAP-v48.69-OC-DTRW-audits.zip"
