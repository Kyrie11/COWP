#!/usr/bin/env bash
set -euo pipefail

export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export SAFE_TEST=${SAFE_TEST:-$OCRAP_ROOT/test_safe}
export NEAR_TEST=${NEAR_TEST:-$OCRAP_ROOT/test_near_contact}
export CONTACT_TEST=${CONTACT_TEST:-$OCRAP_ROOT/test_contact}
export WOMD_VAL=${WOMD_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord}
export WOMD_VAL_INTERACTIVE=${WOMD_VAL_INTERACTIVE:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}

# v48 OC-TRAC-SR evaluation.  The shared observation encoder is fine-tuned
# together with observation-conditioned recovery-value experts.  Deployment
# uses only the calibrated policy-level direct-value certificate; historical
# hand-written rescue certificates are retained as disabled ablations.  Safe
# remains nominal-locked.
export RUN=${RUN:-runs/ocrap_v48_trac_sr_eval}
export BASE_RUN=${BASE_RUN:-}
export SAFE_NOMINAL_ONLY=${SAFE_NOMINAL_ONLY:-0}
if [[ -z "${CKPT:-}" ]]; then
  [[ -n "$BASE_RUN" ]] || { echo "BASE_RUN or CKPT must be set explicitly; refusing legacy fallback" >&2; exit 2; }
  CKPT="$BASE_RUN/model_v48_trac_sr/best.pt"
fi
# Safe paired non-inferiority is nominal locked and does not consume a stress
# certificate.  Requiring gamma/calibration here made the independent Safe
# experiment fail whenever Near/Contact Natural gate had not yet produced those
# files.  Stress runs still require both artifacts below.
if [[ "$SAFE_NOMINAL_ONLY" != "1" ]]; then
  if [[ -z "${CAL:-}" ]]; then
    [[ -n "$BASE_RUN" ]] || { echo "CAL or BASE_RUN must be set explicitly" >&2; exit 2; }
    CAL="$BASE_RUN/calibration/calibration_mix_v48.json"
  fi
  if [[ -z "${GAMMA:-}" ]]; then
    [[ -n "$BASE_RUN" ]] || { echo "GAMMA or BASE_RUN must be set explicitly" >&2; exit 2; }
    GAMMA="$BASE_RUN/calibration/gamma_rec_by_bucket_v48.json"
  fi
else
  CAL="${CAL:-}"
  GAMMA="${GAMMA:-}"
fi
export CKPT CAL GAMMA
mkdir -p "$RUN"

# Exact closed-loop persistence controls. Reusing the same RUN/output filenames
# resumes completed scene/target rollouts from *.json.partial or *.scenes.jsonl.
export CL_RESUME=${CL_RESUME:-true}
export CL_PARTIAL_EVERY=${CL_PARTIAL_EVERY:-4}
export CL_RESUME_FSYNC=${CL_RESUME_FSYNC:-false}

[[ -f "$CKPT" ]] || { echo "missing checkpoint $CKPT; set CKPT=/path/to/best.pt" >&2; exit 2; }
if [[ "$SAFE_NOMINAL_ONLY" != "1" ]]; then
  [[ -f "$GAMMA" ]] || { echo "missing gamma map $GAMMA; set GAMMA=/path/to/gamma.json" >&2; exit 2; }
  [[ -f "$CAL" ]] || { echo "missing calibration $CAL; set CAL=/path/to/calibration.json" >&2; exit 2; }
fi

# v48.24 reads the complete calibrated selector contract.  Earlier versions
# loaded only score/opportunity/harm thresholds and silently left runtime at the
# defaults proposal_top_k=1 and evidence_rerank_top_k=false, even though training
# and calibration used top-k reranking.  That was a deployment-contract mismatch.
DEV_SHADOW_DIAGNOSTIC="${DEV_SHADOW_DIAGNOSTIC:-0}"
read_direct_rule() {
  local path="$1"
  python - "$path" "$DEV_SHADOW_DIAGNOSTIC" <<'PYQ'
import json, math, sys
p=sys.argv[1]
diagnostic=str(sys.argv[2]).strip().lower() in {'1','true','yes','on'}
d=json.load(open(p))
source='deployment'
if d.get('valid_for_deployment', False):
    r=d.get('selector_overrides', {}) or {}
elif diagnostic:
    source='fit_nearest_frontier_diagnostic_only'
    r=d.get('diagnostic_selector_overrides', {}) or {}
    if (not r and d.get('verification_only')
            and (d.get('frozen_rule_source') or {}).get('sha256')):
        # v48.25: the selector was frozen on adaptation-dev before the complete
        # certificate was read. It remains safe for adaptation-dev-only shadow
        # diagnosis even when the independent certificate rejects deployment.
        source='adaptation_dev_frozen_rule_diagnostic_only'
        r=d.get('selector_overrides', {}) or {}
    if not r:
        frontier=d.get('near_miss_frontier') or []
        if not frontier:
            raise SystemExit(f'no fit-derived diagnostic rule in certificate: {p}')
        f=frontier[0]
        r={
          'direct_value_min_advantage_lcb':f.get('score_threshold'),
          'direct_value_opportunity_threshold':f.get('opportunity_threshold'),
          'direct_value_harm_threshold':f.get('harm_threshold'),
          'direct_value_min_rank_margin':f.get('rank_margin_threshold',0.0),
          'direct_value_proposal_top_k':d.get('proposal_top_k',1),
          'direct_value_evidence_rerank_top_k':bool((d.get('constraints') or {}).get('evidence_rerank_top_k',False)),
          'direct_value_conditional_rank_margin':bool((d.get('constraints') or {}).get('conditional_recovery_ranking',False)),
        }
else:
    raise SystemExit(f'invalid OC-TRAC-SR deployment certificate: {p}')

def finite(name, default):
    value=float(r.get(name, default))
    if not math.isfinite(value):
        raise SystemExit(f'non-finite {name} in {p}')
    return value
s=finite('direct_value_min_advantage_lcb', float('inf'))
o=finite('direct_value_opportunity_threshold', float('inf'))
h=finite('direct_value_harm_threshold', float('-inf'))
m=finite('direct_value_min_rank_margin', 0.0)
k=int(r.get('direct_value_proposal_top_k', r.get('proposal_top_k', 1)))
rerank=bool(r.get('direct_value_evidence_rerank_top_k', r.get('evidence_rerank_top_k', False)))
conditional=bool(r.get('direct_value_conditional_rank_margin', False))
if k < 1:
    raise SystemExit(f'invalid direct_value_proposal_top_k={k} in {p}')
print(s, o, h, m, k, str(rerank).lower(), str(conditional).lower(), source)
PYQ
}
if [[ "$SAFE_NOMINAL_ONLY" == "1" ]]; then
  # Safe is nominal-locked by the selector. Stress certificates are irrelevant
  # for this paired non-inferiority probe, so do not require a failed Near/Contact gate.
  export RUN_DIRECT_VALUE=false
  NEAR_DIRECT_THRESHOLD=1000000000
  NEAR_DIRECT_OPPORTUNITY_THRESHOLD=1.0
  NEAR_DIRECT_HARM_THRESHOLD=0.0
  NEAR_DIRECT_MIN_RANK_MARGIN=0.0
  NEAR_DIRECT_PROPOSAL_TOP_K=1
  NEAR_DIRECT_EVIDENCE_RERANK=false
  NEAR_DIRECT_CONDITIONAL_RANK_MARGIN=false
  NEAR_DIRECT_RULE_SOURCE=safe_nominal_lock
  CONTACT_DIRECT_THRESHOLD=1000000000
  CONTACT_DIRECT_OPPORTUNITY_THRESHOLD=1.0
  CONTACT_DIRECT_HARM_THRESHOLD=0.0
  CONTACT_DIRECT_MIN_RANK_MARGIN=0.0
  CONTACT_DIRECT_PROPOSAL_TOP_K=1
  CONTACT_DIRECT_EVIDENCE_RERANK=false
  CONTACT_DIRECT_CONDITIONAL_RANK_MARGIN=false
  CONTACT_DIRECT_RULE_SOURCE=safe_nominal_lock
else
  [[ -n "$BASE_RUN" ]] || { echo "BASE_RUN is required for certificate loading" >&2; exit 2; }
  if [[ -z "${NEAR_DIRECT_THRESHOLD:-}" || -z "${NEAR_DIRECT_OPPORTUNITY_THRESHOLD:-}" || -z "${NEAR_DIRECT_HARM_THRESHOLD:-}" ]]; then
    read -r NEAR_DIRECT_THRESHOLD NEAR_DIRECT_OPPORTUNITY_THRESHOLD NEAR_DIRECT_HARM_THRESHOLD \
      NEAR_DIRECT_MIN_RANK_MARGIN NEAR_DIRECT_PROPOSAL_TOP_K NEAR_DIRECT_EVIDENCE_RERANK \
      NEAR_DIRECT_CONDITIONAL_RANK_MARGIN NEAR_DIRECT_RULE_SOURCE \
      < <(read_direct_rule "$BASE_RUN/calibration/direct_value_risk_near_v48.json")
  fi
  if [[ -z "${CONTACT_DIRECT_THRESHOLD:-}" || -z "${CONTACT_DIRECT_OPPORTUNITY_THRESHOLD:-}" || -z "${CONTACT_DIRECT_HARM_THRESHOLD:-}" ]]; then
    read -r CONTACT_DIRECT_THRESHOLD CONTACT_DIRECT_OPPORTUNITY_THRESHOLD CONTACT_DIRECT_HARM_THRESHOLD \
      CONTACT_DIRECT_MIN_RANK_MARGIN CONTACT_DIRECT_PROPOSAL_TOP_K CONTACT_DIRECT_EVIDENCE_RERANK \
      CONTACT_DIRECT_CONDITIONAL_RANK_MARGIN CONTACT_DIRECT_RULE_SOURCE \
      < <(read_direct_rule "$BASE_RUN/calibration/direct_value_risk_contact_v48.json")
  fi
fi
# Manual threshold overrides remain supported; fill the rest of the selector
# contract explicitly instead of inheriting silent top-1/non-rerank defaults.
NEAR_DIRECT_MIN_RANK_MARGIN="${NEAR_DIRECT_MIN_RANK_MARGIN:-0.0}"
CONTACT_DIRECT_MIN_RANK_MARGIN="${CONTACT_DIRECT_MIN_RANK_MARGIN:-0.0}"
NEAR_DIRECT_PROPOSAL_TOP_K="${NEAR_DIRECT_PROPOSAL_TOP_K:-1}"
CONTACT_DIRECT_PROPOSAL_TOP_K="${CONTACT_DIRECT_PROPOSAL_TOP_K:-1}"
NEAR_DIRECT_EVIDENCE_RERANK="${NEAR_DIRECT_EVIDENCE_RERANK:-false}"
CONTACT_DIRECT_EVIDENCE_RERANK="${CONTACT_DIRECT_EVIDENCE_RERANK:-false}"
NEAR_DIRECT_CONDITIONAL_RANK_MARGIN="${NEAR_DIRECT_CONDITIONAL_RANK_MARGIN:-false}"
CONTACT_DIRECT_CONDITIONAL_RANK_MARGIN="${CONTACT_DIRECT_CONDITIONAL_RANK_MARGIN:-false}"
export NEAR_DIRECT_THRESHOLD CONTACT_DIRECT_THRESHOLD \
  NEAR_DIRECT_OPPORTUNITY_THRESHOLD CONTACT_DIRECT_OPPORTUNITY_THRESHOLD \
  NEAR_DIRECT_HARM_THRESHOLD CONTACT_DIRECT_HARM_THRESHOLD \
  NEAR_DIRECT_MIN_RANK_MARGIN CONTACT_DIRECT_MIN_RANK_MARGIN \
  NEAR_DIRECT_PROPOSAL_TOP_K CONTACT_DIRECT_PROPOSAL_TOP_K \
  NEAR_DIRECT_EVIDENCE_RERANK CONTACT_DIRECT_EVIDENCE_RERANK \
  NEAR_DIRECT_CONDITIONAL_RANK_MARGIN CONTACT_DIRECT_CONDITIONAL_RANK_MARGIN

echo "direct-value OC-TRAC-SR rules: near(source=${NEAR_DIRECT_RULE_SOURCE:-manual},score=$NEAR_DIRECT_THRESHOLD,opp=$NEAR_DIRECT_OPPORTUNITY_THRESHOLD,harm=$NEAR_DIRECT_HARM_THRESHOLD,rank_margin=$NEAR_DIRECT_MIN_RANK_MARGIN,k=$NEAR_DIRECT_PROPOSAL_TOP_K,rerank=$NEAR_DIRECT_EVIDENCE_RERANK) contact(source=${CONTACT_DIRECT_RULE_SOURCE:-manual},score=$CONTACT_DIRECT_THRESHOLD,opp=$CONTACT_DIRECT_OPPORTUNITY_THRESHOLD,harm=$CONTACT_DIRECT_HARM_THRESHOLD,rank_margin=$CONTACT_DIRECT_MIN_RANK_MARGIN,k=$CONTACT_DIRECT_PROPOSAL_TOP_K,rerank=$CONTACT_DIRECT_EVIDENCE_RERANK)"

make_sel() {
  local tag="$1"  # scalar | v48
  local rel_near=false
  local rel_contact=false
  local rec_pool=false
  local brake_rescue=false
  local pcd_rescue=false
  local brake_tail=false
  local brake_tail_bypass_pcd=false
  local direct_value=false
  local is_v27=false
  case "$tag" in
    scalar) rel_near=false; rel_contact=false; rec_pool=false; brake_rescue=false; pcd_rescue=false; brake_tail=false; brake_tail_bypass_pcd=false; direct_value=false ;;
    v48) rel_near=false; rel_contact=false; rec_pool=false; brake_rescue=false; pcd_rescue=false; brake_tail=false; brake_tail_bypass_pcd=false; direct_value=${RUN_DIRECT_VALUE:-true}; is_v27=false ;;
    v40|v27) rel_near=true; rel_contact=true; rec_pool=true; brake_rescue=${CONTACT_BRAKE_RESCUE:-false}; pcd_rescue=true; brake_tail=${CONTACT_BRAKE_TAIL:-false}; brake_tail_bypass_pcd=${CONTACT_TAIL_BYPASS_PCD_GAIN:-false}; direct_value=${RUN_DIRECT_VALUE:-true}; is_v27=true ;;
    *) echo "unknown selector tag: $tag" >&2; exit 2 ;;
  esac

  COMMON_SEL=(
    --set seed=${SEED:-7}
    --set selection.ocrap_selector=calibrated_constrained
    --set selection.auto_regime_from_observation=${AUTO_REGIME_FROM_OBSERVATION:-true}
    --set regime_thresholds.tau_d=${AUTO_REGIME_TAU_D:-2.0}
    --set regime_thresholds.tau_ttc=${AUTO_REGIME_TAU_TTC:-3.0}
    --set regime_thresholds.tau_contact=${AUTO_REGIME_TAU_CONTACT:-0.8}

    # Paper invariant: never execute unadmitted recovery, and never treat
    # exploration/coverage prefixes as recoveries.
    --set selection.require_admitted_intervention_by_bucket.safe=true
    --set selection.require_admitted_intervention_by_bucket.test_safe=true
    --set selection.require_admitted_intervention_by_bucket.near_contact=true
    --set selection.require_admitted_intervention_by_bucket.test_near_contact=true
    --set selection.require_admitted_intervention_by_bucket.contact=true
    --set selection.require_admitted_intervention_by_bucket.test_contact=true
    --set selection.unadmitted_fallback_to_nominal=true
    --set selection.intervention_macro_blocklist_by_bucket.safe=keep,brake,yield,lane_shift,merge,pull_over,stabilize,perturb_nominal
    --set selection.intervention_macro_blocklist_by_bucket.test_safe=keep,brake,yield,lane_shift,merge,pull_over,stabilize,perturb_nominal
    --set selection.intervention_macro_allowlist_by_bucket.near_contact=brake,stabilize,yield,merge,pull_over
    --set selection.intervention_macro_allowlist_by_bucket.test_near_contact=brake,stabilize,yield,merge,pull_over
    --set selection.intervention_macro_allowlist_by_bucket.contact=brake,stabilize,yield,merge,pull_over
    --set selection.intervention_macro_allowlist_by_bucket.test_contact=brake,stabilize,yield,merge,pull_over
    --set selection.intervention_macro_blocklist_by_bucket.near_contact=nominal,keep,perturb_nominal,lane_shift
    --set selection.intervention_macro_blocklist_by_bucket.test_near_contact=nominal,keep,perturb_nominal,lane_shift
    --set selection.intervention_macro_blocklist_by_bucket.contact=nominal,keep,perturb_nominal,lane_shift
    --set selection.intervention_macro_blocklist_by_bucket.test_contact=nominal,keep,perturb_nominal,lane_shift
    --set selection.intervention_require_macro_by_bucket.near_contact=true
    --set selection.intervention_require_macro_by_bucket.test_near_contact=true
    --set selection.intervention_require_macro_by_bucket.contact=true
    --set selection.intervention_require_macro_by_bucket.test_contact=true

    # Safe regime remains nominal-preserving.
    --set selection.safe_force_nominal_when_feasible_by_bucket.safe=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.test_safe=true
    --set selection.safe_force_nominal_mode_by_bucket.safe=feasible
    --set selection.safe_force_nominal_mode_by_bucket.test_safe=feasible

    # Keep old option shortcut off.
    --set selection.option_drs_certificate_by_bucket.safe=false
    --set selection.option_drs_certificate_by_bucket.test_safe=false
    --set selection.option_drs_certificate_by_bucket.near_contact=false
    --set selection.option_drs_certificate_by_bucket.test_near_contact=false
    --set selection.option_drs_certificate_by_bucket.contact=false
    --set selection.option_drs_certificate_by_bucket.test_contact=false

    # Relative Pareto channel: near contact permits semantic recovery macros;
    # contact uses it only for yield/merge.  Brake is routed through the v24
    # macro-stratified brake-rescue certificate below.
    --set selection.relative_recovery_certificate_by_bucket.safe=false
    --set selection.relative_recovery_certificate_by_bucket.test_safe=false
    --set selection.relative_recovery_certificate_by_bucket.near_contact="$rel_near"
    --set selection.relative_recovery_certificate_by_bucket.test_near_contact="$rel_near"
    --set selection.relative_recovery_certificate_by_bucket.contact="$rel_contact"
    --set selection.relative_recovery_certificate_by_bucket.test_contact="$rel_contact"
    --set selection.relative_recovery_use_recovery_pool_by_bucket.near_contact="$rec_pool"
    --set selection.relative_recovery_use_recovery_pool_by_bucket.test_near_contact="$rec_pool"
    --set selection.relative_recovery_use_recovery_pool_by_bucket.contact="$rec_pool"
    --set selection.relative_recovery_use_recovery_pool_by_bucket.test_contact="$rec_pool"
    --set selection.relative_recovery_counts_as_evidence=true
    --set selection.relative_recovery_require_macro_by_bucket.near_contact=true
    --set selection.relative_recovery_require_macro_by_bucket.test_near_contact=true
    --set selection.relative_recovery_require_macro_by_bucket.contact=true
    --set selection.relative_recovery_require_macro_by_bucket.test_contact=true
    --set selection.relative_recovery_macro_allowlist_by_bucket.near_contact=brake,stabilize,yield,merge,pull_over
    --set selection.relative_recovery_macro_allowlist_by_bucket.test_near_contact=brake,stabilize,yield,merge,pull_over
    --set selection.relative_recovery_macro_allowlist_by_bucket.contact=yield,merge
    --set selection.relative_recovery_macro_allowlist_by_bucket.test_contact=yield,merge
    --set selection.relative_recovery_macro_blocklist_by_bucket.near_contact=nominal,keep,perturb_nominal,lane_shift
    --set selection.relative_recovery_macro_blocklist_by_bucket.test_near_contact=nominal,keep,perturb_nominal,lane_shift
    --set selection.relative_recovery_macro_blocklist_by_bucket.contact=nominal,keep,perturb_nominal,pull_over,lane_shift,brake,stabilize
    --set selection.relative_recovery_macro_blocklist_by_bucket.test_contact=nominal,keep,perturb_nominal,pull_over,lane_shift,brake,stabilize

    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.near_contact=-0.30
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.test_near_contact=-0.30
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.contact=-0.42
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.test_contact=-0.42
    --set selection.relative_recovery_nominal_gap_min_by_bucket.near_contact=1.10
    --set selection.relative_recovery_nominal_gap_min_by_bucket.test_near_contact=1.10
    --set selection.relative_recovery_nominal_gap_min_by_bucket.contact=0.95
    --set selection.relative_recovery_nominal_gap_min_by_bucket.test_contact=0.95
    --set selection.relative_recovery_nominal_drs_max_by_bucket.near_contact=0.82
    --set selection.relative_recovery_nominal_drs_max_by_bucket.test_near_contact=0.82
    --set selection.relative_recovery_nominal_drs_max_by_bucket.contact=0.86
    --set selection.relative_recovery_nominal_drs_max_by_bucket.test_contact=0.86

    --set selection.relative_recovery_gate_by_bucket.near_contact=pareto
    --set selection.relative_recovery_gate_by_bucket.test_near_contact=pareto
    --set selection.relative_recovery_min_improvement_axes_by_bucket.near_contact=2
    --set selection.relative_recovery_min_improvement_axes_by_bucket.test_near_contact=2
    --set selection.relative_recovery_min_rec_gain_by_bucket.near_contact=0.08
    --set selection.relative_recovery_min_rec_gain_by_bucket.test_near_contact=0.08
    --set selection.relative_recovery_min_gap_reduction_by_bucket.near_contact=0.12
    --set selection.relative_recovery_min_gap_reduction_by_bucket.test_near_contact=0.12
    --set selection.relative_recovery_min_drs_gain_by_bucket.near_contact=0.025
    --set selection.relative_recovery_min_drs_gain_by_bucket.test_near_contact=0.025
    --set selection.relative_recovery_max_drs_drop_by_bucket.near_contact=0.00
    --set selection.relative_recovery_max_drs_drop_by_bucket.test_near_contact=0.00
    --set selection.relative_recovery_max_rec_lcb_drop_by_bucket.near_contact=0.02
    --set selection.relative_recovery_max_rec_lcb_drop_by_bucket.test_near_contact=0.02
    --set selection.relative_recovery_min_drs_by_bucket.near_contact=0.80
    --set selection.relative_recovery_min_drs_by_bucket.test_near_contact=0.80
    --set selection.relative_recovery_max_gap_by_bucket.near_contact=1.35
    --set selection.relative_recovery_max_gap_by_bucket.test_near_contact=1.35
    --set selection.relative_recovery_max_gap_increase_by_bucket.near_contact=0.00
    --set selection.relative_recovery_max_gap_increase_by_bucket.test_near_contact=0.00

    --set selection.relative_recovery_gate_by_bucket.contact=pareto
    --set selection.relative_recovery_gate_by_bucket.test_contact=pareto
    --set selection.relative_recovery_min_improvement_axes_by_bucket.contact=2
    --set selection.relative_recovery_min_improvement_axes_by_bucket.test_contact=2
    --set selection.relative_recovery_min_rec_gain_by_bucket.contact=0.04
    --set selection.relative_recovery_min_rec_gain_by_bucket.test_contact=0.04
    --set selection.relative_recovery_min_gap_reduction_by_bucket.contact=0.08
    --set selection.relative_recovery_min_gap_reduction_by_bucket.test_contact=0.08
    --set selection.relative_recovery_min_drs_gain_by_bucket.contact=0.035
    --set selection.relative_recovery_min_drs_gain_by_bucket.test_contact=0.035
    --set selection.relative_recovery_max_drs_drop_by_bucket.contact=0.00
    --set selection.relative_recovery_max_drs_drop_by_bucket.test_contact=0.00
    --set selection.relative_recovery_max_rec_lcb_drop_by_bucket.contact=0.02
    --set selection.relative_recovery_max_rec_lcb_drop_by_bucket.test_contact=0.02
    --set selection.relative_recovery_min_drs_by_bucket.contact=0.70
    --set selection.relative_recovery_min_drs_by_bucket.test_contact=0.70
    --set selection.relative_recovery_max_gap_by_bucket.contact=1.45
    --set selection.relative_recovery_max_gap_by_bucket.test_contact=1.45
    --set selection.relative_recovery_max_gap_increase_by_bucket.contact=0.00
    --set selection.relative_recovery_max_gap_increase_by_bucket.test_contact=0.00

    --set selection.recovery_cert_max_hard_by_bucket.near_contact=1.0
    --set selection.recovery_cert_max_hard_by_bucket.test_near_contact=1.0
    --set selection.recovery_cert_max_hard_by_bucket.contact=1.0
    --set selection.recovery_cert_max_hard_by_bucket.test_contact=1.0
    --set selection.recovery_cert_max_harm_by_bucket.near_contact=0.55
    --set selection.recovery_cert_max_harm_by_bucket.test_near_contact=0.55
    --set selection.recovery_cert_max_harm_by_bucket.contact=0.70
    --set selection.recovery_cert_max_harm_by_bucket.test_contact=0.70

    # v24: disable the old generic brake/stabilize protective channel in contact.
    # Brake uses the macro-stratified rescue certificate; stabilize must satisfy
    # the stricter near-contact relative gate or abstain.
    --set selection.protective_macro_certificate_by_bucket.safe=false
    --set selection.protective_macro_certificate_by_bucket.test_safe=false
    --set selection.protective_macro_certificate_by_bucket.near_contact=false
    --set selection.protective_macro_certificate_by_bucket.test_near_contact=false
    --set selection.protective_macro_certificate_by_bucket.contact=false
    --set selection.protective_macro_certificate_by_bucket.test_contact=false

    # v24 macro-stratified brake-rescue certificate.  The candidate must be a
    # moderate-gap brake, not a nominal-like perturbation and not a high-gap artifact.
    --set selection.brake_rescue_certificate_by_bucket.near_contact="$brake_rescue"
    --set selection.brake_rescue_certificate_by_bucket.test_near_contact="$brake_rescue"
    --set selection.brake_rescue_certificate_by_bucket.contact="$brake_rescue"
    --set selection.brake_rescue_certificate_by_bucket.test_contact="$brake_rescue"
    --set selection.brake_rescue_macro_name_by_bucket.contact=brake
    --set selection.brake_rescue_macro_name_by_bucket.test_contact=brake
    --set selection.brake_rescue_min_pred_drs_by_bucket.contact=${CONTACT_BRAKE_RESCUE_MIN_DRS:-0.90}
    --set selection.brake_rescue_min_pred_drs_by_bucket.test_contact=${CONTACT_BRAKE_RESCUE_MIN_DRS:-0.90}
    --set selection.brake_rescue_min_pred_r_dep_by_bucket.contact=-0.70
    --set selection.brake_rescue_min_pred_r_dep_by_bucket.test_contact=-0.70
    --set selection.brake_rescue_min_candidate_gap_by_bucket.contact=${CONTACT_BRAKE_RESCUE_MIN_GAP:-0.18}
    --set selection.brake_rescue_min_candidate_gap_by_bucket.test_contact=${CONTACT_BRAKE_RESCUE_MIN_GAP:-0.18}
    --set selection.brake_rescue_max_candidate_gap_by_bucket.contact=${CONTACT_BRAKE_RESCUE_MAX_GAP:-0.34}
    --set selection.brake_rescue_max_candidate_gap_by_bucket.test_contact=${CONTACT_BRAKE_RESCUE_MAX_GAP:-0.34}
    --set selection.brake_rescue_max_hard_by_bucket.contact=1.0
    --set selection.brake_rescue_max_hard_by_bucket.test_contact=1.0
    --set selection.brake_rescue_max_harm_by_bucket.contact=0.70
    --set selection.brake_rescue_max_harm_by_bucket.test_contact=0.70
    --set selection.brake_rescue_require_nominal_unadmitted_by_bucket.contact=true
    --set selection.brake_rescue_require_nominal_unadmitted_by_bucket.test_contact=true
    --set selection.brake_rescue_nominal_rec_lcb_max_by_bucket.contact=0.60
    --set selection.brake_rescue_nominal_rec_lcb_max_by_bucket.test_contact=0.60
    --set selection.brake_rescue_nominal_gap_min_by_bucket.contact=0.02
    --set selection.brake_rescue_nominal_gap_min_by_bucket.test_contact=0.02
    --set selection.brake_rescue_nominal_drs_max_by_bucket.contact=1.01
    --set selection.brake_rescue_nominal_drs_max_by_bucket.test_contact=1.01
    --set selection.brake_rescue_counts_as_evidence=true
    --set selection.brake_rescue_budget_bypass_by_bucket.near_contact=false
    --set selection.brake_rescue_budget_bypass_by_bucket.test_near_contact=false
    --set selection.brake_rescue_budget_bypass_by_bucket.contact=false
    --set selection.brake_rescue_budget_bypass_by_bucket.test_contact=false

    # v34 residual brake-tail certificate: captures contact cases where paper-best
    # brake has high DRS but learned nominal PCD is over-confident, while avoiding
    # early nominal-best brake false positives by requiring high-gap or low-R_dep shape.
    --set selection.brake_tail_rescue_certificate_by_bucket.safe=false
    --set selection.brake_tail_rescue_certificate_by_bucket.test_safe=false
    --set selection.brake_tail_rescue_certificate_by_bucket.near_contact=${RUN_NEAR_TAIL:-false}
    --set selection.brake_tail_rescue_certificate_by_bucket.test_near_contact=${RUN_NEAR_TAIL:-false}
    --set selection.brake_tail_rescue_certificate_by_bucket.contact="$brake_tail"
    --set selection.brake_tail_rescue_certificate_by_bucket.test_contact="$brake_tail"
    --set selection.brake_tail_min_pred_drs_by_bucket.contact=${CONTACT_TAIL_MIN_DRS:-0.78}
    --set selection.brake_tail_min_pred_drs_by_bucket.test_contact=${CONTACT_TAIL_MIN_DRS:-0.78}
    --set selection.brake_tail_min_pred_r_dep_by_bucket.contact=${CONTACT_TAIL_MIN_R_DEP:--0.55}
    --set selection.brake_tail_min_pred_r_dep_by_bucket.test_contact=${CONTACT_TAIL_MIN_R_DEP:--0.55}
    --set selection.brake_tail_min_pred_pcd_by_bucket.contact=${CONTACT_TAIL_MIN_PCD:-0.24}
    --set selection.brake_tail_min_pred_pcd_by_bucket.test_contact=${CONTACT_TAIL_MIN_PCD:-0.24}
    --set selection.brake_tail_min_candidate_gap_by_bucket.contact=${CONTACT_TAIL_MIN_GAP:-0.08}
    --set selection.brake_tail_min_candidate_gap_by_bucket.test_contact=${CONTACT_TAIL_MIN_GAP:-0.08}
    --set selection.brake_tail_max_candidate_gap_by_bucket.contact=${CONTACT_TAIL_MAX_GAP:-0.42}
    --set selection.brake_tail_max_candidate_gap_by_bucket.test_contact=${CONTACT_TAIL_MAX_GAP:-0.42}
    --set selection.brake_tail_high_gap_min_by_bucket.contact=${CONTACT_TAIL_HIGH_GAP:-0.34}
    --set selection.brake_tail_high_gap_min_by_bucket.test_contact=${CONTACT_TAIL_HIGH_GAP:-0.34}
    --set selection.brake_tail_low_r_dep_max_by_bucket.contact=${CONTACT_TAIL_LOW_R_DEP_MAX:-0.00}
    --set selection.brake_tail_low_r_dep_max_by_bucket.test_contact=${CONTACT_TAIL_LOW_R_DEP_MAX:-0.00}
    --set selection.brake_tail_max_hard_by_bucket.contact=1.0
    --set selection.brake_tail_max_hard_by_bucket.test_contact=1.0
    --set selection.brake_tail_max_harm_by_bucket.contact=0.70
    --set selection.brake_tail_max_harm_by_bucket.test_contact=0.70
    --set selection.brake_tail_counts_as_evidence=true
    --set selection.brake_tail_budget_bypass_by_bucket.contact=false
    --set selection.brake_tail_budget_bypass_by_bucket.test_contact=false
    --set selection.brake_tail_challenge_budget_bypass_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_BUDGET_BYPASS:-true}
    --set selection.brake_tail_challenge_budget_bypass_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_BUDGET_BYPASS:-true}
    --set selection.brake_tail_challenge_cooldown_bypass_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_COOLDOWN_BYPASS:-true}
    --set selection.brake_tail_challenge_cooldown_bypass_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_COOLDOWN_BYPASS:-true}
    --set selection.brake_tail_challenge_bypass_pcd_gain_by_bucket.contact="$brake_tail_bypass_pcd"
    --set selection.brake_tail_challenge_bypass_pcd_gain_by_bucket.test_contact="$brake_tail_bypass_pcd"
    --set selection.brake_tail_challenge_min_pred_drs_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_MIN_DRS:-0.90}
    --set selection.brake_tail_challenge_min_pred_drs_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_MIN_DRS:-0.90}
    --set selection.brake_tail_challenge_min_pred_pcd_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_MIN_PCD:-0.32}
    --set selection.brake_tail_challenge_min_pred_pcd_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_MIN_PCD:-0.32}
    --set selection.brake_tail_challenge_min_candidate_gap_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_MIN_GAP:-0.085}
    --set selection.brake_tail_challenge_min_candidate_gap_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_MIN_GAP:-0.085}
    --set selection.brake_tail_challenge_max_candidate_gap_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_MAX_GAP:-0.42}
    --set selection.brake_tail_challenge_max_candidate_gap_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_MAX_GAP:-0.42}
    --set selection.brake_tail_challenge_high_gap_min_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_HIGH_GAP:-0.115}
    --set selection.brake_tail_challenge_high_gap_min_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_HIGH_GAP:-0.115}
    --set selection.brake_tail_challenge_low_r_dep_max_by_bucket.contact=${CONTACT_TAIL_CHALLENGE_LOW_R_DEP_MAX:--0.14}
    --set selection.brake_tail_challenge_low_r_dep_max_by_bucket.test_contact=${CONTACT_TAIL_CHALLENGE_LOW_R_DEP_MAX:--0.14}
    --set selection.brake_tail_challenge_max_consecutive_by_bucket.contact=${CONTACT_TAIL_MAX_CONSECUTIVE:-2}
    --set selection.brake_tail_challenge_max_consecutive_by_bucket.test_contact=${CONTACT_TAIL_MAX_CONSECUTIVE:-2}
    --set selection.brake_tail_challenge_max_consecutive_by_bucket.near_contact=${NEAR_TAIL_MAX_CONSECUTIVE:-1}
    --set selection.brake_tail_challenge_max_consecutive_by_bucket.test_near_contact=${NEAR_TAIL_MAX_CONSECUTIVE:-1}
    --set selection.brake_tail_min_nominal_deviation_by_bucket.near_contact=${NEAR_TAIL_MIN_DEVIATION:-0.002}
    --set selection.brake_tail_min_nominal_deviation_by_bucket.test_near_contact=${NEAR_TAIL_MIN_DEVIATION:-0.002}
    --set selection.brake_tail_min_nominal_deviation_by_bucket.contact=${CONTACT_TAIL_MIN_DEVIATION:-0.002}
    --set selection.brake_tail_min_nominal_deviation_by_bucket.test_contact=${CONTACT_TAIL_MIN_DEVIATION:-0.002}
    --set selection.brake_tail_min_pred_drs_by_bucket.near_contact=${NEAR_TAIL_MIN_DRS:-0.93}
    --set selection.brake_tail_min_pred_drs_by_bucket.test_near_contact=${NEAR_TAIL_MIN_DRS:-0.93}
    --set selection.brake_tail_min_pred_r_dep_by_bucket.near_contact=${NEAR_TAIL_MIN_R_DEP:--0.20}
    --set selection.brake_tail_min_pred_r_dep_by_bucket.test_near_contact=${NEAR_TAIL_MIN_R_DEP:--0.20}
    --set selection.brake_tail_min_pred_pcd_by_bucket.near_contact=${NEAR_TAIL_MIN_PCD:-0.33}
    --set selection.brake_tail_min_pred_pcd_by_bucket.test_near_contact=${NEAR_TAIL_MIN_PCD:-0.33}
    --set selection.brake_tail_min_candidate_gap_by_bucket.near_contact=${NEAR_TAIL_MIN_GAP:-0.14}
    --set selection.brake_tail_min_candidate_gap_by_bucket.test_near_contact=${NEAR_TAIL_MIN_GAP:-0.14}
    --set selection.brake_tail_max_candidate_gap_by_bucket.near_contact=${NEAR_TAIL_MAX_GAP:-0.36}
    --set selection.brake_tail_max_candidate_gap_by_bucket.test_near_contact=${NEAR_TAIL_MAX_GAP:-0.36}
    --set selection.brake_tail_high_gap_min_by_bucket.near_contact=${NEAR_TAIL_HIGH_GAP:-0.15}
    --set selection.brake_tail_high_gap_min_by_bucket.test_near_contact=${NEAR_TAIL_HIGH_GAP:-0.15}
    --set selection.brake_tail_low_r_dep_max_by_bucket.near_contact=${NEAR_TAIL_LOW_R_DEP_MAX:--0.05}
    --set selection.brake_tail_low_r_dep_max_by_bucket.test_near_contact=${NEAR_TAIL_LOW_R_DEP_MAX:--0.05}
    --set selection.brake_tail_challenge_bypass_pcd_gain_by_bucket.near_contact=${NEAR_TAIL_BYPASS_PCD_GAIN:-false}
    --set selection.brake_tail_challenge_bypass_pcd_gain_by_bucket.test_near_contact=${NEAR_TAIL_BYPASS_PCD_GAIN:-false}
    --set selection.brake_tail_challenge_budget_bypass_by_bucket.near_contact=${NEAR_TAIL_CHALLENGE_BUDGET_BYPASS:-false}
    --set selection.brake_tail_challenge_budget_bypass_by_bucket.test_near_contact=${NEAR_TAIL_CHALLENGE_BUDGET_BYPASS:-false}
    --set selection.brake_tail_challenge_cooldown_bypass_by_bucket.near_contact=${NEAR_TAIL_CHALLENGE_COOLDOWN_BYPASS:-false}
    --set selection.brake_tail_challenge_cooldown_bypass_by_bucket.test_near_contact=${NEAR_TAIL_CHALLENGE_COOLDOWN_BYPASS:-false}

    # v24 BMRC: a budget-aware post-contact-deployability rescue certificate.
    # It allows brake to challenge an over-confident nominal prediction in
    # near/contact, but repeated interventions must still pay the hard exposure
    # budget.  This targets the remaining v23 misses where audit-best PCD is
    # brake while selected nominal has over-confident DRS/gap predictions.
    --set selection.pcd_rescue_certificate_by_bucket.safe=false
    --set selection.pcd_rescue_certificate_by_bucket.test_safe=false
    --set selection.pcd_rescue_certificate_by_bucket.near_contact="$pcd_rescue"
    --set selection.pcd_rescue_certificate_by_bucket.test_near_contact="$pcd_rescue"
    --set selection.pcd_rescue_certificate_by_bucket.contact="$pcd_rescue"
    --set selection.pcd_rescue_certificate_by_bucket.test_contact="$pcd_rescue"
    --set selection.pcd_rescue_macro_allowlist_by_bucket.near_contact=brake
    --set selection.pcd_rescue_macro_allowlist_by_bucket.test_near_contact=brake
    --set selection.pcd_rescue_macro_allowlist_by_bucket.contact=${CONTACT_FRONTIER_MACROS:-brake,yield}
    --set selection.pcd_rescue_macro_allowlist_by_bucket.test_contact=${CONTACT_FRONTIER_MACROS:-brake,yield}
    --set selection.pcd_rescue_macro_blocklist_by_bucket.near_contact=nominal,keep,perturb_nominal,pull_over,lane_shift,merge,yield,stabilize
    --set selection.pcd_rescue_macro_blocklist_by_bucket.test_near_contact=nominal,keep,perturb_nominal,pull_over,lane_shift,merge,yield,stabilize
    --set selection.pcd_rescue_macro_blocklist_by_bucket.contact=nominal,keep,perturb_nominal,lane_shift
    --set selection.pcd_rescue_macro_blocklist_by_bucket.test_contact=nominal,keep,perturb_nominal,lane_shift
    --set selection.pcd_rescue_min_pred_pcd_by_bucket.near_contact=0.32
    --set selection.pcd_rescue_min_pred_pcd_by_bucket.test_near_contact=0.32
    --set selection.pcd_rescue_min_pred_pcd_by_bucket.contact=${CONTACT_FRONTIER_MIN_PCD:-0.28}
    --set selection.pcd_rescue_min_pred_pcd_by_bucket.test_contact=${CONTACT_FRONTIER_MIN_PCD:-0.28}
    --set selection.pcd_rescue_min_pcd_gain_by_bucket.near_contact=-1.0
    --set selection.pcd_rescue_min_pcd_gain_by_bucket.test_near_contact=-1.0
    --set selection.pcd_rescue_min_pcd_gain_by_bucket.contact=${CONTACT_FRONTIER_MIN_PCD_GAIN:-0.0}
    --set selection.pcd_rescue_min_pcd_gain_by_bucket.test_contact=${CONTACT_FRONTIER_MIN_PCD_GAIN:-0.0}
    --set selection.pcd_rescue_min_pred_drs_by_bucket.near_contact=0.88
    --set selection.pcd_rescue_min_pred_drs_by_bucket.test_near_contact=0.88
    --set selection.pcd_rescue_min_pred_drs_by_bucket.contact=${CONTACT_FRONTIER_MIN_DRS:-0.62}
    --set selection.pcd_rescue_min_pred_drs_by_bucket.test_contact=${CONTACT_FRONTIER_MIN_DRS:-0.62}
    --set selection.pcd_rescue_min_pred_r_dep_by_bucket.near_contact=-0.75
    --set selection.pcd_rescue_min_pred_r_dep_by_bucket.test_near_contact=-0.75
    --set selection.pcd_rescue_min_pred_r_dep_by_bucket.contact=${CONTACT_FRONTIER_MIN_R_DEP:--1.20}
    --set selection.pcd_rescue_min_pred_r_dep_by_bucket.test_contact=${CONTACT_FRONTIER_MIN_R_DEP:--1.20}
    --set selection.pcd_rescue_min_candidate_gap_by_bucket.near_contact=0.02
    --set selection.pcd_rescue_min_candidate_gap_by_bucket.test_near_contact=0.02
    --set selection.pcd_rescue_min_candidate_gap_by_bucket.contact=0.02
    --set selection.pcd_rescue_min_candidate_gap_by_bucket.test_contact=0.02
    --set selection.pcd_rescue_max_candidate_gap_by_bucket.near_contact=0.34
    --set selection.pcd_rescue_max_candidate_gap_by_bucket.test_near_contact=0.34
    --set selection.pcd_rescue_max_candidate_gap_by_bucket.contact=${CONTACT_FRONTIER_MAX_GAP:-1.45}
    --set selection.pcd_rescue_max_candidate_gap_by_bucket.test_contact=${CONTACT_FRONTIER_MAX_GAP:-1.45}
    --set selection.pcd_rescue_max_hard_by_bucket.near_contact=1.0
    --set selection.pcd_rescue_max_hard_by_bucket.test_near_contact=1.0
    --set selection.pcd_rescue_max_hard_by_bucket.contact=1.0
    --set selection.pcd_rescue_max_hard_by_bucket.test_contact=1.0
    --set selection.pcd_rescue_max_harm_by_bucket.near_contact=0.55
    --set selection.pcd_rescue_max_harm_by_bucket.test_near_contact=0.55
    --set selection.pcd_rescue_max_harm_by_bucket.contact=0.70
    --set selection.pcd_rescue_max_harm_by_bucket.test_contact=0.70
    --set selection.pcd_rescue_require_nominal_low_headroom_by_bucket.near_contact=true
    --set selection.pcd_rescue_require_nominal_low_headroom_by_bucket.test_near_contact=true
    --set selection.pcd_rescue_require_nominal_low_headroom_by_bucket.contact=true
    --set selection.pcd_rescue_require_nominal_low_headroom_by_bucket.test_contact=true
    --set selection.pcd_rescue_require_nominal_unadmitted_by_bucket.near_contact=false
    --set selection.pcd_rescue_require_nominal_unadmitted_by_bucket.test_near_contact=false
    --set selection.pcd_rescue_require_nominal_unadmitted_by_bucket.contact=false
    --set selection.pcd_rescue_require_nominal_unadmitted_by_bucket.test_contact=false
    --set selection.pcd_rescue_nominal_rec_lcb_max_by_bucket.near_contact=0.65
    --set selection.pcd_rescue_nominal_rec_lcb_max_by_bucket.test_near_contact=0.65
    --set selection.pcd_rescue_nominal_rec_lcb_max_by_bucket.contact=0.65
    --set selection.pcd_rescue_nominal_rec_lcb_max_by_bucket.test_contact=0.65
    --set selection.pcd_rescue_nominal_gap_min_by_bucket.near_contact=0.02
    --set selection.pcd_rescue_nominal_gap_min_by_bucket.test_near_contact=0.02
    --set selection.pcd_rescue_nominal_gap_min_by_bucket.contact=0.02
    --set selection.pcd_rescue_nominal_gap_min_by_bucket.test_contact=0.02
    --set selection.pcd_rescue_nominal_drs_max_by_bucket.near_contact=1.01
    --set selection.pcd_rescue_nominal_drs_max_by_bucket.test_near_contact=1.01
    --set selection.pcd_rescue_nominal_drs_max_by_bucket.contact=1.01
    --set selection.pcd_rescue_nominal_drs_max_by_bucket.test_contact=1.01
    --set selection.pcd_rescue_max_utility_drop_by_bucket.near_contact=-1.0
    --set selection.pcd_rescue_max_utility_drop_by_bucket.test_near_contact=-1.0
    --set selection.pcd_rescue_max_utility_drop_by_bucket.contact=-1.0
    --set selection.pcd_rescue_max_utility_drop_by_bucket.test_contact=-1.0
    --set selection.pcd_rescue_bonus_by_bucket.near_contact=0.15
    --set selection.pcd_rescue_bonus_by_bucket.test_near_contact=0.15
    --set selection.pcd_rescue_bonus_by_bucket.contact=0.12
    --set selection.pcd_rescue_bonus_by_bucket.test_contact=0.12
    --set selection.pcd_rescue_counts_as_evidence=true
    --set selection.pcd_rescue_budget_bypass_by_bucket.near_contact=false
    --set selection.pcd_rescue_budget_bypass_by_bucket.test_near_contact=false
    --set selection.pcd_rescue_budget_bypass_by_bucket.contact=false
    --set selection.pcd_rescue_budget_bypass_by_bucket.test_contact=false

    # v26 NARC: a certified brake/PCD rescue may challenge nominal even when
    # the nominal learned LCB is over-confident, but exposure is controlled by
    # an explicit closed-loop cooldown.  Contact is enabled by default; near is
    # conservative by default to test whether the NUP drop was repeated braking.
    --set selection.stress_rescue_challenge_nominal_by_bucket.safe=false
    --set selection.stress_rescue_challenge_nominal_by_bucket.test_safe=false
    --set selection.stress_rescue_challenge_nominal_by_bucket.near_contact=${RUN_NEAR_CHALLENGE:-true}
    --set selection.stress_rescue_challenge_nominal_by_bucket.test_near_contact=${RUN_NEAR_CHALLENGE:-true}
    --set selection.stress_rescue_challenge_nominal_by_bucket.contact=true
    --set selection.stress_rescue_challenge_nominal_by_bucket.test_contact=true
    --set selection.intervention_cooldown_steps_by_bucket.near_contact=${NEAR_COOLDOWN_STEPS:-12}
    --set selection.intervention_cooldown_steps_by_bucket.test_near_contact=${NEAR_COOLDOWN_STEPS:-12}
    --set selection.intervention_cooldown_steps_by_bucket.contact=${CONTACT_COOLDOWN_STEPS:-1}
    --set selection.intervention_cooldown_steps_by_bucket.test_contact=${CONTACT_COOLDOWN_STEPS:-1}

    # v26 Guarded-NARC: v25 proved that challenge itself works, but contact
    # over-selected brake because the challenge frontier was identical to the
    # coarse rescue certificate.  These extra guards restrict nominal challenge
    # to a moderate ambiguity band and cap high-utility false-brake artifacts.
    --set selection.rescue_challenge_min_candidate_gap_by_bucket.near_contact=${NEAR_CHALLENGE_MIN_GAP:-0.14}
    --set selection.rescue_challenge_min_candidate_gap_by_bucket.test_near_contact=${NEAR_CHALLENGE_MIN_GAP:-0.14}
    --set selection.rescue_challenge_max_candidate_gap_by_bucket.near_contact=${NEAR_CHALLENGE_MAX_GAP:-0.36}
    --set selection.rescue_challenge_max_candidate_gap_by_bucket.test_near_contact=${NEAR_CHALLENGE_MAX_GAP:-0.36}
    --set selection.rescue_challenge_min_pred_drs_by_bucket.near_contact=${NEAR_CHALLENGE_MIN_DRS:-0.90}
    --set selection.rescue_challenge_min_pred_drs_by_bucket.test_near_contact=${NEAR_CHALLENGE_MIN_DRS:-0.90}
    --set selection.rescue_challenge_min_pred_pcd_by_bucket.near_contact=${NEAR_CHALLENGE_MIN_PCD:-0.31}
    --set selection.rescue_challenge_min_pred_pcd_by_bucket.test_near_contact=${NEAR_CHALLENGE_MIN_PCD:-0.31}
    --set selection.rescue_challenge_max_pred_utility_by_bucket.near_contact=${NEAR_CHALLENGE_MAX_UTILITY:--1}
    --set selection.rescue_challenge_max_pred_utility_by_bucket.test_near_contact=${NEAR_CHALLENGE_MAX_UTILITY:--1}
    --set selection.rescue_challenge_max_used_by_bucket.near_contact=${NEAR_CHALLENGE_MAX_USED:-3}
    --set selection.rescue_challenge_max_used_by_bucket.test_near_contact=${NEAR_CHALLENGE_MAX_USED:-3}

    --set selection.rescue_challenge_min_candidate_gap_by_bucket.contact=${CONTACT_CHALLENGE_MIN_GAP:-0.08}
    --set selection.rescue_challenge_min_candidate_gap_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_GAP:-0.08}
    --set selection.rescue_challenge_max_candidate_gap_by_bucket.contact=${CONTACT_CHALLENGE_MAX_GAP:-1.45}
    --set selection.rescue_challenge_max_candidate_gap_by_bucket.test_contact=${CONTACT_CHALLENGE_MAX_GAP:-1.45}
    --set selection.rescue_challenge_min_pred_drs_by_bucket.contact=${CONTACT_CHALLENGE_MIN_DRS:-0.62}
    --set selection.rescue_challenge_min_pred_drs_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_DRS:-0.62}
    --set selection.rescue_challenge_min_pred_pcd_by_bucket.contact=${CONTACT_CHALLENGE_MIN_PCD:-0.28}
    --set selection.rescue_challenge_min_pred_pcd_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_PCD:-0.28}
    --set selection.rescue_challenge_max_pred_utility_by_bucket.contact=${CONTACT_CHALLENGE_MAX_UTILITY:--1}
    --set selection.rescue_challenge_max_pred_utility_by_bucket.test_contact=${CONTACT_CHALLENGE_MAX_UTILITY:--1}
    --set selection.rescue_challenge_max_used_by_bucket.contact=${CONTACT_CHALLENGE_MAX_USED:-8}
    --set selection.rescue_challenge_max_used_by_bucket.test_contact=${CONTACT_CHALLENGE_MAX_USED:-8}
    --set selection.rescue_challenge_score_pcd_weight_by_bucket.contact=1.0
    --set selection.rescue_challenge_score_pcd_weight_by_bucket.test_contact=1.0
    --set selection.rescue_challenge_score_drs_weight_by_bucket.contact=0.20
    --set selection.rescue_challenge_score_drs_weight_by_bucket.test_contact=0.20
    --set selection.rescue_challenge_score_gap_weight_by_bucket.contact=0.30
    --set selection.rescue_challenge_score_gap_weight_by_bucket.test_contact=0.30
    --set selection.rescue_challenge_score_utility_weight_by_bucket.contact=0.02
    --set selection.rescue_challenge_score_utility_weight_by_bucket.test_contact=0.02

    --set selection.require_intervention_evidence_by_bucket.safe=true
    --set selection.require_intervention_evidence_by_bucket.test_safe=true
    --set selection.require_intervention_evidence_by_bucket.near_contact=true
    --set selection.require_intervention_evidence_by_bucket.test_near_contact=true
    --set selection.require_intervention_evidence_by_bucket.contact=true
    --set selection.require_intervention_evidence_by_bucket.test_contact=true
    --set selection.intervention_min_pred_drs_by_bucket.near_contact=0.80
    --set selection.intervention_min_pred_drs_by_bucket.test_near_contact=0.80
    --set selection.intervention_min_pred_drs_by_bucket.contact=0.64
    --set selection.intervention_min_pred_drs_by_bucket.test_contact=0.64
    --set selection.intervention_max_pred_gap_by_bucket.near_contact=1.35
    --set selection.intervention_max_pred_gap_by_bucket.test_near_contact=1.35
    --set selection.intervention_max_pred_gap_by_bucket.contact=1.45
    --set selection.intervention_max_pred_gap_by_bucket.test_contact=1.45
    --set selection.intervention_min_rec_lcb_gain_by_bucket.near_contact=0.00
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_near_contact=0.00
    --set selection.intervention_min_rec_lcb_gain_by_bucket.contact=0.00
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.near_contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.test_near_contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.test_contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.near_contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.test_near_contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.test_contact=0.00
    --set selection.prefer_admitted_by_bucket.safe=true
    --set selection.prefer_admitted_by_bucket.near_contact=true
    --set selection.prefer_admitted_by_bucket.contact=true
    --set selection.calibrated_shortfall_penalty_by_bucket.safe=0.05
    --set selection.calibrated_shortfall_penalty_by_bucket.near_contact=0.80
    --set selection.calibrated_shortfall_penalty_by_bucket.contact=0.70
    --set selection.calibrated_gap_penalty_by_bucket.safe=0.00
    --set selection.calibrated_gap_penalty_by_bucket.near_contact=0.35
    --set selection.calibrated_gap_penalty_by_bucket.contact=0.34
    --set selection.deployability_bonus_by_bucket.near_contact=0.50
    --set selection.deployability_bonus_by_bucket.contact=0.35
    --set selection.contact_deployability_bonus_by_bucket.contact=0.18
    --set selection.contact_gap_penalty_by_bucket.contact=0.30
    --set selection.relative_recovery_bonus_by_bucket.near_contact=0.40
    --set selection.relative_recovery_bonus_by_bucket.test_near_contact=0.40
    --set selection.relative_recovery_bonus_by_bucket.contact=0.15
    --set selection.relative_recovery_bonus_by_bucket.test_contact=0.15
    --set selection.intervention_budget_rate_by_bucket.safe=0.0
    --set selection.intervention_budget_rate_by_bucket.near_contact=0.06
    --set selection.intervention_budget_rate_by_bucket.contact=${CONTACT_BUDGET_RATE:-0.025}
    --set selection.intervention_budget_penalty_by_bucket.safe=50.0
    --set selection.intervention_budget_penalty_by_bucket.near_contact=6.0
    --set selection.intervention_budget_penalty_by_bucket.contact=6.0
    --set selection.intervention_budget_hard_by_bucket.contact=${CONTACT_BUDGET_HARD:-true}
    --set selection.intervention_budget_hard_by_bucket.test_contact=${CONTACT_BUDGET_HARD:-true}
    --set selection.intervention_budget_hard_min_rec_gain_by_bucket.contact=0.04
    --set selection.intervention_budget_hard_min_rec_gain_by_bucket.test_contact=0.04
    --set selection.intervention_budget_hard_min_drs_gain_by_bucket.contact=0.03
    --set selection.intervention_budget_hard_min_drs_gain_by_bucket.test_contact=0.03
    --set selection.intervention_budget_hard_min_gap_reduction_by_bucket.contact=0.08
    --set selection.intervention_budget_hard_min_gap_reduction_by_bucket.test_contact=0.08
    --set selection.deviation_penalty_by_bucket.safe=3.0
    --set selection.deviation_penalty_by_bucket.near_contact=0.12
    --set selection.deviation_penalty_by_bucket.contact=0.18
    --set selection.intervention_penalty_by_bucket.safe=2.0
    --set selection.intervention_penalty_by_bucket.near_contact=0.030
    --set selection.intervention_penalty_by_bucket.contact=0.060

    # v48 OC-TRAC-SR: a held-out-verified top-1 value certificate may augment
    # admission in Near/Contact only. Safe remains nominal-locked.
    --set selection.direct_value_certificate_by_bucket.safe=false
    --set selection.direct_value_certificate_by_bucket.test_safe=false
    --set selection.direct_value_certificate_by_bucket.near_contact="$direct_value"
    --set selection.direct_value_certificate_by_bucket.test_near_contact="$direct_value"
    --set selection.direct_value_certificate_by_bucket.contact="$direct_value"
    --set selection.direct_value_certificate_by_bucket.test_contact="$direct_value"
    --set selection.direct_value_score_mode_by_bucket.near_contact=true
    --set selection.direct_value_score_mode_by_bucket.test_near_contact=true
    --set selection.direct_value_score_mode_by_bucket.contact=true
    --set selection.direct_value_score_mode_by_bucket.test_contact=true
    --set selection.direct_value_top1_only_by_bucket.near_contact=${DIRECT_TOP1_ONLY:-true}
    --set selection.direct_value_top1_only_by_bucket.test_near_contact=${DIRECT_TOP1_ONLY:-true}
    --set selection.direct_value_top1_only_by_bucket.contact=${DIRECT_TOP1_ONLY:-true}
    --set selection.direct_value_top1_only_by_bucket.test_contact=${DIRECT_TOP1_ONLY:-true}
    --set selection.direct_value_evidence_rerank_top_k_by_bucket.near_contact=${NEAR_DIRECT_EVIDENCE_RERANK}
    --set selection.direct_value_evidence_rerank_top_k_by_bucket.test_near_contact=${NEAR_DIRECT_EVIDENCE_RERANK}
    --set selection.direct_value_evidence_rerank_top_k_by_bucket.contact=${CONTACT_DIRECT_EVIDENCE_RERANK}
    --set selection.direct_value_evidence_rerank_top_k_by_bucket.test_contact=${CONTACT_DIRECT_EVIDENCE_RERANK}
    --set selection.direct_value_proposal_top_k_by_bucket.near_contact=${NEAR_DIRECT_PROPOSAL_TOP_K}
    --set selection.direct_value_proposal_top_k_by_bucket.test_near_contact=${NEAR_DIRECT_PROPOSAL_TOP_K}
    --set selection.direct_value_proposal_top_k_by_bucket.contact=${CONTACT_DIRECT_PROPOSAL_TOP_K}
    --set selection.direct_value_proposal_top_k_by_bucket.test_contact=${CONTACT_DIRECT_PROPOSAL_TOP_K}
    --set selection.direct_value_min_rank_margin_by_bucket.near_contact=${NEAR_DIRECT_MIN_RANK_MARGIN}
    --set selection.direct_value_min_rank_margin_by_bucket.test_near_contact=${NEAR_DIRECT_MIN_RANK_MARGIN}
    --set selection.direct_value_min_rank_margin_by_bucket.contact=${CONTACT_DIRECT_MIN_RANK_MARGIN}
    --set selection.direct_value_min_rank_margin_by_bucket.test_contact=${CONTACT_DIRECT_MIN_RANK_MARGIN}
    --set selection.direct_value_conditional_rank_margin_by_bucket.near_contact=${NEAR_DIRECT_CONDITIONAL_RANK_MARGIN}
    --set selection.direct_value_conditional_rank_margin_by_bucket.test_near_contact=${NEAR_DIRECT_CONDITIONAL_RANK_MARGIN}
    --set selection.direct_value_conditional_rank_margin_by_bucket.contact=${CONTACT_DIRECT_CONDITIONAL_RANK_MARGIN}
    --set selection.direct_value_conditional_rank_margin_by_bucket.test_contact=${CONTACT_DIRECT_CONDITIONAL_RANK_MARGIN}
    --set selection.direct_value_opportunity_threshold_by_bucket.near_contact=${NEAR_DIRECT_OPPORTUNITY_THRESHOLD}
    --set selection.direct_value_opportunity_threshold_by_bucket.test_near_contact=${NEAR_DIRECT_OPPORTUNITY_THRESHOLD}
    --set selection.direct_value_opportunity_threshold_by_bucket.contact=${CONTACT_DIRECT_OPPORTUNITY_THRESHOLD}
    --set selection.direct_value_opportunity_threshold_by_bucket.test_contact=${CONTACT_DIRECT_OPPORTUNITY_THRESHOLD}
    --set selection.direct_value_harm_threshold_by_bucket.near_contact=${NEAR_DIRECT_HARM_THRESHOLD}
    --set selection.direct_value_harm_threshold_by_bucket.test_near_contact=${NEAR_DIRECT_HARM_THRESHOLD}
    --set selection.direct_value_harm_threshold_by_bucket.contact=${CONTACT_DIRECT_HARM_THRESHOLD}
    --set selection.direct_value_harm_threshold_by_bucket.test_contact=${CONTACT_DIRECT_HARM_THRESHOLD}
    --set selection.direct_value_risk_controlled_admission_by_bucket.safe=false
    --set selection.direct_value_risk_controlled_admission_by_bucket.test_safe=false
    --set selection.direct_value_risk_controlled_admission_by_bucket.near_contact=${RSC_AUGMENT_ADMISSION:-true}
    --set selection.direct_value_risk_controlled_admission_by_bucket.test_near_contact=${RSC_AUGMENT_ADMISSION:-true}
    --set selection.direct_value_risk_controlled_admission_by_bucket.contact=${RSC_AUGMENT_ADMISSION:-true}
    --set selection.direct_value_risk_controlled_admission_by_bucket.test_contact=${RSC_AUGMENT_ADMISSION:-true}
    --set selection.direct_value_macro_allowlist_by_bucket.near_contact=${NEAR_DIRECT_MACROS:-brake,yield,merge,pull_over,stabilize}
    --set selection.direct_value_macro_allowlist_by_bucket.test_near_contact=${NEAR_DIRECT_MACROS:-brake,yield,merge,pull_over,stabilize}
    --set selection.direct_value_macro_allowlist_by_bucket.contact=${CONTACT_DIRECT_MACROS:-brake,yield,merge,pull_over,stabilize}
    --set selection.direct_value_macro_allowlist_by_bucket.test_contact=${CONTACT_DIRECT_MACROS:-brake,yield,merge,pull_over,stabilize}
    --set selection.direct_value_min_nominal_deviation_by_bucket.near_contact=${NEAR_DIRECT_MIN_DEVIATION:-0.002}
    --set selection.direct_value_min_nominal_deviation_by_bucket.test_near_contact=${NEAR_DIRECT_MIN_DEVIATION:-0.002}
    --set selection.direct_value_min_nominal_deviation_by_bucket.contact=${CONTACT_DIRECT_MIN_DEVIATION:-0.002}
    --set selection.direct_value_min_nominal_deviation_by_bucket.test_contact=${CONTACT_DIRECT_MIN_DEVIATION:-0.002}
    --set selection.direct_value_uncertainty_mode_by_bucket.near_contact=risk_selective
    --set selection.direct_value_additive_q_by_bucket.near_contact=0
    --set selection.direct_value_uncertainty_mode_by_bucket.test_near_contact=risk_selective
    --set selection.direct_value_additive_q_by_bucket.test_near_contact=0
    --set selection.direct_value_uncertainty_mode_by_bucket.contact=risk_selective
    --set selection.direct_value_additive_q_by_bucket.contact=0
    --set selection.direct_value_uncertainty_mode_by_bucket.test_contact=risk_selective
    --set selection.direct_value_additive_q_by_bucket.test_contact=0
    --set selection.direct_value_min_advantage_lcb_by_bucket.near_contact=${NEAR_DIRECT_MIN_ADV:-$NEAR_DIRECT_THRESHOLD}
    --set selection.direct_value_min_advantage_lcb_by_bucket.test_near_contact=${NEAR_DIRECT_MIN_ADV:-$NEAR_DIRECT_THRESHOLD}
    --set selection.direct_value_min_advantage_lcb_by_bucket.contact=${CONTACT_DIRECT_MIN_ADV:-$CONTACT_DIRECT_THRESHOLD}
    --set selection.direct_value_min_advantage_lcb_by_bucket.test_contact=${CONTACT_DIRECT_MIN_ADV:-$CONTACT_DIRECT_THRESHOLD}
    --set selection.direct_value_min_candidate_value_by_bucket.near_contact=${NEAR_DIRECT_MIN_VALUE:--1000000000.0}
    --set selection.direct_value_min_candidate_value_by_bucket.test_near_contact=${NEAR_DIRECT_MIN_VALUE:--1000000000.0}
    --set selection.direct_value_min_candidate_value_by_bucket.contact=${CONTACT_DIRECT_MIN_VALUE:--1000000000.0}
    --set selection.direct_value_min_candidate_value_by_bucket.test_contact=${CONTACT_DIRECT_MIN_VALUE:--1000000000.0}
    --set selection.direct_value_max_candidate_std_by_bucket.near_contact=${NEAR_DIRECT_MAX_STD:-0.24}
    --set selection.direct_value_max_candidate_std_by_bucket.test_near_contact=${NEAR_DIRECT_MAX_STD:-0.24}
    --set selection.direct_value_max_candidate_std_by_bucket.contact=${CONTACT_DIRECT_MAX_STD:-0.28}
    --set selection.direct_value_max_candidate_std_by_bucket.test_contact=${CONTACT_DIRECT_MAX_STD:-0.28}
    --set selection.direct_value_max_hard_by_bucket.near_contact=0.0
    --set selection.direct_value_max_hard_by_bucket.test_near_contact=0.0
    --set selection.direct_value_max_hard_by_bucket.contact=1.0
    --set selection.direct_value_max_hard_by_bucket.test_contact=1.0
    --set selection.direct_value_max_harm_by_bucket.near_contact=0.55
    --set selection.direct_value_max_harm_by_bucket.test_near_contact=0.55
    --set selection.direct_value_max_harm_by_bucket.contact=0.70
    --set selection.direct_value_max_harm_by_bucket.test_contact=0.70
    --set selection.direct_value_bonus_by_bucket.near_contact=${NEAR_DIRECT_BONUS:-0.20}
    --set selection.direct_value_bonus_by_bucket.test_near_contact=${NEAR_DIRECT_BONUS:-0.20}
    --set selection.direct_value_bonus_by_bucket.contact=${CONTACT_DIRECT_BONUS:-0.18}
    --set selection.direct_value_bonus_by_bucket.test_contact=${CONTACT_DIRECT_BONUS:-0.18}
    --set selection.direct_value_max_consecutive_by_bucket.near_contact=${NEAR_DIRECT_MAX_CONSECUTIVE:-1}
    --set selection.direct_value_max_consecutive_by_bucket.test_near_contact=${NEAR_DIRECT_MAX_CONSECUTIVE:-1}
    --set selection.direct_value_max_consecutive_by_bucket.contact=${CONTACT_DIRECT_MAX_CONSECUTIVE:-2}
    --set selection.direct_value_max_consecutive_by_bucket.test_contact=${CONTACT_DIRECT_MAX_CONSECUTIVE:-2}
    --set selection.intervention_cooldown_steps_by_bucket.near_contact=0
    --set selection.intervention_cooldown_steps_by_bucket.test_near_contact=0
    --set selection.intervention_cooldown_steps_by_bucket.contact=0
    --set selection.intervention_cooldown_steps_by_bucket.test_contact=0
  )

  if [[ "$SAFE_NOMINAL_ONLY" == "1" ]]; then
    COMMON_SEL+=(
      --set closed_loop.require_calibrated_selector=false
      --set closed_loop.require_gamma_by_bucket=false
      --set evaluation.require_calibrated_selector=false
      --set evaluation.require_gamma_by_bucket=false
    )
  else
    COMMON_SEL+=(
      --set selection.gamma_rec_by_bucket_file="$GAMMA"
      --set closed_loop.require_calibrated_selector=true
      --set closed_loop.require_gamma_by_bucket=true
      --set evaluation.require_calibrated_selector=true
      --set evaluation.require_gamma_by_bucket=true
    )
  fi

  if [[ "$is_v27" == "true" ]]; then
    COMMON_SEL+=(
      # v27 DDC principle:
      # safe is locked nominal; near-contact is nominal-preserving unless a
      # recovery macro has material deployability dominance; contact allows
      # post-contact recovery but still requires positive PCD/DRS/gap evidence.
      --set selection.stress_rescue_challenge_nominal_by_bucket.near_contact=${RUN_NEAR_CHALLENGE:-true}
      --set selection.stress_rescue_challenge_nominal_by_bucket.test_near_contact=${RUN_NEAR_CHALLENGE:-true}
      --set selection.stress_rescue_challenge_nominal_by_bucket.contact=true
      --set selection.stress_rescue_challenge_nominal_by_bucket.test_contact=true
      --set selection.brake_rescue_certificate_by_bucket.near_contact=${RUN_NEAR_BRAKE_RESCUE:-false}
      --set selection.brake_rescue_certificate_by_bucket.test_near_contact=${RUN_NEAR_BRAKE_RESCUE:-false}
      --set selection.brake_rescue_certificate_by_bucket.contact=${CONTACT_BRAKE_RESCUE:-false}
      --set selection.brake_rescue_certificate_by_bucket.test_contact=${CONTACT_BRAKE_RESCUE:-false}

      # Near-contact: preserve nominal utility; no repeated brake artifacts.
      --set selection.relative_recovery_min_drs_by_bucket.near_contact=0.90
      --set selection.relative_recovery_min_drs_by_bucket.test_near_contact=0.90
      --set selection.relative_recovery_min_drs_gain_by_bucket.near_contact=0.00
      --set selection.relative_recovery_min_drs_gain_by_bucket.test_near_contact=0.00
      --set selection.relative_recovery_min_gap_reduction_by_bucket.near_contact=0.08
      --set selection.relative_recovery_min_gap_reduction_by_bucket.test_near_contact=0.08
      --set selection.relative_recovery_min_rec_gain_by_bucket.near_contact=0.10
      --set selection.relative_recovery_min_rec_gain_by_bucket.test_near_contact=0.10
      --set selection.relative_recovery_min_improvement_axes_by_bucket.near_contact=2
      --set selection.relative_recovery_min_improvement_axes_by_bucket.test_near_contact=2
      --set selection.relative_recovery_macro_allowlist_by_bucket.near_contact=${NEAR_RELATIVE_MACROS:-merge,yield,stabilize}
      --set selection.relative_recovery_macro_allowlist_by_bucket.test_near_contact=${NEAR_RELATIVE_MACROS:-merge,yield,stabilize}
      --set selection.pcd_rescue_macro_allowlist_by_bucket.near_contact=${NEAR_PCD_MACROS:-merge,yield,stabilize}
      --set selection.pcd_rescue_macro_allowlist_by_bucket.test_near_contact=${NEAR_PCD_MACROS:-merge,yield,stabilize}
      --set selection.pcd_rescue_macro_blocklist_by_bucket.near_contact=nominal,keep,perturb_nominal,pull_over,lane_shift,brake
      --set selection.pcd_rescue_macro_blocklist_by_bucket.test_near_contact=nominal,keep,perturb_nominal,pull_over,lane_shift,brake
      --set selection.pcd_rescue_min_pcd_gain_by_bucket.near_contact=0.020
      --set selection.pcd_rescue_min_pcd_gain_by_bucket.test_near_contact=0.020
      --set selection.pcd_rescue_max_utility_drop_by_bucket.near_contact=0.12
      --set selection.pcd_rescue_max_utility_drop_by_bucket.test_near_contact=0.12
      --set selection.pcd_rescue_nominal_low_headroom_min_axes_by_bucket.near_contact=2
      --set selection.pcd_rescue_nominal_low_headroom_min_axes_by_bucket.test_near_contact=2
      --set selection.intervention_budget_rate_by_bucket.near_contact=0.035
      --set selection.intervention_budget_rate_by_bucket.test_near_contact=0.035
      --set selection.intervention_budget_hard_by_bucket.near_contact=true
      --set selection.intervention_budget_hard_by_bucket.test_near_contact=true
      --set selection.intervention_cooldown_steps_by_bucket.near_contact=${NEAR_COOLDOWN_STEPS:-8}
      --set selection.intervention_cooldown_steps_by_bucket.test_near_contact=${NEAR_COOLDOWN_STEPS:-8}
      --set selection.stress_nominal_anchor_by_bucket.near_contact=true
      --set selection.stress_nominal_anchor_by_bucket.test_near_contact=true
      --set selection.stress_anchor_drs_floor_by_bucket.near_contact=0.90
      --set selection.stress_anchor_drs_floor_by_bucket.test_near_contact=0.90
      --set selection.stress_anchor_max_gap_by_bucket.near_contact=0.25
      --set selection.stress_anchor_max_gap_by_bucket.test_near_contact=0.25
      --set selection.stress_anchor_min_drs_gain_by_bucket.near_contact=0.04
      --set selection.stress_anchor_min_drs_gain_by_bucket.test_near_contact=0.04
      --set selection.stress_anchor_min_rec_gain_by_bucket.near_contact=0.05
      --set selection.stress_anchor_min_rec_gain_by_bucket.test_near_contact=0.05
      --set selection.stress_anchor_min_gap_reduction_by_bucket.near_contact=0.04
      --set selection.stress_anchor_min_gap_reduction_by_bucket.test_near_contact=0.04

      # Contact: prioritize post-contact deployability, but do not accept a
      # rescue challenge without deployability dominance over nominal.
      --set selection.pcd_rescue_macro_allowlist_by_bucket.contact=${CONTACT_FRONTIER_MACROS:-brake,yield}
      --set selection.pcd_rescue_macro_allowlist_by_bucket.test_contact=${CONTACT_FRONTIER_MACROS:-brake,yield}
      --set selection.pcd_rescue_macro_blocklist_by_bucket.contact=nominal,keep,perturb_nominal,lane_shift
      --set selection.pcd_rescue_macro_blocklist_by_bucket.test_contact=nominal,keep,perturb_nominal,lane_shift
      --set selection.pcd_rescue_min_pcd_gain_by_bucket.contact=${CONTACT_PCD_RESCUE_MIN_GAIN:-0.000}
      --set selection.pcd_rescue_min_pcd_gain_by_bucket.test_contact=${CONTACT_PCD_RESCUE_MIN_GAIN:-0.000}
      --set selection.pcd_rescue_max_utility_drop_by_bucket.contact=${CONTACT_PCD_RESCUE_MAX_UTILITY_DROP:-0.35}
      --set selection.pcd_rescue_max_utility_drop_by_bucket.test_contact=${CONTACT_PCD_RESCUE_MAX_UTILITY_DROP:-0.35}
      --set selection.pcd_rescue_nominal_low_headroom_min_axes_by_bucket.contact=1
      --set selection.pcd_rescue_nominal_low_headroom_min_axes_by_bucket.test_contact=1
      --set selection.relative_recovery_macro_allowlist_by_bucket.contact=${CONTACT_RELATIVE_MACROS:-yield}
      --set selection.relative_recovery_macro_allowlist_by_bucket.test_contact=${CONTACT_RELATIVE_MACROS:-yield}
      --set selection.relative_recovery_macro_blocklist_by_bucket.contact=nominal,keep,perturb_nominal,pull_over,lane_shift,brake
      --set selection.relative_recovery_macro_blocklist_by_bucket.test_contact=nominal,keep,perturb_nominal,pull_over,lane_shift,brake
      --set selection.intervention_budget_rate_by_bucket.contact=${CONTACT_BUDGET_RATE:-0.025}
      --set selection.intervention_budget_rate_by_bucket.test_contact=${CONTACT_BUDGET_RATE:-0.025}
      --set selection.intervention_budget_hard_by_bucket.contact=${CONTACT_BUDGET_HARD:-true}
      --set selection.intervention_budget_hard_by_bucket.test_contact=${CONTACT_BUDGET_HARD:-true}
      --set selection.intervention_cooldown_steps_by_bucket.contact=${CONTACT_COOLDOWN_STEPS:-1}
      --set selection.intervention_cooldown_steps_by_bucket.test_contact=${CONTACT_COOLDOWN_STEPS:-1}

      # New code-level DDC guards.  These are the main v27 addition.
      --set selection.rescue_challenge_min_pcd_gain_by_bucket.near_contact=0.035
      --set selection.rescue_challenge_min_pcd_gain_by_bucket.test_near_contact=0.035
      --set selection.rescue_challenge_min_rec_lcb_gain_by_bucket.near_contact=0.060
      --set selection.rescue_challenge_min_rec_lcb_gain_by_bucket.test_near_contact=0.060
      --set selection.rescue_challenge_min_drs_gain_by_bucket.near_contact=0.000
      --set selection.rescue_challenge_min_drs_gain_by_bucket.test_near_contact=0.000
      --set selection.rescue_challenge_min_gap_reduction_by_bucket.near_contact=0.040
      --set selection.rescue_challenge_min_gap_reduction_by_bucket.test_near_contact=0.040
      --set selection.rescue_challenge_min_improvement_axes_by_bucket.near_contact=3
      --set selection.rescue_challenge_min_improvement_axes_by_bucket.test_near_contact=3
      --set selection.rescue_challenge_macro_allowlist_by_bucket.near_contact=${NEAR_CHALLENGE_MACROS:-merge,yield,stabilize}
      --set selection.rescue_challenge_macro_allowlist_by_bucket.test_near_contact=${NEAR_CHALLENGE_MACROS:-merge,yield,stabilize}
      --set selection.rescue_challenge_max_used_by_bucket.near_contact=${NEAR_CHALLENGE_MAX_USED:-2}
      --set selection.rescue_challenge_max_used_by_bucket.test_near_contact=${NEAR_CHALLENGE_MAX_USED:-2}

      --set selection.rescue_challenge_min_pcd_gain_by_bucket.contact=${CONTACT_CHALLENGE_MIN_PCD_GAIN:-0.000}
      --set selection.rescue_challenge_min_pcd_gain_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_PCD_GAIN:-0.000}
      --set selection.rescue_challenge_min_rec_lcb_gain_by_bucket.contact=${CONTACT_CHALLENGE_MIN_REC_GAIN:--1.000}
      --set selection.rescue_challenge_min_rec_lcb_gain_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_REC_GAIN:--1.000}
      --set selection.rescue_challenge_min_drs_gain_by_bucket.contact=0.000
      --set selection.rescue_challenge_min_drs_gain_by_bucket.test_contact=0.000
      --set selection.rescue_challenge_min_gap_reduction_by_bucket.contact=${CONTACT_CHALLENGE_MIN_GAP_REDUCTION:--1.000}
      --set selection.rescue_challenge_min_gap_reduction_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_GAP_REDUCTION:--1.000}
      --set selection.rescue_challenge_min_improvement_axes_by_bucket.contact=${CONTACT_CHALLENGE_MIN_AXES:-1}
      --set selection.rescue_challenge_min_improvement_axes_by_bucket.test_contact=${CONTACT_CHALLENGE_MIN_AXES:-1}
      --set selection.rescue_challenge_macro_allowlist_by_bucket.contact=${CONTACT_FRONTIER_MACROS:-brake,yield}
      --set selection.rescue_challenge_macro_allowlist_by_bucket.test_contact=${CONTACT_FRONTIER_MACROS:-brake,yield}
      --set selection.rescue_challenge_max_used_by_bucket.contact=${CONTACT_CHALLENGE_MAX_USED:-8}
      --set selection.rescue_challenge_max_used_by_bucket.test_contact=${CONTACT_CHALLENGE_MAX_USED:-8}
      --set selection.rescue_challenge_score_pcd_weight_by_bucket.contact=0.20
      --set selection.rescue_challenge_score_pcd_weight_by_bucket.test_contact=0.20
      --set selection.rescue_challenge_score_drs_weight_by_bucket.contact=1.00
      --set selection.rescue_challenge_score_drs_weight_by_bucket.test_contact=1.00
      --set selection.rescue_challenge_score_gap_weight_by_bucket.contact=0.08
      --set selection.rescue_challenge_score_gap_weight_by_bucket.test_contact=0.08
      --set selection.rescue_challenge_nominal_guard_min_pcd_by_bucket.contact=-1.0
      --set selection.rescue_challenge_nominal_guard_min_pcd_by_bucket.test_contact=-1.0
      --set selection.rescue_challenge_nominal_guard_max_gap_by_bucket.contact=0.08
      --set selection.rescue_challenge_nominal_guard_max_gap_by_bucket.test_contact=0.08
    )
  fi
}

run_eval() {
  local tag="$1"; local d="$2"; local gpu="$3"
  make_sel "$tag"
  local dataset="$SAFE_TEST"
  [[ "$d" == "near_contact" ]] && dataset="$NEAR_TEST"
  [[ "$d" == "contact" ]] && dataset="$CONTACT_TEST"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate \
    --dataset "$dataset" --checkpoint "$CKPT" --calibration "$CAL" --split test \
    --output "$RUN/eval_${d}_v48_${tag}.json" \
    --set evaluation.delta=0.05 \
    --set evaluation.group_by_dataset=true \
    --set evaluation.fallback_to_all_if_empty_split=true \
    --set evaluation.use_running_intervention_budget=${EVAL_RUNNING_BUDGET:-true} \
    "${COMMON_SEL[@]}" \
    --set 'evaluation.methods=[nominal,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]' \
    | tee "$RUN/eval_${d}_v48_${tag}.log"
}

run_audit() {
  local tag="$1"; local b="$2"; local gpu="$3"; local targets="${4:-32}"; local labels="${5:-384}"
  make_sel "$tag"
  local bucket="$NEAR_TEST"
  local shadow_womd_source
  if [[ -n "${DEV_SHADOW_WOMD_SOURCE:-}" ]]; then
    shadow_womd_source="$DEV_SHADOW_WOMD_SOURCE"
  elif [[ "$WOMD_VAL" == *@* ]]; then
    shadow_womd_source="$WOMD_VAL"
  else
    shadow_womd_source="$WOMD_VAL@150"
  fi
  [[ "$b" == "contact" ]] && bucket="$CONTACT_TEST"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$shadow_womd_source" --checkpoint "$CKPT" \
    --output "$RUN/audit_${b}_selected_topk_v48_${tag}.json" \
    "${COMMON_SEL[@]}" \
    --set closed_loop.method=ocrap \
    --set closed_loop.resume="$CL_RESUME" \
    --set closed_loop.resume_allow_legacy_partial=true \
    --set closed_loop.partial_write_every_scenes="$CL_PARTIAL_EVERY" \
    --set closed_loop.resume_fsync="$CL_RESUME_FSYNC" \
    --set closed_loop.save_partial=true \
    --set closed_loop.bucket_dataset="$bucket" \
    --set closed_loop.bucket_split=${BUCKET_SPLIT:-test} \
    --set closed_loop.max_bucket_targets="$targets" \
    --set closed_loop.max_targets_per_scene=${AUDIT_MAX_TARGETS_PER_SCENE:-1} \
    --set closed_loop.max_rollouts=${AUDIT_MAX_ROLLOUTS:-12} \
    --set closed_loop.require_bucket_targets=true \
    --set closed_loop.allow_legacy_source_index_targets=${DEV_SHADOW_ALLOW_LEGACY_SOURCE_INDEX_TARGETS:-true} \
    --set waymax.retain_official_scenario_id=true \
    --set closed_loop.raw_max_scenarios=${DEV_SHADOW_RAW_MAX_SCENARIOS:-0} \
    --set closed_loop.max_steps=${AUDIT_MAX_STEPS:-20} \
    --set closed_loop.replan_interval_steps=${AUDIT_REPLAN_INTERVAL:-1} \
    --set closed_loop.num_candidate_prefixes=${AUDIT_NUM_CANDIDATES:-12} \
    --set closed_loop.num_recovery_options=${AUDIT_NUM_RECOVERY_OPTIONS:-8} \
    --set closed_loop.label_mode=${AUDIT_LABEL_MODE:-selected_topk} \
    --set closed_loop.audit_every_n_steps=${AUDIT_EVERY_N_STEPS:-4} \
    --set closed_loop.audit_max_labels="$labels" \
    --set closed_loop.audit_top_k=${AUDIT_TOP_K:-10} \
    --set closed_loop.audit_max_extra_candidates=${AUDIT_MAX_EXTRA_CANDIDATES:-9} \
    --set closed_loop.render_trace=${CL_RENDER_TRACE:-false} \
    --set closed_loop.render_max_agents=${CL_RENDER_MAX_AGENTS:-64} \
    --set closed_loop.progress_every_steps=1 \
    | tee -a "$RUN/audit_${b}_selected_topk_v48_${tag}.log"
  assert_json "$RUN/audit_${b}_selected_topk_v48_${tag}.json"
}


assert_json() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "missing expected JSON output: $path" >&2
    exit 7
  fi
  python - <<'JSONCHECK' "$path"
import json, sys
with open(sys.argv[1]) as f:
    doc=json.load(f)
if doc.get("bucket_dataset") and int(doc.get("bucket_matched_rollouts", 0) or 0) <= 0:
    raise SystemExit(f"targeted closed-loop audit matched zero rollouts: {sys.argv[1]}")
if doc.get("metrics_valid") is False:
    raise SystemExit(f"closed-loop metrics are invalid/empty: {sys.argv[1]}")
JSONCHECK
}

run_safe_closed_loop_one() {
  local tag="$1" gpu="$2" output="$3"
  make_sel "$tag"
  local safe_womd_source
  if [[ -n "${SAFE_WOMD_SOURCE:-}" ]]; then
    safe_womd_source="$SAFE_WOMD_SOURCE"
  elif [[ "$WOMD_VAL" == *@* ]]; then
    safe_womd_source="$WOMD_VAL"
  else
    safe_womd_source="$WOMD_VAL@150"
  fi
  python tools/validate_womd_shards_v48_16.py --source "$safe_womd_source" --expected-shards 150
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$safe_womd_source" --checkpoint "$CKPT" \
    --output "$output" \
    "${COMMON_SEL[@]}" \
    --set closed_loop.method=ocrap \
    --set closed_loop.resume="$CL_RESUME" \
    --set closed_loop.resume_allow_legacy_partial=true \
    --set closed_loop.partial_write_every_scenes="$CL_PARTIAL_EVERY" \
    --set closed_loop.resume_fsync="$CL_RESUME_FSYNC" \
    --set closed_loop.save_partial=true \
    --set closed_loop.bucket_dataset="$SAFE_TEST" \
    --set closed_loop.bucket_split="${SAFE_BUCKET_SPLIT:-}" \
    --set closed_loop.require_bucket_targets=true \
    --set closed_loop.max_bucket_targets=${SAFE_MAX_TARGETS:-80} \
    --set closed_loop.max_targets_per_scene=1 \
    --set closed_loop.max_rollouts=${SAFE_MAX_ROLLOUTS:-32} \
    --set closed_loop.raw_max_scenarios=${SAFE_RAW_MAX_SCENARIOS:-0} \
    --set closed_loop.max_steps=${SAFE_MAX_STEPS:-40} \
    --set closed_loop.replan_interval_steps=${SAFE_REPLAN_INTERVAL:-1} \
    --set closed_loop.num_candidate_prefixes=${SAFE_NUM_CANDIDATES:-16} \
    --set closed_loop.num_recovery_options=${SAFE_NUM_RECOVERY_OPTIONS:-8} \
    --set closed_loop.label_mode=fast \
    --set closed_loop.render_trace=${CL_RENDER_TRACE:-false} \
    --set closed_loop.render_max_agents=${CL_RENDER_MAX_AGENTS:-64} \
    --set closed_loop.progress_every_steps=5 \
    | tee -a "${output%.json}.log"
  assert_json "$output"
}

run_safe_closed_loop() {
  if [[ "${RUN_SAFE_PAIRED_SCALAR:-0}" == "1" ]]; then
    local scalar_out="$RUN/closed_loop_safe_fast_v48_scalar.json"
    local model_out="$RUN/closed_loop_safe_fast_v48.json"
    run_safe_closed_loop_one scalar "${GPU_SAFE_BASELINE:-0}" "$scalar_out" & local p0=$!
    run_safe_closed_loop_one v48 "${GPU_SAFE:-1}" "$model_out" & local p1=$!
    wait "$p0"; wait "$p1"
    python tools/analyze_safe_paired_noninferiority_v48_8.py \
      --baseline "$scalar_out" --candidate "$model_out" \
      --output "$RUN/safe_paired_noninferiority_v48_8.json"
  else
    run_safe_closed_loop_one v48 "${GPU_SAFE:-0}" "$RUN/closed_loop_safe_fast_v48.json"
  fi
}

summarize() {
  python - <<'PY' "$RUN" | tee "$RUN/summary_all_v48.txt"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
print("\n===== Offline eval v48 =====")
for p in sorted(root.glob('eval_*_v48_*.json')):
    d=json.load(open(p)); print('\n', p.name)
    for m,r in d.get('methods',{}).items():
        if m in ['nominal','backup_filter','contingency','oracle_filter','ocrap','ocrap_teacher']:
            print(f"  {m:14s} FRA={r.get('FRA_exec')} DRS={r.get('DRS')} NUP={r.get('bounded_NUP')} ODG={r.get('ODG')} artifact={r.get('artifact_selection_rate')} PCD={r.get('post_contact_deployability')} int={r.get('intervention_rate')} reason={r.get('selection_reason_counts')}")
print("\n===== Closed-loop/audit v48 =====")
keys=['num_decisions','intervention_rate','closed_loop_bounded_NUP','closed_loop_FRA_exec','closed_loop_DRS','closed_loop_ODG','closed_loop_post_contact_deployability','closed_loop_artifact_selection_rate','closed_loop_audit_best_DRS','closed_loop_audit_best_R_dep','closed_loop_audit_selected_R_dep_regret','closed_loop_audit_best_PCD','closed_loop_audit_selected_PCD_regret','closed_loop_audit_pcd_selector_miss_rate','closed_loop_audit_paper_best_PCD','closed_loop_audit_paper_selected_PCD_regret','closed_loop_audit_paper_pcd_selector_miss_rate','closed_loop_audit_selector_miss_rate','closed_loop_audit_recoverable_candidate_rate','closed_loop_pred_r_dep','closed_loop_pred_gap','closed_loop_pred_DRS_proxy']
for p in sorted(root.glob('*v48*.json')):
    try: d=json.load(open(p))
    except Exception: continue
    if 'closed_loop_bounded_NUP' not in d: continue
    print('\n', p.name)
    for k in keys:
        if k in d: print(f'  {k}: {d[k]}')
    print('  macro_counts:', d.get('macro_counts'))
    print('  audit_best_macro_counts:', d.get('audit_best_macro_counts'))
    print('  audit_miss_best_macro_counts:', d.get('audit_miss_best_macro_counts'))
    print('  audit_pcd_best_macro_counts:', d.get('audit_pcd_best_macro_counts'))
    print('  audit_pcd_miss_best_macro_counts:', d.get('audit_pcd_miss_best_macro_counts'))
    print('  audit_paper_pcd_best_macro_counts:', d.get('audit_paper_pcd_best_macro_counts'))
    print('  audit_paper_pcd_miss_best_macro_counts:', d.get('audit_paper_pcd_miss_best_macro_counts'))
    print('  audit_miss_selected_macro_counts:', d.get('audit_miss_selected_macro_counts'))
    print('  selection_reason_counts:', d.get('selection_reason_counts'))
PY
}

# Staged execution switches.  Training/offline diagnostics should eliminate a
# weak candidate before any Waymax audit; the quick gate uses six rollouts per
# stress regime. Safe closed-loop is deferred until a stress candidate passes.
RUN_OFFLINE_EVAL=${RUN_OFFLINE_EVAL:-1}
RUN_AUDITS=${RUN_AUDITS:-1}
RUN_SAFE_CLOSED_LOOP=${RUN_SAFE_CLOSED_LOOP:-0}
if [[ "$RUN_OFFLINE_EVAL" == "1" ]]; then
  if [[ "${RUN_SCALAR_BASELINES:-1}" == "1" ]]; then
    run_eval scalar safe 0 & run_eval v48 safe 1 & wait
    run_eval scalar near_contact 0 & run_eval v48 near_contact 1 & wait
    run_eval scalar contact 0 & run_eval v48 contact 1 & wait
  else
    if [[ "${OFFLINE_SINGLE_GPU:-0}" == "1" ]]; then
      g=${GPU_SAFE:-0}
      run_eval v48 safe "$g"
      run_eval v48 near_contact "$g"
      run_eval v48 contact "$g"
    else
      run_eval v48 safe ${GPU_SAFE:-0} & run_eval v48 near_contact ${GPU_NEAR:-1} & wait
      run_eval v48 contact ${GPU_CONTACT:-0}
    fi
  fi
fi
if [[ "$RUN_AUDITS" == "1" ]]; then
  RUN_NEAR_AUDIT=${RUN_NEAR_AUDIT:-1}
  RUN_CONTACT_AUDIT=${RUN_CONTACT_AUDIT:-1}
  if [[ "${RUN_SCALAR_BASELINES:-1}" == "1" && "${PAIR_BUCKETS_ON_TWO_GPUS:-1}" == "1" ]]; then
    if [[ "$RUN_NEAR_AUDIT" == "1" ]]; then
      (run_audit scalar near_contact ${GPU_NEAR:-0} ${AUDIT_TARGETS:-16} ${AUDIT_LABELS:-192}; \
       run_audit v48 near_contact ${GPU_NEAR:-0} ${AUDIT_TARGETS:-16} ${AUDIT_LABELS:-192}) &
      P_NEAR=$!
    fi
    if [[ "$RUN_CONTACT_AUDIT" == "1" ]]; then
      (run_audit scalar contact ${GPU_CONTACT:-1} ${AUDIT_TARGETS:-16} ${AUDIT_LABELS:-192}; \
       run_audit v48 contact ${GPU_CONTACT:-1} ${AUDIT_TARGETS:-16} ${AUDIT_LABELS:-192}) &
      P_CONTACT=$!
    fi
    [[ -z "${P_NEAR:-}" ]] || wait "$P_NEAR"
    [[ -z "${P_CONTACT:-}" ]] || wait "$P_CONTACT"
  elif [[ "${RUN_SCALAR_BASELINES:-1}" == "1" ]]; then
    [[ "$RUN_NEAR_AUDIT" != "1" ]] || { run_audit scalar near_contact 0 ${AUDIT_TARGETS:-16} ${AUDIT_LABELS:-192} & run_audit v48 near_contact 1 ${AUDIT_TARGETS:-16} ${AUDIT_LABELS:-192} & wait; }
    [[ "$RUN_CONTACT_AUDIT" != "1" ]] || { run_audit scalar contact 0 ${AUDIT_TARGETS:-16} ${AUDIT_LABELS:-192} & run_audit v48 contact 1 ${AUDIT_TARGETS:-16} ${AUDIT_LABELS:-192} & wait; }
  else
    if [[ "$RUN_NEAR_AUDIT" == "1" ]]; then
      run_audit v48 near_contact ${GPU_NEAR:-0} ${AUDIT_TARGETS:-16} ${AUDIT_LABELS:-192} &
      P_NEAR=$!
    fi
    if [[ "$RUN_CONTACT_AUDIT" == "1" ]]; then
      run_audit v48 contact ${GPU_CONTACT:-1} ${AUDIT_TARGETS:-16} ${AUDIT_LABELS:-192} &
      P_CONTACT=$!
    fi
    [[ -z "${P_NEAR:-}" ]] || wait "$P_NEAR"
    [[ -z "${P_CONTACT:-}" ]] || wait "$P_CONTACT"
  fi
fi
if [[ "$RUN_SAFE_CLOSED_LOOP" == "1" ]]; then run_safe_closed_loop; fi
summarize
