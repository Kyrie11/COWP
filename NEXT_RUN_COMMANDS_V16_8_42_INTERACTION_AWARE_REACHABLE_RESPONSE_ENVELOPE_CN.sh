#!/usr/bin/env bash
set -euo pipefail

# v16.8.42: Root-Conditioned Interaction-Aware Reachable-Response Envelope (RC-IARE)
#
# Scientific branch selected by the frozen V40/V41 decision rule after repaired
# V41 produced real interval-only actions but failed the inherited outcome gate.
# The exact V39 conflict-window tube is the nested first branch. Only on a
# V39-empty fallback step may RC-IARE replace an exact fixed-path blocker with a
# universal high-mass same-root response envelope. Every retained root must own
# a low-burden, drivable, Waymax-kinematic response for current and one-step-
# shifted ego tubes; simultaneous blockers must admit a jointly compatible
# response assignment. All remaining physical/model/protocol layers stay frozen.
#
# Counterfactual48 is development evidence. The six-item conjunction gate is
# unchanged and fail-closed. Mechanism diagnostics never override that gate.

MODE="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
BASE_RUN="${BASE_RUN:-outputs/v16_8_24_compact5k_all}"
BASE_CKPT="${BASE_CKPT:-$BASE_RUN/cowp_all_best.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/v16_8_42_interaction_aware_reachable_response_envelope}"
TFEX_INDEX="${TFEX_INDEX:-$COWP_ROOT/tfexample_id_index_validation.jsonl}"
BCOT_BUDGET="${BCOT_BUDGET:-0.50}"

EXACT_IDS="$SCRIPT_DIR/reference_manifests/waymax_probe_200_ids.txt"
EQ16_IDS="$SCRIPT_DIR/reference_manifests/waymax_v16_8_30_equivalence16_ids.txt"
CF48_IDS="$SCRIPT_DIR/reference_manifests/waymax_v16_8_30_counterfactual48_ids.txt"
FRESH37_IDS="$SCRIPT_DIR/reference_manifests/waymax_v16_8_32_fresh37_ids.txt"

REF_EQ16="$SCRIPT_DIR/reference_results/v16_8_29_equivalence16_cowp_reference.json"
REF_CF48_COWP="$SCRIPT_DIR/reference_results/v16_8_29_counterfactual48_cowp_reference.json"
REF_CF48_RVR="$SCRIPT_DIR/reference_results/v16_8_29_counterfactual48_rvr_reference.json"
REF_CF48_V33_ROSH="$SCRIPT_DIR/reference_results/v16_8_33_counterfactual48_rosh_reference.json"
REF_CF48_V35_CPOSH="$SCRIPT_DIR/reference_results/v16_8_35_counterfactual48_cposh_reference.json"
REF_CF48_V36_FRONTIER="$SCRIPT_DIR/reference_results/v16_8_36_counterfactual48_recovery_frontier_reference.json"
REF_CF48_V37_RRB="$SCRIPT_DIR/reference_results/v16_8_37_counterfactual48_recourse_returnability_bridge_reference.json"
REF_CF48_V38_TUBE="$SCRIPT_DIR/reference_results/v16_8_38_counterfactual48_shift_closed_control_reachable_tube_reference.json"
REF_CF48_V39_TUBE="$SCRIPT_DIR/reference_results/v16_8_39_counterfactual48_conflict_window_control_reachable_tube_reference.json"
REF_FRESH37_COWP="$SCRIPT_DIR/reference_results/v16_8_32_fresh37_cowp_reference.json"
REF_FRESH37_RVR="$SCRIPT_DIR/reference_results/v16_8_32_fresh37_rvr_reference.json"
REF_EXACT_COWP="$SCRIPT_DIR/reference_results/v16_8_29_exact200_cowp_reference.json"
REF_EXACT_RVR="$SCRIPT_DIR/reference_results/v16_8_29_exact200_rvr_reference.json"

EXPECTED_EXACT_SHA="3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f"
EXPECTED_EQ16_SHA="81d0319da0446d1452b4c3a0361ffa6941dfa226b2f14027cac5576f9571c760"
EXPECTED_CF48_SHA="ee3c231c240878d5d20020aec3c98efbb4932cdbf1f1e309b9b7b26bddc40ab0"
EXPECTED_FRESH37_SHA="ecce3321d8f4cd57bbd3189b3673784bec8fde185b882e9c11c38430265a1481"
mkdir -p "$OUT_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

validate_manifest() {
  local p="$1" n="$2" expected="$3"
  [[ -s "$p" ]] || { echo "Missing ID manifest: $p" >&2; exit 2; }
  python - "$p" "$n" "$expected" <<'PY'
import hashlib,sys
p,n,expected=sys.argv[1],int(sys.argv[2]),sys.argv[3]
ids=[x.strip() for x in open(p,encoding='utf-8') if x.strip()]
if len(ids)!=n: raise SystemExit(f"expected {n} IDs, got {len(ids)} in {p}")
if len(set(ids))!=n: raise SystemExit(f"duplicate IDs in {p}")
h=hashlib.sha256("\n".join(ids).encode()).hexdigest()
if h!=expected: raise SystemExit(f"logical ID SHA mismatch: {h} != {expected}")
print(f"manifest OK: n={n} logical_sha256={h} path={p}")
PY
}
ensure_ids() {
  validate_manifest "$EXACT_IDS" 200 "$EXPECTED_EXACT_SHA"
  validate_manifest "$EQ16_IDS" 16 "$EXPECTED_EQ16_SHA"
  validate_manifest "$CF48_IDS" 48 "$EXPECTED_CF48_SHA"
  validate_manifest "$FRESH37_IDS" 37 "$EXPECTED_FRESH37_SHA"
  cp "$EXACT_IDS" "$OUT_ROOT/waymax_exact_200_ids.txt"
  cp "$EQ16_IDS" "$OUT_ROOT/waymax_equivalence16_ids.txt"
  cp "$CF48_IDS" "$OUT_ROOT/waymax_counterfactual48_ids.txt"
  cp "$FRESH37_IDS" "$OUT_ROOT/waymax_v16_8_32_fresh37_ids.txt"
}

require_gate_pass() {
  local analysis_json="$1" methods="$2" stage_label="$3"
  [[ -s "$analysis_json" ]] || { echo "ERROR: missing $stage_label analysis: $analysis_json" >&2; exit 4; }
  python - "$analysis_json" "$methods" "$stage_label" <<'PY' || exit 4
import json,sys
p,methods,stage=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(p,encoding='utf-8'))
gates=d.get('preregistered_gate',{})
for method in [x.strip() for x in methods.split(',') if x.strip()]:
    if method!='cowp_interaction_aware_reachable_response_envelope':
        raise SystemExit(f'ERROR: {method} is not the V16.8.42 RC-IARE promotion candidate')
    g=gates.get('interaction_aware_reachable_response_envelope')
    if not g or g.get('pass') is not True:
        failed=[k for k,v in (g or {}).get('checks',{}).items() if not v]
        raise SystemExit(f'ERROR: {method} did not pass {stage} preregistered gate; failed={failed}')
print(f'promotion gate OK for {methods} at {stage}')
PY
}

index_arg=()
if [[ -s "$TFEX_INDEX" ]]; then
  index_arg=(--tfexample-index-jsonl "$TFEX_INDEX")
else
  echo "[warning] TFExample ID index missing: $TFEX_INDEX" >&2
fi
common_waymax=(
  --data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml --eval-config configs/eval_cowp_v16_8.yaml
  --mode waymax --checkpoint "$BASE_CKPT" --waymax-split validation --rollout-horizon-steps 80
  --waymax-action-mode absolute_xy_yaw --ncf-gate-mode priority --witness-threshold 0.70 --bcot-risk-budget "$BCOT_BUDGET"
  --outcome-risk-penalty 0 --waymax-standard-metrics
  --waymax-standard-metric-names OverlapMetric,OffroadMetric,ProgressionMetric,KinematicsInfeasibilityMetric
  --reuse-waymax-env --prefilter-waymax-shards --jit-waymax-env --jit-waymax-metrics
)
run_colocated_shard() {
  local gpu="$1" shard="$2" method="$3" ids="$4" out="$5"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m cowp.scripts.04_eval_closed_loop "${common_waymax[@]}" "${index_arg[@]}" \
    --method "$method" --scenario-ids-file "$ids" --device cuda:0 --waymax-device gpu \
    --jax-visible-devices 0 --jax-preallocate false --num-shards 2 --shard-index "$shard" \
    --no-progress --status-every 2 --output "$out"
}
run_parallel_methods() {
  local tag="$1" ids="$2" methods="$3"
  IFS=',' read -r -a arr <<< "$methods"
  for method in "${arr[@]}"; do
    echo "=== $tag method: $method ==="
    local start end; start=$(date +%s)
    run_colocated_shard 0 0 "$method" "$ids" "$OUT_ROOT/${tag}_${method}_s0.json" & p0=$!
    run_colocated_shard 1 1 "$method" "$ids" "$OUT_ROOT/${tag}_${method}_s1.json" & p1=$!
    wait "$p0"; wait "$p1"
    end=$(date +%s); echo $((end-start)) > "$OUT_ROOT/${tag}_${method}_wall_seconds.txt"
    python -m cowp.scripts.79_merge_waymax_exact_shards \
      --inputs "$OUT_ROOT/${tag}_${method}_s0.json" "$OUT_ROOT/${tag}_${method}_s1.json" \
      --output "$OUT_ROOT/${tag}_${method}_merged.json"
  done
}
analyze_set() {
  local stage="$1" tag="$2" cowp="$3" rvr="$4"
  local args=(--cowp "$cowp" --rvr "$rvr" --stage "$stage" --development-selected \
    --interaction-aware-reachable-response-envelope "$OUT_ROOT/${tag}_cowp_interaction_aware_reachable_response_envelope_merged.json" \
    --output "$OUT_ROOT/${tag}_interaction_aware_reachable_response_envelope_analysis.json")
  if [[ "$stage" == "counterfactual48" ]]; then
    args+=(--v33-rosh "$REF_CF48_V33_ROSH" \
      --v35-control-projected-spectrum "$REF_CF48_V35_CPOSH" \
      --v36-recovery-frontier "$REF_CF48_V36_FRONTIER" \
      --v37-recourse-returnability-bridge "$REF_CF48_V37_RRB" \
      --v38-shift-closed-tube "$REF_CF48_V38_TUBE" \
      --v39-conflict-window-tube "$REF_CF48_V39_TUBE")
  fi
  python -m cowp.scripts.95_analyze_interaction_aware_reachable_response_envelope "${args[@]}"
}

case "$MODE" in
  sanity)
    python -m py_compile cowp/waymax_eval/policy_wrapper.py cowp/waymax_eval/rollout.py cowp/waymax_eval/metrics_cowp.py \
      cowp/scripts/04_eval_closed_loop.py cowp/scripts/79_merge_waymax_exact_shards.py \
      cowp/scripts/82_verify_cowp_equivalence.py cowp/scripts/88_analyze_control_projected_option_spectrum.py \
      cowp/scripts/89_analyze_control_projected_recovery_frontier.py cowp/scripts/90_analyze_recourse_returnability_bridge.py \
      cowp/scripts/92_analyze_conflict_window_control_reachable_tube.py cowp/scripts/93_analyze_shift_closed_first_action_viability_interval.py cowp/scripts/94_analyze_shift_closure_semantic_fidelity_repair.py cowp/scripts/95_analyze_interaction_aware_reachable_response_envelope.py
    python -m pytest -q \
      tests/test_v16_8_42_interaction_aware_reachable_response_envelope.py \
      tests/test_v16_8_41_shift_closure_semantic_fidelity_repair.py \
      tests/test_v16_8_40_shift_closed_first_action_viability_interval.py \
      tests/test_v16_8_39_conflict_window_control_reachable_tube.py \
      tests/test_v16_8_38_shift_closed_control_reachable_tube.py \
      tests/test_v16_8_37_recourse_returnability_bridge.py \
      tests/test_v16_8_36_control_projected_recovery_frontier.py \
      tests/test_v16_8_35_control_projected_option_spectrum.py tests/test_v16_8_34_executable_option_spectrum.py \
      tests/test_v16_8_33_recovery_option_spectrum.py tests/test_v16_8_32_temporal_option_persistence.py \
      tests/test_v16_8_31_bihorizon_option_viability.py tests/test_v16_8_30_successor_option_viability.py \
      tests/test_v16_8_29_recursive_viability.py tests/test_v16_8_28_execution_integrity.py \
      tests/test_v16_8_27_integrity.py tests/test_v16_8_26_fallback_outcome.py tests/test_v16_8_25_ctu_selector.py
    if grep -R "conventional_check=False" -n cowp/waymax_eval/policy_wrapper.py; then
      echo "ERROR: conventional-safety bypass exists" >&2; exit 3
    fi
    ensure_ids
    echo "v16.8.42 RC-IARE semantic/integrity sanity passed"
    ;;
  make_ids) ensure_ids ;;
  build_tfindex)
    python -m cowp.scripts.78_build_tfexample_id_index --data-config configs/data.yaml --split validation --output "$TFEX_INDEX" ;;
  base_equivalence16_parallel2)
    ensure_ids; run_parallel_methods "equivalence16" "$EQ16_IDS" "cowp"
    python -m cowp.scripts.82_verify_cowp_equivalence --reference "$REF_EQ16" \
      --candidate "$OUT_ROOT/equivalence16_cowp_merged.json" --output "$OUT_ROOT/equivalence16_cowp_vs_v16_8_29.json" ;;
  counterfactual48_parallel2)
    ensure_ids
    run_parallel_methods "counterfactual48_v42" "$CF48_IDS" "${METHODS:-cowp_interaction_aware_reachable_response_envelope}" ;;
  analyze_counterfactual48)
    analyze_set "counterfactual48" "counterfactual48_v42" "$REF_CF48_COWP" "$REF_CF48_RVR" ;;
  fresh37_parallel2)
    ensure_ids
    promoted="${PROMOTED_METHODS:-cowp_interaction_aware_reachable_response_envelope}"
    require_gate_pass "$OUT_ROOT/counterfactual48_v42_interaction_aware_reachable_response_envelope_analysis.json" "$promoted" "counterfactual48"
    run_parallel_methods "fresh37_v42" "$FRESH37_IDS" "$promoted" ;;
  analyze_fresh37)
    analyze_set "fresh37" "fresh37_v42" "$REF_FRESH37_COWP" "$REF_FRESH37_RVR" ;;
  confirm200_parallel2)
    ensure_ids
    promoted="${PROMOTED_METHODS:-cowp_interaction_aware_reachable_response_envelope}"
    require_gate_pass "$OUT_ROOT/fresh37_v42_interaction_aware_reachable_response_envelope_analysis.json" "$promoted" "fresh37"
    run_parallel_methods "confirm_exact200" "$EXACT_IDS" "$promoted" ;;
  analyze_confirm200)
    analyze_set "exact200" "confirm_exact200" "$REF_EXACT_COWP" "$REF_EXACT_RVR" ;;
  help|*)
    cat <<EOF
V16.8.42 RC-IARE fail-closed run order

1) $0 sanity
2) $0 make_ids
3) $0 base_equivalence16_parallel2
4) $0 counterfactual48_parallel2
5) $0 analyze_counterfactual48

STOP unless:
  preregistered_gate.interaction_aware_reachable_response_envelope.pass == true

The counterfactual48 Stage-1 conjunction remains exactly:
  retain >=5/10 historical RVR rescues; avoid >=7/9 RVR-induced collisions;
  remove >=3 net COWP collisions; add <=1 net kinematics failure;
  paired mean EP delta >= -0.05; nonzero action-changing intervention.

Only after Stage-1 pass:
  PROMOTED_METHODS=cowp_interaction_aware_reachable_response_envelope $0 fresh37_parallel2
  $0 analyze_fresh37

Only after fresh37 pass:
  PROMOTED_METHODS=cowp_interaction_aware_reachable_response_envelope $0 confirm200_parallel2
  $0 analyze_confirm200

Do not run fresh37 after a Stage-1 failure. Do not tune root probability floors,
root mass coverage, burden budgets, response profiles, release schedules, action
grids, controller limits, horizon, or scalar scores on counterfactual48. RC-IARE
mechanism fields are attribution diagnostics, not substitute promotion criteria.
EOF
    ;;
esac
