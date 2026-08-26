#!/usr/bin/env bash
set -euo pipefail

# v16.8.30: Successor Option Viability Recovery (SOVR) + RVR Pareto diagnostic guard.
# Scientific discipline:
# - NO dataset/label/cache rebuild, NO retraining, NO RCOT/BCOT/certificate/frontier retuning.
# - The two new branches act ONLY when full conventional-safe set is empty and valid candidates exist.
# - counterfactual48 / balanced_dev96 / exact200 are DEVELOPMENT sets, never final paper evidence.
# - Unchanged v16.8.29 COWP/RVR exact results are bundled as references, so confirm no longer reruns COWP by default.

MODE="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
BASE_RUN="${BASE_RUN:-outputs/v16_8_24_compact5k_all}"
BASE_CKPT="${BASE_CKPT:-$BASE_RUN/cowp_all_best.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/v16_8_30_successor_option_viability}"
TFEX_INDEX="${TFEX_INDEX:-$COWP_ROOT/tfexample_id_index_validation.jsonl}"
BCOT_BUDGET="${BCOT_BUDGET:-0.50}"

EXACT_IDS="$SCRIPT_DIR/reference_manifests/waymax_probe_200_ids.txt"
EQ16_IDS="$SCRIPT_DIR/reference_manifests/waymax_v16_8_30_equivalence16_ids.txt"
CF48_IDS="$SCRIPT_DIR/reference_manifests/waymax_v16_8_30_counterfactual48_ids.txt"
DEV96_IDS="$SCRIPT_DIR/reference_manifests/waymax_v16_8_30_balanced_dev96_ids.txt"
REF_EQ16="$SCRIPT_DIR/reference_results/v16_8_29_equivalence16_cowp_reference.json"
REF_CF48_COWP="$SCRIPT_DIR/reference_results/v16_8_29_counterfactual48_cowp_reference.json"
REF_CF48_RVR="$SCRIPT_DIR/reference_results/v16_8_29_counterfactual48_rvr_reference.json"
REF_DEV96_COWP="$SCRIPT_DIR/reference_results/v16_8_29_balanced_dev96_cowp_reference.json"
REF_DEV96_RVR="$SCRIPT_DIR/reference_results/v16_8_29_balanced_dev96_rvr_reference.json"
REF_EXACT_COWP="$SCRIPT_DIR/reference_results/v16_8_29_exact200_cowp_reference.json"
REF_EXACT_RVR="$SCRIPT_DIR/reference_results/v16_8_29_exact200_rvr_reference.json"

EXPECTED_EXACT_SHA="3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f"
EXPECTED_EQ16_SHA="81d0319da0446d1452b4c3a0361ffa6941dfa226b2f14027cac5576f9571c760"
EXPECTED_CF48_SHA="ee3c231c240878d5d20020aec3c98efbb4932cdbf1f1e309b9b7b26bddc40ab0"
EXPECTED_DEV96_SHA="8ca509bd1263aec10e31fbd4a4ff2df21ae22b83287efce61b072897de8e7783"
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
  validate_manifest "$DEV96_IDS" 96 "$EXPECTED_DEV96_SHA"
  cp "$EXACT_IDS" "$OUT_ROOT/waymax_exact_200_ids.txt"
  cp "$EQ16_IDS" "$OUT_ROOT/waymax_equivalence16_ids.txt"
  cp "$CF48_IDS" "$OUT_ROOT/waymax_counterfactual48_ids.txt"
  cp "$DEV96_IDS" "$OUT_ROOT/waymax_balanced_dev96_ids.txt"
}

index_arg=()
if [[ -s "$TFEX_INDEX" ]]; then
  index_arg=(--tfexample-index-jsonl "$TFEX_INDEX")
else
  echo "[warning] TFExample ID index missing: $TFEX_INDEX" >&2
  echo "          correctness is unchanged; exact-ID loading may be slower." >&2
fi

common_waymax=(
  --data-config configs/data.yaml
  --label-config configs/label_cowp_v16_8.yaml
  --eval-config configs/eval_cowp_v16_8.yaml
  --mode waymax --checkpoint "$BASE_CKPT" --waymax-split validation
  --rollout-horizon-steps 80 --waymax-action-mode absolute_xy_yaw
  --ncf-gate-mode priority --witness-threshold 0.70 --bcot-risk-budget "$BCOT_BUDGET"
  --outcome-risk-penalty 0 --waymax-standard-metrics
  --waymax-standard-metric-names OverlapMetric,OffroadMetric,ProgressionMetric,KinematicsInfeasibilityMetric
  --reuse-waymax-env --prefilter-waymax-shards --jit-waymax-env --jit-waymax-metrics
)

run_colocated_shard() {
  local gpu="$1" shard="$2" method="$3" ids="$4" out="$5"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m cowp.scripts.04_eval_closed_loop "${common_waymax[@]}" "${index_arg[@]}" \
    --method "$method" --scenario-ids-file "$ids" --device cuda:0 \
    --waymax-device gpu --jax-visible-devices 0 --jax-preallocate false \
    --num-shards 2 --shard-index "$shard" \
    --no-progress --status-every 2 --output "$out"
}

run_parallel_methods() {
  local tag="$1" ids="$2" methods="$3"
  IFS=',' read -r -a arr <<< "$methods"
  for method in "${arr[@]}"; do
    echo "=== $tag method: $method ==="
    local start end
    start=$(date +%s)
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
  local tag="$1" cowp="$2" rvr="$3" devflag="$4"
  local extra=()
  [[ "$devflag" == "1" ]] && extra=(--development-selected)
  python -m cowp.scripts.83_analyze_successor_option_viability \
    --cowp "$cowp" --rvr "$rvr" \
    --guard "$OUT_ROOT/${tag}_cowp_rvr_pareto_guard_merged.json" \
    --successor "$OUT_ROOT/${tag}_cowp_successor_option_viability_merged.json" \
    "${extra[@]}" --output "$OUT_ROOT/${tag}_successor_option_analysis.json"
}

case "$MODE" in
  sanity)
    python -m py_compile \
      cowp/waymax_eval/policy_wrapper.py cowp/waymax_eval/rollout.py cowp/waymax_eval/metrics_cowp.py \
      cowp/scripts/04_eval_closed_loop.py cowp/scripts/79_merge_waymax_exact_shards.py \
      cowp/scripts/82_verify_cowp_equivalence.py cowp/scripts/83_analyze_successor_option_viability.py
    python -m pytest -q \
      tests/test_v16_8_30_successor_option_viability.py \
      tests/test_v16_8_29_recursive_viability.py \
      tests/test_v16_8_28_execution_integrity.py \
      tests/test_v16_8_27_integrity.py \
      tests/test_v16_8_26_fallback_outcome.py \
      tests/test_v16_8_25_ctu_selector.py
    if grep -R "conventional_check=False" -n cowp/waymax_eval/policy_wrapper.py; then
      echo "ERROR: conventional-safety bypass exists" >&2; exit 3
    fi
    ensure_ids
    echo "v16.8.30 focused semantic/integrity sanity passed"
    ;;

  make_ids)
    ensure_ids
    ;;

  build_tfindex)
    python -m cowp.scripts.78_build_tfexample_id_index \
      --data-config configs/data.yaml --split validation --output "$TFEX_INDEX"
    ;;

  base_equivalence16_parallel2)
    ensure_ids
    run_parallel_methods "equivalence16" "$EQ16_IDS" "cowp"
    python -m cowp.scripts.82_verify_cowp_equivalence \
      --reference "$REF_EQ16" \
      --candidate "$OUT_ROOT/equivalence16_cowp_merged.json" \
      --output "$OUT_ROOT/equivalence16_cowp_vs_v16_8_29.json"
    ;;

  counterfactual48_parallel2)
    ensure_ids
    run_parallel_methods "counterfactual48" "$CF48_IDS" "${METHODS:-cowp_rvr_pareto_guard,cowp_successor_option_viability}"
    ;;

  analyze_counterfactual48)
    analyze_set "counterfactual48" "$REF_CF48_COWP" "$REF_CF48_RVR" 1
    ;;

  balanced96_parallel2)
    ensure_ids
    run_parallel_methods "balanced_dev96" "$DEV96_IDS" "${METHODS:-cowp_rvr_pareto_guard,cowp_successor_option_viability}"
    ;;

  analyze_balanced96)
    analyze_set "balanced_dev96" "$REF_DEV96_COWP" "$REF_DEV96_RVR" 1
    ;;

  confirm200_parallel2)
    ensure_ids
    # By default run ONLY the promoted new method.  The unchanged COWP/RVR exact200
    # outputs are bundled as immutable references, saving ~53 min of redundant COWP.
    run_parallel_methods "confirm_exact200" "$EXACT_IDS" "${PROMOTED_METHODS:-cowp_successor_option_viability}"
    ;;

  analyze_confirm200)
    # If only successor was promoted, synthesize the guard arg only if it exists.
    args=(--cowp "$REF_EXACT_COWP" --rvr "$REF_EXACT_RVR" --output "$OUT_ROOT/confirm_exact200_successor_option_analysis.json")
    [[ -s "$OUT_ROOT/confirm_exact200_cowp_rvr_pareto_guard_merged.json" ]] && args+=(--guard "$OUT_ROOT/confirm_exact200_cowp_rvr_pareto_guard_merged.json")
    [[ -s "$OUT_ROOT/confirm_exact200_cowp_successor_option_viability_merged.json" ]] && args+=(--successor "$OUT_ROOT/confirm_exact200_cowp_successor_option_viability_merged.json")
    python -m cowp.scripts.83_analyze_successor_option_viability "${args[@]}"
    ;;

  help|*)
    cat <<EOF
v16.8.30 recommended order

0) Point BASE_RUN/BASE_CKPT to the existing v16.8.24 checkpoint. Do not retrain.

1) Mandatory:
   $0 sanity
   $0 make_ids

2) One small common-path equivalence gate (16 scenes only):
   $0 base_equivalence16_parallel2
   This must pass before interpreting either new branch.

3) Mechanism counterexample gate (48 selected dev scenes, NOT paper evidence):
   $0 counterfactual48_parallel2
   $0 analyze_counterfactual48
   Composition = all v16.8.29 10 rescued + 9 induced + 24 shared-collision + 5 stable controls.
   This directly tests whether a branch can retain RVR's rescues while eliminating its induced failures.

4) ONLY if step 3 is favorable, broader regression screen (96 dev scenes):
   $0 balanced96_parallel2
   $0 analyze_balanced96

5) ONLY after step 4 promotes one branch:
   PROMOTED_METHODS=cowp_successor_option_viability $0 confirm200_parallel2
   $0 analyze_confirm200
   COWP is NOT rerun by default: v16.8.29 exact200 COWP/RVR are bundled references.

Branch meaning:
- cowp_rvr_pareto_guard = diagnostic only. It tests whether RVR failed because max-prefix
  overrode existing transport/rule/action/pressure evidence. Do not promote as paper novelty.
- cowp_successor_option_viability = main mechanistic branch. When conventional set is empty,
  compare the original COWP fallback vs the RVR alternative by the option set available after
  their ACTUAL one-step projected actions. Switch only if RVR strictly improves the lexicographic
  successor signature: conventional existence -> distinct conventional macros -> conventional
  candidate count -> best drivable collision-safe prefix. Equal successor support keeps COWP.

Do NOT tune prefix weights, shorten the 8 s conventional horizon, retune RCOT/BCOT/outcome,
or add proposal primitives in this round. If both branches fail and successor support is empty
for both choices, that is the clean evidence needed to move to structured proposal refinement.
EOF
    ;;
esac
