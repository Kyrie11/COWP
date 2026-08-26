#!/usr/bin/env bash
set -euo pipefail

# v16.8.31: Bi-Horizon Option Viability (BHOV) + successor-restoration diagnostic.
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
OUT_ROOT="${OUT_ROOT:-outputs/v16_8_31_bihorizon_option_viability}"
TFEX_INDEX="${TFEX_INDEX:-$COWP_ROOT/tfexample_id_index_validation.jsonl}"
BCOT_BUDGET="${BCOT_BUDGET:-0.50}"

EXACT_IDS="$SCRIPT_DIR/reference_manifests/waymax_probe_200_ids.txt"
EQ16_IDS="$SCRIPT_DIR/reference_manifests/waymax_v16_8_30_equivalence16_ids.txt"
CF48_IDS="$SCRIPT_DIR/reference_manifests/waymax_v16_8_30_counterfactual48_ids.txt"
DEV96_IDS="$SCRIPT_DIR/reference_manifests/waymax_v16_8_30_balanced_dev96_ids.txt"
HOLD64_IDS="$SCRIPT_DIR/reference_manifests/waymax_v16_8_31_holdout64_ids.txt"
REF_EQ16="$SCRIPT_DIR/reference_results/v16_8_29_equivalence16_cowp_reference.json"
REF_CF48_COWP="$SCRIPT_DIR/reference_results/v16_8_29_counterfactual48_cowp_reference.json"
REF_CF48_RVR="$SCRIPT_DIR/reference_results/v16_8_29_counterfactual48_rvr_reference.json"
REF_DEV96_COWP="$SCRIPT_DIR/reference_results/v16_8_29_balanced_dev96_cowp_reference.json"
REF_DEV96_RVR="$SCRIPT_DIR/reference_results/v16_8_29_balanced_dev96_rvr_reference.json"
REF_EXACT_COWP="$SCRIPT_DIR/reference_results/v16_8_29_exact200_cowp_reference.json"
REF_EXACT_RVR="$SCRIPT_DIR/reference_results/v16_8_29_exact200_rvr_reference.json"
REF_CF48_SOV="$SCRIPT_DIR/reference_results/v16_8_30_counterfactual48_sov_reference.json"
REF_HOLD64_COWP="$SCRIPT_DIR/reference_results/v16_8_31_holdout64_cowp_reference.json"
REF_HOLD64_RVR="$SCRIPT_DIR/reference_results/v16_8_31_holdout64_rvr_reference.json"

EXPECTED_EXACT_SHA="3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f"
EXPECTED_EQ16_SHA="81d0319da0446d1452b4c3a0361ffa6941dfa226b2f14027cac5576f9571c760"
EXPECTED_CF48_SHA="ee3c231c240878d5d20020aec3c98efbb4932cdbf1f1e309b9b7b26bddc40ab0"
EXPECTED_DEV96_SHA="8ca509bd1263aec10e31fbd4a4ff2df21ae22b83287efce61b072897de8e7783"
EXPECTED_HOLD64_SHA="becdc8430e14bd76190e3446206bed8e7cb9afb966290978e9bdaa61a5202e79"
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
  validate_manifest "$HOLD64_IDS" 64 "$EXPECTED_HOLD64_SHA"
  cp "$EXACT_IDS" "$OUT_ROOT/waymax_exact_200_ids.txt"
  cp "$EQ16_IDS" "$OUT_ROOT/waymax_equivalence16_ids.txt"
  cp "$CF48_IDS" "$OUT_ROOT/waymax_counterfactual48_ids.txt"
  cp "$DEV96_IDS" "$OUT_ROOT/waymax_balanced_dev96_ids.txt"
  cp "$HOLD64_IDS" "$OUT_ROOT/waymax_v16_8_31_holdout64_ids.txt"
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
      cowp/scripts/82_verify_cowp_equivalence.py cowp/scripts/83_analyze_successor_option_viability.py cowp/scripts/84_analyze_bihorizon_option_viability.py
    python -m pytest -q \
      tests/test_v16_8_31_bihorizon_option_viability.py \
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
    echo "v16.8.31 focused semantic/integrity sanity passed"
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
    run_parallel_methods "counterfactual48_v31" "$CF48_IDS" "${METHODS:-cowp_bihorizon_option_viability,cowp_successor_restore_only}"
    ;;

  analyze_counterfactual48)
    python -m cowp.scripts.84_analyze_bihorizon_option_viability \
      --cowp "$REF_CF48_COWP" --rvr "$REF_CF48_RVR" --sov "$REF_CF48_SOV" \
      --bihorizon "$OUT_ROOT/counterfactual48_v31_cowp_bihorizon_option_viability_merged.json" \
      --restore "$OUT_ROOT/counterfactual48_v31_cowp_successor_restore_only_merged.json" \
      --development-selected --output "$OUT_ROOT/counterfactual48_v31_bihorizon_analysis.json"
    ;;

  holdout64_parallel2)
    ensure_ids
    # Outcome-blind within the already-development exact200 pool: SHA256-selected
    # from the 101 IDs outside every v16.8.30 equivalence16/counterfactual48/balanced96 panel.  This is stronger than
    # another result-selected counterexample panel, but still NOT paper evidence.
    run_parallel_methods "holdout64_v31" "$HOLD64_IDS" "${METHODS:-cowp_bihorizon_option_viability}"
    ;;

  analyze_holdout64)
    args=(--cowp "$REF_HOLD64_COWP" --rvr "$REF_HOLD64_RVR" --development-selected \
      --output "$OUT_ROOT/holdout64_v31_bihorizon_analysis.json")
    [[ -s "$OUT_ROOT/holdout64_v31_cowp_bihorizon_option_viability_merged.json" ]] && \
      args+=(--bihorizon "$OUT_ROOT/holdout64_v31_cowp_bihorizon_option_viability_merged.json")
    [[ -s "$OUT_ROOT/holdout64_v31_cowp_successor_restore_only_merged.json" ]] && \
      args+=(--restore "$OUT_ROOT/holdout64_v31_cowp_successor_restore_only_merged.json")
    python -m cowp.scripts.84_analyze_bihorizon_option_viability "${args[@]}"
    ;;

  confirm200_parallel2)
    ensure_ids
    # By default run ONLY the promoted new method.  The unchanged COWP/RVR exact200
    # outputs are bundled as immutable references, saving ~53 min of redundant COWP.
    run_parallel_methods "confirm_exact200" "$EXACT_IDS" "${PROMOTED_METHODS:-cowp_bihorizon_option_viability}"
    ;;

  analyze_confirm200)
    args=(--cowp "$REF_EXACT_COWP" --rvr "$REF_EXACT_RVR" \
      --output "$OUT_ROOT/confirm_exact200_bihorizon_analysis.json")
    [[ -s "$OUT_ROOT/confirm_exact200_cowp_bihorizon_option_viability_merged.json" ]] && \
      args+=(--bihorizon "$OUT_ROOT/confirm_exact200_cowp_bihorizon_option_viability_merged.json")
    [[ -s "$OUT_ROOT/confirm_exact200_cowp_successor_restore_only_merged.json" ]] && \
      args+=(--restore "$OUT_ROOT/confirm_exact200_cowp_successor_restore_only_merged.json")
    python -m cowp.scripts.84_analyze_bihorizon_option_viability "${args[@]}"
    ;;

  help|*)
    cat <<EOF
v16.8.31 recommended order

0) Reuse the existing v16.8.24 checkpoint. Do NOT retrain/rebuild data/cache.

1) Integrity:
   $0 sanity
   $0 make_ids

2) Cheap common-path equivalence (16 COWP scenes):
   $0 base_equivalence16_parallel2
   Must pass before interpreting new branches.

3) Mechanism-selected 48-scene counterexample panel (NOT promotion/paper evidence):
   $0 counterfactual48_parallel2
   $0 analyze_counterfactual48
   Runs BHOV + restoration-only. Existing v16.8.30 SOV is reused as immutable reference.
   Goal: determine whether SOV's low rescue recall came from requiring strict successor
   improvement, and whether only discrete conventional restoration is trustworthy.

4) ONLY if BHOV improves the rescue/harm tradeoff on step 3, run the outcome-blind
   development holdout64 selected by hash from IDs outside all v16.8.30 development panels:
   $0 holdout64_parallel2
   $0 analyze_holdout64

5) ONLY if holdout64 is non-harmful and directionally favorable:
   PROMOTED_METHODS=cowp_bihorizon_option_viability $0 confirm200_parallel2
   $0 analyze_confirm200
   COWP/RVR are immutable references; they are not rerun. exact200 remains development only.

Branch meaning:
- cowp_bihorizon_option_viability (BHOV): zero-conventional only. Compare the same COWP
  fallback and RVR alternative. Switch only when the RVR action is non-worse in BOTH
  current collision-safe prefix and successor option-set signature, with at least one
  strict improvement. This is a two-horizon product partial order, not a scalar cost.
- cowp_successor_restore_only: diagnostic only. Switch only when RVR changes successor
  conventional existence from 0 to 1. It tests whether the lower-order macro/count/prefix
  parts of the v16.8.30 successor signature are too noisy.

Still prohibited this round:
- no prefix weights or shortened 8 s conventional horizon;
- no RCOT/BCOT/frontier/outcome retuning;
- no new proposal primitives/data rebuild;
- no accepted-path kinematics change in the same experiment.
EOF
    ;;
esac
