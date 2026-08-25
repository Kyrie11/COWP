#!/usr/bin/env bash
set -euo pipefail

# v16.8.29: Recursive Viability Recovery (RVR) + exact-equivalent collision-context cache.
# - NO dataset / label / tensor-cache rebuild
# - NO model retraining
# - NO RCOT / BCOT / protected-priority certificate / frontier retuning
# - The new planning mechanism acts ONLY when the full conventional-safe pool is empty.
# - The development 64-ID panel is outcome-enriched and MUST NOT be used as paper evidence.
MODE="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
BASE_RUN="${BASE_RUN:-outputs/v16_8_24_compact5k_all}"
BASE_CKPT="${BASE_CKPT:-$BASE_RUN/cowp_all_best.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/v16_8_29_recursive_viability}"
EXACT_IDS="${EXACT_IDS:-$SCRIPT_DIR/reference_manifests/waymax_probe_200_ids.txt}"
DEV_IDS="${DEV_IDS:-$SCRIPT_DIR/reference_manifests/waymax_viability_dev64_ids.txt}"
V28_DEV_REFERENCE="${V28_DEV_REFERENCE:-$SCRIPT_DIR/reference_results/v16_8_28_dev64_cowp_reference.json}"
EXPECTED_EXACT_SHA="3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f"
EXPECTED_DEV_SHA="0b271cb30febe3feac3a35bb08bb8b9506b048cd6559d5d49bc046bc80c91567"
PROFILE_N="${PROFILE_N:-12}"
BCOT_BUDGET="${BCOT_BUDGET:-0.50}"
TFEX_INDEX="${TFEX_INDEX:-$COWP_ROOT/tfexample_id_index_validation.jsonl}"
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
if len(set(ids))!=len(ids): raise SystemExit(f"duplicate IDs in {p}")
h=hashlib.sha256("\n".join(ids).encode()).hexdigest()
if h!=expected: raise SystemExit(f"logical ID SHA mismatch: {h} != {expected}")
print(f"manifest OK: n={n} logical_sha256={h} path={p}")
PY
}

ensure_ids() {
  validate_manifest "$EXACT_IDS" 200 "$EXPECTED_EXACT_SHA"
  validate_manifest "$DEV_IDS" 64 "$EXPECTED_DEV_SHA"
  cp "$EXACT_IDS" "$OUT_ROOT/waymax_exact_200_ids.txt"
  cp "$DEV_IDS" "$OUT_ROOT/waymax_viability_dev64_ids.txt"
  head -n "$PROFILE_N" "$EXACT_IDS" > "$OUT_ROOT/waymax_profile_${PROFILE_N}_ids.txt"
}

index_arg=()
if [[ -s "$TFEX_INDEX" ]]; then
  index_arg=(--tfexample-index-jsonl "$TFEX_INDEX")
else
  echo "[warning] TFExample ID index not found: $TFEX_INDEX" >&2
  echo "          Result correctness is unchanged, but exact-ID loading may be slow." >&2
  echo "          Run '$0 build_tfindex' once if needed." >&2
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
  local gpu="$1" shard="$2" method="$3" ids="$4" out="$5" prof="${6:-0}"
  local prof_args=()
  if [[ "$prof" == "1" ]]; then
    prof_args=(--profile-waymax-runtime --profile-waymax-sync --profile-policy-runtime --profile-policy-sync)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m cowp.scripts.04_eval_closed_loop "${common_waymax[@]}" "${index_arg[@]}" \
    --method "$method" --scenario-ids-file "$ids" --device cuda:0 \
    --waymax-device gpu --jax-visible-devices 0 --jax-preallocate false \
    --num-shards 2 --shard-index "$shard" "${prof_args[@]}" \
    --no-progress --status-every 2 --output "$out"
}

run_parallel_pair() {
  local tag="$1" ids="$2"
  local methods="${METHODS:-cowp,cowp_recursive_viability}"
  IFS=',' read -r -a arr <<< "$methods"
  for method in "${arr[@]}"; do
    echo "=== $tag exact-ID method: $method ==="
    local start end
    start=$(date +%s)
    run_colocated_shard 0 0 "$method" "$ids" "$OUT_ROOT/${tag}_${method}_s0.json" 0 & p0=$!
    run_colocated_shard 1 1 "$method" "$ids" "$OUT_ROOT/${tag}_${method}_s1.json" 0 & p1=$!
    wait "$p0"; wait "$p1"
    end=$(date +%s)
    echo $((end-start)) > "$OUT_ROOT/${tag}_${method}_wall_seconds.txt"
    python -m cowp.scripts.79_merge_waymax_exact_shards \
      --inputs "$OUT_ROOT/${tag}_${method}_s0.json" "$OUT_ROOT/${tag}_${method}_s1.json" \
      --output "$OUT_ROOT/${tag}_${method}_merged.json"
  done
}

run_split_pair() {
  local tag="$1" ids="$2"
  local methods="${METHODS:-cowp,cowp_recursive_viability}"
  IFS=',' read -r -a arr <<< "$methods"
  for method in "${arr[@]}"; do
    echo "=== $tag split-device method: $method ==="
    local start end
    start=$(date +%s)
    CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python -m cowp.scripts.04_eval_closed_loop "${common_waymax[@]}" "${index_arg[@]}" \
      --method "$method" --scenario-ids-file "$ids" --device cuda:0 \
      --waymax-device gpu --jax-visible-devices 1 --jax-preallocate false \
      --no-progress --status-every 5 --output "$OUT_ROOT/${tag}_${method}.json"
    end=$(date +%s)
    echo $((end-start)) > "$OUT_ROOT/${tag}_${method}_wall_seconds.txt"
  done
}

case "$MODE" in
  sanity)
    python -m py_compile \
      cowp/waymax_eval/policy_wrapper.py cowp/waymax_eval/rollout.py \
      cowp/waymax_eval/metrics_cowp.py cowp/scripts/04_eval_closed_loop.py \
      cowp/scripts/79_merge_waymax_exact_shards.py \
      cowp/scripts/80_compare_waymax_physical_methods.py \
      cowp/scripts/81_summarize_recursive_viability.py \
      cowp/scripts/82_verify_cowp_equivalence.py
    python -m pytest -q \
      tests/test_v16_8_29_recursive_viability.py \
      tests/test_v16_8_28_execution_integrity.py \
      tests/test_v16_8_27_integrity.py \
      tests/test_v16_8_26_fallback_outcome.py \
      tests/test_v16_8_25_ctu_selector.py
    if grep -R "conventional_check=False" -n cowp/waymax_eval/policy_wrapper.py; then
      echo "ERROR: conventional-safety bypass exists" >&2; exit 3
    fi
    ensure_ids
    echo "v16.8.29 focused semantic/integrity sanity passed"
    ;;

  build_tfindex)
    python -m cowp.scripts.78_build_tfexample_id_index \
      --data-config configs/data.yaml --split validation --output "$TFEX_INDEX"
    ;;

  make_ids)
    ensure_ids
    echo "Exact-200, dev64 and profile manifests ready under $OUT_ROOT"
    ;;

  profile_parallel2)
    ensure_ids
    ids="$OUT_ROOT/waymax_profile_${PROFILE_N}_ids.txt"
    start=$(date +%s)
    run_colocated_shard 0 0 cowp "$ids" "$OUT_ROOT/profile_parallel2_${PROFILE_N}_cowp_s0.json" 1 & p0=$!
    run_colocated_shard 1 1 cowp "$ids" "$OUT_ROOT/profile_parallel2_${PROFILE_N}_cowp_s1.json" 1 & p1=$!
    wait "$p0"; wait "$p1"
    end=$(date +%s); echo $((end-start)) > "$OUT_ROOT/profile_parallel2_${PROFILE_N}_wall_seconds.txt"
    python -m cowp.scripts.79_merge_waymax_exact_shards \
      --inputs "$OUT_ROOT/profile_parallel2_${PROFILE_N}_cowp_s0.json" "$OUT_ROOT/profile_parallel2_${PROFILE_N}_cowp_s1.json" \
      --output "$OUT_ROOT/profile_parallel2_${PROFILE_N}_cowp_merged.json"
    ;;

  viability_diag64_parallel2)
    ensure_ids
    run_parallel_pair "viability_dev64" "$OUT_ROOT/waymax_viability_dev64_ids.txt"
    ;;

  analyze_diag64)
    python -m cowp.scripts.82_verify_cowp_equivalence \
      --reference "$V28_DEV_REFERENCE" \
      --candidate "$OUT_ROOT/viability_dev64_cowp_merged.json" \
      --output "$OUT_ROOT/viability_dev64_cowp_base_equivalence.json"
    python -m cowp.scripts.80_compare_waymax_physical_methods \
      --cowp "$OUT_ROOT/viability_dev64_cowp_merged.json" \
      --recursive "$OUT_ROOT/viability_dev64_cowp_recursive_viability_merged.json" \
      --output "$OUT_ROOT/viability_dev64_physical_compare.json"
    python -m cowp.scripts.81_summarize_recursive_viability \
      --cowp "$OUT_ROOT/viability_dev64_cowp_merged.json" \
      --recursive "$OUT_ROOT/viability_dev64_cowp_recursive_viability_merged.json" \
      --development-selected \
      --output "$OUT_ROOT/viability_dev64_mechanism_summary.json"
    ;;

  confirm200_parallel2)
    ensure_ids
    run_parallel_pair "confirm_exact200" "$OUT_ROOT/waymax_exact_200_ids.txt"
    ;;

  analyze_confirm200)
    python -m cowp.scripts.80_compare_waymax_physical_methods \
      --cowp "$OUT_ROOT/confirm_exact200_cowp_merged.json" \
      --recursive "$OUT_ROOT/confirm_exact200_cowp_recursive_viability_merged.json" \
      --output "$OUT_ROOT/confirm_exact200_physical_compare.json"
    python -m cowp.scripts.81_summarize_recursive_viability \
      --cowp "$OUT_ROOT/confirm_exact200_cowp_merged.json" \
      --recursive "$OUT_ROOT/confirm_exact200_cowp_recursive_viability_merged.json" \
      --output "$OUT_ROOT/confirm_exact200_mechanism_summary.json"
    ;;

  confirm200_split)
    ensure_ids
    run_split_pair "confirm_exact200_split" "$OUT_ROOT/waymax_exact_200_ids.txt"
    ;;

  analyze_confirm200_split)
    python -m cowp.scripts.80_compare_waymax_physical_methods \
      --cowp "$OUT_ROOT/confirm_exact200_split_cowp.json" \
      --recursive "$OUT_ROOT/confirm_exact200_split_cowp_recursive_viability.json" \
      --output "$OUT_ROOT/confirm_exact200_split_physical_compare.json"
    python -m cowp.scripts.81_summarize_recursive_viability \
      --cowp "$OUT_ROOT/confirm_exact200_split_cowp.json" \
      --recursive "$OUT_ROOT/confirm_exact200_split_cowp_recursive_viability.json" \
      --output "$OUT_ROOT/confirm_exact200_split_mechanism_summary.json"
    ;;

  help|*)
    cat <<EOF
v16.8.29 recommended research order

1) Mandatory code/integrity check:
   $0 sanity
   $0 make_ids

2) Optional speed profile (12 exact IDs; useful once after changing server/runtime):
   $0 profile_parallel2

3) FAST DEVELOPMENT GATE — 64 outcome-enriched IDs, COWP vs RVR only:
   $0 viability_diag64_parallel2
   $0 analyze_diag64
   IMPORTANT: dev64 is selected using v16.8.28 outcomes and is NOT paper evidence.
   analyze_diag64 first verifies that the unchanged COWP path matches the bundled
   v16.8.28 reference before interpreting RVR.

4) Run only after the dev64 mechanism signal is favorable:
   $0 confirm200_parallel2
   $0 analyze_confirm200

5) If two co-located Torch+JAX processes OOM or benchmark slower:
   $0 confirm200_split
   $0 analyze_confirm200_split

Only build the TFExample index if the launcher warns that it is missing:
   $0 build_tfindex

Why this is faster than v16.8.28:
- dev64 runs 2 methods x 64 scenes = 128 scene-method rollouts instead of
  4 methods x 200 = 800 (6.25x fewer at the experiment level);
- confirm200 reruns only COWP + RVR = 400 scene-method rollouts (2x fewer);
- collision-agent/CV context is cached once per policy step rather than rebuilt
  for every candidate; the conventional boolean itself is unchanged.

Do NOT use dev64 for final claims, do NOT retune RCOT/BCOT/outcome weights, and do
NOT rebuild the dataset.  If RVR fails, upload the decomposition JSON before any
proposal redesign so collision-screen conservatism can be separated from genuine
proposal support failure.
EOF
    ;;
esac
