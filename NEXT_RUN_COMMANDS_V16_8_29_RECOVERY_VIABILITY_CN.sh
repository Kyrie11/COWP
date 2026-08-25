#!/usr/bin/env bash
set -euo pipefail

# v16.8.29: one preregistered algorithm probe + exact-equivalent runtime cache.
# NO dataset/label/cache rebuild, NO retraining, NO RCOT/BCOT/frontier retuning.
# Main COWP behavior is unchanged; cowp_recovery_bridge differs only when the
# full-horizon conventional set is empty but dynamically valid candidates remain.
MODE="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
BASE_RUN="${BASE_RUN:-outputs/v16_8_24_compact5k_all}"
BASE_CKPT="${BASE_CKPT:-$BASE_RUN/cowp_all_best.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/v16_8_29_recovery_viability}"
REFERENCE_IDS="${REFERENCE_IDS:-$SCRIPT_DIR/reference_manifests/waymax_probe_200_ids.txt}"
EXPECTED_IDS_SHA="3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f"
PROBE_N="${PROBE_N:-200}"
PROFILE_N="${PROFILE_N:-12}"
BCOT_BUDGET="${BCOT_BUDGET:-0.50}"
TFEX_INDEX="${TFEX_INDEX:-$COWP_ROOT/tfexample_id_index_validation.jsonl}"
mkdir -p "$OUT_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

validate_reference_ids() {
  [[ -s "$REFERENCE_IDS" ]] || { echo "Missing reference manifest: $REFERENCE_IDS" >&2; exit 2; }
  python - "$REFERENCE_IDS" "$PROBE_N" "$EXPECTED_IDS_SHA" <<'PY'
import hashlib,sys
p,n,expected=sys.argv[1],int(sys.argv[2]),sys.argv[3]
ids=[x.strip() for x in open(p,encoding='utf-8') if x.strip()]
if len(ids)!=n: raise SystemExit(f"expected {n} exact IDs, got {len(ids)} in {p}")
if len(set(ids))!=len(ids): raise SystemExit("duplicate scenario IDs in reference manifest")
h=hashlib.sha256("\n".join(ids).encode()).hexdigest()
if h!=expected: raise SystemExit(f"logical ID SHA mismatch: {h} != {expected}")
print(f"exact-ID manifest OK: n={len(ids)} logical_sha256={h}")
PY
}

ensure_ids() {
  validate_reference_ids
  cp "$REFERENCE_IDS" "$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt"
  head -n "$PROFILE_N" "$REFERENCE_IDS" > "$OUT_ROOT/waymax_profile_${PROFILE_N}_ids.txt"
}

index_arg=()
if [[ -s "$TFEX_INDEX" ]]; then
  index_arg=(--tfexample-index-jsonl "$TFEX_INDEX")
else
  echo "[warning] TFExample ID index not found: $TFEX_INDEX" >&2
  echo "          correctness is unchanged, but TFRecord lookup can be much slower." >&2
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
  # One independent scenario shard per A30.  Torch and JAX are co-located on the
  # same physical GPU inside each process; scenarios are independent and results
  # are merged by exact scenario ID afterwards.
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m cowp.scripts.04_eval_closed_loop "${common_waymax[@]}" "${index_arg[@]}" \
    --method "$method" --scenario-ids-file "$ids" --device cuda:0 \
    --waymax-device gpu --jax-visible-devices 0 --jax-preallocate false \
    --num-shards 2 --shard-index "$shard" "${prof_args[@]}" \
    --no-progress --status-every 2 --output "$out"
}

run_parallel2_method() {
  local method="$1" ids="$2" prof="${3:-0}"
  local stem="$OUT_ROOT/waymax_${PROBE_N}_${method}"
  [[ "$prof" == "1" ]] && stem="$OUT_ROOT/profile_parallel2_${PROFILE_N}_${method}"
  start=$(date +%s)
  run_colocated_shard 0 0 "$method" "$ids" "${stem}_s0.json" "$prof" & p0=$!
  run_colocated_shard 1 1 "$method" "$ids" "${stem}_s1.json" "$prof" & p1=$!
  wait "$p0"; wait "$p1"
  end=$(date +%s)
  echo $((end-start)) > "${stem}_wall_seconds.txt"
  python -m cowp.scripts.79_merge_waymax_exact_shards \
    --inputs "${stem}_s0.json" "${stem}_s1.json" \
    --output "${stem}_merged.json"
}

case "$MODE" in
  sanity)
    python -m py_compile \
      cowp/waymax_eval/policy_wrapper.py cowp/waymax_eval/rollout.py \
      cowp/waymax_eval/metrics_cowp.py cowp/scripts/04_eval_closed_loop.py \
      cowp/scripts/81_compare_recovery_bridge.py
    python -m pytest -q \
      tests/test_v16_8_29_recovery_viability.py \
      tests/test_v16_8_28_execution_integrity.py \
      tests/test_v16_8_27_integrity.py \
      tests/test_v16_8_26_fallback_outcome.py \
      tests/test_v16_8_25_ctu_selector.py
    if grep -R "conventional_check=False" -n cowp/waymax_eval/policy_wrapper.py; then
      echo "ERROR: conventional-safety bypass exists" >&2; exit 3
    fi
    if grep -n 'fallback_reason = "emergency_stop_like"' cowp/waymax_eval/policy_wrapper.py; then
      echo "ERROR: dead emergency_stop_like branch exists" >&2; exit 4
    fi
    ensure_ids
    ;;

  make_ids)
    ensure_ids
    ;;

  build_tfindex)
    # Only needed if the existing v16.8.26+ index was deleted.  This does not
    # rebuild labels/caches or change experimental semantics.
    python -m cowp.scripts.78_build_tfexample_id_index \
      --data-config configs/data.yaml --split validation --output "$TFEX_INDEX"
    ;;

  profile_parallel2)
    ensure_ids
    ids="$OUT_ROOT/waymax_profile_${PROFILE_N}_ids.txt"
    run_parallel2_method cowp "$ids" 1
    ;;

  waymax_recovery200_parallel2)
    # Recommended path.  Only two methods are necessary now: v16.8.28 COWP is
    # rerun to verify bit-exact semantic equivalence after the runtime cache, and
    # the sole new algorithm probe is paired against it on the same 200 IDs.
    ensure_ids
    ids="$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt"
    for method in cowp cowp_recovery_bridge; do
      echo "=== strict exact-ID method: $method ==="
      run_parallel2_method "$method" "$ids" 0
    done
    ;;

  waymax_recovery200_split)
    # MEMORY-SAFE ALTERNATIVE ONLY.  Do not run this after parallel2 succeeds;
    # doing both merely doubles compute and produces redundant evidence.
    ensure_ids
    ids="$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt"
    for method in cowp cowp_recovery_bridge; do
      CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python -m cowp.scripts.04_eval_closed_loop "${common_waymax[@]}" "${index_arg[@]}" \
        --method "$method" --scenario-ids-file "$ids" --device cuda:0 \
        --waymax-device gpu --jax-visible-devices 1 --jax-preallocate false \
        --no-progress --status-every 5 --output "$OUT_ROOT/waymax_${PROBE_N}_${method}.json"
    done
    ;;

  analyze_parallel2)
    python -m cowp.scripts.81_compare_recovery_bridge \
      --cowp "$OUT_ROOT/waymax_${PROBE_N}_cowp_merged.json" \
      --recovery "$OUT_ROOT/waymax_${PROBE_N}_cowp_recovery_bridge_merged.json" \
      --expected-ids-sha256 "$EXPECTED_IDS_SHA" --expected-count "$PROBE_N" \
      --output "$OUT_ROOT/waymax_${PROBE_N}_recovery_viability_attribution.json"
    ;;

  analyze_split)
    python -m cowp.scripts.81_compare_recovery_bridge \
      --cowp "$OUT_ROOT/waymax_${PROBE_N}_cowp.json" \
      --recovery "$OUT_ROOT/waymax_${PROBE_N}_cowp_recovery_bridge.json" \
      --expected-ids-sha256 "$EXPECTED_IDS_SHA" --expected-count "$PROBE_N" \
      --output "$OUT_ROOT/waymax_${PROBE_N}_recovery_viability_attribution.json"
    ;;

  help|*)
    cat <<EOF
v16.8.29 Recovery-Viability Bridge. No training, no dataset/cache/label rebuild.
Recommended order:
  $0 sanity
  $0 make_ids
  $0 build_tfindex                 # ONLY if $TFEX_INDEX is absent
  $0 profile_parallel2             # optional 12-scene runtime check
  $0 waymax_recovery200_parallel2  # two A30s concurrently, only 2 methods
  $0 analyze_parallel2

ONLY if two-process co-location OOMs:
  $0 waymax_recovery200_split
  $0 analyze_split

IMPORTANT: parallel2 and split are alternatives, not two required experiments.
Do not rerun fallback-outcome/conventional/planner-only: v16.8.28 already resolved
those preregistered branches.  Do not retrain or retune RCOT/BCOT/frontier.
EOF
    ;;
esac
