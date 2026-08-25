#!/usr/bin/env bash
set -euo pipefail

# v16.8.28 repair-only: NO dataset/label/cache rebuild and NO model retraining.
# Purpose: repair no-valid execution integrity (never execute zero-padded PAD as
# a valid Waymax action), then rerun the exact same 200-ID physical attribution.
# No planning algorithm/mechanism is changed.
MODE="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
BASE_RUN="${BASE_RUN:-outputs/v16_8_24_compact5k_all}"
BASE_CKPT="${BASE_CKPT:-$BASE_RUN/cowp_all_best.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/v16_8_28_execution_integrity}"
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
  echo "          evaluation remains correct but may scan TFRecords slowly; run '$0 build_tfindex' once." >&2
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

case "$MODE" in
  sanity)
    python -m py_compile \
      cowp/waymax_eval/policy_wrapper.py cowp/waymax_eval/rollout.py \
      cowp/waymax_eval/metrics_cowp.py cowp/scripts/04_eval_closed_loop.py \
      cowp/scripts/80_compare_waymax_physical_methods.py
    python -m pytest -q \
      tests/test_v16_8_28_execution_integrity.py \
      tests/test_v16_8_27_integrity.py \
      tests/test_v16_8_26_fallback_outcome.py \
      tests/test_v16_8_25_ctu_selector.py
    if grep -R "conventional_check=False" -n cowp/waymax_eval/policy_wrapper.py; then
      echo "ERROR: conventional-safety bypass still exists" >&2; exit 3
    fi
    if grep -n 'fallback_reason = "emergency_stop_like"' cowp/waymax_eval/policy_wrapper.py; then
      echo "ERROR: unreachable emergency_stop_like selector branch still exists" >&2; exit 4
    fi
    python - <<'PY'
import numpy as np
from cowp.waymax_eval.policy_wrapper import _resolve_execution_trajectory
pad=np.zeros((8,20,7),dtype=np.float32)
cur=np.zeros(11,dtype=np.float32); cur[:2]=[100.,50.]; cur[5]=8.; cur[6]=0.4; cur[7:9]=[4.8,1.9]
traj,emergency,src=_resolve_execution_trajectory(pad,0,False,cur,{"time":{"dt":0.1},"planning":{"fallback_decel_mps2":-2.0}})
assert emergency and src=="bounded_smooth_stop"
assert np.linalg.norm(traj[0,:2])>50.0 and np.linalg.norm(traj[0,:2]-cur[:2])<2.0
print("no-valid execution integrity OK: padding is never executed")
PY
    ensure_ids
    ;;

  offline_metadata_check)
    # Fast reporting-integrity check only. No model/cache change and no need to
    # rerun full learned-offline metrics from v16.8.26.
    python -m cowp.scripts.04_eval_closed_loop \
      --data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml --eval-config configs/eval_cowp_v16_8.yaml \
      --mode learned_offline --cache-dir "$COWP_ROOT/tensor_cache_val_waymax" --checkpoint "$BASE_CKPT" \
      --methods cowp,cowp_fallback_outcome --ncf-gate-mode priority --witness-threshold 0.70 --bcot-risk-budget "$BCOT_BUDGET" \
      --outcome-risk-penalty 0 --batch-size 8 --device cuda:0 --num-workers 2 --prefetch-factor 1 \
      --learned-max-scenes 64 --output "$OUT_ROOT/offline_metadata_check.json"
    python - "$OUT_ROOT/offline_metadata_check.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
for name, expected in [('cowp',(False,'none')),('cowp_fallback_outcome',(True,'fallback_only'))]:
    row=p[name]
    if isinstance(row,dict) and 'OutcomeHead/UsedForSelection' not in row:
        # budget-sweep-shaped result: take the only operating point
        row=next(iter(row.values()))
    got=(bool(row['OutcomeHead/UsedForSelection']),str(row['OutcomeHead/SelectionScope']))
    if got!=expected: raise SystemExit(f"{name} metadata mismatch: got={got}, expected={expected}")
print('Outcome-head metadata integrity OK')
PY
    ;;

  build_tfindex)
    python -m cowp.scripts.78_build_tfexample_id_index \
      --data-config configs/data.yaml --split validation --output "$TFEX_INDEX"
    ;;

  make_ids)
    ensure_ids
    echo "IDs ready: $OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt"
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

  waymax_diag200_parallel2)
    ensure_ids; ids="$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt"
    methods="${METHODS:-cowp,cowp_fallback_outcome,conventional_safety,planner_score_only}"
    IFS=',' read -r -a arr <<< "$methods"
    for method in "${arr[@]}"; do
      echo "=== strict exact-ID method: $method ==="
      run_colocated_shard 0 0 "$method" "$ids" "$OUT_ROOT/waymax_${PROBE_N}_${method}_s0.json" 0 & p0=$!
      run_colocated_shard 1 1 "$method" "$ids" "$OUT_ROOT/waymax_${PROBE_N}_${method}_s1.json" 0 & p1=$!
      wait "$p0"; wait "$p1"
      python -m cowp.scripts.79_merge_waymax_exact_shards \
        --inputs "$OUT_ROOT/waymax_${PROBE_N}_${method}_s0.json" "$OUT_ROOT/waymax_${PROBE_N}_${method}_s1.json" \
        --output "$OUT_ROOT/waymax_${PROBE_N}_${method}_merged.json"
    done
    ;;

  waymax_diag200_split)
    # Memory-safe fallback if two co-located processes OOM on the A30s.
    ensure_ids; ids="$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt"
    methods="${METHODS:-cowp,cowp_fallback_outcome,conventional_safety,planner_score_only}"
    IFS=',' read -r -a arr <<< "$methods"
    for method in "${arr[@]}"; do
      CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python -m cowp.scripts.04_eval_closed_loop "${common_waymax[@]}" "${index_arg[@]}" \
        --method "$method" --scenario-ids-file "$ids" --device cuda:0 \
        --waymax-device gpu --jax-visible-devices 1 --jax-preallocate false \
        --no-progress --status-every 5 --output "$OUT_ROOT/waymax_${PROBE_N}_${method}.json"
    done
    ;;

  analyze_parallel2)
    python -m cowp.scripts.80_compare_waymax_physical_methods \
      --cowp "$OUT_ROOT/waymax_${PROBE_N}_cowp_merged.json" \
      --guard "$OUT_ROOT/waymax_${PROBE_N}_cowp_fallback_outcome_merged.json" \
      --conventional "$OUT_ROOT/waymax_${PROBE_N}_conventional_safety_merged.json" \
      --planner "$OUT_ROOT/waymax_${PROBE_N}_planner_score_only_merged.json" \
      --output "$OUT_ROOT/waymax_${PROBE_N}_physical_attribution.json"
    ;;

  analyze_split)
    python -m cowp.scripts.80_compare_waymax_physical_methods \
      --cowp "$OUT_ROOT/waymax_${PROBE_N}_cowp.json" \
      --guard "$OUT_ROOT/waymax_${PROBE_N}_cowp_fallback_outcome.json" \
      --conventional "$OUT_ROOT/waymax_${PROBE_N}_conventional_safety.json" \
      --planner "$OUT_ROOT/waymax_${PROBE_N}_planner_score_only.json" \
      --output "$OUT_ROOT/waymax_${PROBE_N}_physical_attribution.json"
    ;;

  help|*)
    cat <<EOF
v16.8.28 is repair-only. No dataset rebuild, no cache rebuild, no training.
Recommended order:
  $0 sanity
  $0 make_ids
  $0 build_tfindex          # only if $TFEX_INDEX does not already exist
  $0 waymax_diag200_parallel2
  $0 analyze_parallel2

Optional (metadata is unchanged from v16.8.27):
  $0 offline_metadata_check
Optional runtime profile:
  $0 profile_parallel2

If two-process co-location OOMs:
  $0 waymax_diag200_split
  $0 analyze_split

Do NOT run planner repair / BCOT retuning / proposal redesign until this repaired
strict attribution has been inspected.  All four methods must be rerun because the
no-valid PAD-execution bug is in the common online execution path.
EOF
    ;;
esac
