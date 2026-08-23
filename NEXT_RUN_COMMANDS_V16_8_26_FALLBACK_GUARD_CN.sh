#!/usr/bin/env bash
set -euo pipefail

# v16.8.26: NO label / cache / dataset reconstruction.
# Purpose: (1) reject/keep CTU conclusion, (2) localize strict-Waymax physical failures,
# (3) test an opt-in fallback-only outcome risk ranker, (4) profile Waymax runtime.
MODE="${1:-help}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
BASE_RUN="${BASE_RUN:-outputs/v16_8_24_compact5k_all}"
BASE_CKPT="${BASE_CKPT:-$BASE_RUN/cowp_all_best.pt}"
OLD_PROBE_ROOT="${OLD_PROBE_ROOT:-outputs/v16_8_25_ctu_probe}"
OUT_ROOT="${OUT_ROOT:-outputs/v16_8_26_fallback_guard}"
BCOT_BUDGET="${BCOT_BUDGET:-0.50}"
PROBE_N="${PROBE_N:-200}"
PROFILE_N="${PROFILE_N:-12}"
TFEX_INDEX="${TFEX_INDEX:-$COWP_ROOT/tfexample_id_index_validation.jsonl}"
mkdir -p "$OUT_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ensure_ids() {
  local dst="$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt"
  if [[ -s "$dst" ]]; then return; fi
  if [[ -s "$OLD_PROBE_ROOT/waymax_probe_${PROBE_N}_ids.txt" ]]; then
    cp "$OLD_PROBE_ROOT/waymax_probe_${PROBE_N}_ids.txt" "$dst"
    return
  fi
  python - "$COWP_ROOT/heldout_test_scene_ids.txt" "$dst" "$PROBE_N" <<'PY'
import hashlib,sys
src,dst,n=sys.argv[1],sys.argv[2],int(sys.argv[3]); ids=[]
for line in open(src,encoding='utf-8'):
    sid=line.strip()
    if sid and sid not in ids: ids.append(sid)
sel=sorted(ids,key=lambda s:hashlib.sha256(("v16.8.25-waymax-probe|"+s).encode()).hexdigest())[:n]
if len(sel)<n: raise SystemExit(f"need {n}, got {len(sel)}")
open(dst,'w',encoding='utf-8').write('\n'.join(sel)+'\n')
print('wrote',len(sel),dst)
PY
}

index_arg=()
if [[ -s "$TFEX_INDEX" ]]; then index_arg=(--tfexample-index-jsonl "$TFEX_INDEX"); fi

common_waymax=(
  --data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml --eval-config configs/eval_cowp_v16_8.yaml
  --mode waymax --checkpoint "$BASE_CKPT" --waymax-split validation
  --rollout-horizon-steps 80 --waymax-action-mode absolute_xy_yaw
  --ncf-gate-mode priority --witness-threshold 0.70 --bcot-risk-budget "$BCOT_BUDGET"
  --outcome-risk-penalty 0 --waymax-standard-metrics
  --waymax-standard-metric-names OverlapMetric,OffroadMetric,ProgressionMetric,KinematicsInfeasibilityMetric
  --reuse-waymax-env --prefilter-waymax-shards --jit-waymax-env --jit-waymax-metrics
)

run_split_profile() {
  local method="$1" ids="$2" out="$3"
  CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m cowp.scripts.04_eval_closed_loop "${common_waymax[@]}" "${index_arg[@]}" \
    --method "$method" --scenario-ids-file "$ids" --device cuda:0 --waymax-device gpu --jax-visible-devices 1 --jax-preallocate false \
    --profile-waymax-runtime --profile-waymax-sync --no-progress --status-every 2 --output "$out"
}

run_colocated_shard() {
  local gpu="$1" shard="$2" method="$3" ids="$4" out="$5" prof="${6:-0}"
  local prof_args=()
  [[ "$prof" == "1" ]] && prof_args=(--profile-waymax-runtime --profile-waymax-sync)
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m cowp.scripts.04_eval_closed_loop "${common_waymax[@]}" "${index_arg[@]}" \
    --method "$method" --scenario-ids-file "$ids" --device cuda:0 --waymax-device gpu --jax-visible-devices 0 --jax-preallocate false \
    --num-shards 2 --shard-index "$shard" "${prof_args[@]}" --no-progress --status-every 2 --output "$out"
}

case "$MODE" in
  outcome_diag)
    # Same checkpoint/cache. The new method differs from COWP only if fallback is needed.
    for split in val heldout; do
      cache="$COWP_ROOT/tensor_cache_val_waymax"; [[ "$split" == heldout ]] && cache="$COWP_ROOT/tensor_cache_heldout_test_waymax"
      python -m cowp.scripts.04_eval_closed_loop \
        --data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml --eval-config configs/eval_cowp_v16_8.yaml \
        --mode learned_offline --cache-dir "$cache" --checkpoint "$BASE_CKPT" \
        --methods cowp,cowp_fallback_outcome --ncf-gate-mode priority --witness-threshold 0.70 --bcot-risk-budget "$BCOT_BUDGET" \
        --outcome-risk-penalty 0 --batch-size 8 --device cuda:0 --num-workers 2 --prefetch-factor 1 \
        --output "$OUT_ROOT/${split}_fallback_outcome_compare.json"
    done
    ;;

  build_tfindex)
    # One-time sparse-evaluation index only; this does NOT rebuild labels/caches.
    python -m cowp.scripts.78_build_tfexample_id_index --data-config configs/data.yaml --split validation --output "$TFEX_INDEX"
    ;;

  make_ids)
    ensure_ids
    head -n "$PROFILE_N" "$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt" > "$OUT_ROOT/waymax_profile_${PROFILE_N}_ids.txt"
    echo "IDs ready under $OUT_ROOT"
    ;;

  profile_split)
    ensure_ids; head -n "$PROFILE_N" "$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt" > "$OUT_ROOT/waymax_profile_${PROFILE_N}_ids.txt"
    run_split_profile cowp "$OUT_ROOT/waymax_profile_${PROFILE_N}_ids.txt" "$OUT_ROOT/profile_split_${PROFILE_N}_cowp.json"
    ;;

  profile_colocated)
    # One A30 carries both PyTorch and JAX. This tests whether CPU<->GPU/JAX split overhead dominates.
    ensure_ids; head -n "$PROFILE_N" "$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt" > "$OUT_ROOT/waymax_profile_${PROFILE_N}_ids.txt"
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python -m cowp.scripts.04_eval_closed_loop "${common_waymax[@]}" "${index_arg[@]}" \
      --method cowp --scenario-ids-file "$OUT_ROOT/waymax_profile_${PROFILE_N}_ids.txt" --device cuda:0 --waymax-device gpu --jax-visible-devices 0 --jax-preallocate false \
      --profile-waymax-runtime --profile-waymax-sync --no-progress --status-every 2 --output "$OUT_ROOT/profile_colocated_${PROFILE_N}_cowp.json"
    ;;

  profile_parallel2)
    # Two processes, one per A30; each co-locates Torch+JAX and receives half the exact IDs.
    ensure_ids; head -n "$PROFILE_N" "$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt" > "$OUT_ROOT/waymax_profile_${PROFILE_N}_ids.txt"
    start=$(date +%s)
    run_colocated_shard 0 0 cowp "$OUT_ROOT/waymax_profile_${PROFILE_N}_ids.txt" "$OUT_ROOT/profile_parallel2_${PROFILE_N}_cowp_s0.json" 1 & p0=$!
    run_colocated_shard 1 1 cowp "$OUT_ROOT/waymax_profile_${PROFILE_N}_ids.txt" "$OUT_ROOT/profile_parallel2_${PROFILE_N}_cowp_s1.json" 1 & p1=$!
    wait "$p0"; wait "$p1"
    end=$(date +%s); echo $((end-start)) > "$OUT_ROOT/profile_parallel2_${PROFILE_N}_wall_seconds.txt"
    python -m cowp.scripts.79_merge_waymax_exact_shards --inputs "$OUT_ROOT/profile_parallel2_${PROFILE_N}_cowp_s0.json" "$OUT_ROOT/profile_parallel2_${PROFILE_N}_cowp_s1.json" --output "$OUT_ROOT/profile_parallel2_${PROFILE_N}_cowp_merged.json"
    ;;

  waymax_diag200_parallel2)
    # Run only after profile_parallel2 fits memory and is faster. If it OOMs, use waymax_diag200_split below.
    ensure_ids; ids="$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt"
    methods="${METHODS:-cowp,cowp_fallback_outcome,conventional_safety,planner_score_only}"
    IFS=',' read -r -a arr <<< "$methods"
    for method in "${arr[@]}"; do
      run_colocated_shard 0 0 "$method" "$ids" "$OUT_ROOT/waymax_${PROBE_N}_${method}_s0.json" 0 & p0=$!
      run_colocated_shard 1 1 "$method" "$ids" "$OUT_ROOT/waymax_${PROBE_N}_${method}_s1.json" 0 & p1=$!
      wait "$p0"; wait "$p1"
      python -m cowp.scripts.79_merge_waymax_exact_shards --inputs "$OUT_ROOT/waymax_${PROBE_N}_${method}_s0.json" "$OUT_ROOT/waymax_${PROBE_N}_${method}_s1.json" --output "$OUT_ROOT/waymax_${PROBE_N}_${method}_merged.json"
    done
    ;;

  waymax_diag200_split)
    # Memory-safe fallback: original split-GPU layout, one method at a time.
    ensure_ids; ids="$OUT_ROOT/waymax_probe_${PROBE_N}_ids.txt"
    methods="${METHODS:-cowp,cowp_fallback_outcome,conventional_safety,planner_score_only}"
    IFS=',' read -r -a arr <<< "$methods"
    for method in "${arr[@]}"; do
      CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python -m cowp.scripts.04_eval_closed_loop "${common_waymax[@]}" "${index_arg[@]}" \
        --method "$method" --scenario-ids-file "$ids" --device cuda:0 --waymax-device gpu --jax-visible-devices 1 --jax-preallocate false \
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
No dataset rebuild. Recommended order:
  $0 outcome_diag
  $0 build_tfindex       # optional but recommended once; sparse WOMD index only
  $0 make_ids
  $0 profile_split
  $0 profile_colocated
  $0 profile_parallel2   # if this fits A30 memory and wins throughput, use parallel2 below

Then strict paired physical attribution:
  $0 waymax_diag200_parallel2
  $0 analyze_parallel2

If co-located two-process mode OOMs or is slower:
  $0 waymax_diag200_split
  $0 analyze_split

Do NOT run planner_repair or rebuild labels yet.
EOF
    ;;
esac
