#!/usr/bin/env bash
set -euo pipefail

# Source-faithful WOMD/COWP adaptations of:
#   GameFormer, DTPP, PLUTO, PlanT 2.0, PDM-Closed
# They share WOMD scenario IDs / history / 80-step horizon / Waymax evaluator,
# but NEVER train on COWP witness, OPR, burden, false-safe or NCF labels.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-help}"
ONE="${2:-all}"

WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
TRAIN_CACHE="${TRAIN_CACHE:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k/tensor_cache_train}"
VAL_CACHE="${VAL_CACHE:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k/tensor_cache_val}"
HELDOUT_CACHE="${HELDOUT_CACHE:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k/labels_heldout_test}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/external_sota5_v16_8_33}"
SCENARIO_IDS_FILE="${SCENARIO_IDS_FILE:-$ROOT/reference_manifests/formal_v16_8_24_compact5k_heldout1200_ids.txt}"
WOMD_VALIDATION_TFEXAMPLE_DIR="${WOMD_VALIDATION_TFEXAMPLE_DIR:-$WOMD_ROOT/uncompressed/tf_example/validation}"
TFEXAMPLE_GLOB="${TFEXAMPLE_GLOB:-$WOMD_VALIDATION_TFEXAMPLE_DIR/*.tfrecord*}"
TFEXAMPLE_INDEX_JSONL="${TFEXAMPLE_INDEX_JSONL:-$OUT_ROOT/womd_validation_tfexample_index.jsonl}"
COWP_JSON="${COWP_JSON:-}"

DEVICE="${DEVICE:-auto}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
VAL_PREFETCH_FACTOR="${VAL_PREFETCH_FACTOR:-2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
EPOCHS_OVERRIDE="${EPOCHS:-}"
LR_OVERRIDE="${LR:-}"
SEED="${SEED:-3407}"
WAYMAX_ACTION_MODE="${WAYMAX_ACTION_MODE:-absolute_xy_yaw}"
WAYMAX_METRICS="${WAYMAX_METRICS:-OverlapMetric,OffroadMetric,WrongWayMetric,ProgressionMetric,OffRouteMetric,KinematicsInfeasibilityMetric,LogDivergenceMetric}"
JAX_PREALLOCATE="${JAX_PREALLOCATE:-false}"
PARALLEL2="${PARALLEL2:-1}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
PROFILE_POLICY_TIMING="${PROFILE_POLICY_TIMING:-0}"
PROFILE_NUM_SCENARIOS="${PROFILE_NUM_SCENARIOS:-24}"
# DTPP public-source defaults favor stability/fidelity: FP32 and global cost
# weights.  These can be explicitly overridden for an ablation after a stable
# reference checkpoint has been established.
DTPP_AMP="${DTPP_AMP:-0}"
DTPP_VARIABLE_COST="${DTPP_VARIABLE_COST:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
WEIGHT_DECAY_OVERRIDE="${WEIGHT_DECAY:-}"

LEARNED=(gameformer dtpp pluto plant2)
ALL_METHODS=(gameformer dtpp pluto plant2 pdm_closed)
mkdir -p "$OUT_ROOT"

contains_method() {
  local x="$1"; shift
  for y in "$@"; do [[ "$x" == "$y" ]] && return 0; done
  return 1
}

selected() {
  local m="$1"
  [[ "$ONE" == "all" || "$ONE" == "$m" ]]
}

require_path() {
  local p="$1"; local what="$2"
  [[ -e "$p" ]] || { echo "ERROR: missing $what: $p" >&2; exit 2; }
}

ensure_index() {
  if [[ -s "$TFEXAMPLE_INDEX_JSONL" ]]; then return 0; fi
  echo "[index] building exact scenario-id -> tf.Example index: $TFEXAMPLE_INDEX_JSONL"
  python -m cowp.scripts.78_build_tfexample_id_index \
    --split validation \
    --tfexample-glob "$TFEXAMPLE_GLOB" \
    --output "$TFEXAMPLE_INDEX_JSONL" \
    --reuse-if-exists
}

train_one() {
  local m="$1"
  contains_method "$m" "${LEARNED[@]}" || { echo "$m has no learned training stage"; return 0; }
  local epochs batch lr wd
  case "$m" in
    gameformer) epochs=20; batch=24; lr=1e-4; wd=1e-4 ;;   # official open-loop defaults + existing adapter WD
    dtpp)       epochs=30; batch=16; lr=2e-4; wd=1e-2 ;;   # official DTPP AdamW default weight_decay=0.01
    pluto)      epochs=25; batch=24; lr=1e-3; wd=1e-4 ;;   # official README full-data recipe + existing adapter WD
    plant2)     epochs=30; batch=16; lr=1e-4; wd=1e-4 ;;   # WOMD adapter budget; native CARLA config is domain-specific
  esac
  [[ -n "$EPOCHS_OVERRIDE" ]] && epochs="$EPOCHS_OVERRIDE"
  [[ -n "$BATCH_SIZE_OVERRIDE" ]] && batch="$BATCH_SIZE_OVERRIDE"
  [[ -n "$LR_OVERRIDE" ]] && lr="$LR_OVERRIDE"
  [[ -n "$WEIGHT_DECAY_OVERRIDE" ]] && wd="$WEIGHT_DECAY_OVERRIDE"
  require_path "$TRAIN_CACHE" "train tensor cache"
  require_path "$VAL_CACHE" "val tensor cache"
  mkdir -p "$OUT_ROOT/$m"
  local history="$OUT_ROOT/$m/external_${m}_history.json"
  local best="$OUT_ROOT/$m/external_${m}_best.pt"
  local complete="$OUT_ROOT/$m/external_${m}_training_complete.json"
  if [[ "$SKIP_COMPLETED" == "1" ]] && python - "$m" "$history" "$best" "$complete" "$epochs" <<'PY_DONE'
import json, pathlib, sys
m, hp, bp, cp, target = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]), int(sys.argv[5])
if not hp.is_file() or not bp.is_file():
    raise SystemExit(1)
try:
    rows = json.loads(hp.read_text())
    last = int(rows[-1].get("epoch", 0)) if rows else 0
except Exception:
    raise SystemExit(1)
if last < target:
    raise SystemExit(1)
# DTPP V2 checkpoints used the now-invalid default (variable cost + BF16) and
# have no V3 completion marker.  Never silently reuse them.
if m == "dtpp":
    if not cp.is_file():
        raise SystemExit(1)
    try:
        done = json.loads(cp.read_text())
        sig = done.get("training_signature", {})
        if not done.get("completed", False) or bool(sig.get("dtpp_variable_cost", True)):
            raise SystemExit(1)
        if bool(sig.get("amp", True)):
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception:
        raise SystemExit(1)
raise SystemExit(0)
PY_DONE
  then
    echo "[train] $m already completed >=${epochs} epochs; SKIP_COMPLETED=1, reusing $best"
    return 0
  fi
  local amp_args=(--amp --amp-dtype bfloat16)
  local dtpp_cost_args=()
  if [[ "$m" == "dtpp" ]]; then
    # Official DTPP public training is FP32 and variable_weights defaults false.
    [[ "$DTPP_AMP" == "1" ]] || amp_args=()
    [[ "$DTPP_VARIABLE_COST" == "1" ]] && dtpp_cost_args=(--dtpp-variable-cost)
  fi
  echo "[train] $m epochs=$epochs batch=$batch lr=$lr wd=$wd amp=${amp_args[*]:-fp32}"
  python -m cowp.scripts.20_train_external_baseline \
    --baseline "$m" \
    --cache-dir "$TRAIN_CACHE" \
    --val-cache-dir "$VAL_CACHE" \
    --output-dir "$OUT_ROOT/$m" \
    --device "$DEVICE" \
    --epochs "$epochs" \
    --batch-size "$batch" \
    --num-workers "$NUM_WORKERS" \
    --val-num-workers "$VAL_NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --val-prefetch-factor "$VAL_PREFETCH_FACTOR" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --lr "$lr" \
    --weight-decay "$wd" \
    --seed "$SEED" \
    --max-neighbors 10 \
    --max-candidates 30 \
    --future-len 80 \
    "${amp_args[@]}" \
    "${dtpp_cost_args[@]}"
}

ckpt_for() {
  local m="$1"
  echo "$OUT_ROOT/$m/external_${m}_best.pt"
}

offline_one() {
  local m="$1"
  # Mechanism audit must use the identical 1200 held-out split as Waymax.
  # The formal build writes full COWP/WOMD NPZs under labels_heldout_test.
  require_path "$HELDOUT_CACHE" "heldout 1200 label/cache directory"
  mkdir -p "$OUT_ROOT/$m"
  echo "[offline mechanism audit] $m"
  if [[ "$m" == "pdm_closed" ]]; then
    python -m cowp.scripts.22_eval_rule_baseline \
      --baseline pdm_closed \
      --mode learned_offline \
      --cache-dir "$HELDOUT_CACHE" \
      --batch-size 64 --num-workers "$NUM_WORKERS" \
      --no-conventional-filter \
      --output "$OUT_ROOT/$m/offline.json"
  else
    local ckpt; ckpt="$(ckpt_for "$m")"
    require_path "$ckpt" "$m checkpoint"
    python -m cowp.scripts.21_eval_external_baseline \
      --mode learned_offline \
      --checkpoint "$ckpt" \
      --cache-dir "$HELDOUT_CACHE" \
      --batch-size 16 --num-workers "$NUM_WORKERS" \
      --device "$DEVICE" \
      --output "$OUT_ROOT/$m/offline.json"
  fi
}

waymax_one_single() {
  local m="$1"
  local shard_index="${2:-}"
  local num_shards="${3:-}"
  local output_path="${4:-$OUT_ROOT/$m/waymax.json}"
  local shard_args=()
  [[ -n "$shard_index" && -n "$num_shards" ]] && shard_args=(--shard-index "$shard_index" --num-shards "$num_shards")
  local profile_args=()
  [[ "$PROFILE_POLICY_TIMING" == "1" ]] && profile_args=(--profile-policy-timing)
  require_path "$SCENARIO_IDS_FILE" "exact scenario manifest"
  ensure_index
  mkdir -p "$OUT_ROOT/$m"
  echo "[Waymax closed loop] $m ids=$(wc -l < "$SCENARIO_IDS_FILE")"
  if [[ "$m" == "pdm_closed" ]]; then
    python -m cowp.scripts.22_eval_rule_baseline \
      --baseline pdm_closed \
      --mode waymax \
      --waymax-split validation \
      --tfexample-glob "$TFEXAMPLE_GLOB" \
      --tfexample-index-jsonl "$TFEXAMPLE_INDEX_JSONL" \
      --scenario-ids-file "$SCENARIO_IDS_FILE" \
      --rollout-horizon-steps 80 \
      --waymax-action-mode "$WAYMAX_ACTION_MODE" \
      --waymax-standard-metrics \
      --waymax-standard-metric-names "$WAYMAX_METRICS" \
      --no-conventional-filter \
      --jax-preallocate "$JAX_PREALLOCATE" \
      "${shard_args[@]}" \
      "${profile_args[@]}" \
      --output "$output_path"
  else
    local ckpt; ckpt="$(ckpt_for "$m")"
    require_path "$ckpt" "$m checkpoint"
    python -m cowp.scripts.21_eval_external_baseline \
      --mode waymax \
      --checkpoint "$ckpt" \
      --device "$DEVICE" \
      --execution-mode auto \
      --waymax-split validation \
      --tfexample-glob "$TFEXAMPLE_GLOB" \
      --tfexample-index-jsonl "$TFEXAMPLE_INDEX_JSONL" \
      --scenario-ids-file "$SCENARIO_IDS_FILE" \
      --rollout-horizon-steps 80 \
      --waymax-action-mode "$WAYMAX_ACTION_MODE" \
      --waymax-standard-metrics \
      --waymax-standard-metric-names "$WAYMAX_METRICS" \
      --jax-preallocate "$JAX_PREALLOCATE" \
      "${shard_args[@]}" \
      "${profile_args[@]}" \
      --output "$output_path"
  fi
}

waymax_one() {
  local m="$1"
  # Build the shared exact-ID index once in the parent process to avoid a
  # two-shard race when the index does not exist yet.
  require_path "$SCENARIO_IDS_FILE" "exact scenario manifest"
  ensure_index
  if [[ "$PARALLEL2" != "1" ]]; then
    waymax_one_single "$m"
    return
  fi
  mkdir -p "$OUT_ROOT/$m"
  local s0="$OUT_ROOT/$m/waymax_s0.json"
  local s1="$OUT_ROOT/$m/waymax_s1.json"
  echo "[Waymax parallel2] $m GPUs=$GPU0,$GPU1"
  (CUDA_VISIBLE_DEVICES="$GPU0" JAX_VISIBLE_DEVICES=0 DEVICE=cuda:0 waymax_one_single "$m" 0 2 "$s0") & p0=$!
  (CUDA_VISIBLE_DEVICES="$GPU1" JAX_VISIBLE_DEVICES=0 DEVICE=cuda:0 waymax_one_single "$m" 1 2 "$s1") & p1=$!
  set +e
  wait "$p0"; s0_status=$?
  wait "$p1"; s1_status=$?
  set -e
  if [[ $s0_status -ne 0 || $s1_status -ne 0 ]]; then
    echo "ERROR: Waymax shards failed: shard0=$s0_status shard1=$s1_status" >&2
    return 1
  fi
  python -m cowp.scripts.79_merge_waymax_exact_shards \
    --inputs "$s0" "$s1" \
    --output "$OUT_ROOT/$m/waymax.json"
}

train_parallel2() {
  # Independent runs: one baseline per GPU.  This preserves every seed, batch,
  # optimizer and scheduler of train_one while reducing all-baseline wall time.
  local methods=()
  for m in "${LEARNED[@]}"; do selected "$m" && methods+=("$m"); done
  local i=0
  while [[ $i -lt ${#methods[@]} ]]; do
    local m0="${methods[$i]}"
    local m1=""
    [[ $((i+1)) -lt ${#methods[@]} ]] && m1="${methods[$((i+1))]}"
    echo "[train parallel2] GPU$GPU0=$m0 ${m1:+GPU$GPU1=$m1}"
    (CUDA_VISIBLE_DEVICES="$GPU0" DEVICE=cuda:0 train_one "$m0") & p0=$!
    if [[ -n "$m1" ]]; then
      (CUDA_VISIBLE_DEVICES="$GPU1" DEVICE=cuda:0 train_one "$m1") & p1=$!
      # Always reap both workers.  With `set -e`, `wait p0; wait p1` could exit
      # after the first failure and leave the peer run orphaned/unreported.
      set +e
      wait "$p0"; s0=$?
      wait "$p1"; s1=$?
      set -e
      if [[ $s0 -ne 0 || $s1 -ne 0 ]]; then
        echo "ERROR: parallel training pair failed: $m0 status=$s0 $m1 status=$s1" >&2
        return 1
      fi
    else
      wait "$p0"
    fi
    i=$((i+2))
  done
}

profile_one() {
  local m="$1"
  require_path "$SCENARIO_IDS_FILE" "exact scenario manifest"
  local profile_ids="$OUT_ROOT/profile_${PROFILE_NUM_SCENARIOS}_ids.txt"
  head -n "$PROFILE_NUM_SCENARIOS" "$SCENARIO_IDS_FILE" > "$profile_ids"
  echo "[profile] $m first $(wc -l < "$profile_ids") exact held-out IDs on raw WOMD validation"
  (
    SCENARIO_IDS_FILE="$profile_ids"
    PROFILE_POLICY_TIMING=1
    PARALLEL2=0
    waymax_one_single "$m" "" "" "$OUT_ROOT/$m/profile_waymax.json"
  )
}

summarize() {
  local cmd=(python -m cowp.scripts.24_summarize_sota_closed_loop
    --results-root "$OUT_ROOT"
    --output-json "$OUT_ROOT/summary_5_baselines_plus_cowp.json"
    --output-csv "$OUT_ROOT/summary_5_baselines_plus_cowp.csv")
  if [[ -n "$COWP_JSON" ]]; then cmd+=(--cowp-json "$COWP_JSON"); fi
  "${cmd[@]}"
}

case "$MODE" in
  train)
    for m in "${LEARNED[@]}"; do selected "$m" && train_one "$m"; done
    ;;
  train_parallel2)
    train_parallel2
    ;;
  offline)
    for m in "${ALL_METHODS[@]}"; do selected "$m" && offline_one "$m"; done
    ;;
  waymax)
    for m in "${ALL_METHODS[@]}"; do selected "$m" && waymax_one "$m"; done
    ;;
  profile)
    for m in "${ALL_METHODS[@]}"; do selected "$m" && profile_one "$m"; done
    ;;
  all)
    for m in "${LEARNED[@]}"; do selected "$m" && train_one "$m"; done
    for m in "${ALL_METHODS[@]}"; do selected "$m" && offline_one "$m"; done
    for m in "${ALL_METHODS[@]}"; do selected "$m" && waymax_one "$m"; done
    summarize
    ;;
  summary)
    [[ -n "${2:-}" ]] && COWP_JSON="$2"
    summarize
    ;;
  help|-h|--help)
    cat <<EOF
Usage:
  bash RUN_5_SOTA_BASELINES_COWP.sh train [all|gameformer|dtpp|pluto|plant2]
  bash RUN_5_SOTA_BASELINES_COWP.sh train_parallel2 [all|gameformer|dtpp|pluto|plant2]
  bash RUN_5_SOTA_BASELINES_COWP.sh offline [all|gameformer|dtpp|pluto|plant2|pdm_closed]
  bash RUN_5_SOTA_BASELINES_COWP.sh profile [all|gameformer|dtpp|pluto|plant2|pdm_closed]
  bash RUN_5_SOTA_BASELINES_COWP.sh waymax [all|gameformer|dtpp|pluto|plant2|pdm_closed]
  COWP_JSON=/path/to/cowp_waymax.json bash RUN_5_SOTA_BASELINES_COWP.sh summary
  bash RUN_5_SOTA_BASELINES_COWP.sh all

Key environment overrides:
  WOMD_ROOT WOMD_VALIDATION_TFEXAMPLE_DIR TRAIN_CACHE VAL_CACHE HELDOUT_CACHE OUT_ROOT
  SCENARIO_IDS_FILE TFEXAMPLE_GLOB TFEXAMPLE_INDEX_JSONL DEVICE EPOCHS BATCH_SIZE LR
  NUM_WORKERS VAL_NUM_WORKERS PREFETCH_FACTOR CHECKPOINT_EVERY SEED COWP_JSON WEIGHT_DECAY
  DTPP_AMP=0 DTPP_VARIABLE_COST=0 SKIP_COMPLETED=1
  PARALLEL2=1 GPU0=0 GPU1=1   # default: match your two-GPU Waymax workflow
  PROFILE_NUM_SCENARIOS=24     # short exact-ID raw-WOMD timing run

Execution fidelity:
  GameFormer / PLUTO / PlanT2: source planner direct trajectory heads in Waymax.
  DTPP: ego-conditioned prediction + learned cost over a tree/proposal adapter.
  PDM-Closed: predictive rule scoring over the WOMD/COWP local proposal adapter.
EOF
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2;;
esac
