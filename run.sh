#!/usr/bin/env bash
# Stable operator: core legacy baseline stages + falsification-first diagnostics.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
MODE="${1:-help}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
BASE_RUN="${BASE_RUN:-$ROOT/outputs/base_training}"
BASE_CKPT="${BASE_CKPT:-$BASE_RUN/cowp_all_best.pt}"
SIDECAR_ROOT="${SIDECAR_ROOT:-$COWP_ROOT/recourse_sidecar}"
RCRSO_RUN="${RCRSO_RUN:-$ROOT/outputs/recourse_operator}"
RCRSO_UNSELECTED="${RCRSO_UNSELECTED:-$RCRSO_RUN/rcrso_best_unselected.pt}"
RCRSO_CKPT="${RCRSO_CKPT:-$RCRSO_RUN/rcrso_stage0_selected.pt}"
STAGE0_JSON="${STAGE0_JSON:-$RCRSO_RUN/stage0_val_support_audit.json}"
DEV_IDS="$ROOT/reference_manifests/development_exact200_ids.txt"
mkdir -p "$RCRSO_RUN" "$ROOT/outputs/research_probe" "$ROOT/outputs/logged_waymax"
require_file(){ [[ -s "$1" ]] || { echo "ERROR missing $1" >&2; exit 3; }; }
sidecar_build(){
  local split="$1" cache="$2" shards="$3" examples="$4" sobol="$5"
  [[ -d "$cache" ]] || { echo "Missing cache $cache" >&2; exit 3; }
  # Explicitly overwrite only the requested sidecar split. No implicit deletion of original data.
  mkdir -p "$SIDECAR_ROOT/$split"
  local pids=()
  for ((i=0;i<shards;i++)); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m cowp.scripts.104_build_rcrso_sidecar \
      --cache-dir "$cache" --output-root "$SIDECAR_ROOT" --split "$split" \
      --label-config configs/label.yaml --data-config configs/data.yaml --eval-config configs/eval.yaml \
      --max-examples-per-scene "$examples" --max-positive-controls 32 \
      --rich-sobol-proposals "$sobol" --control-knots 8 --environment-cap 24 \
      --forbidden-id-file "$DEV_IDS" --num-shards "$shards" --shard-index "$i" \
      > "$RCRSO_RUN/sidecar_${split}_s${i}of${shards}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
  python scripts/111_merge_rcrso_sidecar_shards.py --root "$SIDECAR_ROOT" --split "$split" --num-shards "$shards"
}
run_logged(){
  local method="$1"; require_file "$BASE_CKPT"; require_file "$DEV_IDS"
  # Comparator = logged replay ONLY. Neither reactive safety nor new IRC labels can be inferred.
  local extra=()
  if [[ "$method" == "cowp_verified_root_conditioned_recourse_set_operator" ]]; then
    require_file "$RCRSO_CKPT"; require_file "$STAGE0_JSON"
    python - "$STAGE0_JSON" <<'PY_CHECK'
import json,sys
assert json.load(open(sys.argv[1]))['stage0_support_gate']['pass'] is True, 'Stage-0 not PASS'
PY_CHECK
    extra=(--rcrso-checkpoint "$RCRSO_CKPT")
  fi
  local idx="$COWP_ROOT/tfexample_id_index_validation.jsonl"
  local idx_arg=(); [[ -s "$idx" ]] && idx_arg=(--tfexample-index-jsonl "$idx")
  CUDA_VISIBLE_DEVICES="${GPU:-0}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m cowp.scripts.04_eval_closed_loop \
    --data-config configs/data.yaml --label-config configs/label.yaml --eval-config configs/eval.yaml \
    --mode waymax --checkpoint "$BASE_CKPT" --method "$method" --waymax-split validation \
    --scenario-ids-file "$DEV_IDS" --rollout-horizon-steps 80 \
    --waymax-action-mode absolute_xy_yaw --ncf-gate-mode priority \
    --witness-threshold 0.70 --bcot-risk-budget 0.50 --outcome-risk-penalty 0 \
    --waymax-standard-metrics --waymax-standard-metric-names OverlapMetric,OffroadMetric,ProgressionMetric,KinematicsInfeasibilityMetric \
    --reuse-waymax-env --prefilter-waymax-shards --jit-waymax-env --jit-waymax-metrics \
    --device "${TORCH_DEVICE:-cuda:0}" --waymax-device "${WAYMAX_DEVICE:-gpu}" \
    --jax-preallocate false "${idx_arg[@]}" "${extra[@]}" --num-shards 1 --shard-index 0 \
    --no-progress --status-every 5 --output "$ROOT/outputs/logged_waymax/${method}.json"
}
case "$MODE" in
  sanity)
    python -m compileall -q cowp research_probe
    python -m research_probe.probe --selftest
    python -m pytest -q tests/test_research_probe.py tests/test_critical_input_mapping.py tests/test_rcrso.py
    ;;
  probe_synthetic) python -m research_probe.probe --selftest ;;
  probe_cache)
    python -m research_probe.probe \
      --cache-dir "${PROBE_CACHE:-$COWP_ROOT/tensor_cache_heldout_test}" \
      --exclude-ids "$DEV_IDS" \
      --max-scenes "${PROBE_SCENES:-80}" --max-candidates "${PROBE_CANDIDATES:-10}" \
      --max-neighbors "${PROBE_NEIGHBORS:-3}" \
      --output "$ROOT/outputs/research_probe/development_diagnostics.json" ;;
  analyze_probe)
    python -m research_probe.analyze \
      --inputs "$ROOT/outputs/research_probe/development_diagnostics.json" \
        "$ROOT/outputs/research_probe/alternative_response_bank.json" \
      --output "$ROOT/outputs/research_probe/sensitivity_analysis.json" ;;
  build_tfindex)
    python -m cowp.scripts.78_build_tfexample_id_index \
      --data-config configs/data.yaml --split validation \
      --output "$COWP_ROOT/tfexample_id_index_validation.jsonl" ;;
  logged_rcrso) run_logged cowp_verified_root_conditioned_recourse_set_operator ;;
  logged_cowp) run_logged cowp ;;
  logged_conventional) run_logged conventional_safety ;;
  logged_planner_only) run_logged planner_score_only ;;
  sidecar_smoke)
    python -m cowp.scripts.104_build_rcrso_sidecar \
      --cache-dir "$COWP_ROOT/tensor_cache_val" --output-root "$RCRSO_RUN/sidecar_smoke" \
      --split val --max-scenes 2 --max-examples-per-scene 12 \
      --rich-sobol-proposals 4 --max-positive-controls 8 --forbidden-id-file "$DEV_IDS" \
      --label-config configs/label.yaml --data-config configs/data.yaml --eval-config configs/eval.yaml ;;
  build_sidecar_train_parallel4) sidecar_build train "$COWP_ROOT/tensor_cache_train" "${SIDECAR_TRAIN_SHARDS:-4}" "${SIDECAR_TRAIN_MAX_EXAMPLES_PER_SCENE:-24}" "${SIDECAR_TRAIN_SOBOL_PROPOSALS:-24}" ;;
  build_sidecar_val_parallel2) sidecar_build val "$COWP_ROOT/tensor_cache_val" "${SIDECAR_VAL_SHARDS:-2}" "${SIDECAR_VAL_MAX_EXAMPLES_PER_SCENE:-32}" "${SIDECAR_VAL_SOBOL_PROPOSALS:-48}" ;;
  train_rcrso)
    require_file "$SIDECAR_ROOT/summary_train.json"; require_file "$SIDECAR_ROOT/summary_val.json"
    CUDA_VISIBLE_DEVICES="${RCRSO_TRAIN_GPU:-0}" python -m cowp.scripts.105_train_rcrso \
      --sidecar-root "$SIDECAR_ROOT" --output-dir "$RCRSO_RUN" --epochs "${RCRSO_EPOCHS:-30}" \
      --batch-size "${RCRSO_BATCH_SIZE:-64}" --lr "${RCRSO_LR:-3e-4}" \
      --weight-decay "${RCRSO_WEIGHT_DECAY:-1e-4}" --d-model 128 --nhead 4 --layers 2 \
      --max-queries 16 --control-knots 8 --num-workers "${RCRSO_NUM_WORKERS:-4}" \
      --device cuda:0 --seed 20260845 ;;
  stage0_support)
    require_file "$RCRSO_UNSELECTED";require_file "$SIDECAR_ROOT/summary_val.json"
    CUDA_VISIBLE_DEVICES="${RCRSO_STAGE0_GPU:-0}" python -m cowp.scripts.106_eval_rcrso_support \
      --sidecar-root "$SIDECAR_ROOT" --split val --checkpoint "$RCRSO_UNSELECTED" \
      --output "$STAGE0_JSON" --selected-checkpoint "$RCRSO_CKPT" \
      --k-values 2,4,8,16 --minimum-full-hypothesis-coverage-lift-pp 3.0 \
      --device cuda:0 --label-config configs/label.yaml --data-config configs/data.yaml --eval-config configs/eval.yaml ;;
  help|*)
    cat <<'EOF'
Stable commands (scientific hierarchy):
  bash run.sh sanity
  bash run.sh probe_synthetic      # toy selftest; cannot substantiate real-world failure
  bash run.sh probe_cache          # development cache + independent dynamics recourse screening
  bash run.sh analyze_probe        # requires alternative_response_bank.json generated independently
  bash run.sh logged_cowp          # existing Waymax logged-replay comparator (no causal claims)
  bash run.sh logged_conventional  # matching standard baseline
  bash run.sh logged_planner_only
  bash run.sh logged_rcrso        # only after frozen Stage-0 PASS
  bash run.sh build_tfindex        # only if WOMD TFExample index missing

Existing RCRSO workflow (separate from new scientific thesis):
  bash run.sh sidecar_smoke
  bash run.sh build_sidecar_train_parallel4
  bash run.sh build_sidecar_val_parallel2
  bash run.sh train_rcrso
  bash run.sh stage0_support

Setting COWP_ROOT, BASE_RUN, BASE_CKPT as in the original installation is supported.
All experiment outputs live in outputs/. Historical lost7/exact200 are DEV evidence.
No no-communication reactive-agent backend or decentralized contingency MPC is delivered.
EOF
    ;;
esac
# End of stable dispatcher. Future commands should be added to the case block above.
