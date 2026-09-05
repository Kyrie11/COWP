#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Frozen base experiment contract.
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
BASE_RUN="${BASE_RUN:-/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all}"
BASE_CKPT="${BASE_CKPT:-$BASE_RUN/cowp_all_best.pt}"
TFEX_INDEX="${TFEX_INDEX:-$COWP_ROOT/tfexample_id_index_validation.jsonl}"
BCOT_BUDGET="${BCOT_BUDGET:-0.50}"

# V45 has a separate proposal-completeness operator checkpoint.  The base COWP
# model, NaturalDecoder, RCOT and BCOT remain frozen.
SIDECAR_ROOT="${SIDECAR_ROOT:-$COWP_ROOT/recourse_sidecar_v16_8_45}"
RCRSO_RUN="${RCRSO_RUN:-outputs/v16_8_45_rcrso_operator}"
RCRSO_UNSELECTED="${RCRSO_UNSELECTED:-$RCRSO_RUN/rcrso_best_unselected.pt}"
RCRSO_CKPT="${RCRSO_CKPT:-$RCRSO_RUN/rcrso_stage0_selected.pt}"
STAGE0_JSON="${STAGE0_JSON:-$RCRSO_RUN/stage0_val_support_audit.json}"
OUT_ROOT="${OUT_ROOT:-outputs/v16_8_45_rcrso_closed_loop}"
METHOD="cowp_verified_root_conditioned_recourse_set_operator"
mkdir -p "$RCRSO_RUN" "$OUT_ROOT"

TRAIN_CACHE="${TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train}"
VAL_CACHE="${VAL_CACHE:-$COWP_ROOT/tensor_cache_val}"
HELDOUT_CACHE="${HELDOUT_CACHE:-$COWP_ROOT/tensor_cache_heldout_test}"

# Frozen development manifests.  They are not publication holdouts.
EXACT200="$SCRIPT_DIR/reference_manifests/waymax_probe_200_ids.txt"
EQ16="$SCRIPT_DIR/reference_manifests/waymax_v16_8_30_equivalence16_ids.txt"
PROFILE4="$SCRIPT_DIR/reference_manifests/v16_8_43r3_runtime_profile4_ids.txt"
LOST7="$SCRIPT_DIR/reference_manifests/v16_8_44_lost7_ids.txt"
LOST7_B1="$SCRIPT_DIR/reference_manifests/v16_8_45_lost7_batch1_2_ids.txt"
LOST7_B2="$SCRIPT_DIR/reference_manifests/v16_8_45_lost7_batch2_2_ids.txt"
LOST7_B3="$SCRIPT_DIR/reference_manifests/v16_8_45_lost7_batch3_3_ids.txt"
RETAINED3="$SCRIPT_DIR/reference_manifests/v16_8_44_retained3_ids.txt"
INDUCED9="$SCRIPT_DIR/reference_manifests/v16_8_44_induced9_ids.txt"
REMAIN29="$SCRIPT_DIR/reference_manifests/v16_8_43_counterfactual_remaining29_ids.txt"
CF48="$SCRIPT_DIR/reference_manifests/waymax_v16_8_30_counterfactual48_ids.txt"
FRESH37="$SCRIPT_DIR/reference_manifests/waymax_v16_8_32_fresh37_ids.txt"

REF_EQ16="$SCRIPT_DIR/reference_results/v16_8_29_equivalence16_cowp_reference.json"
REF_CF48_COWP="$SCRIPT_DIR/reference_results/v16_8_29_counterfactual48_cowp_reference.json"
REF_CF48_RVR="$SCRIPT_DIR/reference_results/v16_8_29_counterfactual48_rvr_reference.json"
REF_CF48_V33="$SCRIPT_DIR/reference_results/v16_8_33_counterfactual48_rosh_reference.json"
REF_CF48_V35="$SCRIPT_DIR/reference_results/v16_8_35_counterfactual48_cposh_reference.json"
REF_CF48_V36="$SCRIPT_DIR/reference_results/v16_8_36_counterfactual48_recovery_frontier_reference.json"
REF_CF48_V37="$SCRIPT_DIR/reference_results/v16_8_37_counterfactual48_recourse_returnability_bridge_reference.json"
REF_CF48_V38="$SCRIPT_DIR/reference_results/v16_8_38_counterfactual48_shift_closed_control_reachable_tube_reference.json"
REF_CF48_V39="$SCRIPT_DIR/reference_results/v16_8_39_counterfactual48_conflict_window_control_reachable_tube_reference.json"
REF_CF48_V42="$SCRIPT_DIR/reference_results/v16_8_42_counterfactual48_interaction_aware_reachable_response_envelope_reference.json"
REF_FRESH37_COWP="$SCRIPT_DIR/reference_results/v16_8_32_fresh37_cowp_reference.json"
REF_EXACT200_COWP="$SCRIPT_DIR/reference_results/v16_8_29_exact200_cowp_reference.json"

index_arg=()
[[ -s "$TFEX_INDEX" ]] && index_arg=(--tfexample-index-jsonl "$TFEX_INDEX")

common_waymax=(
  --data-config configs/data.yaml
  --label-config configs/label_cowp_v16_8.yaml
  --eval-config configs/eval_cowp_v16_8.yaml
  --mode waymax
  --checkpoint "$BASE_CKPT"
  --waymax-split validation
  --rollout-horizon-steps 80
  --waymax-action-mode absolute_xy_yaw
  --ncf-gate-mode priority
  --witness-threshold 0.70
  --bcot-risk-budget "$BCOT_BUDGET"
  --outcome-risk-penalty 0
  --waymax-standard-metrics
  --waymax-standard-metric-names OverlapMetric,OffroadMetric,ProgressionMetric,KinematicsInfeasibilityMetric
  --reuse-waymax-env --prefilter-waymax-shards --jit-waymax-env --jit-waymax-metrics
)

require_file(){ [[ -s "$1" ]] || { echo "ERROR: required file missing: $1" >&2; exit 4; }; }
require_json_pass(){
  local file="$1" path="$2"
  require_file "$file"
  python - "$file" "$path" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8')); cur=x
for k in sys.argv[2].split('.'):
    cur=cur[k]
if cur is not True:
    raise SystemExit(4)
print(f"PASS: {sys.argv[2]}")
PY
}
require_stage0(){ require_json_pass "$STAGE0_JSON" stage0_support_gate.pass; require_file "$RCRSO_CKPT"; }

rcrso_extra=(--rcrso-checkpoint "$RCRSO_CKPT")
run_shard(){
  local gpu="$1" shard="$2" method="$3" ids="$4" out="$5" profile="${6:-0}"
  local extra=()
  if [[ "$method" == "$METHOD" ]]; then
    require_stage0
    extra=("${rcrso_extra[@]}")
  fi
  local prof=()
  [[ "$profile" == "1" ]] && prof=(--profile-policy-runtime --profile-policy-sync)
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m cowp.scripts.04_eval_closed_loop "${common_waymax[@]}" "${index_arg[@]}" \
    --method "$method" --scenario-ids-file "$ids" --device cuda:0 --waymax-device gpu \
    --jax-visible-devices 0 --jax-preallocate false --num-shards 2 --shard-index "$shard" \
    "${extra[@]}" "${prof[@]}" --no-progress --status-every 1 --output "$out"
}
run_subset(){
  local tag="$1" ids="$2" method="${3:-$METHOD}" profile="${4:-1}"
  local start end; start=$(date +%s)
  run_shard 0 0 "$method" "$ids" "$OUT_ROOT/${tag}_${method}_s0.json" "$profile" & p0=$!
  run_shard 1 1 "$method" "$ids" "$OUT_ROOT/${tag}_${method}_s1.json" "$profile" & p1=$!
  wait "$p0"; wait "$p1"
  end=$(date +%s); echo $((end-start)) > "$OUT_ROOT/${tag}_${method}_wall_seconds.txt"
  python -m cowp.scripts.79_merge_waymax_exact_shards \
    --inputs "$OUT_ROOT/${tag}_${method}_s0.json" "$OUT_ROOT/${tag}_${method}_s1.json" \
    --output "$OUT_ROOT/${tag}_${method}_merged.json"
}

build_sidecar_split(){
  local split="$1" cache="$2" shards="$3" max_examples="$4" sobol="$5"
  [[ -d "$cache" ]] || { echo "ERROR: cache dir missing: $cache" >&2; exit 4; }
  rm -rf "$SIDECAR_ROOT/$split"
  mkdir -p "$SIDECAR_ROOT/$split"
  local pids=()
  for ((s=0;s<shards;s++)); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m cowp.scripts.104_build_rcrso_sidecar \
      --cache-dir "$cache" --output-root "$SIDECAR_ROOT" --split "$split" \
      --label-config configs/label_cowp_v16_8.yaml --data-config configs/data.yaml --eval-config configs/eval_cowp_v16_8.yaml \
      --max-examples-per-scene "$max_examples" --max-positive-controls 32 --rich-sobol-proposals "$sobol" --control-knots 8 --environment-cap 24 \
      --forbidden-id-file "$EXACT200" --num-shards "$shards" --shard-index "$s" \
      > "$RCRSO_RUN/sidecar_${split}_s${s}of${shards}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
  python scripts/111_merge_rcrso_sidecar_shards.py --root "$SIDECAR_ROOT" --split "$split" --num-shards "$shards"
}

case "$MODE" in
  sanity)
    python -m py_compile \
      cowp/label/safe_responses.py cowp/models/recourse_set_operator.py cowp/data/recourse_sidecar.py \
      cowp/waymax_eval/policy_wrapper.py cowp/waymax_eval/metrics_cowp.py cowp/scripts/04_eval_closed_loop.py \
      cowp/scripts/104_build_rcrso_sidecar.py cowp/scripts/105_train_rcrso.py cowp/scripts/106_eval_rcrso_support.py \
      cowp/scripts/109_analyze_verified_rcrso.py scripts/107_analyze_v45_rcrso_failfast.py scripts/108_stitch_exact_subsets.py \
      scripts/111_merge_rcrso_sidecar_shards.py
    python -m pytest -q \
      tests/test_v16_8_45_rcrso.py \
      tests/test_v16_8_44_root_conditioned_control_reachable_responder_support.py \
      tests/test_v16_8_43r3_runtime_support_reuse.py \
      tests/test_v16_8_43_blocker_conditioned_reachable_response_envelope.py \
      tests/test_v16_8_42_interaction_aware_reachable_response_envelope.py \
      tests/test_v16_8_41_shift_closure_semantic_fidelity_repair.py \
      tests/test_v16_8_40_shift_closed_first_action_viability_interval.py \
      tests/test_v16_8_39_conflict_window_control_reachable_tube.py \
      tests/test_v16_8_38_shift_closed_control_reachable_tube.py \
      tests/test_v16_8_37_recourse_returnability_bridge.py \
      tests/test_v16_8_36_control_projected_recovery_frontier.py \
      tests/test_v16_8_35_control_projected_option_spectrum.py \
      tests/test_v16_8_34_executable_option_spectrum.py \
      tests/test_v16_8_33_recovery_option_spectrum.py \
      tests/test_v16_8_32_temporal_option_persistence.py \
      tests/test_v16_8_31_bihorizon_option_viability.py \
      tests/test_v16_8_30_successor_option_viability.py \
      tests/test_v16_8_29_recursive_viability.py \
      tests/test_v16_8_28_execution_integrity.py tests/test_v16_8_27_integrity.py tests/test_v16_8_26_fallback_outcome.py tests/test_v16_8_25_ctu_selector.py
    ! grep -R "conventional_check=False" -n cowp/waymax_eval/policy_wrapper.py
    ! grep -R "log_trajectory\|mechanism_ground_truth" -n cowp/models/recourse_set_operator.py cowp/scripts/104_build_rcrso_sidecar.py cowp/scripts/105_train_rcrso.py cowp/scripts/106_eval_rcrso_support.py
    echo "V16.8.45 RCRSO semantic/integrity sanity PASS" ;;

  make_ids)
    python - <<'PY'
from pathlib import Path
import hashlib
expected={
'reference_manifests/waymax_probe_200_ids.txt':'3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f',
'reference_manifests/waymax_v16_8_30_equivalence16_ids.txt':'81d0319da0446d1452b4c3a0361ffa6941dfa226b2f14027cac5576f9571c760',
'reference_manifests/v16_8_44_lost7_ids.txt':'d178299e83671bd6b27651c9592f3ccb2d521fe9f42091e505f59abf0250e52f',
'reference_manifests/v16_8_44_retained3_ids.txt':'8ea17b5f12cbc453940d6f2ab7fe44af295838a6daaab0664c83889d27e95afa',
'reference_manifests/v16_8_44_induced9_ids.txt':'8cef4fcb6fe129e4a6b2ce75fdd87c85be05eabac0c57d9db1f13bed389ee250',
'reference_manifests/v16_8_43_counterfactual_remaining29_ids.txt':'8e225f7bd125f9448ac17252fe881ff5c49abf733c6e70b3dd061eb9069aa1b0',
'reference_manifests/waymax_v16_8_30_counterfactual48_ids.txt':'ee3c231c240878d5d20020aec3c98efbb4932cdbf1f1e309b9b7b26bddc40ab0',
'reference_manifests/waymax_v16_8_32_fresh37_ids.txt':'ecce3321d8f4cd57bbd3189b3673784bec8fde185b882e9c11c38430265a1481',
'reference_manifests/v16_8_43r3_runtime_profile4_ids.txt':'69f73aedb8d1cfd0e065218d2c579659ddfd07845463cb4cc767987323e9af01',
}
for raw,want in expected.items():
    ids=[x.strip() for x in Path(raw).read_text().splitlines() if x.strip()]
    got=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    if got!=want: raise SystemExit(f'hash mismatch {raw}: {got} != {want}')
    print(raw,len(ids),got)
PY
    cp "$EQ16" "$OUT_ROOT/waymax_equivalence16_ids.txt"
    cp "$CF48" "$OUT_ROOT/counterfactual48_ids.txt"
    cp "$FRESH37" "$OUT_ROOT/fresh37_ids.txt"
    cp "$EXACT200" "$OUT_ROOT/exact200_ids.txt"
    ;;

  build_tfindex)
    python -m cowp.scripts.78_build_tfexample_id_index --data-config configs/data.yaml --split validation --output "$TFEX_INDEX" ;;

  sidecar_smoke)
    rm -rf "$RCRSO_RUN/sidecar_smoke"
    python -m cowp.scripts.104_build_rcrso_sidecar \
      --cache-dir "$VAL_CACHE" --output-root "$RCRSO_RUN/sidecar_smoke" --split val \
      --max-scenes 2 --max-examples-per-scene 12 --rich-sobol-proposals 4 --max-positive-controls 8 \
      --forbidden-id-file "$EXACT200" ;;

  build_sidecar_train_parallel4)
    build_sidecar_split train "$TRAIN_CACHE" "${SIDECAR_TRAIN_SHARDS:-4}" "${SIDECAR_TRAIN_MAX_EXAMPLES_PER_SCENE:-24}" "${SIDECAR_TRAIN_SOBOL_PROPOSALS:-24}" ;;

  build_sidecar_val_parallel2)
    build_sidecar_split val "$VAL_CACHE" "${SIDECAR_VAL_SHARDS:-2}" "${SIDECAR_VAL_MAX_EXAMPLES_PER_SCENE:-32}" "${SIDECAR_VAL_SOBOL_PROPOSALS:-48}" ;;

  train_rcrso)
    require_file "$SIDECAR_ROOT/summary_train.json"; require_file "$SIDECAR_ROOT/summary_val.json"
    CUDA_VISIBLE_DEVICES="${RCRSO_TRAIN_GPU:-0}" python -m cowp.scripts.105_train_rcrso \
      --sidecar-root "$SIDECAR_ROOT" --output-dir "$RCRSO_RUN" \
      --epochs "${RCRSO_EPOCHS:-30}" --batch-size "${RCRSO_BATCH_SIZE:-64}" \
      --lr "${RCRSO_LR:-3e-4}" --weight-decay "${RCRSO_WEIGHT_DECAY:-1e-4}" \
      --d-model 128 --nhead 4 --layers 2 --max-queries 16 --control-knots 8 \
      --num-workers "${RCRSO_NUM_WORKERS:-4}" --device cuda:0 --seed 20260845 ;;

  stage0_support)
    require_file "$RCRSO_UNSELECTED"; require_file "$SIDECAR_ROOT/summary_val.json"
    CUDA_VISIBLE_DEVICES="${RCRSO_STAGE0_GPU:-0}" python -m cowp.scripts.106_eval_rcrso_support \
      --sidecar-root "$SIDECAR_ROOT" --split val --checkpoint "$RCRSO_UNSELECTED" \
      --output "$STAGE0_JSON" --selected-checkpoint "$RCRSO_CKPT" \
      --k-values 2,4,8,16 --minimum-full-hypothesis-coverage-lift-pp 3.0 --device cuda:0 \
      --label-config configs/label_cowp_v16_8.yaml --data-config configs/data.yaml --eval-config configs/eval_cowp_v16_8.yaml ;;

  base_equivalence16_parallel2)
    run_subset equivalence16 "$EQ16" cowp 0
    python -m cowp.scripts.82_verify_cowp_equivalence --reference "$REF_EQ16" \
      --candidate "$OUT_ROOT/equivalence16_cowp_merged.json" --output "$OUT_ROOT/equivalence16_cowp_vs_v16_8_29.json" ;;

  profile4_parallel2)
    require_stage0; run_subset runtime_profile4_v45 "$PROFILE4" "$METHOD" 1 ;;

  lost7_parallel2)
    require_stage0; run_subset lost7 "$LOST7" "$METHOD" 1 ;;
  analyze_lost7)
    python scripts/102_analyze_v44_failfast_counterexamples.py --stage lost7 \
      --lost7-result "$OUT_ROOT/lost7_${METHOD}_merged.json" --lost7-ids "$LOST7" \
      --output "$OUT_ROOT/lost7_v45_failfast_gate.json" ;;

  lost7_batch1_parallel2) require_stage0; run_subset lost7_b1 "$LOST7_B1" "$METHOD" 1 ;;
  lost7_batch2_parallel2) require_stage0; run_subset lost7_b2 "$LOST7_B2" "$METHOD" 1 ;;
  lost7_batch3_parallel2) require_stage0; run_subset lost7_b3 "$LOST7_B3" "$METHOD" 1 ;;
  analyze_lost7_progressive)
    results=(); manifests=()
    for x in 1 2 3; do
      case "$x" in 1) tag=lost7_b1; man="$LOST7_B1";; 2) tag=lost7_b2; man="$LOST7_B2";; 3) tag=lost7_b3; man="$LOST7_B3";; esac
      f="$OUT_ROOT/${tag}_${METHOD}_merged.json"
      [[ -s "$f" ]] || break
      results+=("$f"); manifests+=("$man")
    done
    [[ ${#results[@]} -gt 0 ]] || { echo "ERROR: no progressive lost7 result found" >&2; exit 4; }
    python scripts/107_analyze_v45_rcrso_failfast.py --results "${results[@]}" --manifests "${manifests[@]}" \
      --total-lost7-manifest "$LOST7" --output "$OUT_ROOT/lost7_v45_progressive_gate.json" ;;

  retained3_parallel2)
    require_json_pass "$OUT_ROOT/lost7_v45_failfast_gate.json" lost7_gate.pass
    run_subset retained3 "$RETAINED3" "$METHOD" 1 ;;
  analyze_rescue10)
    python scripts/102_analyze_v44_failfast_counterexamples.py --stage rescue10 \
      --lost7-result "$OUT_ROOT/lost7_${METHOD}_merged.json" --lost7-ids "$LOST7" \
      --retained3-result "$OUT_ROOT/retained3_${METHOD}_merged.json" --retained3-ids "$RETAINED3" \
      --output "$OUT_ROOT/rescue10_v45_gate.json" ;;

  induced9_parallel2)
    require_json_pass "$OUT_ROOT/rescue10_v45_gate.json" rescue10_gate.pass
    run_subset induced9 "$INDUCED9" "$METHOD" 1 ;;
  analyze_induced9)
    python scripts/102_analyze_v44_failfast_counterexamples.py --stage induced9 \
      --induced9-result "$OUT_ROOT/induced9_${METHOD}_merged.json" --induced9-ids "$INDUCED9" \
      --output "$OUT_ROOT/induced9_v45_gate.json" ;;

  remaining29_parallel2)
    require_json_pass "$OUT_ROOT/induced9_v45_gate.json" induced9_gate.pass
    run_subset remaining29 "$REMAIN29" "$METHOD" 1 ;;
  stitch_counterfactual48)
    require_json_pass "$OUT_ROOT/rescue10_v45_gate.json" rescue10_gate.pass
    require_json_pass "$OUT_ROOT/induced9_v45_gate.json" induced9_gate.pass
    python scripts/108_stitch_exact_subsets.py \
      --inputs "$OUT_ROOT/lost7_${METHOD}_merged.json" "$OUT_ROOT/retained3_${METHOD}_merged.json" "$OUT_ROOT/induced9_${METHOD}_merged.json" "$OUT_ROOT/remaining29_${METHOD}_merged.json" \
      --target-ids "$CF48" --output "$OUT_ROOT/counterfactual48_${METHOD}_merged.json" ;;
  analyze_counterfactual48)
    python -m cowp.scripts.109_analyze_verified_rcrso \
      --cowp "$REF_CF48_COWP" --rvr "$REF_CF48_RVR" --v33-rosh "$REF_CF48_V33" \
      --v35-control-projected-spectrum "$REF_CF48_V35" --v36-recovery-frontier "$REF_CF48_V36" \
      --v37-recourse-returnability-bridge "$REF_CF48_V37" --v38-shift-closed-tube "$REF_CF48_V38" \
      --v39-conflict-window-tube "$REF_CF48_V39" --v42-interaction-aware-reachable-response-envelope "$REF_CF48_V42" \
      --verified-root-conditioned-recourse-set-operator "$OUT_ROOT/counterfactual48_${METHOD}_merged.json" \
      --stage counterfactual48 --development-selected --output "$OUT_ROOT/counterfactual48_v45_rcrso_analysis.json" ;;

  fresh37_parallel2)
    require_json_pass "$OUT_ROOT/counterfactual48_v45_rcrso_analysis.json" preregistered_gate.verified_root_conditioned_recourse_set_operator.pass
    run_subset fresh37 "$FRESH37" "$METHOD" 1 ;;
  analyze_fresh37)
    python -m cowp.scripts.109_analyze_verified_rcrso --cowp "$REF_FRESH37_COWP" \
      --verified-root-conditioned-recourse-set-operator "$OUT_ROOT/fresh37_${METHOD}_merged.json" \
      --stage fresh37 --development-selected --output "$OUT_ROOT/fresh37_v45_rcrso_analysis.json" ;;

  confirm200_parallel2)
    require_json_pass "$OUT_ROOT/fresh37_v45_rcrso_analysis.json" preregistered_gate.verified_root_conditioned_recourse_set_operator.pass
    run_subset exact200 "$EXACT200" "$METHOD" 1 ;;
  analyze_confirm200)
    python -m cowp.scripts.109_analyze_verified_rcrso --cowp "$REF_EXACT200_COWP" \
      --verified-root-conditioned-recourse-set-operator "$OUT_ROOT/exact200_${METHOD}_merged.json" \
      --stage exact200 --development-selected --output "$OUT_ROOT/exact200_v45_rcrso_analysis.json" ;;

  help|*)
    cat <<EOF
V16.8.45 RCRSO — recommended order

Offline operator / support gate (no Waymax closed loop):
  $0 sanity
  $0 make_ids
  $0 sidecar_smoke
  $0 build_sidecar_train_parallel4
  $0 build_sidecar_val_parallel2
  $0 train_rcrso
  $0 stage0_support

Only if Stage-0 PASS:
  $0 base_equivalence16_parallel2
  $0 profile4_parallel2          # engineering runtime only
  $0 lost7_parallel2
  $0 analyze_lost7

Only if lost7 >= 2/7:
  $0 retained3_parallel2
  $0 analyze_rescue10

Only if historical rescue10 >= 5/10:
  $0 induced9_parallel2
  $0 analyze_induced9

Only if induced9 avoided >= 7/9:
  $0 remaining29_parallel2
  $0 stitch_counterfactual48
  $0 analyze_counterfactual48

Only if original six-item counterfactual48 gate PASS:
  $0 fresh37_parallel2
  $0 analyze_fresh37

Only if fresh37 no-harm gate PASS:
  $0 confirm200_parallel2
  $0 analyze_confirm200

Optional progressive lost7 diagnostics (fixed order 2+2+3; not needed if you run full lost7 directly):
  $0 lost7_batch1_parallel2 ; $0 analyze_lost7_progressive
  $0 lost7_batch2_parallel2 ; $0 analyze_lost7_progressive
  $0 lost7_batch3_parallel2 ; $0 analyze_lost7_progressive

Important:
- Do NOT tune K on lost7/counterfactual48. K is frozen by Stage-0 validation support coverage and stored in RCRSO_CKPT.
- Do NOT shorten 80-step Waymax rollout.
- Do NOT edit p_min/floor/root mass/beta/RCOT/BCOT/hard verifier to rescue Stage-0 or lost7.
- The exact200 universe is development-selected. A publication claim still needs a new untouched final holdout + >=3 seeds + paired CI.
EOF
    ;;
esac
