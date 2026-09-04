#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
BASE_RUN="${BASE_RUN:-outputs/v16_8_24_compact5k_all}"
BASE_CKPT="${BASE_CKPT:-$BASE_RUN/cowp_all_best.pt}"
OUT_ROOT="${OUT_ROOT:-outputs/v16_8_44_root_conditioned_control_reachable_responder_support}"
TFEX_INDEX="${TFEX_INDEX:-$COWP_ROOT/tfexample_id_index_validation.jsonl}"
BCOT_BUDGET="${BCOT_BUDGET:-0.50}"
METHOD="cowp_root_conditioned_control_reachable_responder_support"

LOST7="$SCRIPT_DIR/reference_manifests/v16_8_44_lost7_ids.txt"
RETAINED3="$SCRIPT_DIR/reference_manifests/v16_8_44_retained3_ids.txt"
INDUCED9="$SCRIPT_DIR/reference_manifests/v16_8_44_induced9_ids.txt"
REMAIN29="$SCRIPT_DIR/reference_manifests/v16_8_43_counterfactual_remaining29_ids.txt"
CF48="$SCRIPT_DIR/reference_manifests/waymax_v16_8_30_counterfactual48_ids.txt"
REF_COWP="$SCRIPT_DIR/reference_results/v16_8_29_counterfactual48_cowp_reference.json"
REF_RVR="$SCRIPT_DIR/reference_results/v16_8_29_counterfactual48_rvr_reference.json"
REF_V33="$SCRIPT_DIR/reference_results/v16_8_33_counterfactual48_rosh_reference.json"
REF_V35="$SCRIPT_DIR/reference_results/v16_8_35_counterfactual48_cposh_reference.json"
REF_V36="$SCRIPT_DIR/reference_results/v16_8_36_counterfactual48_recovery_frontier_reference.json"
REF_V37="$SCRIPT_DIR/reference_results/v16_8_37_counterfactual48_recourse_returnability_bridge_reference.json"
REF_V38="$SCRIPT_DIR/reference_results/v16_8_38_counterfactual48_shift_closed_control_reachable_tube_reference.json"
REF_V39="$SCRIPT_DIR/reference_results/v16_8_39_counterfactual48_conflict_window_control_reachable_tube_reference.json"
REF_V42="$SCRIPT_DIR/reference_results/v16_8_42_counterfactual48_interaction_aware_reachable_response_envelope_reference.json"
mkdir -p "$OUT_ROOT"

index_arg=()
[[ -s "$TFEX_INDEX" ]] && index_arg=(--tfexample-index-jsonl "$TFEX_INDEX")
common=(--data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml --eval-config configs/eval_cowp_v16_8.yaml
  --mode waymax --checkpoint "$BASE_CKPT" --waymax-split validation --rollout-horizon-steps 80
  --waymax-action-mode absolute_xy_yaw --ncf-gate-mode priority --witness-threshold 0.70 --bcot-risk-budget "$BCOT_BUDGET"
  --outcome-risk-penalty 0 --waymax-standard-metrics
  --waymax-standard-metric-names OverlapMetric,OffroadMetric,ProgressionMetric,KinematicsInfeasibilityMetric
  --reuse-waymax-env --prefilter-waymax-shards --jit-waymax-env --jit-waymax-metrics)

run_shard(){ local gpu="$1" shard="$2" ids="$3" out="$4";
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false python -m cowp.scripts.04_eval_closed_loop "${common[@]}" "${index_arg[@]}" \
    --method "$METHOD" --scenario-ids-file "$ids" --device cuda:0 --waymax-device gpu --jax-visible-devices 0 --jax-preallocate false \
    --num-shards 2 --shard-index "$shard" --no-progress --status-every 1 --output "$out"; }
run_subset(){ local tag="$1" ids="$2"; local start end; start=$(date +%s);
  run_shard 0 0 "$ids" "$OUT_ROOT/${tag}_${METHOD}_s0.json" & p0=$!; run_shard 1 1 "$ids" "$OUT_ROOT/${tag}_${METHOD}_s1.json" & p1=$!;
  wait "$p0"; wait "$p1"; end=$(date +%s); echo $((end-start)) > "$OUT_ROOT/${tag}_${METHOD}_wall_seconds.txt";
  python -m cowp.scripts.79_merge_waymax_exact_shards --inputs "$OUT_ROOT/${tag}_${METHOD}_s0.json" "$OUT_ROOT/${tag}_${METHOD}_s1.json" --output "$OUT_ROOT/${tag}_${METHOD}_merged.json"; }
require_json_pass(){ local file="$1" path="$2"; python - "$file" "$path" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); cur=x
for k in sys.argv[2].split('.'): cur=cur[k]
if cur is not True: raise SystemExit(4)
PY
}

case "$MODE" in
  sanity)
    python -m py_compile cowp/label/safe_responses.py cowp/waymax_eval/policy_wrapper.py cowp/waymax_eval/rollout.py cowp/waymax_eval/metrics_cowp.py cowp/scripts/101_analyze_root_conditioned_control_reachable_responder_support.py scripts/102_analyze_v44_failfast_counterexamples.py scripts/103_stitch_v44_counterfactual48.py
    python -m pytest -q tests/test_v16_8_44_root_conditioned_control_reachable_responder_support.py tests/test_v16_8_43r3_runtime_support_reuse.py tests/test_v16_8_43_blocker_conditioned_reachable_response_envelope.py tests/test_v16_8_42_interaction_aware_reachable_response_envelope.py tests/test_v16_8_41_shift_closure_semantic_fidelity_repair.py
    ! grep -R "conventional_check=False" -n cowp/waymax_eval/policy_wrapper.py
    echo "V16.8.44 semantic sanity PASS" ;;
  build_tfindex) python -m cowp.scripts.78_build_tfexample_id_index --data-config configs/data.yaml --split validation --output "$TFEX_INDEX" ;;
  lost7_parallel2) run_subset lost7 "$LOST7" ;;
  analyze_lost7)
    python scripts/102_analyze_v44_failfast_counterexamples.py --stage lost7 --lost7-result "$OUT_ROOT/lost7_${METHOD}_merged.json" --lost7-ids "$LOST7" --output "$OUT_ROOT/lost7_failfast_gate.json" ;;
  retained3_parallel2)
    require_json_pass "$OUT_ROOT/lost7_failfast_gate.json" lost7_gate.pass; run_subset retained3 "$RETAINED3" ;;
  analyze_rescue10)
    python scripts/102_analyze_v44_failfast_counterexamples.py --stage rescue10 --lost7-result "$OUT_ROOT/lost7_${METHOD}_merged.json" --lost7-ids "$LOST7" --retained3-result "$OUT_ROOT/retained3_${METHOD}_merged.json" --retained3-ids "$RETAINED3" --output "$OUT_ROOT/rescue10_gate.json" ;;
  induced9_parallel2)
    require_json_pass "$OUT_ROOT/rescue10_gate.json" rescue10_gate.pass; run_subset induced9 "$INDUCED9" ;;
  analyze_induced9)
    python scripts/102_analyze_v44_failfast_counterexamples.py --stage induced9 --induced9-result "$OUT_ROOT/induced9_${METHOD}_merged.json" --induced9-ids "$INDUCED9" --output "$OUT_ROOT/induced9_gate.json" ;;
  remaining29_parallel2)
    require_json_pass "$OUT_ROOT/induced9_gate.json" induced9_gate.pass; run_subset remaining29 "$REMAIN29" ;;
  stitch_counterfactual48)
    require_json_pass "$OUT_ROOT/rescue10_gate.json" rescue10_gate.pass; require_json_pass "$OUT_ROOT/induced9_gate.json" induced9_gate.pass
    python scripts/103_stitch_v44_counterfactual48.py --inputs "$OUT_ROOT/lost7_${METHOD}_merged.json" "$OUT_ROOT/retained3_${METHOD}_merged.json" "$OUT_ROOT/induced9_${METHOD}_merged.json" "$OUT_ROOT/remaining29_${METHOD}_merged.json" --counterfactual48-ids "$CF48" --output "$OUT_ROOT/counterfactual48_${METHOD}_merged.json" ;;
  analyze_counterfactual48)
    python -m cowp.scripts.101_analyze_root_conditioned_control_reachable_responder_support --cowp "$REF_COWP" --rvr "$REF_RVR" --v33-rosh "$REF_V33" --v35-control-projected-spectrum "$REF_V35" --v36-recovery-frontier "$REF_V36" --v37-recourse-returnability-bridge "$REF_V37" --v38-shift-closed-tube "$REF_V38" --v39-conflict-window-tube "$REF_V39" --v42-interaction-aware-reachable-response-envelope "$REF_V42" --root-conditioned-control-reachable-responder-support "$OUT_ROOT/counterfactual48_${METHOD}_merged.json" --stage counterfactual48 --development-selected --output "$OUT_ROOT/counterfactual48_${METHOD}_analysis.json" ;;
  *) echo "Usage: $0 {sanity|build_tfindex|lost7_parallel2|analyze_lost7|retained3_parallel2|analyze_rescue10|induced9_parallel2|analyze_induced9|remaining29_parallel2|stitch_counterfactual48|analyze_counterfactual48}" ;;
esac
