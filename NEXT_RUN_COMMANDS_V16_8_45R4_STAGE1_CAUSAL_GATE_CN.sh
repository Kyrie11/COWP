#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# R4 is not a scientific-method revision. It binds the Stage-0-GO V45 RCRSO
# checkpoint to the already preregistered closed-loop causal gates.
RCRSO_STAGE0_RUN="${RCRSO_STAGE0_RUN:-outputs/v16_8_45r3_stage0_runtime_observability}"
RCRSO_CKPT="${RCRSO_CKPT:-$RCRSO_STAGE0_RUN/rcrso_stage0_selected.pt}"
STAGE0_JSON="${STAGE0_JSON:-$RCRSO_STAGE0_RUN/stage0_val_support_audit.json}"
STAGE0_PARTIAL_DIR="${STAGE0_PARTIAL_DIR:-$RCRSO_STAGE0_RUN/stage0_partials}"
OUT_ROOT="${OUT_ROOT:-outputs/v16_8_45r4_stage1_causal_gate}"
R3="$SCRIPT_DIR/NEXT_RUN_COMMANDS_V16_8_45R3_STAGE0_RUNTIME_OBSERVABILITY_CN.sh"
mkdir -p "$OUT_ROOT"

pass_stage0(){
  python - "$STAGE0_JSON" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x.get('stage0_support_gate',{}).get('pass') is not True: raise SystemExit(4)
if int(x.get('selected_k',-1)) != 16: print('NOTE: selected K is',x.get('selected_k'))
print('PASS: frozen Stage-0 support gate; selected_k=',x.get('selected_k'))
PY
  [[ -s "$RCRSO_CKPT" ]] || { echo "ERROR: selected checkpoint missing: $RCRSO_CKPT" >&2; exit 4; }
}

run_r3(){
  RCRSO_RUN="$RCRSO_STAGE0_RUN" RCRSO_CKPT="$RCRSO_CKPT" STAGE0_JSON="$STAGE0_JSON" STAGE0_PARTIAL_DIR="$STAGE0_PARTIAL_DIR" OUT_ROOT="$OUT_ROOT" bash "$R3" "$1"
}

case "$MODE" in
  audit_stage0_result)
    args=(--result-dir "$RCRSO_STAGE0_RUN" --output "$OUT_ROOT/V16_8_45R3_STAGE0_RESULT_RELIABILITY_AND_ATTRIBUTION_AUDIT.json")
    [[ -s "$RCRSO_CKPT" ]] && args+=(--selected-checkpoint "$RCRSO_CKPT")
    python scripts/113_audit_v45r3_stage0_result.py "${args[@]}" ;;
  sanity)
    run_r3 sanity
    python -m py_compile scripts/113_audit_v45r3_stage0_result.py
    bash -n "$0"
    echo "V16.8.45R4 Stage-1 causal-gate harness sanity PASS" ;;
  base_equivalence16_parallel2)
    pass_stage0; run_r3 base_equivalence16_parallel2 ;;
  profile4_parallel2)
    pass_stage0; run_r3 profile4_parallel2 ;;
  lost7_batch1_parallel2|lost7_batch2_parallel2|lost7_batch3_parallel2|analyze_lost7_progressive)
    pass_stage0; run_r3 "$MODE" ;;
  lost7_parallel2|analyze_lost7|retained3_parallel2|analyze_rescue10|induced9_parallel2|analyze_induced9|remaining29_parallel2|stitch_counterfactual48|analyze_counterfactual48|fresh37_parallel2|analyze_fresh37|confirm200_parallel2|analyze_confirm200)
    pass_stage0; run_r3 "$MODE" ;;
  help|*)
    cat <<EOF
V16.8.45R4: Stage-1 causal gate only; scientific RCRSO is unchanged.

Required server provenance:
  export RCRSO_STAGE0_RUN=/path/to/outputs/v16_8_45r3_stage0_runtime_observability
  # directory must contain stage0_val_support_audit.json and rcrso_stage0_selected.pt

Run:
  $0 audit_stage0_result
  $0 sanity
  $0 base_equivalence16_parallel2

Only after equivalence16 passes:
  $0 lost7_batch1_parallel2
  $0 analyze_lost7_progressive
  # run batch2 only if JSON continue_progressive=true
  $0 lost7_batch2_parallel2
  $0 analyze_lost7_progressive
  # run batch3 only if still continue_progressive=true
  $0 lost7_batch3_parallel2
  $0 analyze_lost7_progressive

Do not create/tune V46 from Stage-0 alone. lost7 >=2/7 is the next causal policy gate.
EOF
    ;;
esac
