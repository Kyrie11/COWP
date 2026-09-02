#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

: "${OCRAP_RESULTS_ROOT:=runs/ocrap_three_regime_closed_loop_v50}"
: "${EXTERNAL_RESULTS_ROOT:=runs/all_regime_external_baselines_v50}"
: "${OUT:=runs/external_comparison_v50}"
: "${BUILD_VIDEOS:=false}"  # selection is metric-only; true triggers a selective 10-scene trace rerun
: "${ALLOW_INCOMPLETE_RUNS:=false}"
: "${FPS:=10}"
: "${REQUIRE_EXACT_VIDEO_SELECTION:=$BUILD_VIDEOS}"
: "${VIDEO_SELECTION_FALLBACK:=true}"
# Deterministic qualitative-example thresholds.  They can be tightened or
# relaxed without changing the full benchmark results or unsafe-regression
# guards in the selector.
: "${MIN_NEAR_TTC_GAIN_S:=0.25}"
: "${MIN_NEAR_CLEARANCE_GAIN_M:=0.25}"
: "${MIN_NEAR_EXPOSURE_REDUCTION_S:=0.20}"
: "${MIN_CONTACT_TERMINAL_CLEARANCE_GAIN_M:=0.50}"
: "${MIN_CONTACT_AUC_GAIN_M:=0.50}"
: "${MIN_CONTACT_CLEARANCE_GAIN_M:=0.25}"
: "${MIN_CONTACT_OVERLAP_DURATION_REDUCTION_S:=0.20}"
mkdir -p "$OUT/tables" "$OUT/selection"

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }
require_complete_index() {
  local p="$1"
  require_file "$p"
  python - "$p" "$ALLOW_INCOMPLETE_RUNS" <<'PY'
import json,sys
p=sys.argv[1]; allow=sys.argv[2].lower() in {'1','true','yes','on'}; d=json.load(open(p))
if not d.get('complete') and not allow: raise SystemExit(f'run index is incomplete: {p}; failed={d.get("failed_or_incomplete_regimes")}')
PY
}
scene_journal() {
  local p="$1"
  if [[ -f "${p}.scenes.jsonl" ]]; then printf '%s\n' "${p}.scenes.jsonl"
  elif [[ -f "${p%.*}.scenes.jsonl" ]]; then printf '%s\n' "${p%.*}.scenes.jsonl"
  else echo "Missing scene journal for $p" >&2; return 2; fi
}

require_complete_index "$OCRAP_RESULTS_ROOT/OCRAP_THREE_REGIME_RUN_INDEX.json"
require_complete_index "$EXTERNAL_RESULTS_ROOT/EXTERNAL_BASELINE_RUN_INDEX.json"

OCRAP_SAFE="$OCRAP_RESULTS_ROOT/safe/closed_loop_ocrap.json"
OCRAP_NEAR="$OCRAP_RESULTS_ROOT/near/closed_loop_ocrap.json"
OCRAP_CONTACT="$OCRAP_RESULTS_ROOT/contact/closed_loop_ocrap.json"
SAFE_INPUTS=(
  "ocrap=$OCRAP_SAFE"
  "nominal_replay=$EXTERNAL_RESULTS_ROOT/safe/closed_loop_nominal_replay.json"
  "wayformer_bc=$EXTERNAL_RESULTS_ROOT/safe/closed_loop_wayformer_bc.json"
  "gameformer_lite=$EXTERNAL_RESULTS_ROOT/safe/closed_loop_gameformer_lite.json"
  "betopnet_lite=$EXTERNAL_RESULTS_ROOT/safe/closed_loop_betopnet_lite.json"
)
NEAR_INPUTS=(
  "ocrap=$OCRAP_NEAR"
  "gameformer_lite=$EXTERNAL_RESULTS_ROOT/near/closed_loop_gameformer_lite.json"
  "marc_lite=$EXTERNAL_RESULTS_ROOT/near/closed_loop_marc_lite.json"
  "racp_lite=$EXTERNAL_RESULTS_ROOT/near/closed_loop_racp_lite.json"
  "expected_risk_filter=$EXTERNAL_RESULTS_ROOT/near/closed_loop_expected_risk_filter.json"
  "cvar_risk_filter=$EXTERNAL_RESULTS_ROOT/near/closed_loop_cvar_risk_filter.json"
  "dro_cvar_filter=$EXTERNAL_RESULTS_ROOT/near/closed_loop_dro_cvar_filter.json"
  "predictive_safety_filter=$EXTERNAL_RESULTS_ROOT/near/closed_loop_predictive_safety_filter.json"
)
CONTACT_INPUTS=(
  "ocrap=$OCRAP_CONTACT"
  "postimpact_mpc_lite=$EXTERNAL_RESULTS_ROOT/contact/closed_loop_postimpact_mpc_lite.json"
  "post_crash_braking=$EXTERNAL_RESULTS_ROOT/contact/closed_loop_post_crash_braking.json"
  "post_collision_restoration=$EXTERNAL_RESULTS_ROOT/contact/closed_loop_post_collision_restoration.json"
  "severity_minimization=$EXTERNAL_RESULTS_ROOT/contact/closed_loop_severity_minimization.json"
)
for spec in "${SAFE_INPUTS[@]}" "${NEAR_INPUTS[@]}" "${CONTACT_INPUTS[@]}"; do require_file "${spec#*=}"; done

run_table() { local regime="$1"; shift; local args=() spec; for spec in "$@"; do args+=(--input "$spec"); done; python tools/build_regime_comparison_tables.py --regime "$regime" --output-dir "$OUT/tables" "${args[@]}"; }
run_selection() { local regime="$1" output="$2"; shift 2; local args=() spec; for spec in "$@"; do args+=(--input "$spec"); done; python tools/select_best_external_baseline.py --regime "$regime" --output "$output" "${args[@]}"; }
run_table safe "${SAFE_INPUTS[@]}"
run_table near "${NEAR_INPUTS[@]}"
run_table contact "${CONTACT_INPUTS[@]}"
NEAR_EXTERNAL=("${NEAR_INPUTS[@]:1}"); CONTACT_EXTERNAL=("${CONTACT_INPUTS[@]:1}")
run_selection near "$OUT/selection/near_best_external.json" "${NEAR_EXTERNAL[@]}"
run_selection contact "$OUT/selection/contact_best_external.json" "${CONTACT_EXTERNAL[@]}"

eval "$(python - "$OUT/selection/near_best_external.json" "$OUT/selection/contact_best_external.json" <<'PY'
import json,shlex,sys
for prefix,path in zip(('NEAR','CONTACT'),sys.argv[1:]):
 d=json.load(open(path)); b=d['best']; print(f'{prefix}_BEST_METHOD={shlex.quote(b["method"])}'); print(f'{prefix}_BEST_RESULT={shlex.quote(b["path"])}')
PY
)"
NEAR_OCRAP_JOURNAL="$(scene_journal "$OCRAP_NEAR")"; NEAR_BASELINE_JOURNAL="$(scene_journal "$NEAR_BEST_RESULT")"
CONTACT_OCRAP_JOURNAL="$(scene_journal "$OCRAP_CONTACT")"; CONTACT_BASELINE_JOURNAL="$(scene_journal "$CONTACT_BEST_RESULT")"
SELECTION_EXACT_ARGS=()
if [[ "$REQUIRE_EXACT_VIDEO_SELECTION" == true ]]; then SELECTION_EXACT_ARGS+=(--require-exact-positive-count); fi
if [[ "$VIDEO_SELECTION_FALLBACK" == true ]]; then SELECTION_EXACT_ARGS+=(--fallback-topk-nonregressive); fi
python tools/select_critical_scenes_v48_34.py \
  --method-scenes "$NEAR_OCRAP_JOURNAL" --control-scenes "$NEAR_BASELINE_JOURNAL" --regime near \
  --num-positive 5 --num-failure 0 --max-per-scene 1 "${SELECTION_EXACT_ARGS[@]}" \
  --min-near-ttc-gain-s "$MIN_NEAR_TTC_GAIN_S" \
  --min-near-clearance-gain-m "$MIN_NEAR_CLEARANCE_GAIN_M" \
  --min-near-exposure-reduction-s "$MIN_NEAR_EXPOSURE_REDUCTION_S" \
  --output "$OUT/selection/near_selection.json" --target-keys-output "$OUT/selection/near_target_keys.json"
python tools/select_critical_scenes_v48_34.py \
  --method-scenes "$CONTACT_OCRAP_JOURNAL" --control-scenes "$CONTACT_BASELINE_JOURNAL" --regime contact \
  --num-positive 5 --num-failure 0 --max-per-scene 1 "${SELECTION_EXACT_ARGS[@]}" \
  --min-contact-terminal-clearance-gain-m "$MIN_CONTACT_TERMINAL_CLEARANCE_GAIN_M" \
  --min-contact-auc-gain-m "$MIN_CONTACT_AUC_GAIN_M" \
  --min-contact-clearance-gain-m "$MIN_CONTACT_CLEARANCE_GAIN_M" \
  --min-contact-overlap-duration-reduction-s "$MIN_CONTACT_OVERLAP_DURATION_REDUCTION_S" \
  --output "$OUT/selection/contact_selection.json" --target-keys-output "$OUT/selection/contact_target_keys.json"

if [[ "$BUILD_VIDEOS" == true ]]; then
  : "${EXTERNAL_CHECKPOINT_ROOT:?Set EXTERNAL_CHECKPOINT_ROOT when BUILD_VIDEOS=true}"
  SELECTION_ROOT="$OUT/selection" EXTERNAL_CHECKPOINT_ROOT="$EXTERNAL_CHECKPOINT_ROOT" \
  OUT="$OUT/selective_traces" FPS="$FPS" \
    bash scripts/run_selected_recovery_video_traces.sh
fi

python - "$OCRAP_RESULTS_ROOT" "$EXTERNAL_RESULTS_ROOT" "$OUT" "$BUILD_VIDEOS" "$NEAR_BEST_METHOD" "$CONTACT_BEST_METHOD" <<'PY'
import json,pathlib,sys
ocrap,external,out=map(pathlib.Path,sys.argv[1:4]); videos=sys.argv[4].lower()=='true'
files={'safe_table':out/'tables/safe_comparison.json','near_table':out/'tables/near_comparison.json','contact_table':out/'tables/contact_comparison.json','near_best_external':out/'selection/near_best_external.json','contact_best_external':out/'selection/contact_best_external.json','near_selection':out/'selection/near_selection.json','contact_selection':out/'selection/contact_selection.json','near_target_keys':out/'selection/near_target_keys.json','contact_target_keys':out/'selection/contact_target_keys.json'}
if videos: files['top10_video_index']=out/'selective_traces/videos/TOP10_VIDEO_INDEX.json'
doc={'event':'external_comparison_artifacts_v50','ocrap_results_root':str(ocrap),'external_results_root':str(external),'paired_target_set_required':True,'main_metric_protocol':'deployable physical closed-loop metrics; Contact uses post-contact physical recovery metrics','near_best_external':sys.argv[5],'contact_best_external':sys.argv[6],'video_pipeline':'metric-only full run -> deterministic key selection -> selective trace rerun -> MP4','artifacts':{k:str(v) for k,v in files.items()}}
(out/'COMPARISON_INDEX.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'event':doc['event'],'output':str(out),'videos':videos}))
PY
