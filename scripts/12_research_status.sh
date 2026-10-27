#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
OUT="$RIDE_OUTPUT_ROOT/carla"
STATUS="$OUT/research_status.json"
if [[ ! -f "$OUT/all_scenes_audit.json" ]]; then
  echo "Missing $OUT/all_scenes_audit.json; run scripts/04_analyze_carla.sh first" >&2
  exit 3
fi
cd "$RIDE_HOME"
ARGS=(--carla-audit "$OUT/all_scenes_audit.json" --output "$STATUS")
[[ -f "$OUT/risk_ranking_disagreement.json" ]] && ARGS+=(--ranking "$OUT/risk_ranking_disagreement.json")
[[ -f "$OUT/science_gates.json" ]] && ARGS+=(--gates "$OUT/science_gates.json")
[[ -f "$OUT/support_gap_bank_ablation.json" ]] && ARGS+=(--support-gap "$OUT/support_gap_bank_ablation.json")
[[ -f "$RIDE_OUTPUT_ROOT/womd/observational_response_fit.json" ]] && \
  ARGS+=(--womd-fit "$RIDE_OUTPUT_ROOT/womd/observational_response_fit.json")
[[ -f "$RIDE_OUTPUT_ROOT/waymax/native_panel.json" ]] && \
  ARGS+=(--waymax "$RIDE_OUTPUT_ROOT/waymax/native_panel.json")
python -m ride.research_status "${ARGS[@]}"
echo "Wrote $STATUS"
