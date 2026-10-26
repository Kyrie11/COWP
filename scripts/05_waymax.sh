#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
CMD="${1:-waymax_inspect}"
case "$CMD" in waymax_inspect|waymax_smoke|waymax_priority22) ;; *) echo "Unknown Waymax command: $CMD" >&2; exit 2;; esac
mkdir -p "$RIDE_OUTPUT_ROOT/waymax"
cd "$RIDE_HOME"
if [[ "$CMD" == "waymax_inspect" ]]; then
  python - <<'PY'
import sys
from waymax import config, agents, env
print('Python:', sys.version)
print('Environment:', getattr(env,'BaseEnvironment',None),getattr(env,'MultiAgentEnvironment',None))
print('IDMRoutePolicy:',getattr(agents,'IDMRoutePolicy',None))
print('WOD_1_3_1_VALIDATION:',getattr(config,'WOD_1_3_1_VALIDATION',None))
PY
  exit 0
fi
COHORT="${RIDE_PREVIOUS_COHORT:-}"
IDS="${RIDE_PRIORITY_IDS:-}"
if [[ -z "$COHORT" || -z "$IDS" || ! -f "$COHORT/cohort_manifest.json" || ! -f "$IDS" ]]; then
  echo "Historical panel requires RIDE_PREVIOUS_COHORT=/path/to/cohort and RIDE_PRIORITY_IDS=/path/to/22_ids.txt" >&2
  echo "For a NEW blind panel use scripts/09_native_womd_cohort.sh and scripts/05b_waymax_native.sh." >&2
  exit 4
fi
if [[ "$CMD" == "waymax_smoke" ]]; then n=1; steps=40; else n=22; steps=80; fi
OUT="$RIDE_OUTPUT_ROOT/waymax/${CMD}.json"
if [[ -e "$OUT" ]]; then echo "Refusing to overwrite $OUT" >&2; exit 3; fi
python -m ride.waymax.paired_idm \
  --cohort "$COHORT" --raw-root "$RAW_WOMD_ROOT" --scenario-ids-file "$IDS" \
  --partition development --max-scenes "$n" --steps "$steps" --validate-map \
  --output "$OUT" 2>&1 | tee "$RIDE_OUTPUT_ROOT/waymax/${CMD}.log"
