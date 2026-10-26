#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
COHORT="${1:-$RIDE_OUTPUT_ROOT/womd/native_dev_partial}"
if [[ ! -f "$COHORT/cohort_manifest.json" ]]; then
  echo "No decoded WOMD cohort: $COHORT" >&2; exit 3
fi
mkdir -p "$RIDE_OUTPUT_ROOT/womd"
python -m ride.womd.observational \
  --cohort "$COHORT" --partition development \
  --max-scenes "${RIDE_OBS_SCENES:-320}" --steps 10 \
  --output "$RIDE_OUTPUT_ROOT/womd/observational_response_fit.json"
echo 'Observational fit ONLY: logged ego cannot identify do(ego action) responses.'
