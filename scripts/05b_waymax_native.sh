#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
MODE="${1:-smoke}"
case "$MODE" in smoke) n=1; steps=40;; panel) n=22; steps=80;;
  *) echo 'Usage: bash scripts/05b_waymax_native.sh smoke|panel' >&2; exit 2;; esac
COHORT="${RIDE_NATIVE_COHORT:-$RIDE_OUTPUT_ROOT/womd/native_full150_predefined}"
if [[ ! -f "$COHORT/cohort_manifest.json" ]]; then
  COHORT="${RIDE_NATIVE_COHORT:-$RIDE_OUTPUT_ROOT/womd/native_dev_partial}"
fi
if [[ ! -f "$COHORT/cohort_manifest.json" ]]; then
  echo "Missing decoded native WOMD cohort $COHORT. Execute scripts/09_native_womd_cohort.sh dev first." >&2
  exit 3
fi
mkdir -p "$RIDE_OUTPUT_ROOT/waymax"
# Always freeze the full 22-ID panel once; smoke reuses its first ID.
IDS="$RIDE_OUTPUT_ROOT/waymax/native_panel22_ids.txt"
if [[ ! -f "$IDS" ]]; then
  python -m ride.freeze_ids --cohort "$COHORT" --count 22 --output "$IDS"
fi
# For smoke, a separate first ID is reused without altering frozen full panel.
if [[ "$MODE" == "smoke" ]]; then
  IDLIST="$RIDE_OUTPUT_ROOT/waymax/native_smoke_id.txt"
  head -n 1 "$IDS" > "$IDLIST"
else
  IDLIST="$IDS"
fi
python -m research_probe.waymax_paired_idm \
  --cohort "$COHORT" --raw-root "$RAW_WOMD_ROOT" \
  --scenario-ids-file "$IDLIST" --partition development \
  --max-scenes "$n" --steps "$steps" --validate-map \
  --output "$RIDE_OUTPUT_ROOT/waymax/native_${MODE}.json" \
  2>&1 | tee "$RIDE_OUTPUT_ROOT/waymax/native_${MODE}.log"
