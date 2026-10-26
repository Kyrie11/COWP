#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
MODE="${1:-dev}"
mkdir -p "$RIDE_OUTPUT_ROOT/womd"
case "$MODE" in
  dev)
    OUT="$RIDE_OUTPUT_ROOT/womd/native_dev_partial"
    EXTRA=(--max-shards 4 --max-records-per-shard 350 --keep-random 120
           --keep-interactive 120 --keep-chain 75 --keep-crossing 75 --keep-cpa 75)
    ;;
  full150)
    OUT="$RIDE_OUTPUT_ROOT/womd/native_full150_predefined"
    EXTRA=(--max-shards 0 --max-records-per-shard 0 --require-complete-split
           --keep-random 600 --keep-interactive 450 --keep-chain 250
           --keep-crossing 250 --keep-cpa 250 --holdout-fraction 0.20)
    ;;
  *) echo 'Usage: bash scripts/09_native_womd_cohort.sh dev|full150' >&2; exit 2;;
esac
if [[ -e "$OUT/cohort_manifest.json" ]]; then
  echo "Refusing to overwrite existing cohort manifest: $OUT" >&2
  exit 3
fi
cd "$RIDE_HOME"
python -m ride.womd.native --raw-root "$RAW_WOMD_ROOT" \
 --output "$OUT" --split validation --format tfexample \
 --exclude-ids "$RIDE_HOME/references/prior_development_exclusion_ids.txt" \
 "${EXTRA[@]}"
