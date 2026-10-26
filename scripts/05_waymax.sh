#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
CMD="${1:-waymax_smoke}"
case "$CMD" in waymax_inspect|waymax_smoke|waymax_priority22) ;; *) echo 'Unknown Waymax command' >&2; exit 2;; esac
# The retained V4 Waymax scripts require the *previously built* full-WOMD
# cohort and fixed 22-ID list. Fail clearly rather than fabricate results.
if [[ "$CMD" != "waymax_inspect" ]]; then
  ROOT="${EICDRR_FULL_ROOT:-/data0/senzeyu2/dataset/COWP/eicdrr_full_validation_20261014}"
  IDS="${H2_PRIORITY_IDS:-$ROOT/h2_gate_v2/h2_priority_dev_ids.txt}"
  if [[ ! -f "$ROOT/cohort/cohort_manifest.json" || ! -f "$IDS" ]]; then
    echo "REQUIRES previous EICDRR cohort: $ROOT/cohort/cohort_manifest.json and $IDS" >&2
    echo 'Set EICDRR_FULL_ROOT and H2_PRIORITY_IDS to existing files, or rebuild the cohort with the relevant original pipeline.' >&2
    exit 4
  fi
fi
mkdir -p "$RIDE_OUTPUT_ROOT/waymax"
cd "$RIDE_HOME/COWP"
# Dedicated Waymax JAX environment is recommended; CARLA_ROOT not needed.
export EICDRR_V4_ROOT="$RIDE_OUTPUT_ROOT/waymax"
bash NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh "$CMD" 2>&1 | tee "$RIDE_OUTPUT_ROOT/waymax/${CMD}.log"
