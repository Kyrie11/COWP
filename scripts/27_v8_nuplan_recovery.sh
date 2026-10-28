#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
work="${RIDE_REACTSIM_WORK:-$DATASET_ROOT/RIDE/reactsim_workspace}"
coverage="${RIDE_COVERAGE_JSON:-$work/diagnostics/val_db_coverage.json}"
test -s "$coverage" || { echo "Missing $coverage; first run scripts/22_reactsim_db_coverage.sh" >&2;exit 3; }
python -m ride.nuplan_recovery_plan --coverage "$coverage" \
  --output "$work/diagnostics/v8_nuplan_recovery_plan.json" \
  --missing-list "$work/diagnostics/v8_missing_db_filenames.txt"
