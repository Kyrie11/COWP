#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
base="${RIDE_BASELINE_ROOT:-$DATASET_ROOT/RIDE/research_baselines}"
nuplan="${RIDE_NUPLAN_SOURCE_ROOT:-$DATASET_ROOT/CapPlan/data/nuplan}"
work="${RIDE_REACTSIM_WORK:-$DATASET_ROOT/RIDE/reactsim_workspace}"
mkdir -p "$work/diagnostics"
python -m ride.baseline_readiness --baselines "$base" --nuplan "$nuplan" \
  --output "$work/diagnostics/v8_baseline_preflight.json"
