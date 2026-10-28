#!/usr/bin/env bash
# Diagnose whether ReactSim logs are absent or merely outside the staged split.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
repo="${RIDE_REACTSIM_REPO:-$DATASET_ROOT/RIDE/research_baselines/ReactSim_Bench}"
nuplan_source="${RIDE_NUPLAN_SOURCE_ROOT:-$DATASET_ROOT/CapPlan/data/nuplan}"
ids="${1:-$repo/reactsim_data_pipeline/.scenario_ids_val.json}"
out="${2:-${RIDE_REACTSIM_WORK:-$DATASET_ROOT/RIDE/reactsim_workspace}/diagnostics/val_db_coverage.json}"
python -m ride.reactsim_db_coverage --nuplan-root "$nuplan_source" --ids "$ids" \
  --staged-splits ${RIDE_NUPLAN_STAGED_SPLITS:-val train_boston train_pittsburgh train_singapore train_vegas} \
  --output "$out"
