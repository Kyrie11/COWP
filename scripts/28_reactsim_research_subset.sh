#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
repo="${RIDE_REACTSIM_REPO:-$DATASET_ROOT/RIDE/research_baselines/ReactSim_Bench}"
work="${RIDE_REACTSIM_WORK:-$DATASET_ROOT/RIDE/reactsim_workspace}"
root="$work/research_only_subset"
mkdir -p "$root"
python -m ride.reactsim_research_subset \
 --ids "$repo/reactsim_data_pipeline/.scenario_ids_val.json" \
 --db-dir "$repo/data/nuplan/nuplan-v1.1/splits/trainval" \
 --target "${RIDE_SUBSET_TARGET:-32}" --max-groups "${RIDE_SUBSET_MAX_GROUPS:-200}" \
 --output-ids "$root/research_only_val_ids.json" --output-audit "$root/subset_manifest.json"
echo "RESEARCH ONLY subset, not official 2636. Can pass research_only_val_ids.json to author build_cache.py with an ISOLATED --output_dir after reviewing scenario requirements."
