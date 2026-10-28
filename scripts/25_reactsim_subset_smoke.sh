#!/usr/bin/env bash
# Process explicitly labelled NON-BENCHMARK subset for installation smoke only.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
repo="${RIDE_REACTSIM_REPO:-$DATASET_ROOT/RIDE/research_baselines/ReactSim_Bench}"
work="${RIDE_REACTSIM_WORK:-$DATASET_ROOT/RIDE/reactsim_workspace}"
ids="$repo/reactsim_data_pipeline/.scenario_ids_val.json"
db="$repo/data/nuplan/nuplan-v1.1/splits/trainval"
subset="$work/subset_smoke_v8/ids_val_subset.json"
output="$work/subset_smoke_v8/processed_val"
if [[ -e "$subset" || -e "$output" ]]; then
 echo "Refusing to overwrite subset smoke outputs in $work/subset_smoke_v8" >&2;exit 3
fi
python -m ride.reactsim_subset --ids "$ids" --db-dir "$db" \
  --limit "${RIDE_REACTSIM_SMOKE_N:-12}" --output "$subset"
cd "$repo/reactsim_data_pipeline"
python build_cache.py --scenario_ids_path "$subset" \
  --scenario_cache_path "$work/subset_smoke_v8/scenario_objects.pkl" \
  --data_root "$repo/data/nuplan" --db_dir "$db" \
  --map_root "$repo/data/nuplan/maps" --output_dir "$output" \
  --num_workers 1 --skip_existing
printf 'WARNING: only exploratory subset cache; NOT full ReactSim-Bench evaluation.\n' >&2
