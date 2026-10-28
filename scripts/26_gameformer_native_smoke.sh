#!/usr/bin/env bash
# Execute the author's ORIGINAL GameFormer planner with nuPlan IDM agents.
# This is not an RQ3 shared-candidate fair comparison.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
repo="${RIDE_GAMEFORMER_REPO:-$DATASET_ROOT/RIDE/research_baselines/GameFormer-Planner}"
nuplan="${RIDE_NUPLAN_ROOT:-$DATASET_ROOT/CapPlan/data/nuplan}"
model="${GAMEFORMER_MODEL_PTH:-}"
[[ -f "$repo/run_nuplan_test.py" ]] || { echo "GameFormer Planner source missing: $repo" >&2; exit 3; }
[[ -n "$model" && -s "$model" ]] || { echo 'Set GAMEFORMER_MODEL_PTH to a real, architecture-compatible trained model; repository source alone is insufficient.' >&2;exit 3; }
[[ -d "$nuplan/nuplan-v1.1/splits/val" ]] || exit 3
[[ -d "$nuplan/maps" ]] || exit 3
export NUPLAN_EXP_ROOT="${RIDE_GAMEFORMER_EXP_ROOT:-$DATASET_ROOT/RIDE/gameformer_native_exp}"
export NUPLAN_DATA_ROOT="$nuplan"
export NUPLAN_MAPS_ROOT="$nuplan/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
cd "$repo"
python run_nuplan_test.py --experiment_name closed_loop_reactive_agents \
 --data_path "$nuplan/nuplan-v1.1/splits/val" \
 --map_path "$nuplan/maps" --model_path "$model" \
 --scenarios_per_type "${RIDE_GF_PER_TYPE:-1}" \
 --total_scenarios "${RIDE_GF_TOTAL:-5}" --device "${RIDE_GF_DEVICE:-cuda}"
echo 'Author-original nuPlan+IDM smoke only; no RIDE candidate-pair regret has been measured.' >&2
