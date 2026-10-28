#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
mode="${1:-check}";shift || true
case "$mode" in check|apply) ;; *) echo 'Usage: bash scripts/15_reactsim_nuplan_stage.sh check|apply [--splits val ...]' >&2;exit 2;; esac
args=(--nuplan-root "${RIDE_NUPLAN_ROOT:-/data0/senzeyu2/dataset/CapPlan/data/nuplan}" \
  --reactsim-repo "${RIDE_REACTSIM_REPO:-$DATASET_ROOT/RIDE/research_baselines/ReactSim_Bench}" \
  --workspace "${RIDE_REACTSIM_WORK:-$DATASET_ROOT/RIDE/reactsim_workspace}" \
  --behaviors "${RIDE_REACTSIM_BEHAVIORS:-$DATASET_ROOT/RIDE/ReactSim-Bench/custom_behaviors}")
if [[ "$mode" == apply ]]; then args+=(--apply);fi
python -m ride.reactsim_setup "${args[@]}" "$@"
