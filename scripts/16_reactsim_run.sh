#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
cmd="${1:?Usage: bash scripts/16_reactsim_run.sh preflight|preprocess|smart|mtr}";shift
repo="${RIDE_REACTSIM_REPO:-$DATASET_ROOT/RIDE/research_baselines/ReactSim_Bench}"
nuplan="$repo/data/nuplan";scenarios="$repo/data/nuplan_processed/val"
export NUPLAN_DATA_ROOT="$nuplan"
export NUPLAN_MAPS_ROOT="$nuplan/maps"
export NUPLAN_EXP_ROOT="${RIDE_REACTSIM_WORK:-$DATASET_ROOT/RIDE/reactsim_workspace}/nuplan_exp"
cd "$repo"
test -d "$nuplan/maps" || { echo 'Missing staged maps. First run scripts/15_reactsim_nuplan_stage.sh apply' >&2;exit 3; }
test -d "$nuplan/nuplan-v1.1/splits/trainval" || { echo 'Missing staged trainval link' >&2;exit 3; }
test -d data/custom_behaviors || { echo 'Missing author custom behaviors' >&2;exit 3; }
case "$cmd" in
 preflight)
   echo "repo=$repo"
   echo "nuplan=$nuplan"
   echo "NUPLAN_EXP_ROOT=$NUPLAN_EXP_ROOT"
   find -L data/custom_behaviors -maxdepth 1 -name '*.pkl' | wc -l
   echo "available db: $(find -L "$nuplan/nuplan-v1.1/splits/trainval" -name '*.db' | wc -l)"
   if [[ -d "$scenarios/processed_scenarios" ]]; then
     val_count=$(find -L "$scenarios/processed_scenarios" -maxdepth 1 -name '*.pkl' | wc -l)
     echo "processed val cache files: $val_count (full author val=2636; count alone is not proof of token alignment)"
   else echo 'processed val: MISSING (preprocess required)'; fi
   [[ -f ckpts/smart.ckpt ]] && ls -lh ckpts/smart.ckpt || true
   [[ -f ckpts/mtr.pth ]] && ls -lh ckpts/mtr.pth || true
   ;;
 preprocess)
   echo 'RIDE v6: VAL-only verified preprocessing. Train caches are optional and need separate opt-in.' >&2
   bash "$RIDE_HOME/scripts/17_reactsim_preprocess_checked.sh" val "$@"
   ;;
 smart|mtr)
   test -d "$scenarios/processed_scenarios" || { echo "Missing processed val data: $scenarios" >&2;exit 3; }
   ckpt=smart.ckpt;dir=SMART_CATK_TrajTok;config=configs/experiment/smart.yaml;batch=16
   if [[ "$cmd" == mtr ]]; then ckpt=mtr.pth;dir=MTR;config=tools/cfgs/nuplan/mtr.yaml;batch=4;fi
   test -s "ckpts/$ckpt" || { echo "Missing official checkpoint ckpts/$ckpt" >&2;exit 3; }
   test -f "$dir/tools/closed_loop_test.py" || { echo "Author evaluation entry missing in $dir" >&2;exit 3; }
   (cd "$dir" && python tools/closed_loop_test.py \
      --model_path "../ckpts/$ckpt" --config_file "$config" \
      --scenario_dir '../data/nuplan_processed/val' \
      --map_root '../data/nuplan/maps' \
      --custom_behavior_dir '../data/custom_behaviors' \
      --rollout_save_path "outputs/${cmd}_closed_loop/rollouts" \
      --replan_interval 5 --batch_size "$batch" --visualize_mode video "$@")
   ;;
 *) echo "Unknown command: $cmd" >&2;exit 2;;
esac
