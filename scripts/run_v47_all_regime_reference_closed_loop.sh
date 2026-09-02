#!/usr/bin/env bash
set -euo pipefail

export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export SAFE_BUCKET=${SAFE_BUCKET:-$OCRAP_ROOT/val_safe}
export NEAR_BUCKET=${NEAR_BUCKET:-$OCRAP_ROOT/val_near_contact}
export CONTACT_BUCKET=${CONTACT_BUCKET:-$OCRAP_ROOT/val_contact}
export WOMD_VAL=${WOMD_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord}
export WOMD_VAL_INTERACTIVE=${WOMD_VAL_INTERACTIVE:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}
# Must match the raw WOMD source used to build the stress bucket roots.
# The synchronized v47 rebuild uses standard validation, not validation_interactive.
export WOMD_STRESS=${WOMD_STRESS:-$WOMD_VAL}
export RUN=${RUN:-runs/v47_reference_closed_loop}
export REFERENCE_BUCKET_SPLIT=${REFERENCE_BUCKET_SPLIT:-val}
export REFERENCE_MAX_ROLLOUTS=${REFERENCE_MAX_ROLLOUTS:-8}
export REFERENCE_MAX_STEPS=${REFERENCE_MAX_STEPS:-24}
export REFERENCE_MAX_TARGETS=${REFERENCE_MAX_TARGETS:-32}
export CL_USE_SDC_PATHS=${CL_USE_SDC_PATHS:-false}
mkdir -p "$RUN"

# These rollouts are deliberately independent of the learned OC-TRAC admission
# certificate. They provide the required all-regime physical comparison even
# when a candidate checkpoint is rejected before deployment. They must not be
# reported as OC-TRAC learned-policy results.
run_one() {
  local regime="$1" bucket="$2" womd="$3" gpu="$4"
  PYTHONPATH=src python tools/check_closed_loop_dataset_support.py \
    --dataset "$bucket" --womd-pattern "$womd" --split "$REFERENCE_BUCKET_SPLIT" \
    --output "$RUN/preflight_${regime}.json"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "${womd}@${REFERENCE_RAW_MAX_SCENARIOS:-900}" \
    --output "$RUN/closed_loop_${regime}_nominal_reference_v47.json" \
    --set closed_loop.method=nominal \
    --set closed_loop.bucket_dataset="$bucket" \
    --set closed_loop.bucket_split="$REFERENCE_BUCKET_SPLIT" \
    --set closed_loop.max_bucket_targets="$REFERENCE_MAX_TARGETS" \
    --set closed_loop.max_targets_per_scene=1 \
    --set closed_loop.max_rollouts="$REFERENCE_MAX_ROLLOUTS" \
    --set closed_loop.raw_max_scenarios=${REFERENCE_RAW_MAX_SCENARIOS:-900} \
    --set closed_loop.max_steps="$REFERENCE_MAX_STEPS" \
    --set closed_loop.replan_interval_steps=1 \
    --set closed_loop.num_candidate_prefixes=${REFERENCE_NUM_CANDIDATES:-8} \
    --set closed_loop.num_recovery_options=${REFERENCE_NUM_RECOVERY_OPTIONS:-6} \
    --set closed_loop.label_mode=fast \
    --set closed_loop.use_sdc_paths="$CL_USE_SDC_PATHS" \
    --set closed_loop.resume=true \
    --set closed_loop.save_partial=true \
    --set closed_loop.partial_write_every_scenes=2 \
    --set closed_loop.progress_every_steps=5 \
    2>&1 | tee "$RUN/closed_loop_${regime}_nominal_reference_v47.log"
}

run_one safe "$SAFE_BUCKET" "$WOMD_VAL" ${GPU_SAFE:-0} & P0=$!
run_one near_contact "$NEAR_BUCKET" "$WOMD_STRESS" ${GPU_NEAR:-1} & P1=$!
wait "$P0"; wait "$P1"
run_one contact "$CONTACT_BUCKET" "$WOMD_STRESS" ${GPU_CONTACT:-0}

PYTHONPATH=src python - "$RUN" <<'PY' | tee "$RUN/summary_reference_all_regimes_v47.txt"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
keys=[
 'num_scenes','num_decisions','collision_scene_rate','collision_step_rate','offroad_scene_rate','offroad_step_rate','minimum_clearance_m','minimum_ttc_s','intervention_rate','closed_loop_bounded_NUP',
 'closed_loop_FRA_exec','closed_loop_DRS','closed_loop_ODG',
 'overlap_mean','offroad_mean','min_clearance_m_min','ttc_s_min',
 'route_free_path_length_m','route_free_net_displacement_m','route_free_progress_efficiency',
 'longitudinal_accel_abs_mean_mps2','longitudinal_accel_abs_max_mps2','hard_brake_rate',
 'longitudinal_jerk_abs_mean_mps3','longitudinal_jerk_abs_max_mps3','yaw_rate_abs_mean_radps','yaw_rate_abs_max_radps',
]
for p in sorted(root.glob('closed_loop_*_nominal_reference_v47.json')):
    d=json.load(open(p))
    metrics=d.get('waymax_metrics', {}) or {}
    print('\n'+p.name)
    for k in keys:
        if k in d:
            print(f'  {k}: {d[k]}')
        elif k in metrics:
            print(f'  {k}: {metrics[k]}')
PY
