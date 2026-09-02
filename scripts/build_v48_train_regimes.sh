#!/usr/bin/env bash
set -euo pipefail

# Build clean, distribution-matched Near/Contact training roots from WOMD
# training. Existing train_contact/train_near_contact are never modified.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO}"

WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data0/senzeyu2/dataset/OCRAP_v48_train}"
WOMD_TRAIN="${WOMD_TRAIN:-${WOMD_ROOT}/training/training_tfexample.tfrecord@1000}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RESUME="${RESUME:-1}"
NEAR_RAW_PER_WORKER="${NEAR_RAW_PER_WORKER:-5500}"
CONTACT_RAW_PER_WORKER="${CONTACT_RAW_PER_WORKER:-7000}"
PARTITION_STRIDE="${PARTITION_STRIDE:-2}"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

run_ocrap() { "${PYTHON_BIN}" -m ocrap.cli "$@"; }
resume_args=()
[[ "${RESUME}" == "1" ]] && resume_args+=(--resume)

COMMON=(
  --set data_source=womd
  --set simulation_backend=waymax_closed_loop
  --set womd_patterns="${WOMD_TRAIN}"
  --set scenario_start_index=0
  --set scenario_stride="${PARTITION_STRIDE}"
  --set num_candidate_prefixes=24
  --set num_reactive_futures=2
  --set num_roots=8
  --set num_recovery_options=12
  --set waymax.dataloader_include_sdc_paths=true
  --set 'waymax.metrics_to_run=[log_divergence,overlap,offroad,sdc_wrongway,sdc_off_route,sdc_progression,kinematic_infeasibility]'
  --set waymax.teacher_backend=hybrid
  --set waymax.use_jit_scan_rollouts=true
  --set waymax.teacher_metrics_stride=0
  --set waymax.cache_env_objects=true
  --set waymax.cache_postprefix_rollouts=true
  --set waymax.cache_teacher_metric_rollouts=true
  --set waymax.cache_identical_teacher_rollouts=true
  --set waymax.compute_future_metrics=true
  --set waymax.enable_augmented_hidden_roots=true
  --set waymax.enable_visible_perturbation_roots=true
  --set dataset_quality.require_nominal_per_scene_time=true
  --set dataset_quality.keep_nominal_even_if_quality_fails=true
  --set dataset_quality.min_accepted_prefixes_per_scene_time=2
  --set dataset_quality.balanced_two_pass=true
  --set dataset_quality.balanced_rotate_prefix_order=true
  --set dataset_quality.artifact_pair_mode=balanced
  --set dataset_quality.artifact_quota_uses_label=true
  --set dataset_quality.require_artifact_pairs=true
  --set dataset_quality.min_artifact_prefixes_per_scene_time=1
  --set dataset_quality.max_artifact_prefixes_per_scene_time=2
  --set dataset_quality.max_artifact_attempts_per_scene_time=24
  --set dataset_quality.max_nonartifact_attempts_per_scene_time=14
  --set artifact.force_mine=true
  --set artifact.use_margin_override=false
  --set artifact.enable_branch_intent_margin=true
  --set artifact.branch_intent_compatible_margin=1.0
  --set artifact.branch_intent_incompatible_margin=-2.5
  --set regime_thresholds.include_prefix_collision_in_near=false
  --set regime_thresholds.include_prefix_contact_in_post=false
  --set regime_thresholds.use_paper_regime_definitions=true
  --set io.compress_npz=false
  --set io.fsync_npz=false
)

build_near() {
  local worker="$1" output="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  run_ocrap build-dataset "${resume_args[@]}" "${COMMON[@]}" \
    --set scenario_worker_index="${worker}" --set max_scenarios="${NEAR_RAW_PER_WORKER}" \
    --set split.force_id=train --set max_times_per_scenario=4 --set max_biased_times_per_scenario=4 \
    --set num_targeted_futures=8 \
    --set 'targeted_future_kinds=[hidden_vehicle_yields,hidden_vehicle_accelerates,low_friction_braking,control_delay_noise]' \
    --set artifact.mine_probability=0.30 \
    --set dataset_quality.max_accepted_prefixes_per_scene_time=8 \
    --set dataset_quality.min_nonartifact_prefixes_per_scene_time=4 \
    --set dataset_quality.max_nonartifact_prefixes_per_scene_time=6 \
    --set 'dataset_quality.require_nominal_regimes=[near_contact]' \
    --set 'dataset_quality.require_any_regimes=[oracle_artifact]' \
    --set 'dataset_quality.forbid_nominal_regimes=[post_contact,prefix_collision,prefix_contact]' \
    --set 'dataset_quality.forbid_any_regimes=[post_contact,prefix_collision,prefix_contact]' \
    --set dataset_quality.artifact_pass_use_margin_override=true \
    --set dataset_quality.artifact_pass_skip_augmented_waymax=true \
    --set dataset_quality.artifact_pass_apply_override_to_screened=true \
    --set dataset_quality.artifact_pass_compute_future_metrics=false \
    --output "${output}"
}

build_contact() {
  local worker="$1" output="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  run_ocrap build-dataset "${resume_args[@]}" "${COMMON[@]}" \
    --set scenario_worker_index="${worker}" --set max_scenarios="${CONTACT_RAW_PER_WORKER}" \
    --set split.force_id=train --set max_times_per_scenario=5 --set max_biased_times_per_scenario=5 \
    --set num_targeted_futures=10 \
    --set 'targeted_future_kinds=[hidden_vehicle_yields,hidden_vehicle_accelerates,contact_impulse_surrogate,secondary_collision_approach,low_friction_braking,control_delay_noise]' \
    --set artifact.mine_probability=0.25 \
    --set dataset_quality.max_accepted_prefixes_per_scene_time=9 \
    --set dataset_quality.min_nonartifact_prefixes_per_scene_time=5 \
    --set dataset_quality.max_nonartifact_prefixes_per_scene_time=7 \
    --set 'dataset_quality.require_nominal_regimes=[post_contact_counterfactual]' \
    --set 'dataset_quality.require_any_regimes=[oracle_artifact]' \
    --set 'dataset_quality.forbid_nominal_regimes=[post_contact_observed,prefix_collision,prefix_contact]' \
    --set 'dataset_quality.forbid_any_regimes=[post_contact_observed,prefix_collision,prefix_contact]' \
    --set dataset_quality.artifact_pass_use_margin_override=true \
    --set dataset_quality.artifact_pass_skip_augmented_waymax=true \
    --set dataset_quality.artifact_pass_apply_override_to_screened=true \
    --set dataset_quality.artifact_pass_compute_future_metrics=false \
    --output "${output}"
}

wait_pair() {
  local p0="$1" p1="$2" name0="$3" name1="$4"
  local s0=0 s1=0
  set +e; wait "${p0}"; s0=$?; wait "${p1}"; s1=$?; set -e
  if [[ ${s0} -ne 0 || ${s1} -ne 0 ]]; then
    echo "worker failure: ${name0}=${s0}, ${name1}=${s1}" >&2
    tail -n 120 "${LOG_DIR}/${name0}.log" >&2 || true
    tail -n 120 "${LOG_DIR}/${name1}.log" >&2 || true
    exit 3
  fi
}

mkdir -p "${OUTPUT_ROOT}/shards"
build_near 0 "${OUTPUT_ROOT}/shards/train_near_w0" "${GPU0}" >"${LOG_DIR}/train_near_w0.log" 2>&1 & P0=$!
build_near 1 "${OUTPUT_ROOT}/shards/train_near_w1" "${GPU1}" >"${LOG_DIR}/train_near_w1.log" 2>&1 & P1=$!
wait_pair "${P0}" "${P1}" train_near_w0 train_near_w1

build_contact 0 "${OUTPUT_ROOT}/shards/train_contact_w0" "${GPU0}" >"${LOG_DIR}/train_contact_w0.log" 2>&1 & P0=$!
build_contact 1 "${OUTPUT_ROOT}/shards/train_contact_w1" "${GPU1}" >"${LOG_DIR}/train_contact_w1.log" 2>&1 & P1=$!
wait_pair "${P0}" "${P1}" train_contact_w0 train_contact_w1

rm -rf "${OUTPUT_ROOT}/train_near_contact" "${OUTPUT_ROOT}/train_contact"
"${PYTHON_BIN}" tools/merge_dataset_roots.py --hardlink --output "${OUTPUT_ROOT}/train_near_contact" \
  "${OUTPUT_ROOT}/shards/train_near_w0" "${OUTPUT_ROOT}/shards/train_near_w1" | tee "${LOG_DIR}/merge_near.log"
"${PYTHON_BIN}" tools/merge_dataset_roots.py --hardlink --output "${OUTPUT_ROOT}/train_contact" \
  "${OUTPUT_ROOT}/shards/train_contact_w0" "${OUTPUT_ROOT}/shards/train_contact_w1" | tee "${LOG_DIR}/merge_contact.log"

run_ocrap diagnose --dataset "${OUTPUT_ROOT}/train_near_contact" --output "${OUTPUT_ROOT}/diagnose_train_near_contact.json" | tee "${LOG_DIR}/diagnose_near.log"
run_ocrap diagnose --dataset "${OUTPUT_ROOT}/train_contact" --output "${OUTPUT_ROOT}/diagnose_train_contact.json" | tee "${LOG_DIR}/diagnose_contact.log"

echo "v48 train roots complete: ${OUTPUT_ROOT}"
