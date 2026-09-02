#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# OC-RAP clean val/internal-test reconstruction for WOMD Motion v1.3.1
#
# Key design:
#   1) Use Waymax-native sharded path: validation_tfexample.tfrecord@150.
#   2) Use scenario_stride/scenario_worker_index for six deterministic,
#      mutually exclusive scenario partitions.
#   3) Use standard validation for the primary IID val/test datasets.
#   4) Do not use official testing/testing_interactive because their future
#      ground truth is hidden.
#   5) Do not use validation_interactive as the primary contact test source:
#      it is interaction-mined, not a collision-labelled dataset, and mixing it
#      with standard validation creates the drift seen in the previous build.
#
# Important semantic note:
#   test_contact/val_contact below are counterfactual contact-surrogate sets.
#   They are not observed real post-collision datasets.
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OCRAP_REPO="${OCRAP_REPO:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Always import the source tree beside this script.  Without this, an older
# editable/global OC-RAP installation can be imported even though the files in
# the current repository were updated.
export PYTHONPATH="${OCRAP_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${OCRAP_REPO}"

run_ocrap() {
  "${PYTHON_BIN}" -m ocrap.cli "$@"
}

WOMD_VAL_BASE="${WOMD_ROOT}/validation/validation_tfexample.tfrecord"
WOMD_VAL="${WOMD_VAL_BASE}@150"

# Six disjoint modulo partitions over the deterministic Waymax enumeration.
# Every raw scenario belongs to exactly one worker index.
PARTITION_STRIDE=6
VAL_SAFE_WORKER=0
TEST_SAFE_WORKER=1
VAL_NEAR_WORKER=2
TEST_NEAR_WORKER=3
VAL_CONTACT_WORKER=4
TEST_CONTACT_WORKER=5

# Raw-scenario scan budgets. Increase these if the final minimum-count audit
# reports too few accepted scenes. Each worker has roughly one sixth of the
# complete validation split available, so these values are safely below the
# available pool.
VAL_SAFE_RAW="${VAL_SAFE_RAW:-1200}"
TEST_SAFE_RAW="${TEST_SAFE_RAW:-1800}"
VAL_NEAR_RAW="${VAL_NEAR_RAW:-700}"
TEST_NEAR_RAW="${TEST_NEAR_RAW:-1000}"
VAL_CONTACT_RAW="${VAL_CONTACT_RAW:-700}"
TEST_CONTACT_RAW="${TEST_CONTACT_RAW:-1000}"

# Minimum accepted unique-scene requirements used by the final audit.
MIN_VAL_SAFE="${MIN_VAL_SAFE:-100}"
MIN_TEST_SAFE="${MIN_TEST_SAFE:-150}"
MIN_VAL_NEAR="${MIN_VAL_NEAR:-100}"
MIN_TEST_NEAR="${MIN_TEST_NEAR:-150}"
MIN_VAL_CONTACT="${MIN_VAL_CONTACT:-100}"
MIN_TEST_CONTACT="${MIN_TEST_CONTACT:-150}"

RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RESUME="${RESUME:-1}"
ADOPT_LEGACY_RESUME="${ADOPT_LEGACY_RESUME:-0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
REQUIRE_JAX_GPU="${REQUIRE_JAX_GPU:-1}"
REQUIRE_WAYMAX_STACK="${REQUIRE_WAYMAX_STACK:-1}"
PROFILE_BUILD="${PROFILE_BUILD:-0}"
RESET_DATASETS="${RESET_DATASETS:-}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-160}"
# Keep 0 for an exact continuation of datasets originally built with the full
# Waymax teacher.  A positive value enables screened-hybrid acceleration but is
# a semantic contract change and therefore requires clean outputs/shards.
NEAR_TEACHER_TOP_K="${NEAR_TEACHER_TOP_K:-0}"
CONTACT_TEACHER_TOP_K="${CONTACT_TEACHER_TOP_K:-0}"
NEAR_TEACHER_ROLLOUT_MODES="${NEAR_TEACHER_ROLLOUT_MODES:-}"
CONTACT_TEACHER_ROLLOUT_MODES="${CONTACT_TEACHER_ROLLOUT_MODES:-}"
STRESS_COMPUTE_FUTURE_METRICS="${STRESS_COMPUTE_FUTURE_METRICS:-true}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

verify_jax_gpu() {
  local gpu="$1"
  [[ "${REQUIRE_JAX_GPU}" == "1" ]] || return 0
  CUDA_VISIBLE_DEVICES="${gpu}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${PYTHON_BIN}" - <<'PY'
import jax
devices = jax.devices()
print({"jax_devices": [str(d) for d in devices]})
if not any(getattr(d, "platform", "") == "gpu" for d in devices):
    raise SystemExit("JAX cannot see a GPU. Install a CUDA-enabled jaxlib before the long Waymax build.")
PY
}

# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------

[[ -d "${WOMD_ROOT}/validation" ]] \
  || die "Missing WOMD validation directory: ${WOMD_ROOT}/validation"

VAL_SHARD_COUNT="$(
  "${PYTHON_BIN}" - "${WOMD_ROOT}/validation" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = sorted(root.glob("validation_tfexample.tfrecord-*-of-00150"))
print(len(files))
PY
)"

[[ "${VAL_SHARD_COUNT}" -eq 150 ]] \
  || die "Expected 150 standard validation shards, found ${VAL_SHARD_COUNT}."

echo "WOMD standard validation: ${VAL_SHARD_COUNT} shards"
echo "Waymax input pattern: ${WOMD_VAL}"
[[ "${GPU0}" != "${GPU1}" ]] \
  || die "GPU0 and GPU1 both resolve to ${GPU0}. Use two distinct CUDA device ids."
verify_jax_gpu "${GPU0}"
verify_jax_gpu "${GPU1}"
verify_waymax_stack() {
  [[ "${REQUIRE_WAYMAX_STACK}" == "1" ]] || return 0
  CUDA_VISIBLE_DEVICES="${GPU0}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["jax", "tensorflow", "waymax", "waymax.config", "waymax.dataloader"]
loaded = {}
for name in mods:
    module = importlib.import_module(name)
    loaded[name] = getattr(module, "__version__", "available")
print({"waymax_stack": loaded})
PY
}

verify_waymax_stack

if [[ -d "${WOMD_ROOT}/validation_interactive" ]]; then
  INTERACTIVE_COUNT="$(
    "${PYTHON_BIN}" - "${WOMD_ROOT}/validation_interactive" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
print(len(list(root.glob("*tfrecord-*-of-00150"))))
PY
  )"
  echo "WOMD validation_interactive detected (${INTERACTIVE_COUNT} shards);"
  echo "it is intentionally excluded from the primary IID val/test build."
fi

# The clean contact gate requires two explicit labels.  Verify both the source
# file and the module Python will actually import before starting a long build.
REGIME_FILE="${OCRAP_REPO}/src/ocrap/data/build/regimes.py"
[[ -f "${REGIME_FILE}" ]] || die "Missing source file: ${REGIME_FILE}"
grep -q '"post_contact_observed"' "${REGIME_FILE}" \
  || die "${REGIME_FILE} does not define post_contact_observed. Use the synchronized code package."
grep -q '"post_contact_counterfactual"' "${REGIME_FILE}" \
  || die "${REGIME_FILE} does not define post_contact_counterfactual. Use the synchronized code package."

"${PYTHON_BIN}" - "${OCRAP_REPO}" <<'PY'
from pathlib import Path
import inspect
import ocrap.data.build.regimes as regimes
import ocrap.data.build.builder as builder
import sys

repo = Path(sys.argv[1]).resolve()
regime_path = Path(inspect.getsourcefile(regimes) or "").resolve()
builder_path = Path(inspect.getsourcefile(builder) or "").resolve()
expected_regime = (repo / "src/ocrap/data/build/regimes.py").resolve()
expected_builder = (repo / "src/ocrap/data/build/builder.py").resolve()

if regime_path != expected_regime or builder_path != expected_builder:
    raise SystemExit(
        "ERROR: Python is importing a different OC-RAP installation. "
        f"regimes={regime_path}, builder={builder_path}, expected_repo={repo}"
    )

src = inspect.getsource(regimes.assign_regimes)
required = {"post_contact_observed", "post_contact_counterfactual"}
missing = sorted(k for k in required if k not in src)
if missing:
    raise SystemExit(f"ERROR: assign_regimes is missing labels: {missing}")

required_builder_symbols = {
    "_apply_scenario_scan_controls",
    "_scenario_source_config",
    "_selected_scenario_iterator",
}
missing_builder = sorted(name for name in required_builder_symbols if not hasattr(builder, name))
if missing_builder:
    raise SystemExit(
        "ERROR: builder.py is missing centralized WOMD scan-control helpers: "
        f"{missing_builder}"
    )

selected = list(
    builder._apply_scenario_scan_controls(
        iter(range(40)),
        start_index=3,
        stride=6,
        worker_index=4,
        max_scenarios=3,
    )
)
if selected != [7, 13, 19]:
    raise SystemExit(
        "ERROR: builder WOMD scan controls failed the functional single-application test: "
        f"got={selected}, expected=[7, 13, 19]"
    )
source_cfg = builder._scenario_source_config(
    {
        "scenario_start_index": 3,
        "scenario_stride": 6,
        "scenario_worker_index": 4,
        "max_scenarios": 3,
    }
)
expected_source = {
    "scenario_start_index": 0,
    "scenario_stride": 1,
    "scenario_worker_index": 0,
    "max_scenarios": 20,
}
for key, expected in expected_source.items():
    if source_cfg.get(key) != expected:
        raise SystemExit(
            "ERROR: builder source scan neutralization is incorrect: "
            f"key={key}, got={source_cfg.get(key)}, expected={expected}"
        )

print(f"OC-RAP source and WOMD partition preflight passed: {repo}")
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Preflight-only mode passed; no dataset was built."
  exit 0
fi

reset_one_dataset() {
  local name="$1"
  case "${name}" in
    val_safe|test_safe|val_near_contact|test_near_contact|val_contact|test_contact) ;;
    *) die "RESET_DATASETS contains unsupported dataset name: ${name}" ;;
  esac
  "${PYTHON_BIN}" - "${OCRAP_ROOT}" "${OCRAP_ROOT}/${name}" <<'PY'
from pathlib import Path
import shutil, sys
root = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
if target.parent != root:
    raise SystemExit(f"Refusing unsafe reset outside OCRAP_ROOT: {target}")
shutil.rmtree(target, ignore_errors=True)
PY
  rm -f -- "${OCRAP_ROOT}/.${name}.build.log"
  echo "reset dataset output: ${OCRAP_ROOT}/${name}"
}

if [[ -n "${RESET_DATASETS//[[:space:],]/}" ]]; then
  RESET_NORMALIZED="${RESET_DATASETS//,/ }"
  for d in ${RESET_NORMALIZED}; do
    reset_one_dataset "${d}"
  done
fi

if [[ "${RESUME}" != "1" ]]; then
  for d in \
    val_safe test_safe \
    val_near_contact test_near_contact \
    val_contact test_contact
  do
    [[ ! -e "${OCRAP_ROOT}/${d}" ]] \
      || die "${OCRAP_ROOT}/${d} already exists. Remove it or set RESUME=1."
  done
fi

mkdir -p "${OCRAP_ROOT}"

# -----------------------------------------------------------------------------
# Shared configuration
# -----------------------------------------------------------------------------

COMMON=(
  --set data_source=womd
  --set simulation_backend=waymax_closed_loop
  --set womd_patterns="${WOMD_VAL}"
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
  --set waymax.augmented_hidden_from_unknown_only=true

  --set dataset_quality.require_nominal_per_scene_time=true
  --set dataset_quality.keep_nominal_even_if_quality_fails=true
  --set dataset_quality.min_accepted_prefixes_per_scene_time=2

  --set regime_thresholds.include_prefix_collision_in_near=false
  --set regime_thresholds.include_prefix_contact_in_post=false
  --set regime_thresholds.use_paper_regime_definitions=true

  --set io.compress_npz=false
  --set io.fsync_npz=false
  --set profiling.enabled="${PROFILE_BUILD}"
)

resume_args_for() {
  local output="$1"
  RESUME_ARGS=()
  if [[ "${RESUME}" == "1" ]]; then
    RESUME_ARGS+=(--resume)
  fi
  if [[ "${ADOPT_LEGACY_RESUME}" == "1" && -d "${output}" && ! -f "${output}/resume_contract.json" ]]; then
    RESUME_ARGS+=(--adopt-resume-contract)
  fi
}

build_safe() {
  local split="$1"
  local worker="$2"
  local max_scenarios="$3"
  local output="$4"
  local gpu="$5"

  echo
  echo "==== Building ${output} from validation worker ${worker}/${PARTITION_STRIDE} ===="

  resume_args_for "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    run_ocrap build-dataset "${RESUME_ARGS[@]}" \
    "${COMMON[@]}" \
    --set scenario_worker_index="${worker}" \
    --set max_scenarios="${max_scenarios}" \
    --set split.force_id="${split}" \
    --set max_times_per_scenario=3 \
    --set max_biased_times_per_scenario=0 \
    --set dataset_quality.min_uniform_times_per_scenario=3 \
    --set num_targeted_futures=0 \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_rollout_top_k_options=4 \
    --set waymax.enable_augmented_hidden_roots=false \
    --set waymax.enable_visible_perturbation_roots=false \
    --set artifact.force_mine=false \
    --set artifact.mine_probability=0.0 \
    --set artifact.use_margin_override=false \
    --set dataset_quality.balanced_two_pass=false \
    --set dataset_quality.artifact_pair_mode=tag \
    --set dataset_quality.nominal_regime_dataset=true \
    --set dataset_quality.keep_nominal_even_if_quality_fails=false \
    --set dataset_quality.drop_scene_time_if_under_min_quality=true \
    --set dataset_quality.max_accepted_prefixes_per_scene_time=8 \
    --set 'dataset_quality.require_nominal_regimes=[normal]' \
    --set 'dataset_quality.forbid_nominal_regimes=[near_contact,post_contact,oracle_artifact,prefix_collision,prefix_contact]' \
    --set 'dataset_quality.forbid_any_regimes=[near_contact,post_contact,oracle_artifact,prefix_collision,prefix_contact]' \
    --set regime_thresholds.tau_occ=0.75 \
    --set regime_thresholds.tau_normal_occ=0.90 \
    --set regime_thresholds.require_uniform_for_normal=true \
    --output "${output}"
}

build_near() {
  local split="$1"
  local worker="$2"
  local max_scenarios="$3"
  local output="$4"
  local gpu="$5"

  echo
  echo "==== Building ${output} from validation worker ${worker}/${PARTITION_STRIDE} ===="

  resume_args_for "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    run_ocrap build-dataset "${RESUME_ARGS[@]}" \
    "${COMMON[@]}" \
    --set scenario_worker_index="${worker}" \
    --set max_scenarios="${max_scenarios}" \
    --set split.force_id="${split}" \
    --set max_times_per_scenario=3 \
    --set max_biased_times_per_scenario=3 \
    --set num_targeted_futures=8 \
    --set 'targeted_future_kinds=[hidden_vehicle_yields,hidden_vehicle_accelerates,low_friction_braking,control_delay_noise]' \
    --set waymax.compute_future_metrics="${STRESS_COMPUTE_FUTURE_METRICS}" \
    --set waymax.teacher_rollout_top_k_options="${NEAR_TEACHER_TOP_K}" \
    --set "waymax.teacher_rollout_option_modes=[${NEAR_TEACHER_ROLLOUT_MODES}]" \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.enable_augmented_hidden_roots=true \
    --set waymax.enable_visible_perturbation_roots=true \
    --set artifact.force_mine=true \
    --set artifact.mine_probability=0.30 \
    --set artifact.use_margin_override=false \
    --set artifact.enable_branch_intent_margin=true \
    --set artifact.branch_intent_compatible_margin=1.0 \
    --set artifact.branch_intent_incompatible_margin=-2.5 \
    --set dataset_quality.balanced_two_pass=true \
    --set dataset_quality.artifact_pair_mode=balanced \
    --set dataset_quality.artifact_quota_uses_label=true \
    --set dataset_quality.require_artifact_pairs=true \
    --set dataset_quality.max_accepted_prefixes_per_scene_time=8 \
    --set dataset_quality.min_artifact_prefixes_per_scene_time=1 \
    --set dataset_quality.max_artifact_prefixes_per_scene_time=2 \
    --set dataset_quality.min_nonartifact_prefixes_per_scene_time=4 \
    --set dataset_quality.max_nonartifact_prefixes_per_scene_time=6 \
    --set dataset_quality.max_artifact_attempts_per_scene_time=24 \
    --set dataset_quality.max_nonartifact_attempts_per_scene_time=12 \
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
  local split="$1"
  local worker="$2"
  local max_scenarios="$3"
  local output="$4"
  local gpu="$5"

  echo
  echo "==== Building ${output} from validation worker ${worker}/${PARTITION_STRIDE} ===="

  resume_args_for "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    run_ocrap build-dataset "${RESUME_ARGS[@]}" \
    "${COMMON[@]}" \
    --set scenario_worker_index="${worker}" \
    --set max_scenarios="${max_scenarios}" \
    --set split.force_id="${split}" \
    --set max_times_per_scenario=4 \
    --set max_biased_times_per_scenario=4 \
    --set num_targeted_futures=10 \
    --set 'targeted_future_kinds=[hidden_vehicle_yields,hidden_vehicle_accelerates,contact_impulse_surrogate,secondary_collision_approach,low_friction_braking,control_delay_noise]' \
    --set waymax.compute_future_metrics="${STRESS_COMPUTE_FUTURE_METRICS}" \
    --set waymax.teacher_rollout_top_k_options="${CONTACT_TEACHER_TOP_K}" \
    --set "waymax.teacher_rollout_option_modes=[${CONTACT_TEACHER_ROLLOUT_MODES}]" \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.enable_augmented_hidden_roots=true \
    --set waymax.enable_visible_perturbation_roots=true \
    --set artifact.force_mine=true \
    --set artifact.mine_probability=0.25 \
    --set artifact.use_margin_override=false \
    --set artifact.enable_branch_intent_margin=true \
    --set artifact.branch_intent_compatible_margin=1.0 \
    --set artifact.branch_intent_incompatible_margin=-2.5 \
    --set dataset_quality.balanced_two_pass=true \
    --set dataset_quality.artifact_pair_mode=balanced \
    --set dataset_quality.artifact_quota_uses_label=true \
    --set dataset_quality.require_artifact_pairs=true \
    --set dataset_quality.max_accepted_prefixes_per_scene_time=9 \
    --set dataset_quality.min_artifact_prefixes_per_scene_time=1 \
    --set dataset_quality.max_artifact_prefixes_per_scene_time=2 \
    --set dataset_quality.min_nonartifact_prefixes_per_scene_time=5 \
    --set dataset_quality.max_nonartifact_prefixes_per_scene_time=7 \
    --set dataset_quality.max_artifact_attempts_per_scene_time=24 \
    --set dataset_quality.max_nonartifact_attempts_per_scene_time=12 \
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

# -----------------------------------------------------------------------------
# Build six mutually exclusive datasets.  Run exactly one process per GPU;
# launching all six together causes JAX compilation and rollout contention.
# -----------------------------------------------------------------------------

show_build_log() {
  local name="$1"
  local log="${OCRAP_ROOT}/.${name}.build.log"
  echo >&2
  echo "----- ${name} log tail (${log}) -----" >&2
  if [[ -f "${log}" ]]; then
    tail -n "${LOG_TAIL_LINES}" "${log}" >&2 || true
  else
    echo "log file does not exist" >&2
  fi
  echo "----- end ${name} log -----" >&2
}

wait_pair() {
  local p0="$1" p1="$2" name0="$3" name1="$4"
  local s0=0 s1=0
  set +e
  wait "${p0}"; s0=$?
  wait "${p1}"; s1=$?
  set -e
  if [[ "${s0}" != 0 || "${s1}" != 0 ]]; then
    [[ "${s0}" == 0 ]] || show_build_log "${name0}"
    [[ "${s1}" == 0 ]] || show_build_log "${name1}"
    echo >&2
    echo "ERROR: dataset workers failed: ${name0}=${s0}, ${name1}=${s1}" >&2
    echo "Common causes are shown above: WOMD path/shard mismatch, Waymax API/import failure," >&2
    echo "CUDA/JAX failure, or an incompatible legacy resume contract." >&2
    exit 3
  fi
}

mkdir -p "${OCRAP_ROOT}/reports"

build_safe val "${VAL_SAFE_WORKER}" "${VAL_SAFE_RAW}" \
  "${OCRAP_ROOT}/val_safe" "${GPU0}" >"${OCRAP_ROOT}/.val_safe.build.log" 2>&1 & P0=$!
build_safe test "${TEST_SAFE_WORKER}" "${TEST_SAFE_RAW}" \
  "${OCRAP_ROOT}/test_safe" "${GPU1}" >"${OCRAP_ROOT}/.test_safe.build.log" 2>&1 & P1=$!
wait_pair "${P0}" "${P1}" val_safe test_safe

build_near val "${VAL_NEAR_WORKER}" "${VAL_NEAR_RAW}" \
  "${OCRAP_ROOT}/val_near_contact" "${GPU0}" >"${OCRAP_ROOT}/.val_near_contact.build.log" 2>&1 & P0=$!
build_near test "${TEST_NEAR_WORKER}" "${TEST_NEAR_RAW}" \
  "${OCRAP_ROOT}/test_near_contact" "${GPU1}" >"${OCRAP_ROOT}/.test_near_contact.build.log" 2>&1 & P1=$!
wait_pair "${P0}" "${P1}" val_near_contact test_near_contact

build_contact val "${VAL_CONTACT_WORKER}" "${VAL_CONTACT_RAW}" \
  "${OCRAP_ROOT}/val_contact" "${GPU0}" >"${OCRAP_ROOT}/.val_contact.build.log" 2>&1 & P0=$!
build_contact test "${TEST_CONTACT_WORKER}" "${TEST_CONTACT_RAW}" \
  "${OCRAP_ROOT}/test_contact" "${GPU1}" >"${OCRAP_ROOT}/.test_contact.build.log" 2>&1 & P1=$!
wait_pair "${P0}" "${P1}" val_contact test_contact

# -----------------------------------------------------------------------------
# Diagnose each dataset
# -----------------------------------------------------------------------------

if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
  mkdir -p "${OCRAP_ROOT}/reports"

  for d in \
    val_safe test_safe \
    val_near_contact test_near_contact \
    val_contact test_contact
  do
    if [[ "${d}" == *safe ]]; then
      run_ocrap diagnose \
        --dataset "${OCRAP_ROOT}/${d}" \
        --set dataset_quality.nominal_regime_dataset=true \
        --set 'dataset_quality.require_nominal_regimes=[normal]' \
        --output "${OCRAP_ROOT}/reports/diagnose_${d}.json"
    else
      run_ocrap diagnose \
        --dataset "${OCRAP_ROOT}/${d}" \
        --output "${OCRAP_ROOT}/reports/diagnose_${d}.json"
    fi

    run_ocrap papercheck \
      --dataset "${OCRAP_ROOT}/${d}" \
      --output "${OCRAP_ROOT}/reports/papercheck_${d}.json"
  done
fi

# -----------------------------------------------------------------------------
# Hard audit: scene isolation, minimum counts, and regime purity
# -----------------------------------------------------------------------------

"${PYTHON_BIN}" - \
  "${OCRAP_ROOT}" \
  "${MIN_VAL_SAFE}" "${MIN_TEST_SAFE}" \
  "${MIN_VAL_NEAR}" "${MIN_TEST_NEAR}" \
  "${MIN_VAL_CONTACT}" "${MIN_TEST_CONTACT}" <<'PY'
from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

root = Path(sys.argv[1])
minimums = {
    "val_safe": int(sys.argv[2]),
    "test_safe": int(sys.argv[3]),
    "val_near_contact": int(sys.argv[4]),
    "test_near_contact": int(sys.argv[5]),
    "val_contact": int(sys.argv[6]),
    "test_contact": int(sys.argv[7]),
}
names = list(minimums)

rows_by_name: dict[str, list[dict[str, str]]] = {}
scene_ids: dict[str, set[str]] = {}

for name in names:
    manifest = root / name / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"ERROR: missing manifest: {manifest}")

    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows_by_name[name] = rows
    ids = {
        (r.get("original_scenario_id") or r.get("scene_id") or "").strip()
        for r in rows
    }
    ids.discard("")
    scene_ids[name] = ids

    if len(ids) < minimums[name]:
        raise SystemExit(
            f"ERROR: {name} contains only {len(ids)} unique accepted scenes; "
            f"minimum is {minimums[name]}. Increase its *_RAW budget and rebuild."
        )

# All six sources must be scene-disjoint.
for left, right in itertools.combinations(names, 2):
    overlap = scene_ids[left] & scene_ids[right]
    if overlap:
        examples = sorted(overlap)[:5]
        raise SystemExit(
            f"ERROR: raw-scene overlap between {left} and {right}: "
            f"{len(overlap)} scenes; examples={examples}"
        )

def labels(row: dict[str, str]) -> set[str]:
    return {
        x.strip()
        for x in (row.get("regime_label") or "").split(";")
        if x.strip()
    }

for name, rows in rows_by_name.items():
    nominal = [r for r in rows if str(r.get("is_nominal", "0")) == "1"]
    if not nominal:
        raise SystemExit(f"ERROR: {name} contains no nominal samples.")

    if name.endswith("safe"):
        bad = [
            r for r in rows
            if labels(r) & {
                "near_contact", "post_contact", "oracle_artifact",
                "prefix_collision", "prefix_contact",
            }
        ]
        if bad:
            raise SystemExit(f"ERROR: {name} is regime-contaminated ({len(bad)} rows).")
        if any("normal" not in labels(r) for r in nominal):
            raise SystemExit(f"ERROR: {name} contains a non-normal nominal sample.")

    elif "near_contact" in name:
        if any("near_contact" not in labels(r) for r in nominal):
            raise SystemExit(f"ERROR: {name} contains a nominal sample not labelled near_contact.")
        if any("post_contact" in labels(r) for r in rows):
            raise SystemExit(f"ERROR: {name} contains post_contact contamination.")
        if any(
            labels(r) & {"prefix_collision", "prefix_contact"}
            for r in rows
        ):
            raise SystemExit(f"ERROR: {name} contains prefix collision/contact contamination.")

    elif "contact" in name:
        if any("post_contact_counterfactual" not in labels(r) for r in nominal):
            raise SystemExit(
                f"ERROR: {name} contains a nominal sample not labelled "
                "post_contact_counterfactual."
            )
        if any("post_contact_observed" in labels(r) for r in rows):
            raise SystemExit(f"ERROR: {name} contains observed-contact contamination.")
        if any(
            labels(r) & {"prefix_collision", "prefix_contact"}
            for r in rows
        ):
            raise SystemExit(f"ERROR: {name} contains prefix collision/contact contamination.")

print("\nAudit passed.")
for name in names:
    print(
        f"  {name}: {len(scene_ids[name])} unique accepted scenes, "
        f"{len(rows_by_name[name])} samples"
    )
print("  Pairwise raw-scene overlap: 0")
PY

echo
echo "Rebuild complete."
echo "Reports: ${OCRAP_ROOT}/reports"
