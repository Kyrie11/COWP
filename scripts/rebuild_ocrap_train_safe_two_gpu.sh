#!/usr/bin/env bash
set -euo pipefail

# Strict, resumable two-GPU train_safe reconstruction.  The target is a complete
# scene-time-group dataset containing 15k--20k samples, not a candidate-level
# dataset in which 95% of every alternative prefix must be labelled normal.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OCRAP_REPO="${OCRAP_REPO:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export PYTHONPATH="${OCRAP_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${OCRAP_REPO}"

export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WOMD_TRAIN="${WOMD_ROOT}/training/training_tfexample.tfrecord@1000"
TRAIN_SAFE_NAME="${TRAIN_SAFE_NAME:-train_safe}"
OUTPUT="${OCRAP_ROOT}/${TRAIN_SAFE_NAME}"
SHARD_ROOT="${OCRAP_ROOT}/.${TRAIN_SAFE_NAME}_shards"
RAW_PER_WORKER="${RAW_PER_WORKER:-6000}"
MIN_SAMPLES="${MIN_SAMPLES:-15000}"
MAX_SAMPLES="${MAX_SAMPLES:-20000}"
TAU_NORMAL_DEP="${TAU_NORMAL_DEP:-0.50}"
TAU_NORMAL_OCC="${TAU_NORMAL_OCC:-0.90}"
RESUME="${RESUME:-1}"
ADOPT_LEGACY_RESUME="${ADOPT_LEGACY_RESUME:-0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
REQUIRE_JAX_GPU="${REQUIRE_JAX_GPU:-1}"
REQUIRE_WAYMAX_STACK="${REQUIRE_WAYMAX_STACK:-1}"
PROFILE_BUILD="${PROFILE_BUILD:-0}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-160}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -d "${WOMD_ROOT}/training" ]] || die "Missing WOMD training directory: ${WOMD_ROOT}/training"
TRAIN_SHARD_COUNT="$(
  "${PYTHON_BIN}" - "${WOMD_ROOT}/training" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
print(len(list(root.glob("training_tfexample.tfrecord-*-of-01000"))))
PY
)"
[[ "${TRAIN_SHARD_COUNT}" -eq 1000 ]] \
  || die "Expected 1000 WOMD training shards, found ${TRAIN_SHARD_COUNT} under ${WOMD_ROOT}/training"

echo "WOMD training shards: ${TRAIN_SHARD_COUNT}"
echo "Waymax input pattern: ${WOMD_TRAIN}"
mkdir -p "${OCRAP_ROOT}" "${OCRAP_ROOT}/reports"

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

"${PYTHON_BIN}" - "${OCRAP_REPO}" <<'PY'
from pathlib import Path
import inspect
import sys
import ocrap.data.build.builder as builder

repo = Path(sys.argv[1]).resolve()
builder_path = Path(inspect.getsourcefile(builder) or "").resolve()
expected_builder = (repo / "src/ocrap/data/build/builder.py").resolve()
if builder_path != expected_builder:
    raise SystemExit(
        "ERROR: Python is importing a different OC-RAP installation: "
        f"builder={builder_path}, expected={expected_builder}"
    )
required = {
    "_apply_scenario_scan_controls",
    "_scenario_source_config",
    "_selected_scenario_iterator",
}
missing = sorted(name for name in required if not hasattr(builder, name))
if missing:
    raise SystemExit(f"ERROR: builder is missing centralized WOMD scan helpers: {missing}")
selected = list(
    builder._apply_scenario_scan_controls(
        iter(range(20)), start_index=0, stride=2, worker_index=1, max_scenarios=4
    )
)
if selected != [1, 3, 5, 7]:
    raise SystemExit(
        "ERROR: WOMD two-worker partition functional test failed: "
        f"got={selected}, expected=[1, 3, 5, 7]"
    )
print(f"OC-RAP source and WOMD partition preflight passed: {repo}")
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Preflight-only mode passed; no dataset was built."
  exit 0
fi

safe_reset_path() {
  local target="$1"
  "${PYTHON_BIN}" - "${OCRAP_ROOT}" "${target}" <<'PY'
from pathlib import Path
import shutil, sys
root = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
if target == root or root not in target.parents:
    raise SystemExit(f"Refusing unsafe reset outside OCRAP_ROOT: {target}")
shutil.rmtree(target, ignore_errors=True)
PY
}

if [[ "${RESET_OUTPUT}" == "1" ]]; then
  echo "Resetting final train_safe output and hidden worker shards."
  safe_reset_path "${OUTPUT}"
  safe_reset_path "${SHARD_ROOT}"
fi

mkdir -p "${SHARD_ROOT}"

if [[ "${RESUME}" != "1" ]]; then
  [[ ! -e "${OUTPUT}" ]] || die "${OUTPUT} exists. Remove it, use RESET_OUTPUT=1, or set RESUME=1."
  for worker in 0 1; do
    [[ ! -e "${SHARD_ROOT}/worker${worker}" ]] \
      || die "${SHARD_ROOT}/worker${worker} exists. Use RESET_OUTPUT=1 for a clean replacement or RESUME=1."
  done
fi

COMMON=(
  --set data_source=womd
  --set simulation_backend=waymax_closed_loop
  --set womd_patterns="${WOMD_TRAIN}"
  --set scenario_start_index=0
  --set scenario_stride=2
  --set max_scenarios="${RAW_PER_WORKER}"
  --set split.force_id=train
  --set max_times_per_scenario=3
  --set max_biased_times_per_scenario=0
  --set dataset_quality.min_uniform_times_per_scenario=3
  --set num_candidate_prefixes=24
  --set num_reactive_futures=2
  --set num_targeted_futures=0
  --set num_roots=8
  --set num_recovery_options=12
  --set waymax.dataloader_include_sdc_paths=true
  --set 'waymax.metrics_to_run=[log_divergence,overlap,offroad,sdc_wrongway,sdc_off_route,sdc_progression,kinematic_infeasibility]'
  --set waymax.compute_future_metrics=false
  --set waymax.teacher_backend=hybrid
  --set waymax.teacher_rollout_top_k_options=4
  --set waymax.teacher_metrics_stride=0
  --set waymax.use_jit_scan_rollouts=true
  --set waymax.cache_env_objects=true
  --set waymax.cache_postprefix_rollouts=true
  --set waymax.cache_teacher_metric_rollouts=true
  --set waymax.cache_identical_teacher_rollouts=true
  --set waymax.enable_augmented_hidden_roots=false
  --set waymax.enable_visible_perturbation_roots=false
  --set artifact.force_mine=false
  --set artifact.mine_probability=0.0
  --set artifact.use_margin_override=false
  --set dataset_quality.balanced_two_pass=false
  --set dataset_quality.artifact_pair_mode=tag
  --set dataset_quality.nominal_regime_dataset=true
  --set dataset_quality.require_nominal_per_scene_time=true
  --set dataset_quality.keep_nominal_even_if_quality_fails=false
  --set dataset_quality.drop_scene_time_if_under_min_quality=true
  --set dataset_quality.min_accepted_prefixes_per_scene_time=2
  --set dataset_quality.max_accepted_prefixes_per_scene_time=8
  --set 'dataset_quality.require_nominal_regimes=[normal]'
  --set 'dataset_quality.forbid_nominal_regimes=[near_contact,post_contact,oracle_artifact,prefix_collision,prefix_contact]'
  --set 'dataset_quality.forbid_any_regimes=[near_contact,post_contact,oracle_artifact,prefix_collision,prefix_contact]'
  --set regime_thresholds.tau_normal_dep="${TAU_NORMAL_DEP}"
  --set regime_thresholds.tau_occ=0.75
  --set regime_thresholds.tau_normal_occ="${TAU_NORMAL_OCC}"
  --set regime_thresholds.require_uniform_for_normal=true
  --set regime_thresholds.include_prefix_collision_in_near=false
  --set regime_thresholds.include_prefix_contact_in_post=false
  --set regime_thresholds.use_paper_regime_definitions=true
  --set io.compress_npz=false
  --set io.fsync_npz=false
  --set profiling.enabled="${PROFILE_BUILD}"
)

build_worker() {
  # Do not reference ${worker} in the same `local` command that assigns it.
  # With `set -u`, Bash expands all right-hand sides before the assignments are
  # installed, which made both workers fail immediately with "unbound variable".
  local worker="$1"
  local gpu="$2"
  local out="${SHARD_ROOT}/worker${worker}"
  local resume_arg=()
  if [[ "${RESUME}" == "1" ]]; then resume_arg=(--resume); fi
  if [[ "${ADOPT_LEGACY_RESUME}" == "1" && -d "${out}" && ! -f "${out}/resume_contract.json" ]]; then
    resume_arg+=(--adopt-resume-contract)
  fi
  echo "building strict train_safe worker ${worker}/2 on GPU ${gpu}: ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${PYTHON_BIN}" -m ocrap.cli build-dataset \
    "${resume_arg[@]}" "${COMMON[@]}" --set scenario_worker_index="${worker}" --output "${out}"
}

WORKER0_LOG="${SHARD_ROOT}/worker0.launch.log"
WORKER1_LOG="${SHARD_ROOT}/worker1.launch.log"
build_worker 0 "${GPU0}" >"${WORKER0_LOG}" 2>&1 & P0=$!
build_worker 1 "${GPU1}" >"${WORKER1_LOG}" 2>&1 & P1=$!
set +e
wait "${P0}"; S0=$?
wait "${P1}"; S1=$?
set -e
if [[ "${S0}" != 0 || "${S1}" != 0 ]]; then
  for pair in "worker0:${S0}:${WORKER0_LOG}" "worker1:${S1}:${WORKER1_LOG}"; do
    IFS=: read -r worker status log <<<"${pair}"
    [[ "${status}" == 0 ]] && continue
    echo >&2
    echo "----- ${worker} failure log tail (${log}) -----" >&2
    if [[ -f "${log}" ]]; then
      tail -n "${LOG_TAIL_LINES}" "${log}" >&2 || true
    else
      echo "log file does not exist" >&2
    fi
    echo "----- end ${worker} failure log -----" >&2
  done
  echo >&2
  echo "ERROR: train_safe workers failed: worker0=${S0}, worker1=${S1}" >&2
  if grep -Eq "different semantic build config|Legacy partial dataset has no resume contract|Output already contains dataset files" "${WORKER0_LOG}" "${WORKER1_LOG}" 2>/dev/null; then
    echo "Detected a stale/incompatible hidden shard under ${SHARD_ROOT}." >&2
    echo "For a complete replacement run: RESET_OUTPUT=1 RESUME=1 bash scripts/rebuild_ocrap_train_safe_two_gpu.sh" >&2
    echo "For a genuine continuation, keep RESET_OUTPUT=0 and use the exact original semantic parameters." >&2
  else
    echo "Inspect the printed traceback above. Typical causes are WOMD path/shard mismatch," >&2
    echo "Waymax/TensorFlow API import failure, CUDA/JAX failure, or GPU memory exhaustion." >&2
  fi
  exit 3
fi

"${PYTHON_BIN}" tools/merge_dataset_shards.py --replace-output --symlink --output "${OUTPUT}" \
  "${SHARD_ROOT}/worker0" "${SHARD_ROOT}/worker1"
"${PYTHON_BIN}" tools/cap_dataset_scene_time_groups.py --dataset "${OUTPUT}" \
  --min-samples "${MIN_SAMPLES}" --max-samples "${MAX_SAMPLES}" --apply

"${PYTHON_BIN}" -m ocrap.cli diagnose --dataset "${OUTPUT}" \
  --set dataset_quality.nominal_regime_dataset=true \
  --set 'dataset_quality.require_nominal_regimes=[normal]' \
  --output "${OCRAP_ROOT}/reports/${TRAIN_SAFE_NAME}.json"

"${PYTHON_BIN}" - "${OUTPUT}" "${MIN_SAMPLES}" "${MAX_SAMPLES}" <<'PY'
import csv, sys
from collections import defaultdict
from pathlib import Path
root=Path(sys.argv[1]); lo=int(sys.argv[2]); hi=int(sys.argv[3])
rows=list(csv.DictReader((root/'manifest.csv').open(newline='',encoding='utf-8')))
assert lo <= len(rows) <= hi, (len(rows),lo,hi)
groups=defaultdict(list)
for r in rows:
    groups[(r.get('original_scenario_id') or r.get('scene_id'),r.get('time_index'))].append(r)
for key, rs in groups.items():
    nominal=[r for r in rs if r.get('is_nominal')=='1']
    assert len(nominal)==1, (key,len(nominal))
    labels={x for x in nominal[0].get('regime_label','').split(';') if x}
    assert 'normal' in labels, (key,labels)
forbidden={'near_contact','post_contact','oracle_artifact','prefix_collision','prefix_contact'}
bad=[r for r in rows if forbidden & {x for x in r.get('regime_label','').split(';') if x}]
assert not bad, len(bad)
print({'audit':'passed','samples':len(rows),'groups':len(groups),'scenes':len({k[0] for k in groups})})
PY

echo "strict train_safe ready: ${OUTPUT}"
echo "diagnostics: ${OCRAP_ROOT}/reports/${TRAIN_SAFE_NAME}.json"
