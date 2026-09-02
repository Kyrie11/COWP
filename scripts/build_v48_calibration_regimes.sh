#!/usr/bin/env bash
set -euo pipefail

# Dedicated OC-RAP calibration construction.
# Source: standard WOMD validation only.
# The reserved start index is beyond the maximum raw index used by the supplied
# synchronized val/test builder defaults; scene filtering and overlap audit are
# still applied as a second line of protection.

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
WOMD_VAL="${WOMD_VAL:-$WOMD_ROOT/validation/validation_tfexample.tfrecord@150}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data0/senzeyu2/dataset/OCRAP_v48_calibration}"
EVAL_OCRAP_ROOT="${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
RESUME="${RESUME:-1}"
CALIBRATION_START_INDEX="${CALIBRATION_START_INDEX:-11000}"
PARTITION_STRIDE="${PARTITION_STRIDE:-6}"
SAFE_RAW_PER_WORKER="${SAFE_RAW_PER_WORKER:-600}"
NEAR_RAW_PER_WORKER="${NEAR_RAW_PER_WORKER:-700}"
CONTACT_RAW_PER_WORKER="${CONTACT_RAW_PER_WORKER:-700}"
MIN_CAL_SAFE_SCENES="${MIN_CAL_SAFE_SCENES:-80}"
MIN_CAL_NEAR_SCENES="${MIN_CAL_NEAR_SCENES:-120}"
MIN_CAL_CONTACT_SCENES="${MIN_CAL_CONTACT_SCENES:-120}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
REQUIRE_JAX_GPU="${REQUIRE_JAX_GPU:-1}"
START_STAGE="${START_STAGE:-safe}"  # safe | near | contact | merge
case "$START_STAGE" in safe|near|contact|merge) ;; *) echo "invalid START_STAGE=$START_STAGE" >&2; exit 2 ;; esac
stage_rank(){ case "$1" in safe) echo 0;; near) echo 1;; contact) echo 2;; merge) echo 3;; esac; }
SHOULD_START_RANK="$(stage_rank "$START_STAGE")"
should_run(){ [[ "$(stage_rank "$1")" -ge "$SHOULD_START_RANK" ]]; }
LOG_DIR="$OUTPUT_ROOT/logs"
SHARD_DIR="$OUTPUT_ROOT/shards"
mkdir -p "$LOG_DIR" "$SHARD_DIR"
STATUS_FILE="$OUTPUT_ROOT/calibration_build_status.json"
CURRENT_STAGE="preflight"
write_status(){
  local state="$1" stage="${2:-$CURRENT_STAGE}" detail="${3:-}"
  "$PYTHON_BIN" - "$STATUS_FILE" "$state" "$stage" "$detail" <<'PY_STATUS'
import json, os, sys, tempfile
from datetime import datetime, timezone
path, state, stage, detail = sys.argv[1:]
payload = {"state": state, "stage": stage, "detail": detail,
           "updated_at_utc": datetime.now(timezone.utc).isoformat()}
os.makedirs(os.path.dirname(path), exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix='.calibration-status-', dir=os.path.dirname(path), text=True)
with os.fdopen(fd, 'w', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2); f.write('\n')
os.replace(tmp, path)
PY_STATUS
}
trap 'rc=$?; write_status failed "$CURRENT_STAGE" "line=$LINENO exit=$rc"; exit $rc' ERR
write_status running "$CURRENT_STAGE" "controller started"

run_ocrap(){ "$PYTHON_BIN" -m ocrap.cli "$@"; }
die(){ write_status failed "$CURRENT_STAGE" "$*"; echo "ERROR: $*" >&2; exit 1; }
if command -v flock >/dev/null 2>&1; then
  exec 9>"$OUTPUT_ROOT/.calibration_build.lock"
  flock -n 9 || die "another calibration controller already holds $OUTPUT_ROOT/.calibration_build.lock"
fi
resume_args=(); [[ "$RESUME" == 1 ]] && resume_args+=(--resume)

[[ -d "$WOMD_ROOT/validation" ]] || die "missing standard validation directory: $WOMD_ROOT/validation"
VAL_SHARDS=$(find "$WOMD_ROOT/validation" -maxdepth 1 -name 'validation_tfexample.tfrecord-*-of-00150' | wc -l)
[[ "$VAL_SHARDS" -eq 150 ]] || die "expected 150 standard validation shards, found $VAL_SHARDS"
[[ "$GPU0" != "$GPU1" ]] || die "GPU0 and GPU1 must differ"
if [[ "$REQUIRE_JAX_GPU" == 1 ]]; then
  for gpu in "$GPU0" "$GPU1"; do
    CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false "$PYTHON_BIN" - <<'PY'
import jax
x=[str(d) for d in jax.devices()]
print({'jax_devices':x})
if not any(getattr(d,'platform','') == 'gpu' for d in jax.devices()):
    raise SystemExit('JAX cannot see a GPU')
PY
  done
fi

COMMON=(
  --set data_source=womd
  --set simulation_backend=waymax_closed_loop
  --set womd_patterns="$WOMD_VAL"
  --set scenario_start_index="$CALIBRATION_START_INDEX"
  --set scenario_stride="$PARTITION_STRIDE"
  --set split.force_id=calibration
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
)

build_safe(){ local worker="$1" out="$2" gpu="$3";
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false run_ocrap build-dataset "${resume_args[@]}" "${COMMON[@]}" \
    --set scenario_worker_index="$worker" --set max_scenarios="$SAFE_RAW_PER_WORKER" \
    --set max_times_per_scenario=3 --set max_biased_times_per_scenario=0 \
    --set dataset_quality.min_uniform_times_per_scenario=3 \
    --set num_targeted_futures=0 --set waymax.compute_future_metrics=false \
    --set waymax.teacher_rollout_top_k_options=4 \
    --set waymax.enable_augmented_hidden_roots=false --set waymax.enable_visible_perturbation_roots=false \
    --set artifact.force_mine=false --set artifact.mine_probability=0.0 --set artifact.use_margin_override=false \
    --set dataset_quality.balanced_two_pass=false --set dataset_quality.artifact_pair_mode=tag \
    --set dataset_quality.nominal_regime_dataset=true --set dataset_quality.keep_nominal_even_if_quality_fails=false \
    --set dataset_quality.drop_scene_time_if_under_min_quality=true \
    --set dataset_quality.max_accepted_prefixes_per_scene_time=8 \
    --set 'dataset_quality.require_nominal_regimes=[normal]' \
    --set 'dataset_quality.forbid_nominal_regimes=[near_contact,post_contact,oracle_artifact,prefix_collision,prefix_contact]' \
    --set 'dataset_quality.forbid_any_regimes=[near_contact,post_contact,oracle_artifact,prefix_collision,prefix_contact]' \
    --set regime_thresholds.tau_occ=0.75 --set regime_thresholds.tau_normal_occ=0.90 \
    --set regime_thresholds.require_uniform_for_normal=true --output "$out"; }

build_near(){ local worker="$1" out="$2" gpu="$3";
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false run_ocrap build-dataset "${resume_args[@]}" "${COMMON[@]}" \
    --set scenario_worker_index="$worker" --set max_scenarios="$NEAR_RAW_PER_WORKER" \
    --set max_times_per_scenario=3 --set max_biased_times_per_scenario=3 \
    --set num_targeted_futures=8 \
    --set 'targeted_future_kinds=[hidden_vehicle_yields,hidden_vehicle_accelerates,low_friction_braking,control_delay_noise]' \
    --set waymax.compute_future_metrics=true --set waymax.teacher_rollout_top_k_options=0 \
    --set waymax.enable_augmented_hidden_roots=true --set waymax.enable_visible_perturbation_roots=true \
    --set artifact.force_mine=true --set artifact.mine_probability=0.30 --set artifact.use_margin_override=false \
    --set artifact.enable_branch_intent_margin=true --set artifact.branch_intent_compatible_margin=1.0 \
    --set artifact.branch_intent_incompatible_margin=-2.5 \
    --set dataset_quality.balanced_two_pass=true --set dataset_quality.balanced_rotate_prefix_order=true \
    --set dataset_quality.artifact_pair_mode=balanced --set dataset_quality.artifact_quota_uses_label=true \
    --set dataset_quality.require_artifact_pairs=true --set dataset_quality.max_accepted_prefixes_per_scene_time=8 \
    --set dataset_quality.min_artifact_prefixes_per_scene_time=1 --set dataset_quality.max_artifact_prefixes_per_scene_time=2 \
    --set dataset_quality.min_nonartifact_prefixes_per_scene_time=4 --set dataset_quality.max_nonartifact_prefixes_per_scene_time=6 \
    --set dataset_quality.max_artifact_attempts_per_scene_time=24 --set dataset_quality.max_nonartifact_attempts_per_scene_time=12 \
    --set 'dataset_quality.require_nominal_regimes=[near_contact]' --set 'dataset_quality.require_any_regimes=[oracle_artifact]' \
    --set 'dataset_quality.forbid_nominal_regimes=[post_contact,prefix_collision,prefix_contact]' \
    --set 'dataset_quality.forbid_any_regimes=[post_contact,prefix_collision,prefix_contact]' \
    --set dataset_quality.artifact_pass_use_margin_override=true --set dataset_quality.artifact_pass_skip_augmented_waymax=true \
    --set dataset_quality.artifact_pass_apply_override_to_screened=true --set dataset_quality.artifact_pass_compute_future_metrics=false \
    --output "$out"; }

build_contact(){ local worker="$1" out="$2" gpu="$3";
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false run_ocrap build-dataset "${resume_args[@]}" "${COMMON[@]}" \
    --set scenario_worker_index="$worker" --set max_scenarios="$CONTACT_RAW_PER_WORKER" \
    --set max_times_per_scenario=4 --set max_biased_times_per_scenario=4 \
    --set num_targeted_futures=10 \
    --set 'targeted_future_kinds=[hidden_vehicle_yields,hidden_vehicle_accelerates,contact_impulse_surrogate,secondary_collision_approach,low_friction_braking,control_delay_noise]' \
    --set waymax.compute_future_metrics=true --set waymax.teacher_rollout_top_k_options=0 \
    --set waymax.enable_augmented_hidden_roots=true --set waymax.enable_visible_perturbation_roots=true \
    --set artifact.force_mine=true --set artifact.mine_probability=0.25 --set artifact.use_margin_override=false \
    --set artifact.enable_branch_intent_margin=true --set artifact.branch_intent_compatible_margin=1.0 \
    --set artifact.branch_intent_incompatible_margin=-2.5 \
    --set dataset_quality.balanced_two_pass=true --set dataset_quality.balanced_rotate_prefix_order=true \
    --set dataset_quality.artifact_pair_mode=balanced --set dataset_quality.artifact_quota_uses_label=true \
    --set dataset_quality.require_artifact_pairs=true --set dataset_quality.max_accepted_prefixes_per_scene_time=9 \
    --set dataset_quality.min_artifact_prefixes_per_scene_time=1 --set dataset_quality.max_artifact_prefixes_per_scene_time=2 \
    --set dataset_quality.min_nonartifact_prefixes_per_scene_time=5 --set dataset_quality.max_nonartifact_prefixes_per_scene_time=7 \
    --set dataset_quality.max_artifact_attempts_per_scene_time=24 --set dataset_quality.max_nonartifact_attempts_per_scene_time=12 \
    --set 'dataset_quality.require_nominal_regimes=[post_contact_counterfactual]' --set 'dataset_quality.require_any_regimes=[oracle_artifact]' \
    --set 'dataset_quality.forbid_nominal_regimes=[post_contact_observed,prefix_collision,prefix_contact]' \
    --set 'dataset_quality.forbid_any_regimes=[post_contact_observed,prefix_collision,prefix_contact]' \
    --set dataset_quality.artifact_pass_use_margin_override=true --set dataset_quality.artifact_pass_skip_augmented_waymax=true \
    --set dataset_quality.artifact_pass_apply_override_to_screened=true --set dataset_quality.artifact_pass_compute_future_metrics=false \
    --output "$out"; }

wait_pair(){ local p0="$1" p1="$2" n0="$3" n1="$4" s0=0 s1=0; set +e; wait "$p0"; s0=$?; wait "$p1"; s1=$?; set -e;
  [[ "$s0" == 0 && "$s1" == 0 ]] || die "calibration workers failed: $n0=$s0 $n1=$s1"; }

if should_run safe; then
  CURRENT_STAGE="build_safe"; write_status running "$CURRENT_STAGE" "workers 0,1"
  build_safe 0 "$SHARD_DIR/calibration_safe_w0" "$GPU0" >"$LOG_DIR/calibration_safe_w0.log" 2>&1 & P0=$!
  build_safe 1 "$SHARD_DIR/calibration_safe_w1" "$GPU1" >"$LOG_DIR/calibration_safe_w1.log" 2>&1 & P1=$!
  wait_pair "$P0" "$P1" safe_w0 safe_w1
fi
if should_run near; then
  CURRENT_STAGE="build_near_contact"; write_status running "$CURRENT_STAGE" "workers 2,3; contact logs are not expected yet"
  build_near 2 "$SHARD_DIR/calibration_near_w2" "$GPU0" >"$LOG_DIR/calibration_near_w2.log" 2>&1 & P0=$!
  build_near 3 "$SHARD_DIR/calibration_near_w3" "$GPU1" >"$LOG_DIR/calibration_near_w3.log" 2>&1 & P1=$!
  wait_pair "$P0" "$P1" near_w2 near_w3
fi
if should_run contact; then
  CURRENT_STAGE="build_contact"; write_status running "$CURRENT_STAGE" "workers 4,5"
  build_contact 4 "$SHARD_DIR/calibration_contact_w4" "$GPU0" >"$LOG_DIR/calibration_contact_w4.log" 2>&1 & P0=$!
  build_contact 5 "$SHARD_DIR/calibration_contact_w5" "$GPU1" >"$LOG_DIR/calibration_contact_w5.log" 2>&1 & P1=$!
  wait_pair "$P0" "$P1" contact_w4 contact_w5
fi

CURRENT_STAGE="merge_filter_audit"; write_status running "$CURRENT_STAGE" "merging and enforcing scene disjointness"
for required in calibration_safe_w0 calibration_safe_w1 calibration_near_w2 calibration_near_w3 calibration_contact_w4 calibration_contact_w5; do
  [[ -f "$SHARD_DIR/$required/manifest.csv" ]] || die "missing completed shard manifest: $SHARD_DIR/$required/manifest.csv"
done
for d in raw_calibration_safe raw_calibration_near_contact raw_calibration_contact calibration_safe calibration_near_contact calibration_contact; do
  rm -rf "$OUTPUT_ROOT/$d"
done
"$PYTHON_BIN" tools/merge_dataset_roots.py --hardlink --output "$OUTPUT_ROOT/raw_calibration_safe" \
  "$SHARD_DIR/calibration_safe_w0" "$SHARD_DIR/calibration_safe_w1" | tee "$LOG_DIR/merge_safe.log"
"$PYTHON_BIN" tools/merge_dataset_roots.py --hardlink --output "$OUTPUT_ROOT/raw_calibration_near_contact" \
  "$SHARD_DIR/calibration_near_w2" "$SHARD_DIR/calibration_near_w3" | tee "$LOG_DIR/merge_near.log"
"$PYTHON_BIN" tools/merge_dataset_roots.py --hardlink --output "$OUTPUT_ROOT/raw_calibration_contact" \
  "$SHARD_DIR/calibration_contact_w4" "$SHARD_DIR/calibration_contact_w5" | tee "$LOG_DIR/merge_contact.log"

exclude_args=()
for d in val_safe test_safe val_near_contact test_near_contact val_contact test_contact; do
  [[ -d "$EVAL_OCRAP_ROOT/$d" ]] && exclude_args+=(--exclude-root "$EVAL_OCRAP_ROOT/$d")
done
for spec in \
  "raw_calibration_safe calibration_safe" \
  "raw_calibration_near_contact calibration_near_contact" \
  "raw_calibration_contact calibration_contact"; do
  read -r raw final <<<"$spec"
  "$PYTHON_BIN" tools/filter_dataset_scenes_v48.py --overwrite --input "$OUTPUT_ROOT/$raw" --output "$OUTPUT_ROOT/$final" \
    "${exclude_args[@]}" | tee "$LOG_DIR/filter_${final}.log"
done

audit_existing=()
for d in val_safe val_near_contact val_contact test_safe test_near_contact test_contact; do
  [[ -d "$EVAL_OCRAP_ROOT/$d" ]] && audit_existing+=(--development-root "$EVAL_OCRAP_ROOT/$d")
done
python tools/check_scene_overlap_v48.py \
  "${audit_existing[@]}" \
  --test-root "$OUTPUT_ROOT/calibration_safe" --test-root "$OUTPUT_ROOT/calibration_near_contact" --test-root "$OUTPUT_ROOT/calibration_contact" \
  --output "$OUTPUT_ROOT/calibration_overlap_audit.json" --fail-on-development-test-overlap \
  2>&1 | tee "$LOG_DIR/calibration_overlap_audit.log"

if [[ "$RUN_DIAGNOSTICS" == 1 ]]; then
  run_ocrap diagnose --dataset "$OUTPUT_ROOT/calibration_safe" --set dataset_quality.nominal_regime_dataset=true \
    --set 'dataset_quality.require_nominal_regimes=[normal]' --output "$OUTPUT_ROOT/diagnose_calibration_safe.json" | tee "$LOG_DIR/diagnose_safe.log"
  run_ocrap diagnose --dataset "$OUTPUT_ROOT/calibration_near_contact" --output "$OUTPUT_ROOT/diagnose_calibration_near_contact.json" | tee "$LOG_DIR/diagnose_near.log"
  run_ocrap diagnose --dataset "$OUTPUT_ROOT/calibration_contact" --output "$OUTPUT_ROOT/diagnose_calibration_contact.json" | tee "$LOG_DIR/diagnose_contact.log"
fi

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$MIN_CAL_SAFE_SCENES" "$MIN_CAL_NEAR_SCENES" "$MIN_CAL_CONTACT_SCENES" "$CALIBRATION_START_INDEX" "$PARTITION_STRIDE" <<'PY'
import csv, json, sys
from pathlib import Path
root=Path(sys.argv[1]); mins={'calibration_safe':int(sys.argv[2]),'calibration_near_contact':int(sys.argv[3]),'calibration_contact':int(sys.argv[4])}
counts={}
for name, minimum in mins.items():
    with (root/name/'manifest.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    scenes={str(r.get('original_scenario_id') or r.get('scene_id') or '') for r in rows}
    scenes.discard(''); counts[name]={'samples':len(rows),'scenes':len(scenes),'minimum_scenes':minimum}
    if len(scenes) < minimum: raise SystemExit(f'{name}: {len(scenes)} scenes < {minimum}; increase *_RAW_PER_WORKER')
(root/'calibration_build_contract.json').write_text(json.dumps({
    'source':'WOMD standard validation','scenario_start_index':int(sys.argv[5]),'scenario_stride':int(sys.argv[6]),
    'worker_assignment':{'safe':[0,1],'near':[2,3],'contact':[4,5]},
    'excluded_existing_roots':True,'counts':counts},ensure_ascii=False,indent=2)+'\n')
print({'event':'dedicated_calibration_complete','counts':counts})
PY

CURRENT_STAGE="complete"; write_status complete "$CURRENT_STAGE" "dedicated calibration roots complete"
trap - ERR
echo "dedicated calibration roots complete: $OUTPUT_ROOT"
