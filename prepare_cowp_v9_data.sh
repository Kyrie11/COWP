#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
TFEXAMPLE_TRAIN="${TFEXAMPLE_TRAIN:-$WOMD_ROOT/uncompressed/tf_example/training/*.tfrecord*}"
TFEXAMPLE_VAL="${TFEXAMPLE_VAL:-$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*}"

LABELS_TRAIN_IN="${LABELS_TRAIN_IN:-$COWP_ROOT/labels_train}"
LABELS_VAL_IN="${LABELS_VAL_IN:-$COWP_ROOT/labels_val}"
LABELS_TRAIN_V9="${LABELS_TRAIN_V9:-$COWP_ROOT/labels_train_v9_transport}"
LABELS_VAL_V9="${LABELS_VAL_V9:-$COWP_ROOT/labels_val_v9_transport}"
CACHE_TRAIN_V9="${CACHE_TRAIN_V9:-$COWP_ROOT/tensor_cache_train_v9}"
CACHE_VAL_V9="${CACHE_VAL_V9:-$COWP_ROOT/tensor_cache_val_v9}"
CACHE_TRAIN_WAYMAX_V9="${CACHE_TRAIN_WAYMAX_V9:-$COWP_ROOT/tensor_cache_train_waymax_v9}"
CACHE_VAL_WAYMAX_V9="${CACHE_VAL_WAYMAX_V9:-$COWP_ROOT/tensor_cache_val_waymax_v9}"
TRAIN_OUTCOMES="${TRAIN_OUTCOMES:-outputs/waymax_replay/train_cache_bal12_safety.jsonl}"
VAL_OUTCOMES="${VAL_OUTCOMES:-outputs/waymax_replay/val_cache_bal12_safety.jsonl}"

AUGMENT_TRAIN_WORKERS="${AUGMENT_TRAIN_WORKERS:-16}"
AUGMENT_VAL_WORKERS="${AUGMENT_VAL_WORKERS:-8}"
CACHE_TRAIN_WORKERS="${CACHE_TRAIN_WORKERS:-6}"
CACHE_VAL_WORKERS="${CACHE_VAL_WORKERS:-2}"
PARALLEL_SPLITS="${PARALLEL_SPLITS:-1}"
FORCE="${FORCE:-0}"

mkdir -p "$COWP_ROOT" outputs/cowp_v9_data/logs outputs/cowp_v9_data/verification

run_logged() {
  local name="$1"; shift
  echo "[$name] $*"
  "$@" > >(tee "outputs/cowp_v9_data/logs/${name}.log") 2> >(tee -a "outputs/cowp_v9_data/logs/${name}.log" >&2)
}

augment_train() {
  run_logged augment_train "$PYTHON_BIN" -u -m cowp.scripts.26_augment_transport_labels \
    --data-config configs/data.yaml --label-config configs/label_cowp_v9.yaml \
    --input-dir "$LABELS_TRAIN_IN" --output-dir "$LABELS_TRAIN_V9" \
    --num-workers "$AUGMENT_TRAIN_WORKERS" --start-method forkserver \
    --skip-existing --profile-jsonl "$COWP_ROOT/profile_augment_transport_train_v9.jsonl"
}
augment_val() {
  run_logged augment_val "$PYTHON_BIN" -u -m cowp.scripts.26_augment_transport_labels \
    --data-config configs/data.yaml --label-config configs/label_cowp_v9.yaml \
    --input-dir "$LABELS_VAL_IN" --output-dir "$LABELS_VAL_V9" \
    --num-workers "$AUGMENT_VAL_WORKERS" --start-method forkserver \
    --skip-existing --profile-jsonl "$COWP_ROOT/profile_augment_transport_val_v9.jsonl"
}

if [[ "$PARALLEL_SPLITS" == "1" ]]; then
  augment_train & p0=$!
  augment_val & p1=$!
  wait "$p0"; wait "$p1"
else
  augment_train
  augment_val
fi

build_train_cache() {
  run_logged build_train_cache "$PYTHON_BIN" -u -m cowp.scripts.02_build_tensor_cache \
    --data-config configs/data.yaml --split training --tfexample-glob "$TFEXAMPLE_TRAIN" \
    --labels-dir "$LABELS_TRAIN_V9" --output-dir "$CACHE_TRAIN_V9" \
    --num-workers "$CACHE_TRAIN_WORKERS" --start-method forkserver --parallel-scan \
    --require-waymax-ready --skip-existing --no-compress \
    --profile-jsonl "$COWP_ROOT/profile_tensor_cache_train_v9.jsonl" --cpu-only
}
build_val_cache() {
  run_logged build_val_cache "$PYTHON_BIN" -u -m cowp.scripts.02_build_tensor_cache \
    --data-config configs/data.yaml --split validation --tfexample-glob "$TFEXAMPLE_VAL" \
    --labels-dir "$LABELS_VAL_V9" --output-dir "$CACHE_VAL_V9" \
    --num-workers "$CACHE_VAL_WORKERS" --start-method forkserver --parallel-scan \
    --require-waymax-ready --skip-existing --no-compress \
    --profile-jsonl "$COWP_ROOT/profile_tensor_cache_val_v9.jsonl" --cpu-only
}

# Train and validation TFRecord scans can run concurrently on a sufficiently fast disk.
# Set PARALLEL_SPLITS=0 if both jobs contend heavily for one HDD/NFS mount.
if [[ "$PARALLEL_SPLITS" == "1" ]]; then
  build_train_cache & p0=$!
  build_val_cache & p1=$!
  wait "$p0"; wait "$p1"
else
  build_train_cache
  build_val_cache
fi

if [[ ! -s "$TRAIN_OUTCOMES" || ! -s "$VAL_OUTCOMES" ]]; then
  echo "Existing Waymax outcome JSONL is missing. Re-run candidate replay before attach:" >&2
  echo "  train: $TRAIN_OUTCOMES" >&2
  echo "  val:   $VAL_OUTCOMES" >&2
  exit 3
fi

attach_train() {
  run_logged attach_train "$PYTHON_BIN" -u -m cowp.scripts.12_attach_waymax_candidate_outcomes \
    --cache-dir "$CACHE_TRAIN_V9" --output-dir "$CACHE_TRAIN_WAYMAX_V9" \
    --outcomes-jsonl "$TRAIN_OUTCOMES"
}
attach_val() {
  run_logged attach_val "$PYTHON_BIN" -u -m cowp.scripts.12_attach_waymax_candidate_outcomes \
    --cache-dir "$CACHE_VAL_V9" --output-dir "$CACHE_VAL_WAYMAX_V9" \
    --outcomes-jsonl "$VAL_OUTCOMES"
}
attach_train & p0=$!
attach_val & p1=$!
wait "$p0"; wait "$p1"

run_logged verify_train_transport "$PYTHON_BIN" -u -m cowp.scripts.27_verify_transport_cache \
  --cache-dir "$CACHE_TRAIN_WAYMAX_V9" --max-files 2048 \
  --output outputs/cowp_v9_data/verification/train_transport.json & p0=$!
run_logged verify_val_transport "$PYTHON_BIN" -u -m cowp.scripts.27_verify_transport_cache \
  --cache-dir "$CACHE_VAL_WAYMAX_V9" --max-files 2048 \
  --output outputs/cowp_v9_data/verification/val_transport.json & p1=$!
wait "$p0"; wait "$p1"

run_logged verify_train_waymax "$PYTHON_BIN" -u -m cowp.scripts.14_verify_waymax_cache \
  --cache-dir "$CACHE_TRAIN_WAYMAX_V9" & p0=$!
run_logged verify_val_waymax "$PYTHON_BIN" -u -m cowp.scripts.14_verify_waymax_cache \
  --cache-dir "$CACHE_VAL_WAYMAX_V9" & p1=$!
wait "$p0"; wait "$p1"

echo "[cowp_v9_data] complete"
echo "TRAIN_CACHE=$CACHE_TRAIN_WAYMAX_V9"
echo "VAL_CACHE=$CACHE_VAL_WAYMAX_V9"
