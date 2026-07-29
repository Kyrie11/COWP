#!/usr/bin/env bash
set -euo pipefail

# Paper-grade data rebuild for v16.8.  Uses a fresh root because priority labels,
# candidate map screening and candidate replay coverage changed relative to v9/v15.
PYTHON_BIN="${PYTHON_BIN:-python}"
WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v18}"
TRAIN_LIMIT="${TRAIN_LIMIT:-22000}"
VAL_LIMIT="${VAL_LIMIT:-5000}"
LABEL_WORKERS_TRAIN="${LABEL_WORKERS_TRAIN:-32}"
LABEL_WORKERS_VAL="${LABEL_WORKERS_VAL:-24}"
CACHE_WORKERS="${CACHE_WORKERS:-8}"
AUG_WORKERS_TRAIN="${AUG_WORKERS_TRAIN:-12}"
AUG_WORKERS_VAL="${AUG_WORKERS_VAL:-6}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_WAYMAX_REPLAY="${RUN_WAYMAX_REPLAY:-1}"
MAX_REPLAY_CANDIDATES="${MAX_REPLAY_CANDIDATES:-24}"
REPLAY_HORIZON="${REPLAY_HORIZON:-80}"

SCENARIO_TRAIN="$WOMD_ROOT/uncompressed/scenario/training/*.tfrecord*"
SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
TFEXAMPLE_TRAIN="$WOMD_ROOT/uncompressed/tf_example/training/*.tfrecord*"
TFEXAMPLE_VAL="$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*"
LABEL_CFG="configs/label_cowp_v16_8.yaml"
EVAL_CFG="configs/eval_cowp_v16_8.yaml"

INDEX_TRAIN="$COWP_ROOT/index_train.jsonl"
INDEX_VAL="$COWP_ROOT/index_val.jsonl"
LABELS_TRAIN="$COWP_ROOT/labels_train"
LABELS_VAL="$COWP_ROOT/labels_val"
BASE_TRAIN="$COWP_ROOT/tensor_cache_train"
BASE_VAL="$COWP_ROOT/tensor_cache_val"
RAW_TRAIN="$COWP_ROOT/tensor_cache_train_waymax"
RAW_VAL="$COWP_ROOT/tensor_cache_val_waymax"
TRANSPORT_TRAIN="$COWP_ROOT/tensor_cache_train_waymax_transport_v18"
TRANSPORT_VAL="$COWP_ROOT/tensor_cache_val_waymax_transport_v18"
REPLAY_DIR="$COWP_ROOT/waymax_replay_v18"
TRAIN_OUTCOMES="$REPLAY_DIR/train_cache_bal${MAX_REPLAY_CANDIDATES}_safety.jsonl"
VAL_OUTCOMES="$REPLAY_DIR/val_cache_bal${MAX_REPLAY_CANDIDATES}_safety.jsonl"

mkdir -p "$COWP_ROOT/logs" "$REPLAY_DIR"

run() {
  local name="$1"; shift
  echo "[$name] $*"
  "$@" > >(tee "$COWP_ROOT/logs/${name}.log") 2> >(tee -a "$COWP_ROOT/logs/${name}.log" >&2)
}

run index_train "$PYTHON_BIN" -m cowp.scripts.00_index_womd \
  --data-config configs/data.yaml --proto-glob "$SCENARIO_TRAIN" \
  --output "$INDEX_TRAIN" --cpu-only
run index_val "$PYTHON_BIN" -m cowp.scripts.00_index_womd \
  --data-config configs/data.yaml --proto-glob "$SCENARIO_VAL" \
  --output "$INDEX_VAL" --cpu-only

run labels_train "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --proto-glob "$SCENARIO_TRAIN" --output-dir "$LABELS_TRAIN" \
  --index-jsonl "$INDEX_TRAIN" --limit "$TRAIN_LIMIT" \
  --num-workers "$LABEL_WORKERS_TRAIN" --start-method forkserver \
  --max-pending-multiplier 2 --no-compress --skip-existing --skip-diagnostics --cpu-only
run labels_val "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --proto-glob "$SCENARIO_VAL" --output-dir "$LABELS_VAL" \
  --index-jsonl "$INDEX_VAL" --limit "$VAL_LIMIT" \
  --num-workers "$LABEL_WORKERS_VAL" --start-method forkserver \
  --max-pending-multiplier 2 --no-compress --skip-existing --skip-diagnostics --cpu-only

run diagnose_labels_train "$PYTHON_BIN" cowp/scripts/06_diagnose_dataset.py \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --labels-dir "$LABELS_TRAIN" --output-dir "$COWP_ROOT/diagnostics_train" \
  --make-visualizations --max-visualizations 64
run diagnose_labels_val "$PYTHON_BIN" cowp/scripts/06_diagnose_dataset.py \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --labels-dir "$LABELS_VAL" --output-dir "$COWP_ROOT/diagnostics_val" \
  --make-visualizations --max-visualizations 64

run tensor_train "$PYTHON_BIN" -m cowp.scripts.02_build_tensor_cache \
  --data-config configs/data.yaml --split training --tfexample-glob "$TFEXAMPLE_TRAIN" \
  --labels-dir "$LABELS_TRAIN" --output-dir "$BASE_TRAIN" \
  --num-workers "$CACHE_WORKERS" --start-method forkserver --parallel-scan \
  --require-waymax-ready --skip-existing --no-compress \
  --profile-jsonl "$COWP_ROOT/profile_tensor_cache_train.jsonl" --cpu-only
run tensor_val "$PYTHON_BIN" -m cowp.scripts.02_build_tensor_cache \
  --data-config configs/data.yaml --split validation --tfexample-glob "$TFEXAMPLE_VAL" \
  --labels-dir "$LABELS_VAL" --output-dir "$BASE_VAL" \
  --num-workers "$CACHE_WORKERS" --start-method forkserver --parallel-scan \
  --require-waymax-ready --skip-existing --no-compress \
  --profile-jsonl "$COWP_ROOT/profile_tensor_cache_val.jsonl" --cpu-only

if [[ "$RUN_WAYMAX_REPLAY" == "1" ]]; then
  run replay_train env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -u -m cowp.scripts.13_replay_waymax_candidates \
    --data-config configs/data.yaml --label-config "$LABEL_CFG" --eval-config "$EVAL_CFG" \
    --cache-dir "$BASE_TRAIN" --state-source cache --outcomes-jsonl "$TRAIN_OUTCOMES" \
    --candidate-selection balanced --max-candidates-per-scene "$MAX_REPLAY_CANDIDATES" \
    --rollout-horizon-steps "$REPLAY_HORIZON" --waymax-device gpu \
    --waymax-action-mode absolute_xy_yaw --metric-set safety
  run replay_val env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -u -m cowp.scripts.13_replay_waymax_candidates \
    --data-config configs/data.yaml --label-config "$LABEL_CFG" --eval-config "$EVAL_CFG" \
    --cache-dir "$BASE_VAL" --state-source cache --outcomes-jsonl "$VAL_OUTCOMES" \
    --candidate-selection balanced --max-candidates-per-scene "$MAX_REPLAY_CANDIDATES" \
    --rollout-horizon-steps "$REPLAY_HORIZON" --waymax-device gpu \
    --waymax-action-mode absolute_xy_yaw --metric-set safety
fi

[[ -s "$TRAIN_OUTCOMES" ]] || { echo "missing $TRAIN_OUTCOMES" >&2; exit 2; }
[[ -s "$VAL_OUTCOMES" ]] || { echo "missing $VAL_OUTCOMES" >&2; exit 2; }
run attach_train "$PYTHON_BIN" -m cowp.scripts.12_attach_waymax_candidate_outcomes \
  --cache-dir "$BASE_TRAIN" --output-dir "$RAW_TRAIN" --outcomes-jsonl "$TRAIN_OUTCOMES"
run attach_val "$PYTHON_BIN" -m cowp.scripts.12_attach_waymax_candidate_outcomes \
  --cache-dir "$BASE_VAL" --output-dir "$RAW_VAL" --outcomes-jsonl "$VAL_OUTCOMES"
run verify_train "$PYTHON_BIN" -m cowp.scripts.14_verify_waymax_cache --cache-dir "$RAW_TRAIN"
run verify_val "$PYTHON_BIN" -m cowp.scripts.14_verify_waymax_cache --cache-dir "$RAW_VAL"

run augment_train "$PYTHON_BIN" -u -m cowp.scripts.26_augment_transport_labels \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --input-dir "$RAW_TRAIN" --output-dir "$TRANSPORT_TRAIN" \
  --num-workers "$AUG_WORKERS_TRAIN" --chunksize 2 --storage-mode overlay \
  --sidecar-subdir .transport_v18 --force
run augment_val "$PYTHON_BIN" -u -m cowp.scripts.26_augment_transport_labels \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --input-dir "$RAW_VAL" --output-dir "$TRANSPORT_VAL" \
  --num-workers "$AUG_WORKERS_VAL" --chunksize 2 --storage-mode overlay \
  --sidecar-subdir .transport_v18 --force

run align_train "$PYTHON_BIN" -u -m cowp.scripts.33_diagnose_cache_alignment \
  --raw-cache "$RAW_TRAIN" --transport-cache "$TRANSPORT_TRAIN" \
  --max-scenes 2000 --workers 8 --hash-mode sampled --output "$COWP_ROOT/cache_alignment_train.json"
run align_val "$PYTHON_BIN" -u -m cowp.scripts.33_diagnose_cache_alignment \
  --raw-cache "$RAW_VAL" --transport-cache "$TRANSPORT_VAL" \
  --max-scenes 2000 --workers 8 --hash-mode sampled --output "$COWP_ROOT/cache_alignment_val.json"

cat > "$COWP_ROOT/data_manifest_v18.json" <<JSON
{
  "schema_version": "cowp_v18_data_v1",
  "label_config": "$LABEL_CFG",
  "eval_config": "$EVAL_CFG",
  "train_limit": $TRAIN_LIMIT,
  "val_limit": $VAL_LIMIT,
  "max_replay_candidates": $MAX_REPLAY_CANDIDATES,
  "raw_train_cache": "$RAW_TRAIN",
  "raw_val_cache": "$RAW_VAL",
  "transport_train_cache": "$TRANSPORT_TRAIN",
  "transport_val_cache": "$TRANSPORT_VAL",
  "notes": "Fresh labels include explicit root-conditioned response search, transported OPR, corrected priority labels, and map-screened proposals."
}
JSON

echo "[v16.8 data] complete: $COWP_ROOT"
echo "TRAIN_CACHE=$TRANSPORT_TRAIN"
echo "VAL_CACHE=$TRANSPORT_VAL"
