#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_4_pipeline_v9labels_seed2026}"
RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax}"
RAW_VAL_CACHE="${RAW_VAL_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax}"
TRAIN_CACHE="${TRAIN_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax_transport_v9}"
VAL_CACHE="${VAL_CACHE:-/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax_transport_v9}"
WAYMAX_VAL="${WAYMAX_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
TRAIN_NPROC="${TRAIN_NPROC:-2}"
TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-$GPU0,$GPU1}"

RUN_AUGMENT="${RUN_AUGMENT:-0}"
RUN_DIAGNOSE="${RUN_DIAGNOSE:-0}"
DIAG_PROFILE="${DIAG_PROFILE:-fast}"  # fast|full
RUN_NATURAL="${RUN_NATURAL:-1}"
RUN_TRANSPORT="${RUN_TRANSPORT:-1}"
RUN_PLANNER="${RUN_PLANNER:-1}"
RUN_OFFLINE="${RUN_OFFLINE:-1}"
RUN_PROBE="${RUN_PROBE:-1}"
RUN_PARETO_ABLATION="${RUN_PARETO_ABLATION:-1}"
RUN_PAIRMAX_ABLATION="${RUN_PAIRMAX_ABLATION:-1}"
RUN_SEMANTIC_PROBE="${RUN_SEMANTIC_PROBE:-0}"
RUN_FULL="${RUN_FULL:-0}"
FORCE_AUGMENT="${FORCE_AUGMENT:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
ALLOW_QUALITY_GATE_FAILURE="${ALLOW_QUALITY_GATE_FAILURE:-0}"  # engineering smoke only; never for paper claims
DETACH="${DETACH:-0}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-none}"  # none|diagnose|natural|transport|planner|offline|probe

AUG_TRAIN_WORKERS="${AUG_TRAIN_WORKERS:-12}"
AUG_VAL_WORKERS="${AUG_VAL_WORKERS:-6}"
AUG_CHUNKSIZE="${AUG_CHUNKSIZE:-2}"
AUG_STORAGE_MODE="${AUG_STORAGE_MODE:-overlay}"
AUG_SIDECAR_SUBDIR="${AUG_SIDECAR_SUBDIR:-.transport_v16}"
NATURAL_EPOCHS="${NATURAL_EPOCHS:-20}"
TRANSPORT_EPOCHS="${TRANSPORT_EPOCHS:-14}"
PLANNER_EPOCHS="${PLANNER_EPOCHS:-10}"
BATCH_PER_GPU="${BATCH_PER_GPU:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
# Natural batches are much lighter than transport/planner batches.  Use more
# readers and a deeper queue only for natural training; heavy stages retain the
# conservative memory settings that avoided earlier pin/prefetch failures.
NATURAL_NUM_WORKERS="${NATURAL_NUM_WORKERS:-8}"
NATURAL_PREFETCH_FACTOR="${NATURAL_PREFETCH_FACTOR:-2}"
# Validation creates a second worker pool per DDP rank.  Keep it intentionally
# smaller and non-persistent; this only changes input-pipeline resource usage.
NATURAL_VAL_NUM_WORKERS="${NATURAL_VAL_NUM_WORKERS:-2}"
NATURAL_VAL_PREFETCH_FACTOR="${NATURAL_VAL_PREFETCH_FACTOR:-1}"
TRANSPORT_NUM_WORKERS="${TRANSPORT_NUM_WORKERS:-$NUM_WORKERS}"
TRANSPORT_PREFETCH_FACTOR="${TRANSPORT_PREFETCH_FACTOR:-$PREFETCH_FACTOR}"
TRANSPORT_VAL_NUM_WORKERS="${TRANSPORT_VAL_NUM_WORKERS:-2}"
TRANSPORT_VAL_PREFETCH_FACTOR="${TRANSPORT_VAL_PREFETCH_FACTOR:-1}"
PLANNER_NUM_WORKERS="${PLANNER_NUM_WORKERS:-$NUM_WORKERS}"
PLANNER_PREFETCH_FACTOR="${PLANNER_PREFETCH_FACTOR:-$PREFETCH_FACTOR}"
PLANNER_VAL_NUM_WORKERS="${PLANNER_VAL_NUM_WORKERS:-2}"
PLANNER_VAL_PREFETCH_FACTOR="${PLANNER_VAL_PREFETCH_FACTOR:-1}"
TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_system}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-1}"
NATURAL_LR="${NATURAL_LR:-3.0e-5}"
TRANSPORT_LR="${TRANSPORT_LR:-1.5e-5}"
PLANNER_LR="${PLANNER_LR:-8e-6}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
TRAIN_SEED="${TRAIN_SEED:-2026}"
# Precision policy. Natural basis defaults to full fp32; downstream stages use
# bf16 autocast when supported.  Explicit fp16 is retained only for controlled
# diagnosis and will fail fast on non-finite outputs/skipped optimizer steps.
AMP_DTYPE="${AMP_DTYPE:-auto}"                 # auto|bfloat16|float16
NATURAL_AMP="${NATURAL_AMP:-0}"
TRANSPORT_AMP="${TRANSPORT_AMP:-1}"
PLANNER_AMP="${PLANNER_AMP:-1}"

PROBE_SCENARIOS="${PROBE_SCENARIOS:-100}"
FULL_SCENARIOS="${FULL_SCENARIOS:-1000}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-80}"
PAIR_WITNESS_THRESHOLD="${PAIR_WITNESS_THRESHOLD:-0.70}"
DEFAULT_BCOT_BUDGET="${BCOT_RISK_BUDGET:-0.35}"
BCOT_BUDGET_SWEEP="${BCOT_BUDGET_SWEEP:-0.05,0.08,0.10,0.12,0.15,0.18,0.22,0.26,0.30,0.35,0.40,0.45,0.50,0.60,0.70}"
NCF_GATE_MODE="${NCF_GATE_MODE:-priority}"
OFFLINE_OUTCOME_RISK_PENALTY="${OFFLINE_OUTCOME_RISK_PENALTY:-0.0}"
ONLINE_OUTCOME_RISK_PENALTY="${ONLINE_OUTCOME_RISK_PENALTY:-0.0}"
OUTCOME_RISK_THRESHOLD="${OUTCOME_RISK_THRESHOLD:-1.10}"
WAYMAX_ACTION_MODE="${WAYMAX_ACTION_MODE:-absolute_xy_yaw}"

INIT_CKPT="${INIT_CKPT:-outputs/cowp_v10_gct_probe100_seed2026/checkpoints/planner/cowp_planner_best.pt}"
NATURAL_CKPT="${NATURAL_CKPT:-}"
NATURAL_HISTORY="${NATURAL_HISTORY:-}"
TRANSPORT_CKPT="${TRANSPORT_CKPT:-}"
CKPT="${CKPT:-}"
REQUIRE_INIT_CKPT="${REQUIRE_INIT_CKPT:-1}"
DATA_PROTOCOL="${DATA_PROTOCOL:-v9_reuse}"
CACHE_REUSE_REPORT="${CACHE_REUSE_REPORT:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_ROOT"/{logs,configs,checkpoints/natural,checkpoints/transport,checkpoints/planner,eval/learned_offline,eval/probe,eval/waymax,jax_cache}

# Build the exact run configs in a temporary directory first.  Provenance is
# checked before canonical configs are overwritten, so accidentally reusing an
# OUT_ROOT after a code/config change cannot corrupt the old experiment record.
CONFIG_CANDIDATE_DIR="$(mktemp -d "$OUT_ROOT/configs/.candidate_v16_4.XXXXXX")"
cleanup_candidate_configs() { rm -rf "$CONFIG_CANDIDATE_DIR"; }
trap cleanup_candidate_configs EXIT
cp configs/model_cowp_v16.yaml "$CONFIG_CANDIDATE_DIR/model_cowp_v16.yaml"
cp configs/label_cowp_v16.yaml "$CONFIG_CANDIDATE_DIR/label_cowp_v16.yaml"
cp configs/label_cowp_v16_pareto_ablation.yaml "$CONFIG_CANDIDATE_DIR/label_cowp_v16_pareto_ablation.yaml"
cp configs/label_cowp_v16_pairmax_ablation.yaml "$CONFIG_CANDIDATE_DIR/label_cowp_v16_pairmax_ablation.yaml"
cp configs/train_cowp_v16.yaml "$CONFIG_CANDIDATE_DIR/train_cowp_v16.yaml"
cp configs/eval_cowp_v16.yaml "$CONFIG_CANDIDATE_DIR/eval_cowp_v16.yaml"
sed -i -E "0,/^  seed:/{s/^  seed:.*/  seed: ${TRAIN_SEED}/}" "$CONFIG_CANDIDATE_DIR/train_cowp_v16.yaml"

# Never mix checkpoints/evaluations produced by different code or config states
# under one OUT_ROOT. Logical file names keep the signature stable while the
# candidate configs live in a temporary directory.
"$PYTHON_BIN" -u -m cowp.scripts.42_write_run_provenance \
  --output "$OUT_ROOT/configs/run_provenance.json" --strict-existing \
  --data-protocol "$DATA_PROTOCOL" \
  --raw-train-cache "$RAW_TRAIN_CACHE" --raw-val-cache "$RAW_VAL_CACHE" \
  --train-cache "$TRAIN_CACHE" --val-cache "$VAL_CACHE" \
  --file "configs/model_cowp_v16.yaml=$CONFIG_CANDIDATE_DIR/model_cowp_v16.yaml" \
  --file "configs/label_cowp_v16.yaml=$CONFIG_CANDIDATE_DIR/label_cowp_v16.yaml" \
  --file "configs/train_cowp_v16.yaml=$CONFIG_CANDIDATE_DIR/train_cowp_v16.yaml" \
  --file "configs/eval_cowp_v16.yaml=$CONFIG_CANDIDATE_DIR/eval_cowp_v16.yaml" \
  --file "cowp/models/natural_decoder.py=cowp/models/natural_decoder.py" \
  --file "cowp/models/losses.py=cowp/models/losses.py" \
  --file "cowp/models/cowp_model.py=cowp/models/cowp_model.py" \
  --file "cowp/scripts/03_train.py=cowp/scripts/03_train.py" \
  --file "cowp/waymax_eval/policy_wrapper.py=cowp/waymax_eval/policy_wrapper.py" \
  --file "run_cowp_v16_4_dual_gpu.sh=$0" | tee "$OUT_ROOT/logs/run_provenance.log"

# Provenance passed: only now publish the canonical copied configs.
cp "$CONFIG_CANDIDATE_DIR/model_cowp_v16.yaml" "$OUT_ROOT/configs/model_cowp_v16.yaml"
cp "$CONFIG_CANDIDATE_DIR/label_cowp_v16.yaml" "$OUT_ROOT/configs/label_cowp_v16.yaml"
cp "$CONFIG_CANDIDATE_DIR/label_cowp_v16_pareto_ablation.yaml" "$OUT_ROOT/configs/label_cowp_v16_pareto_ablation.yaml"
cp "$CONFIG_CANDIDATE_DIR/label_cowp_v16_pairmax_ablation.yaml" "$OUT_ROOT/configs/label_cowp_v16_pairmax_ablation.yaml"
cp "$CONFIG_CANDIDATE_DIR/train_cowp_v16.yaml" "$OUT_ROOT/configs/train_cowp_v16.yaml"
cp "$CONFIG_CANDIDATE_DIR/eval_cowp_v16.yaml" "$OUT_ROOT/configs/eval_cowp_v16.yaml"
cleanup_candidate_configs
trap - EXIT

printf '{\n  "data_protocol": "%s",\n  "raw_train_cache": "%s",\n  "raw_val_cache": "%s",\n  "train_cache": "%s",\n  "val_cache": "%s"\n}\n' \
  "$DATA_PROTOCOL" "$RAW_TRAIN_CACHE" "$RAW_VAL_CACHE" "$TRAIN_CACHE" "$VAL_CACHE" \
  > "$OUT_ROOT/configs/data_protocol_manifest.json"
printf '{\n  "amp_dtype": "%s",\n  "natural_amp": %s,\n  "transport_amp": %s,\n  "planner_amp": %s\n}\n' \
  "$AMP_DTYPE" "$NATURAL_AMP" "$TRANSPORT_AMP" "$PLANNER_AMP" \
  > "$OUT_ROOT/configs/precision_manifest.json"
MODEL_CFG="$OUT_ROOT/configs/model_cowp_v16.yaml"
LABEL_CFG="$OUT_ROOT/configs/label_cowp_v16.yaml"
PARETO_CFG="$OUT_ROOT/configs/label_cowp_v16_pareto_ablation.yaml"
PAIRMAX_CFG="$OUT_ROOT/configs/label_cowp_v16_pairmax_ablation.yaml"
TRAIN_CFG="$OUT_ROOT/configs/train_cowp_v16.yaml"
EVAL_CFG="$OUT_ROOT/configs/eval_cowp_v16.yaml"

if [[ "$DETACH" == "1" && "${COWP_V16_4_DETACHED:-0}" != "1" ]]; then
  export COWP_V16_4_DETACHED=1
  nohup bash "$0" > "$OUT_ROOT/logs/driver.nohup.log" 2>&1 &
  echo "[cowp_v16.4] detached pid=$! log=$OUT_ROOT/logs/driver.nohup.log"
  exit 0
fi

logrun() {
  local name="$1"; shift
  local log="$OUT_ROOT/logs/${name}.log"
  echo "[$name] -> $log"
  "$@" > >(tee "$log") 2> >(tee -a "$log" >&2)
}

amp_args() {
  local enabled="$1"
  if [[ "$enabled" == "1" ]]; then
    printf '%s\n' --amp --amp-dtype "$AMP_DTYPE"
  fi
}

wait_all() {
  local rc=0 pid
  for pid in "$@"; do
    [[ -n "$pid" ]] || continue
    if ! wait "$pid"; then rc=1; fi
  done
  return "$rc"
}

quality_gate() {
  local name="$1"; shift
  if logrun "$name" "$@"; then
    return 0
  fi
  if [[ "$ALLOW_QUALITY_GATE_FAILURE" == "1" ]]; then
    echo "[engineering-only] quality gate '$name' failed; continuing to exercise downstream code paths." >&2
    printf '%s\n' "$name" >> "$OUT_ROOT/eval/QUALITY_GATES_BYPASSED.txt"
    return 0
  fi
  return 1
}

json_valid() {
  local p="$1"
  [[ "$FORCE_EVAL" != "1" && -s "$p" ]] || return 1
  "$PYTHON_BIN" - "$p" <<'PY' >/dev/null
import json, sys
json.load(open(sys.argv[1], encoding="utf-8"))
PY
}

cache_ready() {
  local d="$1"
  [[ -s "$d/transport_augmentation_summary.json" ]] || return 1
  "$PYTHON_BIN" - "$d/transport_augmentation_summary.json" <<'PY' >/dev/null
import json,sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
total=int(x.get("files_total",0))
if "files_completed" in x:
    done=int(x.get("files_completed",0))
else:
    done=int(x.get("files_written",0))+int(x.get("files_skipped",0))+int(x.get("files_skipped_sidecar",0))+int(x.get("files_skipped_materialized",0))
assert total > 0 and done == total and int(x.get("error_count",0)) == 0
assert bool(x.get("complete", True))
PY
}

if [[ "$RUN_AUGMENT" == "1" ]]; then
  pids=()
  force_aug_args=()
  [[ "$FORCE_AUGMENT" == "1" ]] && force_aug_args=(--force)
  if ! cache_ready "$TRAIN_CACHE" || [[ "$FORCE_AUGMENT" == "1" ]]; then
    mkdir -p "$TRAIN_CACHE"
    (
      env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$PYTHON_BIN" -u -m cowp.scripts.26_augment_transport_labels \
        --data-config configs/data.yaml --label-config "$LABEL_CFG" \
        --input-dir "$RAW_TRAIN_CACHE" --output-dir "$TRAIN_CACHE" \
        --num-workers "$AUG_TRAIN_WORKERS" --chunksize "$AUG_CHUNKSIZE" \
        --storage-mode "$AUG_STORAGE_MODE" --sidecar-subdir "$AUG_SIDECAR_SUBDIR" \
        "${force_aug_args[@]}"
    ) > >(tee "$OUT_ROOT/logs/augment_train.log") 2> >(tee -a "$OUT_ROOT/logs/augment_train.log" >&2) &
    pids+=("$!")
  fi
  if ! cache_ready "$VAL_CACHE" || [[ "$FORCE_AUGMENT" == "1" ]]; then
    mkdir -p "$VAL_CACHE"
    (
      env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$PYTHON_BIN" -u -m cowp.scripts.26_augment_transport_labels \
        --data-config configs/data.yaml --label-config "$LABEL_CFG" \
        --input-dir "$RAW_VAL_CACHE" --output-dir "$VAL_CACHE" \
        --num-workers "$AUG_VAL_WORKERS" --chunksize "$AUG_CHUNKSIZE" \
        --storage-mode "$AUG_STORAGE_MODE" --sidecar-subdir "$AUG_SIDECAR_SUBDIR" \
        "${force_aug_args[@]}"
    ) > >(tee "$OUT_ROOT/logs/augment_val.log") 2> >(tee -a "$OUT_ROOT/logs/augment_val.log" >&2) &
    pids+=("$!")
  fi
  for pid in "${pids[@]:-}"; do [[ -n "$pid" ]] && wait "$pid"; done
fi
NEED_TRANSPORT_CACHE=0
if [[ "$RUN_AUGMENT" == "1" || "$RUN_DIAGNOSE" == "1" || "$RUN_TRANSPORT" == "1" || "$RUN_PLANNER" == "1" || "$RUN_OFFLINE" == "1" || "$RUN_PROBE" == "1" || "$RUN_FULL" == "1" ]]; then
  NEED_TRANSPORT_CACHE=1
fi
if [[ "$NEED_TRANSPORT_CACHE" == "1" ]]; then
  cache_ready "$TRAIN_CACHE" || { echo "Transport train cache is not ready: $TRAIN_CACHE" >&2; exit 2; }
  cache_ready "$VAL_CACHE" || { echo "Transport val cache is not ready: $VAL_CACHE" >&2; exit 2; }
fi

if [[ "$RUN_DIAGNOSE" == "1" ]]; then
  case "$DIAG_PROFILE" in
    fast)
      transport_train_limit="${TRANSPORT_DIAG_TRAIN_SCENES:-2048}"
      transport_val_limit="${TRANSPORT_DIAG_VAL_SCENES:-1024}"
      alignment_scenes="${ALIGNMENT_DIAG_SCENES:-1024}"
      oracle_scenes="${ORACLE_DIAG_SCENES:-1024}"
      ;;
    full)
      transport_train_limit=0
      transport_val_limit=0
      alignment_scenes="${ALIGNMENT_DIAG_SCENES:-5000}"
      oracle_scenes="${ORACLE_DIAG_SCENES:-5000}"
      ;;
    *) echo "Unknown DIAG_PROFILE=$DIAG_PROFILE (expected fast|full)" >&2; exit 2 ;;
  esac

  transport_train_report="$OUT_ROOT/eval/transport_train_diagnostics.json"
  transport_val_report="$OUT_ROOT/eval/transport_val_diagnostics.json"
  diag_pids=()
  if ! json_valid "$transport_train_report"; then
    logrun diagnose_transport_train "$PYTHON_BIN" -u -m cowp.scripts.27_diagnose_transport_labels \
      --cache-dir "$TRAIN_CACHE" --limit "$transport_train_limit" \
      --workers "${TRANSPORT_DIAG_WORKERS:-8}" --output "$transport_train_report" & diag_pids+=("$!")
  fi
  if ! json_valid "$transport_val_report"; then
    logrun diagnose_transport_val "$PYTHON_BIN" -u -m cowp.scripts.27_diagnose_transport_labels \
      --cache-dir "$VAL_CACHE" --limit "$transport_val_limit" \
      --workers "${TRANSPORT_DIAG_WORKERS:-8}" --output "$transport_val_report" & diag_pids+=("$!")
  fi
  wait_all "${diag_pids[@]:-}"

  alignment_train_report="$OUT_ROOT/eval/cache_alignment_train.json"
  alignment_val_report="$OUT_ROOT/eval/cache_alignment_val.json"
  oracle_report="$OUT_ROOT/eval/natural_oracle_val.json"
  diag_pids=()
  if ! json_valid "$alignment_train_report"; then
    logrun diagnose_alignment_train "$PYTHON_BIN" -u -m cowp.scripts.33_diagnose_cache_alignment \
      --raw-cache "$RAW_TRAIN_CACHE" --transport-cache "$TRAIN_CACHE" \
      --max-scenes "$alignment_scenes" --workers "${DIAG_WORKERS:-8}" --hash-mode sampled --output "$alignment_train_report" & diag_pids+=("$!")
  fi
  if ! json_valid "$alignment_val_report"; then
    logrun diagnose_alignment_val "$PYTHON_BIN" -u -m cowp.scripts.33_diagnose_cache_alignment \
      --raw-cache "$RAW_VAL_CACHE" --transport-cache "$VAL_CACHE" \
      --max-scenes "$alignment_scenes" --workers "${DIAG_WORKERS:-8}" --hash-mode sampled --output "$alignment_val_report" & diag_pids+=("$!")
  fi
  if ! json_valid "$oracle_report"; then
    logrun diagnose_natural_oracle_val "$PYTHON_BIN" -u -m cowp.scripts.34_diagnose_natural_oracles \
      --cache-dir "$RAW_VAL_CACHE" --max-scenes "$oracle_scenes" --workers "${DIAG_WORKERS:-8}" \
      --output "$oracle_report" & diag_pids+=("$!")
  fi
  wait_all "${diag_pids[@]:-}"
fi
# Mandatory exact-path preflight: this follows the same dataset -> model anchor
# path as training and exits before any GPU-hours are spent when a 35 m constant
# translation/indexing bug is present.
MODEL_ANCHOR_REPORT="$OUT_ROOT/eval/model_anchor_preflight_val.json"
if [[ ! -s "$MODEL_ANCHOR_REPORT" || "$FORCE_EVAL" == "1" ]]; then
  logrun diagnose_model_anchor "$PYTHON_BIN" -u -m cowp.scripts.35_diagnose_model_anchor \
    --data-config configs/data.yaml --model-config "$MODEL_CFG" \
    --label-config "$LABEL_CFG" --train-config "$TRAIN_CFG" \
    --cache-dir "$RAW_VAL_CACHE" --max-scenes "${ANCHOR_DIAG_SCENES:-1024}" \
    --batch-size "${ANCHOR_DIAG_BATCH:-16}" --num-workers "${DIAG_WORKERS:-8}" \
    --output "$MODEL_ANCHOR_REPORT"
fi
CAUSAL_AUDIT_REPORT="$OUT_ROOT/eval/causal_protocol_audit.json"
ALIGN_VAL_REPORT="$OUT_ROOT/eval/cache_alignment_val.json"
audit_alignment_args=()
[[ -s "$ALIGN_VAL_REPORT" ]] && audit_alignment_args=(--cache-alignment-report "$ALIGN_VAL_REPORT")
audit_reuse_args=()
[[ -n "$CACHE_REUSE_REPORT" && -s "$CACHE_REUSE_REPORT" ]] && audit_reuse_args=(--cache-reuse-report "$CACHE_REUSE_REPORT")
logrun audit_causal_protocol "$PYTHON_BIN" -u -m cowp.scripts.36_audit_causal_protocol \
  --model-config "$MODEL_CFG" --label-config "$LABEL_CFG" \
  --train-config "$TRAIN_CFG" --eval-config "$EVAL_CFG" \
  --data-protocol "$DATA_PROTOCOL" \
  "${audit_alignment_args[@]}" "${audit_reuse_args[@]}" --output "$CAUSAL_AUDIT_REPORT"
if [[ "$STOP_AFTER_STAGE" == "diagnose" ]]; then
  echo "[cowp_v16.4] stopped after diagnostics: $OUT_ROOT"
  exit 0
fi

best_natural() {
  local p="$OUT_ROOT/checkpoints/natural/cowp_natural_best.pt"
  [[ -s "$p" ]] && { echo "$p"; return; }
  return 1
}
best_transport() {
  local p="$OUT_ROOT/checkpoints/transport/cowp_witness_best.pt"
  [[ -s "$p" ]] && { echo "$p"; return; }
  return 1
}
best_planner() {
  local p="$OUT_ROOT/checkpoints/planner/cowp_planner_best.pt"
  [[ -s "$p" ]] && { echo "$p"; return; }
  return 1
}

# Stage 0: repair the natural-option basis while preserving the initialized scene encoder.
# Root-indexed transport is not identifiable when the natural trajectories drift by
# tens of metres, so this stage is a hard prerequisite rather than an optional auxiliary.
if [[ "$RUN_NATURAL" == "1" && "$REQUIRE_INIT_CKPT" == "1" && ! -s "$INIT_CKPT" ]]; then
  echo "Required initialization checkpoint is missing: $INIT_CKPT" >&2
  exit 2
fi
natural_default_history="$OUT_ROOT/checkpoints/natural/history_natural.json"
natural_artifacts_complete() {
  local ckpt="$OUT_ROOT/checkpoints/natural/cowp_natural_best.pt"
  [[ -s "$ckpt" && -s "$natural_default_history" ]] || return 1
  "$PYTHON_BIN" - "$natural_default_history" <<'PY' >/dev/null
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert isinstance(x, list) and len(x) >= 2
PY
}
if [[ "$RUN_NATURAL" == "1" ]]; then
  if natural_artifacts_complete && [[ "$FORCE_TRAIN" != "1" ]]; then
    echo "[natural] keep complete checkpoint+history $(best_natural)"
  else
    if best_natural >/dev/null && [[ ! -s "$natural_default_history" ]]; then
      echo "[natural] checkpoint exists but history is missing; retraining instead of creating a false gate failure." >&2
    fi
    init_args=()
    [[ -s "$INIT_CKPT" ]] && init_args=(--resume "$INIT_CKPT")
    mapfile -t natural_amp_args < <(amp_args "$NATURAL_AMP")
    logrun train_natural_ddp env CUDA_VISIBLE_DEVICES="$TRAIN_VISIBLE_DEVICES" \
      "$TORCHRUN_BIN" --standalone --nproc_per_node="$TRAIN_NPROC" -m cowp.scripts.03_train \
      --data-config configs/data.yaml --model-config "$MODEL_CFG" \
      --label-config "$LABEL_CFG" --train-config "$TRAIN_CFG" \
      --cache-dir "$RAW_TRAIN_CACHE" --val-cache-dir "$RAW_VAL_CACHE" \
      --stage natural --epochs "$NATURAL_EPOCHS" --batch-size "$BATCH_PER_GPU" \
      --lr "$NATURAL_LR" --num-workers "$NATURAL_NUM_WORKERS" --prefetch-factor "$NATURAL_PREFETCH_FACTOR" \
      --val-num-workers "$NATURAL_VAL_NUM_WORKERS" --val-prefetch-factor "$NATURAL_VAL_PREFETCH_FACTOR" \
      --sharing-strategy "$TORCH_SHARING_STRATEGY" \
      --device cuda --output-dir "$OUT_ROOT/checkpoints/natural" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" --early-stop-min-delta 1e-4 \
      --lr-scheduler plateau --min-lr 2e-6 --save-every 2 \
      --no-positive-oversampling --eval-before-train \
      --reset-checkpoint-prefix natural_decoder --natural-graph-warmup-epochs 2 \
      --grad-clip 1.0 --fused-adamw "${natural_amp_args[@]}" "${init_args[@]}"
  fi
fi
if [[ -z "$NATURAL_CKPT" ]]; then NATURAL_CKPT="$(best_natural || true)"; fi
[[ -s "$NATURAL_CKPT" ]] || { echo "No natural-basis checkpoint. Set NATURAL_CKPT or enable RUN_NATURAL." >&2; exit 2; }
if [[ -z "$NATURAL_HISTORY" ]]; then NATURAL_HISTORY="$OUT_ROOT/checkpoints/natural/history_natural.json"; fi
[[ -s "$NATURAL_HISTORY" ]] || {
  echo "Natural-basis history is required for the hard gate: $NATURAL_HISTORY" >&2
  echo "When supplying an external NATURAL_CKPT, also set NATURAL_HISTORY to its history_natural.json." >&2
  exit 2
}
ORACLE_REPORT="$OUT_ROOT/eval/natural_oracle_val.json"
if [[ ! -s "$ORACLE_REPORT" ]]; then
  logrun diagnose_natural_oracle_val "$PYTHON_BIN" -u -m cowp.scripts.34_diagnose_natural_oracles \
    --cache-dir "$RAW_VAL_CACHE" --max-scenes "${ORACLE_DIAG_SCENES:-2000}" --workers "${DIAG_WORKERS:-8}" --output "$ORACLE_REPORT"
fi
quality_gate gate_natural_basis "$PYTHON_BIN" -u -m cowp.scripts.32_gate_natural_basis \
  --history "$NATURAL_HISTORY" --oracle-report "$ORACLE_REPORT" --max-oracle-gap-m 6.0 \
  --output "$OUT_ROOT/eval/learned_offline/natural_basis_gate.json" \
  --max-set-minade-m 8.5 --max-branch-minade-m 3.0 \
  --max-observed-minade-m 4.0 --max-branch-spread-m 3.0 \
  --max-neutral-minade-m 2.0 --max-priority-minade-m 2.0 \
  --max-priority-bce 0.45 --max-neutral-consistency-m 3.0 \
  --max-typed-untyped-gap-m 3.0
LEARNED_NATURAL_REPORT="$OUT_ROOT/eval/learned_offline/learned_natural_effectiveness.json"
logrun diagnose_learned_natural "$PYTHON_BIN" -u -m cowp.scripts.39_diagnose_learned_natural \
  --data-config configs/data.yaml --model-config "$MODEL_CFG" \
  --label-config "$LABEL_CFG" --train-config "$TRAIN_CFG" \
  --cache-dir "$RAW_VAL_CACHE" --checkpoint "$NATURAL_CKPT" \
  --max-scenes "${LEARNED_NATURAL_DIAG_SCENES:-2000}" \
  --batch-size "${LEARNED_NATURAL_DIAG_BATCH:-8}" --num-workers "${DIAG_WORKERS:-8}" \
  --device "${LEARNED_NATURAL_DIAG_DEVICE:-cuda}" --output "$LEARNED_NATURAL_REPORT"
quality_gate gate_natural_effectiveness "$PYTHON_BIN" -u -m cowp.scripts.40_gate_natural_effectiveness \
  --report "$LEARNED_NATURAL_REPORT" \
  --output "$OUT_ROOT/eval/learned_offline/natural_effectiveness_gate.json" \
  --max-learned-8s-m "${MAX_LEARNED_8S_M:-2.5}" \
  --max-obs-8s-m "${MAX_OBS_8S_M:-4.0}" \
  --min-overall-gain-8s-m "${MIN_OVERALL_GAIN_8S_M:-0.03}" \
  --min-obs-gain-8s-m "${MIN_OBS_GAIN_8S_M:-0.05}"
if [[ "$STOP_AFTER_STAGE" == "natural" ]]; then
  echo "[cowp_v16.4] stopped after natural gate: $OUT_ROOT"
  exit 0
fi

# Stage 1: learn explicit mode conflict/retention and same-root response recovery.
# Stage 0 has already validated NATURAL_CKPT.  Do not re-check INIT_CKPT here:
# users may legitimately skip natural training and supply a standalone repaired
# natural checkpoint.
if [[ "$RUN_TRANSPORT" == "1" ]]; then
  if best_transport >/dev/null && [[ "$FORCE_TRAIN" != "1" ]]; then
    echo "[transport] keep existing $(best_transport)"
  else
    resume_args=()
    [[ -s "$NATURAL_CKPT" ]] && resume_args=(--resume "$NATURAL_CKPT")
    mapfile -t transport_amp_args < <(amp_args "$TRANSPORT_AMP")
    logrun train_transport_ddp env CUDA_VISIBLE_DEVICES="$TRAIN_VISIBLE_DEVICES" \
      "$TORCHRUN_BIN" --standalone --nproc_per_node="$TRAIN_NPROC" -m cowp.scripts.03_train \
      --data-config configs/data.yaml --model-config "$MODEL_CFG" \
      --label-config "$LABEL_CFG" --train-config "$TRAIN_CFG" \
      --cache-dir "$TRAIN_CACHE" --val-cache-dir "$VAL_CACHE" \
      --stage witness --epochs "$TRANSPORT_EPOCHS" --batch-size "$BATCH_PER_GPU" \
      --lr "$TRANSPORT_LR" --num-workers "$TRANSPORT_NUM_WORKERS" --prefetch-factor "$TRANSPORT_PREFETCH_FACTOR" \
      --val-num-workers "$TRANSPORT_VAL_NUM_WORKERS" --val-prefetch-factor "$TRANSPORT_VAL_PREFETCH_FACTOR" \
      --sharing-strategy "$TORCH_SHARING_STRATEGY" \
      --device cuda --output-dir "$OUT_ROOT/checkpoints/transport" \
      --freeze-backbone-epochs "$FREEZE_BACKBONE_EPOCHS" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" --early-stop-min-delta 1e-4 \
      --lr-scheduler plateau --min-lr 2e-6 --save-every 2 \
      --no-positive-oversampling --no-response-traj --no-response-components \
      --fused-adamw "${transport_amp_args[@]}" "${resume_args[@]}"
  fi
fi
if [[ -z "$TRANSPORT_CKPT" ]]; then TRANSPORT_CKPT="$(best_transport || true)"; fi
[[ -s "$TRANSPORT_CKPT" ]] || { echo "No transport checkpoint. Set TRANSPORT_CKPT or enable RUN_TRANSPORT." >&2; exit 2; }
if [[ "$STOP_AFTER_STAGE" == "transport" ]]; then
  echo "[cowp_v16.4] stopped after transport: $OUT_ROOT"
  exit 0
fi

# Stage 2: train planner/candidate shields without rewriting the transport backbone.
if [[ "$RUN_PLANNER" == "1" ]]; then
  if best_planner >/dev/null && [[ "$FORCE_TRAIN" != "1" ]]; then
    echo "[planner] keep existing $(best_planner)"
  else
    mapfile -t planner_amp_args < <(amp_args "$PLANNER_AMP")
    logrun train_planner_ddp env CUDA_VISIBLE_DEVICES="$TRAIN_VISIBLE_DEVICES" \
      "$TORCHRUN_BIN" --standalone --nproc_per_node="$TRAIN_NPROC" -m cowp.scripts.03_train \
      --data-config configs/data.yaml --model-config "$MODEL_CFG" \
      --label-config "$LABEL_CFG" --train-config "$TRAIN_CFG" \
      --cache-dir "$TRAIN_CACHE" --val-cache-dir "$VAL_CACHE" \
      --stage planner --epochs "$PLANNER_EPOCHS" --batch-size "$BATCH_PER_GPU" \
      --lr "$PLANNER_LR" --num-workers "$PLANNER_NUM_WORKERS" --prefetch-factor "$PLANNER_PREFETCH_FACTOR" \
      --val-num-workers "$PLANNER_VAL_NUM_WORKERS" --val-prefetch-factor "$PLANNER_VAL_PREFETCH_FACTOR" \
      --sharing-strategy "$TORCH_SHARING_STRATEGY" \
      --device cuda --output-dir "$OUT_ROOT/checkpoints/planner" \
      --with-waymax-outcome-labels --freeze-backbone-epochs "$FREEZE_BACKBONE_EPOCHS" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" --early-stop-min-delta 1e-4 \
      --lr-scheduler plateau --min-lr 1e-6 --save-every 2 \
      --no-positive-oversampling --no-response-traj --no-response-components \
      --fused-adamw "${planner_amp_args[@]}" --resume "$TRANSPORT_CKPT"
  fi
fi
if [[ -z "$CKPT" ]]; then CKPT="$(best_planner || true)"; fi
[[ -s "$CKPT" ]] || { echo "No v15 planner checkpoint. Set CKPT or enable RUN_PLANNER." >&2; exit 2; }
echo "[checkpoint] $CKPT"
if [[ "$STOP_AFTER_STAGE" == "planner" ]]; then
  echo "[cowp_v16.4] stopped after planner: $OUT_ROOT"
  exit 0
fi

if [[ "$RUN_OFFLINE" == "1" ]]; then
  budget_sweep_out="$OUT_ROOT/eval/learned_offline/bcot_budget_sweep.json"
  if ! json_valid "$budget_sweep_out"; then
    logrun learned_offline_bcot_sweep env CUDA_VISIBLE_DEVICES="$GPU0" \
      "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
      --mode learned_offline --method cowp \
      --checkpoint "$CKPT" --cache-dir "$VAL_CACHE" \
      --data-config configs/data.yaml --label-config "$LABEL_CFG" --eval-config "$EVAL_CFG" \
      --device cuda --batch-size 20 --num-workers 4 --prefetch-factor 1 \
      --witness-threshold "$PAIR_WITNESS_THRESHOLD" \
      --bcot-risk-budget "$DEFAULT_BCOT_BUDGET" \
      --bcot-risk-budget-sweep "$BCOT_BUDGET_SWEEP" \
      --ncf-gate-mode "$NCF_GATE_MODE" --offline-fallback stop_like \
      --outcome-risk-penalty "$OFFLINE_OUTCOME_RISK_PENALTY" --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
      --output "$budget_sweep_out" --no-progress
  fi

  logrun calibrate_bcot "$PYTHON_BIN" -u -m cowp.scripts.31_calibrate_bcot_budget \
    --input "$budget_sweep_out" \
    --output "$OUT_ROOT/eval/learned_offline/bcot_calibration.json" \
    --min-ncf-recall 0.30 --min-accepted-rate 0.10 --max-fallback 0.25 \
    --max-selected-false-safe 0.50

  CALIBRATED_BCOT_BUDGET="$("$PYTHON_BIN" - "$OUT_ROOT/eval/learned_offline/bcot_calibration.json" <<'PY2'
import json,sys
print(json.load(open(sys.argv[1]))["bcot_risk_budget"])
PY2
)"
  echo "[calibrated BCOT budget] $CALIBRATED_BCOT_BUDGET"

  combined="$OUT_ROOT/eval/learned_offline/_shared_model_pass.json"
  pairmax_offline="$OUT_ROOT/eval/learned_offline/pairmax_ablation.json"
  offline_pids=()
  if ! json_valid "$combined"; then
    logrun learned_offline_methods env CUDA_VISIBLE_DEVICES="$GPU0" \
      "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
      --mode learned_offline \
      --methods planner_score_only,conventional_safety,soft_burden_cost_only,universal_ncf,cowp \
      --checkpoint "$CKPT" --cache-dir "$VAL_CACHE" \
      --data-config configs/data.yaml --label-config "$LABEL_CFG" --eval-config "$EVAL_CFG" \
      --device cuda --batch-size 20 --num-workers 4 --prefetch-factor 1 \
      --witness-threshold "$PAIR_WITNESS_THRESHOLD" --bcot-risk-budget "$CALIBRATED_BCOT_BUDGET" \
      --ncf-gate-mode "$NCF_GATE_MODE" --offline-fallback stop_like \
      --outcome-risk-penalty "$OFFLINE_OUTCOME_RISK_PENALTY" --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
      --output "$combined" --no-progress &
    offline_pids+=("$!")
  fi
  if [[ "$RUN_PAIRMAX_ABLATION" == "1" ]] && ! json_valid "$pairmax_offline"; then
    logrun learned_offline_pairmax env CUDA_VISIBLE_DEVICES="$GPU1" \
      "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
      --mode learned_offline --method cowp \
      --checkpoint "$CKPT" --cache-dir "$VAL_CACHE" \
      --data-config configs/data.yaml --label-config "$PAIRMAX_CFG" --eval-config "$EVAL_CFG" \
      --device cuda --batch-size 20 --num-workers 4 --prefetch-factor 1 \
      --witness-threshold "$PAIR_WITNESS_THRESHOLD" --bcot-risk-budget "$CALIBRATED_BCOT_BUDGET" \
      --ncf-gate-mode "$NCF_GATE_MODE" --offline-fallback stop_like \
      --outcome-risk-penalty "$OFFLINE_OUTCOME_RISK_PENALTY" --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
      --output "$pairmax_offline" --no-progress &
    offline_pids+=("$!")
  fi
  wait_all "${offline_pids[@]:-}"

  logrun diagnose_bcot_result "$PYTHON_BIN" -u -m cowp.scripts.30_diagnose_bcot_result \
    --input "$combined" --sweep-input "$budget_sweep_out" --method cowp \
    --calibration-json "$OUT_ROOT/eval/learned_offline/bcot_calibration.json" \
    --output "$OUT_ROOT/eval/learned_offline/bcot_readiness.json"
  quality_gate verify_mechanism "$PYTHON_BIN" -u -m cowp.scripts.25_verify_mechanism_effect \
    --input "$budget_sweep_out" --method cowp \
    --calibration-json "$OUT_ROOT/eval/learned_offline/bcot_calibration.json" \
    --min-unique-selection-points 3 --min-ncf-recall 0.30 \
    --min-witness-auprc 0.60 --min-bcot-auprc 0.65 --min-root-transport-auprc 0.65 \
    --min-accepted-rate 0.10 --max-fallback 0.25 \
    --min-false-safe-improvement 0.08 \
    --output "$OUT_ROOT/eval/learned_offline/mechanism_verification.json"
fi
if [[ "$STOP_AFTER_STAGE" == "offline" ]]; then
  echo "[cowp_v16.4] stopped after learned-offline verification: $OUT_ROOT"
  exit 0
fi

CALIBRATION_JSON="$OUT_ROOT/eval/learned_offline/bcot_calibration.json"

if [[ -s "$CALIBRATION_JSON" ]]; then
  ONLINE_BCOT_BUDGET="$("$PYTHON_BIN" - "$CALIBRATION_JSON" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["bcot_risk_budget"])
PY
)"
else
  ONLINE_BCOT_BUDGET="$DEFAULT_BCOT_BUDGET"
fi
ONLINE_WITNESS_THRESHOLD="$PAIR_WITNESS_THRESHOLD"
echo "[pair witness threshold] $ONLINE_WITNESS_THRESHOLD"
echo "[BCOT budget] $ONLINE_BCOT_BUDGET"

# Never permit an expensive Waymax run to bypass the learned-offline mechanism
# gate.  RUN_OFFLINE=0 is supported only when a previously generated passing
# verification report is present under the same OUT_ROOT.
if [[ "$RUN_PROBE" == "1" || "$RUN_FULL" == "1" ]]; then
  MECHANISM_REPORT="$OUT_ROOT/eval/learned_offline/mechanism_verification.json"
  [[ -s "$MECHANISM_REPORT" ]] || {
    echo "Online evaluation blocked: missing $MECHANISM_REPORT. Run RUN_OFFLINE=1 first." >&2
    exit 2
  }
  if [[ "$ALLOW_QUALITY_GATE_FAILURE" != "1" ]]; then
    "$PYTHON_BIN" - "$MECHANISM_REPORT" <<'PY_GATE' >/dev/null
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert x.get("pass") is True, "mechanism_verification.json does not pass"
PY_GATE
  else
    echo "[engineering-only] bypassing mechanism pass assertion to exercise Waymax pipeline." >&2
  fi
fi

run_online_one() {
  local method="$1" gpu="$2" scenarios="$3" out="$4" log="$5" label_cfg="$6" shard_count="${7:-1}" shard_index="${8:-0}" outcome_penalty="${9:-$ONLINE_OUTCOME_RISK_PENALTY}"
  local cfg_tag cache
  cfg_tag="$(basename "$label_cfg" .yaml)"
  cache="$OUT_ROOT/jax_cache/${method}_${cfg_tag}_g${gpu}_s${shard_index}"
  mkdir -p "$cache"
  (
    CUDA_VISIBLE_DEVICES="$gpu" JAX_COMPILATION_CACHE_DIR="$cache" \
    "$PYTHON_BIN" -u -m cowp.scripts.04_eval_closed_loop \
      --mode waymax --method "$method" --checkpoint "$CKPT" --cache-dir "$VAL_CACHE" \
      --data-config configs/data.yaml --label-config "$label_cfg" --eval-config "$EVAL_CFG" \
      --tfexample-glob "$WAYMAX_VAL" --num-shards "$shard_count" --shard-index "$shard_index" \
      --num-scenarios "$scenarios" --rollout-horizon-steps "$ROLLOUT_HORIZON" \
      --device auto --waymax-device gpu --waymax-action-mode "$WAYMAX_ACTION_MODE" \
      --jax-visible-devices 0 --jax-preallocate false --waymax-standard-metrics \
      --reuse-waymax-env --prefilter-waymax-shards --jit-waymax-env --jit-waymax-metrics \
      --witness-threshold "$ONLINE_WITNESS_THRESHOLD" --bcot-risk-budget "$ONLINE_BCOT_BUDGET" --ncf-gate-mode "$NCF_GATE_MODE" \
      --outcome-risk-penalty "$outcome_penalty" --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
      --output "$out" --no-progress
  ) > >(tee "$log") 2> >(tee -a "$log" >&2)
}

if [[ "$RUN_PROBE" == "1" ]]; then
  probe_cowp="$OUT_ROOT/eval/probe/cowp_root_transport_${PROBE_SCENARIOS}.json"
  probe_conv="$OUT_ROOT/eval/probe/conventional_safety_${PROBE_SCENARIOS}.json"
  if ! json_valid "$probe_cowp" || ! json_valid "$probe_conv"; then
    run_online_one cowp "$GPU0" "$PROBE_SCENARIOS" "$probe_cowp" "$OUT_ROOT/logs/probe_cowp_root_transport.log" "$LABEL_CFG" & p0=$!
    run_online_one conventional_safety "$GPU1" "$PROBE_SCENARIOS" "$probe_conv" "$OUT_ROOT/logs/probe_conventional.log" "$LABEL_CFG" & p1=$!
    wait_all "$p0" "$p1"
  fi
  logrun summarize_probe "$PYTHON_BIN" -u -m cowp.scripts.24_summarize_planner_delta \
    --reference "$probe_conv" --candidate "$probe_cowp" \
    --output "$OUT_ROOT/eval/probe/delta_conventional_vs_root_transport.json"

  # Second wave: the critical aggregation ablation (legacy pairwise max/any)
  # and the selector-frontier ablation are independent and occupy one GPU each.
  second_wave_pids=()
  if [[ "$RUN_PAIRMAX_ABLATION" == "1" ]]; then
    probe_pairmax="$OUT_ROOT/eval/probe/cowp_pairmax_${PROBE_SCENARIOS}.json"
    if ! json_valid "$probe_pairmax"; then
      run_online_one cowp "$GPU0" "$PROBE_SCENARIOS" "$probe_pairmax" "$OUT_ROOT/logs/probe_cowp_pairmax.log" "$PAIRMAX_CFG" &
      second_wave_pids+=("$!")
    fi
  fi
  if [[ "$RUN_PARETO_ABLATION" == "1" ]]; then
    probe_pareto="$OUT_ROOT/eval/probe/cowp_pareto_${PROBE_SCENARIOS}.json"
    if ! json_valid "$probe_pareto"; then
      run_online_one cowp "$GPU1" "$PROBE_SCENARIOS" "$probe_pareto" "$OUT_ROOT/logs/probe_cowp_pareto.log" "$PARETO_CFG" &
      second_wave_pids+=("$!")
    fi
  fi
  wait_all "${second_wave_pids[@]:-}"

  if [[ "$RUN_PAIRMAX_ABLATION" == "1" ]]; then
    logrun summarize_pairmax_ablation "$PYTHON_BIN" -u -m cowp.scripts.24_summarize_planner_delta \
      --reference "$probe_pairmax" --candidate "$probe_cowp" \
      --output "$OUT_ROOT/eval/probe/delta_pairmax_vs_bcot.json"
  fi
  if [[ "$RUN_PARETO_ABLATION" == "1" ]]; then
    logrun summarize_frontier_ablation "$PYTHON_BIN" -u -m cowp.scripts.24_summarize_planner_delta \
      --reference "$probe_pareto" --candidate "$probe_cowp" \
      --output "$OUT_ROOT/eval/probe/delta_pareto_vs_bcot.json"
  fi

fi
if [[ "$STOP_AFTER_STAGE" == "probe" ]]; then
  echo "[cowp_v16.4] stopped after probe: $OUT_ROOT"
  exit 0
fi

if [[ "$RUN_FULL" == "1" ]]; then
  conv_out="$OUT_ROOT/eval/waymax/conventional_safety_merged.json"
  planner_out="$OUT_ROOT/eval/waymax/planner_score_only_merged.json"
  if ! json_valid "$conv_out" || ! json_valid "$planner_out"; then
    run_online_one conventional_safety "$GPU0" "$FULL_SCENARIOS" "$conv_out" "$OUT_ROOT/logs/full_conventional.log" "$LABEL_CFG" & p0=$!
    run_online_one planner_score_only "$GPU1" "$FULL_SCENARIOS" "$planner_out" "$OUT_ROOT/logs/full_planner.log" "$LABEL_CFG" & p1=$!
    wait_all "$p0" "$p1"
  fi
  per_shard=$(( (FULL_SCENARIOS + 1) / 2 ))
  c0="$OUT_ROOT/eval/waymax/cowp_root_transport_shard0.json"
  c1="$OUT_ROOT/eval/waymax/cowp_root_transport_shard1.json"
  full_cowp_pids=()
  if ! json_valid "$c0"; then
    run_online_one cowp "$GPU0" "$per_shard" "$c0" "$OUT_ROOT/logs/full_cowp_shard0.log" "$LABEL_CFG" 2 0 & full_cowp_pids+=("$!")
  fi
  if ! json_valid "$c1"; then
    run_online_one cowp "$GPU1" "$per_shard" "$c1" "$OUT_ROOT/logs/full_cowp_shard1.log" "$LABEL_CFG" 2 1 & full_cowp_pids+=("$!")
  fi
  wait_all "${full_cowp_pids[@]:-}"
  cowp_merged="$OUT_ROOT/eval/waymax/cowp_root_transport_merged.json"
  logrun merge_cowp "$PYTHON_BIN" -u -m cowp.scripts.17_merge_waymax_shards \
    --output "$cowp_merged" --inputs "$c0" "$c1"
  logrun summarize_full_vs_conventional "$PYTHON_BIN" -u -m cowp.scripts.24_summarize_planner_delta \
    --reference "$conv_out" --candidate "$cowp_merged" \
    --output "$OUT_ROOT/eval/waymax/delta_conventional_vs_cowp.json"
  logrun summarize_full_vs_planner "$PYTHON_BIN" -u -m cowp.scripts.24_summarize_planner_delta \
    --reference "$planner_out" --candidate "$cowp_merged" \
    --output "$OUT_ROOT/eval/waymax/delta_planner_vs_cowp.json"
fi

completion_level="planner"
if [[ "$RUN_PROBE" == "1" ]]; then completion_level="probe"; fi
if [[ "$RUN_FULL" == "1" ]]; then completion_level="full"; fi
logrun validate_pipeline_outputs "$PYTHON_BIN" -u -m cowp.scripts.44_validate_pipeline_outputs \
  --out-root "$OUT_ROOT" --level "$completion_level" \
  --output "$OUT_ROOT/eval/pipeline_completion_report.json"

echo "[cowp_v16.4] complete: $OUT_ROOT"
