#!/usr/bin/env bash
set -euo pipefail

# v16.8.9 paper-grade rebuild for candidate-conditioned causal audit + affected-root transport.
# Key speed policy: build fresh causal labels/tensor/transport first; full train-set Waymax replay is OPTIONAL and disabled by default.
PYTHON_BIN="${PYTHON_BIN:-python}"
# Multiprocess label generation is scene-parallel.  Prevent each NumPy/BLAS/TensorFlow
# worker from spawning its own thread pool and oversubscribing the host.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit}"
# Stable scene-set reuse is both faster and scientifically cleaner: rebuild fresh
# labels on exactly the train/val scenario IDs that were actually present in the
# audited old tensor cache.  No old COWP label is reused.
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
REUSE_OLD_SCENE_SET="${REUSE_OLD_SCENE_SET:-1}"
OLD_SCENESET_TRAIN_CACHE="${OLD_SCENESET_TRAIN_CACHE:-$SOURCE_DATA_ROOT/tensor_cache_train}"
OLD_SCENESET_VAL_CACHE="${OLD_SCENESET_VAL_CACHE:-$SOURCE_DATA_ROOT/tensor_cache_val}"
SOURCE_INDEX_TRAIN="${SOURCE_INDEX_TRAIN:-$SOURCE_DATA_ROOT/index_train.jsonl}"
SOURCE_INDEX_VAL="${SOURCE_INDEX_VAL:-$SOURCE_DATA_ROOT/index_val.jsonl}"
TRAIN_LIMIT="${TRAIN_LIMIT:-22000}"
VAL_LIMIT="${VAL_LIMIT:-5000}"
LABEL_WORKERS_TRAIN="${LABEL_WORKERS_TRAIN:-32}"
LABEL_WORKERS_VAL="${LABEL_WORKERS_VAL:-24}"
CACHE_WORKERS="${CACHE_WORKERS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_WAYMAX_REPLAY="${RUN_WAYMAX_REPLAY:-0}"
MAX_REPLAY_CANDIDATES="${MAX_REPLAY_CANDIDATES:-24}"
REPLAY_HORIZON="${REPLAY_HORIZON:-80}"
ALLOW_UNVERIFIED_RESUME="${ALLOW_UNVERIFIED_RESUME:-0}"
FORCE_INDEX="${FORCE_INDEX:-0}"
RUN_LABEL_DIAGNOSTICS="${RUN_LABEL_DIAGNOSTICS:-0}"
STRICT_VERDICT="${STRICT_VERDICT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_strict_probe/v16_8_9_strict_verdict.json}"

# Never pay the multi-day rebuild cost unless the strict 400+800 probe explicitly
# authorized it under the same code lineage.
[[ -s "$STRICT_VERDICT" ]] || { echo "missing STRICT_VERDICT=$STRICT_VERDICT; run the strict proposal probe first" >&2; exit 4; }
"$PYTHON_BIN" - "$STRICT_VERDICT" <<'PYSTRICT'
import importlib,json,sys
from pathlib import Path
p=json.load(open(sys.argv[1],encoding='utf-8'))
if not p.get('recommend_full_rebuild',False):
    raise SystemExit('strict proposal verdict does not authorize full rebuild')
mod=importlib.import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol')
expected=mod.current_fingerprint(Path.cwd())
actual=str(p.get('code_fingerprint_sha256',''))
if actual != expected:
    raise SystemExit(f'strict verdict/code fingerprint mismatch: verdict={actual} current={expected}; rerun strict probe')
print('strict proposal verdict authorizes full rebuild and matches current code fingerprint')
PYSTRICT

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
REPLAY_DIR="$COWP_ROOT/waymax_replay_v16_8_9"
TRAIN_OUTCOMES="$REPLAY_DIR/train_cache_bal${MAX_REPLAY_CANDIDATES}_safety.jsonl"
VAL_OUTCOMES="$REPLAY_DIR/val_cache_bal${MAX_REPLAY_CANDIDATES}_safety.jsonl"

mkdir -p "$COWP_ROOT/logs" "$REPLAY_DIR"

# Indexes contain only WOMD scenario locations/ids and are independent of the
# COWP label algorithm.  Reusing them is safe when rebuilding on the same WOMD release.
if [[ "$FORCE_INDEX" != "1" && ! -s "$INDEX_TRAIN" && -s "$SOURCE_INDEX_TRAIN" ]]; then
  cp "$SOURCE_INDEX_TRAIN" "$INDEX_TRAIN"
  echo "[index_train] copied stable WOMD index from $SOURCE_INDEX_TRAIN"
fi
if [[ "$FORCE_INDEX" != "1" && ! -s "$INDEX_VAL" && -s "$SOURCE_INDEX_VAL" ]]; then
  cp "$SOURCE_INDEX_VAL" "$INDEX_VAL"
  echo "[index_val] copied stable WOMD index from $SOURCE_INDEX_VAL"
fi

TRAIN_ALLOWLIST=""
VAL_ALLOWLIST=""
if [[ "$REUSE_OLD_SCENE_SET" == "1" ]]; then
  [[ -d "$OLD_SCENESET_TRAIN_CACHE" ]] || { echo "missing OLD_SCENESET_TRAIN_CACHE=$OLD_SCENESET_TRAIN_CACHE" >&2; exit 3; }
  [[ -d "$OLD_SCENESET_VAL_CACHE" ]] || { echo "missing OLD_SCENESET_VAL_CACHE=$OLD_SCENESET_VAL_CACHE" >&2; exit 3; }
  TRAIN_ALLOWLIST="$COWP_ROOT/rebuild_train_scenario_ids.txt"
  VAL_ALLOWLIST="$COWP_ROOT/rebuild_val_scenario_ids.txt"
  find "$OLD_SCENESET_TRAIN_CACHE" -maxdepth 1 -type f -name '*.npz' -printf '%f\n' | sed 's/\.npz$//' | LC_ALL=C sort -u > "$TRAIN_ALLOWLIST"
  find "$OLD_SCENESET_VAL_CACHE" -maxdepth 1 -type f -name '*.npz' -printf '%f\n' | sed 's/\.npz$//' | LC_ALL=C sort -u > "$VAL_ALLOWLIST"
  TRAIN_TARGETS="$(wc -l < "$TRAIN_ALLOWLIST")"
  VAL_TARGETS="$(wc -l < "$VAL_ALLOWLIST")"
  [[ "$TRAIN_TARGETS" -gt 0 && "$VAL_TARGETS" -gt 0 ]] || { echo "old scene-set allowlist is empty" >&2; exit 3; }
  echo "[scene set] paired old-cache IDs: train=$TRAIN_TARGETS val=$VAL_TARGETS"
fi

# Resume is allowed only when the proposal/label implementation fingerprint is
# identical. This prevents a four-day build from silently mixing pre-fix and
# post-fix candidate files under the same root.
FINGERPRINT_FILE="$COWP_ROOT/build_fingerprint.sha256"
CURRENT_FINGERPRINT="$($PYTHON_BIN - <<'PYHASH'
from pathlib import Path
from importlib import import_module
from pathlib import Path
current_fingerprint = import_module("cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol").current_fingerprint
print(current_fingerprint(Path.cwd()))
PYHASH
)"
if [[ -s "$FINGERPRINT_FILE" ]]; then
  STORED_FINGERPRINT="$(tr -d '[:space:]' < "$FINGERPRINT_FILE")"
  [[ "$STORED_FINGERPRINT" == "$CURRENT_FINGERPRINT" ]] || {
    echo "build fingerprint mismatch; use a new COWP_ROOT instead of mixing data versions" >&2
    exit 3
  }
elif [[ -d "$LABELS_TRAIN" || -d "$LABELS_VAL" ]] && [[ "$ALLOW_UNVERIFIED_RESUME" != "1" ]]; then
  echo "existing labels without a v16.8.9 fingerprint; use a new COWP_ROOT or set ALLOW_UNVERIFIED_RESUME=1 after manual audit" >&2
  exit 3
else
  printf '%s\n' "$CURRENT_FINGERPRINT" > "$FINGERPRINT_FILE"
fi

run() {
  local name="$1"; shift
  echo "[$name] $*"
  "$@" > >(tee "$COWP_ROOT/logs/${name}.log") 2> >(tee -a "$COWP_ROOT/logs/${name}.log" >&2)
}

# Fail before the expensive label build if the raw data is not actually the
# future-visible WOMD 1.3.1 contract expected by COWP + full Waymax metrics.
run womd_v131_preflight "$PYTHON_BIN" -m cowp.scripts.64_validate_womd_v131_contract \
  --tfexample-train-glob "$TFEXAMPLE_TRAIN" --tfexample-val-glob "$TFEXAMPLE_VAL" \
  --scenario-train-glob "$SCENARIO_TRAIN" --scenario-val-glob "$SCENARIO_VAL" \
  --sample-shards 64 --scenario-sample-shards 32 --require-sdc-paths \
  --output "$COWP_ROOT/womd_v1_3_1_preflight.json"

if [[ "$FORCE_INDEX" == "1" || ! -s "$INDEX_TRAIN" ]]; then
  run index_train "$PYTHON_BIN" -m cowp.scripts.00_index_womd \
    --data-config configs/data.yaml --proto-glob "$SCENARIO_TRAIN" \
    --output "$INDEX_TRAIN" --cpu-only
else
  echo "[index_train] reuse $INDEX_TRAIN"
fi
if [[ "$FORCE_INDEX" == "1" || ! -s "$INDEX_VAL" ]]; then
  run index_val "$PYTHON_BIN" -m cowp.scripts.00_index_womd \
    --data-config configs/data.yaml --proto-glob "$SCENARIO_VAL" \
    --output "$INDEX_VAL" --cpu-only
else
  echo "[index_val] reuse $INDEX_VAL"
fi

# Exact split-leakage guard.  The algorithm consumes future-visible train/val
# labels, so any scenario-id overlap would invalidate held-out validation even
# if every per-scene tensor passed its local integrity checks.
run split_leakage_check "$PYTHON_BIN" -m cowp.scripts.09_check_splits \
  --train "$INDEX_TRAIN" --val "$INDEX_VAL" --fail-on-overlap \
  --output "$COWP_ROOT/train_val_split_audit.json"

LABEL_TRAIN_EXTRA=(--limit "$TRAIN_LIMIT")
LABEL_VAL_EXTRA=(--limit "$VAL_LIMIT")
if [[ -n "$TRAIN_ALLOWLIST" ]]; then LABEL_TRAIN_EXTRA=(--allow-scenario-ids "$TRAIN_ALLOWLIST"); fi
if [[ -n "$VAL_ALLOWLIST" ]]; then LABEL_VAL_EXTRA=(--allow-scenario-ids "$VAL_ALLOWLIST"); fi
run labels_train "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --proto-glob "$SCENARIO_TRAIN" --output-dir "$LABELS_TRAIN" \
  --index-jsonl "$INDEX_TRAIN" "${LABEL_TRAIN_EXTRA[@]}" \
  --num-workers "$LABEL_WORKERS_TRAIN" --start-method forkserver \
  --max-pending-multiplier 2 --no-compress --skip-existing --skip-diagnostics \
  --profile-jsonl "$COWP_ROOT/profile_labels_train.jsonl" --cpu-only
run labels_val "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --proto-glob "$SCENARIO_VAL" --output-dir "$LABELS_VAL" \
  --index-jsonl "$INDEX_VAL" "${LABEL_VAL_EXTRA[@]}" \
  --num-workers "$LABEL_WORKERS_VAL" --start-method forkserver \
  --max-pending-multiplier 2 --no-compress --skip-existing --skip-diagnostics \
  --profile-jsonl "$COWP_ROOT/profile_labels_val.jsonl" --cpu-only

# Audit active learned-label support before spending additional I/O on WOMD
# tensor merges. This is stricter than the original seven-head class check.
run model_support_labels_train "$PYTHON_BIN" -m cowp.scripts.65_audit_model_support \
  --cache-dir "$LABELS_TRAIN" --sample-scenes 0 --min-class-examples 128 --min-source-examples 128 --strict \
  --output "$COWP_ROOT/model_support_audit_labels_train.json"
run model_support_labels_val "$PYTHON_BIN" -m cowp.scripts.65_audit_model_support \
  --cache-dir "$LABELS_VAL" --sample-scenes 0 --min-class-examples 32 --min-source-examples 32 --strict \
  --output "$COWP_ROOT/model_support_audit_labels_val.json"

if [[ "$RUN_LABEL_DIAGNOSTICS" == "1" ]]; then
run diagnose_labels_train "$PYTHON_BIN" cowp/scripts/06_diagnose_dataset.py \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --labels-dir "$LABELS_TRAIN" --output-dir "$COWP_ROOT/diagnostics_train" \
  --make-visualizations --max-visualizations 64
run diagnose_labels_val "$PYTHON_BIN" cowp/scripts/06_diagnose_dataset.py \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --labels-dir "$LABELS_VAL" --output-dir "$COWP_ROOT/diagnostics_val" \
  --make-visualizations --max-visualizations 64
else
  echo "[diagnostics] skipped during build; run after promotion/training if needed"
fi

run tensor_train "$PYTHON_BIN" -m cowp.scripts.02_build_tensor_cache \
  --data-config configs/data.yaml --split training --tfexample-glob "$TFEXAMPLE_TRAIN" \
  --labels-dir "$LABELS_TRAIN" --output-dir "$BASE_TRAIN" \
  --num-workers "$CACHE_WORKERS" --start-method forkserver --parallel-scan \
  --require-waymax-ready --require-sdc-paths --skip-existing --no-compress \
  --profile-jsonl "$COWP_ROOT/profile_tensor_cache_train.jsonl" --cpu-only
run tensor_val "$PYTHON_BIN" -m cowp.scripts.02_build_tensor_cache \
  --data-config configs/data.yaml --split validation --tfexample-glob "$TFEXAMPLE_VAL" \
  --labels-dir "$LABELS_VAL" --output-dir "$BASE_VAL" \
  --num-workers "$CACHE_WORKERS" --start-method forkserver --parallel-scan \
  --require-waymax-ready --require-sdc-paths --skip-existing --no-compress \
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
  [[ -s "$TRAIN_OUTCOMES" ]] || { echo "missing $TRAIN_OUTCOMES" >&2; exit 2; }
  [[ -s "$VAL_OUTCOMES" ]] || { echo "missing $VAL_OUTCOMES" >&2; exit 2; }
  run attach_train "$PYTHON_BIN" -m cowp.scripts.12_attach_waymax_candidate_outcomes \
    --cache-dir "$BASE_TRAIN" --output-dir "$RAW_TRAIN" --outcomes-jsonl "$TRAIN_OUTCOMES"
  run attach_val "$PYTHON_BIN" -m cowp.scripts.12_attach_waymax_candidate_outcomes \
    --cache-dir "$BASE_VAL" --output-dir "$RAW_VAL" --outcomes-jsonl "$VAL_OUTCOMES"
  run verify_train "$PYTHON_BIN" -m cowp.scripts.14_verify_waymax_cache --cache-dir "$RAW_TRAIN"
  run verify_val "$PYTHON_BIN" -m cowp.scripts.14_verify_waymax_cache --cache-dir "$RAW_VAL"
  EFFECTIVE_RAW_TRAIN="$RAW_TRAIN"
  EFFECTIVE_RAW_VAL="$RAW_VAL"
else
  # Candidate outcomes are optional auxiliary planner labels.  Core NCF/BCOT/RCOT
  # training and real online Waymax evaluation do not require replaying every
  # train candidate.  This saves the most expensive non-label data stage.
  EFFECTIVE_RAW_TRAIN="$BASE_TRAIN"
  EFFECTIVE_RAW_VAL="$BASE_VAL"
  echo "[waymax replay] skipped; planner must export USE_WAYMAX_OUTCOME_LABELS=0"
fi

# Fresh v16.8.9 causal-audit data is intentionally self-contained.  The fresh
# label engine already emits the complete transport supervision, including
# root_min_safe_burden and canonical_root_weight, so a second augmentation pass
# would duplicate expensive geometry work and reintroduce fragile overlay symlinks.
TRAIN_CACHE_FINAL="$EFFECTIVE_RAW_TRAIN"
VAL_CACHE_FINAL="$EFFECTIVE_RAW_VAL"

VERIFY_TRAIN_ALLOW=()
VERIFY_VAL_ALLOW=()
if [[ -n "$TRAIN_ALLOWLIST" ]]; then VERIFY_TRAIN_ALLOW=(--allowlist "$TRAIN_ALLOWLIST"); fi
if [[ -n "$VAL_ALLOWLIST" ]]; then VERIFY_VAL_ALLOW=(--allowlist "$VAL_ALLOWLIST"); fi
run verify_fresh_train "$PYTHON_BIN" -m cowp.scripts.60_verify_fresh_v16_8_9_cache \
  --cache-dir "$TRAIN_CACHE_FINAL" "${VERIFY_TRAIN_ALLOW[@]}" \
  --sample-scenes 0 --require-sdc-paths --output "$COWP_ROOT/fresh_cache_integrity_train.json"
run verify_fresh_val "$PYTHON_BIN" -m cowp.scripts.60_verify_fresh_v16_8_9_cache \
  --cache-dir "$VAL_CACHE_FINAL" "${VERIFY_VAL_ALLOW[@]}" \
  --sample-scenes 0 --require-sdc-paths --output "$COWP_ROOT/fresh_cache_integrity_val.json"
run supervision_train "$PYTHON_BIN" -m cowp.scripts.62_audit_training_supervision \
  --cache-dir "$TRAIN_CACHE_FINAL" --sample-scenes 0 --min-class-examples 128 --strict \
  --output "$COWP_ROOT/training_supervision_audit_train.json"
run supervision_val "$PYTHON_BIN" -m cowp.scripts.62_audit_training_supervision \
  --cache-dir "$VAL_CACHE_FINAL" --sample-scenes 0 --min-class-examples 32 --strict \
  --output "$COWP_ROOT/training_supervision_audit_val.json"
run model_support_cache_train "$PYTHON_BIN" -m cowp.scripts.65_audit_model_support \
  --cache-dir "$TRAIN_CACHE_FINAL" --sample-scenes 0 --min-class-examples 128 --min-source-examples 128 --strict \
  --output "$COWP_ROOT/model_support_audit_cache_train.json"
run model_support_cache_val "$PYTHON_BIN" -m cowp.scripts.65_audit_model_support \
  --cache-dir "$VAL_CACHE_FINAL" --sample-scenes 0 --min-class-examples 32 --min-source-examples 32 --strict \
  --output "$COWP_ROOT/model_support_audit_cache_val.json"

# Recompute the full validation proposal ceiling from the actual post-merge cache.
# This is a final dataset-level guard before any GPU training starts.
run proposal_ceiling_val "$PYTHON_BIN" -m cowp.scripts.45_diagnose_proposal_ceiling \
  --cache-dir "$VAL_CACHE_FINAL" --output "$COWP_ROOT/fresh_proposal_ceiling_val.json" \
  --hard-count 0 --random-count 0 --control-count 0 --seed 2026
run proposal_source_ablation_val "$PYTHON_BIN" -m cowp.scripts.50_ablate_proposal_sources \
  --cache-dir "$VAL_CACHE_FINAL" --output "$COWP_ROOT/fresh_proposal_source_ablation_val.json"
run causal_audit_val "$PYTHON_BIN" -m cowp.scripts.57_diagnose_causal_audit \
  --cache-dir "$VAL_CACHE_FINAL" --output "$COWP_ROOT/fresh_causal_audit_val.json"
"$PYTHON_BIN" - "$COWP_ROOT/fresh_proposal_ceiling_val.json" "$COWP_ROOT/fresh_proposal_source_ablation_val.json" "$COWP_ROOT/fresh_causal_audit_val.json" <<'PYCHECK'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
a=json.load(open(sys.argv[2],encoding='utf-8'))
d=json.load(open(sys.argv[3],encoding='utf-8'))
r=p['scene_rates']
src=p.get('proposal_source_stats',{})
psy=src.get('PRIORITY_SMOOTH_YIELD',{})
all_bank=a.get('ablations',{}).get('all',{})
no_psy=a.get('ablations',{}).get('without_priority_smooth_yield',{})
checks={
 'any_valid': r['any_valid'] >= 0.99,
 'any_ncf': r['any_ncf'] >= 0.40,
 'false_safe_floor': r['best_case_selected_false_safe_lower_bound'] <= 0.55,
 'pbtr_floor': r['best_case_pbtr_lower_bound'] <= 0.45,
 'audit_no_read_errors': bool(d.get('integrity',{}).get('no_read_errors',False)),
 'audit_no_silent_blockers': bool(d.get('integrity',{}).get('no_silent_blockers',False)),
 'audit_no_irrelevant_blockers': bool(d.get('integrity',{}).get('no_irrelevant_blockers',False)),
 'audit_transport_affected_match': bool(d.get('integrity',{}).get('transport_affected_matches_audit',False)),
 'audit_transport_conflict_match': bool(d.get('integrity',{}).get('transport_conflict_matches_audit',False)),
 'audit_transport_retain_match': bool(d.get('integrity',{}).get('transport_retain_matches_audit',False)),
 'audit_canonical_weight_match': bool(d.get('integrity',{}).get('canonical_root_weight_matches_transport',False)),
 'audit_no_irrelevant_responses': bool(d.get('integrity',{}).get('no_responses_for_irrelevant_pairs',False)),
 'audit_relevance_non_degenerate': 0.01 <= float(d.get('pair_rates',{}).get('relevant',0.0)) <= 0.95,
 'affected_definition_consistent': bool(d.get('integrity',{}).get('affected_definition_consistent',False)),
 'burden_only_definition_consistent': bool(d.get('integrity',{}).get('burden_only_definition_consistent',False)),
 'proposal_union_monotone_any_ncf': float(all_bank.get('any_ncf_scene_rate',0.0)) + 1e-12 >= float(no_psy.get('any_ncf_scene_rate',0.0)),
 'proposal_union_monotone_false_safe': float(all_bank.get('best_case_selected_false_safe_lower_bound',1.0)) <= float(no_psy.get('best_case_selected_false_safe_lower_bound',1.0)) + 1e-12,
 'proposal_union_monotone_pbtr': float(all_bank.get('best_case_pbtr_lower_bound',1.0)) <= float(no_psy.get('best_case_pbtr_lower_bound',1.0)) + 1e-12,
}
print('postbuild proposal gates:', checks)
if not all(checks.values()):
 raise SystemExit('Fresh full cache fails proposal/source-integrity gates; do not train.')
PYCHECK

cat > "$COWP_ROOT/data_manifest_v16_8_9.json" <<JSON
{
  "schema_version": "cowp_v16_8_9_causal_audit_self_contained_data_v1",
  "build_fingerprint_sha256": "$CURRENT_FINGERPRINT",
  "label_config": "$LABEL_CFG",
  "eval_config": "$EVAL_CFG",
  "train_limit": $TRAIN_LIMIT,
  "val_limit": $VAL_LIMIT,
  "reuse_old_scene_set": $([[ "$REUSE_OLD_SCENE_SET" == "1" ]] && echo true || echo false),
  "old_sceneset_train_cache": "$OLD_SCENESET_TRAIN_CACHE",
  "old_sceneset_val_cache": "$OLD_SCENESET_VAL_CACHE",
  "max_replay_candidates": $MAX_REPLAY_CANDIDATES,
  "waymax_replay_enabled": $([[ "$RUN_WAYMAX_REPLAY" == "1" ]] && echo true || echo false),
  "raw_train_cache": "$EFFECTIVE_RAW_TRAIN",
  "raw_val_cache": "$EFFECTIVE_RAW_VAL",
  "transport_storage": "inline_self_contained",
  "transport_train_cache": "$TRAIN_CACHE_FINAL",
  "transport_val_cache": "$VAL_CACHE_FINAL",
  "notes": "Fresh labels use fixed-anchor global critical selection plus candidate-conditioned causal audit relevance. Affected-root transport is the exact root-level audit support (unsafe OR direct budget crossing); burden-only prevalence is reported but is not artificially required. Responses/witness/RCOT share the same audit support; silent/irrelevant blockers are forbidden. Transport supervision is serialized inline; no symlink overlay is required."
}
JSON

echo "[v16.8.9 self-contained causal-audit data] complete: $COWP_ROOT"
echo "RAW_TRAIN_CACHE=$EFFECTIVE_RAW_TRAIN"
echo "RAW_VAL_CACHE=$EFFECTIVE_RAW_VAL"
echo "TRAIN_CACHE=$TRAIN_CACHE_FINAL"
echo "VAL_CACHE=$VAL_CACHE_FINAL"
echo "USE_WAYMAX_OUTCOME_LABELS=$([[ "$RUN_WAYMAX_REPLAY" == "1" ]] && echo 1 || echo 0)"
