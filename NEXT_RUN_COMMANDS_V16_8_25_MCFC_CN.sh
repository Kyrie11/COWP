#!/usr/bin/env bash
set -euo pipefail

# v16.8.25 evidence-first execution plan.
#
# The inspected v16.8.24 held-out set has already influenced algorithm design.
# It is therefore DEVELOPMENT/DIAGNOSTIC data from this point onward.  The
# `freeze_final_split` stage creates a new content-blind final holdout using
# scenario IDs only; do not build/decode/evaluate it until the algorithm,
# hyperparameters, BCOT operating point, and checkpoint-selection rule are frozen.
#
# Recommended order:
#   1) freeze_final_split   # freeze IDs before inspecting any new experiment output
#   2) strict_v24_dev
#   3) learned_v24_diagnostics
#   4) proposal_probe
#   5) only if proposal_probe says promote_mcfc_to_full_rebuild=true:
#        full_rebuild -> attach_waymax -> train_v25 -> eval_v25_dev
#   6) after everything is frozen:
#        build_final_blind -> eval_final_blind
#
# Usage examples:
#   bash NEXT_RUN_COMMANDS_V16_8_25_MCFC_CN.sh strict_v24_dev
#   bash NEXT_RUN_COMMANDS_V16_8_25_MCFC_CN.sh proposal_probe
#
# Required for stages touching WOMD:
#   export WOMD_ROOT=/data0/.../waymo_open_dataset_motion_v_1_3_1

STAGE="${1:-help}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OLD_ROOT="${OLD_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
OLD_RUN="${OLD_RUN:-outputs/v16_8_24_compact5k_all}"
OLD_CKPT="${OLD_CKPT:-$OLD_RUN/cowp_all_best.pt}"
WOMD_ROOT="${WOMD_ROOT:-}"
NEW_ROOT="${NEW_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_25_mcfc_compact_full_5k}"
NEW_RUN="${NEW_RUN:-outputs/v16_8_25_mcfc_compact5k}"
PROBE_ROOT="${PROBE_ROOT:-outputs/v16_8_25_mcfc_proposal_probe_val}"
FINAL_ROOT="${FINAL_ROOT:-/data0/senzeyu2/dataset/COWP/final_blind_v16_8_25_mcfc}"
FINAL_SPLIT_ROOT="${FINAL_SPLIT_ROOT:-outputs/final_blind_split_v1}"
FINAL_IDS="${FINAL_IDS:-$FINAL_SPLIT_ROOT/final_blind_scene_ids.txt}"
FINAL_MANIFEST="${FINAL_MANIFEST:-$FINAL_SPLIT_ROOT/final_blind_split_manifest.json}"
LABEL_CFG_OLD="${LABEL_CFG_OLD:-configs/label_cowp_v16_8.yaml}"
LABEL_CFG_NEW="${LABEL_CFG_NEW:-configs/label_cowp_v16_8_25_mcfc.yaml}"
MODEL_CFG="${MODEL_CFG:-configs/model_cowp_v16_8.yaml}"
TRAIN_CFG="${TRAIN_CFG:-configs/train_cowp_v16_8.yaml}"
EVAL_CFG="${EVAL_CFG:-configs/eval_cowp_v16_8.yaml}"
WITNESS_THRESHOLD="${WITNESS_THRESHOLD:-0.70}"
BCOT_BUDGET_V24="${BCOT_BUDGET_V24:-0.50}"
STRICT_ID_FILE="${STRICT_ID_FILE:-$OLD_ROOT/heldout_test_scene_ids.txt}"
SEED="${SEED:-2026}"

# Stable thread settings for PyTorch/data workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

need_womd(){
  if [[ -z "$WOMD_ROOT" ]]; then
    echo "ERROR: export WOMD_ROOT=/path/to/waymo_open_dataset_motion_v_1_3_1" >&2
    exit 2
  fi
}

need_file(){
  [[ -f "$1" ]] || { echo "ERROR: missing file: $1" >&2; exit 2; }
}

need_dir(){
  [[ -d "$1" ]] || { echo "ERROR: missing directory: $1" >&2; exit 2; }
}

json_get(){
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
for part in sys.argv[2].split('.'):
    x=x[part]
print(x)
PY
}

optional_tfexample_index_args(){
  # The exact-ID evaluator works without an index.  If a reusable index exists,
  # pass it to avoid scanning irrelevant validation shards.
  local p="${TFEXAMPLE_INDEX_VAL:-}"
  if [[ -n "$p" && -f "$p" ]]; then
    printf '%s\0%s\0' --tfexample-index-jsonl "$p"
  fi
}

strict_v24_dev(){
  need_womd
  need_file "$OLD_CKPT"
  need_file "$STRICT_ID_FILE"
  mkdir -p "$OLD_RUN/strict_waymax_exact_dev"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false

  # Use one method/process at a time.  Exact-ID mode fails if any requested ID is
  # missing; never add --allow-missing-scenario-ids for paper evidence.
  local -a idx_args=()
  if [[ -n "${TFEXAMPLE_INDEX_VAL:-}" && -f "${TFEXAMPLE_INDEX_VAL}" ]]; then
    idx_args=(--tfexample-index-jsonl "$TFEXAMPLE_INDEX_VAL")
  fi
  local methods=(cowp conventional_safety planner_score_only soft_burden_cost_only universal_ncf idm_lattice)
  for method in "${methods[@]}"; do
    "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
      --data-config configs/data.yaml \
      --label-config "$LABEL_CFG_OLD" \
      --eval-config "$EVAL_CFG" \
      --mode waymax \
      --method "$method" \
      --checkpoint "$OLD_CKPT" \
      --device cuda:0 \
      --waymax-device gpu \
      --jax-visible-devices "${JAX_VISIBLE_DEVICES:-1}" \
      --jax-preallocate false \
      --waymax-split validation \
      --tfexample-glob "$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*" \
      --scenario-ids-file "$STRICT_ID_FILE" \
      "${idx_args[@]}" \
      --rollout-horizon-steps 80 \
      --waymax-action-mode absolute_xy_yaw \
      --ncf-gate-mode priority \
      --witness-threshold "$WITNESS_THRESHOLD" \
      --bcot-risk-budget "$BCOT_BUDGET_V24" \
      --waymax-standard-metrics \
      --waymax-standard-metric-names OverlapMetric,OffroadMetric,ProgressionMetric,KinematicsInfeasibilityMetric \
      --reuse-waymax-env --prefilter-waymax-shards --jit-waymax-env --jit-waymax-metrics \
      --status-every 20 --no-progress \
      --output "$OLD_RUN/strict_waymax_exact_dev/${method}.json"
  done

  "$PYTHON_BIN" - "$OLD_RUN/strict_waymax_exact_dev" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1])
rows=[]
for p in sorted(root.glob('*.json')):
    x=json.loads(p.read_text())
    # Keep this deliberately generic because Waymax versions may expose the
    # standard-metric aggregate under slightly different nested keys.
    rows.append({
        'method':p.stem,
        'scenario_ids_requested':x.get('scenario_ids_requested_count'),
        'scenario_ids_requested_sha256':x.get('scenario_ids_requested_sha256'),
        'num_results':len(x.get('results',[])) if isinstance(x.get('results'),list) else x.get('num_results'),
        'output':str(p),
    })
print(json.dumps({'strict_exact_outputs':rows},indent=2,ensure_ascii=False))
PY
}

learned_v24_diagnostics(){
  need_file "$OLD_CKPT"
  need_dir "$OLD_ROOT/tensor_cache_val_waymax"
  need_dir "$OLD_ROOT/tensor_cache_heldout_test_waymax"
  mkdir -p "$OLD_RUN/diagnostics_v25_reanalysis"

  # A wider validation budget sweep is needed because the previous sweep ended
  # at 0.50 while calibration was still proposal-infeasible.  This is diagnostic
  # only; do not select an operating point using the already-inspected heldout.
  "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
    --data-config configs/data.yaml --label-config "$LABEL_CFG_OLD" --eval-config "$EVAL_CFG" \
    --mode learned_offline --cache-dir "$OLD_ROOT/tensor_cache_val_waymax" \
    --checkpoint "$OLD_CKPT" --method cowp --ncf-gate-mode priority \
    --witness-threshold "$WITNESS_THRESHOLD" \
    --bcot-risk-budget-sweep 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70 \
    --witness-threshold-sweep 0.50,0.60,0.70,0.80,0.90 \
    --batch-size 8 --device cuda:0 --num-workers 2 --prefetch-factor 1 \
    --output "$OLD_RUN/diagnostics_v25_reanalysis/val_budget_witness_sweep.json"

  "$PYTHON_BIN" -m cowp.scripts.31_calibrate_bcot_budget \
    --input "$OLD_RUN/diagnostics_v25_reanalysis/val_budget_witness_sweep.json" \
    --output "$OLD_RUN/diagnostics_v25_reanalysis/val_calibration.json"

  "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
    --data-config configs/data.yaml --label-config "$LABEL_CFG_OLD" --eval-config "$EVAL_CFG" \
    --mode learned_offline --cache-dir "$OLD_ROOT/tensor_cache_heldout_test_waymax" \
    --checkpoint "$OLD_CKPT" \
    --methods cowp,conventional_safety,planner_score_only,soft_burden_cost_only,universal_ncf,outcome_oracle \
    --ncf-gate-mode priority --witness-threshold "$WITNESS_THRESHOLD" \
    --bcot-risk-budget "$BCOT_BUDGET_V24" \
    --batch-size 8 --device cuda:0 --num-workers 2 --prefetch-factor 1 \
    --output "$OLD_RUN/diagnostics_v25_reanalysis/heldout_internal_methods.json"

  # Universal-veto and no-gate are inference-policy diagnostics, not architecture
  # ablations.  True branch/module ablations require separately rebuilt labels /
  # retrained checkpoints; current evaluator intentionally rejects fake no-op names.
  "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
    --data-config configs/data.yaml --label-config "$LABEL_CFG_OLD" --eval-config "$EVAL_CFG" \
    --mode learned_offline --cache-dir "$OLD_ROOT/tensor_cache_heldout_test_waymax" \
    --checkpoint "$OLD_CKPT" --method cowp --ncf-gate-mode none \
    --witness-threshold "$WITNESS_THRESHOLD" --bcot-risk-budget "$BCOT_BUDGET_V24" \
    --batch-size 8 --device cuda:0 --num-workers 2 --prefetch-factor 1 \
    --output "$OLD_RUN/diagnostics_v25_reanalysis/heldout_no_ncf_gate.json"
}

proposal_probe(){
  need_womd
  need_dir "$OLD_ROOT/tensor_cache_val_waymax"
  mkdir -p "$PROBE_ROOT"
  local hard_ids="$PROBE_ROOT/hard_scene_ids.txt"
  local random_ids="$PROBE_ROOT/random_scene_ids.txt"
  local probe_ids="$PROBE_ROOT/probe_scene_ids.txt"
  local old_diag="$PROBE_ROOT/old_proposal_ceiling.json"
  local labels="$PROBE_ROOT/labels"
  local cache="$PROBE_ROOT/tensor_cache"
  local profile="$PROBE_ROOT/profile_labels.jsonl"
  local index_val="$WOMD_ROOT/.cowp_v131_indices/scenario_location_index_validation.jsonl"
  local scenario_val="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
  local tfexample_val="$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*"

  need_file "$index_val"
  "$PYTHON_BIN" -m cowp.scripts.45_diagnose_proposal_ceiling \
    --cache-dir "$OLD_ROOT/tensor_cache_val_waymax" --output "$old_diag" \
    --hard-definition protected --hard-count 300 --hard-scene-ids "$hard_ids" \
    --random-count 300 --random-scene-ids "$random_ids" \
    --probe-total-count 600 --random-exclude-hard-probe --seed 20260823

  cat "$hard_ids" "$random_ids" | awk 'NF && !seen[$1]++ {print $1}' > "$probe_ids"
  echo "probe ids: $(wc -l < "$probe_ids")"

  # Build a fresh sparse label/cache under the opt-in MCFC config.  Do not reuse
  # stale labels: the proposal source and the resulting mechanism labels must be
  # regenerated together for an attribution-valid probe.
  rm -rf "$labels" "$cache"
  rm -f "$profile"
  mkdir -p "$labels" "$cache"
  "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
    --data-config configs/data.yaml --label-config "$LABEL_CFG_NEW" \
    --proto-glob "$scenario_val" --output-dir "$labels" \
    --allow-scenario-ids "$probe_ids" --index-jsonl "$index_val" --require-all-allowed-resolved \
    --num-workers "${PROBE_LABEL_WORKERS:-24}" --start-method forkserver \
    --max-pending-multiplier 2 --no-compress --skip-diagnostics --cpu-only \
    --profile-jsonl "$profile"

  "$PYTHON_BIN" -m cowp.scripts.49_summarize_label_build_profile \
    --input "$profile" --output "$PROBE_ROOT/profile_summary.json"

  "$PYTHON_BIN" -m cowp.scripts.02_build_tensor_cache \
    --data-config configs/data.yaml --split validation --tfexample-glob "$tfexample_val" \
    --labels-dir "$labels" --output-dir "$cache" \
    --num-workers "${PROBE_CACHE_WORKERS:-8}" --start-method forkserver --parallel-scan \
    --require-waymax-ready --require-sdc-paths --require-all-labels-matched \
    --no-compress --cpu-only --profile-jsonl "$PROBE_ROOT/profile_tensor_cache.jsonl"

  "$PYTHON_BIN" -m cowp.scripts.46_compare_proposal_probe \
    --old-cache "$OLD_ROOT/tensor_cache_val_waymax" --new-cache "$cache" \
    --representative-scene-ids "$probe_ids" --hard-scene-ids "$hard_ids" \
    --new-build-profile "$profile" \
    --min-overall-any-valid 0.99 --min-overall-any-ncf 0.40 \
    --max-false-safe-floor 0.55 --max-pbtr-floor 0.45 \
    --min-hard-recovery 0.20 --max-rmr-target-tta-error-s 0.20 \
    --output "$PROBE_ROOT/paired_proposal_probe.json"

  "$PYTHON_BIN" -m cowp.scripts.50_ablate_proposal_sources \
    --cache-dir "$cache" --output "$PROBE_ROOT/proposal_source_ablation.json"

  "$PYTHON_BIN" -m cowp.scripts.57_diagnose_causal_audit \
    --cache-dir "$cache" --scene-ids "$probe_ids" --output "$PROBE_ROOT/causal_audit.json"

  # The source-attributed screen is deliberately stricter than "some MCFC
  # candidates were generated".  It requires scene-level option-coverage gains
  # and lower oracle floors before spending on a 5k rebuild/retrain.
  "$PYTHON_BIN" -m cowp.scripts.85_screen_v16_8_25_mcfc_probe \
    --paired-probe "$PROBE_ROOT/paired_proposal_probe.json" \
    --source-ablation "$PROBE_ROOT/proposal_source_ablation.json" \
    --output "$PROBE_ROOT/mcfc_promotion_screen.json"

  echo "MCFC promotion verdict: $(json_get "$PROBE_ROOT/mcfc_promotion_screen.json" promote_mcfc_to_full_rebuild)"
}

freeze_final_split(){
  need_womd
  local index_val="$WOMD_ROOT/.cowp_v131_indices/scenario_location_index_validation.jsonl"
  need_file "$index_val"
  need_file "$OLD_ROOT/val_scene_ids.txt"
  need_file "$OLD_ROOT/heldout_test_scene_ids.txt"
  mkdir -p "$FINAL_SPLIT_ROOT"
  "$PYTHON_BIN" -m cowp.scripts.84_make_blind_final_holdout \
    --scenario-index-jsonl "$index_val" \
    --exclude-ids "$OLD_ROOT/val_scene_ids.txt" \
    --exclude-ids "$OLD_ROOT/heldout_test_scene_ids.txt" \
    --count "${FINAL_COUNT:-1200}" --seed "${FINAL_SPLIT_SEED:-cowp-ccfa-final-v1}" \
    --output "$FINAL_IDS" --manifest "$FINAL_MANIFEST"
  echo "Final-blind IDs frozen at $FINAL_IDS"
  echo "Do NOT decode/evaluate them until algorithm/hyperparameters/checkpoint rule are frozen."
}

require_mcfc_promotion(){
  local gate="$PROBE_ROOT/mcfc_promotion_screen.json"
  need_file "$gate"
  local verdict
  verdict="$(json_get "$gate" promote_mcfc_to_full_rebuild)"
  if [[ "$verdict" != "True" && "$verdict" != "true" ]]; then
    echo "STOP: MCFC proposal probe did not pass source-attributed promotion gate." >&2
    echo "Do not rebuild/retrain v16.8.25. Inspect $gate and revise the proposal mechanism first." >&2
    exit 3
  fi
}

full_rebuild(){
  need_womd
  require_mcfc_promotion
  export WOMD_ROOT
  export SOURCE_DATA_ROOT="$OLD_ROOT"
  export COWP_ROOT="$NEW_ROOT"
  export LABEL_CFG="$LABEL_CFG_NEW"
  bash PREPARE_COWP_V16_8_25_MCFC_DATA_CN.sh

  # Paired development split must remain identical to v16.8.24; otherwise the
  # proposal/mechanism comparison is confounded by a split change.
  cmp "$OLD_ROOT/train_scene_ids.txt" "$NEW_ROOT/train_scene_ids.txt"
  cmp "$OLD_ROOT/val_scene_ids.txt" "$NEW_ROOT/val_scene_ids.txt"
  cmp "$OLD_ROOT/heldout_test_scene_ids.txt" "$NEW_ROOT/heldout_test_scene_ids.txt"
  echo "PASS: v16.8.24 and v16.8.25 development splits are exact-ID identical."
}

attach_waymax(){
  need_womd
  require_mcfc_promotion
  need_dir "$NEW_ROOT/tensor_cache_train"
  export WOMD_ROOT COWP_ROOT="$NEW_ROOT" LABEL_CFG="$LABEL_CFG_NEW"
  bash ATTACH_WAYMAX_OUTCOMES_V16_8_25_MCFC_CN.sh
}

make_seeded_train_config(){
  local out="$1"
  "$PYTHON_BIN" - "$TRAIN_CFG" "$out" "$SEED" <<'PY'
from pathlib import Path
import sys,yaml
src,out,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
x=yaml.safe_load(Path(src).read_text(encoding='utf-8'))
x['seed']=seed
Path(out).parent.mkdir(parents=True,exist_ok=True)
Path(out).write_text(yaml.safe_dump(x,sort_keys=False),encoding='utf-8')
print(out)
PY
}

train_v25(){
  require_mcfc_promotion
  need_dir "$NEW_ROOT/tensor_cache_train_waymax"
  need_dir "$NEW_ROOT/tensor_cache_val_waymax"
  mkdir -p "$NEW_RUN"
  local seeded_cfg="$NEW_RUN/train_seed_${SEED}.yaml"
  make_seeded_train_config "$seeded_cfg"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" -m cowp.scripts.03_train \
    --data-config configs/data.yaml --model-config "$MODEL_CFG" --label-config "$LABEL_CFG_NEW" \
    --train-config "$seeded_cfg" \
    --cache-dir "$NEW_ROOT/tensor_cache_train_waymax" --val-cache-dir "$NEW_ROOT/tensor_cache_val_waymax" \
    --stage all --with-waymax-outcome-labels --batch-size "${TRAIN_BATCH_SIZE:-4}" \
    --amp --amp-dtype bfloat16 --num-workers "${TRAIN_WORKERS:-4}" --val-num-workers "${VAL_WORKERS:-2}" \
    --prefetch-factor 1 --val-prefetch-factor 1 --sharing-strategy file_system --fused-adamw \
    --eval-before-train --output-dir "$NEW_RUN"
}

calibrate_v25(){
  local ckpt="${NEW_CKPT:-$NEW_RUN/cowp_all_best.pt}"
  need_file "$ckpt"
  need_dir "$NEW_ROOT/tensor_cache_val_waymax"
  mkdir -p "$NEW_RUN/eval_dev"
  "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
    --data-config configs/data.yaml --label-config "$LABEL_CFG_NEW" --eval-config "$EVAL_CFG" \
    --mode learned_offline --cache-dir "$NEW_ROOT/tensor_cache_val_waymax" --checkpoint "$ckpt" \
    --method cowp --ncf-gate-mode priority --witness-threshold "$WITNESS_THRESHOLD" \
    --bcot-risk-budget-sweep 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70 \
    --batch-size 8 --device cuda:0 --num-workers 2 --prefetch-factor 1 \
    --output "$NEW_RUN/eval_dev/val_budget_sweep.json"
  "$PYTHON_BIN" -m cowp.scripts.31_calibrate_bcot_budget \
    --input "$NEW_RUN/eval_dev/val_budget_sweep.json" \
    --output "$NEW_RUN/eval_dev/val_calibration.json"
  echo "Calibration status: $(json_get "$NEW_RUN/eval_dev/val_calibration.json" status)"
}

get_v25_budget(){
  local p="$NEW_RUN/eval_dev/val_calibration.json"
  need_file "$p"
  "$PYTHON_BIN" - "$p" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
status=str(x.get('status',''))
if status not in {'constraints_satisfied','ok','success'}:
    raise SystemExit('Calibration is not constraint-satisfied: status='+status+'. Do not call this a calibrated operating point.')
for key in ('selected_bcot_risk_budget','selected_budget','bcot_risk_budget'):
    if key in x:
        print(float(x[key])); break
else:
    # Some historical calibrator versions place the chosen point in selected.
    sel=x.get('selected',{})
    for key in ('bcot_risk_budget','budget'):
        if key in sel:
            print(float(sel[key])); break
    else: raise SystemExit('Could not locate selected BCOT budget in calibration JSON')
PY
}

eval_v25_dev(){
  require_mcfc_promotion
  local ckpt="${NEW_CKPT:-$NEW_RUN/cowp_all_best.pt}"
  need_file "$ckpt"
  need_dir "$NEW_ROOT/tensor_cache_heldout_test_waymax"
  calibrate_v25
  local budget
  budget="$(get_v25_budget)"
  echo "Using validation-calibrated BCOT budget: $budget"

  "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
    --data-config configs/data.yaml --label-config "$LABEL_CFG_NEW" --eval-config "$EVAL_CFG" \
    --mode learned_offline --cache-dir "$NEW_ROOT/tensor_cache_heldout_test_waymax" --checkpoint "$ckpt" \
    --methods cowp,conventional_safety,planner_score_only,soft_burden_cost_only,universal_ncf,outcome_oracle \
    --ncf-gate-mode priority --witness-threshold "$WITNESS_THRESHOLD" --bcot-risk-budget "$budget" \
    --batch-size 8 --device cuda:0 --num-workers 2 --prefetch-factor 1 \
    --output "$NEW_RUN/eval_dev/heldout_internal_methods.json"

  # This development heldout was already inspected; use it only to diagnose
  # online/offline parity and physical-safety regressions, not as the final claim.
  STRICT_ID_FILE="$NEW_ROOT/heldout_test_scene_ids.txt" \
  OLD_ROOT="$NEW_ROOT" OLD_RUN="$NEW_RUN/eval_dev" OLD_CKPT="$ckpt" \
  LABEL_CFG_OLD="$LABEL_CFG_NEW" BCOT_BUDGET_V24="$budget" strict_v24_dev
}

build_final_blind(){
  # Science guard: this stage intentionally exists but should only be called
  # after the model/checkpoint-selection rule and hyperparameters are frozen.
  need_womd
  need_file "$FINAL_IDS"
  local scenario_val="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
  local tfexample_val="$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*"
  local index_val="$WOMD_ROOT/.cowp_v131_indices/scenario_location_index_validation.jsonl"
  mkdir -p "$FINAL_ROOT/labels" "$FINAL_ROOT/tensor_cache"
  "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
    --data-config configs/data.yaml --label-config "$LABEL_CFG_NEW" --proto-glob "$scenario_val" \
    --output-dir "$FINAL_ROOT/labels" --allow-scenario-ids "$FINAL_IDS" \
    --index-jsonl "$index_val" --require-all-allowed-resolved \
    --num-workers "${FINAL_LABEL_WORKERS:-32}" --start-method forkserver --max-pending-multiplier 2 \
    --no-compress --skip-diagnostics --cpu-only --profile-jsonl "$FINAL_ROOT/profile_labels.jsonl"
  "$PYTHON_BIN" -m cowp.scripts.02_build_tensor_cache \
    --data-config configs/data.yaml --split validation --tfexample-glob "$tfexample_val" \
    --labels-dir "$FINAL_ROOT/labels" --output-dir "$FINAL_ROOT/tensor_cache" \
    --num-workers "${FINAL_CACHE_WORKERS:-8}" --start-method forkserver --parallel-scan \
    --require-waymax-ready --require-sdc-paths --require-all-labels-matched --no-compress --cpu-only \
    --profile-jsonl "$FINAL_ROOT/profile_tensor_cache.jsonl"
}

eval_final_blind(){
  need_womd
  # Run exactly once after freeze.  Do not tune anything from these outputs.
  local ckpt="${NEW_CKPT:-$NEW_RUN/cowp_all_best.pt}"
  local budget="${FINAL_BCOT_BUDGET:-}"
  need_file "$ckpt"; need_file "$FINAL_IDS"; need_dir "$FINAL_ROOT/tensor_cache"
  if [[ -z "$budget" ]]; then budget="$(get_v25_budget)"; fi
  mkdir -p "$NEW_RUN/final_blind"

  "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
    --data-config configs/data.yaml --label-config "$LABEL_CFG_NEW" --eval-config "$EVAL_CFG" \
    --mode learned_offline --cache-dir "$FINAL_ROOT/tensor_cache" --checkpoint "$ckpt" \
    --methods cowp,conventional_safety,planner_score_only,soft_burden_cost_only,universal_ncf,outcome_oracle \
    --ncf-gate-mode priority --witness-threshold "$WITNESS_THRESHOLD" --bcot-risk-budget "$budget" \
    --batch-size 8 --device cuda:0 --num-workers 2 --prefetch-factor 1 \
    --output "$NEW_RUN/final_blind/learned_internal_methods.json"

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  local -a idx_args=()
  if [[ -n "${TFEXAMPLE_INDEX_VAL:-}" && -f "${TFEXAMPLE_INDEX_VAL}" ]]; then
    idx_args=(--tfexample-index-jsonl "$TFEXAMPLE_INDEX_VAL")
  fi
  local methods=(cowp conventional_safety planner_score_only soft_burden_cost_only universal_ncf idm_lattice)
  for method in "${methods[@]}"; do
    "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
      --data-config configs/data.yaml --label-config "$LABEL_CFG_NEW" --eval-config "$EVAL_CFG" \
      --mode waymax --method "$method" --checkpoint "$ckpt" --device cuda:0 \
      --waymax-device gpu --jax-visible-devices "${JAX_VISIBLE_DEVICES:-1}" --jax-preallocate false \
      --waymax-split validation --tfexample-glob "$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*" \
      --scenario-ids-file "$FINAL_IDS" "${idx_args[@]}" \
      --rollout-horizon-steps 80 --waymax-action-mode absolute_xy_yaw \
      --ncf-gate-mode priority --witness-threshold "$WITNESS_THRESHOLD" --bcot-risk-budget "$budget" \
      --waymax-standard-metrics \
      --waymax-standard-metric-names OverlapMetric,OffroadMetric,ProgressionMetric,KinematicsInfeasibilityMetric \
      --reuse-waymax-env --prefilter-waymax-shards --jit-waymax-env --jit-waymax-metrics \
      --status-every 20 --no-progress --output "$NEW_RUN/final_blind/waymax_${method}.json"
  done
}

case "$STAGE" in
  strict_v24_dev) strict_v24_dev ;;
  learned_v24_diagnostics) learned_v24_diagnostics ;;
  proposal_probe) proposal_probe ;;
  freeze_final_split) freeze_final_split ;;
  full_rebuild) full_rebuild ;;
  attach_waymax) attach_waymax ;;
  train_v25) train_v25 ;;
  calibrate_v25) calibrate_v25 ;;
  eval_v25_dev) eval_v25_dev ;;
  build_final_blind) build_final_blind ;;
  eval_final_blind) eval_final_blind ;;
  help|*)
    cat <<EOF
Stages:
  strict_v24_dev          exact-ID strict Waymax diagnosis on current inspected heldout
  learned_v24_diagnostics wider validation sweeps + internal learned-offline diagnostics
  proposal_probe          600-scene validation-only MCFC source-attributed promotion probe
  freeze_final_split      create content-blind final IDs (ID hash only; no scenario decoding)
  full_rebuild            gated v16.8.25 5k/1k/1.2k development-data rebuild
  attach_waymax           gated cached candidate Waymax outcome attachment
  train_v25               gated training; set SEED=2026/2027/2028 for three independent runs
  calibrate_v25           validation-only BCOT calibration
  eval_v25_dev            diagnostic heldout learned + strict Waymax (not final blind evidence)
  build_final_blind       decode/build the frozen final blind set only after method freeze
  eval_final_blind        one-shot final learned + strict Waymax evaluation
EOF
    ;;
esac
