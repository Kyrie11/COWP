#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export OLD_VAL_CACHE="${OLD_VAL_CACHE:-$COWP_ROOT/tensor_cache_val}"
export SOURCE_PROBE_ROOT="${SOURCE_PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_8_refinement_smoke}"
export SMOKE_ROOT="${SMOKE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_smoke}"
export HARD_COUNT="${HARD_COUNT:-48}"
export RANDOM_COUNT="${RANDOM_COUNT:-48}"
export LABEL_WORKERS="${LABEL_WORKERS:-24}"
export FORCE_REBUILD_SMOKE="${FORCE_REBUILD_SMOKE:-1}"
export PREV_FRESH_CACHE="${PREV_FRESH_CACHE:-}"

SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
LABEL_CFG="configs/label_cowp_v16_8.yaml"
FRESH_LABELS="$SMOKE_ROOT/labels_val_v16_8_9"
PROFILE="$SMOKE_ROOT/fresh_profile.jsonl"
PROFILE_SUMMARY="$SMOKE_ROOT/fresh_profile_summary.json"
HARD_IDS="$SMOKE_ROOT/hard_scene_ids.txt"
RANDOM_IDS="$SMOKE_ROOT/random_scene_ids.txt"
UNION_IDS="$SMOKE_ROOT/union_scene_ids.txt"
PAIRED="$SMOKE_ROOT/paired_probe.json"
ABLATION="$SMOKE_ROOT/proposal_source_ablation.json"
AUDIT="$SMOKE_ROOT/causal_audit_diagnostic.json"
SUPERVISION="$SMOKE_ROOT/training_supervision_audit.json"
MODEL_SUPPORT="$SMOKE_ROOT/model_support_audit.json"
SCREEN="$SMOKE_ROOT/v16_8_9_smoke_verdict.json"

mkdir -p "$SMOKE_ROOT/logs"
SOURCE_HARD="$SOURCE_PROBE_ROOT/hard_scene_ids.txt"
SOURCE_RANDOM="$SOURCE_PROBE_ROOT/random_scene_ids.txt"
[[ -s "$SOURCE_RANDOM" ]] || SOURCE_RANDOM="$SOURCE_PROBE_ROOT/representative_random_scene_ids.txt"
[[ -s "$SOURCE_HARD" ]] || { echo "missing source hard ids: $SOURCE_HARD" >&2; exit 2; }
[[ -s "$SOURCE_RANDOM" ]] || { echo "missing source random ids under $SOURCE_PROBE_ROOT" >&2; exit 2; }
head -n "$HARD_COUNT" "$SOURCE_HARD" > "$HARD_IDS"
# Keep the smoke contract at exactly HARD_COUNT + RANDOM_COUNT distinct scenes,
# even when an older representative-random manifest happened to overlap the hard list.
awk 'NR==FNR{hard[$1]=1; next} NF && !hard[$1] && !seen[$1]++ {print $1}' "$HARD_IDS" "$SOURCE_RANDOM" | head -n "$RANDOM_COUNT" > "$RANDOM_IDS"
cat "$HARD_IDS" "$RANDOM_IDS" > "$UNION_IDS"
[[ "$(wc -l < "$HARD_IDS")" -eq "$HARD_COUNT" ]] || { echo "smoke hard manifest is shorter than HARD_COUNT=$HARD_COUNT" >&2; exit 3; }
[[ "$(wc -l < "$RANDOM_IDS")" -eq "$RANDOM_COUNT" ]] || { echo "smoke random manifest lacks $RANDOM_COUNT IDs after removing hard overlap" >&2; exit 3; }
[[ "$(sort -u "$UNION_IDS" | wc -l)" -eq $((HARD_COUNT + RANDOM_COUNT)) ]] || { echo "smoke manifest is not unique/disjoint" >&2; exit 3; }

if [[ "$FORCE_REBUILD_SMOKE" == "1" ]]; then
  rm -rf "$FRESH_LABELS"
  rm -f "$PROFILE"
fi
mkdir -p "$FRESH_LABELS"

CODE_FP="$($PYTHON_BIN - <<'PY'
from pathlib import Path
from importlib import import_module
from pathlib import Path
current_fingerprint = import_module("cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol").current_fingerprint
print(current_fingerprint(Path.cwd()))
PY
)"
FP_FILE="$SMOKE_ROOT/v16_8_9_code_fingerprint.sha256"
EXISTING="$(find "$FRESH_LABELS" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
if [[ "$EXISTING" -gt 0 ]]; then
  [[ -s "$FP_FILE" ]] || { echo "existing smoke labels lack v16.8.9 fingerprint" >&2; exit 3; }
  [[ "$(tr -d '[:space:]' < "$FP_FILE")" == "$CODE_FP" ]] || { echo "smoke label/code fingerprint mismatch; set FORCE_REBUILD_SMOKE=1" >&2; exit 3; }
else
  printf '%s\n' "$CODE_FP" > "$FP_FILE"
fi

run(){ local name="$1"; shift; echo "[$name] $*"; "$@" > >(tee "$SMOKE_ROOT/logs/${name}.log") 2> >(tee -a "$SMOKE_ROOT/logs/${name}.log" >&2); }

run build_fresh "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --proto-glob "$SCENARIO_VAL" --output-dir "$FRESH_LABELS" \
  --allow-scenario-ids "$UNION_IDS" --num-workers "$LABEL_WORKERS" \
  --start-method forkserver --max-pending-multiplier 2 --no-compress --skip-existing \
  --profile-jsonl "$PROFILE" --skip-diagnostics --cpu-only

run summarize "$PYTHON_BIN" -m cowp.scripts.49_summarize_label_build_profile \
  --input "$PROFILE" --output "$PROFILE_SUMMARY" --top-slow 20

run compare "$PYTHON_BIN" -m cowp.scripts.46_compare_proposal_probe \
  --old-cache "$OLD_VAL_CACHE" --new-cache "$FRESH_LABELS" \
  --representative-scene-ids "$RANDOM_IDS" --hard-scene-ids "$HARD_IDS" \
  --new-build-profile "$PROFILE" --output "$PAIRED" \
  --min-overall-any-valid 0.99 --min-overall-any-ncf 0.30 \
  --max-false-safe-floor 0.65 --max-pbtr-floor 0.50 \
  --min-hard-recovery 0.12 --max-rmr-target-tta-error-s 0.20

run source_ablation "$PYTHON_BIN" -m cowp.scripts.50_ablate_proposal_sources \
  --cache-dir "$FRESH_LABELS" --output "$ABLATION"

run audit "$PYTHON_BIN" -m cowp.scripts.57_diagnose_causal_audit \
  --cache-dir "$FRESH_LABELS" --scene-ids "$UNION_IDS" --output "$AUDIT"
run supervision "$PYTHON_BIN" -m cowp.scripts.62_audit_training_supervision \
  --cache-dir "$FRESH_LABELS" --sample-scenes 0 --min-class-examples 8 --output "$SUPERVISION" --strict

run model_support "$PYTHON_BIN" -m cowp.scripts.65_audit_model_support \
  --cache-dir "$FRESH_LABELS" --sample-scenes 0 --min-class-examples 8 --min-source-examples 8 \
  --output "$MODEL_SUPPORT" --strict

if [[ -n "$PREV_FRESH_CACHE" && -d "$PREV_FRESH_CACHE" ]]; then
  run compare_previous "$PYTHON_BIN" -m cowp.scripts.46_compare_proposal_probe \
    --old-cache "$PREV_FRESH_CACHE" --new-cache "$FRESH_LABELS" \
    --representative-scene-ids "$RANDOM_IDS" --hard-scene-ids "$HARD_IDS" \
    --new-build-profile "$PROFILE" --output "$SMOKE_ROOT/paired_vs_previous_fresh.json" \
    --min-overall-any-valid 0.0 --min-overall-any-ncf 0.0 \
    --max-false-safe-floor 1.0 --max-pbtr-floor 1.0 --min-hard-recovery 0.0 \
    --max-rmr-target-tta-error-s 1.0
fi

set +e
run screen "$PYTHON_BIN" -m cowp.scripts.58_screen_v16_8_9_causal_audit_probe \
  --paired-probe "$PAIRED" --source-ablation "$ABLATION" \
  --profile-summary "$PROFILE_SUMMARY" --audit-diagnostic "$AUDIT" --output "$SCREEN"
STATUS=$?
set -e

echo "SMOKE_VERDICT=$SCREEN"
echo "CAUSAL_AUDIT=$AUDIT"
echo "TRAINING_SUPERVISION_AUDIT=$SUPERVISION"
echo "MODEL_SUPPORT_AUDIT=$MODEL_SUPPORT"
if [[ "$STATUS" -eq 0 ]]; then
  echo "SMOKE PASS: run NEXT_RUN_COMMANDS_V16_8_9_STRICT_PROPOSAL_PROBE_CN.sh. Do NOT full-rebuild yet."
else
  echo "SMOKE FAIL: do NOT full-rebuild. Upload the four JSON diagnostics for another algorithm/data-contract iteration."
fi
exit "$STATUS"
