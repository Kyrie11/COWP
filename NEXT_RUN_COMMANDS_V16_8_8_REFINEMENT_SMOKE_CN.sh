#!/usr/bin/env bash
set -euo pipefail

# v16.8.8 fast screen: fix candidate-bank-dependent critical-agent selection,
# preserve the base bank, and test Priority-Smooth-Yield (PSY) on a small subset.
# PASS only authorizes the strict 400+800 probe, never a full rebuild.
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
export SOURCE_PROBE_ROOT="${SOURCE_PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment_micro_probe}"
export SMOKE_ROOT="${SMOKE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_8_refinement_smoke}"
export HARD_COUNT="${HARD_COUNT:-48}"
export RANDOM_COUNT="${RANDOM_COUNT:-48}"
export LABEL_WORKERS="${LABEL_WORKERS:-24}"
export FORCE_REBUILD_SMOKE="${FORCE_REBUILD_SMOKE:-1}"

SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
LABEL_CFG="configs/label_cowp_v16_8.yaml"
FRESH_LABELS="$SMOKE_ROOT/labels_val_v16_8_8"
PROFILE="$SMOKE_ROOT/fresh_profile.jsonl"
PROFILE_SUMMARY="$SMOKE_ROOT/fresh_profile_summary.json"
HARD_IDS="$SMOKE_ROOT/hard_scene_ids.txt"
RANDOM_IDS="$SMOKE_ROOT/random_scene_ids.txt"
UNION_IDS="$SMOKE_ROOT/union_scene_ids.txt"
PAIRED="$SMOKE_ROOT/paired_probe.json"
ABLATION="$SMOKE_ROOT/proposal_source_ablation.json"
SCREEN="$SMOKE_ROOT/v16_8_8_smoke_verdict.json"

mkdir -p "$SMOKE_ROOT/logs"
[[ -s "$SOURCE_PROBE_ROOT/hard_scene_ids.txt" ]] || { echo "missing source hard ids: $SOURCE_PROBE_ROOT/hard_scene_ids.txt" >&2; exit 2; }
[[ -s "$SOURCE_PROBE_ROOT/representative_random_scene_ids.txt" ]] || { echo "missing source random ids: $SOURCE_PROBE_ROOT/representative_random_scene_ids.txt" >&2; exit 2; }
head -n "$HARD_COUNT" "$SOURCE_PROBE_ROOT/hard_scene_ids.txt" > "$HARD_IDS"
head -n "$RANDOM_COUNT" "$SOURCE_PROBE_ROOT/representative_random_scene_ids.txt" > "$RANDOM_IDS"
cat "$HARD_IDS" "$RANDOM_IDS" | awk 'NF && !seen[$1]++ {print $1}' > "$UNION_IDS"

if [[ "$FORCE_REBUILD_SMOKE" == "1" ]]; then
  rm -rf "$FRESH_LABELS"
  rm -f "$PROFILE"
fi
mkdir -p "$FRESH_LABELS"

# Never resume a smoke label directory under changed proposal/certificate code.
# Mixing files across critical-set definitions would recreate exactly the
# non-monotone dataset bug this revision is designed to diagnose.
CODE_FP="$($PYTHON_BIN - <<'PYFP'
import importlib
from pathlib import Path
m=importlib.import_module('cowp.scripts.53_gate_fresh_v16_8_6_cache_protocol')
print(m.current_fingerprint(Path.cwd()))
PYFP
)"
CODE_FP_FILE="$SMOKE_ROOT/v16_8_8_code_fingerprint.sha256"
EXISTING_LABELS="$(find "$FRESH_LABELS" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
if [[ "$EXISTING_LABELS" -gt 0 ]]; then
  [[ -s "$CODE_FP_FILE" ]] || { echo "existing v16.8.8 smoke labels lack code fingerprint; refuse mixed resume" >&2; exit 3; }
  [[ "$(tr -d '[:space:]' < "$CODE_FP_FILE")" == "$CODE_FP" ]] || { echo "smoke label/code fingerprint mismatch; use FORCE_REBUILD_SMOKE=1" >&2; exit 3; }
else
  printf '%s\n' "$CODE_FP" > "$CODE_FP_FILE"
fi

run() {
  local name="$1"; shift
  echo "[$name] $*"
  "$@" > >(tee "$SMOKE_ROOT/logs/${name}.log") 2> >(tee -a "$SMOKE_ROOT/logs/${name}.log" >&2)
}

run build_fresh "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" \
  --proto-glob "$SCENARIO_VAL" --output-dir "$FRESH_LABELS" \
  --allow-scenario-ids "$UNION_IDS" \
  --num-workers "$LABEL_WORKERS" --start-method forkserver \
  --max-pending-multiplier 2 --no-compress --skip-existing \
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

run screen "$PYTHON_BIN" -m cowp.scripts.56_screen_v16_8_8_refinement_probe \
  --paired-probe "$PAIRED" --source-ablation "$ABLATION" \
  --profile-summary "$PROFILE_SUMMARY" --output "$SCREEN"

echo "SCREEN=$SCREEN"
echo "PASS => run NEXT_RUN_COMMANDS_V16_8_8_STRICT_PROPOSAL_PROBE_CN.sh. Do NOT full-rebuild from the smoke alone."
