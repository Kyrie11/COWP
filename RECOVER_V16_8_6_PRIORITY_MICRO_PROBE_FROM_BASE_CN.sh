#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
OLD_VAL_CACHE="${OLD_VAL_CACHE:-$COWP_ROOT/tensor_cache_val}"
PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment_micro_probe}"
FRESH_LABELS="$PROBE_ROOT/labels_val_bcs_rmr_bcte"
HARD_IDS="$PROBE_ROOT/hard_scene_ids.txt"
RANDOM_IDS="$PROBE_ROOT/representative_random_scene_ids.txt"
PROFILE_JSONL="$PROBE_ROOT/fresh_probe_profile.jsonl"
COMPARE_JSON="$PROBE_ROOT/paired_proposal_probe.json"
SCREEN_JSON="$PROBE_ROOT/priority_commitment_micro_screen.json"
CURRENT_DIAG="$PROBE_ROOT/current_proposal_ceiling_from_tensor_cache_val.json"

[[ -d "$OLD_VAL_CACHE" ]] || { echo "missing clean base cache: $OLD_VAL_CACHE" >&2; exit 2; }
[[ -d "$FRESH_LABELS" ]] || { echo "missing already-built fresh labels: $FRESH_LABELS" >&2; exit 2; }
[[ -s "$HARD_IDS" && -s "$RANDOM_IDS" && -s "$PROFILE_JSONL" ]] || { echo "probe metadata/profile incomplete" >&2; exit 2; }

# Diagnose the same legacy proposal bank directly from tensor_cache_val.  The
# deleted tensor_cache_val_waymax only added waymax/* arrays, which are not used
# by proposal-ceiling or paired-probe metrics.
"$PYTHON_BIN" -m cowp.scripts.45_diagnose_proposal_ceiling \
  --cache-dir "$OLD_VAL_CACHE" --output "$CURRENT_DIAG" \
  --hard-count 0 --random-count 0 --control-count 0 --seed 2026

# Reuse the 191 already-built fresh labels.  Do NOT rerun label generation.
"$PYTHON_BIN" -m cowp.scripts.46_compare_proposal_probe \
  --old-cache "$OLD_VAL_CACHE" --new-cache "$FRESH_LABELS" \
  --representative-scene-ids "$RANDOM_IDS" --hard-scene-ids "$HARD_IDS" \
  --output "$COMPARE_JSON" --new-build-profile "$PROFILE_JSONL" \
  --min-overall-any-valid 0.99 --min-overall-any-ncf 0.40 \
  --max-false-safe-floor 0.55 --max-pbtr-floor 0.45 \
  --min-hard-recovery 0.20 --max-rmr-target-tta-error-s 0.20

# The micro screen has deliberately looser screening thresholds than the strict
# full-rebuild promotion gate.
"$PYTHON_BIN" -m cowp.scripts.52_screen_priority_commitment_probe \
  --paired-probe "$COMPARE_JSON" --output "$SCREEN_JSON"

echo "COMPARE_JSON=$COMPARE_JSON"
echo "MICRO_SCREEN=$SCREEN_JSON"
echo "No fresh label was rebuilt by this recovery script."
