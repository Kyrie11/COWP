#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export OLD_VAL_CACHE="${OLD_VAL_CACHE:-$COWP_ROOT/tensor_cache_val}"
export SMOKE_ROOT="${SMOKE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_smoke}"
export FRESH_LABELS="${FRESH_LABELS:-$SMOKE_ROOT/labels_val_v16_8_9}"
PROFILE_SUMMARY="$SMOKE_ROOT/fresh_profile_summary.json"
HARD_IDS="$SMOKE_ROOT/hard_scene_ids.txt"
RANDOM_IDS="$SMOKE_ROOT/random_scene_ids.txt"
UNION_IDS="$SMOKE_ROOT/union_scene_ids.txt"
PAIRED="$SMOKE_ROOT/paired_probe.json"
ABLATION="$SMOKE_ROOT/proposal_source_ablation.json"
AUDIT="$SMOKE_ROOT/causal_audit_diagnostic.json"
SUPERVISION="$SMOKE_ROOT/training_supervision_audit.json"
SCREEN="$SMOKE_ROOT/v16_8_9_smoke_verdict.json"
REPAIR="$SMOKE_ROOT/v16_8_9_contract_repair.json"
[[ -d "$FRESH_LABELS" ]] || { echo "missing existing smoke labels: $FRESH_LABELS" >&2; exit 2; }
[[ -s "$HARD_IDS" && -s "$RANDOM_IDS" && -s "$UNION_IDS" ]] || { echo "missing smoke scene-id files under $SMOKE_ROOT" >&2; exit 2; }
run(){ local n="$1"; shift; echo "[$n] $*"; "$@" > >(tee "$SMOKE_ROOT/logs/$n.log") 2> >(tee -a "$SMOKE_ROOT/logs/$n.log" >&2); }
mkdir -p "$SMOKE_ROOT/logs"
run repair_contract "$PYTHON_BIN" -m cowp.scripts.61_repair_v16_8_9_audit_transport_contract \
  --cache-dir "$FRESH_LABELS" --label-config configs/label_cowp_v16_8.yaml --output "$REPAIR"
run compare "$PYTHON_BIN" -m cowp.scripts.46_compare_proposal_probe \
  --old-cache "$OLD_VAL_CACHE" --new-cache "$FRESH_LABELS" \
  --representative-scene-ids "$RANDOM_IDS" --hard-scene-ids "$HARD_IDS" \
  --new-build-profile "$SMOKE_ROOT/fresh_profile.jsonl" --output "$PAIRED" \
  --min-overall-any-valid 0.99 --min-overall-any-ncf 0.30 --max-false-safe-floor 0.65 \
  --max-pbtr-floor 0.50 --min-hard-recovery 0.12 --max-rmr-target-tta-error-s 0.20
run source_ablation "$PYTHON_BIN" -m cowp.scripts.50_ablate_proposal_sources --cache-dir "$FRESH_LABELS" --output "$ABLATION"
run audit "$PYTHON_BIN" -m cowp.scripts.57_diagnose_causal_audit --cache-dir "$FRESH_LABELS" --scene-ids "$UNION_IDS" --output "$AUDIT"
run supervision "$PYTHON_BIN" -m cowp.scripts.62_audit_training_supervision --cache-dir "$FRESH_LABELS" --sample-scenes 0 --min-class-examples 8 --output "$SUPERVISION"
# The repaired cache now matches the current code/data contract. Refresh the
# lineage fingerprint so subsequent diagnostics cannot confuse it with the
# pre-repair serialization contract.
CODE_FP="$($PYTHON_BIN - <<'PY'
from pathlib import Path
from importlib import import_module
current_fingerprint = import_module("cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol").current_fingerprint
print(current_fingerprint(Path.cwd()))
PY
)"
printf '%s\n' "$CODE_FP" > "$SMOKE_ROOT/v16_8_9_code_fingerprint.sha256"
set +e
run screen "$PYTHON_BIN" -m cowp.scripts.58_screen_v16_8_9_causal_audit_probe \
  --paired-probe "$PAIRED" --source-ablation "$ABLATION" --profile-summary "$PROFILE_SUMMARY" \
  --audit-diagnostic "$AUDIT" --output "$SCREEN"
STATUS=$?
set -e
echo "REPAIR_MANIFEST=$REPAIR"
echo "SMOKE_VERDICT=$SCREEN"
echo "CAUSAL_AUDIT=$AUDIT"
echo "TRAINING_SUPERVISION_AUDIT=$SUPERVISION"
if [[ "$STATUS" -eq 0 ]]; then
  echo "CONTRACT-REPAIRED SMOKE PASS: run NEXT_RUN_COMMANDS_V16_8_9_STRICT_PROPOSAL_PROBE_CN.sh. Do NOT full-rebuild yet."
else
  echo "CONTRACT-REPAIRED SMOKE FAIL: do NOT full-rebuild; inspect the updated integrity checks."
fi
exit "$STATUS"
