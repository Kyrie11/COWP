#!/usr/bin/env bash
set -euo pipefail

export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export OLD_VAL_CACHE="${OLD_VAL_CACHE:-$COWP_ROOT/tensor_cache_val}"
export PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_8_refinement_strict_probe}"
export HARD_COUNT="${HARD_COUNT:-400}"
export RANDOM_COUNT="${RANDOM_COUNT:-800}"
export LABEL_WORKERS="${LABEL_WORKERS:-24}"
export SEED="${SEED:-2026}"
export FORCE_REBUILD_PROBE="${FORCE_REBUILD_PROBE:-1}"

FRESH_LABELS="$PROBE_ROOT/labels_val_bcs_rmr_bcte"
mkdir -p "$PROBE_ROOT"
if [[ "$FORCE_REBUILD_PROBE" == "1" ]]; then
  rm -f "$PROBE_ROOT/v16_8_8_code_fingerprint.sha256"
fi
CODE_FP="$(python - <<'PYFP'
import importlib
from pathlib import Path
m=importlib.import_module('cowp.scripts.53_gate_fresh_v16_8_6_cache_protocol')
print(m.current_fingerprint(Path.cwd()))
PYFP
)"
CODE_FP_FILE="$PROBE_ROOT/v16_8_8_code_fingerprint.sha256"
EXISTING_LABELS=0
if [[ -d "$FRESH_LABELS" ]]; then
  EXISTING_LABELS="$(find "$FRESH_LABELS" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
fi
if [[ "$EXISTING_LABELS" -gt 0 && "$FORCE_REBUILD_PROBE" != "1" ]]; then
  [[ -s "$CODE_FP_FILE" ]] || { echo "existing strict labels predate v16.8.8 lineage guard; refuse resume" >&2; exit 3; }
  [[ "$(tr -d '[:space:]' < "$CODE_FP_FILE")" == "$CODE_FP" ]] || { echo "strict label/code fingerprint mismatch; do not mix proposal semantics" >&2; exit 3; }
else
  printf '%s\n' "$CODE_FP" > "$CODE_FP_FILE"
fi

bash NEXT_RUN_COMMANDS_V16_8_4_PROPOSAL_PROBE_CN.sh

ABLATION="$PROBE_ROOT/proposal_source_ablation_v16_8_8.json"
SCREEN="$PROBE_ROOT/v16_8_8_strict_verdict.json"
python -m cowp.scripts.50_ablate_proposal_sources \
  --cache-dir "$PROBE_ROOT/labels_val_bcs_rmr_bcte" --output "$ABLATION"
python -m cowp.scripts.56_screen_v16_8_8_refinement_probe \
  --paired-probe "$PROBE_ROOT/paired_proposal_probe.json" \
  --source-ablation "$ABLATION" \
  --profile-summary "$PROBE_ROOT/fresh_probe_profile_summary.json" \
  --output "$SCREEN" --strict
python - "$SCREEN" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
if not p.get('recommend_full_rebuild',False):
    raise SystemExit('STRICT PROBE FAILED: do not full-rebuild.')
print('STRICT PROBE PASSED: full fresh v16.8.8 rebuild is justified.')
PY
