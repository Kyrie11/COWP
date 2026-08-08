#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}" TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}" MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export OLD_VAL_CACHE="${OLD_VAL_CACHE:-$COWP_ROOT/tensor_cache_val}"
export PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_strict_probe}"
export HARD_COUNT="${HARD_COUNT:-400}" RANDOM_COUNT="${RANDOM_COUNT:-800}" LABEL_WORKERS="${LABEL_WORKERS:-24}" SEED="${SEED:-2026}" FORCE_REBUILD_PROBE="${FORCE_REBUILD_PROBE:-1}"
SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
FRESH="$PROBE_ROOT/labels_val_v16_8_9"; PROFILE="$PROBE_ROOT/fresh_probe_profile.jsonl"; SUMMARY="$PROBE_ROOT/fresh_probe_profile_summary.json"
HARD="$PROBE_ROOT/hard_scene_ids.txt"; RANDOM="$PROBE_ROOT/representative_random_scene_ids.txt"; UNION="$PROBE_ROOT/probe_union_scene_ids.txt"
PAIRED="$PROBE_ROOT/paired_proposal_probe.json"; ABL="$PROBE_ROOT/proposal_source_ablation.json"; AUDIT="$PROBE_ROOT/causal_audit_diagnostic.json"; VERDICT="$PROBE_ROOT/v16_8_9_strict_verdict.json"
mkdir -p "$PROBE_ROOT/logs"
if [[ "$FORCE_REBUILD_PROBE" == 1 ]]; then rm -rf "$FRESH"; rm -f "$PROFILE"; fi; mkdir -p "$FRESH"
CODE_FP="$($PYTHON_BIN - <<'PY'
from pathlib import Path
from importlib import import_module
from pathlib import Path
current_fingerprint = import_module("cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol").current_fingerprint
print(current_fingerprint(Path.cwd()))
PY
)"; FP="$PROBE_ROOT/v16_8_9_code_fingerprint.sha256"
EXISTING="$(find "$FRESH" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
if [[ "$EXISTING" -gt 0 ]]; then [[ -s "$FP" && "$(tr -d '[:space:]' < "$FP")" == "$CODE_FP" ]] || { echo 'strict probe lineage mismatch; rebuild in a fresh PROBE_ROOT' >&2; exit 3; }; else printf '%s\n' "$CODE_FP" > "$FP"; fi
run(){ local n="$1"; shift; echo "[$n] $*"; "$@" > >(tee "$PROBE_ROOT/logs/$n.log") 2> >(tee -a "$PROBE_ROOT/logs/$n.log" >&2); }
run old_ceiling "$PYTHON_BIN" -m cowp.scripts.45_diagnose_proposal_ceiling --cache-dir "$OLD_VAL_CACHE" --output "$PROBE_ROOT/current_proposal_ceiling.json" --hard-scene-ids "$HARD" --hard-count "$HARD_COUNT" --random-scene-ids "$RANDOM" --random-count "$RANDOM_COUNT" --control-count 0 --seed "$SEED"
cat "$HARD" "$RANDOM" | awk 'NF && !seen[$1]++{print $1}' > "$UNION"
run build "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto --data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml --proto-glob "$SCENARIO_VAL" --output-dir "$FRESH" --allow-scenario-ids "$UNION" --num-workers "$LABEL_WORKERS" --start-method forkserver --max-pending-multiplier 2 --no-compress --skip-existing --profile-jsonl "$PROFILE" --skip-diagnostics --cpu-only
run summarize "$PYTHON_BIN" -m cowp.scripts.49_summarize_label_build_profile --input "$PROFILE" --output "$SUMMARY" --top-slow 30
run compare "$PYTHON_BIN" -m cowp.scripts.46_compare_proposal_probe --old-cache "$OLD_VAL_CACHE" --new-cache "$FRESH" --representative-scene-ids "$RANDOM" --hard-scene-ids "$HARD" --new-build-profile "$PROFILE" --output "$PAIRED" --min-overall-any-valid 0.99 --min-overall-any-ncf 0.40 --max-false-safe-floor 0.55 --max-pbtr-floor 0.45 --min-hard-recovery 0.20 --max-rmr-target-tta-error-s 0.20
run source_ablation "$PYTHON_BIN" -m cowp.scripts.50_ablate_proposal_sources --cache-dir "$FRESH" --output "$ABL"
run audit "$PYTHON_BIN" -m cowp.scripts.57_diagnose_causal_audit --cache-dir "$FRESH" --scene-ids "$UNION" --output "$AUDIT"
set +e
run screen "$PYTHON_BIN" -m cowp.scripts.58_screen_v16_8_9_causal_audit_probe --paired-probe "$PAIRED" --source-ablation "$ABL" --profile-summary "$SUMMARY" --audit-diagnostic "$AUDIT" --output "$VERDICT" --strict
STATUS=$?
set -e
if [[ "$STATUS" -eq 0 ]]; then echo "STRICT PASS: full rebuild is justified. VERDICT=$VERDICT"; else echo "STRICT FAIL: DO NOT FULL REBUILD. VERDICT=$VERDICT"; fi
exit "$STATUS"
