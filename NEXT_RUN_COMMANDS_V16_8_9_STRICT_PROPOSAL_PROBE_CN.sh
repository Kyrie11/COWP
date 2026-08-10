#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}" TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}" MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export OLD_VAL_CACHE="${OLD_VAL_CACHE:-$COWP_ROOT/tensor_cache_val}"
export PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_strict_probe}"
export HARD_COUNT="${HARD_COUNT:-400}" RANDOM_COUNT="${RANDOM_COUNT:-800}" LABEL_WORKERS="${LABEL_WORKERS:-24}" SEED="${SEED:-2026}" FORCE_REBUILD_PROBE="${FORCE_REBUILD_PROBE:-0}"
SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
FRESH="$PROBE_ROOT/labels_val_v16_8_9"; PROFILE="$PROBE_ROOT/fresh_probe_profile.jsonl"; SUMMARY="$PROBE_ROOT/fresh_probe_profile_summary.json"
HARD_IDS_FILE="$PROBE_ROOT/hard_scene_ids.txt"; RANDOM_IDS_FILE="$PROBE_ROOT/representative_random_scene_ids.txt"; UNION_IDS_FILE="$PROBE_ROOT/probe_union_scene_ids.txt"
PAIRED="$PROBE_ROOT/paired_proposal_probe.json"; ABL="$PROBE_ROOT/proposal_source_ablation.json"; AUDIT="$PROBE_ROOT/causal_audit_diagnostic.json"; SUPERVISION="$PROBE_ROOT/training_supervision_audit.json"; MODEL_SUPPORT="$PROBE_ROOT/model_support_audit.json"; PROBE_MANIFEST="$PROBE_ROOT/probe_manifest_audit.json"; VERDICT="$PROBE_ROOT/v16_8_9_strict_verdict.json"
mkdir -p "$PROBE_ROOT/logs"
if [[ "$FORCE_REBUILD_PROBE" == 1 ]]; then
  rm -rf "$FRESH"
  rm -f "$PROFILE" "$SUMMARY" "$PAIRED" "$ABL" "$AUDIT" "$SUPERVISION" "$VERDICT" \
        "$HARD_IDS_FILE" "$RANDOM_IDS_FILE" "$UNION_IDS_FILE" "$MODEL_SUPPORT" "$PROBE_MANIFEST"
fi
mkdir -p "$FRESH"
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
emit_early_verdict(){
  local stage="$1" rc="$2" message="$3"
  "$PYTHON_BIN" - "$VERDICT" "$stage" "$rc" "$message" "$CODE_FP" <<'PYFAIL'
import json,sys
from pathlib import Path
out,stage,rc,msg,fp=sys.argv[1:]
payload={
  "schema_version":"cowp_v16_8_9_strict_wrapper_failure_v1",
  "strict":True,
  "screen_pass":False,
  "recommend_strict_probe":False,
  "recommend_full_rebuild":False,
  "code_fingerprint_sha256":fp,
  "failure_stage":stage,
  "failure_returncode":int(rc),
  "next_action":f"STRICT PIPELINE ERROR at {stage}: {msg}. Do not full-rebuild.",
}
Path(out).parent.mkdir(parents=True,exist_ok=True)
Path(out).write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(payload,indent=2,ensure_ascii=False))
PYFAIL
}
run(){
  local n="$1"; shift
  echo "[$n] $*"
  if "$@" > >(tee "$PROBE_ROOT/logs/$n.log") 2> >(tee -a "$PROBE_ROOT/logs/$n.log" >&2); then
    return 0
  else
    local rc=$?
    if [[ "$n" != "screen" ]]; then emit_early_verdict "$n" "$rc" "required stage failed"; fi
    return "$rc"
  fi
}
run old_ceiling "$PYTHON_BIN" -m cowp.scripts.45_diagnose_proposal_ceiling --cache-dir "$OLD_VAL_CACHE" --output "$PROBE_ROOT/current_proposal_ceiling.json" --hard-scene-ids "$HARD_IDS_FILE" --hard-count "$HARD_COUNT" --random-scene-ids "$RANDOM_IDS_FILE" --random-count "$RANDOM_COUNT" --random-exclude-hard-probe --control-count 0 --seed "$SEED"
[[ -s "$HARD_IDS_FILE" ]] || { emit_early_verdict "old_ceiling" 3 "hard-scene ID file was not produced"; exit 3; }
[[ -s "$RANDOM_IDS_FILE" ]] || { emit_early_verdict "old_ceiling" 3 "representative-random ID file was not produced"; exit 3; }
cat "$HARD_IDS_FILE" "$RANDOM_IDS_FILE" | awk 'NF && !seen[$1]++{print $1}' > "$UNION_IDS_FILE"
run manifest "$PYTHON_BIN" -m cowp.scripts.63_validate_probe_manifest --ceiling-json "$PROBE_ROOT/current_proposal_ceiling.json" --hard-ids "$HARD_IDS_FILE" --random-ids "$RANDOM_IDS_FILE" --union-ids "$UNION_IDS_FILE" --expected-hard "$HARD_COUNT" --expected-random "$RANDOM_COUNT" --output "$PROBE_MANIFEST"
run build "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto --data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml --proto-glob "$SCENARIO_VAL" --output-dir "$FRESH" --allow-scenario-ids "$UNION_IDS_FILE" --num-workers "$LABEL_WORKERS" --start-method forkserver --max-pending-multiplier 2 --no-compress --skip-existing --profile-jsonl "$PROFILE" --skip-diagnostics --cpu-only
run summarize "$PYTHON_BIN" -m cowp.scripts.49_summarize_label_build_profile --input "$PROFILE" --output "$SUMMARY" --top-slow 30
run compare "$PYTHON_BIN" -m cowp.scripts.46_compare_proposal_probe --old-cache "$OLD_VAL_CACHE" --new-cache "$FRESH" --representative-scene-ids "$RANDOM_IDS_FILE" --hard-scene-ids "$HARD_IDS_FILE" --new-build-profile "$PROFILE" --output "$PAIRED" --min-overall-any-valid 0.99 --min-overall-any-ncf 0.40 --max-false-safe-floor 0.55 --max-pbtr-floor 0.45 --min-hard-recovery 0.20 --max-rmr-target-tta-error-s 0.20
run source_ablation "$PYTHON_BIN" -m cowp.scripts.50_ablate_proposal_sources --cache-dir "$FRESH" --output "$ABL"
run audit "$PYTHON_BIN" -m cowp.scripts.57_diagnose_causal_audit --cache-dir "$FRESH" --scene-ids "$UNION_IDS_FILE" --output "$AUDIT"
run supervision "$PYTHON_BIN" -m cowp.scripts.62_audit_training_supervision --cache-dir "$FRESH" --sample-scenes 0 --min-class-examples 32 --output "$SUPERVISION" --strict
run model_support "$PYTHON_BIN" -m cowp.scripts.65_audit_model_support --cache-dir "$FRESH" --sample-scenes 0 --min-class-examples 32 --min-source-examples 32 --output "$MODEL_SUPPORT" --strict
set +e
run screen "$PYTHON_BIN" -m cowp.scripts.58_screen_v16_8_9_causal_audit_probe --paired-probe "$PAIRED" --source-ablation "$ABL" --profile-summary "$SUMMARY" --audit-diagnostic "$AUDIT" --output "$VERDICT" --strict
STATUS=$?
set -e
if [[ ! -s "$VERDICT" ]]; then emit_early_verdict "screen" "$STATUS" "screen stage exited without writing verdict"; fi
if [[ "$STATUS" -eq 0 ]]; then echo "STRICT PASS: full rebuild is justified. VERDICT=$VERDICT"; else echo "STRICT FAIL: DO NOT FULL REBUILD. VERDICT=$VERDICT"; fi
exit "$STATUS"
