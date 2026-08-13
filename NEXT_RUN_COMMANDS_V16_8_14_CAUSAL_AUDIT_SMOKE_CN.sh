#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}" TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}" MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export OLD_VAL_CACHE="${OLD_VAL_CACHE:-$COWP_ROOT/tensor_cache_val}"
export SMOKE_ROOT="${SMOKE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_14_support_smoke}"
export HARD_COUNT="${HARD_COUNT:-48}" RANDOM_COUNT="${RANDOM_COUNT:-48}" MIN_HARD_COUNT="${MIN_HARD_COUNT:-24}" LABEL_WORKERS="${LABEL_WORKERS:-24}" FORCE_REBUILD_SMOKE="${FORCE_REBUILD_SMOKE:-1}" SEED="${SEED:-2026}"
export PREV_FRESH_CACHE="${PREV_FRESH_CACHE:-}"

SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
LABEL_CFG="configs/label_cowp_v16_8.yaml"
FRESH_LABELS="$SMOKE_ROOT/labels_val_v16_8_14"
PROFILE="$SMOKE_ROOT/fresh_profile.jsonl"
PROFILE_SUMMARY="$SMOKE_ROOT/fresh_profile_summary.json"
HARD_IDS="$SMOKE_ROOT/hard_scene_ids.txt"; RANDOM_IDS="$SMOKE_ROOT/random_scene_ids.txt"; UNION_IDS="$SMOKE_ROOT/union_scene_ids.txt"
OLD_CEILING="$SMOKE_ROOT/old_val_proposal_ceiling.json"; MANIFEST_AUDIT="$SMOKE_ROOT/probe_manifest_audit.json"
PAIRED="$SMOKE_ROOT/paired_probe.json"; ABLATION="$SMOKE_ROOT/proposal_source_ablation.json"; AUDIT="$SMOKE_ROOT/causal_audit_diagnostic.json"
SUPERVISION="$SMOKE_ROOT/training_supervision_audit.json"; MODEL_SUPPORT="$SMOKE_ROOT/model_support_audit.json"; NATURAL_DIAG="$SMOKE_ROOT/natural_support_diagnostic.json"
BASE_SCREEN="$SMOKE_ROOT/base_screen_verdict.json"; VERDICT="$SMOKE_ROOT/v16_8_14_smoke_verdict.json"; FP_FILE="$SMOKE_ROOT/v16_8_14_code_fingerprint.sha256"
mkdir -p "$SMOKE_ROOT/logs"

SMOKE_TOTAL=$((HARD_COUNT + RANDOM_COUNT))
[[ -d "$OLD_VAL_CACHE" ]] || { echo "missing OLD_VAL_CACHE=$OLD_VAL_CACHE" >&2; exit 2; }
# v16.8.14: derive the stress manifest from the CURRENT old validation cache.
# Never depend on a historical v16.8.8 smoke manifest; hard-scene prevalence is
# an algorithm/cache property, not a fixed WOMD split cardinality.
"$PYTHON_BIN" -m cowp.scripts.45_diagnose_proposal_ceiling \
  --cache-dir "$OLD_VAL_CACHE" --output "$OLD_CEILING" \
  --hard-scene-ids "$HARD_IDS" --hard-count "$HARD_COUNT" \
  --random-scene-ids "$RANDOM_IDS" --random-count "$RANDOM_COUNT" \
  --probe-total-count "$SMOKE_TOTAL" --random-exclude-hard-probe --control-count 0 --seed "$SEED"
cat "$HARD_IDS" "$RANDOM_IDS" | awk 'NF && !seen[$1]++{print $1}' > "$UNION_IDS"
"$PYTHON_BIN" -m cowp.scripts.63_validate_probe_manifest \
  --ceiling-json "$OLD_CEILING" --hard-ids "$HARD_IDS" --random-ids "$RANDOM_IDS" \
  --union-ids "$UNION_IDS" --expected-hard -1 --expected-random -1 \
  --expected-total "$SMOKE_TOTAL" --min-hard "$MIN_HARD_COUNT" \
  --output "$MANIFEST_AUDIT"

CODE_FP="$($PYTHON_BIN - <<'PY'
from pathlib import Path
from importlib import import_module
print(import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol').current_fingerprint(Path.cwd()))
PY
)"
if [[ "$FORCE_REBUILD_SMOKE" == "1" ]]; then
  rm -rf "$FRESH_LABELS"
  rm -f "$PROFILE" "$PROFILE_SUMMARY" "$PAIRED" "$ABLATION" "$AUDIT" "$SUPERVISION" "$MODEL_SUPPORT" "$NATURAL_DIAG" "$BASE_SCREEN" "$VERDICT" "$FP_FILE"
fi
mkdir -p "$FRESH_LABELS"
EXISTING="$(find "$FRESH_LABELS" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
if [[ "$EXISTING" -gt 0 ]]; then
  [[ -s "$FP_FILE" && "$(tr -d '[:space:]' < "$FP_FILE")" == "$CODE_FP" ]] || { echo "smoke label/code fingerprint mismatch; use FORCE_REBUILD_SMOKE=1" >&2; exit 3; }
else
  printf '%s\n' "$CODE_FP" > "$FP_FILE"
fi

run(){ local n="$1"; shift; echo "[$n] $*"; "$@" > >(tee "$SMOKE_ROOT/logs/$n.log") 2> >(tee -a "$SMOKE_ROOT/logs/$n.log" >&2); }
run_semantic(){ local n="$1"; shift; echo "[$n] $*"; "$@" > >(tee "$SMOKE_ROOT/logs/$n.log") 2> >(tee -a "$SMOKE_ROOT/logs/$n.log" >&2); }

run build_fresh "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto --data-config configs/data.yaml --label-config "$LABEL_CFG" --proto-glob "$SCENARIO_VAL" --output-dir "$FRESH_LABELS" --allow-scenario-ids "$UNION_IDS" --num-workers "$LABEL_WORKERS" --start-method forkserver --max-pending-multiplier 2 --no-compress --skip-existing --profile-jsonl "$PROFILE" --skip-diagnostics --cpu-only
run summarize "$PYTHON_BIN" -m cowp.scripts.49_summarize_label_build_profile --input "$PROFILE" --output "$PROFILE_SUMMARY" --top-slow 20
run natural_support "$PYTHON_BIN" -m cowp.scripts.68_summarize_natural_support_diagnostics --input "$PROFILE" --output "$NATURAL_DIAG"
run compare "$PYTHON_BIN" -m cowp.scripts.46_compare_proposal_probe --old-cache "$OLD_VAL_CACHE" --new-cache "$FRESH_LABELS" --representative-scene-ids "$RANDOM_IDS" --hard-scene-ids "$HARD_IDS" --new-build-profile "$PROFILE" --output "$PAIRED" --min-overall-any-valid 0.99 --min-overall-any-ncf 0.30 --max-false-safe-floor 0.65 --max-pbtr-floor 0.50 --min-hard-recovery 0.12 --max-rmr-target-tta-error-s 0.20
run source_ablation "$PYTHON_BIN" -m cowp.scripts.50_ablate_proposal_sources --cache-dir "$FRESH_LABELS" --output "$ABLATION"
run audit "$PYTHON_BIN" -m cowp.scripts.57_diagnose_causal_audit --cache-dir "$FRESH_LABELS" --scene-ids "$UNION_IDS" --output "$AUDIT"
set +e
run_semantic supervision "$PYTHON_BIN" -m cowp.scripts.62_audit_training_supervision --cache-dir "$FRESH_LABELS" --sample-scenes 0 --min-class-examples 8 --output "$SUPERVISION" --strict; SUP_RC=$?
run_semantic model_support "$PYTHON_BIN" -m cowp.scripts.65_audit_model_support --cache-dir "$FRESH_LABELS" --sample-scenes 0 --min-class-examples 8 --min-source-examples 8 --max-unauditable-critical-rate 0.01 --min-certificate-complete-scene-rate 0.98 --output "$MODEL_SUPPORT" --strict; MS_RC=$?
run_semantic screen "$PYTHON_BIN" -m cowp.scripts.58_screen_v16_8_9_causal_audit_probe --paired-probe "$PAIRED" --source-ablation "$ABLATION" --profile-summary "$PROFILE_SUMMARY" --audit-diagnostic "$AUDIT" --output "$BASE_SCREEN"; SCREEN_RC=$?
set -e

"$PYTHON_BIN" - "$VERDICT" "$CODE_FP" "$BASE_SCREEN" "$SUPERVISION" "$MODEL_SUPPORT" "$NATURAL_DIAG" "$SUP_RC" "$MS_RC" "$SCREEN_RC" <<'PY'
import json,sys
from pathlib import Path
out,fp,base_p,sup_p,ms_p,nat_p,sup_rc,ms_rc,screen_rc=sys.argv[1:]
def load(p):
    try: return json.load(open(p,encoding='utf-8'))
    except Exception: return {}
base,sup,ms,nat=map(load,(base_p,sup_p,ms_p,nat_p))
max_unauditable=0.01
checks={
  'proposal_causal_screen_pass': int(screen_rc)==0,
  'training_supervision_pass': bool(sup.get('pass',False)),
  'model_support_pass': bool(ms.get('pass',False)),
  'auditability_coverage': float(nat.get('mechanism_unauditable_rate',1.0)) <= max_unauditable,
  'natural_rootless_zero_on_auditable': int(nat.get('rootless_critical_agents',-1))==0,
  'natural_lt2_low_burden_zero_on_auditable': int(nat.get('critical_agents_with_lt2_low_burden_roots',-1))==0,
  'protected_prio_complete': int(nat.get('protected_without_prio_root',-1))==0,
}
passed=all(checks.values())
payload={
 'schema_version':'cowp_v16_8_14_smoke_composite_verdict_v2', 'code_fingerprint_sha256':fp,
 'pass':passed, 'recommend_strict_probe':passed, 'recommend_full_rebuild':False,
 'checks':checks, 'semantic_returncodes':{'supervision':int(sup_rc),'model_support':int(ms_rc),'screen':int(screen_rc)},
 'base_screen':base,
 'natural_support':{k:nat.get(k) for k in ('critical_agents_selected','mechanism_auditable_critical_agents','mechanism_unauditable_critical_agents','mechanism_unauditable_rate','rootless_critical_agents','critical_agents_with_lt2_low_burden_roots','protected_without_prio_root','empirical_corridor_roots','rootless_dominant_rejection','lt2_low_burden_dominant_rejection','priority_rejection_reason_counts','map_rejected_min_distance_summary_m','map_rejected_max_distance_summary_m')},
 'next_action':'Run strict only after this composite smoke verdict passes.' if passed else 'Do not run strict/full rebuild; inspect natural_support_diagnostic.json and the semantic audit JSONs.',
}
Path(out).write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(payload,indent=2,ensure_ascii=False))
PY

echo "SMOKE_VERDICT=$VERDICT"
if "$PYTHON_BIN" - "$VERDICT" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1],encoding='utf-8')).get('recommend_strict_probe') else 2)
PY
then
  echo "SMOKE PASS: proceed to fastpath-ab, then strict. Do NOT full-rebuild yet."
else
  echo "SMOKE FAIL: verdict was written; do NOT strict/full-rebuild."
  exit 2
fi
