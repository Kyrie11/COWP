#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}" TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}" MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export OLD_VAL_CACHE="${OLD_VAL_CACHE:-$COWP_ROOT/tensor_cache_val}"
export PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_18_support_strict_probe}"
export HARD_COUNT="${HARD_COUNT:-400}" RANDOM_COUNT="${RANDOM_COUNT:-800}" MIN_HARD_COUNT="${MIN_HARD_COUNT:-128}" LABEL_WORKERS="${LABEL_WORKERS:-24}" SEED="${SEED:-2026}" FORCE_REBUILD_PROBE="${FORCE_REBUILD_PROBE:-1}"
export INDEX_ROOT="${INDEX_ROOT:-$COWP_ROOT/womd_scenario_location_indexes_v1}"
VAL_SCENARIO_INDEX="$INDEX_ROOT/scenario_validation_locations.jsonl"
VAL_SCENARIO_INDEX_META="$INDEX_ROOT/scenario_validation_locations.meta.json"
SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
FRESH="$PROBE_ROOT/labels_val_v16_8_18"; PROFILE="$PROBE_ROOT/fresh_probe_profile.jsonl"; SUMMARY="$PROBE_ROOT/fresh_probe_profile_summary.json"
HARD_IDS_FILE="$PROBE_ROOT/hard_scene_ids.txt"; RANDOM_IDS_FILE="$PROBE_ROOT/representative_random_scene_ids.txt"; UNION_IDS_FILE="$PROBE_ROOT/probe_union_scene_ids.txt"
PAIRED="$PROBE_ROOT/paired_proposal_probe.json"; ABL="$PROBE_ROOT/proposal_source_ablation.json"; AUDIT="$PROBE_ROOT/causal_audit_diagnostic.json"; SUPERVISION="$PROBE_ROOT/training_supervision_audit.json"; MODEL_SUPPORT="$PROBE_ROOT/model_support_audit.json"; NATURAL_DIAG="$PROBE_ROOT/natural_support_diagnostic.json"; PROBE_MANIFEST="$PROBE_ROOT/probe_manifest_audit.json"
BASE_VERDICT="$PROBE_ROOT/base_screen_verdict.json"; VERDICT="$PROBE_ROOT/v16_8_18_strict_verdict.json"; FP="$PROBE_ROOT/v16_8_18_code_fingerprint.sha256"
SPARSE_INTEGRITY="$PROBE_ROOT/sparse_label_build_integrity.json"
PIPELINE_STATUS="$PROBE_ROOT/strict_pipeline_status.json"
mkdir -p "$PROBE_ROOT/logs"
CURRENT_STAGE="init"
PIPELINE_DONE=0
write_pipeline_failure(){
  local rc="$1"
  [[ "$PIPELINE_DONE" == "1" ]] && return 0
  "$PYTHON_BIN" - "$PIPELINE_STATUS" "$CURRENT_STAGE" "$rc" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
 'schema_version':'cowp_v16_8_18_strict_pipeline_status_v1',
 'pipeline_complete':False,'failure_stage':sys.argv[2],'return_code':int(sys.argv[3]),
 'composite_verdict_written':False,
 'next_action':'Repair/resume this pipeline failure before interpreting strict model-support results.'
},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
PY
}
trap 'rc=$?; write_pipeline_failure "$rc"' EXIT
CODE_FP="$($PYTHON_BIN - <<'PY'
from pathlib import Path
from importlib import import_module
print(import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol').current_fingerprint(Path.cwd()))
PY
)"
if [[ "$FORCE_REBUILD_PROBE" == "1" ]]; then rm -rf "$FRESH"; rm -f "$PROFILE" "$SUMMARY" "$PAIRED" "$ABL" "$AUDIT" "$SUPERVISION" "$MODEL_SUPPORT" "$NATURAL_DIAG" "$PROBE_MANIFEST" "$BASE_VERDICT" "$VERDICT" "$FP" "$HARD_IDS_FILE" "$RANDOM_IDS_FILE" "$UNION_IDS_FILE" "$SPARSE_INTEGRITY" "$PIPELINE_STATUS"; fi
mkdir -p "$FRESH"
EXISTING="$(find "$FRESH" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
if [[ "$EXISTING" -gt 0 ]]; then [[ -s "$FP" && "$(tr -d '[:space:]' < "$FP")" == "$CODE_FP" ]] || { echo 'strict probe fingerprint mismatch; rebuild in a fresh PROBE_ROOT' >&2; exit 3; }; else printf '%s\n' "$CODE_FP" > "$FP"; fi
run(){ local n="$1"; shift; echo "[$n] $*"; "$@" > >(tee "$PROBE_ROOT/logs/$n.log") 2> >(tee -a "$PROBE_ROOT/logs/$n.log" >&2); }
run_semantic(){ local n="$1"; shift; echo "[$n] $*"; "$@" > >(tee "$PROBE_ROOT/logs/$n.log") 2> >(tee -a "$PROBE_ROOT/logs/$n.log" >&2); }
PROBE_TOTAL=$((HARD_COUNT + RANDOM_COUNT))
run old_ceiling "$PYTHON_BIN" -m cowp.scripts.45_diagnose_proposal_ceiling --cache-dir "$OLD_VAL_CACHE" --output "$PROBE_ROOT/current_proposal_ceiling.json" --hard-scene-ids "$HARD_IDS_FILE" --hard-count "$HARD_COUNT" --random-scene-ids "$RANDOM_IDS_FILE" --random-count "$RANDOM_COUNT" --probe-total-count "$PROBE_TOTAL" --random-exclude-hard-probe --control-count 0 --seed "$SEED"
cat "$HARD_IDS_FILE" "$RANDOM_IDS_FILE" | awk 'NF && !seen[$1]++{print $1}' > "$UNION_IDS_FILE"
run manifest "$PYTHON_BIN" -m cowp.scripts.63_validate_probe_manifest --ceiling-json "$PROBE_ROOT/current_proposal_ceiling.json" --hard-ids "$HARD_IDS_FILE" --random-ids "$RANDOM_IDS_FILE" --union-ids "$UNION_IDS_FILE" --expected-hard -1 --expected-random -1 --expected-total "$PROBE_TOTAL" --min-hard "$MIN_HARD_COUNT" --output "$PROBE_MANIFEST"
CURRENT_STAGE="scenario_location_index"
run scenario_location_index "$PYTHON_BIN" -m cowp.scripts.72_build_scenario_location_index \
  --proto-glob "$SCENARIO_VAL" --output "$VAL_SCENARIO_INDEX" --meta-output "$VAL_SCENARIO_INDEX_META" --reuse-if-valid
CURRENT_STAGE="build_sparse_labels"
run build "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto --data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml --proto-glob "$SCENARIO_VAL" --output-dir "$FRESH" --allow-scenario-ids "$UNION_IDS_FILE" --index-jsonl "$VAL_SCENARIO_INDEX" --require-all-allowed-resolved --num-workers "$LABEL_WORKERS" --start-method forkserver --max-pending-multiplier 2 --no-compress --skip-existing --profile-jsonl "$PROFILE" --skip-diagnostics --cpu-only
CURRENT_STAGE="validate_sparse_build"
run sparse_integrity "$PYTHON_BIN" -m cowp.scripts.71_validate_sparse_label_build --labels-dir "$FRESH" --scene-ids "$UNION_IDS_FILE" --profile-jsonl "$PROFILE" --output "$SPARSE_INTEGRITY"
CURRENT_STAGE="summarize_profile"
run summarize "$PYTHON_BIN" -m cowp.scripts.49_summarize_label_build_profile --input "$PROFILE" --output "$SUMMARY" --top-slow 30
run natural_support "$PYTHON_BIN" -m cowp.scripts.68_summarize_natural_support_diagnostics --input "$PROFILE" --output "$NATURAL_DIAG"
run compare "$PYTHON_BIN" -m cowp.scripts.46_compare_proposal_probe --old-cache "$OLD_VAL_CACHE" --new-cache "$FRESH" --representative-scene-ids "$RANDOM_IDS_FILE" --hard-scene-ids "$HARD_IDS_FILE" --new-build-profile "$PROFILE" --output "$PAIRED" --min-overall-any-valid 0.99 --min-overall-any-ncf 0.40 --max-false-safe-floor 0.55 --max-pbtr-floor 0.45 --min-hard-recovery 0.20 --max-rmr-target-tta-error-s 0.20
run source_ablation "$PYTHON_BIN" -m cowp.scripts.50_ablate_proposal_sources --cache-dir "$FRESH" --output "$ABL"
run audit "$PYTHON_BIN" -m cowp.scripts.57_diagnose_causal_audit --cache-dir "$FRESH" --scene-ids "$UNION_IDS_FILE" --output "$AUDIT"
set +e
run_semantic supervision "$PYTHON_BIN" -m cowp.scripts.62_audit_training_supervision --cache-dir "$FRESH" --sample-scenes 0 --min-class-examples 32 --output "$SUPERVISION" --strict; SUP_RC=$?
run_semantic model_support "$PYTHON_BIN" -m cowp.scripts.65_audit_model_support \
  --cache-dir "$FRESH" --sample-scenes 0 --min-class-examples 32 --min-source-examples 32 \
  --max-unauditable-critical-rate 0.05 --min-certificate-complete-scene-rate 0.75 \
  --min-protected-prio-coverage 0.98 --coverage-gate-mode point \
  --hard-scene-ids "$HARD_IDS_FILE" --random-scene-ids "$RANDOM_IDS_FILE" \
  --max-auditability-stratum-gap 0.03 --max-certificate-stratum-gap 0.08 \
  --output "$MODEL_SUPPORT" --strict; MS_RC=$?
run_semantic screen "$PYTHON_BIN" -m cowp.scripts.58_screen_v16_8_9_causal_audit_probe --paired-probe "$PAIRED" --source-ablation "$ABL" --profile-summary "$SUMMARY" --audit-diagnostic "$AUDIT" --output "$BASE_VERDICT" --strict; SCREEN_RC=$?
set -e
CURRENT_STAGE="write_composite_verdict"
"$PYTHON_BIN" - "$VERDICT" "$CODE_FP" "$BASE_VERDICT" "$SUPERVISION" "$MODEL_SUPPORT" "$NATURAL_DIAG" "$SPARSE_INTEGRITY" "$SUP_RC" "$MS_RC" "$SCREEN_RC" <<'PY'
import json,sys
from pathlib import Path
out,fp,base_p,sup_p,ms_p,nat_p,sparse_p,sup_rc,ms_rc,screen_rc=sys.argv[1:]
def load(p):
  try:return json.load(open(p,encoding='utf-8'))
  except Exception:return {}
base,sup,ms,nat,sparse=map(load,(base_p,sup_p,ms_p,nat_p,sparse_p))
msc=ms.get('checks',{}) or {}
checks={'sparse_label_build_complete':bool(sparse.get('pass',False)),'proposal_causal_screen_pass':int(screen_rc)==0,'training_supervision_pass':bool(sup.get('pass',False)),'model_support_pass':bool(ms.get('pass',False)),'auditability_coverage':bool(msc.get('auditability_coverage',False)),'certificate_complete_scene_coverage':bool(msc.get('certificate_complete_scene_coverage',False)),'auditability_stratum_balance':bool(msc.get('auditability_stratum_balance',False)),'certificate_stratum_balance':bool(msc.get('certificate_stratum_balance',False)),'natural_rootless_zero_on_auditable':int(nat.get('rootless_critical_agents',-1))==0,'natural_lt2_low_burden_zero_on_auditable':int(nat.get('critical_agents_with_lt2_low_burden_roots',-1))==0,'protected_prio_coverage':float(nat.get('protected_prio_root_coverage',0.0))>=0.98}
passed=all(checks.values())
payload={'schema_version':'cowp_v16_8_18_strict_composite_verdict_v4','strict':True,'code_fingerprint_sha256':fp,'pass':passed,'recommend_full_rebuild':passed,'checks':checks,'sparse_build_integrity':sparse,'semantic_returncodes':{'supervision':int(sup_rc),'model_support':int(ms_rc),'screen':int(screen_rc)},'base_screen':base,'model_support_coverage':{'statistics':ms.get('coverage_statistics',{}),'by_stratum':ms.get('coverage_by_stratum',{})},'natural_support':{k:nat.get(k) for k in ('critical_agents_selected','mechanism_auditable_critical_agents','mechanism_unauditable_critical_agents','mechanism_unauditable_rate','mechanism_unauditable_future_support','mechanism_unauditable_finalizer_counts','mechanism_unauditable_with_sufficient_future_but_no_substantial_route_geometry','mechanism_unauditable_with_insufficient_future','rootless_critical_agents','critical_agents_with_lt2_low_burden_roots','protected_auditable_critical_agents','protected_without_prio_root','protected_prio_root_coverage','empirical_corridor_roots','rootless_dominant_rejection','lt2_low_burden_dominant_rejection','priority_rejection_reason_counts','map_rejected_min_distance_summary_m','map_rejected_max_distance_summary_m')},'next_action':'Run train-pilot with the identical fingerprint before full-core.' if passed else 'Do not full-rebuild; inspect strict natural/model-support diagnostics.'}
Path(out).write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps(payload,indent=2,ensure_ascii=False))
PY
"$PYTHON_BIN" - "$PIPELINE_STATUS" "$VERDICT" <<'PY'
import json,sys
from pathlib import Path
v=json.load(open(sys.argv[2],encoding='utf-8'))
Path(sys.argv[1]).write_text(json.dumps({
 'schema_version':'cowp_v16_8_18_strict_pipeline_status_v1','pipeline_complete':True,
 'composite_verdict_written':True,'recommend_full_rebuild':bool(v.get('recommend_full_rebuild',False)),
 'next_action':v.get('next_action')
},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
PY
PIPELINE_DONE=1
if "$PYTHON_BIN" - "$VERDICT" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1],encoding='utf-8')).get('recommend_full_rebuild') else 2)
PY
then echo "STRICT PASS: $VERDICT"; else echo "STRICT FAIL: verdict was written; DO NOT FULL REBUILD. $VERDICT"; exit 2; fi
