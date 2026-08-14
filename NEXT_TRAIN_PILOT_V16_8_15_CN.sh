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
export SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export OLD_TRAIN_CACHE="${OLD_TRAIN_CACHE:-$SOURCE_DATA_ROOT/tensor_cache_train}"
export TRAIN_PILOT_ROOT="${TRAIN_PILOT_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_15_train_pilot}"
export HARD_COUNT="${HARD_COUNT:-400}"
export RANDOM_COUNT="${RANDOM_COUNT:-800}"
export MIN_HARD_COUNT="${MIN_HARD_COUNT:-128}"
export LABEL_WORKERS="${LABEL_WORKERS:-32}"
export SEED="${SEED:-2026}"
export FORCE_REBUILD_TRAIN_PILOT="${FORCE_REBUILD_TRAIN_PILOT:-1}"

SCENARIO_TRAIN="$WOMD_ROOT/uncompressed/scenario/training/*.tfrecord*"
FRESH="$TRAIN_PILOT_ROOT/labels_train_v16_8_15"
PROFILE="$TRAIN_PILOT_ROOT/profile_train_pilot.jsonl"
SUMMARY="$TRAIN_PILOT_ROOT/profile_train_pilot_summary.json"
CEILING_OLD="$TRAIN_PILOT_ROOT/old_train_proposal_ceiling.json"
HARD_IDS="$TRAIN_PILOT_ROOT/hard_scene_ids.txt"
RANDOM_IDS="$TRAIN_PILOT_ROOT/random_scene_ids.txt"
UNION_IDS="$TRAIN_PILOT_ROOT/pilot_union_scene_ids.txt"
MANIFEST="$TRAIN_PILOT_ROOT/pilot_manifest_audit.json"
SUPERVISION="$TRAIN_PILOT_ROOT/training_supervision_audit.json"
MODEL_SUPPORT="$TRAIN_PILOT_ROOT/model_support_audit.json"
CAUSAL_AUDIT="$TRAIN_PILOT_ROOT/causal_audit_diagnostic.json"
NATURAL_DIAG="$TRAIN_PILOT_ROOT/natural_support_diagnostic.json"
CEILING_FRESH="$TRAIN_PILOT_ROOT/fresh_train_proposal_ceiling.json"
VERDICT="$TRAIN_PILOT_ROOT/v16_8_15_train_pilot_verdict.json"
FP_FILE="$TRAIN_PILOT_ROOT/v16_8_15_code_fingerprint.sha256"

mkdir -p "$TRAIN_PILOT_ROOT/logs"
[[ -d "$OLD_TRAIN_CACHE" ]] || { echo "missing OLD_TRAIN_CACHE=$OLD_TRAIN_CACHE" >&2; exit 3; }

CODE_FP="$($PYTHON_BIN - <<'PY'
from pathlib import Path
from importlib import import_module
m=import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol')
print(m.current_fingerprint(Path.cwd()))
PY
)"

if [[ "$FORCE_REBUILD_TRAIN_PILOT" == "1" ]]; then
  rm -rf "$FRESH"
  rm -f "$PROFILE" "$SUMMARY" "$CEILING_OLD" "$HARD_IDS" "$RANDOM_IDS" "$UNION_IDS" \
        "$MANIFEST" "$SUPERVISION" "$MODEL_SUPPORT" "$CAUSAL_AUDIT" "$NATURAL_DIAG" \
        "$CEILING_FRESH" "$VERDICT" "$FP_FILE"
fi
mkdir -p "$FRESH"
printf '%s\n' "$CODE_FP" > "$FP_FILE"

run(){
  local name="$1"; shift
  echo "[$name] $*"
  "$@" > >(tee "$TRAIN_PILOT_ROOT/logs/${name}.log") 2> >(tee -a "$TRAIN_PILOT_ROOT/logs/${name}.log" >&2)
}

# Deliberately sample both old-cache hard scenes and representative random train
# scenes. Validation strict success does not prove that the training split has a
# usable natural/response/witness support distribution.
run choose_train_pilot "$PYTHON_BIN" -m cowp.scripts.45_diagnose_proposal_ceiling \
  --cache-dir "$OLD_TRAIN_CACHE" --output "$CEILING_OLD" \
  --hard-scene-ids "$HARD_IDS" --hard-count "$HARD_COUNT" \
  --random-scene-ids "$RANDOM_IDS" --random-count "$RANDOM_COUNT" --probe-total-count "$((HARD_COUNT + RANDOM_COUNT))" --random-exclude-hard-probe \
  --control-count 0 --seed "$SEED"
cat "$HARD_IDS" "$RANDOM_IDS" | awk 'NF && !seen[$1]++{print $1}' > "$UNION_IDS"
run validate_manifest "$PYTHON_BIN" -m cowp.scripts.63_validate_probe_manifest \
  --ceiling-json "$CEILING_OLD" --hard-ids "$HARD_IDS" --random-ids "$RANDOM_IDS" \
  --union-ids "$UNION_IDS" --expected-hard -1 --expected-random -1 --expected-total "$((HARD_COUNT + RANDOM_COUNT))" --min-hard "$MIN_HARD_COUNT" \
  --output "$MANIFEST"

run build "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
  --data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml \
  --proto-glob "$SCENARIO_TRAIN" --output-dir "$FRESH" --allow-scenario-ids "$UNION_IDS" \
  --num-workers "$LABEL_WORKERS" --start-method forkserver --max-pending-multiplier 2 \
  --no-compress --skip-existing --profile-jsonl "$PROFILE" --skip-diagnostics --cpu-only

run profile "$PYTHON_BIN" -m cowp.scripts.49_summarize_label_build_profile \
  --input "$PROFILE" --output "$SUMMARY" --top-slow 30
run natural_support "$PYTHON_BIN" -m cowp.scripts.68_summarize_natural_support_diagnostics \
  --input "$PROFILE" --output "$NATURAL_DIAG"
# Supervision/model-support are semantic gates: they intentionally return non-zero
# on a scientifically invalid pilot.  Capture those return codes, but continue so
# causal/ceiling diagnostics and, critically, the final FAIL verdict are written.
set +e
run supervision "$PYTHON_BIN" -m cowp.scripts.62_audit_training_supervision \
  --cache-dir "$FRESH" --sample-scenes 0 --min-class-examples 32 --strict --output "$SUPERVISION"
SUP_RC=$?
run model_support "$PYTHON_BIN" -m cowp.scripts.65_audit_model_support \
  --cache-dir "$FRESH" --sample-scenes 0 --min-class-examples 32 --min-source-examples 32 --max-unauditable-critical-rate 0.01 --min-certificate-complete-scene-rate 0.98 --strict \
  --output "$MODEL_SUPPORT"
MS_RC=$?
set -e
run causal_audit "$PYTHON_BIN" -m cowp.scripts.57_diagnose_causal_audit \
  --cache-dir "$FRESH" --scene-ids "$UNION_IDS" --output "$CAUSAL_AUDIT"
run fresh_ceiling "$PYTHON_BIN" -m cowp.scripts.45_diagnose_proposal_ceiling \
  --cache-dir "$FRESH" --output "$CEILING_FRESH" --hard-count 0 --random-count 0 --control-count 0 --seed "$SEED"

"$PYTHON_BIN" - "$VERDICT" "$CODE_FP" "$SUPERVISION" "$MODEL_SUPPORT" "$CAUSAL_AUDIT" "$NATURAL_DIAG" "$CEILING_FRESH" "$SUP_RC" "$MS_RC" <<'PY'
import json,sys
from pathlib import Path
out,fp,sup_p,ms_p,ca_p,nat_p,ceil_p,sup_rc,ms_rc=sys.argv[1:]
sup=json.load(open(sup_p,encoding='utf-8'))
ms=json.load(open(ms_p,encoding='utf-8'))
ca=json.load(open(ca_p,encoding='utf-8'))
nat=json.load(open(nat_p,encoding='utf-8'))
ceil=json.load(open(ceil_p,encoding='utf-8'))
integ=ca.get('integrity',{})
rates=ceil.get('scene_rates',{})
max_unauditable=0.01
checks={
  'training_supervision_pass': bool(sup.get('pass',False)),
  'model_support_pass': bool(ms.get('pass',False)),
  'auditability_coverage': float(nat.get('mechanism_unauditable_rate',1.0)) <= max_unauditable,
  'natural_rootless_zero_on_auditable': int(nat.get('rootless_critical_agents',-1)) == 0,
  'natural_lt2_low_burden_zero_on_auditable': int(nat.get('critical_agents_with_lt2_low_burden_roots',-1)) == 0,
  'protected_prio_complete': int(nat.get('protected_without_prio_root',-1)) == 0,
  'causal_no_read_errors': bool(integ.get('no_read_errors',False)),
  'causal_no_silent_blockers': bool(integ.get('no_silent_blockers',False)),
  'causal_no_irrelevant_blockers': bool(integ.get('no_irrelevant_blockers',False)),
  'candidate_any_valid': float(rates.get('any_valid',0.0)) >= 0.99,
  # This is a support gate, not a publication metric. Candidate-level NCF is
  # intentionally conservative on certificate-incomplete scenes; the separate
  # auditability gate prevents passing by hiding difficult critical relations.
  'candidate_any_ncf_support': float(rates.get('any_ncf',0.0)) >= 0.30,
}
passed=all(checks.values())
payload={
  'schema_version':'cowp_v16_8_15_train_pilot_verdict_v2',
  'code_fingerprint_sha256':fp,
  'pass':passed,
  'recommend_full_rebuild':passed,
  'checks':checks,
  'semantic_returncodes':{'supervision':int(sup_rc),'model_support':int(ms_rc)},
  'scene_rates':rates,
  'natural_support':{
    'critical_agents_selected':nat.get('critical_agents_selected'),
    'mechanism_auditable_critical_agents':nat.get('mechanism_auditable_critical_agents'),
    'mechanism_unauditable_critical_agents':nat.get('mechanism_unauditable_critical_agents'),
    'mechanism_unauditable_rate':nat.get('mechanism_unauditable_rate'),
    'rootless_critical_agents':nat.get('rootless_critical_agents'),
    'critical_agents_with_lt2_low_burden_roots':nat.get('critical_agents_with_lt2_low_burden_roots'),
    'protected_without_prio_root':nat.get('protected_without_prio_root'),
    'empirical_corridor_roots':nat.get('empirical_corridor_roots'),
    'pair_neutral_unsafe_rate':nat.get('pair_neutral_unsafe_rate'),
  },
  'next_action': 'Full rebuild may proceed only if the validation strict verdict also passes.' if passed else 'Do not full-rebuild; inspect the train-pilot support diagnostics.',
}
Path(out).write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(payload,indent=2,ensure_ascii=False))
PY

echo "TRAIN_PILOT_VERDICT=$VERDICT"
if "$PYTHON_BIN" - "$VERDICT" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1],encoding='utf-8')).get('recommend_full_rebuild') else 2)
PY
then
  echo "TRAIN PILOT PASS: validation strict with the identical fingerprint is still required."
else
  echo "TRAIN PILOT FAIL: verdict was written; DO NOT full-rebuild."
  exit 2
fi
