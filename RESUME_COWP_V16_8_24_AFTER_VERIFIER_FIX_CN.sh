#!/usr/bin/env bash
set -euo pipefail

# Resume the v16.8.24 compact full-core build after fixing
# cowp/scripts/60_verify_fresh_v16_8_9_cache.py.
# This intentionally REUSES existing labels and the completed train tensor cache.

PYTHON_BIN="${PYTHON_BIN:-python}"
WOMD_ROOT="${WOMD_ROOT:?export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
CACHE_WORKERS="${CACHE_WORKERS:-12}"
LABEL_CFG="${LABEL_CFG:-configs/label_cowp_v16_8.yaml}"

[[ -d cowp ]] || { echo "Run this script from the COWP repository root." >&2; exit 2; }

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="-1"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

TFEXAMPLE_TRAIN="$WOMD_ROOT/uncompressed/tf_example/training/*.tfrecord*"
TFEXAMPLE_VAL="$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*"
ALLOW_TRAIN="$COWP_ROOT/train_scene_ids.txt"
ALLOW_VAL="$COWP_ROOT/val_scene_ids.txt"
ALLOW_TEST="$COWP_ROOT/heldout_test_scene_ids.txt"
LABELS_TRAIN="$COWP_ROOT/labels_train"
LABELS_VAL="$COWP_ROOT/labels_val"
LABELS_TEST="$COWP_ROOT/labels_heldout_test"
CACHE_TRAIN="$COWP_ROOT/tensor_cache_train"
CACHE_VAL="$COWP_ROOT/tensor_cache_val"
CACHE_TEST="$COWP_ROOT/tensor_cache_heldout_test"
LOG_ROOT="$COWP_ROOT/logs_resume_after_verifier_fix"
mkdir -p "$LOG_ROOT"

run(){
  local name="$1"; shift
  echo "[$name] $*"
  "$@" > >(tee "$LOG_ROOT/${name}.log") 2> >(tee -a "$LOG_ROOT/${name}.log" >&2)
}

for p in "$ALLOW_TRAIN" "$ALLOW_VAL" "$ALLOW_TEST"; do
  [[ -f "$p" ]] || { echo "Missing allowlist: $p" >&2; exit 3; }
done
for p in "$LABELS_TRAIN" "$LABELS_VAL" "$LABELS_TEST" "$CACHE_TRAIN"; do
  [[ -d "$p" ]] || { echo "Missing completed artifact: $p" >&2; exit 3; }
done

# 1) Re-verify the already-completed 5k train cache with the corrected mechanism domain.
run verify_train "$PYTHON_BIN" -m cowp.scripts.60_verify_fresh_v16_8_9_cache \
  --cache-dir "$CACHE_TRAIN" --allowlist "$ALLOW_TRAIN" --sample-scenes 0 \
  --require-sdc-paths --output "$COWP_ROOT/verify_cache_train.json"

# 2) Build only the tensor caches that the original set -e run never reached.
cache_one(){
  local name="$1" split="$2" glob="$3" labels="$4" out="$5" allow="$6"
  run "cache_${name}" "$PYTHON_BIN" -m cowp.scripts.02_build_tensor_cache \
    --data-config configs/data.yaml --split "$split" --tfexample-glob "$glob" \
    --labels-dir "$labels" --output-dir "$out" --num-workers "$CACHE_WORKERS" \
    --start-method forkserver --parallel-scan --require-waymax-ready --require-sdc-paths \
    --require-all-labels-matched --skip-existing --no-compress \
    --profile-jsonl "$COWP_ROOT/profile_tensor_cache_${name}.jsonl" --cpu-only
  run "verify_${name}" "$PYTHON_BIN" -m cowp.scripts.60_verify_fresh_v16_8_9_cache \
    --cache-dir "$out" --allowlist "$allow" --sample-scenes 0 --require-sdc-paths \
    --output "$COWP_ROOT/verify_cache_${name}.json"
}
cache_one val validation "$TFEXAMPLE_VAL" "$LABELS_VAL" "$CACHE_VAL" "$ALLOW_VAL"
cache_one heldout_test validation "$TFEXAMPLE_VAL" "$LABELS_TEST" "$CACHE_TEST" "$ALLOW_TEST"

# 3) Regenerate the cache/code fingerprint manifest after the verifier code change.
"$PYTHON_BIN" - "$COWP_ROOT" "$CACHE_TRAIN" "$CACHE_VAL" "$ALLOW_TRAIN" "$ALLOW_VAL" <<'PY'
import importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); tr=Path(sys.argv[2]); va=Path(sys.argv[3])
m=importlib.import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol')
fp=m.current_fingerprint(Path.cwd()); lfp=m.current_label_semantic_fingerprint(Path.cwd())
(root/'build_fingerprint.sha256').write_text(fp+'\n',encoding='utf-8')
(root/'label_semantic_fingerprint.sha256').write_text(lfp+'\n',encoding='utf-8')
manifest={
 'schema_version':'cowp_v16_8_9_causal_audit_self_contained_data_v1',
 'producer_version':'v16.8.24-rebuild-ready+verifier-mechanism-mask-fix',
 'build_fingerprint_sha256':fp, 'label_semantic_fingerprint_sha256':lfp,
 'raw_train_cache':str(tr), 'raw_val_cache':str(va),
 'transport_train_cache':str(tr), 'transport_val_cache':str(va),
 'train_allowlist':str(Path(sys.argv[4])), 'val_allowlist':str(Path(sys.argv[5])),
 'transport_storage':'inline_same_npz', 'womd_contract':'1.3.1+sdc_paths',
}
(root/'data_manifest_v16_8_9.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
PY

run fresh_protocol "$PYTHON_BIN" -m cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol \
  --cowp-root "$COWP_ROOT" --raw-train "$CACHE_TRAIN" --raw-val "$CACHE_VAL" \
  --transport-train "$CACHE_TRAIN" --transport-val "$CACHE_VAL" --sample-scenes 512 \
  --output "$COWP_ROOT/fresh_cache_protocol.json"

# 4) Run ALL remaining hard gates. Do not loosen these thresholds to make a run pass.
audit_split(){
  local split="$1" cache="$2"
  run "supervision_${split}" "$PYTHON_BIN" -m cowp.scripts.62_audit_training_supervision \
    --cache-dir "$cache" --output "$COWP_ROOT/training_supervision_${split}.json" --sample-scenes 0 --strict
  run "model_support_${split}" "$PYTHON_BIN" -m cowp.scripts.65_audit_model_support \
    --cache-dir "$cache" --output "$COWP_ROOT/model_support_${split}.json" --sample-scenes 0 --strict \
    --max-unauditable-critical-rate 0.05 --min-certificate-complete-scene-rate 0.75 \
    --min-protected-prio-coverage 0.98
  run "mechanism_contrast_${split}" "$PYTHON_BIN" -m cowp.scripts.74_audit_mechanism_contrast \
    --cache-dir "$cache" --output "$COWP_ROOT/mechanism_contrast_${split}.json" --sample-scenes 0 --strict
  run "causal_${split}" "$PYTHON_BIN" -m cowp.scripts.57_diagnose_causal_audit \
    --cache-dir "$cache" --output "$COWP_ROOT/causal_audit_${split}.json" --limit 0
}
audit_split train "$CACHE_TRAIN"
audit_split val "$CACHE_VAL"
audit_split heldout_test "$CACHE_TEST"

# 5) Produce the same final compact-full verdict as the official v24 builder.
"$PYTHON_BIN" - "$COWP_ROOT" "$CACHE_TRAIN" "$CACHE_VAL" "$CACHE_TEST" "$ALLOW_TRAIN" "$ALLOW_VAL" "$ALLOW_TEST" <<'PY'
from pathlib import Path
import importlib,json,sys
root=Path(sys.argv[1])
tr,va,te=map(Path,sys.argv[2:5]); a_tr,a_va,a_te=map(Path,sys.argv[5:8])
ids=lambda p:set(x.strip() for x in p.read_text().splitlines() if x.strip())
T,V,E=ids(a_tr),ids(a_va),ids(a_te)
split_disjoint=not (T&V or T&E or V&E)
reports={}
for name in ('fresh_cache_protocol',
             'training_supervision_train','training_supervision_val','training_supervision_heldout_test',
             'model_support_train','model_support_val','model_support_heldout_test',
             'mechanism_contrast_train','mechanism_contrast_val','mechanism_contrast_heldout_test',
             'causal_audit_train','causal_audit_val','causal_audit_heldout_test'):
    reports[name]=json.loads((root/f'{name}.json').read_text())
def ok(name,x):
    if name.startswith('causal_audit_'):
        integ=x.get('integrity',{})
        return bool(integ) and all(bool(v) for v in integ.values())
    if 'pass' in x:return bool(x['pass'])
    if 'recommend_full_rebuild' in x:return bool(x['recommend_full_rebuild'])
    return not bool(x.get('reasons'))
checks={k:ok(k,v) for k,v in reports.items()}; checks['scenario_id_disjoint']=split_disjoint
m=importlib.import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol')
manifest={
 'schema_version':'cowp_v16_8_24_compact_full_manifest_v1',
 'producer_version':'v16.8.24-rebuild-ready+verifier-mechanism-mask-fix',
 'code_fingerprint_sha256':m.current_fingerprint(Path.cwd()),
 'label_semantic_fingerprint_sha256':m.current_label_semantic_fingerprint(Path.cwd()),
 'train_cache':str(tr),'val_cache':str(va),'heldout_test_cache':str(te),
 'train_allowlist':str(a_tr),'val_allowlist':str(a_va),'heldout_test_allowlist':str(a_te),
 'heldout_test_source':'official WOMD validation split (future GT available); scenario-id disjoint from internal val',
 'official_womd_test_used_for_cowp_labels':False,
 'womd_contract':'1.3.1+sdc_paths',
}
(root/'data_manifest_v16_8_24.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
payload={'schema_version':'cowp_v16_8_24_compact_full_support_v1','pass':all(checks.values()),'checks':checks,
         'failed_checks':[k for k,v in checks.items() if not v],
         'next_action':'Attach Waymax outcomes, then train/evaluate.' if all(checks.values()) else 'STOP: inspect failed_checks; do not train on this compact full build.'}
(root/'full_core_support_verdict_v16_8_24.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(payload,indent=2,ensure_ascii=False))
if not payload['pass']: raise SystemExit(2)
PY

echo "RESUMED V16.8.24 FULL-CORE PASS: $COWP_ROOT"
echo "Next: bash NEXT_EXECUTION_V16_8_24_CN.sh outcomes"
