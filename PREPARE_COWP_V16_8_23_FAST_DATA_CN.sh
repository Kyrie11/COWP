#!/usr/bin/env bash
set -euo pipefail

# Compact, property-complete v16.8.23 full dataset builder.
# Official WOMD test hides the 8 s future, so the COWP held-out test is a
# scenario-disjoint subset of the official WOMD validation split.

PYTHON_BIN="${PYTHON_BIN:-python}"
WOMD_ROOT="${WOMD_ROOT:?export WOMD_ROOT=/path/to/waymo_open_dataset_motion_v_1_3_1}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:?export SOURCE_DATA_ROOT=/path/to/previous/promoted/COWP/dataset}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_23_compact_full}"
TRAIN_LIMIT="${TRAIN_LIMIT:-6000}"
VAL_LIMIT="${VAL_LIMIT:-1200}"
TEST_LIMIT="${TEST_LIMIT:-1500}"
SPLIT_SEED="${SPLIT_SEED:-cowp-v16.8.23-compact}"
LABEL_CFG="${LABEL_CFG:-configs/label_cowp_v16_8.yaml}"
LABEL_WORKERS_TRAIN="${LABEL_WORKERS_TRAIN:-16}"
LABEL_WORKERS_VAL="${LABEL_WORKERS_VAL:-12}"
LABEL_WORKERS_TEST="${LABEL_WORKERS_TEST:-12}"
CACHE_WORKERS="${CACHE_WORKERS:-8}"

# Avoid BLAS/OpenMP oversubscription across ProcessPool workers. These label
# kernels are dominated by many small NumPy/geometry calls, not large GEMMs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="-1"

SCENARIO_TRAIN="$WOMD_ROOT/uncompressed/scenario/training/*.tfrecord*"
SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
TFEXAMPLE_TRAIN="$WOMD_ROOT/uncompressed/tf_example/training/*.tfrecord*"
TFEXAMPLE_VAL="$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*"

INDEX_TRAIN="$COWP_ROOT/scenario_location_index_train.jsonl"
INDEX_VAL="$COWP_ROOT/scenario_location_index_validation.jsonl"
ALLOW_TRAIN="$COWP_ROOT/train_scene_ids.txt"
ALLOW_VAL="$COWP_ROOT/val_scene_ids.txt"
ALLOW_TEST="$COWP_ROOT/heldout_test_scene_ids.txt"
LABELS_TRAIN="$COWP_ROOT/labels_train"
LABELS_VAL="$COWP_ROOT/labels_val"
LABELS_TEST="$COWP_ROOT/labels_heldout_test"
CACHE_TRAIN="$COWP_ROOT/tensor_cache_train"
CACHE_VAL="$COWP_ROOT/tensor_cache_val"
CACHE_TEST="$COWP_ROOT/tensor_cache_heldout_test"
PROFILE_TRAIN="$COWP_ROOT/profile_labels_train.jsonl"
PROFILE_VAL="$COWP_ROOT/profile_labels_val.jsonl"
PROFILE_TEST="$COWP_ROOT/profile_labels_heldout_test.jsonl"
LOG_ROOT="$COWP_ROOT/logs"
mkdir -p "$COWP_ROOT" "$LOG_ROOT"

run(){
  local name="$1"; shift
  echo "[$name] $*"
  "$@" > >(tee "$LOG_ROOT/${name}.log") 2> >(tee -a "$LOG_ROOT/${name}.log" >&2)
}

find_source_cache(){
  local split="$1"
  local -a candidates=(
    "$SOURCE_DATA_ROOT/tensor_cache_${split}_waymax_transport_v16_8"
    "$SOURCE_DATA_ROOT/tensor_cache_${split}_waymax_transport_v15"
    "$SOURCE_DATA_ROOT/tensor_cache_${split}_waymax"
    "$SOURCE_DATA_ROOT/tensor_cache_${split}"
    "$SOURCE_DATA_ROOT/labels_${split}"
  )
  for p in "${candidates[@]}"; do
    if [[ -d "$p" ]] && find "$p" -maxdepth 1 -name '*.npz' -print -quit | grep -q .; then
      printf '%s\n' "$p"; return 0
    fi
  done
  return 1
}

SRC_TRAIN="$(find_source_cache train || true)"
SRC_VAL="$(find_source_cache val || true)"
[[ -n "$SRC_TRAIN" && -n "$SRC_VAL" ]] || {
  echo "Cannot find historical train/val NPZ caches under $SOURCE_DATA_ROOT." >&2
  echo "They are used ONLY for deterministic scenario-ID selection; all v16.8.23 labels are rebuilt from WOMD Scenario protos." >&2
  exit 3
}

# IMPORTANT v16.8.23 fix: caps apply even when reusing the historical scene set.
# Validation is partitioned into internal validation + held-out test because
# official WOMD test has no future GT and cannot produce COWP transport/witness labels.
run select_compact_ids "$PYTHON_BIN" - "$SRC_TRAIN" "$SRC_VAL" "$ALLOW_TRAIN" "$ALLOW_VAL" "$ALLOW_TEST" "$TRAIN_LIMIT" "$VAL_LIMIT" "$TEST_LIMIT" "$SPLIT_SEED" <<'PY'
from pathlib import Path
import hashlib, json, sys
src_tr,src_va,out_tr,out_va,out_te=map(Path,sys.argv[1:6])
ntr,nva,nte=map(int,sys.argv[6:9]); seed=sys.argv[9]
def ids(root):
    return [p.stem for p in root.glob('*.npz') if p.is_file() and not p.name.startswith('.')]
def ordered(values, tag):
    return sorted(set(values), key=lambda sid:(hashlib.sha256(f'{seed}|{tag}|{sid}'.encode()).hexdigest(), sid))
tr=ordered(ids(src_tr),'train')
va=ordered(ids(src_va),'validation')
if len(tr)<ntr: raise SystemExit(f'train source cache has only {len(tr)} ids, need {ntr}')
if len(va)<nva+nte: raise SystemExit(f'validation source cache has only {len(va)} ids, need {nva+nte}')
tr=tr[:ntr]; val=va[:nva]; test=va[nva:nva+nte]
assert not (set(tr)&set(val) or set(tr)&set(test) or set(val)&set(test))
for path,vals in ((out_tr,tr),(out_va,val),(out_te,test)):
    path.write_text('\n'.join(vals)+'\n',encoding='utf-8')
meta={'seed':seed,'train':len(tr),'val':len(val),'heldout_test':len(test),
      'source_train':str(src_tr),'source_validation':str(src_va),
      'test_definition':'held-out subset of official WOMD validation; official WOMD test future is hidden'}
(out_tr.parent/'compact_split_manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(meta,indent=2,ensure_ascii=False))
PY

run index_train "$PYTHON_BIN" -m cowp.scripts.72_build_scenario_location_index \
  --proto-glob "$SCENARIO_TRAIN" --output "$INDEX_TRAIN" --meta-output "$COWP_ROOT/scenario_location_index_train.meta.json" --reuse-if-valid
run index_validation "$PYTHON_BIN" -m cowp.scripts.72_build_scenario_location_index \
  --proto-glob "$SCENARIO_VAL" --output "$INDEX_VAL" --meta-output "$COWP_ROOT/scenario_location_index_validation.meta.json" --reuse-if-valid

common_label=(
  --data-config configs/data.yaml --label-config "$LABEL_CFG" --no-compress --skip-existing --skip-diagnostics --cpu-only
  --start-method forkserver --max-pending-multiplier 2
)
run labels_train "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto "${common_label[@]}" \
  --proto-glob "$SCENARIO_TRAIN" --output-dir "$LABELS_TRAIN" --allow-scenario-ids "$ALLOW_TRAIN" \
  --index-jsonl "$INDEX_TRAIN" --require-all-allowed-resolved --num-workers "$LABEL_WORKERS_TRAIN" --profile-jsonl "$PROFILE_TRAIN"
run labels_val "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto "${common_label[@]}" \
  --proto-glob "$SCENARIO_VAL" --output-dir "$LABELS_VAL" --allow-scenario-ids "$ALLOW_VAL" \
  --index-jsonl "$INDEX_VAL" --require-all-allowed-resolved --num-workers "$LABEL_WORKERS_VAL" --profile-jsonl "$PROFILE_VAL"
run labels_test "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto "${common_label[@]}" \
  --proto-glob "$SCENARIO_VAL" --output-dir "$LABELS_TEST" --allow-scenario-ids "$ALLOW_TEST" \
  --index-jsonl "$INDEX_VAL" --require-all-allowed-resolved --num-workers "$LABEL_WORKERS_TEST" --profile-jsonl "$PROFILE_TEST"

run profile_train "$PYTHON_BIN" -m cowp.scripts.44_summarize_label_build_profile --input "$PROFILE_TRAIN" --output "$COWP_ROOT/profile_train_summary.json"
run natural_train "$PYTHON_BIN" -m cowp.scripts.68_summarize_natural_support_diagnostics --input "$PROFILE_TRAIN" --output "$COWP_ROOT/natural_support_train.json"
run profile_val "$PYTHON_BIN" -m cowp.scripts.44_summarize_label_build_profile --input "$PROFILE_VAL" --output "$COWP_ROOT/profile_val_summary.json"
run natural_val "$PYTHON_BIN" -m cowp.scripts.68_summarize_natural_support_diagnostics --input "$PROFILE_VAL" --output "$COWP_ROOT/natural_support_val.json"
run profile_heldout_test "$PYTHON_BIN" -m cowp.scripts.44_summarize_label_build_profile --input "$PROFILE_TEST" --output "$COWP_ROOT/profile_heldout_test_summary.json"
run natural_heldout_test "$PYTHON_BIN" -m cowp.scripts.68_summarize_natural_support_diagnostics --input "$PROFILE_TEST" --output "$COWP_ROOT/natural_support_heldout_test.json"

cache_one(){
  local name="$1" split="$2" glob="$3" labels="$4" out="$5" allow="$6"
  run "cache_${name}" "$PYTHON_BIN" -m cowp.scripts.02_build_tensor_cache \
    --data-config configs/data.yaml --split "$split" --tfexample-glob "$glob" \
    --labels-dir "$labels" --output-dir "$out" --num-workers "$CACHE_WORKERS" \
    --start-method forkserver --parallel-scan --require-waymax-ready --require-sdc-paths \
    --require-all-labels-matched --skip-existing --no-compress --profile-jsonl "$COWP_ROOT/profile_tensor_cache_${name}.jsonl" --cpu-only
  run "verify_${name}" "$PYTHON_BIN" -m cowp.scripts.60_verify_fresh_v16_8_9_cache \
    --cache-dir "$out" --allowlist "$allow" --sample-scenes 0 --require-sdc-paths --output "$COWP_ROOT/verify_cache_${name}.json"
}
cache_one train training "$TFEXAMPLE_TRAIN" "$LABELS_TRAIN" "$CACHE_TRAIN" "$ALLOW_TRAIN"
cache_one val validation "$TFEXAMPLE_VAL" "$LABELS_VAL" "$CACHE_VAL" "$ALLOW_VAL"
cache_one heldout_test validation "$TFEXAMPLE_VAL" "$LABELS_TEST" "$CACHE_TEST" "$ALLOW_TEST"

# Stable v16.8.9 self-contained cache manifest expected by the protocol gate.
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
 'producer_version':'v16.8.23-fast',
 'build_fingerprint_sha256':fp, 'label_semantic_fingerprint_sha256':lfp,
 'raw_train_cache':str(tr), 'raw_val_cache':str(va),
 'transport_train_cache':str(tr), 'transport_val_cache':str(va),
 'train_allowlist':str(Path(sys.argv[4])), 'val_allowlist':str(Path(sys.argv[5])),
 'transport_storage':'inline_same_npz', 'womd_contract':'1.3.1+sdc_paths',
}
(root/'data_manifest_v16_8_9.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
PY

# Stable train/val cache protocol plus held-out-test one-to-one verification above.
run fresh_protocol "$PYTHON_BIN" -m cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol \
  --cowp-root "$COWP_ROOT" --raw-train "$CACHE_TRAIN" --raw-val "$CACHE_VAL" \
  --transport-train "$CACHE_TRAIN" --transport-val "$CACHE_VAL" --sample-scenes 512 \
  --output "$COWP_ROOT/fresh_cache_protocol.json"

audit_split(){
  local split="$1" cache="$2"
  run "supervision_${split}" "$PYTHON_BIN" -m cowp.scripts.62_audit_training_supervision --cache-dir "$cache" --output "$COWP_ROOT/training_supervision_${split}.json" --sample-scenes 0 --strict
  run "model_support_${split}" "$PYTHON_BIN" -m cowp.scripts.65_audit_model_support --cache-dir "$cache" --output "$COWP_ROOT/model_support_${split}.json" --sample-scenes 0 --strict \
    --max-unauditable-critical-rate 0.05 --min-certificate-complete-scene-rate 0.75 --min-protected-prio-coverage 0.98
  run "mechanism_contrast_${split}" "$PYTHON_BIN" -m cowp.scripts.74_audit_mechanism_contrast --cache-dir "$cache" --output "$COWP_ROOT/mechanism_contrast_${split}.json" --sample-scenes 0 --strict
  run "causal_${split}" "$PYTHON_BIN" -m cowp.scripts.57_diagnose_causal_audit --cache-dir "$cache" --output "$COWP_ROOT/causal_audit_${split}.json" --limit 0
}
audit_split train "$CACHE_TRAIN"
audit_split val "$CACHE_VAL"
audit_split heldout_test "$CACHE_TEST"

# Final compact-full verdict; every split must independently support the six-layer mechanism.
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
             'mechanism_contrast_train','mechanism_contrast_val','mechanism_contrast_heldout_test'):
    reports[name]=json.loads((root/f'{name}.json').read_text())
def ok(x):
    if 'pass' in x:return bool(x['pass'])
    if 'recommend_full_rebuild' in x:return bool(x['recommend_full_rebuild'])
    return not bool(x.get('reasons'))
checks={k:ok(v) for k,v in reports.items()}; checks['scenario_id_disjoint']=split_disjoint
m=importlib.import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol')
manifest={
 'schema_version':'cowp_v16_8_23_compact_full_manifest_v1',
 'producer_version':'v16.8.23-fast',
 'code_fingerprint_sha256':m.current_fingerprint(Path.cwd()),
 'label_semantic_fingerprint_sha256':m.current_label_semantic_fingerprint(Path.cwd()),
 'train_cache':str(tr),'val_cache':str(va),'heldout_test_cache':str(te),
 'train_allowlist':str(a_tr),'val_allowlist':str(a_va),'heldout_test_allowlist':str(a_te),
 'heldout_test_source':'official WOMD validation split (future GT available); scenario-id disjoint from internal val',
 'official_womd_test_used_for_cowp_labels':False,
 'womd_contract':'1.3.1+sdc_paths',
}
(root/'data_manifest_v16_8_23.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
payload={'schema_version':'cowp_v16_8_23_compact_full_support_v1','pass':all(checks.values()),'checks':checks,
         'failed_checks':[k for k,v in checks.items() if not v],
         'next_action':'Attach Waymax outcomes, then train/evaluate.' if all(checks.values()) else 'STOP: inspect failed_checks; do not train on this compact full build.'}
(root/'full_core_support_verdict_v16_8_23.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(payload,indent=2,ensure_ascii=False))
if not payload['pass']: raise SystemExit(2)
PY

echo "V16.8.23 COMPACT FULL train/val/heldout-test BUILD PASS: $COWP_ROOT"
echo "Next: attach Waymax outcomes on held-out splits, then train/evaluate."
