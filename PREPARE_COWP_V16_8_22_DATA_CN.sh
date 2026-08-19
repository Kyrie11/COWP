#!/usr/bin/env bash
set -euo pipefail

# v16.8.22 self-contained full-core builder.
# It rebuilds Scenario->COWP labels and TFExample tensor caches from WOMD 1.3.1.
# By default only the historical scenario-ID set is reused; no old label tensor is reused.

PYTHON_BIN="${PYTHON_BIN:-python}"
WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_22_full}"
REUSE_OLD_SCENE_SET="${REUSE_OLD_SCENE_SET:-1}"
TRAIN_LIMIT="${TRAIN_LIMIT:-22000}"
VAL_LIMIT="${VAL_LIMIT:-5000}"
LABEL_WORKERS_TRAIN="${LABEL_WORKERS_TRAIN:-32}"
LABEL_WORKERS_VAL="${LABEL_WORKERS_VAL:-24}"
CACHE_WORKERS="${CACHE_WORKERS:-8}"
LABEL_CFG="${LABEL_CFG:-configs/label_cowp_v16_8.yaml}"

SCENARIO_TRAIN="$WOMD_ROOT/uncompressed/scenario/training/*.tfrecord*"
SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
TFEXAMPLE_TRAIN="$WOMD_ROOT/uncompressed/tf_example/training/*.tfrecord*"
TFEXAMPLE_VAL="$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*"

INDEX_TRAIN="$COWP_ROOT/scenario_location_index_train.jsonl"
INDEX_VAL="$COWP_ROOT/scenario_location_index_val.jsonl"
ALLOW_TRAIN="$COWP_ROOT/train_scene_ids.txt"
ALLOW_VAL="$COWP_ROOT/val_scene_ids.txt"
LABELS_TRAIN="$COWP_ROOT/labels_train"
LABELS_VAL="$COWP_ROOT/labels_val"
CACHE_TRAIN="$COWP_ROOT/tensor_cache_train"
CACHE_VAL="$COWP_ROOT/tensor_cache_val"
PROFILE_TRAIN="$COWP_ROOT/profile_labels_train.jsonl"
PROFILE_VAL="$COWP_ROOT/profile_labels_val.jsonl"
LOG_ROOT="$COWP_ROOT/logs"

mkdir -p "$COWP_ROOT" "$LOG_ROOT"

run(){
  local name="$1"; shift
  echo "[$name] $*"
  "$@" > >(tee "$LOG_ROOT/${name}.log") 2> >(tee -a "$LOG_ROOT/${name}.log" >&2)
}

# Select a historical cache only to recover the paired scenario-ID set.
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

write_allowlist_from_cache(){
  local cache="$1" out="$2"
  "$PYTHON_BIN" - "$cache" "$out" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1]); out=Path(sys.argv[2])
ids=sorted(p.stem for p in root.glob('*.npz') if p.is_file() and not p.name.startswith('.'))
if not ids:
    raise SystemExit(f'no NPZ scenario ids found under {root}')
out.write_text('\n'.join(ids)+'\n',encoding='utf-8')
print(f'{out}: {len(ids)} scenario ids from {root}')
PY
}

if [[ "$REUSE_OLD_SCENE_SET" == "1" ]]; then
  SRC_TRAIN="$(find_source_cache train || true)"
  SRC_VAL="$(find_source_cache val || true)"
  [[ -n "$SRC_TRAIN" && -n "$SRC_VAL" ]] || {
    echo "REUSE_OLD_SCENE_SET=1 but a historical train/val NPZ cache could not be found under $SOURCE_DATA_ROOT" >&2
    echo "Set SOURCE_DATA_ROOT to the promoted historical dataset, or set REUSE_OLD_SCENE_SET=0 for an explicit new scene set." >&2
    exit 3
  }
  write_allowlist_from_cache "$SRC_TRAIN" "$ALLOW_TRAIN"
  write_allowlist_from_cache "$SRC_VAL" "$ALLOW_VAL"
else
  rm -f "$ALLOW_TRAIN" "$ALLOW_VAL"
fi

# Build location-aware Scenario indexes before label generation.  This prevents a sparse
# allow-list build from repeatedly scanning unrelated WOMD records.
run index_train "$PYTHON_BIN" -m cowp.scripts.72_build_scenario_location_index \
  --proto-glob "$SCENARIO_TRAIN" --output "$INDEX_TRAIN" --meta-output "$COWP_ROOT/scenario_location_index_train.meta.json" --reuse-if-valid
run index_val "$PYTHON_BIN" -m cowp.scripts.72_build_scenario_location_index \
  --proto-glob "$SCENARIO_VAL" --output "$INDEX_VAL" --meta-output "$COWP_ROOT/scenario_location_index_val.meta.json" --reuse-if-valid

common_label=(
  --data-config configs/data.yaml --label-config "$LABEL_CFG" --no-compress --skip-existing --skip-diagnostics --cpu-only
  --start-method forkserver --max-pending-multiplier 2
)

if [[ "$REUSE_OLD_SCENE_SET" == "1" ]]; then
  run labels_train "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
    "${common_label[@]}" --proto-glob "$SCENARIO_TRAIN" --output-dir "$LABELS_TRAIN" \
    --allow-scenario-ids "$ALLOW_TRAIN" --index-jsonl "$INDEX_TRAIN" --require-all-allowed-resolved \
    --num-workers "$LABEL_WORKERS_TRAIN" --profile-jsonl "$PROFILE_TRAIN"
  run labels_val "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
    "${common_label[@]}" --proto-glob "$SCENARIO_VAL" --output-dir "$LABELS_VAL" \
    --allow-scenario-ids "$ALLOW_VAL" --index-jsonl "$INDEX_VAL" --require-all-allowed-resolved \
    --num-workers "$LABEL_WORKERS_VAL" --profile-jsonl "$PROFILE_VAL"
else
  run labels_train "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
    "${common_label[@]}" --proto-glob "$SCENARIO_TRAIN" --output-dir "$LABELS_TRAIN" \
    --index-jsonl "$INDEX_TRAIN" --limit "$TRAIN_LIMIT" --num-workers "$LABEL_WORKERS_TRAIN" --profile-jsonl "$PROFILE_TRAIN"
  run labels_val "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
    "${common_label[@]}" --proto-glob "$SCENARIO_VAL" --output-dir "$LABELS_VAL" \
    --index-jsonl "$INDEX_VAL" --limit "$VAL_LIMIT" --num-workers "$LABEL_WORKERS_VAL" --profile-jsonl "$PROFILE_VAL"
  "$PYTHON_BIN" - "$LABELS_TRAIN" "$ALLOW_TRAIN" "$LABELS_VAL" "$ALLOW_VAL" <<'PY'
from pathlib import Path
import sys
for root,out in ((sys.argv[1],sys.argv[2]),(sys.argv[3],sys.argv[4])):
    ids=sorted(p.stem for p in Path(root).glob('*.npz') if p.is_file())
    Path(out).write_text('\n'.join(ids)+'\n',encoding='utf-8')
PY
fi

run natural_train "$PYTHON_BIN" -m cowp.scripts.68_summarize_natural_support_diagnostics --input "$PROFILE_TRAIN" --output "$COWP_ROOT/natural_support_train.json"
run natural_val "$PYTHON_BIN" -m cowp.scripts.68_summarize_natural_support_diagnostics --input "$PROFILE_VAL" --output "$COWP_ROOT/natural_support_val.json"

# Fresh TFExample tensor caches.  WOMD 1.3.1 SDC-path readiness and one-to-one label
# matching are hard requirements for the promoted full dataset.
run cache_train "$PYTHON_BIN" -m cowp.scripts.02_build_tensor_cache \
  --data-config configs/data.yaml --split training --tfexample-glob "$TFEXAMPLE_TRAIN" \
  --labels-dir "$LABELS_TRAIN" --output-dir "$CACHE_TRAIN" --num-workers "$CACHE_WORKERS" \
  --start-method forkserver --parallel-scan --require-waymax-ready --require-sdc-paths \
  --require-all-labels-matched --skip-existing --no-compress --profile-jsonl "$COWP_ROOT/profile_tensor_cache_train.jsonl" --cpu-only
run cache_val "$PYTHON_BIN" -m cowp.scripts.02_build_tensor_cache \
  --data-config configs/data.yaml --split validation --tfexample-glob "$TFEXAMPLE_VAL" \
  --labels-dir "$LABELS_VAL" --output-dir "$CACHE_VAL" --num-workers "$CACHE_WORKERS" \
  --start-method forkserver --parallel-scan --require-waymax-ready --require-sdc-paths \
  --require-all-labels-matched --skip-existing --no-compress --profile-jsonl "$COWP_ROOT/profile_tensor_cache_val.jsonl" --cpu-only

run verify_train "$PYTHON_BIN" -m cowp.scripts.60_verify_fresh_v16_8_9_cache --cache-dir "$CACHE_TRAIN" --allowlist "$ALLOW_TRAIN" --sample-scenes 0 --require-sdc-paths --output "$COWP_ROOT/verify_cache_train.json"
run verify_val "$PYTHON_BIN" -m cowp.scripts.60_verify_fresh_v16_8_9_cache --cache-dir "$CACHE_VAL" --allowlist "$ALLOW_VAL" --sample-scenes 0 --require-sdc-paths --output "$COWP_ROOT/verify_cache_val.json"

# Write the legacy-named manifest because the fresh-cache gate is the stable self-contained
# cache contract introduced in v16.8.9.  Raw and transport paths intentionally point to the
# same NPZ tree: transport supervision is serialized inline in fresh v16.8.22 caches.
"$PYTHON_BIN" - "$COWP_ROOT" "$CACHE_TRAIN" "$CACHE_VAL" "$ALLOW_TRAIN" "$ALLOW_VAL" <<'PY'
import importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); tr=Path(sys.argv[2]); va=Path(sys.argv[3])
m=importlib.import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol')
fp=m.current_fingerprint(Path.cwd())
lfp=m.current_label_semantic_fingerprint(Path.cwd())
(root/'build_fingerprint.sha256').write_text(fp+'\n',encoding='utf-8')
manifest={
  'schema_version':'cowp_v16_8_9_causal_audit_self_contained_data_v1',
  'producer_version':'v16.8.22',
  'build_fingerprint_sha256':fp,
  'label_semantic_fingerprint_sha256':lfp,
  'raw_train_cache':str(tr), 'raw_val_cache':str(va),
  'transport_train_cache':str(tr), 'transport_val_cache':str(va),
  'train_allowlist':str(Path(sys.argv[4])), 'val_allowlist':str(Path(sys.argv[5])),
  'transport_storage':'inline_same_npz',
  'womd_contract':'1.3.1+sdc_paths',
}
(root/'data_manifest_v16_8_9.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(root/'label_semantic_fingerprint.sha256').write_text(lfp+'\n',encoding='utf-8')
print(json.dumps(manifest,indent=2,ensure_ascii=False))
PY

run fresh_protocol "$PYTHON_BIN" -m cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol \
  --cowp-root "$COWP_ROOT" --raw-train "$CACHE_TRAIN" --raw-val "$CACHE_VAL" \
  --transport-train "$CACHE_TRAIN" --transport-val "$CACHE_VAL" --sample-scenes 512 \
  --output "$COWP_ROOT/fresh_cache_protocol.json"

# Training-facing support.  These are evidence/learnability checks, not population-prevalence gates.
for split in train val; do
  cache_var="CACHE_${split^^}"; cache="${!cache_var}"
  run "supervision_${split}" "$PYTHON_BIN" -m cowp.scripts.62_audit_training_supervision \
    --cache-dir "$cache" --output "$COWP_ROOT/training_supervision_${split}.json" --sample-scenes 0 --strict
  run "model_support_${split}" "$PYTHON_BIN" -m cowp.scripts.65_audit_model_support \
    --cache-dir "$cache" --output "$COWP_ROOT/model_support_${split}.json" --sample-scenes 0 --strict \
    --max-unauditable-critical-rate 0.05 --min-certificate-complete-scene-rate 0.75 --min-protected-prio-coverage 0.98
  run "mechanism_contrast_${split}" "$PYTHON_BIN" -m cowp.scripts.74_audit_mechanism_contrast \
    --cache-dir "$cache" --output "$COWP_ROOT/mechanism_contrast_${split}.json" --sample-scenes 0 --strict
  run "causal_${split}" "$PYTHON_BIN" -m cowp.scripts.57_diagnose_causal_audit \
    --cache-dir "$cache" --output "$COWP_ROOT/causal_audit_${split}.json" --limit 0
done

"$PYTHON_BIN" - "$COWP_ROOT" <<'PY'
import importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
m=importlib.import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol')
fp=m.current_fingerprint(Path.cwd())
reports={}
for name in ('fresh_cache_protocol','training_supervision_train','training_supervision_val','model_support_train','model_support_val','mechanism_contrast_train','mechanism_contrast_val'):
    p=root/f'{name}.json'; reports[name]=json.loads(p.read_text(encoding='utf-8'))
def ok(x):
    if 'pass' in x: return bool(x['pass'])
    if 'recommend_full_rebuild' in x: return bool(x['recommend_full_rebuild'])
    return not bool(x.get('reasons'))
checks={k:ok(v) for k,v in reports.items()}
payload={
  'schema_version':'cowp_v16_8_22_full_core_support_v1',
  'code_fingerprint_sha256':fp,
  'pass':all(checks.values()),
  'checks':checks,
  'failed_checks':[k for k,v in checks.items() if not v],
  'next_action':'Attach Waymax outcomes, then train/evaluate.' if all(checks.values()) else 'Do not attach outcomes or train; inspect failed_checks.',
}
(root/'full_core_support_verdict_v16_8_22.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(payload,indent=2,ensure_ascii=False))
if not payload['pass']:
    raise SystemExit(2)
PY

echo "V16.8.22 FULL CORE BUILD + SIX-LAYER SUPPORT AUDIT PASS: $COWP_ROOT"
echo "Next: COWP_ROOT=$COWP_ROOT bash ATTACH_WAYMAX_OUTCOMES_V16_8_10_CN.sh"
