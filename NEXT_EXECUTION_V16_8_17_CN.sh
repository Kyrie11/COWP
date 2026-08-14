#!/usr/bin/env bash
set -euo pipefail

# COWP v16.8.17 dataset-support promotion driver.
# It intentionally keeps the expensive stages separated so a failed smoke/strict
# probe cannot consume full-rebuild or Waymax rollout resources.
PYTHON_BIN="${PYTHON_BIN:-python}"
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export OLD_VAL_CACHE="${OLD_VAL_CACHE:-$SOURCE_DATA_ROOT/tensor_cache_val}"
export SMOKE_ROOT="${SMOKE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_17_support_smoke}"
export SOURCE_V16_8_16_SMOKE_ROOT="${SOURCE_V16_8_16_SMOKE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_16_support_smoke}"
export PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_17_support_strict_probe}"
export FULL_ROOT="${FULL_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_17_full}"
export TRAIN_PILOT_ROOT="${TRAIN_PILOT_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_17_train_pilot}"

SCENARIO_TRAIN="$WOMD_ROOT/uncompressed/scenario/training/*.tfrecord*"
SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
TFEXAMPLE_TRAIN="$WOMD_ROOT/uncompressed/tf_example/training/*.tfrecord*"
TFEXAMPLE_VAL="$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*"
SMOKE_VERDICT="$SMOKE_ROOT/v16_8_17_smoke_verdict.json"
STRICT_VERDICT="$PROBE_ROOT/v16_8_17_strict_verdict.json"
TRAIN_PILOT_VERDICT="$TRAIN_PILOT_ROOT/v16_8_17_train_pilot_verdict.json"

usage(){
  cat <<USAGE
Usage: bash $0 <mode>

Modes:
  split-audit   Inventory local WOMD primary/challenge splits; optionally audit validation_interactive overlap.
  preflight     Verify COMPLETE Scenario/tf.Example train+val WOMD 1.3.1 shard + sdc_paths contract.
  reaudit-smoke Re-evaluate reviewed v16.8.16 smoke labels under the v16.8.17 coverage policy; no label rebuild.
  smoke         Fresh 96-scene smoke: prefer up to 48 current-cache hard scenes, fill remainder randomly.
  fastpath-ab   Rebuild a small smoke subset with the exact fast path disabled and require bitwise label equality.
  strict        Fresh 1200-scene strict probe: prefer up to 400 hard, fill remainder randomly; require >=128 hard.
  train-pilot   Fresh 1200-scene TRAIN pilot: prefer up to 400 hard, fill remainder randomly; require >=128 hard.
  full-core     Build full fresh labels + tensor caches only after strict + train-pilot both authorize it; Waymax replay deferred.
  outcomes      Attach balanced Waymax safety outcomes to FULL_ROOT (delegates to ATTACH_WAYMAX_OUTCOMES_V16_8_10_CN.sh).
  check         Print strict/full promotion status and key generated paths.

Recommended order:
  split-audit -> preflight -> reaudit-smoke (preferred) OR fresh smoke -> strict -> train-pilot -> full-core -> outcomes
USAGE
}

reaudit_smoke(){
  rm -rf "$SMOKE_ROOT"
  mkdir -p "$SMOKE_ROOT"
  "$PYTHON_BIN" -m cowp.scripts.70_reaudit_v16_8_16_smoke_policy \
    --source-smoke-root "$SOURCE_V16_8_16_SMOKE_ROOT" \
    --output-root "$SMOKE_ROOT" \
    --max-unauditable-critical-rate 0.05 \
    --min-certificate-complete-scene-rate 0.75 \
    --min-protected-prio-coverage 0.95 \
    --max-auditability-stratum-gap 0.03 \
    --max-certificate-stratum-gap 0.10
}


split_audit(){
  mkdir -p "$SMOKE_ROOT"
  local extra=()
  if [[ "${FULL_INTERACTIVE_OVERLAP:-0}" == "1" ]]; then extra+=(--full-validation-interactive-overlap); fi
  "$PYTHON_BIN" -m cowp.scripts.69_audit_womd_split_layout \
    --womd-root "$WOMD_ROOT" --sample-scenario-shards "${SPLIT_AUDIT_SAMPLE_SHARDS:-16}" \
    "${extra[@]}" --output "$SMOKE_ROOT/womd_v1_3_1_split_layout.json"
}

preflight(){
  mkdir -p "$SMOKE_ROOT"
  "$PYTHON_BIN" -m cowp.scripts.64_validate_womd_v131_contract \
    --tfexample-train-glob "$TFEXAMPLE_TRAIN" --tfexample-val-glob "$TFEXAMPLE_VAL" \
    --scenario-train-glob "$SCENARIO_TRAIN" --scenario-val-glob "$SCENARIO_VAL" \
    --sample-shards "${PREFLIGHT_TF_SHARDS:-64}" --scenario-sample-shards "${PREFLIGHT_SCENARIO_SHARDS:-32}" \
    --require-sdc-paths --require-complete-primary-splits --output "$SMOKE_ROOT/womd_v1_3_1_preflight.json"
}

smoke(){
  export HARD_COUNT="${HARD_COUNT:-48}"
  export RANDOM_COUNT="${RANDOM_COUNT:-48}"
  export LABEL_WORKERS="${LABEL_WORKERS:-24}"
  export FORCE_REBUILD_SMOKE="${FORCE_REBUILD_SMOKE:-1}"
  export MIN_HARD_COUNT="${MIN_HARD_COUNT:-24}"
  set +e
  bash NEXT_RUN_COMMANDS_V16_8_17_CAUSAL_AUDIT_SMOKE_CN.sh
  local rc=$?
  set -e
  [[ -s "$SMOKE_ROOT/v16_8_17_smoke_verdict.json" ]] && cat "$SMOKE_ROOT/v16_8_17_smoke_verdict.json" || true
  return "$rc"
}

fastpath_ab(){
  local labels="$SMOKE_ROOT/labels_val_v16_8_17"
  local ids="$SMOKE_ROOT/union_scene_ids.txt"
  if [[ ! -d "$labels" && -d "$SOURCE_V16_8_16_SMOKE_ROOT/labels_val_v16_8_16" ]]; then
    # Policy-only re-audit path: label semantics are fingerprint-verified equal
    # to v16.8.16, so the already-built smoke labels are the correct reference.
    labels="$SOURCE_V16_8_16_SMOKE_ROOT/labels_val_v16_8_16"
    ids="$SOURCE_V16_8_16_SMOKE_ROOT/union_scene_ids.txt"
  fi
  local ab_root="${FASTPATH_AB_ROOT:-$SMOKE_ROOT/fastpath_ab}"
  local n="${FASTPATH_AB_SCENES:-12}"
  [[ -d "$labels" && -s "$ids" ]] || { echo "Run smoke first: missing $labels or $ids" >&2; exit 3; }
  rm -rf "$ab_root"
  mkdir -p "$ab_root/labels_no_fastpath"
  head -n "$n" "$ids" > "$ab_root/scene_ids.txt"
  "$PYTHON_BIN" - "$ab_root/label_no_fastpath.yaml" <<'PY'
from pathlib import Path
import yaml,sys
src=Path('configs/label_cowp_v16_8.yaml')
cfg=yaml.safe_load(src.read_text(encoding='utf-8'))
cfg.setdefault('engineering',{})['risk_known_zero_fastpath']=False
Path(sys.argv[1]).write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
PY
  "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
    --data-config configs/data.yaml --label-config "$ab_root/label_no_fastpath.yaml" \
    --proto-glob "$SCENARIO_VAL" --output-dir "$ab_root/labels_no_fastpath" \
    --allow-scenario-ids "$ab_root/scene_ids.txt" --num-workers "${FASTPATH_AB_WORKERS:-4}" \
    --start-method forkserver --max-pending-multiplier 2 --no-compress --skip-diagnostics --cpu-only
  "$PYTHON_BIN" -m cowp.scripts.66_compare_label_semantic_equivalence \
    --reference-dir "$labels" --candidate-dir "$ab_root/labels_no_fastpath" \
    --scene-ids "$ab_root/scene_ids.txt" --max-scenes 0 \
    --output "$ab_root/fastpath_semantic_equivalence.json"
  echo "FASTPATH A/B PASS: $ab_root/fastpath_semantic_equivalence.json"
}

strict(){
  [[ -s "$SMOKE_VERDICT" ]] || { echo "Missing composite smoke verdict: $SMOKE_VERDICT" >&2; exit 4; }
  "$PYTHON_BIN" - "$SMOKE_VERDICT" <<'PY'
import importlib,json,sys
from pathlib import Path
r=json.load(open(sys.argv[1],encoding='utf-8'))
assert r.get('recommend_strict_probe') is True, 'SMOKE DOES NOT AUTHORIZE STRICT PROBE'
fp=importlib.import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol').current_fingerprint(Path.cwd())
assert r.get('code_fingerprint_sha256') == fp, 'SMOKE/CODE FINGERPRINT MISMATCH; rerun smoke'
PY
  export HARD_COUNT="${HARD_COUNT:-400}"
  export RANDOM_COUNT="${RANDOM_COUNT:-800}"
  export SEED="${SEED:-2026}"
  export LABEL_WORKERS="${LABEL_WORKERS:-24}"
  export FORCE_REBUILD_PROBE="${FORCE_REBUILD_PROBE:-1}"
  export MIN_HARD_COUNT="${MIN_HARD_COUNT:-128}"
  set +e
  bash NEXT_RUN_COMMANDS_V16_8_17_STRICT_PROPOSAL_PROBE_CN.sh
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    [[ -s "$STRICT_VERDICT" ]] && cat "$STRICT_VERDICT" || true
    return "$rc"
  fi
  "$PYTHON_BIN" - "$STRICT_VERDICT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
print(json.dumps(r,indent=2,ensure_ascii=False))
assert r.get('recommend_full_rebuild') is True, 'DO NOT FULL REBUILD'
PY
}

train_pilot(){
  [[ -s "$STRICT_VERDICT" ]] || { echo "Missing validation strict verdict: $STRICT_VERDICT" >&2; exit 4; }
  "$PYTHON_BIN" - "$STRICT_VERDICT" <<'PY'
import importlib,json,sys
from pathlib import Path
r=json.load(open(sys.argv[1],encoding='utf-8'))
assert r.get('recommend_full_rebuild') is True, 'VALIDATION STRICT DOES NOT AUTHORIZE TRAIN PILOT'
fp=importlib.import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol').current_fingerprint(Path.cwd())
assert r.get('code_fingerprint_sha256') == fp, 'STRICT/CODE FINGERPRINT MISMATCH; rerun strict'
PY
  TRAIN_PILOT_ROOT="$TRAIN_PILOT_ROOT" SOURCE_DATA_ROOT="$SOURCE_DATA_ROOT" WOMD_ROOT="$WOMD_ROOT" \
    bash NEXT_TRAIN_PILOT_V16_8_17_CN.sh
}

full_core(){
  [[ -s "$STRICT_VERDICT" ]] || { echo "Missing strict verdict: $STRICT_VERDICT" >&2; exit 4; }
  [[ -s "$TRAIN_PILOT_VERDICT" ]] || { echo "Missing train-pilot verdict: $TRAIN_PILOT_VERDICT" >&2; exit 4; }
  "$PYTHON_BIN" - "$STRICT_VERDICT" "$TRAIN_PILOT_VERDICT" <<'PY'
import importlib,json,sys
from pathlib import Path
strict=json.load(open(sys.argv[1],encoding='utf-8'))
train=json.load(open(sys.argv[2],encoding='utf-8'))
assert strict.get('recommend_full_rebuild') is True, 'VALIDATION STRICT DOES NOT AUTHORIZE FULL REBUILD'
assert train.get('recommend_full_rebuild') is True, 'TRAIN PILOT DOES NOT AUTHORIZE FULL REBUILD'
m=importlib.import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol')
fp=m.current_fingerprint(Path.cwd())
assert strict.get('code_fingerprint_sha256') == fp, 'STRICT/CODE FINGERPRINT MISMATCH'
assert train.get('code_fingerprint_sha256') == fp, 'TRAIN PILOT/CODE FINGERPRINT MISMATCH'
assert strict.get('code_fingerprint_sha256') == train.get('code_fingerprint_sha256'), 'STRICT/TRAIN-PILOT FINGERPRINT MISMATCH'
print('validation strict + train-pilot both authorize full rebuild and match current code fingerprint')
PY
  export COWP_ROOT="$FULL_ROOT"
  export STRICT_VERDICT="$STRICT_VERDICT"
  export REUSE_OLD_SCENE_SET="${REUSE_OLD_SCENE_SET:-1}"
  export LABEL_WORKERS_TRAIN="${LABEL_WORKERS_TRAIN:-32}"
  export LABEL_WORKERS_VAL="${LABEL_WORKERS_VAL:-24}"
  export CACHE_WORKERS="${CACHE_WORKERS:-8}"
  export RUN_WAYMAX_REPLAY=0
  export RUN_LABEL_DIAGNOSTICS="${RUN_LABEL_DIAGNOSTICS:-0}"
  bash PREPARE_COWP_V16_8_9_DATA_FAST_CN.sh
  echo "CORE BUILD PASS. Do not run planner/mainline training yet; attach Waymax outcomes next."
}

outcomes(){
  COWP_ROOT="$FULL_ROOT" bash ATTACH_WAYMAX_OUTCOMES_V16_8_10_CN.sh
}

check(){
  echo "SMOKE_ROOT=$SMOKE_ROOT"
  echo "PROBE_ROOT=$PROBE_ROOT"
  echo "FULL_ROOT=$FULL_ROOT"
  echo "TRAIN_PILOT_ROOT=$TRAIN_PILOT_ROOT"
  if [[ -s "$SMOKE_VERDICT" ]]; then
    "$PYTHON_BIN" - "$SMOKE_VERDICT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
print('smoke recommend_strict_probe =',r.get('recommend_strict_probe'))
print('smoke next_action =',r.get('next_action'))
PY
  else
    echo "smoke verdict: missing"
  fi
  if [[ -s "$STRICT_VERDICT" ]]; then
    "$PYTHON_BIN" - "$STRICT_VERDICT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
print('strict recommend_full_rebuild =',r.get('recommend_full_rebuild'))
print('failure_stage =',r.get('failure_stage'))
print('next_action =',r.get('next_action'))
PY
  else
    echo "strict verdict: missing"
  fi
  if [[ -s "$TRAIN_PILOT_VERDICT" ]]; then
    "$PYTHON_BIN" - "$TRAIN_PILOT_VERDICT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
print('train-pilot recommend_full_rebuild =',r.get('recommend_full_rebuild'))
print('train-pilot next_action =',r.get('next_action'))
PY
  else
    echo "train-pilot verdict: missing"
  fi
  for p in "$FULL_ROOT/tensor_cache_train" "$FULL_ROOT/tensor_cache_val" "$FULL_ROOT/tensor_cache_train_waymax" "$FULL_ROOT/tensor_cache_val_waymax"; do
    [[ -d "$p" ]] && echo "$p : $(find "$p" -maxdepth 1 -type f -name '*.npz' | wc -l) npz" || true
  done
}

mode="${1:-}"
case "$mode" in
  split-audit) split_audit ;;
  preflight) preflight ;;
  reaudit-smoke) reaudit_smoke ;;
  smoke) smoke ;;
  fastpath-ab) fastpath_ab ;;
  strict) strict ;;
  train-pilot) train_pilot ;;
  full-core) full_core ;;
  outcomes) outcomes ;;
  check) check ;;
  *) usage; [[ -n "$mode" ]] && exit 2 || exit 0 ;;
esac
