#!/usr/bin/env bash
set -euo pipefail

# COWP v16.8.22 six-layer mechanism-support driver.
# Scenario->label semantics are unchanged from v16.8.20/v16.8.21.
# v16.8.22 changes:
#   (1) protected-priority becomes the exact primary BCOT/planner target;
#   (2) all-critical remains an auxiliary global stress head;
#   (3) train-pilot promotion uses evidence counts + within-root intervention contrast,
#       not an all-critical population-prevalence threshold.

export PYTHON_BIN="${PYTHON_BIN:-python}"
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export SOURCE_STRICT_ROOT="${SOURCE_STRICT_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_21_support_strict_reaudit}"
export SOURCE_TRAIN_PILOT_ROOT="${SOURCE_TRAIN_PILOT_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_21_train_pilot}"
export SOURCE_TRAIN_CACHE="${SOURCE_TRAIN_CACHE:-$SOURCE_TRAIN_PILOT_ROOT/labels_train_v16_8_18}"
export PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_22_support_strict_reaudit}"
export TRAIN_PILOT_ROOT="${TRAIN_PILOT_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_22_train_pilot_reaudit}"
export FULL_ROOT="${FULL_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_22_full}"
export FRESH_TRAIN_PILOT_ROOT="${FRESH_TRAIN_PILOT_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_22_train_pilot_fresh_source}"

usage(){
  cat <<'EOF'
Usage: bash NEXT_EXECUTION_V16_8_22_CN.sh <mode>

Recommended:
  reaudit-strict
  reaudit-train-pilot
  full-core
  outcomes
  check

Other modes:
  fresh-train-pilot   Rebuild a fresh 1200-scene pilot, then apply the v16.8.22 gate.
  mechanism-audit     Run only the new within-scene/root mechanism-contrast audit.
  split-audit | preflight

Important:
  - v16.8.22 does not change Scenario->label semantics.
  - Do NOT rebuild the v16.8.21 strict labels merely because model/training code changed.
  - The compact zip uploaded for review has no NPZs; reaudit-train-pilot must be run
    on the machine retaining labels_train_v16_8_18, or set SOURCE_TRAIN_CACHE.
EOF
}

require_current_strict(){
  local verdict="$PROBE_ROOT/v16_8_18_strict_verdict.json"
  [[ -s "$verdict" ]] || { echo "Missing v16.8.22 strict verdict: $verdict" >&2; exit 4; }
  "$PYTHON_BIN" - "$verdict" <<'PY'
import importlib,json,sys
from pathlib import Path
r=json.load(open(sys.argv[1],encoding='utf-8'))
m=importlib.import_module('cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol')
fp=m.current_fingerprint(Path.cwd())
assert r.get('recommend_full_rebuild') is True, 'STRICT DOES NOT AUTHORIZE TRAIN PILOT'
assert r.get('code_fingerprint_sha256') == fp, 'STRICT/CODE FINGERPRINT MISMATCH; rerun reaudit-strict'
PY
}

reaudit_strict(){
  rm -rf "$PROBE_ROOT"
  mkdir -p "$PROBE_ROOT"
  "$PYTHON_BIN" -m cowp.scripts.76_reaudit_v16_8_21_strict_for_v16_8_22 \
    --source-strict-root "$SOURCE_STRICT_ROOT" --output-root "$PROBE_ROOT"
}

mechanism_audit(){
  [[ -d "$SOURCE_TRAIN_CACHE" ]] || { echo "SOURCE_TRAIN_CACHE missing: $SOURCE_TRAIN_CACHE" >&2; exit 4; }
  mkdir -p "$TRAIN_PILOT_ROOT"
  "$PYTHON_BIN" -m cowp.scripts.74_audit_mechanism_contrast \
    --cache-dir "$SOURCE_TRAIN_CACHE" --output "$TRAIN_PILOT_ROOT/mechanism_contrast_audit.json" \
    --sample-scenes 0 --strict
}

reaudit_train_pilot(){
  require_current_strict
  rm -rf "$TRAIN_PILOT_ROOT"
  mkdir -p "$TRAIN_PILOT_ROOT"
  "$PYTHON_BIN" -m cowp.scripts.75_reaudit_v16_8_21_train_pilot \
    --source-train-root "$SOURCE_TRAIN_PILOT_ROOT" \
    --cache-dir "$SOURCE_TRAIN_CACHE" \
    --output-root "$TRAIN_PILOT_ROOT"
}

fresh_train_pilot(){
  require_current_strict
  rm -rf "$FRESH_TRAIN_PILOT_ROOT" "$TRAIN_PILOT_ROOT"
  mkdir -p "$FRESH_TRAIN_PILOT_ROOT"
  # The historical builder may exit non-zero only because its embedded v16.8.18
  # prevalence gate is obsolete. Preserve the generated NPZ/evidence and let the
  # v16.8.22 re-audit decide scientific support.
  set +e
  PROBE_ROOT="$PROBE_ROOT" TRAIN_PILOT_ROOT="$FRESH_TRAIN_PILOT_ROOT" \
    SOURCE_DATA_ROOT="$SOURCE_DATA_ROOT" WOMD_ROOT="$WOMD_ROOT" HARD_DEFINITION=protected \
    bash NEXT_EXECUTION_V16_8_18_CN.sh train-pilot
  local legacy_rc=$?
  set -e
  echo "legacy train-pilot driver return code=$legacy_rc (its old prevalence gate is not authoritative in v16.8.22)"
  [[ -d "$FRESH_TRAIN_PILOT_ROOT/labels_train_v16_8_18" ]] || {
    echo "Fresh pilot did not produce labels; this is a real build failure." >&2
    exit 5
  }
  "$PYTHON_BIN" -m cowp.scripts.75_reaudit_v16_8_21_train_pilot \
    --source-train-root "$FRESH_TRAIN_PILOT_ROOT" \
    --cache-dir "$FRESH_TRAIN_PILOT_ROOT/labels_train_v16_8_18" \
    --output-root "$TRAIN_PILOT_ROOT"
}

full_core(){
  require_current_strict
  local tv="$TRAIN_PILOT_ROOT/v16_8_18_train_pilot_verdict.json"
  [[ -s "$tv" ]] || { echo "Missing v16.8.22 train-pilot verdict: $tv" >&2; exit 4; }
  # v16.8.22 is self-contained.  The v16.8.21 archive inherited a call to a
  # PREPARE_COWP_V16_8_9_DATA_FAST_CN.sh file that is not present in the distributed repo.
  # Rebuild fresh labels/cache here instead of delegating to that broken historical wrapper.
  COWP_ROOT="$FULL_ROOT" SOURCE_DATA_ROOT="$SOURCE_DATA_ROOT" WOMD_ROOT="$WOMD_ROOT" \
    REUSE_OLD_SCENE_SET="${REUSE_OLD_SCENE_SET:-1}" \
    bash PREPARE_COWP_V16_8_22_DATA_CN.sh
}

outcomes(){
  COWP_ROOT="$FULL_ROOT" bash ATTACH_WAYMAX_OUTCOMES_V16_8_10_CN.sh
}

check(){
  echo "SOURCE_STRICT_ROOT=$SOURCE_STRICT_ROOT"
  echo "SOURCE_TRAIN_PILOT_ROOT=$SOURCE_TRAIN_PILOT_ROOT"
  echo "SOURCE_TRAIN_CACHE=$SOURCE_TRAIN_CACHE"
  echo "PROBE_ROOT=$PROBE_ROOT"
  echo "TRAIN_PILOT_ROOT=$TRAIN_PILOT_ROOT"
  echo "FULL_ROOT=$FULL_ROOT"
  for f in "$PROBE_ROOT/v16_8_22_strict_verdict.json" "$TRAIN_PILOT_ROOT/v16_8_22_train_pilot_verdict.json" "$FULL_ROOT/full_core_support_verdict_v16_8_22.json"; do
    if [[ -s "$f" ]]; then
      "$PYTHON_BIN" - "$f" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
print(sys.argv[1])
print('  pass=',r.get('pass'),' recommend_full_rebuild=',r.get('recommend_full_rebuild'))
print('  failed_checks=',r.get('failed_checks'))
print('  next_action=',r.get('next_action'))
PY
    else
      echo "$f : missing"
    fi
  done
}

mode="${1:-}"
case "$mode" in
  reaudit-strict) reaudit_strict ;;
  reaudit-train-pilot|train-pilot) reaudit_train_pilot ;;
  fresh-train-pilot) fresh_train_pilot ;;
  mechanism-audit) mechanism_audit ;;
  full-core) full_core ;;
  outcomes) outcomes ;;
  check) check ;;
  split-audit|preflight)
    PROBE_ROOT="$PROBE_ROOT" TRAIN_PILOT_ROOT="$TRAIN_PILOT_ROOT" FULL_ROOT="$FULL_ROOT" \
      SOURCE_DATA_ROOT="$SOURCE_DATA_ROOT" WOMD_ROOT="$WOMD_ROOT" \
      exec bash NEXT_EXECUTION_V16_8_18_CN.sh "$mode"
    ;;
  *) usage; exit 2 ;;
esac
