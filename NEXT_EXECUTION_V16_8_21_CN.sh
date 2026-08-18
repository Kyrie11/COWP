#!/usr/bin/env bash
set -euo pipefail

# COWP v16.8.21 evidence-aligned support wrapper.
# v16.8.21 does NOT change Scenario->label semantics relative to v16.8.20.
# Therefore a reviewed v16.8.20 strict evidence bundle may be policy-re-audited
# without rebuilding its NPZ labels, provided both provenance fingerprints match.

export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export SOURCE_STRICT_ROOT="${SOURCE_STRICT_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_20_support_strict_probe}"
export SMOKE_ROOT="${SMOKE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_21_support_smoke}"
export PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_21_support_strict_reaudit}"
export TRAIN_PILOT_ROOT="${TRAIN_PILOT_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_21_train_pilot}"
export FULL_ROOT="${FULL_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_21_full}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

mode="${1:-}"
case "$mode" in
  reaudit-strict)
    if [[ "$(readlink -f "$PROBE_ROOT")" == "$(readlink -f "$SOURCE_STRICT_ROOT")" ]]; then
      echo "PROBE_ROOT must differ from SOURCE_STRICT_ROOT; refusing to overwrite reviewed v16.8.20 evidence." >&2
      exit 3
    fi
    rm -rf "$PROBE_ROOT"
    mkdir -p "$PROBE_ROOT"
    "$PYTHON_BIN" -m cowp.scripts.73_reaudit_v16_8_20_strict_policy \
      --source-strict-root "$SOURCE_STRICT_ROOT" \
      --output-root "$PROBE_ROOT"
    ;;
  fresh-smoke|smoke)
    # Optional independent rerun.  Because policy code changes the broad code
    # fingerprint, fresh smoke is required before a genuinely fresh strict run.
    exec bash NEXT_EXECUTION_V16_8_18_CN.sh fresh-smoke
    ;;
  strict)
    exec bash NEXT_EXECUTION_V16_8_18_CN.sh strict
    ;;
  split-audit|preflight|fastpath-ab|train-pilot|full-core|outcomes|check)
    exec bash NEXT_EXECUTION_V16_8_18_CN.sh "$mode"
    ;;
  *)
    cat >&2 <<'EOF'
Usage: bash NEXT_EXECUTION_V16_8_21_CN.sh {
  reaudit-strict | fresh-smoke | strict | train-pilot | full-core | outcomes | check |
  split-audit | preflight | fastpath-ab
}

Recommended v16.8.21 path for an already-complete reviewed v16.8.20 strict probe:
  reaudit-strict -> train-pilot -> full-core -> outcomes

Use fresh-smoke -> strict only if you intentionally want to rebuild the probe evidence
under the new broad code fingerprint.  Label semantics are unchanged.
EOF
    exit 2
    ;;
esac
