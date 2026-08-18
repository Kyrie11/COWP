#!/usr/bin/env bash
set -euo pipefail

# COWP v16.8.20 dataset-support wrapper.
# IMPORTANT: v16.8.20 changes label semantics (causal critical selection,
# ego-ranked conflict regions, explicit protected-priority candidate labels), so
# v16.8.19 smoke/strict NPZ files are NOT valid inputs for reaudit.
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export SMOKE_ROOT="${SMOKE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_20_support_smoke}"
export PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_20_support_strict_probe}"
export TRAIN_PILOT_ROOT="${TRAIN_PILOT_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_20_train_pilot}"
export FULL_ROOT="${FULL_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_20_full}"

mode="${1:-}"
case "$mode" in
  smoke|fresh-smoke)
    # Always fresh: every v16.8.19 label cache encodes stale selector/conflict/
    # priority-candidate semantics.
    exec bash NEXT_EXECUTION_V16_8_18_CN.sh fresh-smoke
    ;;
  reaudit-smoke)
    echo "v16.8.20 changes label semantics; reaudit-smoke is disabled. Use smoke/fresh-smoke." >&2
    exit 2
    ;;
  split-audit|preflight|fastpath-ab|strict|train-pilot|full-core|outcomes|check)
    exec bash NEXT_EXECUTION_V16_8_18_CN.sh "$mode"
    ;;
  *)
    echo "Usage: bash $0 {split-audit|preflight|smoke|fastpath-ab|strict|train-pilot|full-core|outcomes|check}" >&2
    exit 2
    ;;
esac
