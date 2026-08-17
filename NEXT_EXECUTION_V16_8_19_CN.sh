#!/usr/bin/env bash
set -euo pipefail

# v16.8.19 dataset-support wrapper.
# The v16.8.18 scripts remain the audited gate implementation; this wrapper only
# isolates outputs and prevents semantic reuse of pre-patch smoke labels.
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export SMOKE_ROOT="${SMOKE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_19_support_smoke}"
export PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_19_support_strict_probe}"
export TRAIN_PILOT_ROOT="${TRAIN_PILOT_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_19_train_pilot}"
export FULL_ROOT="${FULL_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_19_full}"

mode="${1:-}"
case "$mode" in
  smoke|fresh-smoke)
    # Code/label semantics changed: a policy-only re-audit of v16.8.16 labels is invalid.
    exec bash NEXT_EXECUTION_V16_8_18_CN.sh fresh-smoke
    ;;
  reaudit-smoke)
    echo "v16.8.19 changes label semantics; reaudit-smoke is intentionally disabled. Use smoke (fresh build)." >&2
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
