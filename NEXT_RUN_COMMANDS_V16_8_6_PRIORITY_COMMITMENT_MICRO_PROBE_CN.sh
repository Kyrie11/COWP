#!/usr/bin/env bash
set -euo pipefail

# Cheap v16.8.6 screen: 64 old hard scenes + 128 unbiased validation scenes.
# This does NOT authorize a full rebuild. It only decides whether the new
# Priority-Commitment proposal family deserves the stricter 400+800 probe.
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export OLD_VAL_CACHE="${OLD_VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8}"
export PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment_micro_probe}"
export HARD_COUNT="${HARD_COUNT:-64}"
export RANDOM_COUNT="${RANDOM_COUNT:-128}"
export LABEL_WORKERS="${LABEL_WORKERS:-24}"
export SEED="${SEED:-2026}"
# v16.8.6 changes proposal semantics, so do not mix it with old v16.8.4/5 labels.
export FORCE_REBUILD_PROBE="${FORCE_REBUILD_PROBE:-1}"

bash NEXT_RUN_COMMANDS_V16_8_4_PROPOSAL_PROBE_CN.sh

python -m cowp.scripts.52_screen_priority_commitment_probe \
  --paired-probe "$PROBE_ROOT/paired_proposal_probe.json" \
  --output "$PROBE_ROOT/priority_commitment_micro_screen.json"

echo "MICRO_SCREEN=$PROBE_ROOT/priority_commitment_micro_screen.json"
echo "PASS only means: proceed to the strict 400+800 proposal probe. It does NOT mean full rebuild yet."
