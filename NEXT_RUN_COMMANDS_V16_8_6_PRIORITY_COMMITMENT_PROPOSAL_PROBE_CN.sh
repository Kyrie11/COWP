#!/usr/bin/env bash
set -euo pipefail

# Paper-grade proposal promotion probe after the 192-scene micro screen passes.
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export OLD_VAL_CACHE="${OLD_VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8}"
export PROBE_ROOT="${PROBE_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment_proposal_probe}"
export HARD_COUNT="${HARD_COUNT:-400}"
export RANDOM_COUNT="${RANDOM_COUNT:-800}"
export LABEL_WORKERS="${LABEL_WORKERS:-24}"
export SEED="${SEED:-2026}"
export FORCE_REBUILD_PROBE="${FORCE_REBUILD_PROBE:-1}"

bash NEXT_RUN_COMMANDS_V16_8_4_PROPOSAL_PROBE_CN.sh

python - "$PROBE_ROOT/paired_proposal_probe.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
if not p.get('promote_to_full_rebuild',False):
    raise SystemExit('STRICT PROBE FAILED: do not full-rebuild.')
print('STRICT PROBE PASSED: full fresh v16.8.6 rebuild is now justified.')
PY
