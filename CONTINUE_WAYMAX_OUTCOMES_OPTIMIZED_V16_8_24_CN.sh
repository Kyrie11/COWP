#!/usr/bin/env bash
set -euo pipefail

# Resume the existing v16.8.24 full Waymax outcomes after the exact JIT gate.
# This script intentionally fixes the dataset-defining parameters to the values
# already used by the partial shard JSONLs.

export PYTHON_BIN="${PYTHON_BIN:-python}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
REPORT="${WAYMAX_JIT_CHECK_REPORT:-$COWP_ROOT/waymax_jit_equivalence_v16_8_24/compare.json}"

[[ -f "$REPORT" ]] || {
  echo "Missing JIT equivalence report: $REPORT" >&2
  echo "Run CHECK_WAYMAX_JIT_EQUIVALENCE_V16_8_24_CN.sh first." >&2
  exit 3
}
"$PYTHON_BIN" - "$REPORT" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
if not p.get('pass'):
    raise SystemExit('JIT equivalence report did not pass; refusing optimized full resume')
print('JIT semantic equivalence: PASS')
print('measured gate speedup_x:',p.get('speedup_x'))
PY

# Dataset-defining invariants: DO NOT change these while resuming the existing
# *_bal24_safety_shard000/001_of_002.jsonl files.
export WAYMAX_GPUS="${WAYMAX_GPUS:-0,1}"
export REPLAY_NUM_SHARDS=2
export MAX_REPLAY_CANDIDATES=24
export REPLAY_HORIZON=80

# Exact metric cadence / action mode / selection remain hard-coded in
# ATTACH_WAYMAX_OUTCOMES_V16_8_24_CN.sh: balanced, absolute_xy_yaw,
# safety, metric_eval_mode=step, interval=1, done_check_interval=1.

# Execution-only acceleration. These do not change the metric definitions.
export WAYMAX_JIT_ENV_STEP=1
export WAYMAX_JIT_SAFETY_METRICS=1
export WAYMAX_JIT_ENV_RESET=0
export WAYMAX_PROFILE_DETAIL="${WAYMAX_PROFILE_DETAIL:-scene}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export WAYMAX_JAX_COMPILATION_CACHE_DIR="${WAYMAX_JAX_COMPILATION_CACHE_DIR:-$COWP_ROOT/.jax_compilation_cache_waymax_v16_8_24}"

exec bash NEXT_EXECUTION_V16_8_24_CN.sh outcomes
