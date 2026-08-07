#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_5_bcs_rmr_bcte}"
export COWP_ROOT="$DATA_ROOT"
# External planners only need fresh WOMD-state/core cache.  They do not need the
# RCOT transport overlay or cached candidate Waymax outcomes.
export TRAIN_CACHE="${TRAIN_CACHE:-$DATA_ROOT/tensor_cache_train}"
export VAL_CACHE="${VAL_CACHE:-$DATA_ROOT/tensor_cache_val}"
export OUT_ROOT="${OUT_ROOT:-outputs/external_baselines_v16_8_5_fresh}"
export MODE="${MODE:-smoke}"  # smoke first; set full only after COWP probe is healthy.
export BASELINES="${BASELINES:-gameformer dtpp idm_lattice frenet_optimal state_lattice}"
export PARALLEL_LEARNED_TRAIN="${PARALLEL_LEARNED_TRAIN:-1}"
export AMP="${AMP:-1}"
exec bash "$ROOT/run_external_baselines.sh"
