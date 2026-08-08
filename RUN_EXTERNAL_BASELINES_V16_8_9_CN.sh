#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"; cd "$ROOT"
# Paper-grade matched baseline run. GameFormer/DTPP in this repository score the
# candidate bank, so train them only after the same v16.8.9 fresh bank exists.
export DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit}"
export COWP_ROOT="$DATA_ROOT"
export TRAIN_CACHE="${TRAIN_CACHE:-$DATA_ROOT/tensor_cache_train}"
export VAL_CACHE="${VAL_CACHE:-$DATA_ROOT/tensor_cache_val}"
export OUT_ROOT="${OUT_ROOT:-outputs/external_baselines_v16_8_9_matched}"
export LABEL_CONFIG="${LABEL_CONFIG:-configs/label_cowp_v16_8.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_cowp_v16_8.yaml}"
export MODE="${MODE:-smoke}"
export RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-0}"
export BASELINES="${BASELINES:-gameformer dtpp idm_lattice frenet_optimal state_lattice}"
export AMP="${AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-auto}"
export MAX_SKIP_FRACTION="${MAX_SKIP_FRACTION:-0.02}"
export FORCE_RERUN="${FORCE_RERUN:-1}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
exec bash "$ROOT/run_external_baselines.sh"
