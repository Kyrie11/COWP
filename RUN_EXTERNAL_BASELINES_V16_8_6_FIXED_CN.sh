#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"

# Re-run learned external baselines after the v16.8.6 numerical/data-frame fix.
# The legacy-bank default is ONLY a numerical/training smoke.  GameFormer/DTPP
# score the candidate bank, so final paper comparison must retrain them on the
# same fresh v16.8.6 bank used by COWP.  Do not interpret legacy-bank learned
# Waymax as a fair final comparison.
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export TRAIN_CACHE="${TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax_transport_v16_8}"
export VAL_CACHE="${VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8}"
export OUT_ROOT="${OUT_ROOT:-outputs/external_baselines_v16_8_6_fixed_oldbank}"
export LABEL_CONFIG="${LABEL_CONFIG:-configs/label_cowp_v16_8.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_cowp_v16_8.yaml}"
export MODE="${MODE:-smoke}"
# Legacy-bank default: numerical training/offline smoke only. Set 1 only for a
# fair fresh-bank run after rebuilding.
export RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-0}"
export BASELINES="${BASELINES:-gameformer dtpp idm_lattice frenet_optimal state_lattice}"
export AMP="${AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-auto}"
export MAX_SKIP_FRACTION="${MAX_SKIP_FRACTION:-0.02}"
export FORCE_RERUN="${FORCE_RERUN:-1}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

exec bash "$ROOT/run_external_baselines.sh"
