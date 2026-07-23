#!/usr/bin/env bash
set -euo pipefail

COWP_CODE_ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$COWP_CODE_ROOT"

export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v15}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 先执行快速回归测试；任何因果、坐标或指标协议测试失败都会终止。
pytest -q

# v15 修改了自然替代标签，因此必须使用新目录重建标签、tensor cache、Waymax outcome 附加和 transport overlay。
RUN_WAYMAX_REPLAY="${RUN_WAYMAX_REPLAY:-1}" \
RUN_TRANSPORT_AUGMENT=1 \
bash prepare_cowp_v15_data.sh

export RAW_TRAIN_CACHE="$COWP_ROOT/tensor_cache_train_waymax"
export RAW_VAL_CACHE="$COWP_ROOT/tensor_cache_val_waymax"
export TRAIN_CACHE="$COWP_ROOT/tensor_cache_train_waymax_transport_v15"
export VAL_CACHE="$COWP_ROOT/tensor_cache_val_waymax_transport_v15"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v15_causal_natural_seed2026}"
export TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-0,1}"
export TRAIN_NPROC="${TRAIN_NPROC:-2}"
export RUN_AUGMENT=0
export RUN_DIAGNOSE=1
export RUN_NATURAL=1
export RUN_TRANSPORT=1
export RUN_PLANNER=1
export RUN_OFFLINE=1
export RUN_PROBE=1
export RUN_FULL="${RUN_FULL:-0}"
export FORCE_TRAIN="${FORCE_TRAIN:-0}"
export FORCE_EVAL="${FORCE_EVAL:-0}"

bash run_cowp_v15_dual_gpu.sh
