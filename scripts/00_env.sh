#!/usr/bin/env bash
# Source this file:  source scripts/00_env.sh
_RIDE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RIDE_HOME="${RIDE_HOME:-$_RIDE_ROOT}"
export CARLA_ROOT="${CARLA_ROOT:-/data0/senzeyu2/dataset/CARLA_0.9.16}"
export DATASET_ROOT="${DATASET_ROOT:-/data0/senzeyu2/dataset}"
export RAW_WOMD_ROOT="${RAW_WOMD_ROOT:-$DATASET_ROOT/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
export RIDE_OUTPUT_ROOT="${RIDE_OUTPUT_ROOT:-$DATASET_ROOT/COWP/ride_evidence_20261026}"
export PYTHONPATH="$RIDE_HOME:$RIDE_HOME/COWP:$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
