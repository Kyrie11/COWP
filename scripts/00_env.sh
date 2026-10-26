#!/usr/bin/env bash
# Source or execute via other scripts. The package root is always derived from
# this file; Python modules NEVER depend on a sibling legacy source tree.
_RIDE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RIDE_HOME="$_RIDE_ROOT"
export CARLA_ROOT="${CARLA_ROOT:-/data0/senzeyu2/dataset/CARLA_0.9.16}"
export DATASET_ROOT="${DATASET_ROOT:-/data0/senzeyu2/dataset}"
export RAW_WOMD_ROOT="${RAW_WOMD_ROOT:-$DATASET_ROOT/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
export RIDE_OUTPUT_ROOT="${RIDE_OUTPUT_ROOT:-$DATASET_ROOT/RIDE/evidence}"
export PYTHONPATH="$RIDE_HOME${PYTHONPATH:+:$PYTHONPATH}"
# Only include the version-matched CARLA navigation-agent sources if present.
# `pip install carla` alone does not install BehaviorAgent.
if [[ -f "$CARLA_ROOT/PythonAPI/carla/agents/navigation/behavior_agent.py" ]]; then
  export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH"
fi
