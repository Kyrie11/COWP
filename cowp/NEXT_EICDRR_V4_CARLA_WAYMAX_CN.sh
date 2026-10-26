#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TASK="${1:-help}"
ROOT="${EICDRR_FULL_ROOT:-/data0/senzeyu2/dataset/COWP/eicdrr_full_validation_20261014}"
RAW="${RAW_WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
IDS="${H2_PRIORITY_IDS:-$ROOT/h2_gate_v2/h2_priority_dev_ids.txt}"
OUT="${EICDRR_V4_ROOT:-$ROOT/h2_gate_v4}"
TOWN="${CARLA_TOWN:-Town04}"
mkdir -p "$OUT"
case "$TASK" in
 tests)
  PYTHONPATH=. python -m pytest -q tests --ignore=tests/test_core.py ;;
 waymax_inspect)
  PYTHONPATH=. python - <<'PY'
import inspect,sys
import waymax
from waymax import env,config,agents,dataloader
for x in ['BaseEnvironment','MultiAgentEnvironment']:
 print('waymax.env.'+x, getattr(env,x,None))
print('IDMRoutePolicy:',getattr(agents,'IDMRoutePolicy',None))
print('WOD_1_3_1_VALIDATION:',getattr(config,'WOD_1_3_1_VALIDATION',None))
print('Python:',sys.version)
PY
  ;;
 waymax_smoke)
  PYTHONPATH=. python -m research_probe.waymax_paired_idm \
    --cohort "$ROOT/cohort" --raw-root "$RAW" --scenario-ids-file "$IDS" \
    --partition development --max-scenes 1 --steps 40 --validate-map \
    --output "$OUT/waymax_smoke_1.json" ;;
 waymax_priority22)
  PYTHONPATH=. python -m research_probe.waymax_paired_idm \
    --cohort "$ROOT/cohort" --raw-root "$RAW" --scenario-ids-file "$IDS" \
    --partition development --max-scenes 22 --steps 80 --validate-map \
    --output "$OUT/waymax_priority22.json" ;;
 carla_preflight)
  PYTHONPATH=. python - <<'PY'
import carla,sys
from agents.navigation.controller import VehiclePIDController
from agents.navigation.behavior_agent import BehaviorAgent
c=carla.Client('127.0.0.1',2000);c.set_timeout(8)
w=c.get_world()
print('CARLA client imported:',carla.__file__)
print('CARLA server:',c.get_server_version(),'Client:',c.get_client_version())
print('Current map:',w.get_map().name)
print('Python:',sys.version)
if not str(c.get_server_version()).startswith('0.9.16'):
 raise RuntimeError('Expected CARLA 0.9.16 server')
PY
  ;;
 carla_smoke)
  PYTHONPATH=. python -m research_probe.carla_h2.run_carla \
    --town "$TOWN" --anchors 1 --seed 17 --repetitions 1 \
    --families tm behavior --profiles cruise early_slow late_slow \
    --output "$OUT/carla_smoke.jsonl"
  PYTHONPATH=. python -m research_probe.carla_h2.analyze_carla \
    --jsonl "$OUT/carla_smoke.jsonl" --out "$OUT/carla_smoke_analysis.json" ;;
 carla_mechanism_dev)
  PYTHONPATH=. python -m research_probe.carla_h2.run_carla \
    --town "$TOWN" --anchors "${CARLA_ANCHORS:-3}" --seed 17 --repetitions "${CARLA_REPEATS:-3}" \
    --families tm behavior --profiles cruise early_slow late_slow mild_early mild_late \
    --output "$OUT/carla_mechanism_dev.jsonl"
  PYTHONPATH=. python -m research_probe.carla_h2.analyze_carla \
    --jsonl "$OUT/carla_mechanism_dev.jsonl" --out "$OUT/carla_mechanism_analysis.json" ;;
 carla_joint_h1_pilot)
  PYTHONPATH=. python -m research_probe.carla_h2.run_carla \
    --town "$TOWN" --anchors 1 --seed 17 --repetitions 1 \
    --families tm behavior --profiles early_slow \
    --joint-profile early_slow --joint-limit "${JOINT_LIMIT:-9}" \
    --joint-target-grid 6 9 12 --output "$OUT/carla_joint_h1.jsonl"
  PYTHONPATH=. python -m research_probe.carla_h2.analyze_carla \
    --jsonl "$OUT/carla_joint_h1.jsonl" --out "$OUT/carla_joint_h1_analysis.json" ;;
 carla_h3_backup_pilot)
  PYTHONPATH=. python -m research_probe.carla_h2.run_carla \
    --town "$TOWN" --anchors 1 --seed 17 --repetitions 1 \
    --families tm behavior --profiles early_slow \
    --recovery-profile early_slow --recovery-time-s 3.5 --recovery-options stop cruise \
    --output "$OUT/carla_h3_backup.jsonl"
  PYTHONPATH=. python -m research_probe.carla_h2.analyze_carla \
    --jsonl "$OUT/carla_h3_backup.jsonl" --out "$OUT/carla_h3_backup_analysis.json" ;;
 *)
 cat <<'EOF'
Usage: bash NEXT_EICDRR_V4_CARLA_WAYMAX_CN.sh COMMAND
  tests
  waymax_inspect             # Inspect installed Waymax API; no output claims
  waymax_smoke               # ONE actual WOMD map rollout; must succeed before 22
  waymax_priority22          # 22 cases; preliminary, NOT human response
  carla_preflight            # CARLA 0.9.16 server/client and PythonAPI check
  carla_smoke                # 1 corridor × 1 seed × 2 families × 3 actions
  carla_mechanism_dev        # 3 corridors × 3 seeds × 2 families × 5 actions
  carla_joint_h1_pilot       # finite constructive joint response candidates
  carla_h3_backup_pilot      # same initial action prefix, different backups
Important: CARLA is SYNTHETIC SCENE validation, not replay of WOMD!
Exports remain preliminary; frozen sealed holdout and real human counterfactual unavailable.
EOF
 ;;
esac
