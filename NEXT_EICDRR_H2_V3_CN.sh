#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TASK="${1:-help}"
BASE="${EICDRR_FULL_ROOT:-/data0/senzeyu2/dataset/COWP/eicdrr_full_validation_20261014}"
RAW="${RAW_WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
COHORT="$BASE/cohort"
OUT="$BASE/h2_gate_v3"
IDS="${H2_PRIORITY_IDS:-$BASE/h2_gate_v2/h2_priority_dev_ids.txt}"
mkdir -p "$OUT"
case "$TASK" in
 tests) PYTHONPATH=. python -m pytest -q tests --ignore=tests/test_core.py ;;
 inspect_waymax)
   python - <<'PY'
import importlib,inspect
for name in ['tensorflow','jax','waymax','waymax.agents','waymax.env','waymax.config']:
 try:
  mod=importlib.import_module(name)
  print(name,'OK',getattr(mod,'__version__','version unprovided'))
 except Exception as e:print(name,'NOT READY',repr(e))
try:
 from waymax import agents,config
 print('IDMRoutePolicy',inspect.signature(agents.IDMRoutePolicy))
 print('WOD_1_3_1_VALIDATION',hasattr(config,'WOD_1_3_1_VALIDATION'))
except Exception as e:print('WAYMAX ADAPTER NOT READY',repr(e))
PY
 ;;
 focal_calibration)
   python -m research_probe.h2_observational_stratified --cohort "$COHORT" --partition development \
     --max-scenes "${FIT_DEV_N:-320}" --steps 10 --output "$OUT/observational_fit_focal.json" ;;
 fixed_prefix_4_to_8)
   python -m research_probe.h2_fixed_prefix_audit --cohort "$COHORT" \
     --ids-file "$IDS" --short-steps 20 --long-steps 40 --output "$OUT/fixed_prefix_4_to_8.json" ;;
 distinct_rule_probe)
   python -m research_probe.h2_ttc_rule_probe --cohort "$COHORT" \
     --ids-file "$IDS" --steps 20 --output "$OUT/ttc_rule_dev.json" ;;
 distinct_rule_4_to_8)
   python -m research_probe.h2_ttc_rule_probe --cohort "$COHORT" \
     --ids-file "$IDS" --steps 40 --pulse-steps 20 --output "$OUT/ttc_rule_fixed_prefix_8s.json" ;;
 physical_waymax_pilot)
   python -m research_probe.waymax_paired_idm --cohort "$COHORT" --raw-root "$RAW" \
     --scenario-ids-file "$IDS" --partition development --max-scenes "${WAYMAX_PILOT_N:-22}" \
     --steps 80 --validate-map --output "$OUT/waymax_idm_route_policy_8s.json" ;;
 h1_joint_witness_check)
   : "${H1_JOINT_WITNESSES_JSONL:?Export full joint responses with xy, yaw and valid road mask first}"
   python -m research_probe.h1_joint_witness_gate --jsonl "$H1_JOINT_WITNESSES_JSONL" \
     --output "$OUT/h1_joint_witness.json" ;;
 independent_model_check)
   : "${MODEL_PAIR_JSONL:?Must contain ACTUAL same-actions paired trials from TWO independent response implementations}"
   python -m research_probe.independent_model_gate --jsonl "$MODEL_PAIR_JSONL" \
     --output "$OUT/independent_model_gate.json" ;;
 planner_choice_check)
   : "${PLANNER_TRIALS_JSONL:?Must export actually executed baseline planner choices on SHARED candidate actions}"
   python -m research_probe.h4_planner_choice_gate --jsonl "$PLANNER_TRIALS_JSONL" \
     --output "$OUT/planner_choice_gate.json" ;;
 interactive_inventory)
   python -m research_probe.split_inventory --raw-root "$RAW" --split validation_interactive \
     --output "$OUT/validation_interactive_inventory.json" ;;
 *) cat <<'EOF'
Steps (development, no blind holdout):
  tests
  focal_calibration
  fixed_prefix_4_to_8
  distinct_rule_probe
  inspect_waymax
  physical_waymax_pilot
  interactive_inventory
Then after ACTUAL exports from independent closed-loop response models and baseline planners:
  independent_model_check
  h1_joint_witness_check
  planner_choice_check

WARNING: Waymax pilot changes ego TARGET SPEED (not cosine accel): supports
physical/model sensitivity exploratory pilot only, not same-action H2 closure.
TTC rule uses shared simplified simulator and is NOT independently calibrated.
EOF
 ;;
esac
