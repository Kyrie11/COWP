#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
CHOICES="${1:?Usage: bash scripts/06_compare_locked_planners.sh choices_locked.csv choices_locked.sha.json}"
COMMIT="${2:?Missing choices_locked.sha.json frozen before response rollouts}"
python -m ride.decision --rollouts "$RIDE_OUTPUT_ROOT/carla/all_scenes.jsonl" \
 --choices "$CHOICES" --commit-manifest "$COMMIT" --output "$RIDE_OUTPUT_ROOT/carla/planner_decision_regret.json"
python -m ride.gates --ranking "$RIDE_OUTPUT_ROOT/carla/risk_ranking_disagreement.json" --audit "$RIDE_OUTPUT_ROOT/carla/all_scenes_audit.json" \
 --decision "$RIDE_OUTPUT_ROOT/carla/planner_decision_regret.json" \
 --output "$RIDE_OUTPUT_ROOT/carla/science_gates_with_baselines.json"
