#!/usr/bin/env bash
set -euo pipefail
# Fast diagnostic wrapper for RC-MPNCF/COWP.  It intentionally evaluates COWP
# first on a small scenario set so failures in COWP do not block cheap feedback.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CKPT:?Set CKPT to the planner checkpoint, e.g. outputs/rc_mpncf_next/checkpoints/planner/cowp_planner_best.pt}"
MODE="${MODE:-smoke}" \
RUN_TESTS="${RUN_TESTS:-0}" \
RUN_TRAIN="${RUN_TRAIN:-0}" \
RUN_OFFLINE_EVAL="${RUN_OFFLINE_EVAL:-1}" \
RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-1}" \
FORCE_EVAL="${FORCE_EVAL:-1}" \
ONLINE_METHODS="${ONLINE_METHODS:-cowp}" \
TOTAL_ONLINE_SCENARIOS="${TOTAL_ONLINE_SCENARIOS:-80}" \
ROLLOUT_HORIZON_STEPS="${ROLLOUT_HORIZON_STEPS:-60}" \
WAYMAX_STANDARD_METRICS="${WAYMAX_STANDARD_METRICS:-1}" \
CLEAR_ACCELERATOR_CACHE="${CLEAR_ACCELERATOR_CACHE:-0}" \
NO_PROGRESS="${NO_PROGRESS:-1}" \
OUTCOME_RISK_PENALTY="${OUTCOME_RISK_PENALTY:-0.75}" \
OUTCOME_RISK_THRESHOLD="${OUTCOME_RISK_THRESHOLD:-1.10}" \
OUT_ROOT="${OUT_ROOT:-outputs/rc_mpncf_fast_probe}" \
bash "$SCRIPT_DIR/run_mpncf_next.sh"
