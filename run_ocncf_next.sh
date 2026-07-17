#!/usr/bin/env bash
set -euo pipefail
# Outcome-Calibrated Non-Coercive Frontier (OC-NCF) experiment wrapper.
# Defaults are chosen for the two-GPU workstation described in the prompt.
export OUT_ROOT="${OUT_ROOT:-outputs/ocncf_next}"
export NUM_GPUS="${NUM_GPUS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MODE="${MODE:-full}"
export RUN_TESTS="${RUN_TESTS:-0}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export PLANNER_ONLY_RETRAIN="${PLANNER_ONLY_RETRAIN:-1}"
export FORCE_TRAIN="${FORCE_TRAIN:-1}"
export RUN_OFFLINE_EVAL="${RUN_OFFLINE_EVAL:-1}"
export RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-1}"
export FORCE_EVAL="${FORCE_EVAL:-1}"
export CKPT_SELECTION="${CKPT_SELECTION:-latest}"
export EPOCH_PLANNER="${EPOCH_PLANNER:-20}"
export PER_GPU_BATCH_PLANNER="${PER_GPU_BATCH_PLANNER:-16}"
export OUTCOME_RISK_PENALTY="${OUTCOME_RISK_PENALTY:-0.85}"
export OUTCOME_RISK_THRESHOLD="${OUTCOME_RISK_THRESHOLD:-1.15}"
export TOTAL_ONLINE_SCENARIOS="${TOTAL_ONLINE_SCENARIOS:-300}"
export ROLLOUT_HORIZON_STEPS="${ROLLOUT_HORIZON_STEPS:-80}"
export WAYMAX_STANDARD_METRICS="${WAYMAX_STANDARD_METRICS:-1}"
export CLEAR_ACCELERATOR_CACHE="${CLEAR_ACCELERATOR_CACHE:-0}"
export NO_PROGRESS="${NO_PROGRESS:-1}"
export ONLINE_METHODS="${ONLINE_METHODS:-planner_score_only conventional_safety cowp}"
# Reuse the previous witness unless WITNESS_CKPT is explicitly set.
export WITNESS_CKPT="${WITNESS_CKPT:-outputs/rc_mpncf_next/checkpoints/witness/cowp_witness_best.pt}"
exec bash run_mpncf_next.sh "$@"
