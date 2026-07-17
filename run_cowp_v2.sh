#!/usr/bin/env bash
set -euo pipefail
# COWP v2: outcome-calibrated certificate frontier.
# This wrapper keeps every run in a new output directory by default and preserves
# FORCE_TRAIN=0 restart behavior from run_mpncf_next.sh.
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v2}"
export NUM_GPUS="${NUM_GPUS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MODE="${MODE:-full}"
export RUN_TESTS="${RUN_TESTS:-0}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export PLANNER_ONLY_RETRAIN="${PLANNER_ONLY_RETRAIN:-1}"
export FORCE_TRAIN="${FORCE_TRAIN:-0}"
export RUN_OFFLINE_EVAL="${RUN_OFFLINE_EVAL:-1}"
export RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-1}"
export FORCE_EVAL="${FORCE_EVAL:-0}"
export CKPT_SELECTION="${CKPT_SELECTION:-latest}"
export EPOCH_PLANNER="${EPOCH_PLANNER:-28}"
export PER_GPU_BATCH_PLANNER="${PER_GPU_BATCH_PLANNER:-16}"
export OUTCOME_RISK_PENALTY="${OUTCOME_RISK_PENALTY:-1.00}"
export OUTCOME_RISK_THRESHOLD="${OUTCOME_RISK_THRESHOLD:-1.05}"
export TOTAL_ONLINE_SCENARIOS="${TOTAL_ONLINE_SCENARIOS:-300}"
export ROLLOUT_HORIZON_STEPS="${ROLLOUT_HORIZON_STEPS:-80}"
export WAYMAX_STANDARD_METRICS="${WAYMAX_STANDARD_METRICS:-1}"
export WAYMAX_STANDARD_METRIC_NAMES="${WAYMAX_STANDARD_METRIC_NAMES:-OverlapMetric,OffroadMetric,ProgressionMetric,KinematicsInfeasibilityMetric}"
export CLEAR_ACCELERATOR_CACHE="${CLEAR_ACCELERATOR_CACHE:-0}"
export NO_PROGRESS="${NO_PROGRESS:-1}"
export WAYMAX_STATUS_EVERY="${WAYMAX_STATUS_EVERY:-10}"
export ONLINE_METHODS="${ONLINE_METHODS:-planner_score_only conventional_safety cowp}"
# Reuse the strongest available witness unless explicitly overridden.
export WITNESS_CKPT="${WITNESS_CKPT:-outputs/rc_mpncf_next/checkpoints/witness/cowp_witness_best.pt}"
echo "[cowp_v2] OUT_ROOT=$OUT_ROOT"
echo "[cowp_v2] CKPT_SELECTION=$CKPT_SELECTION FORCE_TRAIN=$FORCE_TRAIN FORCE_EVAL=$FORCE_EVAL"
exec bash run_mpncf_next.sh "$@"
