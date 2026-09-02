#!/usr/bin/env bash
set -Eeuo pipefail
# v48.41 FCFR — Factorized Component Frontier Reserve.
# One continuous regime-free selector is shared across Safe/Near/Contact.
# This release keeps the v48.40 dual task OCAF result that improved harm
# discrimination, removes the unsupported frontier_tanh target, factorizes the
# harm interaction/calibrator path by physical veto component, and adds a
# monotone low-capacity skip from the frozen preference advantage into the
# bounded HAF benefit residual.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
if [[ "${V4841_ALLOW_EXACT_FACTOR_CACHE:-0}" != 1 ]]; then
  unset V4836_FACTOR_CACHE_BALANCED V4836_FACTOR_CACHE_PRECISION || true
fi
export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_41_fcfr_main}"
export RESUME_AFTER_ADAPTATION=0
export OCRAP_ALGORITHM_VERSION="v48.41-FCFR"
export OCRAP_IMPLEMENTATION_VERSION="v48.41-FCFR-FACTORIZED-COMPONENT-FRONTIER"
export V4838_FACTOR_ALGORITHM_FAMILY="v48.41-D-factorized-harm-rank-skip"

# Retain only mechanisms supported by prior ablations.
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false
export EVIDENCE_UNBOUNDED_HARM_FACTORS=false
export EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0
export EVIDENCE_COMPONENT_SCALE=6.0
export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT=1.00
export FACTOR_BENEFIT_MARGIN_TEMPERATURE=0.050
export FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT=1.00
export FACTOR_COMPONENT_TAIL_WEIGHT=0.75
# v48.40 C and D did not support frontier_tanh; return to the raw physical
# component margin while preserving the exact zero frontier.
export FACTOR_COMPONENT_MARGIN_TARGET_MODE=raw
export FACTOR_COMPONENT_MARGIN_TARGET_SCALE=0.10

# Retain v48.40's meaningful task-level gradient decoupling and extend it inside
# the harm task to one continuous OCAF context/head per physical veto factor.
export EVIDENCE_DUAL_INTERACTION_BRIDGE=true
export EVIDENCE_FACTORIZED_HARM_INTERACTION=true
# Frozen rank_adv has strong safe-positive ordering signal, especially Contact.
# Use it only as a positive-gain skip inside the bounded HAF benefit residual;
# no ranking loss, regime router, or deployment threshold is changed.
export EVIDENCE_RANK_BENEFIT_SKIP=true
export EVIDENCE_RANK_BENEFIT_GAIN_INIT=1.0

# Deterministic aligned noncompensatory reserve; no learned admission stage.
export EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true
export EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve
export EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050
export V4838_RFR_RESERVE_ONLY=1
export V4837_FACTOR_PRESERVING_IDENTITY=0
export V4836_IDENTITY_TRAIN_ALL=0
export V4836_COUPLE_ADMISSION_PRIOR=0
export V4836_ADAPTIVE_IDENTITY_MARGIN=0
export V4836_ENABLE_FINAL_CALIBRATION=0

# Explicitly disable previously falsified/unsupported experiments.
export FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT=0
export FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT=0
export FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT=0
export FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT=0
export PROPOSAL_TOP_K=5
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction
exec bash scripts/run_v48_36_ocaf_dedicated.sh "$@"
