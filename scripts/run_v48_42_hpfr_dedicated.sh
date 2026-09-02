#!/usr/bin/env bash
set -Eeuo pipefail
# v48.42 HPFR — Hierarchical Partial-Pooling Frontier Reserve.
# Main D combines two independently testable mechanisms: (1) the empirically
# stronger shared dual-task OCAF harm representation plus small detached,
# zero-init component residuals, and (2) the v48.41 rank-benefit skip whose
# previous RC=30 was caused by an exact-parameter checker bug rather than an
# algorithmic gate result. The same continuous physics is used in all regimes.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
if [[ "${V4842_ALLOW_EXACT_FACTOR_CACHE:-0}" != 1 ]]; then
  unset V4836_FACTOR_CACHE_BALANCED V4836_FACTOR_CACHE_PRECISION || true
fi
export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_42_hpfr_main}"
export RESUME_AFTER_ADAPTATION=0
export OCRAP_ALGORITHM_VERSION="v48.42-HPFR"
export OCRAP_IMPLEMENTATION_VERSION="v48.42.1-HPFR-METRIC-PROVENANCE-HOTFIX"
export V4838_FACTOR_ALGORITHM_FAMILY="v48.42-D-partial-pool-harm-rank-skip"

# Keep mechanisms with positive evidence; explicitly disable falsified routes.
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false
export EVIDENCE_UNBOUNDED_HARM_FACTORS=false
export EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0
export EVIDENCE_COMPONENT_SCALE=6.0
export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT=1.00
export FACTOR_BENEFIT_MARGIN_TEMPERATURE=0.050
export FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT=1.00
export FACTOR_COMPONENT_TAIL_WEIGHT=0.75
export FACTOR_COMPONENT_MARGIN_TARGET_MODE=raw
export FACTOR_COMPONENT_MARGIN_TARGET_SCALE=0.10
export EVIDENCE_DUAL_INTERACTION_BRIDGE=true
# v48.41-B full harm factorisation degraded Near/Contact rare-frontier AUC;
# return to the shared harm bridge/trunk and add only bounded component residuals.
export EVIDENCE_FACTORIZED_HARM_INTERACTION=false
export EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=true
export EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL_SCALE=0.50
# Re-run the rank skip after fixing its exact-parameter engineering contract.
export EVIDENCE_RANK_BENEFIT_SKIP=true
export EVIDENCE_RANK_BENEFIT_GAIN_INIT=1.0

export EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true
export EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve
export EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050
export V4838_RFR_RESERVE_ONLY=1
export V4837_FACTOR_PRESERVING_IDENTITY=0
export V4836_IDENTITY_TRAIN_ALL=0
export V4836_COUPLE_ADMISSION_PRIOR=0
export V4836_ADAPTIVE_IDENTITY_MARGIN=0
export V4836_ENABLE_FINAL_CALIBRATION=0
export FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT=0
export FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT=0
export FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT=0
export FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT=0
export PROPOSAL_TOP_K=5
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction
exec bash scripts/run_v48_36_ocaf_dedicated.sh "$@"
