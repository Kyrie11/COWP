#!/usr/bin/env bash
set -Eeuo pipefail
# v48.40 DCFR — Decoupled Context Frontier Reserve.
# One continuous physical selector is shared across Safe/Near/Contact. Benefit
# and harm use the same regime-free observations/actions but separate trainable
# OCAF interaction parameters to prevent cross-task gradient interference.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
if [[ "${V4840_ALLOW_EXACT_FACTOR_CACHE:-0}" != 1 ]]; then
  unset V4836_FACTOR_CACHE_BALANCED V4836_FACTOR_CACHE_PRECISION || true
fi
export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_40_dcfr_main}"
export RESUME_AFTER_ADAPTATION=0
export OCRAP_ALGORITHM_VERSION="v48.40-DCFR"
export OCRAP_IMPLEMENTATION_VERSION="v48.40-DCFR-DECOUPLED-CONTEXT-FRONTIER"
export V4838_FACTOR_ALGORITHM_FAMILY="v48.40-DCFR-dual-ocaf-frontier-normalized"

# Retain only mechanisms supported by prior ablations.
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false
export EVIDENCE_UNBOUNDED_HARM_FACTORS=false
export EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0
export EVIDENCE_COMPONENT_SCALE=6.0
export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT=1.00
export FACTOR_BENEFIT_MARGIN_TEMPERATURE=0.050
export FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT=1.00
export FACTOR_COMPONENT_TAIL_WEIGHT=0.75

# New mechanism 1: task-decoupled observation-conditioned physical context.
export EVIDENCE_DUAL_INTERACTION_BRIDGE=true
# New mechanism 2: preserve the zero safety frontier while compressing large
# teacher violations so scarce near-boundary safe-positive examples retain
# regression resolution. Sign BCE and deployment margins remain unchanged.
export FACTOR_COMPONENT_MARGIN_TARGET_MODE=frontier_tanh
export FACTOR_COMPONENT_MARGIN_TARGET_SCALE=0.10

# Retain aligned deterministic noncompensatory reserve; no learned admission.
export EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true
export EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve
export EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050
export V4838_RFR_RESERVE_ONLY=1
export V4837_FACTOR_PRESERVING_IDENTITY=0
export V4836_IDENTITY_TRAIN_ALL=0
export V4836_COUPLE_ADMISSION_PRIOR=0
export V4836_ADAPTIVE_IDENTITY_MARGIN=0
export V4836_ENABLE_FINAL_CALIBRATION=0

# Explicitly disable previously unsupported v48.38/v48.39 experiments.
export FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT=0
export FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT=0
export FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT=0
export FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT=0
export PROPOSAL_TOP_K=5
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction
exec bash scripts/run_v48_36_ocaf_dedicated.sh "$@"
