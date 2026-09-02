#!/usr/bin/env bash
set -Eeuo pipefail
# v48.38 RFR — Robust Frontier Reserve.
# One observation-conditioned continuous policy is used for Safe/Near/Contact.
# Regime labels remain audit-only and never enter the model or thresholds.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

# A cache is reusable only after the exact factor semantic fingerprint passes.
# Ignore arbitrary ambient cache paths by default to prevent mislabeled runs.
if [[ "${V4838_ALLOW_EXACT_FACTOR_CACHE:-0}" != "1" ]]; then
  unset V4836_FACTOR_CACHE_BALANCED V4836_FACTOR_CACHE_PRECISION || true
fi

export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_38_rfr_main}"
export RESUME_AFTER_ADAPTATION="0"
export OCRAP_ALGORITHM_VERSION="v48.38-RFR"
export OCRAP_IMPLEMENTATION_VERSION="v48.38-RFR-ROBUST-JOINT-RESERVE"
export V4838_FACTOR_ALGORITHM_FAMILY="v48.38-RFR-full-factor"

# Preserve the HAF benefit boundary that showed a positive Near-contact signal.
export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT="1.00"
export FACTOR_BENEFIT_MARGIN_TEMPERATURE="0.050"

# RFR-1: directly correct the two certificate-tail errors observed in v48.37:
# (a) dangerous component-veto violations underestimated as safe;
# (b) true safe-positive actions overestimated as harmful.
# These are one-sided continuous-margin losses, not class/regime oversampling.
export FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT="0.75"
export FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT="1.00"

# RFR-2: regress the shared physical recoverability reserve
#   min(benefit headroom, worst-component safety headroom)
# and spend extra capacity near its unique zero boundary.
export FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT="1.00"
export FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT="2.00"
export FACTOR_JOINT_RESERVE_BOUNDARY_WIDTH="0.050"
export EVIDENCE_JOINT_RESERVE_TEMPERATURE="0.050"

# v48.37 C/D repeatedly selected identity epoch zero.  Do not relearn a sparse
# admission residual. The factor checkpoint itself becomes the deployable model;
# identity/final directories are byte-identical provenance materializations.
export V4838_RFR_RESERVE_ONLY="1"
export V4837_FACTOR_PRESERVING_IDENTITY="0"
export V4836_IDENTITY_TRAIN_ALL="0"
export V4836_COUPLE_ADMISSION_PRIOR="0"
export V4836_ADAPTIVE_IDENTITY_MARGIN="0"
export V4836_ENABLE_FINAL_CALIBRATION="0"
export EVIDENCE_ADMISSION_PRIOR_MODE="joint_reserve"

# Frozen protocol.  Do not turn these into arm- or regime-specific knobs.
export PROPOSAL_TOP_K="5"
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="physical_interaction"

exec bash scripts/run_v48_36_ocaf_dedicated.sh "$@"
