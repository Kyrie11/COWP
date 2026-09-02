#!/usr/bin/env bash
set -Eeuo pipefail
# v48.39 DRFR — Dynamic-Range Frontier Reserve.
# One observation-conditioned continuous policy is shared across Safe/Near/Contact.
# Regime labels remain audit-only and never enter the model, loss routing, or rule.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

# Reuse is allowed only when the factor-cache fingerprint proves every physical
# factor semantic bit. Ignore ambient paths by default to prevent mislabeled runs.
if [[ "${V4839_ALLOW_EXACT_FACTOR_CACHE:-0}" != "1" ]]; then
  unset V4836_FACTOR_CACHE_BALANCED V4836_FACTOR_CACHE_PRECISION || true
fi

export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_39_drfr_main}"
export RESUME_AFTER_ADAPTATION="0"
export OCRAP_ALGORITHM_VERSION="v48.39-DRFR"
export OCRAP_IMPLEMENTATION_VERSION="v48.39-DRFR-DYNAMIC-RANGE-FRONTIER"
export V4838_FACTOR_ALGORITHM_FAMILY="v48.39-DRFR-full-dynamic-range"

# HAF result retained: a signed benefit boundary is useful and the factor
# coordinate system must not be rotated by a sparse identity/admission stage.
export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT="1.00"
export FACTOR_BENEFIT_MARGIN_TEMPERATURE="0.050"
export FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT="1.00"
export FACTOR_COMPONENT_TAIL_WEIGHT="0.75"

# v48.39 DRFR: remove the representational ceilings seen in v48.38. The final
# layers are zero-initialized, so unbounded signed residuals preserve source
# identity at step zero while allowing Smooth-L1 physical-margin supervision to
# span the actual teacher range. These are global factor semantics, not regime
# branches or per-regime weights.
export EVIDENCE_BENEFIT_RESIDUAL_SCALE="6.0"
export EVIDENCE_COMPONENT_SCALE="6.0"
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR="true"
export EVIDENCE_UNBOUNDED_HARM_FACTORS="true"

# Align deployment reserve exactly with the factor coordinates supervised in
# training. v48.38 subtracted the pre-pin nominal a second time, which cancelled
# the semantic component prior and shifted the safety frontier.
export EVIDENCE_RESERVE_FACTOR_ALIGNMENT="true"

# v48.38 ablation C did not support the one-sided tail losses; B/D were also
# confounded by the reserve semantic bug. Do not keep stacking those losses.
# Let dense signed benefit/component regression learn the physical factors and
# compose them deterministically at deployment.
export FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT="0"
export FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT="0"
export FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT="0"
export FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT="0"
export FACTOR_JOINT_RESERVE_BOUNDARY_WIDTH="0.050"
export EVIDENCE_JOINT_RESERVE_TEMPERATURE="0.050"

# No learned admission residual. identity/final are honest zero-update,
# byte-identical materializations of the trained factor checkpoint.
export V4838_RFR_RESERVE_ONLY="1"
export V4837_FACTOR_PRESERVING_IDENTITY="0"
export V4836_IDENTITY_TRAIN_ALL="0"
export V4836_COUPLE_ADMISSION_PRIOR="0"
export V4836_ADAPTIVE_IDENTITY_MARGIN="0"
export V4836_ENABLE_FINAL_CALIBRATION="0"
export EVIDENCE_ADMISSION_PRIOR_MODE="joint_reserve"

# Frozen protocol.
export PROPOSAL_TOP_K="5"
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="physical_interaction"

exec bash scripts/run_v48_36_ocaf_dedicated.sh "$@"
