#!/usr/bin/env bash
set -Eeuo pipefail
# v48.38 RFR mechanism ablation.  All arms use one shared continuous policy,
# identical data/proposal/gate protocol, and no regime-conditioned routing.
# Usage: bash scripts/run_v48_38_rfr_ablation_arm.sh A|B|C|D
# A: v48.37 HAF full reference (learned factor-preserving admission residual)
# B: deterministic joint reserve + reserve regression, no new tail losses
# C: bidirectional frontier-tail losses + v48.37 factor-preserving residual
# D: full RFR = B + tail losses; identical to the v48.38 main method
ARM="${1:?usage: $0 A|B|C|D}"
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

if [[ "${V4838_ALLOW_EXACT_FACTOR_CACHE:-0}" != "1" ]]; then
  unset V4836_FACTOR_CACHE_BALANCED V4836_FACTOR_CACHE_PRECISION || true
fi
export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_38_rfr_ablation_${ARM}}"
export RESUME_AFTER_ADAPTATION="0"
export PROPOSAL_TOP_K="5"
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="physical_interaction"
export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT="1.00"
export FACTOR_BENEFIT_MARGIN_TEMPERATURE="0.050"
export V4836_ADAPTIVE_IDENTITY_MARGIN="0"
export V4836_ENABLE_FINAL_CALIBRATION="0"
export EVIDENCE_JOINT_RESERVE_TEMPERATURE="0.050"

# Reset every mechanism knob explicitly so an ambient shell cannot cross-contaminate arms.
export FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT="0"
export FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT="0"
export FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT="0"
export FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT="0"
export FACTOR_JOINT_RESERVE_BOUNDARY_WIDTH="0.050"
export V4838_RFR_RESERVE_ONLY="0"
export V4837_FACTOR_PRESERVING_IDENTITY="1"
export V4836_IDENTITY_TRAIN_ALL="0"
export V4836_COUPLE_ADMISSION_PRIOR="0"
export EVIDENCE_ADMISSION_PRIOR_MODE="frontier_capped_slack"

case "$ARM" in
  A)
    export OCRAP_ALGORITHM_VERSION="v48.38-RFR-ablation-A-HAF-reference"
    export OCRAP_IMPLEMENTATION_VERSION="v48.38-RFR-ABLATION-A-HAF-REFERENCE"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.38-factor-HAF-reference"
    ;;
  B)
    export OCRAP_ALGORITHM_VERSION="v48.38-RFR-ablation-B-joint-reserve"
    export OCRAP_IMPLEMENTATION_VERSION="v48.38-RFR-ABLATION-B-JOINT-RESERVE"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.38-factor-joint-reserve"
    export FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT="1.00"
    export FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT="2.00"
    export V4838_RFR_RESERVE_ONLY="1"
    export V4837_FACTOR_PRESERVING_IDENTITY="0"
    export EVIDENCE_ADMISSION_PRIOR_MODE="joint_reserve"
    ;;
  C)
    export OCRAP_ALGORITHM_VERSION="v48.38-RFR-ablation-C-tail-calibration"
    export OCRAP_IMPLEMENTATION_VERSION="v48.38-RFR-ABLATION-C-TAIL-CALIBRATION"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.38-factor-tail-calibration"
    export FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT="0.75"
    export FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT="1.00"
    ;;
  D)
    export OCRAP_ALGORITHM_VERSION="v48.38-RFR-ablation-D-full"
    export OCRAP_IMPLEMENTATION_VERSION="v48.38-RFR-ABLATION-D-FULL"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.38-RFR-full-factor"
    export FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT="0.75"
    export FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT="1.00"
    export FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT="1.00"
    export FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT="2.00"
    export V4838_RFR_RESERVE_ONLY="1"
    export V4837_FACTOR_PRESERVING_IDENTITY="0"
    export EVIDENCE_ADMISSION_PRIOR_MODE="joint_reserve"
    ;;
  *) echo "unknown RFR ablation arm: $ARM" >&2; exit 2 ;;
esac

exec bash scripts/run_v48_36_ocaf_dedicated.sh
