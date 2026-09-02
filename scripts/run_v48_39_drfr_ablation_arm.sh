#!/usr/bin/env bash
set -Eeuo pipefail
# v48.39 DRFR dynamic-range mechanism ablation.
# All arms use the same aligned deterministic reserve, data, top-k and one shared
# rule. Only the global benefit/harm factor parameterization changes.
# A: bounded benefit + bounded harm (aligned-reserve control)
# B: bounded benefit + unbounded harm
# C: unbounded benefit + bounded harm
# D: unbounded benefit + unbounded harm (main DRFR)
ARM="${1:?usage: $0 A|B|C|D}"
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

if [[ "${V4839_ALLOW_EXACT_FACTOR_CACHE:-0}" != "1" ]]; then
  unset V4836_FACTOR_CACHE_BALANCED V4836_FACTOR_CACHE_PRECISION || true
fi
export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_39_drfr_ablation_${ARM}}"
export RESUME_AFTER_ADAPTATION="0"
export PROPOSAL_TOP_K="5"
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="physical_interaction"
export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT="1.00"
export FACTOR_BENEFIT_MARGIN_TEMPERATURE="0.050"
export FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT="1.00"
export FACTOR_COMPONENT_TAIL_WEIGHT="0.75"
export EVIDENCE_BENEFIT_RESIDUAL_SCALE="6.0"
export EVIDENCE_COMPONENT_SCALE="6.0"
export EVIDENCE_RESERVE_FACTOR_ALIGNMENT="true"
export EVIDENCE_JOINT_RESERVE_TEMPERATURE="0.050"
export FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT="0"
export FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT="0"
export FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT="0"
export FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT="0"
export V4838_RFR_RESERVE_ONLY="1"
export V4837_FACTOR_PRESERVING_IDENTITY="0"
export V4836_IDENTITY_TRAIN_ALL="0"
export V4836_COUPLE_ADMISSION_PRIOR="0"
export V4836_ADAPTIVE_IDENTITY_MARGIN="0"
export V4836_ENABLE_FINAL_CALIBRATION="0"
export EVIDENCE_ADMISSION_PRIOR_MODE="joint_reserve"

# Reset the two tested switches before selecting the arm; ambient shells cannot
# silently change mechanism identity.
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR="false"
export EVIDENCE_UNBOUNDED_HARM_FACTORS="false"
case "$ARM" in
  A)
    export OCRAP_ALGORITHM_VERSION="v48.39-DRFR-ablation-A-bounded-both"
    export OCRAP_IMPLEMENTATION_VERSION="v48.39-DRFR-ABLATION-A-BOUNDED-BOTH"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.39-factor-bounded-both"
    ;;
  B)
    export OCRAP_ALGORITHM_VERSION="v48.39-DRFR-ablation-B-unbounded-harm"
    export OCRAP_IMPLEMENTATION_VERSION="v48.39-DRFR-ABLATION-B-UNBOUNDED-HARM"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.39-factor-unbounded-harm"
    export EVIDENCE_UNBOUNDED_HARM_FACTORS="true"
    ;;
  C)
    export OCRAP_ALGORITHM_VERSION="v48.39-DRFR-ablation-C-unbounded-benefit"
    export OCRAP_IMPLEMENTATION_VERSION="v48.39-DRFR-ABLATION-C-UNBOUNDED-BENEFIT"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.39-factor-unbounded-benefit"
    export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR="true"
    ;;
  D)
    export OCRAP_ALGORITHM_VERSION="v48.39-DRFR-ablation-D-full"
    export OCRAP_IMPLEMENTATION_VERSION="v48.39-DRFR-ABLATION-D-FULL"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.39-DRFR-full-dynamic-range"
    export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR="true"
    export EVIDENCE_UNBOUNDED_HARM_FACTORS="true"
    ;;
  *) echo "unknown DRFR ablation arm: $ARM" >&2; exit 2 ;;
esac

exec bash scripts/run_v48_36_ocaf_dedicated.sh
