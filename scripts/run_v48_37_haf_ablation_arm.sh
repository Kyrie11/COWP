#!/usr/bin/env bash
set -Eeuo pipefail
# Pre-registered v48.37 HAF 2x2 mechanism ablation.
# Usage: bash scripts/run_v48_37_haf_ablation_arm.sh A|B|C|D
# A: v48.36 training reference under the repaired harness
# B: + signed benefit-headroom anchoring only
# C: + factor-preserving admission only
# D: full HAF = B + C (same as run_v48_37_haf_dedicated.sh)
ARM="${1:?usage: $0 A|B|C|D}"
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
# Each arm is an independent preregistered run.  Avoid accidental cross-arm
# factor-cache reuse unless an exact matching cache is explicitly requested.
if [[ "${V4837_ALLOW_EXACT_FACTOR_CACHE:-0}" != "1" ]]; then
  unset V4836_FACTOR_CACHE_BALANCED V4836_FACTOR_CACHE_PRECISION || true
fi
export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_37_haf_ablation_${ARM}}"
export RESUME_AFTER_ADAPTATION="${RESUME_AFTER_ADAPTATION:-0}"
case "$ARM" in
  A)
    export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT=0
    export V4837_FACTOR_PRESERVING_IDENTITY=0
    export V4836_IDENTITY_TRAIN_ALL=1
    export V4836_COUPLE_ADMISSION_PRIOR=1
    ;;
  B)
    export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT="1.00"
    export V4837_FACTOR_PRESERVING_IDENTITY=0
    export V4836_IDENTITY_TRAIN_ALL=1
    export V4836_COUPLE_ADMISSION_PRIOR=1
    ;;
  C)
    export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT=0
    export V4837_FACTOR_PRESERVING_IDENTITY=1
    export V4836_IDENTITY_TRAIN_ALL=0
    export V4836_COUPLE_ADMISSION_PRIOR=0
    ;;
  D)
    export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT="1.00"
    export V4837_FACTOR_PRESERVING_IDENTITY=1
    export V4836_IDENTITY_TRAIN_ALL=0
    export V4836_COUPLE_ADMISSION_PRIOR=0
    ;;
  *) echo "unknown HAF ablation arm: $ARM" >&2; exit 2 ;;
esac
export FACTOR_BENEFIT_MARGIN_TEMPERATURE="0.050"
export OCRAP_ALGORITHM_VERSION="v48.37-HAF-ablation-$ARM"
export OCRAP_IMPLEMENTATION_VERSION="v48.37-HAF-ABLATION-$ARM"
export V4836_ADAPTIVE_IDENTITY_MARGIN=0
export V4836_ENABLE_FINAL_CALIBRATION=0
export PROPOSAL_TOP_K="5"
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="physical_interaction"
export EVIDENCE_ADMISSION_PRIOR_MODE="frontier_capped_slack"
# The factor-cache fingerprint includes both HAF mechanism switches. Reusing an
# old cache is therefore fail-closed; leave cache variables unset unless the
# caller intentionally provides an exact v48.37 matching cache.
exec bash scripts/run_v48_36_ocaf_dedicated.sh
