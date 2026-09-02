#!/usr/bin/env bash
set -Eeuo pipefail
# v48.37 HAF (Headroom-Aligned Frontier) algorithm wrapper.
# The audited v48.36 controller/gate protocol is intentionally reused unchanged;
# this wrapper changes only the training algorithm and records a distinct
# implementation/algorithm version.  Safe/Near/Contact remain audit strata only.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

# A v48.36 factor cache cannot satisfy the HAF factor contract.  Ignore ambient
# cache variables by default so a stale user shell cannot turn an algorithm run
# into an avoidable engineering RC=30.  Exact HAF caches may be opted in.
if [[ "${V4837_ALLOW_EXACT_FACTOR_CACHE:-0}" != "1" ]]; then
  unset V4836_FACTOR_CACHE_BALANCED V4836_FACTOR_CACHE_PRECISION || true
fi
export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_37_haf_main}"
export RESUME_AFTER_ADAPTATION="${RESUME_AFTER_ADAPTATION:-0}"

export OCRAP_ALGORITHM_VERSION="v48.37-HAF"
export OCRAP_IMPLEMENTATION_VERSION="v48.37-HAF-DUAL-HEADROOM-FACTOR-PRESERVE"

# HAF-1: make opportunity P=0.5 a physical boundary, symmetric with the existing
# signed component-veto margin heads.  No regime label enters this target.
export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT="1.00"
export FACTOR_BENEFIT_MARGIN_TEMPERATURE="0.050"

# HAF-2: preserve the factor-stage coordinate system while learning admission.
# The identity stage trains the shared admission residual only; benefit, harm and
# the OCAF action-observation bridge remain frozen and are byte-audited.
export V4837_FACTOR_PRESERVING_IDENTITY="1"
export V4836_IDENTITY_TRAIN_ALL="0"
export V4836_COUPLE_ADMISSION_PRIOR="0"
export V4836_ADAPTIVE_IDENTITY_MARGIN="0"
export V4836_ENABLE_FINAL_CALIBRATION="0"

# Keep the preregistered proposal/gate semantics unchanged for attribution.
export PROPOSAL_TOP_K="5"
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="physical_interaction"
export EVIDENCE_ADMISSION_PRIOR_MODE="frontier_capped_slack"

exec bash scripts/run_v48_36_ocaf_dedicated.sh "$@"
