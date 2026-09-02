#!/usr/bin/env bash
set -Eeuo pipefail
# v48.40 mechanism ablation; all arms use bounded factors, aligned reserve,
# top-k=5, identical data and one shared deployment rule.
# A shared OCAF + raw margin regression (aligned bounded reference)
# B dual OCAF + raw margin regression
# C shared OCAF + frontier-normalized component regression
# D dual OCAF + frontier-normalized regression (main DCFR)
ARM="${1:?usage: $0 A|B|C|D}"
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
if [[ "${V4840_ALLOW_EXACT_FACTOR_CACHE:-0}" != 1 ]]; then
  unset V4836_FACTOR_CACHE_BALANCED V4836_FACTOR_CACHE_PRECISION || true
fi
export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_40_dcfr_ablation_${ARM}}"
export RESUME_AFTER_ADAPTATION=0
export OCRAP_IMPLEMENTATION_VERSION="v48.40-DCFR-DECOUPLED-CONTEXT-FRONTIER"
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false
export EVIDENCE_UNBOUNDED_HARM_FACTORS=false
export EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0
export EVIDENCE_COMPONENT_SCALE=6.0
export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT=1.00
export FACTOR_BENEFIT_MARGIN_TEMPERATURE=0.050
export FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT=1.00
export FACTOR_COMPONENT_TAIL_WEIGHT=0.75
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
export EVIDENCE_DUAL_INTERACTION_BRIDGE=false
export FACTOR_COMPONENT_MARGIN_TARGET_MODE=raw
export FACTOR_COMPONENT_MARGIN_TARGET_SCALE=0.10
case "$ARM" in
 A) export OCRAP_ALGORITHM_VERSION="v48.40-DCFR-ablation-A"; export V4838_FACTOR_ALGORITHM_FAMILY="v48.40-A-shared-raw" ;;
 B) export OCRAP_ALGORITHM_VERSION="v48.40-DCFR-ablation-B"; export V4838_FACTOR_ALGORITHM_FAMILY="v48.40-B-dual-raw"; export EVIDENCE_DUAL_INTERACTION_BRIDGE=true ;;
 C) export OCRAP_ALGORITHM_VERSION="v48.40-DCFR-ablation-C"; export V4838_FACTOR_ALGORITHM_FAMILY="v48.40-C-shared-frontier-normalized"; export FACTOR_COMPONENT_MARGIN_TARGET_MODE=frontier_tanh ;;
 D) export OCRAP_ALGORITHM_VERSION="v48.40-DCFR"; export V4838_FACTOR_ALGORITHM_FAMILY="v48.40-D-dual-frontier-normalized"; export EVIDENCE_DUAL_INTERACTION_BRIDGE=true; export FACTOR_COMPONENT_MARGIN_TARGET_MODE=frontier_tanh ;;
 *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
exec bash scripts/run_v48_36_ocaf_dedicated.sh "$@"
