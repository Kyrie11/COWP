#!/usr/bin/env bash
set -Eeuo pipefail
# v48.44 ROCT main == D arm: bounded Recovery-Option Compatibility Transport
# to the unified benefit coordinate and the deployability component only.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_44_roct_main}"
exec bash scripts/run_v48_44_roct_ablation_arm.sh D
