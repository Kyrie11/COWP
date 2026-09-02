#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_45_sowr_main}"
exec bash scripts/run_v48_45_sowr_ablation_arm.sh D
