#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
: "${OUTPUTDIR:=runs/ocrap_v48_34_1_barrier_crossfit_dedicated_48341}"
export OUTPUTDIR
exec bash scripts/run_v48_34_barrier_crossfit_dedicated.sh "$@"
