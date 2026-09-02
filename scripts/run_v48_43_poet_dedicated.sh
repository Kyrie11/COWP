#!/usr/bin/env bash
set -Eeuo pipefail
# v48.43 POET main == D arm: post-prefix observation-equivalence transport to
# both sparse benefit and dense harm evidence.  It intentionally removes the
# v48.42 partial-pool and rank-skip mechanisms after their valid negative ablations.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_43_poet_main}"
exec bash scripts/run_v48_43_poet_ablation_arm.sh D
