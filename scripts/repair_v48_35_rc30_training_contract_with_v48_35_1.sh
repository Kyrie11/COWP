#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: configure OUTPUTDIR/OCRAP_ROOT/SOURCE_RUN/PROTOCOL_ROOT/CAL_SAFE/GPU0/GPU1, then run:
  bash scripts/repair_v48_35_rc30_training_contract_with_v48_35_1.sh

This command validates the exact known RC=30 signature and reuses existing adaptation checkpoints.
It never retrains and rejects any positional arguments or changed checkpoint/index contracts.
EOF
  exit 0
fi
if (( $# != 0 )); then
  echo "unexpected arguments; use --help" >&2
  exit 2
fi

# Repair only the known v48.35 stale training-contract metadata-key failure.
# This wrapper never launches adaptation training. The controller first validates
# the original RC=30 signature and every checkpoint/config/hash, then reruns the
# protocol/index/model/training contracts and the registered certificate.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_35_continuous_frontier_dedicated_4835}"
export OUTPUTDIR

# Rebuilding either index would invalidate the byte-identical checkpoint reuse
# contract. The resumed controller therefore requires the original audited files.
unset REBUILD_ADAPT_INDEX REBUILD_ADAPT_DEV_INDEX
export RESUME_AFTER_ADAPTATION=1

bash scripts/run_v48_35_continuous_frontier_dedicated.sh
