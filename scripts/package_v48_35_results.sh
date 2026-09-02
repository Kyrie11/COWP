#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/tools:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
RUN="${RUN:?RUN is required}"
OUTPUT_ZIP="${OUTPUT_ZIP:-${RUN%/}-clean-result.zip}"
args=(--run "$RUN" --output "$OUTPUT_ZIP")
[[ "${INCLUDE_CHECKPOINTS:-0}" == 1 ]] && args+=(--include-checkpoints)
[[ "${INCLUDE_STATUS_HISTORY:-0}" == 1 ]] && args+=(--include-status-history)
python tools/package_v48_35_results.py "${args[@]}"
