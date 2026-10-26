#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
PRED="${1:?Usage: bash scripts/06a_lock_proxy_selectors.sh predictions.csv hard_courtesy}"
POLICY="${2:-hard_courtesy}"
OUT="$RIDE_OUTPUT_ROOT/frozen/${POLICY}"
mkdir -p "$RIDE_OUTPUT_ROOT/frozen"
python -m ride.selectors --predictions "$PRED" --policy "$POLICY" \
 --choices "${OUT}_choices.csv" --manifest "${OUT}_choices.sha.json"
# DO NOT run response rollouts until all planner choices and hashes are frozen.
