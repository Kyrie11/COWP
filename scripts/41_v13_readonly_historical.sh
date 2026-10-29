#!/usr/bin/env bash
# Read-only on a previously completed BOTH-family CARLA chain panel.
# May NOT establish ex-ante decision errors, only screen potential contexts.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
PANEL="${1:?usage: bash scripts/41_v13_readonly_historical.sh path/to/chain_cut_in.jsonl output.json}"
OUTPUT="${2:?set a new output.json}"
python -m ride.v13_model_transfer retrospective \
 --panel "$PANEL" --source-family behavior --target-family tm \
 --tie-resolution 0.05 --meaningful-delta 0.25 --comfort-limit 2.0 \
 --output "$OUTPUT"
