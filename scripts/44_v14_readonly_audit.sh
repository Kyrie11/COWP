#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_V14_AUDIT_ROOTS:?Space-separated extracted v13 experiment directories}"
: "${RIDE_V14_AUDIT_OUTPUT:?New, unique audit output path}"
read -r -a INPUTS <<< "$RIDE_V14_AUDIT_ROOTS"
python -m ride.v14_counterfactual_audit --roots "${INPUTS[@]}" --output "$RIDE_V14_AUDIT_OUTPUT"
