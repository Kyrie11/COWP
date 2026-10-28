#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_V11_AUDIT_OUTPUT:?Set JSON report path}"
(( $# >= 1 )) || { echo 'Usage: RIDE_V11_AUDIT_OUTPUT=... bash scripts/34_v11_audit_uploaded.sh path1.jsonl ...' >&2; exit 2; }
python -m ride.v11_chain_evidence --input "$@" --output "$RIDE_V11_AUDIT_OUTPUT"
