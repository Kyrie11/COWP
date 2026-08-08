#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/NEXT_RUN_COMMANDS_V16_8_9_CAUSAL_AUDIT_SMOKE_CN.sh" "$@"
