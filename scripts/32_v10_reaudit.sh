#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
INPUT="${1:?Usage: bash scripts/32_v10_reaudit.sh INPUT.jsonl [OUTPUT.json]}"
OUTPUT="${2:-$(dirname "$INPUT")/v10_mechanism_report.json}"
python -m ride.v10_mechanism_report --input "$INPUT" --output "$OUTPUT"
python -m ride.carla.m0_gate --input "$INPUT" --output "$(dirname "$OUTPUT")/m0_physics_gate_v10.json" || true
echo "V10 re-audit saved: $OUTPUT"
