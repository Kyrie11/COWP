#!/usr/bin/env bash
# Dedicated evidence-first research run. DOES NOT modify training checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
export BASE_RUN="${BASE_RUN:-/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all}"
export BASE_CKPT="${BASE_CKPT:-$BASE_RUN/cowp_all_best.pt}"
MODE="${1:-help}"
case "$MODE" in
 sanity) bash run.sh sanity ;;
 historical_audit) bash run.sh audit_previous ;;
 geometry_primary)
    export PROBE_TAG=primary RESPONSE_ACCELS=-4,-2,0,1.5
    bash run.sh probe_geometry_circle
    bash run.sh probe_geometry_oriented
    bash run.sh analyze_geometry
    ;;
 geometry_alternate)
    export PROBE_TAG=alternate RESPONSE_ACCELS=-5,-3,0,2
    bash run.sh probe_geometry_circle
    bash run.sh probe_geometry_oriented
    bash run.sh analyze_geometry
    ;;
 full_geometry)
    bash "$0" geometry_primary
    bash "$0" geometry_alternate
    python -m research_probe.analyze \
      --inputs outputs/research_probe/primary_oriented.json outputs/research_probe/alternate_oriented.json \
      --output outputs/research_probe/oriented_response_sensitivity.json
    ;;
 *)
 echo 'Usage: bash NEXT_EXPERIMENT_COMMANDS_CN.sh {sanity|historical_audit|geometry_primary|geometry_alternate|full_geometry}'
 exit 2 ;;
esac
