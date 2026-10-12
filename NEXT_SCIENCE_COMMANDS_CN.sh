#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
WOMD_RAW="${WOMD_RAW:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
COWP_DATA="${COWP_DATA:-/data0/senzeyu2/dataset/COWP}"
AUDIT_DIR="${AUDIT_DIR:-$COWP_DATA/research_native_womd}"
mkdir -p "$AUDIT_DIR" outputs/research_probe
case "${1:-help}" in
 sanity)
   python -m compileall -q research_probe
   python -m pytest -q tests/test_native_womd.py tests/test_geometry.py tests/test_research_probe.py ;;
 audit_upload)
   python -m research_probe.audit_geometry_results --folder "${PROBE_INPUT_DIR:-outputs/research_probe}" --output outputs/research_probe/geometry_independent_decision.json ;;
 discover_raw)
   python - "$WOMD_RAW" <<'PY_RAW'
from pathlib import Path
import sys
p=Path(sys.argv[1]);print('exists=',p.exists());print('sample_shards=')
for q in list(p.rglob('*tfrecord*'))[:15]:print(q)
PY_RAW
   ;;
 sample_raw)
   python -m research_probe.native_womd --raw-root "$WOMD_RAW" --split "${WOMD_SPLIT:-validation}" --format "${WOMD_FORMAT:-auto}" --output "$AUDIT_DIR/cohort" --max-shards "${WOMD_SHARDS:-4}" --max-records-per-shard "${WOMD_PER_SHARD:-350}" --keep-random "${WOMD_RANDOM:-120}" --keep-interactive "${WOMD_INTERACTIVE:-120}" ;;
 simulate_reactive)
   python -m research_probe.reactive_stress --cohort "$AUDIT_DIR/cohort" --output "$AUDIT_DIR/reactive_screen.json" --max-scenes "${REACTIVE_SCENES:-80}" ;;
 export_r3r)
   python -m research_probe.r3r_export --cohort "$AUDIT_DIR/cohort" --output "$AUDIT_DIR/r3r_native_inputs.json" --max-scenes "${R3R_EXPORT_SCENES:-30}" ;;
 *) echo 'Usage: bash NEXT_SCIENCE_COMMANDS_CN.sh {sanity|audit_upload|discover_raw|sample_raw|simulate_reactive|export_r3r}' ;exit 2 ;;
esac
