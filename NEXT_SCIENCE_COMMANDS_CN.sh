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
   python -m pytest -q tests/test_native_womd.py tests/test_native_tfexample_regression.py tests/test_baseline_readiness.py tests/test_gameformer_report.py tests/test_geometry.py tests/test_research_probe.py ;;
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
 smoke_raw)
   extra=(); if [[ -n "${WOMD_EXCLUDE_IDS:-}" ]]; then extra+=(--exclude-ids "$WOMD_EXCLUDE_IDS"); fi
   python -m research_probe.native_womd --raw-root "$WOMD_RAW" --split "${WOMD_SPLIT:-validation}" --format "${WOMD_FORMAT:-tfexample}" --output "$AUDIT_DIR/cohort_smoke" --max-shards 1 --max-records-per-shard 48 --keep-random 12 --keep-interactive 12 "${extra[@]}" ;;
 sample_raw)
   extra=(); if [[ -n "${WOMD_EXCLUDE_IDS:-}" ]]; then extra+=(--exclude-ids "$WOMD_EXCLUDE_IDS"); fi
   python -m research_probe.native_womd --raw-root "$WOMD_RAW" --split "${WOMD_SPLIT:-validation}" --format "${WOMD_FORMAT:-auto}" --output "$AUDIT_DIR/cohort" --max-shards "${WOMD_SHARDS:-4}" --max-records-per-shard "${WOMD_PER_SHARD:-350}" --keep-random "${WOMD_RANDOM:-120}" --keep-interactive "${WOMD_INTERACTIVE:-120}" "${extra[@]}" ;;
 simulate_reactive)
   python -m research_probe.reactive_stress --cohort "$AUDIT_DIR/cohort" --output "$AUDIT_DIR/reactive_screen.json" --max-scenes "${REACTIVE_SCENES:-80}" ;;
 baseline_readiness)
   extra=(); if [[ -n "${WOMD_SCENARIO:-}" ]]; then extra+=(--scenario-root "$WOMD_SCENARIO"); fi
   if [[ -n "${GAMEFORMER_CKPT:-}" ]]; then extra+=(--gameformer-checkpoint "$GAMEFORMER_CKPT"); fi
   python -m research_probe.baseline_readiness --cohort "$AUDIT_DIR/cohort" --gameformer-root "${GAMEFORMER_SRC:-$PWD/third_party/GameFormer-main}" --ra-cbf-root "${RA_CBF_SRC:-$PWD/third_party/learning_responsibility_allocation-main}" --output "$AUDIT_DIR/baseline_readiness.json" "${extra[@]}" ;;
 prepare_gameformer_subset)
   if [[ -z "${WOMD_SCENARIO:-}" ]]; then echo 'Set WOMD_SCENARIO to Scenario proto validation directory (not tf_example)' >&2; exit 2; fi
   python -m research_probe.gameformer_subset --cohort "$AUDIT_DIR/cohort" --scenario-dir "$WOMD_SCENARIO" --output-csv "$AUDIT_DIR/gameformer_cohort.csv" --report "$AUDIT_DIR/gameformer_cohort_match.json" ;;
 gameformer_report)
   if [[ -z "${GAMEFORMER_RESULT_CSV:-}" ]]; then echo 'Set GAMEFORMER_RESULT_CSV to authors testing_log.csv' >&2; exit 2; fi
   python -m research_probe.gameformer_report --cohort "$AUDIT_DIR/cohort" --gameformer-csv "$GAMEFORMER_RESULT_CSV" --output "$AUDIT_DIR/gameformer_open_loop_report.json" ;;
 export_r3r)
   python -m research_probe.r3r_export --cohort "$AUDIT_DIR/cohort" --output "$AUDIT_DIR/r3r_native_inputs.json" --max-scenes "${R3R_EXPORT_SCENES:-30}" ;;
 *) echo 'Usage: bash NEXT_SCIENCE_COMMANDS_CN.sh {sanity|audit_upload|discover_raw|smoke_raw|sample_raw|simulate_reactive|baseline_readiness|prepare_gameformer_subset|gameformer_report|export_r3r}' ;exit 2 ;;
esac
