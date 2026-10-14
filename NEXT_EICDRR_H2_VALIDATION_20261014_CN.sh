#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TASK="${1:-help}"
RAW="${RAW_WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
BASE="${EICDRR_FULL_ROOT:-/data0/senzeyu2/dataset/COWP/eicdrr_full_validation_20261014}"
COHORT="$BASE/cohort";OUT="$BASE/results";GATE="$BASE/h2_gate_v2"
mkdir -p "$OUT" "$GATE"
case "$TASK" in
 tests)
   PYTHONPATH=. python -m pytest -q tests --ignore=tests/test_core.py ;;
 audit_uploaded)
   MATCHED_JSON="${MATCHED_JSON:-$OUT/matched_dev.json}"
   python -m research_probe.h2_evidence_gate --matched "$MATCHED_JSON" \
     --inventory "$OUT/validation_inventory.json" \
     --cohort-manifest "$COHORT/cohort_manifest.json" --out-dir "$GATE" ;;
 raw_all)
   python -m research_probe.verify_cohort_identity --cohort "$COHORT" \
     --raw-root "$RAW" --output "$OUT/identity_raw_all.json" ;;
 observational_fit)
   python -m research_probe.observational_response_calibration --cohort "$COHORT" \
     --partition development --max-scenes "${FIT_DEV_N:-320}" \
     --steps 10 --output "$GATE/observational_fit_dev.json" ;;
 matched_dev_320)
   python -m research_probe.matched_response_test --cohort "$COHORT" --partition development \
     --max-scenes "${MATCHED_DEV_N:-320}" --steps 20 --progress-tolerance 3 \
     --output "$OUT/matched_dev_320_v2.json" ;;
 matched_dev_all)
   python -m research_probe.matched_response_test --cohort "$COHORT" --partition development \
     --max-scenes 0 --steps 20 --progress-tolerance 3 \
     --output "$OUT/matched_dev_all_v2.json" ;;
 triage_deep)
   python -m research_probe.matched_response_test --cohort "$COHORT" --partition development \
      --scenario-ids-file "$GATE/h2_priority_dev_ids.txt" --max-scenes 0 \
      --steps 40 --progress-tolerance 3 --output "$GATE/matched_priority_8s_proxy.json" ;;
 waymax_pair)
   python -m research_probe.waymax_paired_idm --cohort "$COHORT" --raw-root "$RAW" \
      --scenario-ids-file "$GATE/h2_priority_dev_ids.txt" --partition development \
      --max-scenes "${WAYMAX_PILOT_N:-8}" --steps 40 --output "$GATE/waymax_priority_pilot.json" ;;
 interactive_inventory)
   python -m research_probe.split_inventory --raw-root "$RAW" \
     --split validation_interactive --output "$GATE/validation_interactive_inventory.json" ;;
 interactive_cohort)
   [[ ! -f "$BASE/interactive/cohort_manifest.json" ]] || { echo 'Refuse overwrite interactive cohort' >&2;exit 2; }
   python -m research_probe.native_womd --raw-root "$RAW" --output "$BASE/interactive" \
     --split validation_interactive --format tfexample --require-complete-split \
     --max-shards 0 --max-records-per-shard 0 --max-neighbors 20 \
     --keep-random 350 --keep-interactive 500 --keep-chain 350 \
     --keep-crossing 350 --keep-cpa 350 --neighbor-radius 75 \
     --holdout-fraction 0.30 \
     --exclude-ids "$(pwd)/reference_manifests/compact_all_7200_ids.txt" ;;
 interactive_overlap)
   python -m research_probe.cohort_overlap_audit \
     --a "$COHORT/cohort_manifest.json" --b "$BASE/interactive/cohort_manifest.json" \
     --output "$GATE/interactive_overlap.json" ;;
 independent_gate)
   : "${MODEL_PAIR_JSONL:?Export actual independent simulator pair trials and set MODEL_PAIR_JSONL}"
   python -m research_probe.independent_model_gate --jsonl "$MODEL_PAIR_JSONL" \
     --output "$GATE/independent_model_gate.json" ;;
 *) cat <<'EOF'
Read RESEARCH_DECISION_AND_PROTOCOL_CN.md first.
  tests -> audit_uploaded -> raw_all -> observational_fit
  -> matched_dev_320 -> audit_uploaded (MATCHED_JSON=$OUT/matched_dev_320_v2.json)
  -> matched_dev_all (only if pilot robust; expensive)
  -> triage_deep -> waymax_pair -> [actual SECOND architecture simulator] -> independent_gate
  interactive_inventory -> [if absent download ONLY validation_interactive] -> interactive_cohort
  sealed holdout: DO NOT RUN until all thresholds, models, metrics, protocol frozen.
No automatic claims about real human counterfactual behavior.
EOF
esac
