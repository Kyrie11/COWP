#!/usr/bin/env bash
# Full 150-shard, raw-identity gated, H1/H2/H3 staged EICDRR evidence pipeline.
# No scientific causal claim is automatically enabled.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODE="${1:-help}"
RAW="${RAW_WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
BASE="${EICDRR_FULL_ROOT:-/data0/senzeyu2/dataset/COWP/eicdrr_full_validation_20261014}"
COHORT="$BASE/cohort"; OUT="$BASE/results"
EXCLUDE="$(pwd)/reference_manifests/compact_all_7200_ids.txt"
mkdir -p "$OUT"
assert_pass() {
    python - "$1" <<'PY'
import json,sys
p=sys.argv[1];x=json.load(open(p))
assert x.get('pass') is True, f'IDENTITY_GATE_FAILED: {p}'
PY
}
case "$MODE" in
 inventory)
   python -m research_probe.split_inventory --raw-root "$RAW" --split validation --output "$OUT/validation_inventory.json"
   ;;
 tests)
   PYTHONPATH=. python -m pytest -q tests --ignore=tests/test_core.py
   ;;
 cohort_full)
   python -m research_probe.split_inventory --raw-root "$RAW" --split validation --output "$OUT/validation_inventory.json"
   [[ ! -e "$COHORT/cohort_manifest.json" ]] || { echo 'cohort exists; never overwrite a frozen sample' >&2;exit 2; }
   mkdir -p "$COHORT"
   python -m research_probe.native_womd --raw-root "$RAW" --output "$COHORT" \
     --split validation --format tfexample --require-complete-split \
     --max-shards 0 --max-records-per-shard 0 \
     --keep-random 600 --keep-interactive 700 --keep-chain 500 --keep-crossing 500 --keep-cpa 500 \
     --max-neighbors 20 --neighbor-radius 75 \
     --holdout-fraction 0.30 --exclude-ids "$EXCLUDE" \
      | tee "$OUT/full150_scan_stdout.txt"
   ;;
 identity)
   python -m research_probe.verify_cohort_identity --cohort "$COHORT" --historical-ids "$EXCLUDE" --output "$OUT/identity_npz.json"
   python -m research_probe.verify_cohort_identity --cohort "$COHORT" --historical-ids "$EXCLUDE" \
      --raw-root "$RAW" --limit 32 --output "$OUT/identity_raw32.json"
   ;;
 matched_dev)
   assert_pass "$OUT/identity_npz.json"
   assert_pass "$OUT/identity_raw32.json"
   python -m research_probe.matched_response_test --cohort "$COHORT" --selection-order hash \
     --partition development --max-scenes "${DEV_N:-160}" \
     --progress-tolerance 3 --steps 20 --output "$OUT/matched_dev.json"
   ;;
 matched_dev_all)
   assert_pass "$OUT/identity_npz.json"
   assert_pass "$OUT/identity_raw32.json"
   python -m research_probe.matched_response_test --cohort "$COHORT" --partition development \
      --max-scenes 0 --progress-tolerance 3 --steps 20 --output "$OUT/matched_dev_all.json"
   ;;
 holdout_once)
   [[ "${EICDRR_HOLDOUT_UNLOCK:-}" == "yes_frozen_protocol" ]] || { echo 'Requires EICDRR_HOLDOUT_UNLOCK=yes_frozen_protocol after freezing protocol' >&2;exit 2; }
   assert_pass "$OUT/identity_npz.json"
   assert_pass "$OUT/identity_raw32.json"
   python -m research_probe.matched_response_test --cohort "$COHORT" --partition sealed_holdout \
     --max-scenes 0 --progress-tolerance 3 --steps 20 --output "$OUT/matched_holdout_frozen.json"
   ;;
 raw_all)
   assert_pass "$OUT/identity_npz.json"
   python -m research_probe.verify_cohort_identity --cohort "$COHORT" --historical-ids "$EXCLUDE" \
      --raw-root "$RAW" --output "$OUT/identity_raw_all.json"
   ;;
 finite_h1_dev)
   assert_pass "$OUT/identity_npz.json"
   python -m research_probe.h1_compatibility_csp --cohort "$COHORT" \
     --partition development --max-scenes "${H1_SCENES:-160}" --max-agents 8 --output "$OUT/h1_finite_dev.json"
   ;;
 finite_h3_dev)
   assert_pass "$OUT/identity_npz.json"
   python -m research_probe.h3_successor_proxy --cohort "$COHORT" \
     --partition development --max-scenes "${H3_SCENES:-160}" --output "$OUT/h3_finite_dev.json"
   ;;
 waymax_pilot)
   assert_pass "$OUT/identity_raw32.json"
   python -m research_probe.waymax_paired_idm --cohort "$COHORT" --raw-root "$RAW" \
     --partition development --max-scenes "${WAYMAX_PILOT_N:-16}" --steps 40 --ego-speeds 6 12 18 \
     --output "$OUT/waymax_paired_idm_pilot.json"
   ;;
 audit_uploaded)
   : "${MATCHED_JSON:?Set MATCHED_JSON from uploaded results}"
   : "${GHOST_JSON:?Set GHOST_JSON from uploaded results}"
   ID_ARGS=()
   [[ -f "${IDENTITY_JSON:-$OUT/identity_raw32.json}" ]] && ID_ARGS=(--raw-identity "${IDENTITY_JSON:-$OUT/identity_raw32.json}")
   python -m research_probe.audit_latest_results --matched "$MATCHED_JSON" --ghost "$GHOST_JSON" \
     "${ID_ARGS[@]}" --output "$OUT/audit_latest_results.json"
   ;;
 *) cat <<'EOF'
Recommended order:
  inventory -> tests -> cohort_full -> identity -> matched_dev
  -> matched_dev_all (exploration) -> raw_all -> waymax_pilot
  -> [freeze protocol & assess] -> holdout_once
  -> finite_h1_dev -> finite_h3_dev -> H4 external baselines
Training: do not download before validation evidence and independent-model validation.
Testing: WOMD motion benchmark future labels are hidden; cannot calibrate ADE/H2 there.
Important: positive simulations are hypothesis candidates, not observed human do-effects.
EOF
   ;;
esac
