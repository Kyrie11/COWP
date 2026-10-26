#!/usr/bin/env bash
# EICDRR mechanism discovery -> provenance -> validation. No external model is faked.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${1:-help}"
RAW_ROOT="${RAW_WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
DATA_ROOT="${EICDRR_DATA_ROOT:-/data0/senzeyu2/dataset/COWP/eicdrr_identity_audited_20261013}"
COHORT="${EICDRR_COHORT:-${DATA_ROOT}/cohort}"
OUT="${EICDRR_RESULTS_DIR:-${DATA_ROOT}/results}"
EXCLUDE="$ROOT/reference_manifests/compact_all_7200_ids.txt"
mkdir -p "$OUT"
case "$MODE" in
  tests)
    python -m pytest -q tests --ignore=tests/test_core.py
    echo 'NOTE: tests/test_core.py references unbundled legacy ocrap, not part of tested COWP package'
    ;;
  old_audit)
    : "${ORIGINAL_SCREEN_JSON:?Set ORIGINAL_SCREEN_JSON}"
    : "${ORIGINAL_MANIFEST_JSON:?Set ORIGINAL_MANIFEST_JSON}"
    python -m research_probe.paired_evidence \
      --screen "$ORIGINAL_SCREEN_JSON" --manifest "$ORIGINAL_MANIFEST_JSON" \
      --progress-tolerance 3 --output "$OUT/old_screen_audit.json"
    ;;
  cohort_dev)
    [[ ! -e "$COHORT/cohort_manifest.json" ]] || { echo 'Refusing to overwrite existing cohort manifest' >&2; exit 2; }
    python -m research_probe.native_womd --raw-root "$RAW_ROOT" \
       --output "$COHORT" --split validation --format tfexample \
       --max-shards 4 --max-records-per-shard 350 \
       --keep-random 120 --keep-interactive 120 --exclude-ids "$EXCLUDE"
    echo 'WARNING: DEVELOPMENT_ONLY first four WOMD shards; not a population sample/holdout'
    ;;
  cohort_full)
    [[ ! -e "$COHORT/cohort_manifest.json" ]] || { echo 'Refusing to overwrite existing cohort manifest' >&2; exit 2; }
    python -m research_probe.native_womd --raw-root "$RAW_ROOT" \
      --output "$COHORT" --split validation --format tfexample \
      --max-shards 0 --max-records-per-shard 0 --require-complete-split \
      --keep-random 500 --keep-interactive 500 --keep-chain 400 --keep-crossing 400 \
      --exclude-ids "$EXCLUDE"
    ;;
  identity_npz)
    python -m research_probe.verify_cohort_identity --cohort "$COHORT" \
      --historical-ids "$EXCLUDE" --output "$OUT/identity_npz.json"
    ;;
  identity_raw12)
    python -m research_probe.verify_cohort_identity --cohort "$COHORT" \
      --historical-ids "$EXCLUDE" --raw-root "$RAW_ROOT" --limit 12 \
      --output "$OUT/identity_raw12.json"
    ;;
  identity_raw_all)
    python -m research_probe.verify_cohort_identity --cohort "$COHORT" \
      --historical-ids "$EXCLUDE" --raw-root "$RAW_ROOT" \
      --output "$OUT/identity_raw_all.json"
    ;;
  matched_dev)
    test -f "$OUT/identity_npz.json" || { echo 'Run identity_npz first' >&2; exit 2; }
    python - "$OUT/identity_npz.json" <<'PY'
import sys,json
assert json.load(open(sys.argv[1]))['pass'], 'Cohort identity verification FAILED'
PY
    python -m research_probe.matched_response_test --cohort "$COHORT" \
      --max-scenes "${DEV_N:-80}" --steps 20 --selection-order hash \
      --progress-tolerance 3 --output "$OUT/matched_dev.json"
    ;;
  matched_full)
    test -f "$OUT/identity_npz.json" || { echo 'Run identity_npz first' >&2; exit 2; }
    python - "$OUT/identity_npz.json" <<'PY'
import sys,json
assert json.load(open(sys.argv[1]))['pass'], 'Cohort identity verification FAILED'
PY
    python -m research_probe.matched_response_test --cohort "$COHORT" \
      --max-scenes 0 --steps 20 --selection-order hash --progress-tolerance 3 \
      --output "$OUT/matched_all.json"
    echo 'NOTE: matched_full is not a sealed holdout if data/scenes were previously inspected'
    ;;
  ghost)
    python -m research_probe.paired_intervention --cohort "$COHORT" \
      --max-scenes 0 --selection-order hash --progress-tolerance 3 \
      --output "$OUT/paired_ghost.json"
    ;;
  finite_h1)
    python -m research_probe.h1_compatibility_csp --cohort "$COHORT" \
      --max-scenes 0 --max-agents 6 --output "$OUT/h1_csp_finite.json"
    ;;
  finite_h3)
    python -m research_probe.h3_successor_proxy --cohort "$COHORT" \
      --max-scenes 0 --prefix-steps 5 --total-steps 20 --output "$OUT/h3_backup_finite.json"
    ;;
  waymax_idm_pilot)
    test -f "$OUT/identity_raw12.json" || { echo 'Run identity_raw12 first' >&2; exit 2; }
    python - "$OUT/identity_raw12.json" <<'PY'
import sys,json
assert json.load(open(sys.argv[1]))['pass'], 'Raw identity verification FAILED'
PY
    python -m research_probe.waymax_paired_idm --cohort "$COHORT" --raw-root "$RAW_ROOT" \
      --max-scenes "${WAYMAX_PILOT_N:-12}" --steps 40 --ego-speeds 6 12 18 \
      --progress-tolerance 3 --output "$OUT/waymax_idm_pilot.json"
    echo 'WARNING: Waymax IDM pilot only, not independently validated human policy, and not GameFormer/RA-CBF'
    ;;
  baseline_status)
    python -m research_probe.baseline_readiness --cohort "$COHORT" \
       --gameformer-root "$ROOT/third_party/GameFormer-main" \
       --ra-cbf-root "$ROOT/third_party/learning_responsibility_allocation-main" \
       --output "$OUT/baseline_readiness.json"
    ;;
  help|*)
    cat <<'EOF'
Research gate pipeline:
  tests                local numpy/torch COWP tests (except unrelated missing ocrap legacy)
  old_audit            original JSON re-audit (provide ORIGINAL_SCREEN_JSON/MANIFEST_JSON)
  cohort_dev           new 4-shard dev dataset, all 7200 historic IDs excluded
  cohort_full          full 150 shard cohort, full-record processing (slow)
  identity_npz         ego_index, agent selection, and historical-ID consistency
  identity_raw12       verify first 12 cases against actual raw WOMD SDC/source record
  identity_raw_all     source SDC verify all cohort entries (slow)
  matched_dev          80 scenes, smooth progress-matched intervention hypothesis screen
  matched_full         all cohort scenes (exploratory, not sealed holdout)
  ghost                old four-control paired ghost-leader contrasts
  finite_h1            finite CSP search; no real infeasibility guarantee
  finite_h3            finite backup screen; no recursive viability guarantee
  waymax_idm_pilot     12-scene Waymax reactive IDM paired-policy pilot (needs server)
  baseline_status      readiness only; no claim GameFormer / RA-CBF was run
Environment: RAW_WOMD_ROOT, EICDRR_DATA_ROOT, EICDRR_COHORT, EICDRR_RESULTS_DIR,
  DEV_N, WAYMAX_PILOT_N, ORIGINAL_SCREEN_JSON, ORIGINAL_MANIFEST_JSON.
EOF
    ;;
esac
