#!/usr/bin/env bash
# Native WOMD / EIC-DRR hypotheses verification entrypoint.
# USAGE: bash NEXT_EICDRR_COMMANDS_CN.sh audit|cohort_pilot|cohort_full|screen|paired|waymax_pilot|readiness
# Run from extracted COWP_revision directory. Outputs are isolated: no checkpoint deletion.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$ROOT/COWP_revision" ]]; then ROOT="$ROOT/COWP_revision"; fi
cd "$ROOT"
RAW_ROOT="${RAW_WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
DATA_ROOT="${EICDRR_DATA_ROOT:-/data0/senzeyu2/dataset/COWP/research_native_womd/eicdrr_nonoverlap_20261013}"
COHORT="${EICDRR_COHORT:-${DATA_ROOT}/cohort}"
OUT="${EICDRR_RESULTS_DIR:-${DATA_ROOT}/results}"
MODE="${1:-help}"
mkdir -p "$OUT"
case "$MODE" in
  audit)
    # The original JSON files are required; use environment paths if needed.
    SCREEN="${ORIGINAL_SCREEN_JSON:-/mnt/data/reactive_screen.json}"
    MANIFEST="${ORIGINAL_MANIFEST_JSON:-/mnt/data/cohort_manifest.json}"
    python -m research_probe.paired_evidence \
      --screen "$SCREEN" --manifest "$MANIFEST" --progress-tolerance 3.0 \
      --output "$OUT/original_paired_audit.json"
    ;;
  cohort_pilot)
    # Avoid old 7,200 train/val/heldout IDs; require a clean output folder.
    [[ ! -e "$COHORT/cohort_manifest.json" ]] || { echo 'ERROR: cohort exists; set a new EICDRR_COHORT path' >&2; exit 2; }
    python -m research_probe.native_womd \
      --raw-root "$RAW_ROOT" --output "$COHORT" --split validation --format tfexample \
      --max-shards 4 --max-records-per-shard 350 \
      --keep-random 120 --keep-interactive 120 \
      --exclude-ids reference_manifests/compact_all_7200_ids.txt
    ;;
  cohort_full)
    # Heavy I/O: all 150 validation shards. No maximum per shard; register salt and split first.
    [[ ! -e "$COHORT/cohort_manifest.json" ]] || { echo 'ERROR: cohort exists; set a new EICDRR_COHORT path' >&2; exit 2; }
    python -m research_probe.native_womd \
      --raw-root "$RAW_ROOT" --output "$COHORT" --split validation --format tfexample \
      --max-shards 0 --max-records-per-shard 0 --require-complete-split \
      --keep-random 500 --keep-interactive 500 --keep-chain 400 --keep-crossing 400 \
      --exclude-ids reference_manifests/compact_all_7200_ids.txt
    ;;
  verify_full)
    python -m research_probe.verify_native_coverage --cohort "$COHORT" --expect-shards 150 --require-full-records
    ;;
  screen)
    python -m research_probe.reactive_stress \
      --cohort "$COHORT" --max-scenes 0 --selection-order hash \
      --progress-tolerance 3.0 --ade-max 4.0 \
      --output "$OUT/reactive_screen_hash.json"
    python -m research_probe.paired_evidence \
      --screen "$OUT/reactive_screen_hash.json" \
      --manifest "$COHORT/cohort_manifest.json" \
      --progress-tolerance 3.0 --output "$OUT/paired_audit_hash.json"
    ;;
  paired)
    python -m research_probe.paired_intervention \
      --cohort "$COHORT" --max-scenes 0 --selection-order hash \
      --progress-tolerance 3.0 --output "$OUT/paired_ghost.json"
    ;;
  waymax_pilot)
    # Requires a SEPARATE compatible Waymax + TensorFlow/JAX installation and full TFExample shards.
    # This version-sensitive adapter is a preliminary Waymax IDM pilot, not validated science.
    python -m research_probe.waymax_paired_idm \
      --cohort "$COHORT" --raw-root "$RAW_ROOT" \
      --max-scenes "${WAYMAX_PILOT_N:-12}" --steps 40 \
      --ego-speeds 6 12 18 --progress-tolerance 3.0 \
      --output "$OUT/waymax_pilot.json"
    ;;
  dense_response)
    for D in 0 3 6; do
      python -m research_probe.paired_intervention --cohort "$COHORT" \
        --max-scenes 0 --selection-order hash --progress-tolerance 3.0 \
        --responder-delay-steps "$D" \
        --ego-accelerations -3 -2.75 -2.5 -2.25 -2 -1.75 -1.5 -1.25 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.25 1.5 \
        --output "$OUT/paired_dense_delay_${D}.json"
    done
    ;;
  h1_finite)
    python -m research_probe.h1_compatibility_csp --cohort "$COHORT" --max-scenes 0 --max-agents 6 --output "$OUT/h1_csp.json"
    ;;
  h3_finite)
    python -m research_probe.h3_successor_proxy --cohort "$COHORT" --max-scenes 0 --prefix-steps 5 --total-steps 20 --output "$OUT/h3_backup.json"
    ;;
  waymax_targets)
    [[ -n "${WAYMAX_TARGET_IDS_FILE:-}" ]] || { echo 'Set WAYMAX_TARGET_IDS_FILE to frozen, discovery-only scenario list' >&2; exit 2; }
    python -m research_probe.waymax_paired_idm --cohort "$COHORT" --raw-root "$RAW_ROOT" \
      --scenario-ids-file "$WAYMAX_TARGET_IDS_FILE" --max-scenes "${WAYMAX_PILOT_N:-12}" \
      --steps 40 --ego-speeds 6 12 18 --progress-tolerance 3.0 --output "$OUT/waymax_targets.json"
    ;;
  readiness)
    python -m research_probe.baseline_readiness \
      --cohort "$COHORT" \
      --gameformer-root "$ROOT/third_party/GameFormer-main" \
      --ra-cbf-root "$ROOT/third_party/learning_responsibility_allocation-main" \
      --output "$OUT/baseline_readiness_latest.json"
    ;;
  help|*)
    cat <<'USAGE'
Run in extracted COWP_revision using bash NEXT_EICDRR_COMMANDS_CN.sh MODE
  audit          Re-audit uploaded JSON (set ORIGINAL_SCREEN_JSON, ORIGINAL_MANIFEST_JSON)
  cohort_pilot   Fresh 4-shard native cohort excluding all Compact-5k 7200 IDs
  cohort_full    Full shard native cohort, time and storage intensive, separate output folder
  verify_full    Strict 150-shard manifest coverage acceptance
  screen         Longitudinal IDM-like reactive screen, all cohort scenes, hash order
  paired         Same-scene ghost-ego-to-others reactive contrast (simplified model)
  waymax_pilot   Experimental Waymax IDM route policy paired rollout; requires compatible runtime
  dense_response  Fine 0.25m/s2 ego grid with delayed-response sensitivity 0/0.6/1.2s
  h1_finite      Exhaustive finite-bank joint-response CSP; never real infeasibility proof
  h3_finite      Execution-prefix finite backup survival; never certified viability
  waymax_targets Pre-screened discovery-only Waymax pilot (explicit ID file)
  readiness      Honest GameFormer/RA-CBF/JFP baseline readiness audit
Env: RAW_WOMD_ROOT, EICDRR_DATA_ROOT, EICDRR_COHORT, EICDRR_RESULTS_DIR
Never treat zero errors in a proxy script as Waymax closed-loop validation.
USAGE
    ;;
esac
