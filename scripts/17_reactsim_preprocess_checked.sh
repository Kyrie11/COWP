#!/usr/bin/env bash
# No silent missing-token skipping or misleading 'full' benchmark labels.
# This script stages none of the raw data: use 15_reactsim_nuplan_stage.sh apply first.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
mode="${1:-help}"; shift || true
repo="${RIDE_REACTSIM_REPO:-$DATASET_ROOT/RIDE/research_baselines/ReactSim_Bench}"
workspace="${RIDE_REACTSIM_WORK:-$DATASET_ROOT/RIDE/reactsim_workspace}"
dbdir="$repo/data/nuplan/nuplan-v1.1/splits/trainval"
idsdir="$repo/reactsim_data_pipeline"
mkdir -p "$workspace/diagnostics"
case "$mode" in
  token)
    dbname="${1:-2021.08.27.03.47.52_veh-53_01182_01302.db}"
    token="${2:-2cdb26865dfa5261}"
    python -m ride.reactsim_tokens --db "$dbdir/$dbname" --token "$token" --quick-check \
      --output "$workspace/diagnostics/token_${token}.json"
    ;;
  inspect-val|inspect-train)
    split="${mode#inspect-}"
    ids="$idsdir/.scenario_ids_${split}.json"
    test -s "$ids" || { echo "Missing author scenario ids $ids" >&2; exit 3; }
    max="${RIDE_REACTSIM_AUDIT_LIMIT:-0}"
    echo "Read-only $split IDs audit; limit=$max (0 means all)" >&2
    python -m ride.reactsim_tokens --ids "$ids" --db-dir "$dbdir" \
       --max-tokens "$max" --output "$workspace/diagnostics/${split}_token_audit.json"
    ;;
  val|train)
    if (( $# )); then
      echo 'Refusing extra build_cache options: full official val/train IDs must not be silently truncated.' >&2
      exit 2
    fi
    split="$mode"
    ids="$idsdir/.scenario_ids_${split}.json"
    test -s "$ids" || { echo "Missing author scenario ids $ids" >&2; exit 3; }
    test -d "$dbdir" || { echo "DB links missing: $dbdir" >&2; exit 3; }
    if [[ "$split" == train && "${RIDE_ALLOW_FULL_REACTSIM_TRAIN:-0}" != 1 ]]; then
      echo 'The full training cache can be ~642k scenarios. Set RIDE_ALLOW_FULL_REACTSIM_TRAIN=1 explicitly.' >&2
      exit 3
    fi
    echo "Auditing ALL $split official tokens before preprocessing. No missing IDs are dropped." >&2
    python -m ride.reactsim_tokens --ids "$ids" --db-dir "$dbdir" \
      --output "$workspace/diagnostics/${split}_token_audit.json" --enforce
    out="$repo/data/nuplan_processed/$split"
    workers="${RIDE_REACTSIM_WORKERS:-4}"
    [[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid workers' >&2; exit 2; }
    # Do not run author's preprocess.sh: at the verified official main commit,
    # its default output path is data/test/nuplan_processed, not data/nuplan_processed.
    # Use the public build_cache.py directly with the correct output directory.
    echo "Official $split preprocessing output=$out workers=$workers (NO train/test mixing)" >&2
    (cd "$idsdir" && python build_cache.py \
      --scenario_ids_path "$ids" \
      --data_root "$repo/data/nuplan" \
      --db_dir "$dbdir" --map_root "$repo/data/nuplan/maps" \
      --output_dir "$out" --num_workers "$workers" --skip_existing)
    ;;
  *)
    cat <<EOF
Usage:
 bash scripts/17_reactsim_preprocess_checked.sh token [DB_FILENAME] [HEX_TOKEN]
 bash scripts/17_reactsim_preprocess_checked.sh inspect-val
 bash scripts/17_reactsim_preprocess_checked.sh inspect-train
 bash scripts/17_reactsim_preprocess_checked.sh val
 RIDE_ALLOW_FULL_REACTSIM_TRAIN=1 bash scripts/17_reactsim_preprocess_checked.sh train
The val command audits all official val tokens, then builds the official VAL cache.
Do not filter out missing IDs and call it full ReactSim evaluation.
EOF
    [[ "$mode" == help ]] || exit 2
    ;;
esac
