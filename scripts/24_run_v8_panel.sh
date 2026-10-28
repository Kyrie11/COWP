#!/usr/bin/env bash
# Usage: bash scripts/24_run_v8_panel.sh 17 dev
#        bash scripts/24_run_v8_panel.sh 23 confirm
set -euo pipefail
seed="${1:?Specify response seed}"; stage="${2:?Specify dev or confirm}"
[[ "$seed" =~ ^[1-9][0-9]*$ ]] || exit 2
case "$stage" in dev|confirm) ;; *) echo 'Use dev or confirm' >&2; exit 2;; esac
if [[ "$stage" == dev ]]; then
  export RIDE_V8_ANCHOR_SEED_EXPECTED=9701
  unset RIDE_FORBIDDEN_ANCHORS_JSONL
else
  export RIDE_V8_ANCHOR_SEED_EXPECTED="${RIDE_V8_CONFIRM_ANCHOR_SEED:-9901}"
  dev_manifest="${RIDE_V8_EVIDENCE_ROOT:-${DATASET_ROOT:-/data0/senzeyu2/dataset}/RIDE/evidence/rq12_v8_locked}/dev/seed_17/rear_gap_35/rear_delta_0/frozen/cut_in_anchors.jsonl"
  [[ -f "$dev_manifest" ]] || { echo "RUN DEV FIRST: $dev_manifest" >&2; exit 3; }
  export RIDE_FORBIDDEN_ANCHORS_JSONL="$dev_manifest"
fi
source "$(dirname "$0")/23_v8_env_lock.sh"
export RIDE_SEED="$seed"
if [[ "$stage" == dev ]]; then export RIDE_ANCHORS=10; else export RIDE_ANCHORS=8; fi
export RIDE_FACTORIAL_OUTPUT_ROOT="${RIDE_V8_EVIDENCE_ROOT:-$DATASET_ROOT/RIDE/evidence/rq12_v8_locked}/$stage/seed_${seed}"
if [[ -e "$RIDE_FACTORIAL_OUTPUT_ROOT" ]]; then
  echo "REFUSE_OVERWRITE: $RIDE_FACTORIAL_OUTPUT_ROOT" >&2; exit 3
fi
bash "$RIDE_HOME/scripts/19_cut_in_factorial_pilot.sh"
bash "$RIDE_HOME/scripts/20_factorial_report.sh"
bash "$RIDE_HOME/scripts/21_audit_factorial_mechanisms.sh"
python -m ride.v7_forensic_panel \
  --panel "$RIDE_FACTORIAL_OUTPUT_ROOT/rear_gap_35/rear_delta_0/carla/cut_in.jsonl" \
  --output "$RIDE_FACTORIAL_OUTPUT_ROOT/v8_forensic_status.json"
