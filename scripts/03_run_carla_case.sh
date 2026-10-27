#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
CASE="${1:?Usage: bash scripts/03_run_carla_case.sh convoy|cut_in|intersection|support_gap}"
case "$CASE" in convoy|cut_in|intersection|support_gap) ;; *) echo "Unknown case: $CASE" >&2; exit 2;; esac
mkdir -p "$RIDE_OUTPUT_ROOT/carla"
OUT="$RIDE_OUTPUT_ROOT/carla/${CASE}.jsonl"
FROZEN="$RIDE_OUTPUT_ROOT/frozen/${CASE}_anchors.jsonl"
if [[ ! -f "$FROZEN" ]]; then echo "Freeze anchors first: bash scripts/03_precommit_scenes.sh $CASE" >&2; exit 3; fi
if [[ -e "$OUT" ]]; then echo "REFUSING TO OVERWRITE outcome file $OUT; use a new RIDE_OUTPUT_ROOT" >&2; exit 3; fi
EXTRA=()
if [[ "$CASE" == "intersection" ]]; then
  EXTRA+=(--joint-profile early_slow --joint-limit 9 --joint-target-grid 6 9 12)
fi
if [[ "$CASE" == "support_gap" ]]; then
  EXTRA+=(--joint-profile early_slow --joint-limit 36 --joint-target-grid 4 6 8 9 10 12
    --recovery-profile early_slow --recovery-time-s 3.5 --recovery-options stop cruise)
fi
cd "$RIDE_HOME"
python -m ride.carla.ports --mode client --host "$CARLA_HOST" \
  --rpc-port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT"
python -m ride.carla.run \
  --host "$CARLA_HOST" --port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT" \
  --archetype "$CASE" --town "${CARLA_TOWN:-Town04}" \
  --anchors "${RIDE_ANCHORS:-1}" --repetitions "${RIDE_REPEATS:-1}" \
  --seed "${RIDE_SEED:-17}" --cruise-mps "${RIDE_CRUISE_MPS:-9}" \
  --warmup-ticks 0 --no-rendering --anchors-from "$FROZEN" \
  --families tm behavior \
  --profiles early_slow late_slow mild_early mild_late \
  "${EXTRA[@]}" --output "$OUT" \
  2>&1 | tee "$RIDE_OUTPUT_ROOT/carla/${CASE}.log"
python -m ride.evidence --input "$OUT" --output "$RIDE_OUTPUT_ROOT/carla/${CASE}_audit.json"
python -m ride.carla.analyze --jsonl "$OUT" --out "$RIDE_OUTPUT_ROOT/carla/${CASE}_joint_backup_audit.json"
