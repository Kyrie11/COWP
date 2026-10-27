#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
FROZEN="$RIDE_OUTPUT_ROOT/frozen/support_gap_anchors.jsonl"
if [[ ! -f "$FROZEN" ]]; then echo "First freeze support_gap anchors" >&2; exit 3; fi
mkdir -p "$RIDE_OUTPUT_ROOT/carla"
NARROW="$RIDE_OUTPUT_ROOT/carla/support_gap_NARROW.jsonl"
WIDE="$RIDE_OUTPUT_ROOT/carla/support_gap_WIDE.jsonl"
if [[ -e "$NARROW" || -e "$WIDE" ]]; then echo "Refusing to overwrite finite-bank evidence" >&2; exit 3; fi
cd "$RIDE_HOME"
python -m ride.carla.ports --mode client --host "$CARLA_HOST" \
  --rpc-port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT"
COMMON=(--host "$CARLA_HOST" --port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT" --archetype support_gap --town "${CARLA_TOWN:-Town04}" \
  --anchors "${RIDE_ANCHORS:-1}" --repetitions "${RIDE_REPEATS:-1}" \
  --seed "${RIDE_SEED:-17}" --cruise-mps "${RIDE_CRUISE_MPS:-9}" \
  --no-rendering --anchors-from "$FROZEN" --families tm behavior \
  --profiles early_slow late_slow --joint-profile early_slow)
# Narrow 3x3 candidate bank must be a strict subset of the wide 6x6 bank.
python -m ride.carla.run "${COMMON[@]}" --joint-limit 9 \
  --joint-target-grid 6 9 12 --output "$NARROW"
python -m ride.carla.run "${COMMON[@]}" --joint-limit 36 \
  --joint-target-grid 4 6 8 9 10 12 --output "$WIDE"
python -m ride.support --narrow "$NARROW" --wide "$WIDE" \
  --output "$RIDE_OUTPUT_ROOT/carla/support_gap_bank_ablation.json"
