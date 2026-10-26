#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
mkdir -p "$RIDE_OUTPUT_ROOT/carla"
cd "$RIDE_HOME"
# Narrow bank 3x3, pre-registered SUBSET of wide 6x6 bank.
python -m research_probe.carla_h2.run_carla --archetype support_gap \
  --town "${CARLA_TOWN:-Town04}" --anchors "${RIDE_ANCHORS:-1}" \
  --repetitions 1 --seed "${RIDE_SEED:-17}" --no-rendering \
  --families tm behavior --profiles early_slow late_slow \
  --joint-profile early_slow --joint-limit 9 --joint-target-grid 6 9 12 \
  --output "$RIDE_OUTPUT_ROOT/carla/support_gap_NARROW.jsonl"
# Wide bank MUST use same scenario seed, settings, horizon and physical simulator.
python -m research_probe.carla_h2.run_carla --archetype support_gap \
  --town "${CARLA_TOWN:-Town04}" --anchors "${RIDE_ANCHORS:-1}" \
  --repetitions 1 --seed "${RIDE_SEED:-17}" --no-rendering \
  --families tm behavior --profiles early_slow late_slow \
  --joint-profile early_slow --joint-limit 36 --joint-target-grid 4 6 8 9 10 12 \
  --output "$RIDE_OUTPUT_ROOT/carla/support_gap_WIDE.jsonl"
python -m ride.support \
 --narrow "$RIDE_OUTPUT_ROOT/carla/support_gap_NARROW.jsonl" \
 --wide "$RIDE_OUTPUT_ROOT/carla/support_gap_WIDE.jsonl" \
 --output "$RIDE_OUTPUT_ROOT/carla/support_gap_bank_ablation.json"
