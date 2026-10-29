#!/usr/bin/env bash
# REQUIRES actual CARLA 0.9.16 headless server; never silently alters a prior frozen protocol.
# Precommitted, exploratory four-car response delay/brake-command heterogeneity.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
: "${RIDE_V14_STRESS_ROOT:?Set a new output directory}"
: "${CARLA_TOWN:?set Town04, Town05 etc and preflight first}"
[[ ! -e "$RIDE_V14_STRESS_ROOT" ]] || { echo "REFUSE_EXISTING_ROOT $RIDE_V14_STRESS_ROOT" >&2; exit 3; }
export RIDE_OUTPUT_ROOT="$RIDE_V14_STRESS_ROOT"
export RIDE_REPEATS=1
unset RIDE_V14_SOURCE_INACTIVE_STRESS_METADATA
export RIDE_FAMILIES='tm mixed_b_tm_c_behavior mixed_b_tm_c_delayed_behavior'
export RIDE_PROFILES='cruise early_slow late_slow mild_early mild_late speed_early speed_late'
export RIDE_SMOKE_REQUIRE_VALID=0
export RIDE_CRUISE_MPS="${RIDE_CRUISE_MPS:-7.5}"
export RIDE_CUTIN_EGO_GAP_M="${RIDE_CUTIN_EGO_GAP_M:-20}"
export RIDE_CUTIN_REAR_GAP_M="${RIDE_CUTIN_REAR_GAP_M:-35}"
export RIDE_CHAIN_BC_GAP_M="${RIDE_CHAIN_BC_GAP_M:-20}"
export RIDE_CHAIN_C_SPEED_DELTA_MPS="${RIDE_CHAIN_C_SPEED_DELTA_MPS:-0.5}"
export RIDE_CUTIN_LATE_SWITCH_S="${RIDE_CUTIN_LATE_SWITCH_S:-3.5}"
export RIDE_CUTIN_LATERAL_LOOKAHEAD_M="${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-20}"
export RIDE_CUTIN_PAIRED_AMPLITUDE_MPS="${RIDE_CUTIN_PAIRED_AMPLITUDE_MPS:-0.5}"
export RIDE_BEHAVIOR_ROUTE_PROTOCOL=frozen_corridor
export RIDE_BEHAVIOR_SPEED_PROTOCOL=nominal_cap
: "${RIDE_CHAIN_C_CONTROL_DELAY_S:?Explicitly set command delay (e.g. 0.3 or 0.6 s)}"
: "${RIDE_CHAIN_C_BRAKE_COMMAND_CAP:?Explicitly set command cap (e.g. 0.7)}"
# V16: do not confuse an inactive brake cap with a tested braking limit.
if [[ -n "${RIDE_V16_CAP_REFERENCE_PANEL:-}" && "$RIDE_CHAIN_C_BRAKE_COMMAND_CAP" != "1.0" && "$RIDE_CHAIN_C_BRAKE_COMMAND_CAP" != "1" ]]; then
  if ! bash "$RIDE_HOME/scripts/47_v16_cap_preflight.sh"; then
    if [[ "${RIDE_V16_ALLOW_INACTIVE_CAP_AS_PLACEBO:-0}" != "1" ]]; then
      echo "V16_CAP_NOT_ACTIVE_IN_REFERENCE: select a preregistered effective cap or label this as PLACEBO" >&2
      exit 4
    fi
    echo "V16_INTENTIONAL_INACTIVE_CAP_PLACEBO=1" >&2
  fi
fi
python -m ride.v15_protocol_check --mode panel \
  --families tm mixed_b_tm_c_behavior mixed_b_tm_c_delayed_behavior
bash "$RIDE_HOME/scripts/29_v9_carla_preflight.sh" "$CARLA_TOWN"
if [[ -n "${RIDE_V14_STRESS_BASE_FROZEN:-}" ]]; then
  # Strictly preserve road/seed/actor geometry across all delay/cap cells.
  # Reuse only the ORIGINAL baseline frozen manifest, never its outcomes.
  mkdir -p "$RIDE_OUTPUT_ROOT/frozen"
  python -m ride.v15_stress_frozen \
    --input "$RIDE_V14_STRESS_BASE_FROZEN" \
    --output "$RIDE_OUTPUT_ROOT/frozen/chain_cut_in_anchors.jsonl" \
    --delay-s "$RIDE_CHAIN_C_CONTROL_DELAY_S" \
    --brake-cap "$RIDE_CHAIN_C_BRAKE_COMMAND_CAP"
  eval "$(python -m ride.v14_frozen_replay emit-env --input \
    "$RIDE_OUTPUT_ROOT/frozen/chain_cut_in_anchors.jsonl")"
else
  bash "$RIDE_HOME/scripts/03_precommit_scenes.sh" chain_cut_in
fi
bash "$RIDE_HOME/scripts/03_run_carla_case.sh" chain_cut_in
# Keep execution and scientific validity separate. A requested C actuation
# must be visibly applied; otherwise record the experiment as an inactive control.
_ACTUATION_ARGS=()
if [[ "${RIDE_V16_ALLOW_INACTIVE_CAP_AS_PLACEBO:-0}" == "1" ]]; then
  _ACTUATION_ARGS+=(--allow-inactive-placebo)
fi
python -m ride.v16_cross_stress actuation-gate \
  --panel "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
  --output "$RIDE_OUTPUT_ROOT/carla/v16_C_actuation_gate.json" \
  "${_ACTUATION_ARGS[@]}"
python -m ride.v14_safety_envelope \
  --panel "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
  --output "$RIDE_OUTPUT_ROOT/carla/v14_hypothetical_stop_margin.json" \
  --reaction-s 0.6 --c-brake-mps2 3.5 --b-brake-mps2 5
python -m ride.v14_safety_stress_audit \
  --panel "$RIDE_OUTPUT_ROOT/carla/chain_cut_in.jsonl" \
  --output "$RIDE_OUTPUT_ROOT/carla/v14_physical_stress_audit.json"
printf 'C stress conditions are HYPOTHETICAL, not human-calibrated. Output: %s\n' "$RIDE_OUTPUT_ROOT"
