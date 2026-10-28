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
# Small smoke runs can select only one responder family/profile without
# patching the Python code or changing the frozen anchor definitions.
read -r -a RIDE_FAMILY_LIST <<< "${RIDE_FAMILIES:-tm behavior}"
read -r -a RIDE_PROFILE_LIST <<< "${RIDE_PROFILES:-early_slow late_slow mild_early mild_late}"
if (( ${#RIDE_FAMILY_LIST[@]} == 0 || ${#RIDE_PROFILE_LIST[@]} == 0 )); then
  echo 'ERROR: RIDE_FAMILIES and RIDE_PROFILES must not be empty' >&2; exit 2
fi
printf 'RIDE trial: case=%s rpc=%s tm=%s reset=%s families=%s profiles=%s timeout=%s\n' \
  "$CASE" "$CARLA_RPC_PORT" "$CARLA_TM_PORT" "${RIDE_WORLD_RESET:-actors}" \
  "${RIDE_FAMILIES:-tm behavior}" "${RIDE_PROFILES:-early_slow late_slow mild_early mild_late}" \
  "${RIDE_CARLA_TIMEOUT:-120}" >&2
cd "$RIDE_HOME"
python -m ride.carla.ports --mode client --host "$CARLA_HOST" \
  --rpc-port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT"
python -m ride.carla.run \
  --host "$CARLA_HOST" --port "$CARLA_RPC_PORT" --tm-port "$CARLA_TM_PORT" \
  --archetype "$CASE" --town "${CARLA_TOWN:-Town04}" \
  --anchors "${RIDE_ANCHORS:-1}" --repetitions "${RIDE_REPEATS:-1}" \
  --seed "${RIDE_SEED:-17}" --cruise-mps "${RIDE_CRUISE_MPS:-8}" \
  --rear-speed-delta-mps "${RIDE_REAR_SPEED_DELTA_MPS:-0}" \
  --cutin-ego-gap-m "${RIDE_CUTIN_EGO_GAP_M:-20}" \
  --cutin-rear-gap-m "${RIDE_CUTIN_REAR_GAP_M:-45}" \
  --cutin-late-switch-s "${RIDE_CUTIN_LATE_SWITCH_S:-4.5}" \
  --cutin-lateral-lookahead-m "${RIDE_CUTIN_LATERAL_LOOKAHEAD_M:-15}" \
  --warmup-ticks "${RIDE_WARMUP_TICKS:-20}" --no-rendering --anchors-from "$FROZEN" \
  --timeout "${RIDE_CARLA_TIMEOUT:-120}" --world-reset "${RIDE_WORLD_RESET:-actors}" \
  --families "${RIDE_FAMILY_LIST[@]}" \
  --profiles "${RIDE_PROFILE_LIST[@]}" \
  "${EXTRA[@]}" --output "$OUT" \
  2>&1 | tee "$RIDE_OUTPUT_ROOT/carla/${CASE}.log"
python -m ride.evidence --input "$OUT" --output "$RIDE_OUTPUT_ROOT/carla/${CASE}_audit.json"
python -m ride.carla.analyze --jsonl "$OUT" --out "$RIDE_OUTPUT_ROOT/carla/${CASE}_joint_backup_audit.json"
# Explicit separate smoke validity gate. Analysis artifacts are kept even on failure.
if [[ "${RIDE_SMOKE_REQUIRE_VALID:-1}" == 1 ]]; then
  SMOKE_ARGS=()
  if [[ "${RIDE_SMOKE_REQUIRE_COMPLETE:-1}" == 1 ]]; then
    SMOKE_ARGS+=(--require-complete)
  fi
  # A collision or a deliberate off-road candidate is a recorded outcome, not
  # a missing trial.  Require complete traces by default; use REQUIRE_ALL=1
  # for the ordinary (non-bank) scene smoke gate when every trial must also
  # satisfy the validity predicate.
  if [[ "${RIDE_SMOKE_REQUIRE_ALL:-0}" == 1 ]]; then
    SMOKE_ARGS+=(--require-all)
  fi
  if [[ "${RIDE_SMOKE_REQUIRE_MECHANISM:-0}" == 1 ]]; then
    SMOKE_ARGS+=(--require-mechanism)
  fi
  python -m ride.carla.smoke_validate --jsonl "$OUT" "${SMOKE_ARGS[@]}"
fi
