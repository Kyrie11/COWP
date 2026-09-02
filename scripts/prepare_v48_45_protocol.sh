#!/usr/bin/env bash
set -Eeuo pipefail
# Engineering-only bootstrap for the scene-disjoint v48.45 calibration protocol.
# It reads calibration_{near_contact,contact,safe} only. No test_* root is read.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}"
CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
PROTOCOL_SEED="${V4845_PROTOCOL_SEED:-4814}"
ADAPT_TRAIN_FRACTION="${V4845_ADAPT_TRAIN_FRACTION:-0.45}"
ADAPT_DEV_FRACTION="${V4845_ADAPT_DEV_FRACTION:-0.15}"
LINK_MODE="${V4845_PROTOCOL_LINK_MODE:-hardlink}"
export OCRAP_ROOT PROTOCOL_ROOT CAL_NEAR CAL_CONTACT CAL_SAFE

for d in "$CAL_NEAR" "$CAL_CONTACT" "$CAL_SAFE"; do
  [[ -d "$d" ]] || { echo "v48.45 protocol bootstrap: missing calibration dataset: $d" >&2; exit 4; }
  [[ -s "$d/manifest.csv" ]] || { echo "v48.45 protocol bootstrap: missing/empty manifest: $d/manifest.csv" >&2; exit 4; }
done
for d in "$CAL_NEAR" "$CAL_CONTACT" "$CAL_SAFE"; do
  case "$(basename "$d")" in
    test|test_*|*_test) echo "v48.45 protocol bootstrap refuses test dataset input: $d" >&2; exit 4 ;;
  esac
done
[[ "$(basename "$PROTOCOL_ROOT")" == "calibration_v48_14_prism_4814" ]] || {
  echo "v48.45 protocol bootstrap requires canonical protocol leaf calibration_v48_14_prism_4814: $PROTOCOL_ROOT" >&2; exit 4;
}

parent="$(dirname "$PROTOCOL_ROOT")"
mkdir -p "$parent"
lock="${PROTOCOL_ROOT}.prepare.lock"
if ! mkdir "$lock" 2>/dev/null; then
  echo "v48.45 protocol bootstrap lock exists (another preparation may be active): $lock" >&2
  exit 4
fi
backup=""
seal_probe="$parent/.v48_45_protocol_seal_probe.$$.json"
cleanup() {
  rm -f "$seal_probe" 2>/dev/null || true
  rmdir "$lock" 2>/dev/null || true
}
rollback() {
  local rc="$?"
  if [[ "$rc" -ne 0 && -n "$backup" && -d "$backup" ]]; then
    rm -rf "$PROTOCOL_ROOT" 2>/dev/null || true
    mv "$backup" "$PROTOCOL_ROOT" 2>/dev/null || true
    echo "v48.45 protocol bootstrap failed; previous protocol restored from $backup" >&2
  fi
  cleanup
  exit "$rc"
}
trap rollback EXIT

# Reuse only if an independent seal recomputation validates the exact sources,
# seed, fractions, role labels, scene partition, and sample paths.
if [[ -d "$PROTOCOL_ROOT" ]]; then
  set +e
  python tools/check_v48_45_protocol_seal.py \
    --protocol-root "$PROTOCOL_ROOT" \
    --near-source "$CAL_NEAR" --contact-source "$CAL_CONTACT" --safe-root "$CAL_SAFE" \
    --seed "$PROTOCOL_SEED" --adapt-train-fraction "$ADAPT_TRAIN_FRACTION" --adapt-dev-fraction "$ADAPT_DEV_FRACTION" \
    --output "$seal_probe" >/dev/null 2>&1
  probe_rc=$?
  set -e
  if [[ "$probe_rc" == 0 ]]; then
    mv "$seal_probe" "$PROTOCOL_ROOT/V48_45_PROTOCOL_SEAL.json"
    echo "[v48.45 protocol] valid deterministic protocol already exists; reusing $PROTOCOL_ROOT"
    trap - EXIT
    cleanup
    exit 0
  fi
fi

# Invalid/partial existing protocol is not reused. Back it up until the new
# deterministic build has passed both the legacy role audit and the stronger seal.
if [[ -e "$PROTOCOL_ROOT" ]]; then
  backup="$parent/.calibration_v48_14_prism_4814.backup.$$.${RANDOM}"
  mv "$PROTOCOL_ROOT" "$backup"
  echo "[v48.45 protocol] moved invalid/partial protocol aside: $backup"
fi

set +e
python tools/partition_dedicated_calibration_v48_14.py \
  --near "$CAL_NEAR" --contact "$CAL_CONTACT" \
  --output-root "$PROTOCOL_ROOT" \
  --adapt-train-fraction "$ADAPT_TRAIN_FRACTION" \
  --adapt-dev-fraction "$ADAPT_DEV_FRACTION" \
  --seed "$PROTOCOL_SEED" --link-mode "$LINK_MODE"
partition_rc=$?
set -e
if [[ "$partition_rc" != 0 ]]; then
  echo "v48.45 protocol partition failed: rc=$partition_rc" >&2
  exit "$partition_rc"
fi

python tools/audit_dedicated_protocol_v48_16.py \
  --protocol-root "$PROTOCOL_ROOT" \
  --output "$PROTOCOL_ROOT/V48_16_DEDICATED_PROTOCOL_AUDIT.json"

python tools/check_v48_45_protocol_seal.py \
  --protocol-root "$PROTOCOL_ROOT" \
  --near-source "$CAL_NEAR" --contact-source "$CAL_CONTACT" --safe-root "$CAL_SAFE" \
  --seed "$PROTOCOL_SEED" --adapt-train-fraction "$ADAPT_TRAIN_FRACTION" --adapt-dev-fraction "$ADAPT_DEV_FRACTION" \
  --output "$PROTOCOL_ROOT/V48_45_PROTOCOL_SEAL.json"

if [[ -n "$backup" && -d "$backup" ]]; then
  rm -rf "$backup"
  backup=""
fi
printf '[v48.45 protocol] READY: %s\n' "$PROTOCOL_ROOT"

trap - EXIT
cleanup
exit 0
