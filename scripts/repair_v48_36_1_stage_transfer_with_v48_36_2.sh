#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Repair the exact v48.36.1 false RC=30 stage-transfer failure without retraining,
then resume calibration/certificate with the byte-identical checkpoints.

Required/typical environment variables:
  OUTPUTDIR       Existing v48.36.1 run directory.
  SOURCE_RUN      Original v48.13 source run.
  PROTOCOL_ROOT   Dedicated adaptation/calibration protocol root.
  OCRAP_ROOT      Dataset root used by the controller.
  CAL_SAFE        Safe calibration root.
  GPU0, GPU1      Training/certificate GPU IDs.
  REPAIR_ONLY=1   Stop after repair authorization; do not launch the controller.

No positional arguments are accepted.
EOF
}

if [[ $# -gt 0 ]]; then
  case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    *) echo "unexpected positional argument: $1" >&2; usage >&2; exit 2 ;;
  esac
fi

OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_36_1_ocaf_cuda_hotfix_48361}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

python tools/repair_v48_36_1_stage_transfer_failure.py \
  --run "$OUTPUTDIR" \
  --source-run "$SOURCE_RUN" \
  --protocol-root "$PROTOCOL_ROOT" \
  --repo "$REPO" \
  --output "$OUTPUTDIR/V48_36_2_STAGE_TRANSFER_REPAIR.json"

python - "$OUTPUTDIR/V48_36_2_STAGE_TRANSFER_REPAIR.json" <<'PY_REPAIR_VALID'
import json, pathlib, sys
path=pathlib.Path(sys.argv[1]); doc=json.loads(path.read_text())
if doc.get('valid') is not True or doc.get('algorithm_changed') is not False or doc.get('retraining_performed') is not False:
    raise SystemExit(f'repair contract invalid: {path}')
print('stage-transfer repair authorized; no retraining was performed')
PY_REPAIR_VALID

if [[ "${REPAIR_ONLY:-0}" == 1 ]]; then
  exit 0
fi

set +e
OUTPUTDIR="$OUTPUTDIR" \
OCRAP_ROOT="$OCRAP_ROOT" \
SOURCE_RUN="$SOURCE_RUN" \
PROTOCOL_ROOT="$PROTOCOL_ROOT" \
CAL_SAFE="$CAL_SAFE" \
GPU0="$GPU0" GPU1="$GPU1" \
RESUME_AFTER_ADAPTATION=1 \
REBUILD_ADAPT_INDEX=0 \
REBUILD_ADAPT_DEV_INDEX=0 \
OCRAP_IMPLEMENTATION_VERSION=v48.36.2-STAGE-TRANSFER-HOTFIX \
bash scripts/run_v48_36_ocaf_dedicated.sh \
  >"$OUTPUTDIR/controller.v48.36.2-resume.stdout.log" 2>&1
rc=$?
set -e
printf 'v48.36.2 resumed controller RC=%s\n' "$rc"
exit "$rc"
