#!/usr/bin/env bash
set -Eeuo pipefail
# Engineering-only fast rerun for the exact v48.42 HPFR factor checkpoints.
# It NEVER mutates/reuses the failed OUTPUTDIR as the new run. Instead it asks
# the audited factor-cache contract to byte-verify both old factor stages, then
# materializes a fresh v48.42.1 run and reruns calibration/certificate.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

OLD_OUTPUTDIR="${OLD_OUTPUTDIR:?OLD_OUTPUTDIR must point to the failed v48.42 main run}"
OUTPUTDIR="${OUTPUTDIR:?OUTPUTDIR must be a NEW run directory}"
OLD_OUTPUTDIR="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$OLD_OUTPUTDIR")"
OUTPUTDIR="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$OUTPUTDIR")"
if [[ "$OLD_OUTPUTDIR" == "$OUTPUTDIR" ]]; then
  echo "refusing in-place rerun: OUTPUTDIR must differ from OLD_OUTPUTDIR" >&2
  exit 30
fi
if [[ -e "$OUTPUTDIR" && -n "$(find "$OUTPUTDIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing non-empty OUTPUTDIR: $OUTPUTDIR" >&2
  exit 30
fi
for variant in balanced precision; do
  stage="$OLD_OUTPUTDIR/candidates/$variant/factor_stage"
  [[ -f "$stage/FACTOR_CACHE_CONTRACT.json" ]] || { echo "missing exact factor-cache contract: $stage" >&2; exit 30; }
  [[ -f "$stage/model_v48_trac_sr/best.pt" ]] || { echo "missing factor checkpoint: $stage/model_v48_trac_sr/best.pt" >&2; exit 30; }
  [[ -f "$stage/model_v48_trac_sr/train_summary.json" ]] || { echo "missing factor train summary: $stage" >&2; exit 30; }
done

# Recover source/protocol identity from the failed controller completion unless
# the operator explicitly overrides them.
readarray -t RECORDED < <(python - "$OLD_OUTPUTDIR/V48_36_COMPLETE.json" "$OLD_OUTPUTDIR" <<'PY'
import json,pathlib,sys
complete=pathlib.Path(sys.argv[1]); old=pathlib.Path(sys.argv[2])
d=json.loads(complete.read_text(encoding='utf-8'))
old_repo=old.parent.parent
source=pathlib.Path(str(d.get('source_run') or ''))
if not source.is_absolute(): source=(old_repo/source).resolve()
protocol=pathlib.Path(str(d.get('protocol_root') or '')).expanduser().resolve()
print(source)
print(protocol)
PY
)
export SOURCE_RUN="${SOURCE_RUN:-${RECORDED[0]}}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-${RECORDED[1]}}"
export OCRAP_ROOT="${OCRAP_ROOT:-$(dirname -- "$PROTOCOL_ROOT")}" 
export OLD_OUTPUTDIR OUTPUTDIR
export V4842_ALLOW_EXACT_FACTOR_CACHE=1
export V4836_FACTOR_CACHE_BALANCED="$OLD_OUTPUTDIR/candidates/balanced/factor_stage"
export V4836_FACTOR_CACHE_PRECISION="$OLD_OUTPUTDIR/candidates/precision/factor_stage"

# The v48.42.1 main wrapper force-pins all algorithm flags. Cache reuse is
# accepted only if manage_v48_32_factor_cache.py proves the complete semantic
# contract and checkpoint/index hashes are byte-identical.
exec bash scripts/run_v48_42_hpfr_dedicated.sh "$@"
