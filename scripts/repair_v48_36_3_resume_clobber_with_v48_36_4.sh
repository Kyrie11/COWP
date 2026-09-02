#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_36_1_ocaf_cuda_hotfix_48361}"
mkdir -p "$OUTPUTDIR/logs"

python tools/check_v48_36_reentry_contract.py \
  --run "$OUTPUTDIR" --mode resume \
  --output "$OUTPUTDIR/V48_36_REENTRY_CONTRACT.json"
action="$(python - "$OUTPUTDIR/V48_36_REENTRY_CONTRACT.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('action','refuse'))
PY
)"
case "$action" in
  return_existing_terminal)
    : # already repaired; idempotent success
    ;;
  restore_archived_terminal)
    python tools/restore_v48_36_terminal_state_after_refused_resume.py \
      --run "$OUTPUTDIR" --repo "$REPO" \
      --output "$OUTPUTDIR/V48_36_4_REENTRY_RESTORE.json"
    ;;
  *)
    echo "v48.36.4 resume-clobber repair refused: action=$action" >&2
    exit 30
    ;;
esac

python - "$OUTPUTDIR" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
state=json.loads((root/'AUTHORITATIVE_RUN_STATUS.json').read_text())
complete=json.loads((root/'V48_36_COMPLETE.json').read_text())
rc=int(state.get('authoritative_exit_code',-1))
assert state.get('valid') is True and rc in (0,20), state
assert complete.get('pipeline_valid') is True and int(complete.get('pipeline_exit_code',-1)) == rc, complete
print(f'v48.36.4 terminal-state management PASS: authoritative RC={rc}; active state preserved/restored')
raise SystemExit(rc)
PY
