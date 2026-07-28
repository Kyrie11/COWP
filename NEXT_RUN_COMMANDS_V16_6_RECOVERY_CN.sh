#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_6_natural_recovery_v9labels_seed2026}"
export FORCE_TRAIN="${FORCE_TRAIN:-1}"
export FORCE_EVAL="${FORCE_EVAL:-1}"
export NATURAL_AMP="${NATURAL_AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-auto}"
export NATURAL_EPOCHS="${NATURAL_EPOCHS:-20}"
export STOP_AFTER_STAGE=natural
export RUN_DIAGNOSE="${RUN_DIAGNOSE:-1}"
export RUN_NATURAL=1
export RUN_TRANSPORT=0
export RUN_PLANNER=0
export RUN_OFFLINE=0
export RUN_PROBE=0
export RUN_FULL=0
export ALLOW_QUALITY_GATE_FAILURE=0
export BACKGROUND="${BACKGROUND:-1}"

# A fresh v16.6 output root is required.  If a prior v16.6 launch was interrupted
# before writing any natural history, preserve its provenance manifest as an audit
# backup so the repaired/relaunched run can recreate a consistent manifest.
PROVENANCE="$OUT_ROOT/configs/run_provenance.json"
NATURAL_HISTORY="$OUT_ROOT/checkpoints/natural/history_natural.json"
if [[ -s "$PROVENANCE" && ! -s "$NATURAL_HISTORY" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  mv "$PROVENANCE" "${PROVENANCE}.interrupted_before_natural_history.${stamp}.bak"
fi

exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_6_CN.sh"
