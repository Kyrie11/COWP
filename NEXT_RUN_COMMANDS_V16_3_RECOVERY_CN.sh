#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_3_natural_recovery_v9labels_seed2026}"
export FORCE_TRAIN="${FORCE_TRAIN:-1}"
export FORCE_EVAL="${FORCE_EVAL:-1}"
export NATURAL_AMP=0
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

# The previous run failed before epoch 0 completed, but it already wrote the
# strict code-signature manifest.  A numerical code repair changes that hash.
# Keep the same OUT_ROOT and preserve the old manifest as an audit backup; only
# rotate it when no completed natural history exists.
PROVENANCE="$OUT_ROOT/configs/run_provenance.json"
NATURAL_HISTORY="$OUT_ROOT/checkpoints/natural/history_natural.json"
if [[ -s "$PROVENANCE" && ! -s "$NATURAL_HISTORY" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  mv "$PROVENANCE" "${PROVENANCE}.failed_before_numeric_fix.${stamp}.bak"
fi

exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_3_CN.sh"
