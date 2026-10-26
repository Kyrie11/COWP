#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
# Clone AUTHOR repositories only. Do not claim that they have been run,
# that their code includes a WOMD closed-loop planner, or that a port is official.
ROOT="$DATASET_ROOT/research_baselines"
mkdir -p "$ROOT"
clone() {
  local name="$1" url="$2"
  if [[ -e "$ROOT/$name" ]]; then
    echo "EXISTS, not overwriting: $ROOT/$name"
  else
    git clone --depth 1 "$url" "$ROOT/$name"
  fi
}
clone GameFormer https://github.com/MCZhi/GameFormer.git
clone RA_CBF https://github.com/rkcosner/learning_responsibility_allocation.git
clone Bosch_SMART_reactive https://github.com/boschresearch/interactive-closed-loop.git
clone ReactSim_Bench https://github.com/Thinklab-SJTU/ReactSim-Bench.git
clone R3R_Julia https://github.com/MarshallVielmetti/r3r_decentralized_gatekeeper.git
printf '%s\n' 'Downloaded source repositories only. Keep independent environments and record commit hashes.'
