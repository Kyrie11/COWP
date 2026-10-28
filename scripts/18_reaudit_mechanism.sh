#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
case_name="${1:-cut_in}"
case "$case_name" in convoy|cut_in|chain_cut_in|intersection|support_gap) ;; *) echo "Bad case $case_name" >&2;exit 2;; esac
path="${2:-$RIDE_OUTPUT_ROOT/carla/$case_name.jsonl}"
test -s "$path" || { echo "Missing trial JSONL: $path" >&2;exit 3; }
out="${3:-$(dirname "$path")/${case_name}_mechanism_diagnostics.json}"
python -m ride.carla.mechanism_audit --input "$path" --output "$out"
echo "Exploratory mechanism audit: $out"
