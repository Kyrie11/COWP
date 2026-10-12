#!/usr/bin/env bash
# This file documents runnable order, not an auto-launcher of long jobs.
set -euo pipefail
# Run inside the freshly extracted CLEAN project directory.
# export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
# export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
# export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

if [[ "${1:-help}" == "quick" ]]; then
  bash run.sh sanity
  bash run.sh probe_synthetic
  PROBE_SCENES=80 bash run.sh probe_cache
  python -m research_probe.probe \
    --cache-dir "$COWP_ROOT/tensor_cache_heldout_test" \
    --exclude-ids reference_manifests/development_exact200_ids.txt \
    --max-scenes 80 --max-candidates 10 --max-neighbors 3 \
    --response-accels=-5,-3,0,2 \
    --output outputs/research_probe/alternative_response_bank.json
  bash run.sh analyze_probe
else
  cat <<'EOF'
Run commands individually (no execution until explicit `quick`):
  export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
  export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
  export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"
  bash run.sh sanity
  bash run.sh probe_synthetic
  PROBE_SCENES=80 bash run.sh probe_cache
  python -m research_probe.probe --cache-dir "$COWP_ROOT/tensor_cache_heldout_test" \
    --exclude-ids reference_manifests/development_exact200_ids.txt \
    --max-scenes 80 --max-candidates 10 --max-neighbors 3 \
    --response-accels=-5,-3,0,2 \
    --output outputs/research_probe/alternative_response_bank.json
  bash run.sh analyze_probe
Optional logged replay only:
  bash run.sh logged_conventional
  bash run.sh logged_planner_only
  bash run.sh logged_cowp
Do not run stage0 training again unless you explicitly choose the RCRSO workstream.
EOF
fi
