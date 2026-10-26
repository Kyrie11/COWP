#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
# ONLY the author's 28 MB counterfactual AV behavior files, not nuPlan states/maps.
mkdir -p "$DATASET_ROOT/ReactSim-Bench/custom_behaviors"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
root=os.environ['DATASET_ROOT']
dst=os.path.join(root,'ReactSim-Bench','custom_behaviors')
print('Downloading ReactSim AV-only behavior data to',dst,flush=True)
snapshot_download('zzzy0729/ReactSim-Bench-Data',repo_type='dataset',local_dir=dst)
print('Complete. Full evaluation ALSO requires separately acquired nuPlan state/map data.')
PY
