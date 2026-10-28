#!/usr/bin/env bash
# Run in a separate ReactSim environment. Downloads ONLY requested author models.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
weight="${1:?Usage: bash scripts/14_download_reactsim_weights.sh smart.ckpt|mtr.pth|ctg.ckpt|vbd.ckpt|clsft.ckpt|trajtok.ckpt}"
case "$weight" in smart.ckpt|mtr.pth|ctg.ckpt|vbd.ckpt|clsft.ckpt|trajtok.ckpt);; *) echo 'Unknown official checkpoint' >&2; exit 2;; esac
out="${RIDE_REACTSIM_CHECKPOINTS:-$DATASET_ROOT/RIDE/research_baselines/ReactSim_Bench/ckpts}"
mkdir -p "$out"
if [[ -e "$out/$weight" ]]; then
  echo "Refusing to overwrite $out/$weight" >&2;exit 3
fi
if ! command -v hf >/dev/null 2>&1; then
  echo 'Install huggingface_hub in the ReactSim env: python -m pip install -U huggingface_hub' >&2
  exit 3
fi
hf download zzzy0729/ReactSim-Bench-Models --repo-type dataset --include "$weight" --local-dir "$out"
test -s "$out/$weight" || { echo 'No nonempty model produced' >&2; exit 3; }
ls -lh "$out/$weight"
