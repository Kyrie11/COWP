#!/usr/bin/env bash
# Run in a separate ReactSim environment. Downloads ONLY requested author models.
#
# The ReactSim model repository is a *model* repository, not a dataset
# repository.  Passing --repo-type dataset makes the Hugging Face CLI look for
# a different namespace and reports the rather misleading "Dataset ... not
# found" error even though the public model files exist.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
weight="${1:?Usage: bash scripts/14_download_reactsim_weights.sh smart.ckpt|mtr.pth|ctg.ckpt|vbd.ckpt|clsft.ckpt|trajtok.ckpt}"
case "$weight" in smart.ckpt|mtr.pth|ctg.ckpt|vbd.ckpt|clsft.ckpt|trajtok.ckpt);; *) echo 'Unknown official checkpoint' >&2; exit 2;; esac
out="${RIDE_REACTSIM_CHECKPOINTS:-$DATASET_ROOT/RIDE/research_baselines/ReactSim_Bench/ckpts}"
repo_id="${RIDE_REACTSIM_MODEL_REPO:-zzzy0729/ReactSim-Bench-Models}"
# Pin the public revision used by the current ReactSim release.  `main` is
# still available via RIDE_REACTSIM_MODEL_REVISION=main, but a moving branch
# must not silently change a published evaluation.
revision="${RIDE_REACTSIM_MODEL_REVISION:-db679ec17c5a5423bd3878a3679eb79b77be22f4}"
mkdir -p "$out"
if [[ -e "$out/$weight" ]]; then
  echo "Refusing to overwrite $out/$weight" >&2
  echo 'Remove it explicitly or choose another RIDE_REACTSIM_CHECKPOINTS directory after checking its checksum.' >&2
  exit 3
fi
if ! command -v hf >/dev/null 2>&1; then
  echo 'Install huggingface_hub in the ReactSim env: python -m pip install -U huggingface_hub' >&2
  exit 3
fi
# ReactSim-Bench-Models is hosted as a model repo. The public snapshot has
# root-level files, while the publisher's upload helper uses ckpts/<name>.
# Probe both layouts in a temporary directory and flatten only the requested
# file into the evaluator's checkpoint directory.
echo "Downloading $repo_id/$weight (repo_type=model, revision=$revision) to $out" >&2
tmp="$(mktemp -d "$out/.download.XXXXXX")"
cleanup(){ rm -rf "$tmp"; }
trap cleanup EXIT
hf_root_status=0
hf download "$repo_id" --repo-type model --revision "$revision" \
    --include "$weight" --local-dir "$tmp" || hf_root_status=$?
src=""
for candidate in "$tmp/$weight" "$tmp/ckpts/$weight"; do
  if [[ -s "$candidate" ]]; then src="$candidate"; break; fi
done
if [[ -z "$src" ]]; then
  echo "Root-level $weight unavailable (status=$hf_root_status); trying ckpts/$weight" >&2
  hf download "$repo_id" --repo-type model --revision "$revision" \
    --include "ckpts/$weight" --local-dir "$tmp"
  for candidate in "$tmp/ckpts/$weight" "$tmp/$weight"; do
    if [[ -s "$candidate" ]]; then src="$candidate"; break; fi
  done
fi
if [[ -z "$src" ]]; then
  echo "No nonempty checkpoint found in either root/$weight or ckpts/$weight" >&2
  exit 3
fi
cp -- "$src" "$out/$weight"
test -s "$out/$weight" || { echo 'No nonempty model produced' >&2; exit 3; }

# Hugging Face may leave a Git-LFS/Xet pointer when the large-file transfer is
# unavailable.  Detect it before the checkpoint reaches an evaluator.
if head -c 64 "$out/$weight" | grep -q 'version https://git-lfs.github.com/spec'; then
  echo "Downloaded a Git-LFS pointer instead of model bytes: $out/$weight" >&2
  echo 'Install/upgrade huggingface_hub (and xet), authenticate if required, and retry in a clean output directory.' >&2
  rm -f "$out/$weight"
  exit 3
fi

# Always emit a checksum manifest beside the model.  For the two primary
# models we know the publisher's SHA256 at the pinned revision and fail closed
# on a mismatch.  Other official files are still recorded and can be pinned by
# setting RIDE_REACTSIM_EXPECTED_SHA256 explicitly.
declare -A known_sha256=(
  [smart.ckpt]=773c3899496c9aa060d23bb09c16f67922ce33bb31ae0d755a6ea62905e7005a
  [mtr.pth]=414dd464e9dc801f14197bc890eefd9746e31c7694d6fc52c90dd9f7517c1de2
)
actual_sha256="$(sha256sum "$out/$weight" | awk '{print $1}')"
expected_sha256="${RIDE_REACTSIM_EXPECTED_SHA256:-${known_sha256[$weight]:-}}"
if [[ -n "$expected_sha256" && "$actual_sha256" != "$expected_sha256" ]]; then
  echo "SHA256 mismatch for $weight: expected=$expected_sha256 actual=$actual_sha256" >&2
  rm -f "$out/$weight"
  exit 3
fi
printf '%s  %s\n' "$actual_sha256" "$weight" > "$out/$weight.sha256"
echo "sha256=$actual_sha256"
ls -lh "$out/$weight"
