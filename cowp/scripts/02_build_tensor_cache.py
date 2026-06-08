from __future__ import annotations

import argparse
import os

from cowp.core.config import load_config
from cowp.data.build_cache import build_tensor_cache


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge WOMD tf.Example tensors with COWP proto-derived labels into NPZ tensor cache.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--tfexample-glob", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", help="Skip merged cache files that already exist.")
    ap.add_argument("--verify-cache", action="store_true", help="Re-open newly written NPZ files to catch corrupt writes. Disabled by default for speed.")
    ap.add_argument("--compress", dest="compress", action="store_true", help="Use np.savez_compressed. Smaller but often much slower for WOMD tensor cache.")
    ap.add_argument("--no-compress", dest="compress", action="store_false", help="Use np.savez. This is the default for fast tensor-cache construction.")
    ap.set_defaults(compress=False)
    ap.add_argument("--max-examples-scanned", type=int, default=None, help="Stop after scanning this many tf.Example records; useful for smoke/debug runs.")
    ap.add_argument("--profile-jsonl", default=None, help="Optional per-written-cache timing JSONL.")
    ap.add_argument("--tfexample-index-jsonl", default=None, help="Optional scenario-id index for tf.Example shards. If present, scan only shards containing label ids.")
    ap.add_argument("--build-tfexample-index", action="store_true", help="Build or rebuild --tfexample-index-jsonl before merging. Useful when matched stays at zero.")
    ap.add_argument("--cpu-only", action="store_true", help="Hide CUDA devices before TensorFlow is imported; tensor-cache construction is CPU/I/O-bound.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress display.")
    args = ap.parse_args()
    if args.cpu_only:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    cfg = load_config(args.data_config)
    n = build_tensor_cache(
        args.tfexample_glob or cfg["womd"]["tfexample_glob"],
        args.labels_dir or cfg["outputs"]["labels_dir"],
        args.output_dir or cfg["outputs"]["tensor_cache_dir"],
        limit=args.limit,
        progress=not args.no_progress,
        verify_cache=args.verify_cache,
        skip_existing=args.skip_existing,
        compress=args.compress,
        max_examples_scanned=args.max_examples_scanned,
        profile_jsonl=args.profile_jsonl,
        tfexample_index_jsonl=args.tfexample_index_jsonl,
        build_index_if_missing=args.build_tfexample_index,
    )
    mode = "compressed" if args.compress else "uncompressed"
    print(f"Built {n} merged tensor cache files ({mode})")


if __name__ == "__main__":
    main()
