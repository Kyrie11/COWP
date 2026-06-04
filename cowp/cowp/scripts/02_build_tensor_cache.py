from __future__ import annotations

import argparse

from cowp.core.config import load_config
from cowp.data.build_cache import build_tensor_cache


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge WOMD tf.Example tensors with COWP proto-derived labels into NPZ tensor cache.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--tfexample-glob", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.data_config)
    n = build_tensor_cache(
        args.tfexample_glob or cfg["womd"]["tfexample_glob"],
        args.labels_dir or cfg["outputs"]["labels_dir"],
        args.output_dir or cfg["outputs"]["tensor_cache_dir"],
        limit=args.limit,
    )
    print(f"Built {n} merged tensor cache files")


if __name__ == "__main__":
    main()
