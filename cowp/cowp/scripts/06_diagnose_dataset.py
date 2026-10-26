from __future__ import annotations

import argparse

from cowp.core.config import load_config
from cowp.data.validation import diagnose_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Run COWP per-scene and dataset-level diagnostics.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--make-visualizations", action="store_true", help="Write a small gallery of high-signal witness plots.")
    ap.add_argument("--max-visualizations", type=int, default=16)
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress display.")
    args = ap.parse_args()
    cfg = load_config(args.label_config, args.data_config)
    df = diagnose_dataset(
        args.labels_dir or cfg["outputs"]["labels_dir"],
        cfg,
        args.output_dir or cfg["outputs"]["diagnostics_dir"],
        progress=not args.no_progress,
        make_visualizations=args.make_visualizations,
        max_visualizations=args.max_visualizations,
    )
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()
