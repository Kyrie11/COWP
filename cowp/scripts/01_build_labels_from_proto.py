from __future__ import annotations

import argparse
from pathlib import Path

from cowp.core.config import load_config
from cowp.data.build_cache import build_labels_from_proto
from cowp.data.validation import diagnose_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Build COWP counterfactual labels from WOMD Scenario protos.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--proto-glob", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--diagnostics-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", help="Skip label files that already exist in the output directory.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress display.")
    ap.add_argument("--diagnostic-visualizations", action="store_true", help="Write a small gallery of high-signal witness diagnostic plots.")
    ap.add_argument("--max-visualizations", type=int, default=16)
    ap.add_argument("--no-obs-branch", action="store_true")
    ap.add_argument("--no-neutral-branch", action="store_true")
    ap.add_argument("--no-priority-branch", action="store_true")
    ap.add_argument("--no-option-preservation", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.label_config, args.data_config)
    proto_glob = args.proto_glob or cfg["womd"]["scenario_proto_glob"]
    output_dir = args.output_dir or cfg["outputs"]["labels_dir"]
    ablation = {
        "use_obs_branch": not args.no_obs_branch,
        "use_neutral_branch": not args.no_neutral_branch,
        "use_priority_branch": not args.no_priority_branch,
        "use_option_preservation": not args.no_option_preservation,
    }
    n = build_labels_from_proto(
        proto_glob,
        output_dir,
        cfg,
        limit=args.limit,
        ablation=ablation,
        progress=not args.no_progress,
        skip_existing=args.skip_existing,
    )
    print(f"Built {n} label files in {output_dir}")
    diag = args.diagnostics_dir or cfg["outputs"]["diagnostics_dir"]
    diagnose_dataset(
        output_dir,
        cfg,
        diag,
        progress=not args.no_progress,
        make_visualizations=args.diagnostic_visualizations,
        max_visualizations=args.max_visualizations,
    )
    print(f"Wrote diagnostics to {diag}")


if __name__ == "__main__":
    main()
