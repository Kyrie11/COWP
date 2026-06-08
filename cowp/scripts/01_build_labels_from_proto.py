from __future__ import annotations

import argparse
import json
import os
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
    ap.add_argument("--limit", type=int, default=None, help="Stop after this many written label files.")
    ap.add_argument("--max-scenarios-scanned", type=int, default=None, help="Stop after scanning this many Scenario records, even if many are filtered.")
    ap.add_argument("--num-workers", type=int, default=1, help="Parallel CPU workers for label generation. Use 4-16 depending on CPU/RAM.")
    ap.add_argument("--no-compress", action="store_true", help="Use np.savez instead of np.savez_compressed for faster label writes.")
    ap.add_argument("--profile-jsonl", default=None, help="Optional per-scenario build timing JSONL.")
    ap.add_argument("--cpu-only", action="store_true", help="Hide CUDA devices before TensorFlow is imported; label construction is CPU-bound.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip label files that already exist in the output directory.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress display.")
    ap.add_argument("--diagnostic-visualizations", action="store_true", help="Write a small gallery of high-signal witness diagnostic plots.")
    ap.add_argument("--max-visualizations", type=int, default=16)
    ap.add_argument("--no-obs-branch", action="store_true")
    ap.add_argument("--no-neutral-branch", action="store_true")
    ap.add_argument("--no-priority-branch", action="store_true")
    ap.add_argument("--no-option-preservation", action="store_true")
    args = ap.parse_args()
    if args.cpu_only:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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
        num_workers=args.num_workers,
        max_scenarios_scanned=args.max_scenarios_scanned,
        compress=not args.no_compress,
        profile_jsonl=args.profile_jsonl,
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
    summary_path = Path(diag) / "dataset_diagnostics_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        keys = [
            "num_scenes",
            "quality_assessment",
            "positive_pair_ratio",
            "false_safe_candidate_ratio",
            "ncf_candidate_ratio",
            "stress_eligible_scene_ratio",
            "response_safe_pair_ratio",
            "mean_natural_alternatives",
            "validation_error_files",
        ]
        compact = {k: summary.get(k) for k in keys if k in summary}
        print("Dataset diagnostics summary:")
        print(json.dumps(compact, indent=2, ensure_ascii=False))
    print(f"Wrote diagnostics to {diag}")


if __name__ == "__main__":
    main()
