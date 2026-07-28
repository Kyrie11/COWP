from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cowp.core.config import load_config
from cowp.data.build_cache import build_labels_from_proto
from cowp.data.validation import diagnose_dataset


def _read_id_file(path: str | None) -> set[str] | None:
    if not path:
        return None
    ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sid = row.get("scenario_id", row.get("id"))
                if sid is not None:
                    ids.add(str(sid))
                    continue
            except Exception:
                pass
            ids.add(line.split()[0])
    return ids


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
    ap.add_argument("--index-jsonl", default=None, help="Optional Scenario index JSONL used only to display a truthful progress total.")
    ap.add_argument("--allow-scenario-ids", default=None, help="Optional txt/jsonl scenario-id allowlist; other scenarios are filtered before label construction.")
    ap.add_argument("--exclude-scenario-ids", default=None, help="Optional txt/jsonl scenario-id blocklist, used to prevent train/val leakage.")
    ap.add_argument("--start-method", default=None, choices=["fork", "forkserver", "spawn"], help="Multiprocessing start method. Use spawn or forkserver to avoid forking after TensorFlow init.")
    ap.add_argument("--max-pending-multiplier", type=int, default=4, help="Queue this many tasks per worker to hide slow-scenario stragglers.")
    ap.add_argument("--continue-on-error", action="store_true", help="Log worker exceptions to profile JSONL and continue instead of failing immediately.")
    ap.add_argument("--cpu-only", action="store_true", help="Hide CUDA devices before TensorFlow is imported; label construction is CPU-bound.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip label files that already exist in the output directory.")
    ap.add_argument("--skip-diagnostics", action="store_true", help="Only build labels; do not run dataset diagnostics at the end. Use cowp/scripts/06_diagnose_dataset.py separately.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress display.")
    ap.add_argument("--diagnostic-visualizations", action="store_true", help="Write a small gallery of high-signal witness diagnostic plots.")
    ap.add_argument("--max-visualizations", type=int, default=16)
    ap.add_argument("--no-obs-branch", action="store_true")
    ap.add_argument("--no-neutral-branch", action="store_true")
    ap.add_argument("--no-priority-branch", action="store_true")
    ap.add_argument("--no-option-preservation", action="store_true")
    ap.add_argument("--no-dual-edge", action="store_true", help="Record model ablation switch; label geometry is unchanged.")
    ap.add_argument("--no-conflict-query", action="store_true", help="Record model ablation switch; label geometry is unchanged.")
    ap.add_argument("--no-hard-witness-rejection", action="store_true", help="Record planner ablation switch for downstream manifests.")
    ap.add_argument("--soft-burden-cost-only", action="store_true", help="Record planner ablation switch for downstream manifests.")
    args = ap.parse_args()
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    cfg = load_config(args.label_config, args.data_config)
    proto_glob = args.proto_glob or cfg["womd"]["scenario_proto_glob"]
    output_dir = args.output_dir or cfg["outputs"]["labels_dir"]
    ablation = {
        "use_obs_branch": not args.no_obs_branch,
        "use_neutral_branch": not args.no_neutral_branch,
        "use_priority_branch": not args.no_priority_branch,
        "use_option_preservation": not args.no_option_preservation,
        "use_dual_edge": not args.no_dual_edge,
        "use_conflict_query": not args.no_conflict_query,
        "use_hard_witness_rejection": not args.no_hard_witness_rejection,
        "soft_burden_cost_only": bool(args.soft_burden_cost_only),
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
        index_jsonl=args.index_jsonl,
        start_method=args.start_method,
        max_pending_multiplier=args.max_pending_multiplier,
        fail_on_error=not args.continue_on_error,
        allow_scenario_ids=_read_id_file(args.allow_scenario_ids),
        exclude_scenario_ids=_read_id_file(args.exclude_scenario_ids),
    )
    print(f"Built {n} label files in {output_dir}")
    if args.skip_diagnostics:
        print("Skipped diagnostics (--skip-diagnostics). Run 06_diagnose_dataset.py after label generation when needed.")
        return
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
