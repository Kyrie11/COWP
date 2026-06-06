from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cowp.core.config import load_config
from cowp.data.build_cache import build_labels_from_proto
from cowp.data.validation import diagnose_dataset
from cowp.label.stress_set import write_stress_manifest


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Quick COWP dataset paper-check: build a bounded label sample, diagnostics, stress manifest, timing profile, and visualizations."
    )
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--proto-glob", default=None)
    ap.add_argument("--output-dir", default="outputs/cowp_papercheck")
    ap.add_argument("--limit-labels", type=int, default=64, help="Target number of written label files.")
    ap.add_argument("--max-scenarios-scanned", type=int, default=2000, help="Hard cap on scanned Scenario records.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-visualizations", type=int, default=64)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-compress", action="store_true", help="Faster writes for temporary papercheck outputs.")
    ap.add_argument("--cpu-only", action="store_true", help="Hide CUDA devices before TensorFlow is imported; this pipeline is CPU/I/O-bound.")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    if args.cpu_only:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    cfg = load_config(args.label_config, args.data_config)
    proto_glob = args.proto_glob or cfg["womd"]["scenario_proto_glob"]
    root = Path(args.output_dir)
    labels_dir = root / "labels"
    diagnostics_dir = root / "diagnostics"
    stress_path = root / "stress_set" / "cowp_stress_manifest.jsonl"
    profile_path = root / "label_build_profile.jsonl"
    root.mkdir(parents=True, exist_ok=True)

    n_labels = build_labels_from_proto(
        proto_glob,
        labels_dir,
        cfg,
        limit=args.limit_labels,
        max_scenarios_scanned=args.max_scenarios_scanned,
        num_workers=args.num_workers,
        progress=not args.no_progress,
        skip_existing=args.skip_existing,
        compress=not args.no_compress,
        profile_jsonl=profile_path,
    )
    df = diagnose_dataset(
        labels_dir,
        cfg,
        diagnostics_dir,
        progress=not args.no_progress,
        make_visualizations=True,
        max_visualizations=args.max_visualizations,
    )
    n_stress = write_stress_manifest(labels_dir, stress_path, progress=not args.no_progress)

    summary_path = diagnostics_dir / "dataset_diagnostics_summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = {
        "labels_written": int(n_labels),
        "diagnostic_rows": int(len(df)),
        "stress_rows": int(n_stress),
        "labels_dir": str(labels_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "stress_manifest": str(stress_path),
        "profile_jsonl": str(profile_path),
        "summary": summary,
    }
    out = root / "papercheck_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
