#!/usr/bin/env python3
"""Summarize OC-RAP dataset-build bottlenecks from profiling artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _f(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def summarize(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    stage = _load_json(root / "build_stage_profile.json")
    summary = _load_json(root / "dataset_summary.json")
    scene_rows: list[dict[str, str]] = []
    scene_path = root / "build_scene_time_profile.csv"
    if scene_path.exists():
        with scene_path.open(newline="", encoding="utf-8") as f:
            scene_rows = list(csv.DictReader(f))

    stage_totals = stage.get("stage_totals_s", summary.get("stage_totals_s", {})) or {}
    ranked = sorted(
        ((str(k), _f(v)) for k, v in stage_totals.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    elapsed = _f(stage.get("elapsed_wall_s", summary.get("elapsed_wall_s", 0.0)))
    teacher = sum(_f(r.get("sample_teacher_margins_sum_s", 0.0)) for r in scene_rows)
    future = sum(_f(r.get("sample_future_generation_sum_s", 0.0)) for r in scene_rows)
    write = sum(_f(r.get("npz_write_s", 0.0)) for r in scene_rows)
    scene_times = [_f(r.get("scene_time_total_s", 0.0)) for r in scene_rows if _f(r.get("scene_time_total_s", 0.0)) > 0]
    result = {
        "dataset": str(root),
        "samples": int(summary.get("num_samples", stage.get("num_samples_total", 0)) or 0),
        "new_samples_written": int(summary.get("new_samples_written", stage.get("new_samples_written", 0)) or 0),
        "elapsed_wall_s": elapsed,
        "samples_per_hour": _f(summary.get("samples_per_hour", stage.get("samples_per_hour", 0.0))),
        "top_stage_totals_s": ranked[:8],
        "scene_time_groups_profiled": len(scene_rows),
        "median_scene_time_s": median(scene_times) if scene_times else 0.0,
        "teacher_margin_compute_sum_s": teacher,
        "future_generation_sum_s": future,
        "npz_write_sum_s": write,
        "teacher_to_write_ratio": teacher / max(write, 1.0e-9),
        "teacher_share_of_profiled_compute": teacher / max(teacher + future + write, 1.0e-9),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", help="Dataset output directories")
    parser.add_argument("--output", default="", help="Optional JSON output")
    args = parser.parse_args()
    rows = [summarize(Path(p)) for p in args.datasets]
    text = json.dumps(rows, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
