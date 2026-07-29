#!/usr/bin/env python3
"""Preflight the OC-RAP offline-root -> WOMD/Waymax closed-loop contract.

This checker deliberately does not claim that every target is present in the raw
TFRecords; proving that requires scanning the shards.  It verifies the necessary
conditions that the current evaluator consumes: non-empty offline samples,
scene_id/time_index metadata, unique scene-time targets, and resolvable raw WOMD
paths.  The actual closed-loop runner then records matched/missing targets.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path


def _strip_limit(spec: str) -> str:
    text = str(spec or "").strip()
    if "@" in text:
        base, suffix = text.rsplit("@", 1)
        if suffix.isdigit():
            return base
    return text


def _resolve_raw_specs(spec: str) -> list[str]:
    found: list[str] = []
    for part in (x.strip() for x in str(spec or "").split(",")):
        if not part:
            continue
        part = _strip_limit(part)
        matches = sorted(glob.glob(part)) if any(c in part for c in "*?[") else ([part] if Path(part).exists() else [])
        found.extend(matches)
    return sorted(dict.fromkeys(found))


def _root_label(path: Path) -> str:
    for parent in [path.parent, *path.parents]:
        low = parent.name.lower()
        if "near" in low:
            return "near_contact"
        if "contact" in low:
            return "contact"
        if "safe" in low:
            return "safe"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="Check whether an OC-RAP bucket can drive Waymax closed-loop replay.")
    ap.add_argument("--dataset", required=True, help="OC-RAP offline dataset root/spec containing .npz samples")
    ap.add_argument("--womd-pattern", required=True, help="Raw WOMD TFRecord path/glob used by closed-loop")
    ap.add_argument("--split", default="", help="Optional split_id filter, e.g. val or test")
    ap.add_argument("--max-scan", type=int, default=0, help="Limit metadata scan; 0 means all samples")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    paths = list(iter_sample_paths_many(args.dataset))
    if args.max_scan > 0:
        paths = paths[: args.max_scan]
    split_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    missing_scene = 0
    missing_time = 0
    valid_rows = 0
    unique_targets: set[tuple[str, int]] = set()
    examples: list[dict[str, Any]] = []

    for p in paths:
        split = str(scalar_metadata_for_path(p, "split_id", ""))
        split_counts[split or "<missing>"] += 1
        if args.split and split != args.split:
            continue
        scene = str(scalar_metadata_for_path(p, "scene_id", "") or "").strip()
        if not scene:
            missing_scene += 1
            continue
        raw_time = scalar_metadata_for_path(p, "time_index", None)
        try:
            if raw_time is None:
                raise ValueError
            time_index = int(float(raw_time))
        except Exception:
            missing_time += 1
            continue
        valid_rows += 1
        bucket = _root_label(Path(p))
        bucket_counts[bucket] += 1
        unique_targets.add((scene, time_index))
        if len(examples) < 5:
            examples.append({"scene_id": scene, "time_index": time_index, "bucket": bucket, "sample": str(p)})

    raw_files = _resolve_raw_specs(args.womd_pattern)
    report = {
        "dataset": args.dataset,
        "womd_pattern": args.womd_pattern,
        "split_filter": args.split,
        "num_samples_scanned": len(paths),
        "num_valid_sample_rows": valid_rows,
        "num_unique_scene_time_targets": len(unique_targets),
        "missing_scene_id_rows": missing_scene,
        "missing_time_index_rows": missing_time,
        "split_counts": dict(sorted(split_counts.items())),
        "bucket_counts_after_split": dict(sorted(bucket_counts.items())),
        "resolved_womd_files": raw_files[:20],
        "num_resolved_womd_files": len(raw_files),
        "examples": examples,
        "schema_supports_closed_loop": bool(valid_rows > 0 and unique_targets and raw_files),
        "limitations": [
            "This preflight verifies the metadata/path contract only.",
            "The closed-loop runner must still match scene_id values while scanning the specified WOMD shards.",
            "Route metrics require raw v1.3.1 path_samples/sdc_paths fields and closed_loop.use_sdc_paths=true; route-free dynamics metrics do not.",
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["schema_supports_closed_loop"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
