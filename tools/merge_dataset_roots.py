#!/usr/bin/env python3
"""Merge several OC-RAP dataset roots produced by scenario sharding.

Each input root is expected to contain a manifest.csv and samples/*.npz.  The
script copies samples into OUT/samples and rewrites manifest paths so downstream
train/diagnose can use a single dataset root.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

MANIFEST_FIELDS = [
    "path", "scene_id", "original_scenario_id", "time_index", "candidate_index", "split_id",
    "is_nominal", "r_orc_star", "r_dep_star", "oracle_gap_star", "i_art_star", "regime_label",
]


def _unique_target(sample_dir: Path, name: str) -> Path:
    target = sample_dir / name
    if not target.exists():
        return target
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 1
    while True:
        candidate = sample_dir / f"{stem}__dup{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def merge(inputs: list[Path], output: Path, *, copy: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    split_counts: dict[str, int] = {}
    scene_ids: set[str] = set()
    copied = 0
    for root in inputs:
        manifest = root / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"missing manifest: {manifest}")
        with manifest.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = row.get("path", "")
                src = root / rel
                if not src.exists():
                    raise FileNotFoundError(f"manifest sample missing: {src}")
                dst = _unique_target(sample_dir, src.name)
                if copy:
                    shutil.copy2(src, dst)
                else:
                    try:
                        dst.hardlink_to(src)
                    except Exception:
                        shutil.copy2(src, dst)
                out_row = {field: row.get(field, "") for field in MANIFEST_FIELDS}
                out_row["path"] = f"samples/{dst.name}"
                rows.append(out_row)
                copied += 1
                split = str(out_row.get("split_id", ""))
                split_counts[split] = split_counts.get(split, 0) + 1
                if out_row.get("scene_id"):
                    scene_ids.add(str(out_row["scene_id"]))
    manifest_out = output / "manifest.csv"
    with manifest_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "num_samples": len(rows),
        "num_input_roots": len(inputs),
        "input_roots": [str(p) for p in inputs],
        "sample_dir": str(sample_dir),
        "manifest": str(manifest_out),
        "split_counts": split_counts,
        "unique_scene_ids": len(scene_ids),
    }
    (output / "merged_dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--hardlink", action="store_true", help="Try hardlinks before falling back to copy2.")
    args = ap.parse_args()
    print(json.dumps(merge([Path(x).expanduser() for x in args.inputs], Path(args.output).expanduser(), copy=not args.hardlink), indent=2))


if __name__ == "__main__":
    main()
