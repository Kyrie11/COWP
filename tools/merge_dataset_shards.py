#!/usr/bin/env python3
"""Merge OC-RAP dataset shards produced with scenario_stride/scenario_worker_index.

Example:
  python tools/merge_dataset_shards.py \
    --replace-output --output dataset/train_safe \
    dataset/.train_safe_shards/worker0 dataset/.train_safe_shards/worker1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

MANIFEST_FIELDS = [
    "path",
    "scene_id",
    "original_scenario_id",
    "time_index",
    "candidate_index",
    "split_id",
    "is_nominal",
    "r_orc_star",
    "r_dep_star",
    "oracle_gap_star",
    "i_art_star",
    "regime_label",
]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def merge_shards(output: Path, shards: list[Path], *, copy: bool = True) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    merged_rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    split_counts: dict[str, int] = {"train": 0, "val": 0, "calibration": 0, "test": 0}
    shard_summaries: list[dict[str, Any]] = []

    for shard_idx, shard in enumerate(shards):
        shard = shard.expanduser().resolve()
        shard_summaries.append(_load_json(shard / "dataset_summary.json"))
        for row in _read_manifest(shard / "manifest.csv"):
            rel = Path(row.get("path", ""))
            src = shard / rel
            if not src.exists():
                # Older/partial manifests can be stale; skip missing rows instead
                # of creating a broken merged dataset.
                continue
            dst_name = src.name
            if dst_name in seen_names:
                dst_name = f"shard{shard_idx:02d}_{src.name}"
            seen_names.add(dst_name)
            dst = sample_dir / dst_name
            if copy:
                shutil.copy2(src, dst)
            else:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.symlink(src, dst)
            new_row = dict(row)
            new_row["path"] = str(Path("samples") / dst_name)
            merged_rows.append(new_row)
            split = str(new_row.get("split_id", ""))
            split_counts[split] = split_counts.get(split, 0) + 1

    manifest = output / "manifest.csv"
    _write_manifest(manifest, merged_rows)
    summary = {
        "num_samples": int(len(merged_rows)),
        "split_counts": split_counts,
        "manifest": str(manifest),
        "sample_dir": str(sample_dir),
        "merged_from": [str(s.expanduser().resolve()) for s in shards],
        "merge_mode": "copy" if copy else "symlink",
        "shard_summaries": shard_summaries,
    }
    with (output / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
    return summary


def merge_shards_replace(output: Path, shards: list[Path], *, copy: bool = True) -> dict[str, Any]:
    """Build a clean sibling directory, then atomically replace the output.

    Re-merging directly into an existing samples/ directory can leave stale NPZ
    files that are not present in the new manifest.  A temporary complete merge
    followed by a same-filesystem rename prevents that class of contamination.
    """
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}_{int(time.time() * 1e6)}"
    temp = output.parent / f".{output.name}.merge_tmp_{token}"
    backup = output.parent / f".{output.name}.merge_backup_{token}"
    shutil.rmtree(temp, ignore_errors=True)
    summary = merge_shards(temp, shards, copy=copy)
    moved_old = False
    installed_new = False
    try:
        if output.exists() or output.is_symlink():
            os.replace(output, backup)
            moved_old = True
        os.replace(temp, output)
        installed_new = True
        summary["manifest"] = str(output / "manifest.csv")
        summary["sample_dir"] = str(output / "samples")
        with (output / "dataset_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        if moved_old:
            shutil.rmtree(backup, ignore_errors=True)
        return summary
    except Exception:
        if installed_new and output.exists():
            shutil.rmtree(output, ignore_errors=True)
        if moved_old and backup.exists():
            os.replace(backup, output)
        raise
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge OC-RAP dataset shards into one manifest/sample directory.")
    parser.add_argument("shards", nargs="+", help="Shard dataset directories containing manifest.csv and samples/.")
    parser.add_argument("--output", required=True, help="Merged dataset output directory.")
    parser.add_argument("--symlink", action="store_true", help="Symlink samples instead of copying them.")
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Merge into a clean temporary directory and atomically replace the output.",
    )
    args = parser.parse_args()
    fn = merge_shards_replace if args.replace_output else merge_shards
    summary = fn(Path(args.output), [Path(s) for s in args.shards], copy=not args.symlink)
    print(summary)


if __name__ == "__main__":
    main()
