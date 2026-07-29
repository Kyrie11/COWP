#!/usr/bin/env python3
"""Deterministically cap an OC-RAP dataset without splitting scene-time groups."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

MANIFEST_FIELDS = [
    "path", "scene_id", "original_scenario_id", "time_index", "candidate_index",
    "split_id", "is_nominal", "r_orc_star", "r_dep_star", "oracle_gap_star",
    "i_art_star", "regime_label",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def group_key(row: dict[str, str]) -> tuple[str, str]:
    scene = (row.get("original_scenario_id") or row.get("scene_id") or "").strip()
    return scene, str(row.get("time_index", "")).strip()


def stable_key(key: tuple[str, str], seed: str) -> str:
    return hashlib.sha1(f"{seed}|{key[0]}|{key[1]}".encode("utf-8", errors="replace")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, required=True)
    ap.add_argument("--min-samples", type=int, default=0)
    ap.add_argument("--seed", default="ocrap-safe-cap-v1")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    manifest = args.dataset / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    rows = read_rows(manifest)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    ordered = sorted(groups.items(), key=lambda kv: stable_key(kv[0], args.seed))
    selected_keys: set[tuple[str, str]] = set()
    selected_count = 0
    for key, group in ordered:
        if selected_count + len(group) > args.max_samples:
            continue
        selected_keys.add(key)
        selected_count += len(group)

    if len(rows) <= args.max_samples:
        selected_keys = set(groups)
        selected_count = len(rows)
    if selected_count < args.min_samples:
        raise SystemExit(
            f"selected only {selected_count} samples, below min_samples={args.min_samples}; "
            "increase the raw-scenario scan budget and rebuild"
        )

    selected_rows = [row for row in rows if group_key(row) in selected_keys]
    selected_rows.sort(key=lambda r: (group_key(r), int(r.get("candidate_index", 0) or 0)))
    nominal_per_group: dict[tuple[str, str], int] = defaultdict(int)
    for row in selected_rows:
        nominal_per_group[group_key(row)] += int(str(row.get("is_nominal", "0")) == "1")
    bad_nominal = [key for key, count in nominal_per_group.items() if count != 1]
    if bad_nominal:
        raise SystemExit(f"cannot cap dataset: {len(bad_nominal)} groups do not contain exactly one nominal")

    report = {
        "dataset": str(args.dataset),
        "original_samples": len(rows),
        "selected_samples": selected_count,
        "original_groups": len(groups),
        "selected_groups": len(selected_keys),
        "selected_scenes": len({key[0] for key in selected_keys}),
        "max_samples": args.max_samples,
        "min_samples": args.min_samples,
        "apply": bool(args.apply),
    }
    if not args.apply:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    selected_paths = {str(row.get("path", "")) for row in selected_rows}
    removed = 0
    for row in rows:
        rel = str(row.get("path", ""))
        if rel in selected_paths:
            continue
        path = args.dataset / rel
        if path.exists() or path.is_symlink():
            path.unlink()
            removed += 1
    write_rows(manifest, selected_rows)

    summary_path = args.dataset / "dataset_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    summary.update({
        "num_samples": selected_count,
        "split_counts": {
            split: sum(1 for row in selected_rows if str(row.get("split_id", "")) == split)
            for split in sorted({str(row.get("split_id", "")) for row in selected_rows})
        },
        "manifest": str(manifest),
        "sample_dir": str(args.dataset / "samples"),
        "scene_time_groups": len(selected_keys),
        "unique_raw_scene_ids": len({key[0] for key in selected_keys}),
        "capped_from_samples": len(rows),
        "cap_seed": args.seed,
    })
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    report["removed_files"] = removed
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
