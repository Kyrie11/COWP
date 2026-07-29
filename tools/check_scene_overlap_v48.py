#!/usr/bin/env python3
"""Audit scene identity overlap across OC-RAP dataset roots.

The tool reads only manifest.csv files, so it is cheap enough to run before every
experiment.  It treats original_scenario_id as authoritative when available.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def scenes(root: Path) -> set[str]:
    manifest = root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    out: set[str] = set()
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scene = (row.get("original_scenario_id") or row.get("scene_id") or "").strip()
            if not scene:
                raise ValueError(f"manifest row in {manifest} lacks scene identity")
            out.add(scene)
    return out


def union(roots: Iterable[Path]) -> set[str]:
    out: set[str] = set()
    for root in roots:
        out.update(scenes(root))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", action="append", type=Path, default=[])
    ap.add_argument("--development-root", action="append", type=Path, default=[])
    ap.add_argument("--test-root", action="append", type=Path, default=[])
    ap.add_argument("--output", type=Path)
    ap.add_argument("--fail-on-train-development-overlap", action="store_true")
    ap.add_argument("--fail-on-development-test-overlap", action="store_true")
    args = ap.parse_args()
    if not args.train_root and not args.development_root and not args.test_root:
        ap.error("provide at least one dataset root")

    train = union(args.train_root)
    dev = union(args.development_root)
    test = union(args.test_root)
    result = {
        "train_roots": [str(p) for p in args.train_root],
        "development_roots": [str(p) for p in args.development_root],
        "test_roots": [str(p) for p in args.test_root],
        "train_scenes": len(train),
        "development_scenes": len(dev),
        "test_scenes": len(test),
        "train_development_overlap": len(train & dev),
        "train_test_overlap": len(train & test),
        "development_test_overlap": len(dev & test),
        "train_development_overlap_examples": sorted(train & dev)[:20],
        "development_test_overlap_examples": sorted(dev & test)[:20],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.fail_on_train_development_overlap and result["train_development_overlap"]:
        return 3
    if args.fail_on_development_test_overlap and result["development_test_overlap"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
