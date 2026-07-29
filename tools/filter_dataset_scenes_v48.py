#!/usr/bin/env python3
"""Create a linked OC-RAP root after excluding scenes found in other roots."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def scene_id(row: dict[str, str]) -> str:
    value = (row.get("original_scenario_id") or row.get("scene_id") or "").strip()
    if not value:
        raise ValueError("manifest row missing original_scenario_id/scene_id")
    return value


def read_rows(root: Path) -> tuple[list[dict[str, str]], list[str]]:
    manifest = root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            mode = "symlink"
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            mode = "copy"
    shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--exclude-root", action="append", type=Path, default=[])
    ap.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows, fields = read_rows(args.input)
    excluded: set[str] = set()
    for root in args.exclude_root:
        other, _ = read_rows(root)
        excluded.update(scene_id(row) for row in other)

    kept = [row for row in rows if scene_id(row) not in excluded]
    if not kept:
        raise ValueError("all scenes were excluded")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        shutil.rmtree(args.output)
    (args.output / "samples").mkdir(parents=True)

    out_rows: list[dict[str, str]] = []
    for row in kept:
        raw = Path(row.get("path", ""))
        src = raw if raw.is_absolute() else args.input / raw
        if not src.exists():
            alt = args.input / "samples" / raw.name
            if not alt.exists():
                raise FileNotFoundError(src)
            src = alt
        dst = args.output / "samples" / src.name
        link(src, dst, args.link_mode)
        new = dict(row)
        new["path"] = str(Path("samples") / dst.name)
        out_rows.append(new)

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(out_rows)
    summary = {
        "source": str(args.input.resolve()),
        "exclude_roots": [str(p.resolve()) for p in args.exclude_root],
        "input_samples": len(rows),
        "output_samples": len(out_rows),
        "input_scenes": len({scene_id(r) for r in rows}),
        "output_scenes": len({scene_id(r) for r in out_rows}),
        "excluded_scene_count": len({scene_id(r) for r in rows} & excluded),
    }
    (args.output / "scene_filter_provenance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
