#!/usr/bin/env python3
"""Create scene-disjoint calibration/development roots without modifying input.

Files are linked (hardlink by default, symlink/copy fallback), so this is cheap and
keeps the user's original val_* path intact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def _score(scene: str, seed: int) -> float:
    h = hashlib.sha1(f"{seed}|{scene}".encode("utf-8", errors="replace")).digest()
    return int.from_bytes(h[:8], "big") / float(2**64)


def _link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
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
    ap.add_argument("--calibration-output", type=Path, required=True)
    ap.add_argument("--validation-output", type=Path, required=True)
    ap.add_argument("--calibration-fraction", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=4801)
    ap.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest = args.input / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys()) if rows else []
    if not rows:
        raise ValueError("empty manifest")

    by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scene = (row.get("original_scenario_id") or row.get("scene_id") or "").strip()
        if not scene:
            raise ValueError("manifest row missing scene_id/original_scenario_id")
        by_scene[scene].append(row)

    cal_scenes = {s for s in by_scene if _score(s, args.seed) < args.calibration_fraction}
    val_scenes = set(by_scene) - cal_scenes
    if not cal_scenes or not val_scenes:
        raise ValueError("split produced an empty partition; change seed/fraction")

    for out in (args.calibration_output, args.validation_output):
        if out.exists():
            if not args.overwrite:
                raise FileExistsError(f"output exists: {out}; pass --overwrite")
            shutil.rmtree(out)
        (out / "samples").mkdir(parents=True)

    def write_partition(out: Path, scenes: set[str], split_id: str) -> int:
        out_rows: list[dict[str, str]] = []
        for scene in sorted(scenes):
            for row in by_scene[scene]:
                raw = Path(row.get("path", ""))
                src = raw if raw.is_absolute() else args.input / raw
                if not src.exists():
                    # Most manifests use samples/foo.npz relative to root.
                    alt = args.input / "samples" / raw.name
                    if alt.exists():
                        src = alt
                    else:
                        raise FileNotFoundError(src)
                dst = out / "samples" / src.name
                _link(src, dst, args.link_mode)
                new = dict(row)
                new["path"] = str(Path("samples") / dst.name)
                new["split_id"] = split_id
                out_rows.append(new)
        fieldnames = list(fields)
        if "split_id" not in fieldnames:
            fieldnames.append("split_id")
        with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(out_rows)
        (out / "split_provenance.json").write_text(json.dumps({
            "source": str(args.input.resolve()),
            "seed": args.seed,
            "calibration_fraction": args.calibration_fraction,
            "split_id": split_id,
            "num_scenes": len(scenes),
            "num_samples": len(out_rows),
            "scene_disjoint": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(out_rows)

    n_cal = write_partition(args.calibration_output, cal_scenes, "calibration")
    n_val = write_partition(args.validation_output, val_scenes, "val")
    print(json.dumps({
        "event": "scene_disjoint_calibration_split_complete",
        "input": str(args.input),
        "calibration_output": str(args.calibration_output),
        "validation_output": str(args.validation_output),
        "calibration_scenes": len(cal_scenes),
        "validation_scenes": len(val_scenes),
        "calibration_samples": n_cal,
        "validation_samples": n_val,
        "scene_overlap": 0,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
