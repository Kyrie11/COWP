#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from cache_io import (
    add_key_override_arguments,
    discover_npz_files,
    infer_scene_record,
    load_npz,
    parse_key_overrides,
)


ID_KEYS = {
    "scenario_id", "scenarioid", "scene_id", "sceneid",
    "id", "scenario/id", "scenario/id_bytes",
}


def find_id(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            norm = re.sub(r"[^a-z0-9/]+", "", str(key).lower())
            if norm in ID_KEYS and not isinstance(value, (dict, list)):
                return str(value)
        for value in obj.values():
            out = find_id(value)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for value in obj:
            out = find_id(value)
            if out is not None:
                return out
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Select paired hard/random proposal-probe scenes.")
    ap.add_argument("--old-cache", required=True, type=Path)
    ap.add_argument("--index-jsonl", required=True, type=Path)
    ap.add_argument("--hard-count", type=int, default=400)
    ap.add_argument("--random-count", type=int, default=800)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output-index-jsonl", required=True, type=Path)
    ap.add_argument("--output-manifest", required=True, type=Path)
    add_key_override_arguments(ap)
    args = ap.parse_args()

    overrides = parse_key_overrides(args)
    hard_ids = []
    all_ids = []
    errors = []
    for path in discover_npz_files(args.old_cache):
        try:
            rec = infer_scene_record(path, load_npz(path), overrides)
            all_ids.append(rec.scenario_id)
            if rec.any_conventional_safe() and not rec.any_ncf():
                hard_ids.append(rec.scenario_id)
        except Exception as exc:
            errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})

    if errors:
        raise SystemExit(
            "cache parsing failed; run probe_cache_schema.py and provide explicit key overrides. "
            f"first error: {errors[0]}"
        )

    rng = random.Random(args.seed)
    if len(hard_ids) < args.hard_count:
        raise SystemExit(f"only {len(hard_ids)} hard scenes available, need {args.hard_count}")
    selected_hard = rng.sample(sorted(set(hard_ids)), args.hard_count)
    remaining = sorted(set(all_ids) - set(selected_hard))
    if len(remaining) < args.random_count:
        raise SystemExit(f"only {len(remaining)} non-hard-selected scenes remain")
    selected_random = rng.sample(remaining, args.random_count)
    wanted = set(selected_hard) | set(selected_random)

    selected_lines: dict[str, str] = {}
    with args.index_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = find_id(obj)
            if sid in wanted:
                if sid in selected_lines:
                    raise SystemExit(f"duplicate scenario id {sid!r} in index at line {line_no}")
                selected_lines[sid] = line.rstrip("\n")

    missing = sorted(wanted - set(selected_lines))
    if missing:
        raise SystemExit(
            f"{len(missing)} requested IDs are missing from index; examples={missing[:10]}"
        )

    args.output_index_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_index_jsonl.open("w", encoding="utf-8") as f:
        for sid in selected_hard + selected_random:
            f.write(selected_lines[sid] + "\n")

    manifest = {
        "version": "v16.8.4_three_arm_paired_probe",
        "seed": args.seed,
        "old_cache": str(args.old_cache),
        "index_jsonl": str(args.index_jsonl),
        "hard_definition": "old_any_conventional_safe AND NOT old_any_ncf",
        "hard_ids": selected_hard,
        "random_ids": selected_random,
        "all_ids": selected_hard + selected_random,
        "counts": {
            "hard": len(selected_hard),
            "random": len(selected_random),
            "total": len(wanted),
            "old_hard_available": len(set(hard_ids)),
            "old_scenes_available": len(set(all_ids)),
        },
        "note": (
            "Overall proposal-rate estimates must use random_ids only. hard_ids are an "
            "enriched diagnostic stratum and must not be pooled as an unbiased validation sample."
        ),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
