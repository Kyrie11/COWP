#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from cache_io import (
    CacheSchemaError,
    add_key_override_arguments,
    discover_npz_files,
    infer_scene_record,
    load_npz,
    parse_key_overrides,
)


def describe_array(arr: np.ndarray) -> dict:
    out = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.size and np.issubdtype(arr.dtype, np.number):
        finite = np.asarray(arr)[np.isfinite(arr)]
        if finite.size:
            out.update(
                min=float(np.min(finite)),
                max=float(np.max(finite)),
                mean=float(np.mean(finite)),
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect COWP cache schema safely.")
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output", type=Path, required=True)
    add_key_override_arguments(ap)
    args = ap.parse_args()

    files = discover_npz_files(args.cache_dir)
    if not files:
        raise SystemExit(f"no .npz/.npz.gz files found under {args.cache_dir}")
    rng = random.Random(args.seed)
    chosen = files if len(files) <= args.sample else rng.sample(files, args.sample)

    key_presence: Counter[str] = Counter()
    key_shapes: dict[str, Counter[str]] = {}
    examples: list[dict] = []
    resolved: Counter[tuple] = Counter()
    errors: list[dict] = []
    overrides = parse_key_overrides(args)

    for path in chosen:
        try:
            data = load_npz(path)
            for key, arr in data.items():
                key_presence[key] += 1
                key_shapes.setdefault(key, Counter())[str(tuple(arr.shape))] += 1
            rec = infer_scene_record(path, data, overrides)
            resolved[tuple(sorted(rec.resolved_keys.items()))] += 1
            examples.append(
                {
                    "file": str(path),
                    "scenario_id": rec.scenario_id,
                    "num_candidates": int(rec.valid.size),
                    "valid": int(np.sum(rec.valid)),
                    "conventional_safe": int(np.sum(rec.valid & rec.conventional_safe)),
                    "ncf": int(np.sum(rec.valid & rec.ncf)),
                    "priority_eligible_available": rec.priority_eligible is not None,
                    "priority_ncf_available": rec.priority_ncf is not None,
                    "source_available": rec.source is not None,
                    "resolved_keys": rec.resolved_keys,
                }
            )
        except Exception as exc:
            errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})

    report = {
        "cache_dir": str(args.cache_dir),
        "files_total": len(files),
        "files_sampled": len(chosen),
        "sample_seed": args.seed,
        "resolved_schema_variants": [
            {"count": count, "keys": dict(items)} for items, count in resolved.most_common()
        ],
        "key_inventory": [
            {
                "key": key,
                "presence": count,
                "shapes": dict(key_shapes[key]),
            }
            for key, count in key_presence.most_common()
        ],
        "examples": examples,
        "errors": errors,
        "pass": bool(examples) and not errors,
        "next_action": (
            "Run analyze_proposal_cache.py with the resolved keys."
            if examples and not errors
            else "Inspect candidate-level label keys and pass explicit --*-key overrides."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
