#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from cache_io import (
    add_key_override_arguments,
    discover_npz_files,
    infer_scene_record,
    load_npz,
    parse_key_overrides,
)


def safe_rate(num: int, den: int) -> float | None:
    return None if den <= 0 else num / den


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure fixed-bank proposal sufficiency.")
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=0, help="0 means all files")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--promotion-config", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    add_key_override_arguments(ap)
    args = ap.parse_args()

    files = discover_npz_files(args.cache_dir)
    if not files:
        raise SystemExit(f"no cache files found under {args.cache_dir}")
    if 0 < args.sample < len(files):
        files = random.Random(args.seed).sample(files, args.sample)

    overrides = parse_key_overrides(args)
    counts = Counter()
    candidate_counts = Counter()
    source_ncf_scenes = Counter()
    resolved_schema = Counter()
    errors = []

    for path in files:
        try:
            rec = infer_scene_record(path, load_npz(path), overrides)
            counts["scenes"] += 1
            any_valid = rec.any_valid()
            any_conv = rec.any_conventional_safe()
            any_ncf = rec.any_ncf()
            counts["any_valid"] += int(any_valid)
            counts["any_conventional_safe"] += int(any_conv)
            counts["any_ncf"] += int(any_ncf)
            counts["conventional_without_ncf"] += int(any_conv and not any_ncf)
            candidate_counts["valid"] += int(np.sum(rec.valid))
            candidate_counts["conventional_safe"] += int(
                np.sum(rec.valid & rec.conventional_safe)
            )
            candidate_counts["ncf"] += int(np.sum(rec.valid & rec.ncf))

            pe = rec.any_priority_eligible()
            pn = rec.any_priority_ncf()
            if pe is not None:
                counts["priority_available_scenes"] += 1
                counts["any_priority_eligible"] += int(pe)
            if pn is not None:
                counts["priority_ncf_available_scenes"] += 1
                counts["any_priority_ncf"] += int(pn)
                counts["priority_eligible_without_ncf"] += int(bool(pe) and not pn)

            for source in rec.ncf_sources():
                source_ncf_scenes[source] += 1
            resolved_schema[tuple(sorted(rec.resolved_keys.items()))] += 1
        except Exception as exc:
            errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})

    n = counts["scenes"]
    any_conv_rate = safe_rate(counts["any_conventional_safe"], n)
    any_ncf_rate = safe_rate(counts["any_ncf"], n)
    fs_floor = safe_rate(counts["conventional_without_ncf"], n)

    priority_complete = (
        counts["priority_available_scenes"] == n
        and counts["priority_ncf_available_scenes"] == n
    )
    any_pe_rate = safe_rate(counts["any_priority_eligible"], n) if priority_complete else None
    any_pn_rate = safe_rate(counts["any_priority_ncf"], n) if priority_complete else None
    pbtr_floor = (
        safe_rate(
            counts["priority_eligible_without_ncf"],
            counts["any_priority_eligible"],
        )
        if priority_complete
        else None
    )

    thresholds = {
        "min_any_ncf_scene_rate": 0.40,
        "max_best_case_selected_false_safe_lower_bound": 0.55,
        "max_best_case_pbtr_lower_bound": 0.45,
        "require_priority_metrics": True,
        "max_read_error_rate": 0.0,
    }
    if args.promotion_config:
        cfg = yaml.safe_load(args.promotion_config.read_text(encoding="utf-8")) or {}
        thresholds.update(cfg.get("full_rebuild_gate", {}))

    checks = {
        "any_ncf_scene_rate": any_ncf_rate is not None
        and any_ncf_rate >= thresholds["min_any_ncf_scene_rate"],
        "selected_false_safe_floor": fs_floor is not None
        and fs_floor <= thresholds["max_best_case_selected_false_safe_lower_bound"],
        "priority_metrics_available": priority_complete
        or not thresholds.get("require_priority_metrics", True),
        "pbtr_floor": pbtr_floor is not None
        and pbtr_floor <= thresholds["max_best_case_pbtr_lower_bound"],
        "read_error_rate": safe_rate(len(errors), len(files)) is not None
        and safe_rate(len(errors), len(files)) <= thresholds["max_read_error_rate"],
    }
    if not thresholds.get("require_priority_metrics", True) and pbtr_floor is None:
        checks["pbtr_floor"] = True

    report = {
        "version": "v16.8.4_fixed_bank_audit",
        "cache_dir": str(args.cache_dir),
        "files_requested": len(files),
        "scenes_read": n,
        "errors": errors[:100],
        "counts": dict(counts),
        "candidate_counts": dict(candidate_counts),
        "metrics": {
            "AnyValidSceneRate": safe_rate(counts["any_valid"], n),
            "AnyConventionalSafeSceneRate": any_conv_rate,
            "AnyNCFSceneRate": any_ncf_rate,
            "ConventionalWithoutNCFSceneRate": fs_floor,
            "BestCaseSelectedFalseSafeLowerBound": fs_floor,
            "AnyPriorityEligibleSceneRate": any_pe_rate,
            "AnyPriorityNCFSceneRate": any_pn_rate,
            "BestCasePBTRLowerBound": pbtr_floor,
        },
        "source_ncf_scene_counts": dict(source_ncf_scenes.most_common()),
        "resolved_schema_variants": [
            {"count": count, "keys": dict(items)}
            for items, count in resolved_schema.most_common()
        ],
        "thresholds": thresholds,
        "checks": checks,
        "promote_to_paired_probe": all(checks.values()),
        "interpretation": (
            "The cache is only a fixed-bank feasibility audit. Passing does not establish "
            "selector quality, closed-loop safety, causal burden validity, or SOTA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
