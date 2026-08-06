#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
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


def load_records(cache_dir: Path, overrides: dict) -> tuple[dict, list]:
    records = {}
    errors = []
    for path in discover_npz_files(cache_dir):
        try:
            rec = infer_scene_record(path, load_npz(path), overrides)
            if rec.scenario_id in records:
                raise RuntimeError(f"duplicate scenario id {rec.scenario_id}")
            records[rec.scenario_id] = rec
        except Exception as exc:
            errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return records, errors


def rate(num: int, den: int) -> float | None:
    return None if den <= 0 else num / den


def summarize(records: dict, ids: list[str]) -> dict:
    c = Counter()
    source_ncf = Counter()
    source_provenance_scenes = 0
    for sid in ids:
        rec = records[sid]
        conv = rec.any_conventional_safe()
        ncf = rec.any_ncf()
        pe = rec.any_priority_eligible()
        pn = rec.any_priority_ncf()
        c["scenes"] += 1
        c["any_valid"] += int(rec.any_valid())
        c["any_conventional_safe"] += int(conv)
        c["any_ncf"] += int(ncf)
        c["conventional_without_ncf"] += int(conv and not ncf)
        if pe is not None:
            c["priority_available"] += 1
            c["any_priority_eligible"] += int(pe)
        if pn is not None:
            c["priority_ncf_available"] += 1
            c["any_priority_ncf"] += int(pn)
            c["priority_eligible_without_ncf"] += int(bool(pe) and not pn)
        if rec.source is not None:
            source_provenance_scenes += 1
            for source in rec.ncf_sources():
                source_ncf[source] += 1

    n = c["scenes"]
    priority_complete = c["priority_available"] == n and c["priority_ncf_available"] == n
    return {
        "counts": dict(c),
        "metrics": {
            "AnyConventionalSafeSceneRate": rate(c["any_conventional_safe"], n),
            "AnyNCFSceneRate": rate(c["any_ncf"], n),
            "BestCaseSelectedFalseSafeLowerBound": rate(c["conventional_without_ncf"], n),
            "AnyPriorityEligibleSceneRate": (
                rate(c["any_priority_eligible"], n) if priority_complete else None
            ),
            "AnyPriorityNCFSceneRate": (
                rate(c["any_priority_ncf"], n) if priority_complete else None
            ),
            "BestCasePBTRLowerBound": (
                rate(c["priority_eligible_without_ncf"], c["any_priority_eligible"])
                if priority_complete
                else None
            ),
            "ProvenanceSceneCoverage": rate(source_provenance_scenes, n),
        },
        "source_ncf_scene_counts": dict(source_ncf.most_common()),
    }


def rmr_sources(rec) -> set[str]:
    return {
        s for s in rec.ncf_sources()
        if any(tok in s.lower() for tok in ("rmr", "bcte", "timing_envelope", "timing"))
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare old, single-region control, and RMR-BCTE proposal caches."
    )
    ap.add_argument("--old-cache", required=True, type=Path)
    ap.add_argument("--control-cache", type=Path)
    ap.add_argument("--new-cache", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--promotion-config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    add_key_override_arguments(ap)
    args = ap.parse_args()

    overrides = parse_key_overrides(args)
    old, old_errors = load_records(args.old_cache, overrides)
    new, new_errors = load_records(args.new_cache, overrides)
    control, control_errors = (
        load_records(args.control_cache, overrides) if args.control_cache else ({}, [])
    )
    if old_errors or new_errors or control_errors:
        report = {
            "pass": False,
            "errors": {
                "old": old_errors[:20],
                "control": control_errors[:20],
                "new": new_errors[:20],
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    hard_ids = list(manifest["hard_ids"])
    random_ids = list(manifest["random_ids"])
    all_ids = hard_ids + random_ids
    required_sets = [set(old), set(new)]
    if args.control_cache:
        required_sets.append(set(control))
    missing = {
        "old": sorted(set(all_ids) - set(old)),
        "new": sorted(set(all_ids) - set(new)),
        "control": sorted(set(all_ids) - set(control)) if args.control_cache else [],
    }
    paired_complete = not any(missing.values())
    if not paired_complete:
        report = {"pass": False, "missing_requested_ids": missing}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 3

    old_random = summarize(old, random_ids)
    new_random = summarize(new, random_ids)
    control_random = summarize(control, random_ids) if args.control_cache else None

    hard_recovered_new = sum(int(new[sid].any_ncf()) for sid in hard_ids)
    hard_recovered_control = (
        sum(int(control[sid].any_ncf()) for sid in hard_ids) if args.control_cache else None
    )
    old_ncf_ids = [sid for sid in random_ids if old[sid].any_ncf()]
    old_ncf_retained_new = sum(int(new[sid].any_ncf()) for sid in old_ncf_ids)
    old_ncf_retained_control = (
        sum(int(control[sid].any_ncf()) for sid in old_ncf_ids)
        if args.control_cache
        else None
    )
    rmr_recovered_hard = sum(
        int(new[sid].any_ncf() and bool(rmr_sources(new[sid]))) for sid in hard_ids
    )
    rmr_unique_recovered_hard = sum(
        int(
            new[sid].any_ncf()
            and bool(rmr_sources(new[sid]))
            and (not args.control_cache or not control[sid].any_ncf())
        )
        for sid in hard_ids
    )

    cfg = yaml.safe_load(args.promotion_config.read_text(encoding="utf-8")) or {}
    full_gate = cfg["full_rebuild_gate"]
    attribution_gate = cfg["algorithm_attribution_gate"]

    nm = new_random["metrics"]
    checks = {
        "paired_coverage": paired_complete,
        "any_ncf_scene_rate": (
            nm["AnyNCFSceneRate"] is not None
            and nm["AnyNCFSceneRate"] >= full_gate["min_any_ncf_scene_rate"]
        ),
        "selected_false_safe_floor": (
            nm["BestCaseSelectedFalseSafeLowerBound"] is not None
            and nm["BestCaseSelectedFalseSafeLowerBound"]
            <= full_gate["max_best_case_selected_false_safe_lower_bound"]
        ),
        "pbtr_floor": (
            nm["BestCasePBTRLowerBound"] is not None
            and nm["BestCasePBTRLowerBound"]
            <= full_gate["max_best_case_pbtr_lower_bound"]
        ),
        "hard_scene_recovery": (
            rate(hard_recovered_new, len(hard_ids))
            >= full_gate["min_hard_scene_ncf_recovery_rate"]
        ),
        "old_ncf_retention": (
            rate(old_ncf_retained_new, len(old_ncf_ids)) is not None
            and rate(old_ncf_retained_new, len(old_ncf_ids))
            >= full_gate["min_old_ncf_scene_retention"]
        ),
        "provenance_coverage": (
            nm["ProvenanceSceneCoverage"] is not None
            and nm["ProvenanceSceneCoverage"]
            >= full_gate["min_provenance_scene_coverage"]
        ),
    }

    attribution = {
        "available": bool(args.control_cache),
        "single_region_control": control_random,
        "hard_recovery_control": (
            rate(hard_recovered_control, len(hard_ids))
            if hard_recovered_control is not None
            else None
        ),
        "hard_recovery_rmr": rate(hard_recovered_new, len(hard_ids)),
        "rmr_incremental_hard_recovery": (
            rate(hard_recovered_new - hard_recovered_control, len(hard_ids))
            if hard_recovered_control is not None
            else None
        ),
        "rmr_incremental_random_any_ncf": (
            nm["AnyNCFSceneRate"] - control_random["metrics"]["AnyNCFSceneRate"]
            if control_random is not None
            and nm["AnyNCFSceneRate"] is not None
            and control_random["metrics"]["AnyNCFSceneRate"] is not None
            else None
        ),
        "rmr_recovered_hard_scene_rate": rate(rmr_recovered_hard, len(hard_ids)),
        "rmr_unique_recovered_hard_scene_rate": rate(
            rmr_unique_recovered_hard, len(hard_ids)
        ),
    }
    attribution_checks = {
        "control_available": attribution["available"],
        "incremental_random_any_ncf": (
            attribution["rmr_incremental_random_any_ncf"] is not None
            and attribution["rmr_incremental_random_any_ncf"]
            >= attribution_gate["min_incremental_random_any_ncf_scene_rate"]
        ),
        "incremental_hard_recovery": (
            attribution["rmr_incremental_hard_recovery"] is not None
            and attribution["rmr_incremental_hard_recovery"]
            >= attribution_gate["min_incremental_hard_scene_recovery_rate"]
        ),
        "rmr_provenance_recovery": (
            attribution["rmr_unique_recovered_hard_scene_rate"] is not None
            and attribution["rmr_unique_recovered_hard_scene_rate"]
            >= attribution_gate["min_rmr_unique_hard_recovery_rate"]
        ),
    }

    report = {
        "version": "v16.8.4_three_arm_paired_proposal_probe",
        "manifest": str(args.manifest),
        "paired_complete": paired_complete,
        "missing_requested_ids": missing,
        "unbiased_random_stratum": {
            "old": old_random,
            "new_rmr_bcte": new_random,
        },
        "hard_stratum": {
            "scenes": len(hard_ids),
            "new_hard_scene_ncf_recovery_rate": rate(hard_recovered_new, len(hard_ids)),
            "single_region_hard_scene_ncf_recovery_rate": (
                rate(hard_recovered_control, len(hard_ids))
                if hard_recovered_control is not None
                else None
            ),
        },
        "old_ncf_scene_retention": {
            "eligible_old_ncf_random_scenes": len(old_ncf_ids),
            "new_rmr_bcte": rate(old_ncf_retained_new, len(old_ncf_ids)),
            "single_region_control": (
                rate(old_ncf_retained_control, len(old_ncf_ids))
                if old_ncf_retained_control is not None
                else None
            ),
        },
        "algorithm_attribution": attribution,
        "full_rebuild_thresholds": full_gate,
        "full_rebuild_checks": checks,
        "promote_to_full_rebuild": all(checks.values()),
        "algorithm_attribution_thresholds": attribution_gate,
        "algorithm_attribution_checks": attribution_checks,
        "algorithm_increment_demonstrated": all(attribution_checks.values()),
        "paper_interpretation": (
            "Full-rebuild promotion and algorithm-attribution are separate. A proposal "
            "bank can become feasible through an engineering fix without demonstrating "
            "RMR-BCTE novelty. Publication claims require both, followed by retraining and "
            "reactive closed-loop validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
