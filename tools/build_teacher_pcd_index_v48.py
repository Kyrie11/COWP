#!/usr/bin/env python3
"""Precompute exact teacher-PCD metadata for fast, aligned group sampling.

The index is intentionally separate from the dataset: it does not rewrite user
samples and can be regenerated when OC-MERO hyperparameters change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.serialization import load_npz
from ocrap.evaluation.metrics import (
    best_shared_option_index,
    deployable_recovery_success,
    post_contact_deployability_score,
)
from ocrap.models.data import bucket_id_for_path, iter_sample_paths_many


def _scalar(d: dict[str, Any], key: str, default: Any) -> Any:
    a = np.asarray(d.get(key, default))
    return a.item() if a.shape == () else a


def teacher_pcd(d: dict[str, Any], *, alpha: float, beta: float, top_m: int) -> float:
    m = np.asarray(d["m_star"], dtype=np.float64)
    p = np.asarray(d["root_probs"], dtype=np.float64)
    c = np.asarray(d.get("c_star", np.eye(m.shape[0])), dtype=np.float64)
    rv = np.asarray(d.get("root_valid", np.ones(m.shape[0])), dtype=bool)
    ov = np.asarray(d.get("option_valid", np.ones(m.shape[1])), dtype=bool)
    result = oc_mero(
        m, p, c, alpha=alpha, beta=beta, option_valid=ov, root_valid=rv,
        use_lcvar=True, use_obs_kernel=True, top_m=top_m,
    )
    option = best_shared_option_index(result.q, p, gamma=0.0, root_valid=rv, option_valid=ov)
    drs = deployable_recovery_success(m, p, option, root_valid=rv)
    r_dep = float(_scalar(d, "r_dep_star", result.r_dep))
    r_orc = float(_scalar(d, "r_orc_star", result.r_orc))
    return float(post_contact_deployability_score(drs, r_dep, max(0.0, r_orc - r_dep)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Comma-separated OC-RAP roots")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--top-m", type=int, default=8)
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--positive-gain", type=float, default=0.015)
    ap.add_argument("--deployable-macro-ids", default="2,3,5,6,7",
                    help="Comma-separated recovery macros permitted by the deployed selector")
    ap.add_argument("--summary-output", type=Path)
    ap.add_argument("--min-positive-groups-near", type=int, default=0)
    ap.add_argument("--min-positive-groups-contact", type=int, default=0)
    ap.add_argument("--min-positive-scenes-near", type=int, default=0)
    ap.add_argument("--min-positive-scenes-contact", type=int, default=0)
    ap.add_argument(
        "--quality-mode", choices=["strict", "warn", "off"], default="strict",
        help="strict exits non-zero on requested coverage shortfalls; warn records them but continues; off skips target checks",
    )
    args = ap.parse_args()
    deployable_macro_ids = {int(x.strip()) for x in str(args.deployable_macro_ids).split(",") if x.strip()}

    paths = iter_sample_paths_many(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    groups: set[tuple[int, str, int]] = set()
    group_targets: dict[tuple[int, str, int], dict[str, float]] = {}
    with tmp.open("w", encoding="utf-8") as f:
        for i, path in enumerate(paths, 1):
            d = load_npz(path)
            scene = str(_scalar(d, "scene_id", path.stem))
            time_index = int(_scalar(d, "time_index", 0))
            bucket = int(bucket_id_for_path(path))
            row = {
                "path": str(path.resolve()),
                "bucket": bucket,
                "scene": scene,
                "time": time_index,
                "candidate": int(_scalar(d, "candidate_index", 0)),
                "macro": int(_scalar(d, "prefix_macro_type_id", _scalar(d, "prefix_macro_id", -1))),
                "nominal": bool(float(_scalar(d, "is_nominal", 0.0)) > 0.5),
                "teacher_pcd": teacher_pcd(d, alpha=args.alpha, beta=args.beta, top_m=args.top_m),
            }
            key = (bucket, scene, time_index)
            groups.add(key)
            stats = group_targets.setdefault(key, {
                "nominal": float("-inf"), "best_recovery": float("-inf"),
                "best_macro": -1, "best_deployable_recovery": float("-inf"),
                "best_deployable_macro": -1, "scene": scene, "bucket": bucket,
            })
            if row["nominal"]:
                stats["nominal"] = max(stats["nominal"], row["teacher_pcd"])
            else:
                if row["teacher_pcd"] > stats["best_recovery"]:
                    stats["best_recovery"] = row["teacher_pcd"]
                    stats["best_macro"] = row["macro"]
                if row["macro"] in deployable_macro_ids and row["teacher_pcd"] > stats["best_deployable_recovery"]:
                    stats["best_deployable_recovery"] = row["teacher_pcd"]
                    stats["best_deployable_macro"] = row["macro"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i == 1 or i % max(1, args.progress_every) == 0 or i == len(paths):
                print({"event": "teacher_pcd_index_progress", "seen": i, "total": len(paths), "groups": len(groups)}, flush=True)
    tmp.replace(args.output)
    positive_all = [
        stats for stats in group_targets.values()
        if np.isfinite(stats["nominal"]) and np.isfinite(stats["best_recovery"])
        and stats["best_recovery"] - stats["nominal"] >= args.positive_gain
    ]
    positive = [
        stats for stats in group_targets.values()
        if np.isfinite(stats["nominal"]) and np.isfinite(stats["best_deployable_recovery"])
        and stats["best_deployable_recovery"] - stats["nominal"] >= args.positive_gain
    ]

    def bucket_summary(bucket: int, rows_source: list[dict[str, Any]], macro_key: str) -> dict[str, Any]:
        from collections import Counter
        rows = [x for x in rows_source if int(x["bucket"]) == bucket]
        scenes = Counter(str(x["scene"]) for x in rows)
        macros = Counter(int(x[macro_key]) for x in rows)
        n = len(rows)
        return {
            "positive_groups": n,
            "positive_scenes": len(scenes),
            "positive_macro_counts": {str(k): int(v) for k, v in sorted(macros.items())},
            "max_positive_macro_share": (max(macros.values()) / n) if n and macros else None,
            "top10_positive_scene_share": (sum(v for _, v in scenes.most_common(10)) / n) if n else None,
        }

    by_bucket = {"near": bucket_summary(1, positive, "best_deployable_macro"),
                 "contact": bucket_summary(2, positive, "best_deployable_macro")}
    all_macro_by_bucket = {"near": bucket_summary(1, positive_all, "best_macro"),
                           "contact": bucket_summary(2, positive_all, "best_macro")}

    # Development-screening heuristics are deliberately weaker than the final
    # publication coverage targets. They answer whether an algorithm-direction
    # experiment is worth running before paying for a multi-day rebuild.
    screening_thresholds = {
        "near": {"adequate_groups": 80, "adequate_scenes": 40, "marginal_groups": 20, "marginal_scenes": 10},
        "contact": {"adequate_groups": 60, "adequate_scenes": 30, "marginal_groups": 15, "marginal_scenes": 8},
    }
    for name, lim in screening_thresholds.items():
        got = by_bucket[name]
        groups_n = int(got["positive_groups"])
        scenes_n = int(got["positive_scenes"])
        macro_share = got.get("max_positive_macro_share")
        top10_share = got.get("top10_positive_scene_share")
        concentrated = bool(
            (macro_share is not None and float(macro_share) > 0.80)
            or (top10_share is not None and float(top10_share) > 0.60)
        )
        if groups_n >= lim["adequate_groups"] and scenes_n >= lim["adequate_scenes"] and not concentrated:
            status = "adequate_for_direction_screening"
            action = "reuse_existing_dataset"
        elif groups_n >= lim["marginal_groups"] and scenes_n >= lim["marginal_scenes"]:
            status = "marginal_debug_only"
            action = "run_screening_but_do_not_make_strong_claims"
        else:
            status = "data_limited"
            action = "consider_targeted_increment_or_rebuild_after_audit"
        got["screening_status"] = status
        got["screening_concentration_warning"] = concentrated
        got["screening_recommended_action"] = action

    summary = {
        "event": "teacher_pcd_index_complete",
        "output": str(args.output),
        "num_samples": len(paths),
        "num_groups": len(groups),
        "alpha": args.alpha,
        "beta": args.beta,
        "top_m": args.top_m,
        "positive_gain": args.positive_gain,
        "deployable_macro_ids": sorted(deployable_macro_ids),
        "positive_advantage_groups": len(positive),
        "positive_advantage_groups_all_macros": len(positive_all),
        "by_bucket": by_bucket,
        "all_macro_by_bucket": all_macro_by_bucket,
        "quality_mode": args.quality_mode,
    }

    failures: list[str] = []
    limits = {
        "near": (args.min_positive_groups_near, args.min_positive_scenes_near),
        "contact": (args.min_positive_groups_contact, args.min_positive_scenes_contact),
    }
    for name, (min_groups, min_scenes) in limits.items():
        got = by_bucket[name]
        if int(got["positive_groups"]) < min_groups:
            failures.append(f"{name}: positive groups {got['positive_groups']} < {min_groups}")
        if int(got["positive_scenes"]) < min_scenes:
            failures.append(f"{name}: positive scenes {got['positive_scenes']} < {min_scenes}")
    if args.quality_mode == "off":
        failures = []
    summary["quality_failures"] = failures
    summary["quality_passed"] = not failures
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary, flush=True)
    if failures:
        event = "teacher_pcd_index_quality_failure" if args.quality_mode == "strict" else "teacher_pcd_index_quality_warning"
        print({"event": event, "failures": failures}, flush=True)
        if args.quality_mode == "strict":
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
