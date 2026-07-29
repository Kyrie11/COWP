#!/usr/bin/env python3
"""Validate OC-RAP regime diagnostics before expensive experiments.

The central distinction is between *scene-time regime purity* and the regime of
all alternative candidate prefixes.  A clean Safe scene-time must have exactly
one nominal candidate labelled ``normal`` and no forbidden contamination, but it
may retain deliberately poor non-nominal candidates as useful negatives.  Older
versions incorrectly required 95% of every candidate sample to be ``normal``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SAFE_FORBIDDEN = (
    "near_contact",
    "post_contact",
    "oracle_artifact",
    "prefix_collision",
    "prefix_contact",
)


def load(root: Path, name: str) -> dict[str, Any]:
    p = root / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def mean(d: dict[str, Any], *keys: str) -> float | None:
    x: Any = d
    for k in keys:
        if not isinstance(x, dict):
            return None
        x = x.get(k)
    if isinstance(x, dict):
        x = x.get("mean")
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def frac(count: int, n: int) -> float:
    return float(count) / max(1, int(n))


def _names(scope: str) -> list[str]:
    if scope == "all":
        return [f"{s}_{b}" for s in ("train", "val", "test") for b in ("safe", "near_contact", "contact")]
    if scope == "trainval":
        return [f"{s}_{b}" for s in ("train", "val") for b in ("safe", "near_contact", "contact")]
    if scope == "v45dev":
        # v45 trains only Near/Contact lightweight heads and uses val_safe as a
        # preservation gate.  The frozen base may still be stale, so this scope
        # is development-only and must never be used for final paper claims.
        return [
            "val_safe",
            "train_near_contact", "val_near_contact",
            "train_contact", "val_contact",
        ]
    raise ValueError(scope)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("diagnostics_dir", type=Path)
    ap.add_argument("--mode", choices=["development", "paper"], default="development")
    ap.add_argument("--scope", choices=["all", "trainval", "v45dev"], default="all")
    ap.add_argument("--max-near-post-contact", type=float, default=0.10)
    ap.add_argument("--min-contact-post-contact", type=float, default=0.95)
    ap.add_argument("--min-safe-normal", type=float, default=0.95,
                    help="Minimum normal fraction among nominal candidates, not all candidates.")
    ap.add_argument("--max-safe-forbidden-fraction", type=float, default=0.0)
    ap.add_argument("--min-paper-safe-scenes", type=int, default=100)
    ap.add_argument("--min-development-stress-scenes", type=int, default=20)
    ap.add_argument("--min-paper-stress-scenes", type=int, default=100)
    ap.add_argument("--max-rdep-mean-shift", type=float, default=0.75)
    ap.add_argument("--max-hard-mean-shift", type=float, default=0.20)
    args = ap.parse_args()

    root = args.diagnostics_dir
    names = _names(args.scope)
    data = {name: load(root, name) for name in names}
    failures: list[str] = []
    warnings: list[str] = []
    rows: dict[str, dict[str, Any]] = {}

    for key, d in data.items():
        split, bucket = key.split("_", 1)
        n = int(d.get("num_samples", 0) or 0)
        groups = int(d.get("num_scene_time_groups", 0) or 0)
        scenes = int(d.get("num_scenes", 0) or 0)
        regimes_block = d.get("regimes") or {}
        regimes = regimes_block.get("counts", {}) or {}
        nominal_regimes = regimes_block.get("nominal_counts", {}) or {}
        nominal_total_raw = regimes_block.get("nominal_sample_count")
        nominal_total = int(nominal_total_raw) if nominal_total_raw is not None else 0

        post = frac(int(regimes.get("post_contact", 0) or 0), n)
        near = frac(int(regimes.get("near_contact", 0) or 0), n)
        candidate_normal = frac(int(regimes.get("normal", 0) or 0), n)
        nominal_normal = (
            frac(int(nominal_regimes.get("normal", 0) or 0), nominal_total)
            if nominal_total > 0 else None
        )
        nominal_near = (
            frac(int(nominal_regimes.get("near_contact", 0) or 0), nominal_total)
            if nominal_total > 0 else None
        )
        nominal_post = (
            frac(int(nominal_regimes.get("post_contact", 0) or 0), nominal_total)
            if nominal_total > 0 else None
        )
        forbidden_counts = {name: int(regimes.get(name, 0) or 0) for name in SAFE_FORBIDDEN}
        forbidden_total = sum(forbidden_counts.values())
        forbidden_fraction = frac(forbidden_total, n)

        rows[key] = {
            "samples": n,
            "groups": groups,
            "scenes": scenes,
            "post_contact_fraction": post,
            "near_contact_fraction": near,
            "candidate_normal_fraction": candidate_normal,
            "nominal_normal_fraction": nominal_normal,
            "nominal_near_contact_fraction": nominal_near,
            "nominal_post_contact_fraction": nominal_post,
            "safe_forbidden_fraction": forbidden_fraction,
            "safe_forbidden_counts": forbidden_counts,
            "r_dep_mean": mean(d, "recovery_labels", "r_dep_star"),
            "hard_mean": mean(d, "candidate_prefixes", "hard_violation"),
            "negative_deployable_fraction": mean(d, "recovery_labels", "negative_deployable_fraction"),
        }

        if n <= 0 or groups <= 0 or scenes <= 0:
            failures.append(f"{key}: empty or structurally invalid dataset (samples={n}, groups={groups}, scenes={scenes})")
            continue

        if nominal_total > 0 and nominal_total != groups:
            failures.append(f"{key}: nominal_sample_count={nominal_total} != scene_time_groups={groups}")

        if bucket == "near_contact":
            if post > args.max_near_post_contact:
                failures.append(f"{key}: post_contact_fraction={post:.3f} > {args.max_near_post_contact:.3f}")
            if nominal_near is not None and nominal_near < 0.95:
                failures.append(f"{key}: nominal_near_contact_fraction={nominal_near:.3f} < 0.950")

        elif bucket == "contact":
            if post < args.min_contact_post_contact:
                failures.append(f"{key}: post_contact_fraction={post:.3f} < {args.min_contact_post_contact:.3f}")
            if nominal_post is not None and nominal_post < args.min_contact_post_contact:
                failures.append(
                    f"{key}: nominal_post_contact_fraction={nominal_post:.3f} < {args.min_contact_post_contact:.3f}"
                )

        elif bucket == "safe":
            if forbidden_fraction > args.max_safe_forbidden_fraction:
                nonzero = {k: v for k, v in forbidden_counts.items() if v}
                failures.append(
                    f"{key}: forbidden regime contamination={forbidden_fraction:.6f} > "
                    f"{args.max_safe_forbidden_fraction:.6f}; counts={nonzero}"
                )

            if nominal_normal is not None:
                if nominal_normal < args.min_safe_normal:
                    failures.append(
                        f"{key}: nominal_normal_fraction={nominal_normal:.3f} < {args.min_safe_normal:.3f}"
                    )
            else:
                # Legacy reports do not contain nominal regime counts.  A strict
                # necessary condition is still available: there must be at least
                # one normal sample for every scene-time group.  Passing this
                # condition does not prove that the normal sample is nominal.
                normal_count = int(regimes.get("normal", 0) or 0)
                if normal_count < groups:
                    failures.append(
                        f"{key}: legacy diagnostics cannot satisfy nominal-normal contract: "
                        f"normal_count={normal_count} < scene_time_groups={groups}"
                    )
                else:
                    msg = (
                        f"{key}: legacy diagnostics lack nominal regime counts; candidate_normal_fraction="
                        f"{candidate_normal:.3f} is informational only. Rerun diagnose with patched code."
                    )
                    (failures if args.mode == "paper" else warnings).append(msg)

            if candidate_normal < args.min_safe_normal:
                warnings.append(
                    f"{key}: candidate_normal_fraction={candidate_normal:.3f}; this is allowed when hard "
                    "non-nominal negatives are retained, provided nominal purity and forbidden contamination pass"
                )

    if args.scope == "all":
        for bucket in ("near_contact", "contact"):
            v = rows[f"val_{bucket}"]
            t = rows[f"test_{bucket}"]
            if v["r_dep_mean"] is not None and t["r_dep_mean"] is not None:
                shift = abs(float(v["r_dep_mean"]) - float(t["r_dep_mean"]))
                if shift > args.max_rdep_mean_shift:
                    (failures if args.mode == "paper" else warnings).append(
                        f"{bucket}: |test-val r_dep mean shift|={shift:.3f}"
                    )
            if v["hard_mean"] is not None and t["hard_mean"] is not None:
                shift = abs(float(v["hard_mean"]) - float(t["hard_mean"]))
                if shift > args.max_hard_mean_shift:
                    (failures if args.mode == "paper" else warnings).append(
                        f"{bucket}: |test-val hard mean shift|={shift:.3f}"
                    )

    for key, row in rows.items():
        bucket = key.split("_", 1)[1]
        scenes = int(row["scenes"] or 0)
        if bucket == "safe" and key.startswith(("val_", "test_")) and scenes < args.min_paper_safe_scenes:
            (failures if args.mode == "paper" else warnings).append(
                f"{key}: only {scenes} scenes; paper target >= {args.min_paper_safe_scenes}"
            )
        if bucket in {"near_contact", "contact"} and key.startswith(("val_", "test_")):
            target = args.min_paper_stress_scenes if args.mode == "paper" else args.min_development_stress_scenes
            if scenes < target:
                (failures if args.mode == "paper" else warnings).append(
                    f"{key}: only {scenes} independent scenes; target >= {target}"
                )

    if args.scope == "v45dev":
        warnings.append(
            "v45dev scope excludes train_safe and validates only the data directly used by v45 head training plus val_safe; "
            "results are development-only until the base model is retrained on clean train_safe"
        )

    result = {
        "mode": args.mode,
        "scope": args.scope,
        "valid": not failures,
        "rows": rows,
        "failures": sorted(set(failures)),
        "warnings": sorted(set(warnings)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        print("RESULT: FAIL — rebuild/filter datasets before using them for this stage")
        return 2
    print("RESULT: PASS" if not warnings else "RESULT: PASS WITH WARNINGS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
