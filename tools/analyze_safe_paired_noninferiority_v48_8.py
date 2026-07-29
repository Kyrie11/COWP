#!/usr/bin/env python3
"""Scene-paired Safe non-inferiority report for two closed-loop JSON files.

The tool deliberately marks metrics unavailable when the runner did not emit a
scene-level route, jerk, or yaw-rate statistic. It never substitutes unrelated
Waymax metrics for a publication claim.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Callable


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _scene_map(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in doc.get("scenes") or []:
        sid = str(row.get("scene_id") or "")
        if sid:
            out[sid] = row
    return out


def _metric_summary(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metric_summary")
    return value if isinstance(value, dict) else {}


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _get_nested(row: dict[str, Any], *keys: str) -> float | None:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return _finite(cur)


def _bootstrap_mean_ci(values: list[float], *, seed: int, samples: int, alpha: float) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_i = max(0, min(len(means) - 1, int((alpha / 2) * len(means))))
    hi_i = max(0, min(len(means) - 1, int((1 - alpha / 2) * len(means)) - 1))
    return means[lo_i], means[hi_i]


def _evaluate_metric(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    getter: Callable[[dict[str, Any]], float | None],
    *,
    name: str,
    direction: str,
    margin: float | None,
    seed: int,
    samples: int,
    alpha: float,
) -> dict[str, Any]:
    deltas: list[float] = []
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    for baseline, candidate in pairs:
        b = getter(baseline)
        c = getter(candidate)
        if b is None or c is None:
            continue
        baseline_values.append(b)
        candidate_values.append(c)
        deltas.append(c - b)
    if not deltas:
        return {
            "metric": name,
            "available": False,
            "reason": "metric is not emitted at scene level by both runs",
            "direction": direction,
            "noninferiority_margin": margin,
        }
    mean_delta = sum(deltas) / len(deltas)
    lo, hi = _bootstrap_mean_ci(deltas, seed=seed, samples=samples, alpha=alpha)
    passed = None
    if margin is not None:
        if direction == "lower_is_better":
            passed = hi <= margin
        elif direction == "higher_is_better":
            passed = lo >= -margin
    return {
        "metric": name,
        "available": True,
        "paired_scenes": len(deltas),
        "baseline_mean": sum(baseline_values) / len(baseline_values),
        "candidate_mean": sum(candidate_values) / len(candidate_values),
        "candidate_minus_baseline": mean_delta,
        "paired_ci95": [lo, hi],
        "direction": direction,
        "noninferiority_margin": margin,
        "passed_noninferiority": passed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=4880)
    args = ap.parse_args()

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    bmap = _scene_map(baseline)
    cmap = _scene_map(candidate)
    shared = sorted(set(bmap) & set(cmap))
    pairs = [(bmap[s], cmap[s]) for s in shared]

    metrics = [
        _evaluate_metric(
            pairs,
            lambda r: _get_nested(r, "metric_summary", "overlap_any"),
            name="collision_scene_rate",
            direction="lower_is_better",
            margin=0.002,
            seed=args.seed,
            samples=args.bootstrap_samples,
            alpha=0.05,
        ),
        _evaluate_metric(
            pairs,
            lambda r: _get_nested(r, "metric_summary", "offroad_any"),
            name="offroad_scene_rate",
            direction="lower_is_better",
            margin=0.002,
            seed=args.seed + 1,
            samples=args.bootstrap_samples,
            alpha=0.05,
        ),
        _evaluate_metric(
            pairs,
            lambda r: _get_nested(r, "closed_loop_bounded_NUP"),
            name="bounded_NUP",
            direction="higher_is_better",
            margin=0.01,
            seed=args.seed + 2,
            samples=args.bootstrap_samples,
            alpha=0.05,
        ),
        _evaluate_metric(
            pairs,
            lambda r: _get_nested(r, "intervention_episode_rate"),
            name="intervention_episode_rate",
            direction="lower_is_better",
            margin=0.03,
            seed=args.seed + 3,
            samples=args.bootstrap_samples,
            alpha=0.05,
        ),
        _evaluate_metric(
            pairs,
            lambda r: _get_nested(r, "route_progression"),
            name="route_progression",
            direction="higher_is_better",
            margin=0.005,
            seed=args.seed + 4,
            samples=args.bootstrap_samples,
            alpha=0.05,
        ),
        _evaluate_metric(
            pairs,
            lambda r: _get_nested(r, "metric_summary", "jerk_p95"),
            name="jerk_p95",
            direction="lower_is_better",
            margin=None,
            seed=args.seed + 5,
            samples=args.bootstrap_samples,
            alpha=0.05,
        ),
        _evaluate_metric(
            pairs,
            lambda r: _get_nested(r, "metric_summary", "yaw_rate_p95"),
            name="yaw_rate_p95",
            direction="lower_is_better",
            margin=None,
            seed=args.seed + 6,
            samples=args.bootstrap_samples,
            alpha=0.05,
        ),
    ]
    available_required = [m for m in metrics[:4] if m.get("available")]
    doc = {
        "version": "v48.8",
        "method": "scene_paired_bootstrap_noninferiority",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_scenes": len(bmap),
        "candidate_scenes": len(cmap),
        "paired_scenes": len(shared),
        "baseline_only_scenes": sorted(set(bmap) - set(cmap)),
        "candidate_only_scenes": sorted(set(cmap) - set(bmap)),
        "metrics": metrics,
        "core_available_metrics_passed": bool(available_required) and all(
            m.get("passed_noninferiority") is True for m in available_required
        ),
        "paper_safe_claim_ready": len(shared) >= 100
        and all(m.get("available") for m in metrics)
        and all(m.get("passed_noninferiority") is True for m in metrics[:4]),
        "notes": [
            "A small paired probe is diagnostic only; use at least 100 paired scenes for the paper claim.",
            "Unavailable route/jerk/yaw metrics are reported as unavailable rather than inferred from proxies.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
