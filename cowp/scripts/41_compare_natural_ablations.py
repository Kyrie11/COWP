from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _mean(report: dict[str, Any], key: str) -> float | None:
    row = report.get("distributions", {}).get(key, {})
    value = row.get("weighted_mean", row.get("mean"))
    return None if value is None else float(value)


def _p99(report: dict[str, Any], key: str) -> float | None:
    value = report.get("distributions", {}).get(key, {}).get("p99")
    return None if value is None else float(value)


def _metrics(report: dict[str, Any]) -> dict[str, float | None]:
    return {
        "all_8s_m": _mean(report, "all/8s/learned"),
        "obs_8s_m": _mean(report, "source_0/8s/learned"),
        "neutral_8s_m": _mean(report, "source_1/8s/learned"),
        "priority_8s_m": _mean(report, "source_2/8s/learned"),
        "overall_gain_8s_m": _mean(report, "all/8s/gain"),
        "obs_gain_8s_m": _mean(report, "source_0/8s/gain"),
        "mass_soft_path_ratio": _mean(report, "residual/soft_path_ratio"),
        "mass_soft_path_violation": _mean(report, "residual/soft_path_violation"),
        # This is the exact probability-mass-weighted squared excess optimized by
        # the v16.5/v16.6 root-identity loss.  The previous gate checked the mean
        # path ratio, which is not the optimized quantity and can remain nearly
        # unchanged while high-mass violations are removed.
        "mass_soft_path_excess_sq": _mean(report, "residual/soft_path_excess_sq"),
        "emergency_path_ratio_p99": _p99(report, "residual/projected_emergency_path_ratio"),
        "projection_active_mass": _mean(report, "residual/projection_active"),
    }


def _finite(values: Iterable[Any]) -> np.ndarray:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return np.asarray(out, dtype=np.float64)


def _paired_delta(
    main: dict[str, Any],
    ablation: dict[str, Any],
    key: str,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, float | int | None]:
    """Return ablation-minus-main paired effect and percentile CI.

    Positive values mean the proposed main model is better because all supported
    metrics are errors, violations, or excess penalties for which lower is better.
    Scene alignment is by explicit dataset index, never by list position alone.
    """
    main_rows = main.get("paired_scene_metrics", {})
    abl_rows = ablation.get("paired_scene_metrics", {})
    main_ids = list(main_rows.get("scene_index", []))
    abl_ids = list(abl_rows.get("scene_index", []))
    main_values = list(main_rows.get(key, []))
    abl_values = list(abl_rows.get(key, []))
    if not (len(main_ids) == len(main_values) and len(abl_ids) == len(abl_values)):
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}

    main_map = {int(i): v for i, v in zip(main_ids, main_values)}
    abl_map = {int(i): v for i, v in zip(abl_ids, abl_values)}
    common = sorted(set(main_map) & set(abl_map))
    diffs: list[float] = []
    for idx in common:
        mv = main_map[idx]
        av = abl_map[idx]
        if mv is None or av is None:
            continue
        try:
            m = float(mv)
            a = float(av)
        except (TypeError, ValueError):
            continue
        if math.isfinite(m) and math.isfinite(a):
            diffs.append(a - m)
    arr = np.asarray(diffs, dtype=np.float64)
    if not arr.size:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    mean = float(arr.mean())
    if arr.size == 1 or int(bootstrap_samples) <= 0:
        return {"n": int(arr.size), "mean": mean, "ci_low": mean, "ci_high": mean}
    rng = np.random.default_rng(int(seed))
    n_boot = max(int(bootstrap_samples), 200)
    means = np.empty(n_boot, dtype=np.float64)
    # Chunking avoids a large [bootstrap_samples, scenes] allocation.
    for start in range(0, n_boot, 256):
        end = min(start + 256, n_boot)
        sample = rng.integers(0, arr.size, size=(end - start, arr.size))
        means[start:end] = arr[sample].mean(axis=1)
    return {
        "n": int(arr.size),
        "mean": mean,
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def _same_protocol(reports: dict[str, dict[str, Any]]) -> tuple[dict[str, bool], dict[str, Any]]:
    epochs = {name: report.get("checkpoint_epoch") for name, report in reports.items()}
    hashes = {name: report.get("sample_indices_sha256") for name, report in reports.items()}
    scenes = {name: report.get("sampled_scenes") for name, report in reports.items()}
    decoder_types = {name: report.get("decoder_type") for name, report in reports.items()}
    protocols = {name: report.get("diagnostic_protocol", {}) for name, report in reports.items()}
    checks = {
        "same_checkpoint_epoch": len(set(epochs.values())) == 1 and None not in epochs.values(),
        "same_sample_indices": len(set(hashes.values())) == 1 and None not in hashes.values(),
        "same_sample_count": len(set(scenes.values())) == 1 and None not in scenes.values(),
        "same_decoder_family": len(set(decoder_types.values())) == 1 and None not in decoder_types.values(),
        "paired_scene_metrics_present": all(
            bool(protocols[name].get("paired_scene_metrics", False))
            and bool(reports[name].get("paired_scene_metrics", {}).get("scene_index", []))
            for name in reports
        ),
    }
    return checks, {
        "checkpoint_epochs": epochs,
        "sample_indices_sha256": hashes,
        "sampled_scenes": scenes,
        "decoder_types": decoder_types,
    }


def _relative_reduction(main_value: float, ablation_value: float) -> float:
    return float((ablation_value - main_value) / max(abs(ablation_value), 1.0e-8))


def compare_reports(
    main: dict[str, Any],
    no_obs_capacity: dict[str, Any],
    no_mass_trust: dict[str, Any],
    *,
    min_capacity_obs_improvement_m: float = 0.0,
    capacity_obs_noninferiority_margin_m: float = 0.02,
    max_capacity_overall_regression_m: float = 0.02,
    max_capacity_prior_regression_m: float = 0.03,
    min_mass_excess_relative_reduction: float = 0.10,
    min_mass_violation_relative_reduction: float = 0.10,
    max_envelope_obs_regression_m: float = 0.05,
    max_envelope_overall_regression_m: float = 0.05,
    max_emergency_ratio_p99: float = 1.001,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    """Protocol-correct v16.6 natural component attribution.

    The development gate answers a narrow question: are both isolated components
    active, directionally useful, and non-harmful enough to justify collecting the
    downstream closed-loop evidence?  It is deliberately distinct from a paper
    claim.  A publication claim still requires at least three seeds, held-out
    calibration, and confidence intervals that exclude zero.
    """
    reports = {
        "main": main,
        "no_obs_capacity_boost": no_obs_capacity,
        "no_mass_aware_root_envelope": no_mass_trust,
    }
    rows = {name: _metrics(report) for name, report in reports.items()}

    def req(group: str, key: str) -> float:
        value = rows[group].get(key)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Missing/non-finite required metric {group}.{key}")
        return float(value)

    protocol_checks, protocol = _same_protocol(reports)

    deltas = {
        "obs_capacity_obs_improvement_m": req("no_obs_capacity_boost", "obs_8s_m") - req("main", "obs_8s_m"),
        "capacity_overall_regression_m": req("main", "all_8s_m") - req("no_obs_capacity_boost", "all_8s_m"),
        "capacity_neutral_regression_m": req("main", "neutral_8s_m") - req("no_obs_capacity_boost", "neutral_8s_m"),
        "capacity_priority_regression_m": req("main", "priority_8s_m") - req("no_obs_capacity_boost", "priority_8s_m"),
        "mass_excess_sq_reduction": req("no_mass_aware_root_envelope", "mass_soft_path_excess_sq") - req("main", "mass_soft_path_excess_sq"),
        "mass_excess_sq_relative_reduction": _relative_reduction(
            req("main", "mass_soft_path_excess_sq"),
            req("no_mass_aware_root_envelope", "mass_soft_path_excess_sq"),
        ),
        "mass_violation_reduction": req("no_mass_aware_root_envelope", "mass_soft_path_violation") - req("main", "mass_soft_path_violation"),
        "mass_violation_relative_reduction": _relative_reduction(
            req("main", "mass_soft_path_violation"),
            req("no_mass_aware_root_envelope", "mass_soft_path_violation"),
        ),
        "mass_envelope_obs_improvement_m": req("no_mass_aware_root_envelope", "obs_8s_m") - req("main", "obs_8s_m"),
        "mass_envelope_overall_improvement_m": req("no_mass_aware_root_envelope", "all_8s_m") - req("main", "all_8s_m"),
    }

    paired = {
        "capacity_obs_improvement_m": _paired_delta(
            main, no_obs_capacity, "obs_8s_m", seed=bootstrap_seed + 1,
            bootstrap_samples=bootstrap_samples,
        ),
        "capacity_overall_improvement_m": _paired_delta(
            main, no_obs_capacity, "all_8s_m", seed=bootstrap_seed + 2,
            bootstrap_samples=bootstrap_samples,
        ),
        "envelope_obs_improvement_m": _paired_delta(
            main, no_mass_trust, "obs_8s_m", seed=bootstrap_seed + 3,
            bootstrap_samples=bootstrap_samples,
        ),
        "envelope_overall_improvement_m": _paired_delta(
            main, no_mass_trust, "all_8s_m", seed=bootstrap_seed + 4,
            bootstrap_samples=bootstrap_samples,
        ),
        "envelope_mass_excess_sq_reduction": _paired_delta(
            main, no_mass_trust, "mass_soft_path_excess_sq", seed=bootstrap_seed + 5,
            bootstrap_samples=bootstrap_samples,
        ),
        "envelope_mass_violation_reduction": _paired_delta(
            main, no_mass_trust, "mass_soft_path_violation", seed=bootstrap_seed + 6,
            bootstrap_samples=bootstrap_samples,
        ),
    }

    def paired_mean(name: str) -> float | None:
        value = paired[name].get("mean")
        return None if value is None else float(value)

    def paired_low(name: str) -> float | None:
        value = paired[name].get("ci_low")
        return None if value is None else float(value)

    capacity_point = paired_mean("capacity_obs_improvement_m")
    capacity_low = paired_low("capacity_obs_improvement_m")
    envelope_excess_point = paired_mean("envelope_mass_excess_sq_reduction")
    envelope_violation_point = paired_mean("envelope_mass_violation_reduction")

    component_checks = {
        # The continuation gate requires directional OBS benefit and excludes a
        # practically meaningful paired degradation.  It does not pretend that a
        # single seed proves the component for publication.
        "obs_capacity_directionally_improves_obs": (
            capacity_point is not None and capacity_point >= float(min_capacity_obs_improvement_m)
        ),
        "obs_capacity_is_paired_noninferior": (
            capacity_low is not None and capacity_low >= -float(capacity_obs_noninferiority_margin_m)
        ),
        "obs_capacity_preserves_overall": (
            deltas["capacity_overall_regression_m"] <= float(max_capacity_overall_regression_m)
        ),
        "obs_capacity_preserves_neutral": (
            deltas["capacity_neutral_regression_m"] <= float(max_capacity_prior_regression_m)
        ),
        "obs_capacity_preserves_priority": (
            deltas["capacity_priority_regression_m"] <= float(max_capacity_prior_regression_m)
        ),
        # The envelope is attributed against the exact optimized squared excess
        # and the probability mass outside the semantic envelope, not against the
        # unoptimized global mean ratio used by v16.5.
        "mass_envelope_reduces_exact_excess_objective": (
            deltas["mass_excess_sq_relative_reduction"] >= float(min_mass_excess_relative_reduction)
            and envelope_excess_point is not None and envelope_excess_point > 0.0
        ),
        "mass_envelope_reduces_violation_mass": (
            deltas["mass_violation_relative_reduction"] >= float(min_mass_violation_relative_reduction)
            and envelope_violation_point is not None and envelope_violation_point > 0.0
        ),
        "mass_envelope_preserves_obs": (
            deltas["mass_envelope_obs_improvement_m"] >= -float(max_envelope_obs_regression_m)
        ),
        "mass_envelope_preserves_overall": (
            deltas["mass_envelope_overall_improvement_m"] >= -float(max_envelope_overall_regression_m)
        ),
        "emergency_envelope_is_respected": (
            req("main", "emergency_path_ratio_p99") <= float(max_emergency_ratio_p99)
        ),
    }
    checks = {**protocol_checks, **component_checks}

    # Paper readiness is intentionally stricter and cannot be satisfied by the
    # single-seed attribution package alone.  This prevents the development gate
    # from being quoted as statistical evidence in the manuscript.
    paper_claim_checks = {
        "requires_three_or_more_independent_seeds": False,
        "capacity_95pct_ci_excludes_zero": bool(capacity_low is not None and capacity_low > 0.0),
        "envelope_excess_95pct_ci_excludes_zero": bool(
            paired_low("envelope_mass_excess_sq_reduction") is not None
            and float(paired_low("envelope_mass_excess_sq_reduction")) > 0.0
        ),
        "envelope_violation_95pct_ci_excludes_zero": bool(
            paired_low("envelope_mass_violation_reduction") is not None
            and float(paired_low("envelope_mass_violation_reduction")) > 0.0
        ),
    }

    return {
        "pass": bool(all(checks.values())),
        "paper_claim_ready": bool(all(paper_claim_checks.values())),
        "checks": checks,
        "paper_claim_checks": paper_claim_checks,
        "protocol": protocol,
        "metrics": rows,
        "deltas": deltas,
        "paired_bootstrap": paired,
        "thresholds": {
            "min_capacity_obs_improvement_m": float(min_capacity_obs_improvement_m),
            "capacity_obs_noninferiority_margin_m": float(capacity_obs_noninferiority_margin_m),
            "max_capacity_overall_regression_m": float(max_capacity_overall_regression_m),
            "max_capacity_prior_regression_m": float(max_capacity_prior_regression_m),
            "min_mass_excess_relative_reduction": float(min_mass_excess_relative_reduction),
            "min_mass_violation_relative_reduction": float(min_mass_violation_relative_reduction),
            "max_envelope_obs_regression_m": float(max_envelope_obs_regression_m),
            "max_envelope_overall_regression_m": float(max_envelope_overall_regression_m),
            "max_emergency_ratio_p99": float(max_emergency_ratio_p99),
            "bootstrap_samples": int(bootstrap_samples),
        },
        "interpretation": (
            "pass is a protocol-correct development continuation gate for collecting downstream evidence. "
            "It requires aligned checkpoints and paired scenes. paper_claim_ready remains false until the "
            "same isolated effects are reproduced across at least three independent seeds."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare protocol-aligned v16.6 natural-stage ablations.")
    ap.add_argument("--main", required=True)
    ap.add_argument("--no-obs-capacity", required=True)
    ap.add_argument("--no-mass-trust", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-capacity-obs-improvement-m", type=float, default=0.0)
    ap.add_argument("--capacity-obs-noninferiority-margin-m", type=float, default=0.02)
    ap.add_argument("--max-capacity-overall-regression-m", type=float, default=0.02)
    ap.add_argument("--max-capacity-prior-regression-m", type=float, default=0.03)
    ap.add_argument("--min-mass-excess-relative-reduction", type=float, default=0.10)
    ap.add_argument("--min-mass-violation-relative-reduction", type=float, default=0.10)
    ap.add_argument("--max-envelope-obs-regression-m", type=float, default=0.05)
    ap.add_argument("--max-envelope-overall-regression-m", type=float, default=0.05)
    ap.add_argument("--max-emergency-ratio-p99", type=float, default=1.001)
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    args = ap.parse_args()
    report = compare_reports(
        json.loads(Path(args.main).read_text(encoding="utf-8")),
        json.loads(Path(args.no_obs_capacity).read_text(encoding="utf-8")),
        json.loads(Path(args.no_mass_trust).read_text(encoding="utf-8")),
        min_capacity_obs_improvement_m=args.min_capacity_obs_improvement_m,
        capacity_obs_noninferiority_margin_m=args.capacity_obs_noninferiority_margin_m,
        max_capacity_overall_regression_m=args.max_capacity_overall_regression_m,
        max_capacity_prior_regression_m=args.max_capacity_prior_regression_m,
        min_mass_excess_relative_reduction=args.min_mass_excess_relative_reduction,
        min_mass_violation_relative_reduction=args.min_mass_violation_relative_reduction,
        max_envelope_obs_regression_m=args.max_envelope_obs_regression_m,
        max_envelope_overall_regression_m=args.max_envelope_overall_regression_m,
        max_emergency_ratio_p99=args.max_emergency_ratio_p99,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    report["sources"] = {
        "main": args.main,
        "no_obs_capacity": args.no_obs_capacity,
        "no_mass_trust": args.no_mass_trust,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
