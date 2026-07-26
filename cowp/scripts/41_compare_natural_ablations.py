from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
        "emergency_path_ratio_p99": _p99(report, "residual/projected_emergency_path_ratio"),
        "projection_active_mass": _mean(report, "residual/projection_active"),
    }


def compare_reports(
    main: dict[str, Any],
    no_obs_capacity: dict[str, Any],
    no_mass_trust: dict[str, Any],
    *,
    min_capacity_obs_gain_m: float = 0.05,
    max_capacity_prior_regression_m: float = 0.15,
    min_mass_ratio_reduction: float = 0.03,
    min_mass_violation_reduction: float = 0.02,
    max_trust_obs_regression_m: float = 0.15,
    max_trust_overall_regression_m: float = 0.10,
) -> dict[str, Any]:
    """Controlled v16.5 attribution.

    v16.4's ``no_effectiveness_loss`` arm simultaneously removed seven loss
    families, so its result cannot identify a single component.  v16.5 gates only
    the two core additions with one-factor-at-a-time controls: source-adaptive OBS
    capacity and probability-mass-aware root-identity regularization.
    """
    rows = {
        "main": _metrics(main),
        "no_obs_capacity_boost": _metrics(no_obs_capacity),
        "no_mass_aware_root_envelope": _metrics(no_mass_trust),
    }

    def req(group: str, key: str) -> float:
        value = rows[group].get(key)
        if value is None:
            raise ValueError(f"Missing required metric {group}.{key}")
        return float(value)

    deltas = {
        # Positive means the proposed main model is better/lower.
        "obs_capacity_obs_improvement_m": (
            req("no_obs_capacity_boost", "obs_8s_m") - req("main", "obs_8s_m")
        ),
        "capacity_neutral_regression_m": (
            req("main", "neutral_8s_m") - req("no_obs_capacity_boost", "neutral_8s_m")
        ),
        "capacity_priority_regression_m": (
            req("main", "priority_8s_m") - req("no_obs_capacity_boost", "priority_8s_m")
        ),
        "mass_ratio_reduction": (
            req("no_mass_aware_root_envelope", "mass_soft_path_ratio")
            - req("main", "mass_soft_path_ratio")
        ),
        "mass_violation_reduction": (
            req("no_mass_aware_root_envelope", "mass_soft_path_violation")
            - req("main", "mass_soft_path_violation")
        ),
        "mass_trust_obs_improvement_m": (
            req("no_mass_aware_root_envelope", "obs_8s_m") - req("main", "obs_8s_m")
        ),
        "mass_trust_overall_improvement_m": (
            req("no_mass_aware_root_envelope", "all_8s_m") - req("main", "all_8s_m")
        ),
    }
    checks = {
        "obs_capacity_improves_obs": (
            deltas["obs_capacity_obs_improvement_m"] >= float(min_capacity_obs_gain_m)
        ),
        "obs_capacity_preserves_neutral": (
            deltas["capacity_neutral_regression_m"] <= float(max_capacity_prior_regression_m)
        ),
        "obs_capacity_preserves_priority": (
            deltas["capacity_priority_regression_m"] <= float(max_capacity_prior_regression_m)
        ),
        "mass_envelope_reduces_probability_weighted_ratio": (
            deltas["mass_ratio_reduction"] >= float(min_mass_ratio_reduction)
        ),
        "mass_envelope_reduces_probability_weighted_violation": (
            deltas["mass_violation_reduction"] >= float(min_mass_violation_reduction)
        ),
        "mass_envelope_does_not_harm_obs": (
            deltas["mass_trust_obs_improvement_m"] >= -float(max_trust_obs_regression_m)
        ),
        "mass_envelope_does_not_harm_overall": (
            deltas["mass_trust_overall_improvement_m"] >= -float(max_trust_overall_regression_m)
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "metrics": rows,
        "deltas": deltas,
        "thresholds": {
            "min_capacity_obs_gain_m": float(min_capacity_obs_gain_m),
            "max_capacity_prior_regression_m": float(max_capacity_prior_regression_m),
            "min_mass_ratio_reduction": float(min_mass_ratio_reduction),
            "min_mass_violation_reduction": float(min_mass_violation_reduction),
            "max_trust_obs_regression_m": float(max_trust_obs_regression_m),
            "max_trust_overall_regression_m": float(max_trust_overall_regression_m),
        },
        "interpretation": (
            "The gate validates only isolated core components. It intentionally "
            "does not reuse the invalid v16.4 compound no-effectiveness-loss arm."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare controlled v16.5 natural-stage ablations.")
    ap.add_argument("--main", required=True)
    ap.add_argument("--no-obs-capacity", required=True)
    ap.add_argument("--no-mass-trust", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-capacity-obs-gain-m", type=float, default=0.05)
    ap.add_argument("--max-capacity-prior-regression-m", type=float, default=0.15)
    ap.add_argument("--min-mass-ratio-reduction", type=float, default=0.03)
    ap.add_argument("--min-mass-violation-reduction", type=float, default=0.02)
    ap.add_argument("--max-trust-obs-regression-m", type=float, default=0.15)
    ap.add_argument("--max-trust-overall-regression-m", type=float, default=0.10)
    args = ap.parse_args()
    report = compare_reports(
        json.loads(Path(args.main).read_text(encoding="utf-8")),
        json.loads(Path(args.no_obs_capacity).read_text(encoding="utf-8")),
        json.loads(Path(args.no_mass_trust).read_text(encoding="utf-8")),
        min_capacity_obs_gain_m=args.min_capacity_obs_gain_m,
        max_capacity_prior_regression_m=args.max_capacity_prior_regression_m,
        min_mass_ratio_reduction=args.min_mass_ratio_reduction,
        min_mass_violation_reduction=args.min_mass_violation_reduction,
        max_trust_obs_regression_m=args.max_trust_obs_regression_m,
        max_trust_overall_regression_m=args.max_trust_overall_regression_m,
    )
    report["sources"] = {
        "main": args.main,
        "no_obs_capacity": args.no_obs_capacity,
        "no_mass_trust": args.no_mass_trust,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
