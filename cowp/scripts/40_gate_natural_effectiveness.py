from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mean(report: dict, key: str) -> float | None:
    row = report.get("distributions", {}).get(key, {})
    value = row.get("weighted_mean", row.get("mean"))
    return None if value is None else float(value)


def _p99(report: dict, key: str) -> float | None:
    value = report.get("distributions", {}).get(key, {}).get("p99")
    return None if value is None else float(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hard gate for a useful, physical, root-identifiable natural decoder.")
    ap.add_argument("--report", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-learned-8s-m", type=float, default=2.5)
    ap.add_argument("--max-obs-8s-m", type=float, default=4.0)
    ap.add_argument("--min-overall-gain-8s-m", type=float, default=0.03)
    ap.add_argument("--min-obs-gain-8s-m", type=float, default=0.05)
    ap.add_argument("--max-neutral-degradation-8s-m", type=float, default=0.15)
    ap.add_argument("--max-priority-degradation-8s-m", type=float, default=0.15)
    ap.add_argument("--max-velocity-error-mps", type=float, default=0.25)
    ap.add_argument("--max-yaw-error-rad", type=float, default=0.15)
    ap.add_argument("--min-effective-modes", type=float, default=2.0)
    ap.add_argument("--max-mass-soft-path-ratio", type=float, default=1.20)
    ap.add_argument("--max-mass-soft-path-violation", type=float, default=0.25)
    ap.add_argument("--max-emergency-path-ratio-p99", type=float, default=1.001)
    args = ap.parse_args()

    src = json.loads(Path(args.report).read_text(encoding="utf-8"))
    metrics = {
        "learned_8s_m": _mean(src, "all/8s/learned"),
        "overall_gain_8s_m": _mean(src, "all/8s/gain"),
        "obs_8s_m": _mean(src, "source_0/8s/learned"),
        "obs_gain_8s_m": _mean(src, "source_0/8s/gain"),
        "neutral_gain_8s_m": _mean(src, "source_1/8s/gain"),
        "priority_gain_8s_m": _mean(src, "source_2/8s/gain"),
        "velocity_error_mps": _mean(src, "kinematic/velocity_error_mps"),
        "yaw_error_rad": _mean(src, "kinematic/yaw_error_rad"),
        "mass_soft_path_ratio": _mean(src, "residual/soft_path_ratio"),
        "mass_soft_path_violation": _mean(src, "residual/soft_path_violation"),
        "emergency_path_ratio_p99": _p99(src, "residual/projected_emergency_path_ratio"),
        "projection_active_mass": _mean(src, "residual/projection_active"),
        "residual_endpoint_p99_m": _p99(src, "residual/endpoint_m"),
    }
    eff = [float(v.get("effective_modes", 0.0)) for v in src.get("mode_usage", {}).values()]
    metrics["min_effective_modes"] = min(eff) if eff else 0.0

    def le(name: str, threshold: float) -> bool:
        value = metrics.get(name)
        return value is not None and float(value) <= float(threshold)

    def ge(name: str, threshold: float) -> bool:
        value = metrics.get(name)
        return value is not None and float(value) >= float(threshold)

    checks = {
        "learned_absolute_quality": le("learned_8s_m", args.max_learned_8s_m),
        "obs_absolute_quality": le("obs_8s_m", args.max_obs_8s_m),
        "residual_improves_overall": ge("overall_gain_8s_m", args.min_overall_gain_8s_m),
        "residual_improves_obs": ge("obs_gain_8s_m", args.min_obs_gain_8s_m),
        "neutral_prior_preserved": ge("neutral_gain_8s_m", -args.max_neutral_degradation_8s_m),
        "priority_prior_preserved": ge("priority_gain_8s_m", -args.max_priority_degradation_8s_m),
        "velocity_consistency": le("velocity_error_mps", args.max_velocity_error_mps),
        "yaw_consistency": le("yaw_error_rad", args.max_yaw_error_rad),
        "mode_bank_is_used": ge("min_effective_modes", args.min_effective_modes),
        "probability_mass_stays_in_soft_root_envelope": le(
            "mass_soft_path_ratio", args.max_mass_soft_path_ratio
        ),
        "soft_root_envelope_violation_is_limited": le(
            "mass_soft_path_violation", args.max_mass_soft_path_violation
        ),
        "all_modes_respect_emergency_physical_envelope": le(
            "emergency_path_ratio_p99", args.max_emergency_path_ratio_p99
        ),
    }
    report = {
        "pass": bool(all(checks.values())),
        "metrics": metrics,
        "checks": checks,
        "thresholds": vars(args) | {"report": None, "output": None},
        "source_report": args.report,
        "failure_interpretation": (
            "A failure means the natural foundation is not yet both predictive and root-identifiable. "
            "Do not continue to transport/planner for a paper claim."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
