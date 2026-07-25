from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _mean(report: dict[str, Any], key: str) -> float | None:
    row = report.get("distributions", {}).get(key, {})
    value = row.get("weighted_mean", row.get("mean"))
    return None if value is None else float(value)


def _metrics(report: dict[str, Any]) -> dict[str, float | None]:
    return {
        "all_8s_m": _mean(report, "all/8s/learned"),
        "obs_8s_m": _mean(report, "source_0/8s/learned"),
        "neutral_8s_m": _mean(report, "source_1/8s/learned"),
        "priority_8s_m": _mean(report, "source_2/8s/learned"),
        "overall_gain_8s_m": _mean(report, "all/8s/gain"),
        "obs_gain_8s_m": _mean(report, "source_0/8s/gain"),
        "velocity_error_mps": _mean(report, "kinematic/velocity_error_mps"),
        "yaw_error_rad": _mean(report, "kinematic/yaw_error_rad"),
        "residual_endpoint_p99_m": report.get("distributions", {}).get("residual/endpoint_m", {}).get("p99"),
        "residual_budget_saturation_rate": _mean(report, "residual/budget_saturated"),
    }


def compare_reports(
    main: dict[str, Any],
    no_loss: dict[str, Any],
    no_obs_capacity: dict[str, Any],
    no_trust_region: dict[str, Any] | None = None,
    *,
    min_loss_obs_gain_m: float = 0.05,
    min_loss_overall_gain_m: float = 0.02,
    min_capacity_obs_gain_m: float = 0.05,
    max_prior_regression_m: float = 0.15,
    min_trust_tail_reduction_m: float = 5.0,
    max_trust_obs_regression_m: float = 0.15,
) -> dict[str, Any]:
    rows = {
        "main": _metrics(main),
        "no_effectiveness_loss": _metrics(no_loss),
        "no_obs_capacity_boost": _metrics(no_obs_capacity),
    }
    if no_trust_region is not None:
        rows["no_trust_region"] = _metrics(no_trust_region)

    def req(group: str, key: str) -> float:
        value = rows[group].get(key)
        if value is None:
            raise ValueError(f"Missing required metric {group}.{key}")
        return float(value)

    deltas = {
        # Positive means the proposed main model has lower error.
        "new_loss_obs_improvement_m": req("no_effectiveness_loss", "obs_8s_m") - req("main", "obs_8s_m"),
        "new_loss_overall_improvement_m": req("no_effectiveness_loss", "all_8s_m") - req("main", "all_8s_m"),
        "obs_capacity_improvement_m": req("no_obs_capacity_boost", "obs_8s_m") - req("main", "obs_8s_m"),
        "main_vs_no_loss_neutral_delta_m": req("main", "neutral_8s_m") - req("no_effectiveness_loss", "neutral_8s_m"),
        "main_vs_no_loss_priority_delta_m": req("main", "priority_8s_m") - req("no_effectiveness_loss", "priority_8s_m"),
        "main_vs_no_capacity_neutral_delta_m": req("main", "neutral_8s_m") - req("no_obs_capacity_boost", "neutral_8s_m"),
        "main_vs_no_capacity_priority_delta_m": req("main", "priority_8s_m") - req("no_obs_capacity_boost", "priority_8s_m"),
    }
    checks = {
        "new_loss_improves_obs": deltas["new_loss_obs_improvement_m"] >= float(min_loss_obs_gain_m),
        "new_loss_improves_overall": deltas["new_loss_overall_improvement_m"] >= float(min_loss_overall_gain_m),
        "obs_capacity_improves_obs": deltas["obs_capacity_improvement_m"] >= float(min_capacity_obs_gain_m),
        "new_loss_preserves_neutral": deltas["main_vs_no_loss_neutral_delta_m"] <= float(max_prior_regression_m),
        "new_loss_preserves_priority": deltas["main_vs_no_loss_priority_delta_m"] <= float(max_prior_regression_m),
        "obs_capacity_preserves_neutral": deltas["main_vs_no_capacity_neutral_delta_m"] <= float(max_prior_regression_m),
        "obs_capacity_preserves_priority": deltas["main_vs_no_capacity_priority_delta_m"] <= float(max_prior_regression_m),
    }
    if no_trust_region is not None:
        deltas.update({
            "trust_region_tail_reduction_m": (
                req("no_trust_region", "residual_endpoint_p99_m")
                - req("main", "residual_endpoint_p99_m")
            ),
            # Positive means the trust region improved OBS error.
            "trust_region_obs_improvement_m": (
                req("no_trust_region", "obs_8s_m") - req("main", "obs_8s_m")
            ),
        })
        checks.update({
            "trust_region_reduces_residual_tail": (
                deltas["trust_region_tail_reduction_m"] >= float(min_trust_tail_reduction_m)
            ),
            "trust_region_does_not_harm_obs": (
                deltas["trust_region_obs_improvement_m"] >= -float(max_trust_obs_regression_m)
            ),
        })
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "metrics": rows,
        "deltas": deltas,
        "thresholds": {
            "min_loss_obs_gain_m": float(min_loss_obs_gain_m),
            "min_loss_overall_gain_m": float(min_loss_overall_gain_m),
            "min_capacity_obs_gain_m": float(min_capacity_obs_gain_m),
            "max_prior_regression_m": float(max_prior_regression_m),
            "min_trust_tail_reduction_m": float(min_trust_tail_reduction_m),
            "max_trust_obs_regression_m": float(max_trust_obs_regression_m),
        },
        "interpretation": (
            "This is the attribution gate. The main CNOB model must beat a same-decoder loss ablation "
            "and a same-loss OBS-capacity ablation; otherwise the corresponding component claim is unsupported."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare controlled v16 natural-stage ablations.")
    ap.add_argument("--main", required=True)
    ap.add_argument("--no-effectiveness-loss", required=True)
    ap.add_argument("--no-obs-capacity", required=True)
    ap.add_argument("--no-trust-region")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-loss-obs-gain-m", type=float, default=0.05)
    ap.add_argument("--min-loss-overall-gain-m", type=float, default=0.02)
    ap.add_argument("--min-capacity-obs-gain-m", type=float, default=0.05)
    ap.add_argument("--max-prior-regression-m", type=float, default=0.15)
    ap.add_argument("--min-trust-tail-reduction-m", type=float, default=5.0)
    ap.add_argument("--max-trust-obs-regression-m", type=float, default=0.15)
    args = ap.parse_args()
    report = compare_reports(
        json.loads(Path(args.main).read_text(encoding="utf-8")),
        json.loads(Path(args.no_effectiveness_loss).read_text(encoding="utf-8")),
        json.loads(Path(args.no_obs_capacity).read_text(encoding="utf-8")),
        (
            json.loads(Path(args.no_trust_region).read_text(encoding="utf-8"))
            if args.no_trust_region else None
        ),
        min_loss_obs_gain_m=args.min_loss_obs_gain_m,
        min_loss_overall_gain_m=args.min_loss_overall_gain_m,
        min_capacity_obs_gain_m=args.min_capacity_obs_gain_m,
        max_prior_regression_m=args.max_prior_regression_m,
        min_trust_tail_reduction_m=args.min_trust_tail_reduction_m,
        max_trust_obs_regression_m=args.max_trust_obs_regression_m,
    )
    report["sources"] = {
        "main": args.main,
        "no_effectiveness_loss": args.no_effectiveness_loss,
        "no_obs_capacity": args.no_obs_capacity,
    }
    if args.no_trust_region:
        report["sources"]["no_trust_region"] = args.no_trust_region
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
