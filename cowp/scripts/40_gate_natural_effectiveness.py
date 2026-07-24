from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mean(report: dict, key: str) -> float | None:
    row = report.get("distributions", {}).get(key, {})
    value = row.get("weighted_mean", row.get("mean"))
    return None if value is None else float(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hard gate proving that the learned natural residual is useful and physical.")
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
    ap.add_argument("--max-residual-endpoint-p99-m", type=float, default=25.0)
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
        "residual_endpoint_p99_m": src.get("distributions", {}).get("residual/endpoint_m", {}).get("p99"),
    }
    eff = [float(v.get("effective_modes", 0.0)) for v in src.get("mode_usage", {}).values()]
    metrics["min_effective_modes"] = min(eff) if eff else 0.0

    def present(name: str) -> float:
        value = metrics.get(name)
        if value is None:
            raise ValueError(f"Required learned-natural metric is absent: {name}")
        return float(value)

    checks = {
        "learned_absolute_quality": present("learned_8s_m") <= args.max_learned_8s_m,
        "obs_absolute_quality": present("obs_8s_m") <= args.max_obs_8s_m,
        "residual_improves_overall": present("overall_gain_8s_m") >= args.min_overall_gain_8s_m,
        "residual_improves_obs": present("obs_gain_8s_m") >= args.min_obs_gain_8s_m,
        "neutral_prior_preserved": present("neutral_gain_8s_m") >= -args.max_neutral_degradation_8s_m,
        "priority_prior_preserved": present("priority_gain_8s_m") >= -args.max_priority_degradation_8s_m,
        "velocity_consistency": present("velocity_error_mps") <= args.max_velocity_error_mps,
        "yaw_consistency": present("yaw_error_rad") <= args.max_yaw_error_rad,
        "mode_bank_is_used": present("min_effective_modes") >= args.min_effective_modes,
        "residual_is_bounded": present("residual_endpoint_p99_m") <= args.max_residual_endpoint_p99_m,
    }
    report = {
        "pass": bool(all(checks.values())),
        "metrics": metrics,
        "checks": checks,
        "thresholds": vars(args) | {"report": None, "output": None},
        "source_report": args.report,
        "failure_interpretation": (
            "A failure means the decoder/loss/capacity claim is not supported. Do not continue to transport/planner "
            "merely because the absolute natural gate passed."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
