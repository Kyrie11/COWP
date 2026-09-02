from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite(row: dict, key: str, default: float = math.inf) -> float:
    try:
        x = float(row.get(key, default))
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _improvement(rows: list[dict], key: str) -> float | None:
    vals = [_finite(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) < 2:
        return None
    return float(vals[0] - min(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail fast when the learned natural-option basis is unusable for root-indexed transport.")
    ap.add_argument("--history", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--oracle-report", default=None, help="Optional output of 34_diagnose_natural_oracles.py")
    ap.add_argument("--max-oracle-gap-m", type=float, default=6.0)
    ap.add_argument("--max-set-minade-m", type=float, default=12.0)
    ap.add_argument("--max-branch-minade-m", type=float, default=15.0)
    ap.add_argument("--max-observed-minade-m", type=float, default=4.0)
    ap.add_argument("--max-branch-spread-m", type=float, default=3.0)
    ap.add_argument("--max-neutral-minade-m", type=float, default=15.0)
    ap.add_argument("--max-priority-minade-m", type=float, default=15.0)
    ap.add_argument("--max-minade-1s-m", type=float, default=3.0)
    ap.add_argument("--max-minade-3s-m", type=float, default=8.0)
    ap.add_argument("--min-source-improvement", type=float, default=0.01)
    ap.add_argument("--min-priority-improvement", type=float, default=0.005)
    ap.add_argument("--min-neutral-consistency-improvement-m", type=float, default=0.5)
    ap.add_argument("--max-source-ce", type=float, default=0.30, help="Absolute semantic gate; typed roots may already be correct at epoch -1 and need not improve further.")
    ap.add_argument("--max-priority-bce", type=float, default=0.45)
    ap.add_argument("--max-neutral-consistency-m", type=float, default=10.0)
    ap.add_argument("--max-typed-untyped-gap-m", type=float, default=4.0, help="Typed matching must not hide severe loss of geometric coverage.")
    ap.add_argument("--min-validation-points", type=int, default=2)
    args = ap.parse_args()

    rows = json.loads(Path(args.history).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Natural history must be a JSON list")
    val_rows = [r for r in rows if isinstance(r, dict) and math.isfinite(_finite(r, "val/natural/traj"))]
    if len(val_rows) < int(args.min_validation_points):
        raise ValueError(f"Expected at least {args.min_validation_points} validation rows, found {len(val_rows)}")
    best = min(val_rows, key=lambda r: (_finite(r, "checkpoint/score"), _finite(r, "val/natural/traj")))
    metrics = {
        "set_minade_m": _finite(best, "val/natural/traj"),
        "untyped_set_minade_m": _finite(best, "val/natural/untyped_traj"),
        "minade_1s_m": _finite(best, "val/natural/minade_1s"),
        "minade_3s_m": _finite(best, "val/natural/minade_3s"),
        "minade_5s_m": _finite(best, "val/natural/minade_5s"),
        "branch_minade_m": _finite(best, "val/natural/branch_minade"),
        "observed_minade_m": _finite(best, "val/natural/obs_minade"),
        "neutral_minade_m": _finite(best, "val/natural/neutral_minade"),
        "priority_minade_m": _finite(best, "val/natural/prio_minade"),
        "neutral_consistency_m": _finite(best, "val/natural/neutral_consistency"),
        "source_ce": _finite(best, "val/natural/source"),
        "priority_bce": _finite(best, "val/natural/priority"),
        "source_improvement": _improvement(val_rows, "val/natural/source"),
        "priority_improvement": _improvement(val_rows, "val/natural/priority"),
        "neutral_consistency_improvement_m": _improvement(val_rows, "val/natural/neutral_consistency"),
        "checkpoint_score": _finite(best, "checkpoint/score"),
        "epoch": int(best.get("epoch", -1)),
    }

    effective_set_gate = float(args.max_set_minade_m)
    oracle = None
    if args.oracle_report:
        oracle = json.loads(Path(args.oracle_report).read_text(encoding="utf-8"))
        oracle_mean = oracle.get("interpretation", {}).get("oracle_8s_mean_m")
        try:
            oracle_mean = float(oracle_mean)
            if math.isfinite(oracle_mean):
                effective_set_gate = min(effective_set_gate, oracle_mean + float(args.max_oracle_gap_m))
        except Exception:
            oracle_mean = None
    branch_vals = [
        metrics["observed_minade_m"], metrics["neutral_minade_m"], metrics["priority_minade_m"]
    ]
    finite_branch_vals = [x for x in branch_vals if math.isfinite(x)]
    branch_spread = (max(finite_branch_vals) - min(finite_branch_vals)) if finite_branch_vals else math.inf
    metrics["branch_spread_m"] = float(branch_spread)
    metrics["source_identity_is_structural"] = True
    checks = {
        "set_minade_pass": metrics["set_minade_m"] <= effective_set_gate,
        "branch_minade_pass": metrics["branch_minade_m"] <= args.max_branch_minade_m,
        "observed_minade_pass": metrics["observed_minade_m"] <= args.max_observed_minade_m,
        "branch_spread_pass": branch_spread <= args.max_branch_spread_m,
        "neutral_minade_pass": metrics["neutral_minade_m"] <= args.max_neutral_minade_m,
        "priority_minade_pass": metrics["priority_minade_m"] <= args.max_priority_minade_m,
        # Source IDs are permanent typed roots in v15.  Source CE is retained as
        # a diagnostic only; it cannot be used as evidence that semantics were
        # learned from data.
        "structural_source_identity_pass": True,
        "priority_semantics_pass": (
            metrics["priority_bce"] <= args.max_priority_bce
            or (metrics["priority_improvement"] or 0.0) >= args.min_priority_improvement
        ),
        "neutral_consistency_pass": (
            metrics["neutral_consistency_m"] <= args.max_neutral_consistency_m
            or (metrics["neutral_consistency_improvement_m"] or 0.0) >= args.min_neutral_consistency_improvement_m
        ),
    }
    if math.isfinite(metrics["untyped_set_minade_m"]):
        checks["typed_untyped_gap_pass"] = (
            metrics["set_minade_m"] - metrics["untyped_set_minade_m"]
        ) <= args.max_typed_untyped_gap_m

    # Older histories do not contain horizon metrics; do not fabricate a failure.
    if math.isfinite(metrics["minade_1s_m"]):
        checks["minade_1s_pass"] = metrics["minade_1s_m"] <= args.max_minade_1s_m
    if math.isfinite(metrics["minade_3s_m"]):
        checks["minade_3s_pass"] = metrics["minade_3s_m"] <= args.max_minade_3s_m

    report = {
        "pass": bool(all(checks.values())),
        "metrics": metrics,
        "checks": checks,
        "thresholds": {
            "max_set_minade_m": args.max_set_minade_m,
            "effective_set_minade_gate_m": effective_set_gate,
            "max_oracle_gap_m": args.max_oracle_gap_m,
            "max_branch_minade_m": args.max_branch_minade_m,
            "max_observed_minade_m": args.max_observed_minade_m,
            "max_branch_spread_m": args.max_branch_spread_m,
            "max_neutral_minade_m": args.max_neutral_minade_m,
            "max_priority_minade_m": args.max_priority_minade_m,
            "max_minade_1s_m": args.max_minade_1s_m,
            "max_minade_3s_m": args.max_minade_3s_m,
            "min_source_improvement": args.min_source_improvement,
            "min_priority_improvement": args.min_priority_improvement,
            "min_neutral_consistency_improvement_m": args.min_neutral_consistency_improvement_m,
            "max_source_ce": args.max_source_ce,
            "max_priority_bce": args.max_priority_bce,
            "max_neutral_consistency_m": args.max_neutral_consistency_m,
            "max_typed_untyped_gap_m": args.max_typed_untyped_gap_m,
        },
        "oracle_report": args.oracle_report,
        "validation_points": len(val_rows),
        "failure_interpretation": "Do not continue to transport/planner training when this gate fails. v15 treats source identity as structural and requires geometric quality in every branch, especially OBS; near-zero source CE is not counted as learned novelty.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
