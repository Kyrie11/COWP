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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fail fast when the learned natural-option basis is too inaccurate for root-indexed transport."
    )
    ap.add_argument("--history", required=True, help="history_natural.json from cowp.scripts.03_train")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-set-minade-m", type=float, default=12.0)
    ap.add_argument("--max-branch-minade-m", type=float, default=15.0)
    ap.add_argument("--max-neutral-minade-m", type=float, default=15.0)
    ap.add_argument("--max-priority-minade-m", type=float, default=15.0)
    ap.add_argument("--min-validation-points", type=int, default=1)
    args = ap.parse_args()

    rows = json.loads(Path(args.history).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Natural history must be a JSON list")
    val_rows = [
        r for r in rows
        if isinstance(r, dict) and math.isfinite(_finite(r, "val/natural/traj"))
    ]
    if len(val_rows) < int(args.min_validation_points):
        raise ValueError(
            f"Expected at least {args.min_validation_points} validation rows, found {len(val_rows)}"
        )
    best = min(
        val_rows,
        key=lambda r: (
            _finite(r, "checkpoint/score"),
            _finite(r, "val/natural/traj"),
        ),
    )
    metrics = {
        "set_minade_m": _finite(best, "val/natural/traj"),
        "branch_minade_m": _finite(best, "val/natural/branch_minade"),
        "observed_minade_m": _finite(best, "val/natural/obs_minade"),
        "neutral_minade_m": _finite(best, "val/natural/neutral_minade"),
        "priority_minade_m": _finite(best, "val/natural/prio_minade"),
        "neutral_consistency_m": _finite(best, "val/natural/neutral_consistency"),
        "checkpoint_score": _finite(best, "checkpoint/score"),
        "epoch": int(best.get("epoch", -1)),
    }
    checks = {
        "set_minade_pass": metrics["set_minade_m"] <= args.max_set_minade_m,
        "branch_minade_pass": metrics["branch_minade_m"] <= args.max_branch_minade_m,
        "neutral_minade_pass": metrics["neutral_minade_m"] <= args.max_neutral_minade_m,
        "priority_minade_pass": metrics["priority_minade_m"] <= args.max_priority_minade_m,
    }
    report = {
        "pass": bool(all(checks.values())),
        "metrics": metrics,
        "checks": checks,
        "thresholds": {
            "max_set_minade_m": args.max_set_minade_m,
            "max_branch_minade_m": args.max_branch_minade_m,
            "max_neutral_minade_m": args.max_neutral_minade_m,
            "max_priority_minade_m": args.max_priority_minade_m,
        },
        "validation_points": len(val_rows),
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
