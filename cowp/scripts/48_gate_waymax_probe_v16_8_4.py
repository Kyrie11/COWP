#!/usr/bin/env python3
"""Promotion gate from the small paired Waymax probe to expensive full rollout.

This gate is intentionally *not* a publication gate.  It answers a narrower
engineering/scientific question: did COWP survive the small paired closed-loop
probe without obvious safety/progress regression or fallback collapse?  Full
Waymax is blocked when the answer is no.

The gate consumes the two raw JSON outputs produced by 04_eval_closed_loop, not
only the summarized delta file, so missing diagnostics cannot be mistaken for a
zero delta.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _nested(d: dict[str, Any], section: str, key: str) -> float | None:
    x = d.get(section, {})
    if isinstance(x, dict):
        return _finite_float(x.get(key))
    return None


def _first_metric(d: dict[str, Any], candidates: list[tuple[str, str]]) -> tuple[str | None, float | None]:
    for section, key in candidates:
        v = _nested(d, section, key)
        if v is not None:
            return f"{section}/{key}", v
    return None, None


def _load(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, type=Path, help="conventional_safety Waymax probe JSON")
    ap.add_argument("--candidate", required=True, type=Path, help="COWP Waymax probe JSON")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--min-rollouts", type=int, default=80)
    ap.add_argument("--max-collision-regression", type=float, default=0.08)
    ap.add_argument("--max-offroad-regression", type=float, default=0.08)
    ap.add_argument("--max-kinematics-regression", type=float, default=0.08)
    ap.add_argument("--max-progress-regression", type=float, default=0.08)
    ap.add_argument("--max-fallback-episode-rate", type=float, default=0.50)
    ap.add_argument("--max-pred-fsr-regression", type=float, default=0.05)
    ap.add_argument("--max-pred-cbs-regression", type=float, default=0.10)
    ap.add_argument("--max-pred-opr-drop", type=float, default=0.05)
    args = ap.parse_args()

    ref = _load(args.reference)
    cand = _load(args.candidate)
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    nr = int(ref.get("num_rollouts", 0) or 0)
    nc = int(cand.get("num_rollouts", 0) or 0)
    checks["num_rollouts"] = {"reference": nr, "candidate": nc, "minimum": int(args.min_rollouts)}
    if nr < args.min_rollouts or nc < args.min_rollouts:
        failures.append("insufficient completed probe rollouts")
    if nr != nc:
        failures.append("paired probe rollout counts differ")

    ref_policy = str(ref.get("actual_non_ego_policy", ""))
    cand_policy = str(cand.get("actual_non_ego_policy", ""))
    checks["actual_non_ego_policy"] = {"reference": ref_policy, "candidate": cand_policy}
    if ref_policy != cand_policy:
        failures.append("reference/candidate non-ego policies differ")

    # Waymax standard safety.  Older adapters may report CR while newer ones may
    # also expose CollisionRate, so accept either but require one collision metric.
    col_key_r, col_r = _first_metric(ref, [("standard_metric_summary", "CR"), ("standard_metric_summary", "CollisionRate")])
    col_key_c, col_c = _first_metric(cand, [("standard_metric_summary", "CR"), ("standard_metric_summary", "CollisionRate")])
    checks["collision"] = {"reference_key": col_key_r, "candidate_key": col_key_c, "reference": col_r, "candidate": col_c, "max_regression": args.max_collision_regression}
    if col_r is None or col_c is None:
        failures.append("missing Waymax collision metric in probe JSON")
    elif col_c > col_r + args.max_collision_regression:
        failures.append("collision regression exceeds probe tolerance")

    def paired_lower_is_better(name: str, key: str, tolerance: float, required: bool = False) -> None:
        rv = _nested(ref, "standard_metric_summary", key)
        cv = _nested(cand, "standard_metric_summary", key)
        checks[name] = {"reference": rv, "candidate": cv, "max_regression": tolerance}
        if rv is None or cv is None:
            (failures if required else warnings).append(f"missing {key} metric")
        elif cv > rv + tolerance:
            failures.append(f"{key} regression exceeds probe tolerance")

    paired_lower_is_better("offroad", "OffroadRate", args.max_offroad_regression)
    paired_lower_is_better("kinematics", "KinematicsInfeasibilityRate", args.max_kinematics_regression)

    ep_r = _nested(ref, "standard_metric_summary", "EP")
    ep_c = _nested(cand, "standard_metric_summary", "EP")
    checks["progress"] = {"reference": ep_r, "candidate": ep_c, "max_regression": args.max_progress_regression}
    if ep_r is None or ep_c is None:
        failures.append("missing Waymax EP/progress metric in probe JSON")
    elif ep_c < ep_r - args.max_progress_regression:
        failures.append("EP/progress regression exceeds probe tolerance")

    fallback = _nested(cand, "closed_loop_cowp_metric_summary", "FallbackEpisodeRate")
    diag_eps = _nested(cand, "closed_loop_cowp_metric_summary", "EpisodesWithDiagnostics")
    checks["fallback"] = {"candidate": fallback, "episodes_with_diagnostics": diag_eps, "maximum": args.max_fallback_episode_rate}
    if fallback is None or diag_eps is None or diag_eps <= 0:
        failures.append("missing COWP episode diagnostics/fallback metric")
    elif fallback > args.max_fallback_episode_rate:
        failures.append("COWP fallback episode rate is too high for full rollout promotion")

    # Mechanism proxies are only enforced when present for both methods.  They are
    # predictions rather than counterfactual ground truth in logged-replay Waymax,
    # so this gate prevents obvious regression but does not elevate them to claims.
    fs_r = _nested(ref, "closed_loop_cowp_metric_summary", "PredFSR_episode")
    fs_c = _nested(cand, "closed_loop_cowp_metric_summary", "PredFSR_episode")
    checks["predicted_false_safe"] = {"reference": fs_r, "candidate": fs_c, "max_regression": args.max_pred_fsr_regression}
    if fs_r is not None and fs_c is not None and fs_c > fs_r + args.max_pred_fsr_regression:
        failures.append("predicted false-safe episode rate regresses")
    elif fs_r is None or fs_c is None:
        warnings.append("PredFSR_episode unavailable for one/both methods; skipped as promotion check")

    cbs_r = _nested(ref, "closed_loop_cowp_metric_summary", "PredCBS_episode")
    cbs_c = _nested(cand, "closed_loop_cowp_metric_summary", "PredCBS_episode")
    checks["predicted_burden"] = {"reference": cbs_r, "candidate": cbs_c, "max_regression": args.max_pred_cbs_regression}
    if cbs_r is not None and cbs_c is not None and cbs_c > cbs_r + args.max_pred_cbs_regression:
        failures.append("predicted burden proxy regresses")
    elif cbs_r is None or cbs_c is None:
        warnings.append("PredCBS_episode unavailable for one/both methods; skipped as promotion check")

    opr_r = _nested(ref, "closed_loop_cowp_metric_summary", "PredOPR_min_episode")
    opr_c = _nested(cand, "closed_loop_cowp_metric_summary", "PredOPR_min_episode")
    checks["predicted_opr"] = {"reference": opr_r, "candidate": opr_c, "max_drop": args.max_pred_opr_drop}
    if opr_r is not None and opr_c is not None and opr_c < opr_r - args.max_pred_opr_drop:
        failures.append("predicted OPR proxy drops")
    elif opr_r is None or opr_c is None:
        warnings.append("PredOPR_min_episode unavailable for one/both methods; skipped as promotion check")

    report = {
        "schema_version": "cowp_v16_8_4_waymax_probe_promotion_v1",
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
        "pass": not failures,
        "interpretation": (
            "PASS only authorizes the expensive full logged-replay Waymax run. "
            "It is not publication evidence for causal burden transfer; the paper still requires "
            "reactive-agent and human-audited stress-set validation for that claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
