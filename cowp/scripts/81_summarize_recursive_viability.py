from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: str) -> dict[str, Any]:
    obj = json.load(open(path, encoding="utf-8"))
    if "scenario_results" in obj:
        return obj
    ids = [str(x) for x in obj.get("scenario_ids_resolved", [])]
    metrics = list(obj.get("standard_metrics", []))
    diags = list(obj.get("scenario_diagnostics", []))
    obj["scenario_results"] = [
        {
            "scenario_id": sid,
            "standard_metrics": metrics[i] if i < len(metrics) else {},
            "diagnostics": diags[i] if i < len(diags) else {},
        }
        for i, sid in enumerate(ids)
    ]
    return obj


def _metric_bool(row: dict[str, Any], key: str) -> bool:
    return float((row.get("standard_metrics") or {}).get(key, 0.0) or 0.0) > 0.0


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = []
    for row in rows:
        v = (row.get("diagnostics") or {}).get(key)
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            vals.append(x)
    return float(np.mean(vals)) if vals else None


def _summary(obj: dict[str, Any]) -> dict[str, Any]:
    rows = list(obj.get("scenario_results", []))
    coll = [r for r in rows if _metric_bool(r, "CollisionRate")]
    no_coll = [r for r in rows if not _metric_bool(r, "CollisionRate")]
    out: dict[str, Any] = {
        "method": obj.get("method"),
        "num_scenarios": len(rows),
        "standard_metric_summary": obj.get("standard_metric_summary", {}),
        "screen_decomposition": {
            "mean_zero_conventional_step_rate": _mean(rows, "zero_conventional_candidate_step_rate"),
            "mean_zero_conventional_collision_empty_step_rate": _mean(rows, "zero_conventional_collision_empty_step_rate"),
            "mean_zero_conventional_roadgraph_empty_step_rate": _mean(rows, "zero_conventional_roadgraph_empty_step_rate"),
            "mean_zero_conventional_both_empty_step_rate": _mean(rows, "zero_conventional_both_empty_step_rate"),
            "mean_zero_conventional_intersection_empty_step_rate": _mean(rows, "zero_conventional_intersection_empty_step_rate"),
            "mean_valid_candidates": _mean(rows, "mean_valid_candidates"),
            "mean_roadgraph_safe_candidates": _mean(rows, "mean_roadgraph_safe_candidates"),
            "mean_collision_safe_candidates": _mean(rows, "mean_collision_safe_candidates"),
            "mean_conventional_candidates": _mean(rows, "mean_conventional_candidates"),
            "mean_max_collision_safe_prefix_steps": _mean(rows, "mean_max_collision_safe_prefix_steps"),
            "mean_selected_collision_safe_prefix_steps": _mean(rows, "mean_selected_collision_safe_prefix_steps"),
            "mean_recursive_viability_recovery_step_rate": _mean(rows, "recursive_viability_recovery_step_rate"),
        },
        "collision_conditioning": {
            "collision_episodes": len(coll),
            "noncollision_episodes": len(no_coll),
            "collision_mean_zero_conventional_step_rate": _mean(coll, "zero_conventional_candidate_step_rate"),
            "noncollision_mean_zero_conventional_step_rate": _mean(no_coll, "zero_conventional_candidate_step_rate"),
            "collision_mean_max_collision_safe_prefix_steps": _mean(coll, "mean_max_collision_safe_prefix_steps"),
            "noncollision_mean_max_collision_safe_prefix_steps": _mean(no_coll, "mean_max_collision_safe_prefix_steps"),
        },
    }
    return out


def _paired(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    bm = {str(r["scenario_id"]): r for r in base.get("scenario_results", [])}
    om = {str(r["scenario_id"]): r for r in other.get("scenario_results", [])}
    ids = sorted(set(bm) & set(om))
    out: dict[str, Any] = {"paired_scenarios": len(ids)}
    for label, key in (
        ("collision", "CollisionRate"),
        ("offroad", "OffroadRate"),
        ("kinematics", "KinematicsInfeasibilityRate"),
        ("cr", "CR"),
    ):
        b = np.asarray([_metric_bool(bm[s], key) for s in ids], dtype=bool)
        o = np.asarray([_metric_bool(om[s], key) for s in ids], dtype=bool)
        out[label] = {
            "base_rate": float(b.mean()) if len(b) else None,
            "other_rate": float(o.mean()) if len(o) else None,
            "rescued_failures": int((b & ~o).sum()),
            "new_failures": int((~b & o).sum()),
            "shared_failures": int((b & o).sum()),
        }
    ep_delta = []
    for sid in ids:
        a = float((bm[sid].get("standard_metrics") or {}).get("EP", np.nan))
        b = float((om[sid].get("standard_metrics") or {}).get("EP", np.nan))
        if np.isfinite(a) and np.isfinite(b):
            ep_delta.append(b - a)
    if ep_delta:
        d = np.asarray(ep_delta, dtype=np.float64)
        rng = np.random.default_rng(16829)
        boot = np.asarray([rng.choice(d, size=len(d), replace=True).mean() for _ in range(5000)])
        out["ep"] = {
            "paired_finite": int(len(d)),
            "delta_mean": float(d.mean()),
            "delta_median": float(np.median(d)),
            "bootstrap95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        }
    # Directly test the mechanism: on shared no-conventional states, the recursive
    # variant should select a longer causal survival prefix than COWP's unrestricted
    # valid fallback.  This is a descriptive paired diagnostic, not a paper claim.
    diffs = []
    for sid in ids:
        db = bm[sid].get("diagnostics") or {}
        do = om[sid].get("diagnostics") or {}
        xb = db.get("mean_selected_collision_safe_prefix_steps")
        xo = do.get("mean_selected_collision_safe_prefix_steps")
        if xb is None or xo is None:
            continue
        xb, xo = float(xb), float(xo)
        if np.isfinite(xb) and np.isfinite(xo):
            diffs.append(xo - xb)
    if diffs:
        d = np.asarray(diffs, dtype=np.float64)
        out["selected_prefix_delta"] = {
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "positive_scene_rate": float((d > 0).mean()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize v16.8.29 recursive-viability screen decomposition and paired behavior.")
    ap.add_argument("--cowp", required=True)
    ap.add_argument("--recursive", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--development-selected", action="store_true", help="Mark an outcome-enriched development manifest; never treat it as publication evidence.")
    args = ap.parse_args()
    base = _load(args.cowp)
    rec = _load(args.recursive)
    out = {
        "schema_version": "cowp_v16_8_29_recursive_viability_summary_v1",
        "development_selected": bool(args.development_selected),
        "evidence_status": (
            "development_mechanism_probe_only" if args.development_selected else "strict_exact_id_paired_probe"
        ),
        "cowp": _summary(base),
        "recursive_viability": _summary(rec),
        "paired": _paired(base, rec),
    }
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
