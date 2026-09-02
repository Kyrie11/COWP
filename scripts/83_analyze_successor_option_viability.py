from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text())


def _rows(d: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(r["scenario_id"]): r for r in d.get("scenario_results", [])}


def _value(row: dict[str, Any], key: str) -> float:
    return float((row.get("standard_metrics", {}) or {}).get(key, 0.0) or 0.0)


def _event(row: dict[str, Any], key: str) -> bool:
    return _value(row, key) > 0.0


def _mcnemar_exact(rescued: int, induced: int) -> float:
    n = int(rescued + induced)
    if n <= 0:
        return 1.0
    k = min(int(rescued), int(induced))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def _bootstrap_mean_ci(x: np.ndarray, seed: int = 16830, draws: int = 10000) -> list[float] | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        vals[i] = float(rng.choice(x, size=x.size, replace=True).mean())
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def _aggregate(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    event_keys = {
        "CR": "CR",
        "Collision": "CollisionRate",
        "Offroad": "OffroadRate",
        "Kinematics": "KinematicsInfeasibilityRate",
    }
    out: dict[str, Any] = {"n": len(ids)}
    for label, key in event_keys.items():
        out[label] = float(np.mean([_event(rows[s], key) for s in ids])) if ids else 0.0
    ep = np.asarray([_value(rows[s], "EP") for s in ids], dtype=np.float64)
    ep = ep[np.isfinite(ep)]
    out["EP"] = float(ep.mean()) if ep.size else None
    diag_keys = [
        "fallback_step_rate",
        "zero_conventional_candidate_step_rate",
        "mean_conventional_candidates",
        "mean_max_collision_safe_prefix_steps",
        "mean_selected_collision_safe_prefix_steps",
        "recovery_switch_step_rate",
        "successor_option_probe_step_rate",
        "mean_successor_signature_compare_on_probes",
        "mean_recovery_prefix_gain_steps",
        "mean_recovery_action_risk_delta",
        "mean_recovery_rule_risk_delta",
        "mean_recovery_pressure_risk_delta",
    ]
    for key in diag_keys:
        vals = []
        for s in ids:
            v = (rows[s].get("diagnostics", {}) or {}).get(key)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                vals.append(float(v))
        out[key] = float(np.mean(vals)) if vals else None
    return out


def _paired(base: dict[str, dict[str, Any]], alt: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    event_keys = {
        "CR": "CR",
        "Collision": "CollisionRate",
        "Offroad": "OffroadRate",
        "Kinematics": "KinematicsInfeasibilityRate",
    }
    for label, key in event_keys.items():
        rescued = [s for s in ids if _event(base[s], key) and not _event(alt[s], key)]
        induced = [s for s in ids if not _event(base[s], key) and _event(alt[s], key)]
        shared = [s for s in ids if _event(base[s], key) and _event(alt[s], key)]
        out[label] = {
            "rescued": len(rescued),
            "induced": len(induced),
            "shared_failure": len(shared),
            "net_failures_removed": len(rescued) - len(induced),
            "mcnemar_exact_p": _mcnemar_exact(len(rescued), len(induced)),
            "rescued_ids": rescued,
            "induced_ids": induced,
        }
    dep = []
    for s in ids:
        a = _value(base[s], "EP")
        b = _value(alt[s], "EP")
        if math.isfinite(a) and math.isfinite(b):
            dep.append(b - a)
    arr = np.asarray(dep, dtype=np.float64)
    out["EP"] = {
        "paired_n": int(arr.size),
        "delta_mean": float(arr.mean()) if arr.size else None,
        "delta_median": float(np.median(arr)) if arr.size else None,
        "bootstrap95": _bootstrap_mean_ci(arr),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cowp", required=True)
    ap.add_argument("--rvr")
    ap.add_argument("--guard")
    ap.add_argument("--successor")
    ap.add_argument("--development-selected", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = {"cowp": _load(args.cowp), "rvr": _load(args.rvr), "guard": _load(args.guard), "successor": _load(args.successor)}
    data = {k: v for k, v in data.items() if v is not None}
    rowmap = {k: _rows(v) for k, v in data.items()}
    ids = list(rowmap["cowp"].keys())
    idset = set(ids)
    mismatched = {k: sorted(idset.symmetric_difference(set(rows))) for k, rows in rowmap.items() if set(rows) != idset}
    if mismatched:
        raise SystemExit(f"scenario set mismatch: {mismatched}")

    out: dict[str, Any] = {
        "development_selected": bool(args.development_selected),
        "paper_evidence": False if args.development_selected else None,
        "scenario_count": len(ids),
        "methods": {},
        "paired_vs_cowp": {},
    }
    for name, rows in rowmap.items():
        out["methods"][name] = _aggregate(rows, ids)
        if name != "cowp":
            out["paired_vs_cowp"][name] = _paired(rowmap["cowp"], rows, ids)

    if "rvr" in rowmap:
        old_rescued = set(out["paired_vs_cowp"]["rvr"]["Collision"]["rescued_ids"])
        old_induced = set(out["paired_vs_cowp"]["rvr"]["Collision"]["induced_ids"])
        out["v16_8_29_counterexample_retention"] = {}
        for name in ("guard", "successor"):
            if name not in rowmap:
                continue
            rows = rowmap[name]
            retained = sum(not _event(rows[s], "CollisionRate") for s in old_rescued)
            avoided = sum(not _event(rows[s], "CollisionRate") for s in old_induced)
            out["v16_8_29_counterexample_retention"][name] = {
                "old_rvr_rescues_total": len(old_rescued),
                "old_rvr_rescues_retained": int(retained),
                "old_rvr_induced_total": len(old_induced),
                "old_rvr_induced_avoided": int(avoided),
            }

    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
