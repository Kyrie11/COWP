from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def _load(path: str) -> dict:
    p = json.load(open(path, encoding="utf-8"))
    if "scenario_results" in p:
        return p
    ids = [str(x) for x in p.get("scenario_ids_resolved", [])]
    mets = p.get("standard_metrics", [])
    diags = p.get("scenario_diagnostics", [])
    p["scenario_results"] = [
        {
            "scenario_id": sid,
            "standard_metrics": mets[i] if i < len(mets) else {},
            "diagnostics": diags[i] if i < len(diags) else {},
        }
        for i, sid in enumerate(ids)
    ]
    return p


def _logical_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def _integrity(
    base: dict,
    other: dict,
    *,
    expected_ids_sha256: str | None = None,
    expected_count: int | None = None,
) -> dict:
    bi = [str(r["scenario_id"]) for r in base.get("scenario_results", [])]
    oi = [str(r["scenario_id"]) for r in other.get("scenario_results", [])]
    bset, oset = set(bi), set(oi)
    bdecl = base.get("scenario_ids_sha256")
    odecl = other.get("scenario_ids_sha256")
    return {
        "base_n": len(bi),
        "other_n": len(oi),
        "base_unique_n": len(bset),
        "other_unique_n": len(oset),
        "base_has_no_duplicate_ids": len(bi) == len(bset),
        "other_has_no_duplicate_ids": len(oi) == len(oset),
        "same_id_set": bset == oset,
        "same_id_order": bi == oi,
        # These hashes describe result-row order only.  The authoritative exact-ID
        # manifest hash is the declared scenario_ids_sha256 checked below.
        "base_result_order_sha256": _logical_sha(bi),
        "other_result_order_sha256": _logical_sha(oi),
        "declared_base_sha256": bdecl,
        "declared_other_sha256": odecl,
        "same_declared_manifest_sha256": bdecl == odecl,
        "base_declared_matches_expected": expected_ids_sha256 is None or bdecl == expected_ids_sha256,
        "other_declared_matches_expected": expected_ids_sha256 is None or odecl == expected_ids_sha256,
        "base_count_matches_expected": expected_count is None or len(bi) == expected_count,
        "other_count_matches_expected": expected_count is None or len(oi) == expected_count,
        "same_checkpoint": str(base.get("checkpoint")) == str(other.get("checkpoint")),
    }


def _mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _paired(base: dict, other: dict) -> dict:
    bm = {r["scenario_id"]: r for r in base["scenario_results"]}
    om = {r["scenario_id"]: r for r in other["scenario_results"]}
    ids = sorted(set(bm) & set(om))
    out: dict[str, object] = {"paired_scenarios": len(ids)}
    for key in ["CR", "CollisionRate", "OffroadRate", "KinematicsInfeasibilityRate"]:
        b = np.asarray([float(bm[s]["standard_metrics"].get(key, 0)) > 0 for s in ids], bool)
        o = np.asarray([float(om[s]["standard_metrics"].get(key, 0)) > 0 for s in ids], bool)
        worsen = int((~b & o).sum())
        improve = int((b & ~o).sum())
        out[key] = {
            "base_rate": float(b.mean()),
            "other_rate": float(o.mean()),
            "delta": float(o.mean() - b.mean()),
            "base_safe_to_other_fail": worsen,
            "base_fail_to_other_safe": improve,
            "mcnemar_exact_p": _mcnemar_exact(worsen, improve),
        }
    bd, od = [], []
    for sid in ids:
        a = float(bm[sid]["standard_metrics"].get("EP", np.nan))
        b = float(om[sid]["standard_metrics"].get("EP", np.nan))
        if np.isfinite(a) and np.isfinite(b):
            bd.append(a)
            od.append(b)
    if bd:
        delta = np.asarray(od) - np.asarray(bd)
        rng = np.random.default_rng(16829)
        boot = np.asarray([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(5000)])
        out["EP"] = {
            "paired_finite": len(delta),
            "base_mean": float(np.mean(bd)),
            "other_mean": float(np.mean(od)),
            "delta_mean": float(delta.mean()),
            "delta_median": float(np.median(delta)),
            "bootstrap95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "exactly_equal_scenes": int(np.isclose(delta, 0.0, rtol=0.0, atol=1e-12).sum()),
        }
    return out


def _bridge_summary(p: dict) -> dict:
    rows = [r.get("diagnostics", {}) or {} for r in p.get("scenario_results", [])]
    if not rows:
        return {}
    avail = np.asarray([float(r.get("recovery_bridge_available_step_rate", 0.0) or 0.0) for r in rows])
    used = np.asarray([float(r.get("recovery_bridge_step_rate", 0.0) or 0.0) for r in rows])
    no_conv = np.asarray([float(r.get("no_conventional_step_rate", 0.0) or 0.0) for r in rows])
    mean_count = np.asarray([float(r.get("mean_recovery_bridge_candidates", 0.0) or 0.0) for r in rows])
    return {
        "mean_recovery_bridge_available_step_rate": float(avail.mean()),
        "mean_recovery_bridge_used_step_rate": float(used.mean()),
        "episode_with_recovery_available_rate": float((avail > 0).mean()),
        "episode_with_recovery_used_rate": float((used > 0).mean()),
        "mean_no_conventional_step_rate": float(no_conv.mean()),
        "mean_recovery_bridge_candidates": float(mean_count.mean()),
        "bridge_use_fraction_given_no_conventional": float(used.sum() / max(no_conv.sum(), 1e-12)),
    }


def _failure_localization(p: dict) -> dict:
    rows = p.get("scenario_results", [])
    out = {}
    mapping = {"Collision": "CollisionRate", "Offroad": "OffroadRate", "Kinematics": "KinematicsInfeasibilityRate"}
    for label, key in mapping.items():
        pos = [r for r in rows if float(r.get("standard_metrics", {}).get(key, 0)) > 0]
        if not pos:
            continue
        suffix = label.lower()
        reasons: Counter[str] = Counter()
        macros: Counter[str] = Counter()
        recovery_at, fallback_at, conv_at, emergency_at = [], [], [], []
        before = []
        for r in pos:
            d = r.get("diagnostics", {}) or {}
            reasons[str(d.get(f"fallback_reason_at_action_before_first_{suffix}", "none"))] += 1
            macros[str(d.get(f"selected_macro_name_at_action_before_first_{suffix}", "unknown"))] += 1
            recovery_at.append(bool(d.get(f"selected_recovery_bridge_at_action_before_first_{suffix}", False)))
            fallback_at.append(bool(d.get(f"fallback_at_action_before_first_{suffix}", False)))
            conv_at.append(bool(d.get(f"selected_conventional_safe_at_action_before_first_{suffix}", False)))
            emergency_at.append(bool(d.get(f"emergency_action_at_action_before_first_{suffix}", False)))
            if d.get(f"fallback_rate_before_first_{suffix}") is not None:
                before.append(float(d[f"fallback_rate_before_first_{suffix}"]))
        out[label] = {
            "positive_episodes": len(pos),
            "mean_fallback_rate_before_first_event": float(np.mean(before)) if before else None,
            "fallback_immediately_before_first_event_rate": float(np.mean(fallback_at)),
            "recovery_bridge_immediately_before_first_event_rate": float(np.mean(recovery_at)),
            "conventional_safe_immediately_before_first_event_rate": float(np.mean(conv_at)),
            "emergency_immediately_before_first_event_rate": float(np.mean(emergency_at)),
            "fallback_reason_histogram_before_first_event": dict(sorted(reasons.items())),
            "selected_macro_histogram_before_first_event": dict(sorted(macros.items())),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cowp", required=True)
    ap.add_argument("--recovery", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--expected-ids-sha256", default=None)
    ap.add_argument("--expected-count", type=int, default=None)
    args = ap.parse_args()
    base, recovery = _load(args.cowp), _load(args.recovery)
    integrity = _integrity(
        base, recovery,
        expected_ids_sha256=args.expected_ids_sha256,
        expected_count=args.expected_count,
    )
    required = (
        integrity["same_id_set"]
        and integrity["base_has_no_duplicate_ids"]
        and integrity["other_has_no_duplicate_ids"]
        and integrity["same_declared_manifest_sha256"]
        and integrity["base_declared_matches_expected"]
        and integrity["other_declared_matches_expected"]
        and integrity["base_count_matches_expected"]
        and integrity["other_count_matches_expected"]
        and integrity["same_checkpoint"]
    )
    if not required:
        raise SystemExit(f"paired integrity failure: {integrity}")
    out = {
        "schema_version": "cowp_v16_8_29_recovery_viability_compare_v1",
        "integrity": integrity,
        "cowp_vs_recovery_bridge": _paired(base, recovery),
        "cowp_failure_localization": _failure_localization(base),
        "recovery_bridge_usage": _bridge_summary(recovery),
        "recovery_bridge_failure_localization": _failure_localization(recovery),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
