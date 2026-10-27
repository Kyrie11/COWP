"""Produce an evidence-first RIDE research status report.

This report is intentionally conservative.  It separates completed simulator
trials from valid paired contrasts, and it never upgrades CARLA/Waymax pilot
outputs into human-causal or strong-baseline claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf8"))


def build_status(*, carla_audit=None, ranking=None, gates=None,
                 womd_fit=None, waymax=None, support_gap=None):
    """Return a JSON-serialisable status object for a frozen evidence panel."""
    cs = (carla_audit or {}).get("summary", {})
    rs = (ranking or {}).get("summary", {})
    ws = (waymax or {}).get("summary", {})
    ss = (support_gap or {}).get("summary", {})

    valid_pairs = int(cs.get("valid_task_matched_contrasts", 0) or 0)
    strict_reversals = int(rs.get("strict_risk_order_reversals", 0) or 0)
    collision_effects = int(cs.get("cross_family_matching_collision_effects", 0) or 0)
    baseline_status = (gates or {}).get("RQ3", "NOT_TESTED_NO_LOCKED_STRONG_BASELINE_CHOICES")

    if valid_pairs == 0:
        rq1 = "NOT_TESTED_VALIDLY"
        rq2 = "NOT_TESTED_VALIDLY"
    else:
        rq1 = ("SIMULATOR_REVERSALS_REQUIRE_PLANNER_AND_EXTERNAL_REPLICATION"
               if strict_reversals else "NO_STRICT_RQ1_REVERSAL_IN_PANEL")
        rq2 = ("SIMULATOR_PAIRED_EFFECT_REQUIRES_EXTERNAL_REPLICATION"
               if collision_effects else "NO_STABLE_RQ2_EFFECT_IN_PANEL")

    fit_summary = (womd_fit or {}).get("summary", {})
    waymax_allowed = bool(ws.get("scientific_claim_allowed", False))
    status = {
        "project": "RIDE",
        "title": "Response-Model-Induced Decision Errors in Interactive Driving",
        "evidence_scope": "pilot_screening_only",
        "rq1": rq1,
        "rq2": rq2,
        "rq3": baseline_status,
        "start_new_planner_algorithm": False,
        "carla": {
            "total_trials": int(cs.get("total_trials", 0) or 0),
            "valid_task_matched_contrasts": valid_pairs,
            "invalid_reasons": cs.get("invalid_reasons", {}),
            "cross_family_matching_collision_effects": collision_effects,
            "strict_risk_order_reversals": strict_reversals,
        },
        "womd_observational_fit": {
            "attempted": int(fit_summary.get("attempted", 0) or 0),
            "eligible": int(fit_summary.get("eligible", 0) or 0),
            "scientific_claim_allowed": bool(fit_summary.get("scientific_claim_allowed", False)),
            "interpretation": "logged_future_conditioning_is_not_do_effect",
        },
        "waymax": {
            "attempted": int(ws.get("attempted", 0) or 0),
            "complete": int(ws.get("complete", 0) or 0),
            "errors": int(ws.get("errors", 0) or 0),
            "all_regimes_matched": int(ws.get("any_all_regime_matched", 0) or 0),
            "scientific_claim_allowed": waymax_allowed,
        },
        "support_gap": {
            "valid_nested_bank_comparisons": int(ss.get("valid_nested_bank_comparisons", 0) or 0),
            "finite_bank_misses": int(ss.get("finite_bank_misses", 0) or 0),
            "global_unsat_proven": bool(ss.get("global_UNSAT_proven", False)),
        },
        "next_gate": "repair_execution_then_repeat_RQ1_RQ2",
        "claim_boundary": [
            "CARLA TM and BehaviorAgent are simulator policies, not human counterfactual truth",
            "WOMD observational fit does not identify a response under do(ego_action)",
            "missing finite-bank candidates are UNKNOWN, never UNSAT",
            "post-hoc outcome ranking is not an online planner choice",
        ],
    }
    return status


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--carla-audit")
    p.add_argument("--ranking")
    p.add_argument("--gates")
    p.add_argument("--womd-fit")
    p.add_argument("--waymax")
    p.add_argument("--support-gap")
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    result = build_status(carla_audit=_load(a.carla_audit), ranking=_load(a.ranking),
                          gates=_load(a.gates), womd_fit=_load(a.womd_fit),
                          waymax=_load(a.waymax), support_gap=_load(a.support_gap))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
