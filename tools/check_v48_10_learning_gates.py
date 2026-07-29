#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v48.10 COPE conditional preference, ordinal evidence, and Natural gate.")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--preference-top1-min", type=float, default=0.10)
    ap.add_argument("--preference-accuracy-min", type=float, default=0.60)
    ap.add_argument("--near-benefit-auc-min", type=float, default=0.70)
    ap.add_argument("--contact-benefit-auc-min", type=float, default=0.75)
    ap.add_argument("--harm-auc-min", type=float, default=0.60)
    args = ap.parse_args()

    variants: dict[str, dict] = {}
    for variant in ("balanced", "precision"):
        base = args.run / "candidates" / variant
        row: dict[str, dict] = {"preference": {}, "evidence": {}, "certificate": {}}
        for regime in ("near", "contact"):
            final = _load(base / "calibration" / f"direct_value_risk_{regime}_v48.json")
            pref = _load(base / "stages" / "conditional_preference" / "preference_audit" / f"preference_{regime}.json")
            top1 = pref.get("unconstrained_group_top1_correlation")
            acc = pref.get("positive_group_top1_accuracy")
            row["preference"][regime] = {
                "top1_correlation": top1,
                "positive_top1_accuracy": acc,
                "positive_top1_regret": pref.get("positive_group_top1_regret_mean"),
                "rank_margin_correctness_auc": pref.get("top1_correctness_rank_margin_auc"),
                "passed": bool(
                    top1 is not None and float(top1) >= args.preference_top1_min
                    and acc is not None and float(acc) >= args.preference_accuracy_min
                ),
            }
            benefit_auc = final.get("policy_top1_positive_auc")
            harm_auc = final.get("policy_top1_harm_auc")
            benefit_min = args.near_benefit_auc_min if regime == "near" else args.contact_benefit_auc_min
            row["evidence"][regime] = {
                "policy_top1_benefit_auc": benefit_auc,
                "policy_top1_harm_auc": harm_auc,
                "candidate_benefit_auc": final.get("candidate_positive_auc"),
                "candidate_harm_auc": final.get("candidate_risk_harm_auc"),
                "evidence_score_teacher_correlation": final.get("candidate_advantage_correlation"),
                "passed": bool(
                    benefit_auc is not None and float(benefit_auc) >= benefit_min
                    and harm_auc is not None and float(harm_auc) >= args.harm_auc_min
                ),
            }
            verify = final.get("verify") or {}
            row["certificate"][regime] = {
                "valid_for_deployment": bool(final.get("valid_for_deployment", False)),
                "verify_selected": verify.get("num_selected"),
                "precision_lcb90": verify.get("precision_wilson_lcb90"),
                "harmful_selected_ucb90": verify.get("harmful_selected_ucb90"),
                "positive_recall": verify.get("positive_recall"),
                "macro_share": verify.get("max_selected_macro_share"),
                "near_miss_verify_frontier": final.get("near_miss_verify_frontier", [])[:5],
                "passed": bool(final.get("valid_for_deployment", False)),
            }
        row["stage_p_passed"] = all(x["passed"] for x in row["preference"].values())
        row["stage_e_discrimination_passed"] = all(x["passed"] for x in row["evidence"].values())
        row["natural_gate_passed"] = all(x["passed"] for x in row["certificate"].values())
        variants[variant] = row

    doc = {
        "version": "v48.10",
        "algorithm": "OC-TRAC-COPE",
        "run": str(args.run),
        "variants": variants,
        "decision": {
            "continue_to_multiseed": any(v["stage_p_passed"] and v["stage_e_discrimination_passed"] for v in variants.values()),
            "continue_to_stress_closed_loop": any(v["natural_gate_passed"] for v in variants.values()),
        },
        "note": "Stage thresholds are diagnostic. Only the unchanged scene-disjoint Natural gate authorizes stress closed-loop.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    if doc["decision"]["continue_to_stress_closed_loop"]:
        return 0
    if doc["decision"]["continue_to_multiseed"]:
        return 10
    return 20


if __name__ == "__main__":
    raise SystemExit(main())
