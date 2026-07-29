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
    ap = argparse.ArgumentParser(description="Audit v48.9 PACER learning stages before multiseed or closed-loop promotion.")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--preference-top1-min", type=float, default=0.10)
    ap.add_argument("--preference-accuracy-min", type=float, default=0.60)
    ap.add_argument("--false-switch-max", type=float, default=0.45)
    ap.add_argument("--near-policy-auc-min", type=float, default=0.70)
    ap.add_argument("--contact-policy-auc-min", type=float, default=0.75)
    ap.add_argument("--policy-harm-auc-min", type=float, default=0.60)
    args = ap.parse_args()

    variants: dict[str, dict] = {}
    for variant in ("balanced", "precision"):
        base = args.run / "candidates" / variant
        row: dict[str, dict] = {"preference": {}, "gain": {}, "certificate": {}}
        for regime in ("near", "contact"):
            final = _load(base / "calibration" / f"direct_value_risk_{regime}_v48.json")
            pref = _load(base / "stages" / "preference" / "preference_audit" / f"preference_{regime}.json")
            top1 = pref.get("unconstrained_group_top1_correlation")
            acc = pref.get("positive_group_top1_accuracy")
            false_switch = pref.get("nonpositive_group_false_switch_rate")
            pref_pass = bool(
                top1 is not None and float(top1) >= args.preference_top1_min
                and acc is not None and float(acc) >= args.preference_accuracy_min
                and false_switch is not None and float(false_switch) <= args.false_switch_max
            )
            row["preference"][regime] = {
                "top1_correlation": top1,
                "acceptable_top1_accuracy": acc,
                "nonpositive_group_false_switch_rate": false_switch,
                "harmful_ranked_switch_rate": pref.get("harmful_ranked_switch_rate"),
                "positive_group_recovery_activation_rate": pref.get("positive_group_recovery_activation_rate"),
                "passed": pref_pass,
            }

            policy_auc = final.get("policy_top1_positive_auc")
            harm_auc = final.get("policy_top1_harm_auc")
            auc_min = args.near_policy_auc_min if regime == "near" else args.contact_policy_auc_min
            gain_pass = bool(
                policy_auc is not None and float(policy_auc) >= auc_min
                and harm_auc is not None and float(harm_auc) >= args.policy_harm_auc_min
            )
            row["gain"][regime] = {
                "policy_top1_positive_auc": policy_auc,
                "policy_top1_harm_auc": harm_auc,
                "policy_top1_gain_mae": final.get("policy_top1_gain_mae"),
                "candidate_positive_auc": final.get("candidate_positive_auc"),
                "candidate_harm_auc": final.get("candidate_risk_harm_auc"),
                "mean_regret": final.get("positive_group_top1_regret_mean"),
                "passed": gain_pass,
            }

            verify = final.get("verify") or {}
            row["certificate"][regime] = {
                "valid_for_deployment": bool(final.get("valid_for_deployment", False)),
                "verify_selected": verify.get("num_selected"),
                "precision_lcb90": verify.get("precision_wilson_lcb90"),
                "harmful_selected_ucb90": verify.get("harmful_selected_ucb90"),
                "positive_recall": verify.get("positive_recall"),
                "passed": bool(final.get("valid_for_deployment", False)),
                "near_miss_verify_frontier": final.get("near_miss_verify_frontier", [])[:5],
            }

        row["stage_p_passed"] = all(x["passed"] for x in row["preference"].values())
        row["stage_c_discrimination_passed"] = all(x["passed"] for x in row["gain"].values())
        row["natural_gate_passed"] = all(x["passed"] for x in row["certificate"].values())
        variants[variant] = row

    doc = {
        "version": "v48.9",
        "algorithm": "OC-TRAC-PACER",
        "run": str(args.run),
        "variants": variants,
        "decision": {
            "continue_to_multiseed": any(
                v["stage_p_passed"] and v["stage_c_discrimination_passed"]
                for v in variants.values()
            ),
            "continue_to_stress_closed_loop": any(v["natural_gate_passed"] for v in variants.values()),
        },
        "note": "Learning-stage thresholds do not authorize deployment; only the unchanged held-out Natural gate can do so.",
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
