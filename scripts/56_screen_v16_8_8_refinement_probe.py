from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen v16.8.8 stable-critical + Priority-Smooth-Yield proposal probe.")
    ap.add_argument("--paired-probe", required=True)
    ap.add_argument("--source-ablation", required=True)
    ap.add_argument("--profile-summary", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    paired = json.loads(Path(args.paired_probe).read_text(encoding="utf-8"))
    gate_mod = importlib.import_module("cowp.scripts.53_gate_fresh_v16_8_6_cache_protocol")
    code_root = Path(__file__).resolve().parents[2]
    code_fingerprint = gate_mod.current_fingerprint(code_root)
    ablation = json.loads(Path(args.source_ablation).read_text(encoding="utf-8"))
    profile = json.loads(Path(args.profile_summary).read_text(encoding="utf-8"))
    new = paired.get("new", {})
    comp = paired.get("pairing_completeness", {})
    psy_inc = ablation.get("priority_smooth_yield_increment", {})
    all_bank = ablation.get("ablations", {}).get("all", {})
    no_psy = ablation.get("ablations", {}).get("without_priority_smooth_yield", {})
    modes = profile.get("critical_selection_reference_modes", {}) or {}
    attempts = profile.get("candidate_attempted_by_source_all_profiled_scenes", {}) or {}
    accepted = profile.get("candidate_accepted_by_source_all_profiled_scenes", {}) or {}
    psy_attempted = int(attempts.get("PRIORITY_SMOOTH_YIELD", 0))
    psy_accepted = int(accepted.get("PRIORITY_SMOOTH_YIELD", 0))
    psy_accept_rate = float(psy_accepted / max(psy_attempted, 1))

    if args.strict:
        th = {
            "min_any_valid": 0.99,
            "min_any_ncf": 0.40,
            "max_false_safe_floor": 0.55,
            "max_pbtr_floor": 0.45,
            "min_hard_recovery": 0.20,
            "min_psy_scene_rate": 0.05,
            "min_psy_priority_ncf_scene_rate": 0.01,
            "min_psy_acceptance_rate": 0.02,
        }
    else:
        # A smoke gate should reject another catastrophic 0.20-NCF bank without
        # pretending that 96 scenes can authorize a multi-day rebuild.
        th = {
            "min_any_valid": 0.99,
            "min_any_ncf": 0.30,
            "max_false_safe_floor": 0.65,
            "max_pbtr_floor": 0.50,
            "min_hard_recovery": 0.12,
            "min_psy_scene_rate": 0.03,
            "min_psy_priority_ncf_scene_rate": 0.005,
            "min_psy_acceptance_rate": 0.02,
        }

    psy_scene = float(new.get("scene_with_priority_smooth_yield_rate", 0.0))
    psy_pncf_scene = float(new.get("scene_with_priority_smooth_yield_priority_ncf_rate", 0.0))
    checks = {
        "pairing_complete": bool(comp.get("complete", False)) and int(comp.get("build_error_count", 0)) == 0,
        "stable_critical_reference": int(modes.get("fixed_anchor_v1", 0)) > 0 and sum(int(v) for v in modes.values()) == int(profile.get("unique_scenarios", 0)),
        "any_valid": float(new.get("any_valid_scene_rate", 0.0)) >= th["min_any_valid"],
        "any_ncf": float(new.get("any_ncf_scene_rate", 0.0)) >= th["min_any_ncf"],
        "false_safe_floor": float(new.get("best_case_selected_false_safe_lower_bound", 1.0)) <= th["max_false_safe_floor"],
        "pbtr_floor": float(new.get("best_case_pbtr_lower_bound", 1.0)) <= th["max_pbtr_floor"],
        "hard_recovery": float(paired.get("paired", {}).get("hard_scene_ncf_recovery_rate", 0.0)) >= th["min_hard_recovery"],
        "psy_generated": psy_scene >= th["min_psy_scene_rate"],
        "psy_yields_priority_ncf": psy_pncf_scene >= th["min_psy_priority_ncf_scene_rate"],
        "psy_acceptance": psy_accept_rate >= th["min_psy_acceptance_rate"],
        # With fixed critical selection and base-bank preservation, adding PSY
        # must not make the bank ceiling worse.  This is an internal monotonicity
        # check, not merely a performance threshold.
        "proposal_union_monotone_any_ncf": float(all_bank.get("any_ncf_scene_rate", 0.0)) + 1e-12 >= float(no_psy.get("any_ncf_scene_rate", 0.0)),
        "proposal_union_monotone_false_safe": float(all_bank.get("best_case_selected_false_safe_lower_bound", 1.0)) <= float(no_psy.get("best_case_selected_false_safe_lower_bound", 1.0)) + 1e-12,
        "proposal_union_monotone_pbtr": float(all_bank.get("best_case_pbtr_lower_bound", 1.0)) <= float(no_psy.get("best_case_pbtr_lower_bound", 1.0)) + 1e-12,
    }
    passed = bool(all(checks.values()))
    if args.strict:
        next_action = (
            "STRICT PASS: full fresh rebuild is justified; run the self-contained v16.8.8 build."
            if passed else
            "STRICT FAIL: do not full-rebuild; inspect source ablation and profile diagnostics."
        )
    else:
        next_action = (
            "SMOKE PASS: run the strict 400-hard + 800-random probe. Do not full-rebuild yet."
            if passed else
            "SMOKE FAIL: do not full-rebuild; refine proposal/critical-set logic before spending more build cost."
        )
    result = {
        "schema_version": "cowp_v16_8_8_refinement_screen_v2",
        "strict": bool(args.strict),
        "code_fingerprint_sha256": code_fingerprint,
        "screen_pass": passed,
        "checks": checks,
        "thresholds": th,
        "observed": {
            "new_any_valid_scene_rate": float(new.get("any_valid_scene_rate", 0.0)),
            "new_any_ncf_scene_rate": float(new.get("any_ncf_scene_rate", 0.0)),
            "new_false_safe_floor": float(new.get("best_case_selected_false_safe_lower_bound", 1.0)),
            "new_pbtr_floor": float(new.get("best_case_pbtr_lower_bound", 1.0)),
            "hard_scene_ncf_recovery_rate": float(paired.get("paired", {}).get("hard_scene_ncf_recovery_rate", 0.0)),
            "psy_scene_rate": psy_scene,
            "psy_priority_ncf_scene_rate": psy_pncf_scene,
            "psy_attempted": psy_attempted,
            "psy_accepted": psy_accepted,
            "psy_acceptance_rate": psy_accept_rate,
            "psy_increment": psy_inc,
            "all_bank": all_bank,
            "without_psy": no_psy,
            "critical_selection_reference_modes": modes,
        },
        "recommend_full_rebuild": bool(args.strict and passed),
        "recommend_strict_probe": bool((not args.strict) and passed),
        "next_action": next_action,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
