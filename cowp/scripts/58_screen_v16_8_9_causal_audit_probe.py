from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen v16.8.9 candidate-conditioned causal-audit proposal/data probe.")
    ap.add_argument("--paired-probe", required=True)
    ap.add_argument("--source-ablation", required=True)
    ap.add_argument("--profile-summary", required=True)
    ap.add_argument("--audit-diagnostic", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    paired = json.loads(Path(args.paired_probe).read_text(encoding="utf-8"))
    ablation = json.loads(Path(args.source_ablation).read_text(encoding="utf-8"))
    profile = json.loads(Path(args.profile_summary).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.audit_diagnostic).read_text(encoding="utf-8"))
    gate_mod = importlib.import_module("cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol")
    code_fingerprint = gate_mod.current_fingerprint(Path(__file__).resolve().parents[2])

    new = paired.get("new", {})
    comp = paired.get("pairing_completeness", {})
    modes = profile.get("critical_selection_reference_modes", {}) or {}
    pair_rates = audit.get("pair_rates", {}) or {}
    integ = audit.get("integrity", {}) or {}
    root_counts = audit.get("root_counts", {}) or {}
    all_bank = ablation.get("ablations", {}).get("all", {})
    no_psy = ablation.get("ablations", {}).get("without_priority_smooth_yield", {})

    if args.strict:
        th = dict(min_any_valid=0.99, min_any_ncf=0.40, max_false_safe_floor=0.55,
                  max_pbtr_floor=0.45, min_hard_recovery=0.20,
                  min_relevant_pair_rate=0.01, max_relevant_pair_rate=0.95,
                  min_burden_only_root_fraction=0.002)
    else:
        th = dict(min_any_valid=0.99, min_any_ncf=0.30, max_false_safe_floor=0.65,
                  max_pbtr_floor=0.50, min_hard_recovery=0.12,
                  min_relevant_pair_rate=0.01, max_relevant_pair_rate=0.95,
                  min_burden_only_root_fraction=0.001)

    relevant_rate = float(pair_rates.get("relevant", 0.0))
    burden_only_fraction = float(pair_rates.get("burden_only_root_fraction", 0.0))
    unique_scenes = int(profile.get("unique_scenarios", 0))
    checks = {
        "pairing_complete": bool(comp.get("complete", False)) and int(comp.get("build_error_count", 0)) == 0,
        "stable_critical_reference": int(modes.get("fixed_anchor_v1", 0)) == unique_scenes and unique_scenes > 0,
        "any_valid": float(new.get("any_valid_scene_rate", 0.0)) >= th["min_any_valid"],
        "any_ncf": float(new.get("any_ncf_scene_rate", 0.0)) >= th["min_any_ncf"],
        "false_safe_floor": float(new.get("best_case_selected_false_safe_lower_bound", 1.0)) <= th["max_false_safe_floor"],
        "pbtr_floor": float(new.get("best_case_pbtr_lower_bound", 1.0)) <= th["max_pbtr_floor"],
        "hard_recovery": float(paired.get("paired", {}).get("hard_scene_ncf_recovery_rate", 0.0)) >= th["min_hard_recovery"],
        "audit_not_degenerate": th["min_relevant_pair_rate"] <= relevant_rate <= th["max_relevant_pair_rate"],
        "burden_only_affected_signal_present": burden_only_fraction >= th["min_burden_only_root_fraction"],
        "no_read_errors": bool(integ.get("no_read_errors", False)),
        "no_silent_blockers": bool(integ.get("no_silent_blockers", False)),
        "no_irrelevant_blockers": bool(integ.get("no_irrelevant_blockers", False)),
        "transport_affected_matches_audit": bool(integ.get("transport_affected_matches_audit", False)),
        "no_responses_for_irrelevant_pairs": bool(integ.get("no_responses_for_irrelevant_pairs", False)),
        "proposal_union_monotone_any_ncf": float(all_bank.get("any_ncf_scene_rate", 0.0)) + 1e-12 >= float(no_psy.get("any_ncf_scene_rate", 0.0)),
        "proposal_union_monotone_false_safe": float(all_bank.get("best_case_selected_false_safe_lower_bound", 1.0)) <= float(no_psy.get("best_case_selected_false_safe_lower_bound", 1.0)) + 1e-12,
        "proposal_union_monotone_pbtr": float(all_bank.get("best_case_pbtr_lower_bound", 1.0)) <= float(no_psy.get("best_case_pbtr_lower_bound", 1.0)) + 1e-12,
    }
    passed = all(checks.values())
    if args.strict:
        next_action = (
            "STRICT PASS: v16.8.9 data semantics and proposal ceiling justify a full fresh rebuild."
            if passed else
            "STRICT FAIL: do not full-rebuild; inspect audit/blocker/source diagnostics first."
        )
    else:
        next_action = (
            "SMOKE PASS: run the 400-hard + 800-random v16.8.9 strict probe; do not full-rebuild yet."
            if passed else
            "SMOKE FAIL: do not full-rebuild; the causal-audit/data contract still needs repair."
        )

    result = {
        "schema_version": "cowp_v16_8_9_causal_audit_screen_v1",
        "strict": bool(args.strict),
        "code_fingerprint_sha256": code_fingerprint,
        "screen_pass": bool(passed),
        "checks": checks,
        "thresholds": th,
        "observed": {
            "new_any_valid_scene_rate": float(new.get("any_valid_scene_rate", 0.0)),
            "new_any_ncf_scene_rate": float(new.get("any_ncf_scene_rate", 0.0)),
            "new_false_safe_floor": float(new.get("best_case_selected_false_safe_lower_bound", 1.0)),
            "new_pbtr_floor": float(new.get("best_case_pbtr_lower_bound", 1.0)),
            "hard_scene_ncf_recovery_rate": float(paired.get("paired", {}).get("hard_scene_ncf_recovery_rate", 0.0)),
            "audit_pair_rates": pair_rates,
            "audit_root_counts": root_counts,
            "audit_integrity": integ,
            "critical_selection_reference_modes": modes,
            "all_bank": all_bank,
            "without_psy": no_psy,
        },
        "recommend_strict_probe": bool((not args.strict) and passed),
        "recommend_full_rebuild": bool(args.strict and passed),
        "next_action": next_action,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
