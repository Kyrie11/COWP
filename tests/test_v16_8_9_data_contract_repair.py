from __future__ import annotations

import numpy as np

from cowp.data.cache_schema import validate_numeric_invariants


def _record(*, transport_matches: bool = True):
    K, A, M = 2, 1, 3
    affected = np.zeros((K, A, M), dtype=bool)
    unsafe = np.zeros_like(affected)
    budget = np.zeros_like(affected)
    burden_only = np.zeros_like(affected)
    affected[0, 0, 0] = True
    unsafe[0, 0, 0] = True
    affected[1, 0, 1] = True
    budget[1, 0, 1] = True
    burden_only[1, 0, 1] = True
    transport = affected.copy()
    if not transport_matches:
        transport[1, 0, 1] = False
    return {
        "cowp/candidates/valid": np.ones(K, dtype=bool),
        "cowp/critical/valid": np.ones(A, dtype=bool),
        "cowp/audit/pair_relevant": np.asarray([[True], [False]], dtype=bool),
        "cowp/audit/relevance_mass": np.asarray([[1.0], [0.05]], dtype=np.float32),
        "cowp/audit/root_affected": affected,
        "cowp/audit/root_unsafe": unsafe,
        "cowp/audit/root_budget_crossed": budget,
        "cowp/audit/root_burden_only_affected": burden_only,
        "cowp/witness/pair_noncoercive_feasible": np.ones((K, A), dtype=bool),
        "cowp/witness/exists": np.zeros((K, A), dtype=bool),
        "cowp/witness/blocker_code": np.zeros((K, A), dtype=np.int32),
        "cowp/transport/mode_affected": transport,
        "cowp/witness/opr": np.ones((K, A), dtype=np.float32),
    }


def test_transport_affected_must_mirror_audit_even_for_irrelevant_pairs():
    errors = validate_numeric_invariants(_record(transport_matches=True), {"ncf": {}})
    assert not any("mode_affected" in e for e in errors)

    errors = validate_numeric_invariants(_record(transport_matches=False), {"ncf": {}})
    assert any("transport/mode_affected" in e for e in errors)


def test_affected_and_burden_only_identities_are_explicit():
    data = _record()
    bad = dict(data)
    bad["cowp/audit/root_affected"] = data["cowp/audit/root_affected"].copy()
    bad["cowp/audit/root_affected"][1, 0, 1] = False
    errors = validate_numeric_invariants(bad, {"ncf": {}})
    assert any("root_affected" in e for e in errors)

    bad2 = dict(data)
    bad2["cowp/audit/root_burden_only_affected"] = data["cowp/audit/root_burden_only_affected"].copy()
    bad2["cowp/audit/root_burden_only_affected"][0, 0, 0] = True
    errors = validate_numeric_invariants(bad2, {"ncf": {}})
    assert any("root_burden_only_affected" in e for e in errors)


def test_smoke_does_not_hard_gate_arbitrary_burden_only_prevalence(tmp_path):
    import json
    import subprocess
    import sys

    paired = {
        "num_representative_scenes": 48,
        "pairing_completeness": {"complete": True, "build_error_count": 0},
        "new": {
            "any_valid_scene_rate": 1.0,
            "any_ncf_scene_rate": 0.42,
            "best_case_selected_false_safe_lower_bound": 0.50,
            "any_priority_eligible_scene_rate": 0.90,
            "best_case_pbtr_lower_bound": 0.44,
        },
        "paired": {"old_hard_scene_count": 48, "hard_scene_ncf_recovery_rate": 0.21},
    }
    bank = {
        "any_ncf_scene_rate": 0.42,
        "best_case_selected_false_safe_lower_bound": 0.50,
        "best_case_pbtr_lower_bound": 0.44,
    }
    ablation = {"ablations": {"all": bank, "without_priority_smooth_yield": dict(bank)}}
    profile = {"unique_scenarios": 96, "critical_selection_reference_modes": {"fixed_anchor_v1": 96}}
    audit = {
        "pair_rates": {"relevant": 0.4, "burden_only_root_fraction": 0.0, "burden_only_scene_rate": 0.0},
        "root_counts": {"burden_only": 0},
        "integrity": {
            "affected_definition_consistent": True,
            "burden_only_definition_consistent": True,
            "no_read_errors": True,
            "no_silent_blockers": True,
            "no_irrelevant_blockers": True,
            "transport_affected_matches_audit": True,
            "transport_conflict_matches_audit": True,
            "transport_retain_matches_audit": True,
            "canonical_root_weight_matches_transport": True,
            "no_responses_for_irrelevant_pairs": True,
        },
    }
    files = {}
    for name, obj in (("paired", paired), ("ablation", ablation), ("profile", profile), ("audit", audit)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(obj), encoding="utf-8")
        files[name] = path
    out = tmp_path / "verdict.json"
    proc = subprocess.run(
        [sys.executable, "-m", "cowp.scripts.58_screen_v16_8_9_causal_audit_probe",
         "--paired-probe", str(files["paired"]), "--source-ablation", str(files["ablation"]),
         "--profile-summary", str(files["profile"]), "--audit-diagnostic", str(files["audit"]),
         "--output", str(out)],
        cwd=".", check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    verdict = json.loads(out.read_text(encoding="utf-8"))
    assert verdict["screen_pass"] is True
    assert verdict["recommend_strict_probe"] is True
    assert verdict["recommend_full_rebuild"] is False
    assert verdict["advisories"]["burden_only_affected_observed"] is False
