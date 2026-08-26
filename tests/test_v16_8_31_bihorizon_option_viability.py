from __future__ import annotations


def test_bihorizon_product_dominance_is_parameter_free():
    from cowp.waymax_eval.policy_wrapper import _bihorizon_option_dominates

    base = (0, 0, 0, 10)
    # Equal successor support + longer current prefix is admissible.
    assert _bihorizon_option_dominates(base, base, 3, 7)
    # Better successor support with equal current prefix is admissible.
    assert _bihorizon_option_dominates(base, (0, 0, 0, 11), 3, 3)
    # Any successor regression blocks even a large current-prefix gain.
    assert not _bihorizon_option_dominates((1, 1, 2, 80), (0, 9, 9, 80), 3, 80)
    # Any current-prefix regression blocks even a successor improvement.
    assert not _bihorizon_option_dominates(base, (1, 1, 1, 80), 7, 6)
    # Exact tie is not a reason to perturb COWP.
    assert not _bihorizon_option_dominates(base, base, 7, 7)


def test_restore_only_requires_discrete_conventional_restoration():
    from cowp.waymax_eval.policy_wrapper import _successor_restoration_dominates

    assert _successor_restoration_dominates(
        {"conventional_exists": 0}, {"conventional_exists": 1}
    )
    assert not _successor_restoration_dominates(
        {"conventional_exists": 0}, {"conventional_exists": 0, "conventional_candidates": 7}
    )
    assert not _successor_restoration_dominates(
        {"conventional_exists": 1}, {"conventional_exists": 1, "conventional_candidates": 9}
    )


def test_new_methods_keep_priority_gate_defaults():
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    from cowp.waymax_eval.rollout import _method_gate_defaults

    for method in ("cowp_bihorizon_option_viability", "cowp_successor_restore_only"):
        assert _canonical_online_method(method, "hard") == (method, "priority")
        assert _method_gate_defaults(method, "hard") == (method, "priority")


def test_v31_holdout64_is_disjoint_from_all_v30_development_panels():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "reference_manifests"
    def ids(name: str) -> set[str]:
        return {x.strip() for x in (root / name).read_text(encoding="utf-8").splitlines() if x.strip()}

    hold = ids("waymax_v16_8_31_holdout64_ids.txt")
    prior = (
        ids("waymax_v16_8_30_equivalence16_ids.txt")
        | ids("waymax_v16_8_30_counterfactual48_ids.txt")
        | ids("waymax_v16_8_30_balanced_dev96_ids.txt")
    )
    assert len(hold) == 64
    assert hold.isdisjoint(prior)


def test_v31_subset_reference_hashes_match_their_manifests():
    import hashlib
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    cases = [
        ("waymax_v16_8_30_equivalence16_ids.txt", "v16_8_29_equivalence16_cowp_reference.json"),
        ("waymax_v16_8_30_counterfactual48_ids.txt", "v16_8_29_counterfactual48_cowp_reference.json"),
        ("waymax_v16_8_30_counterfactual48_ids.txt", "v16_8_29_counterfactual48_rvr_reference.json"),
        ("waymax_v16_8_31_holdout64_ids.txt", "v16_8_31_holdout64_cowp_reference.json"),
        ("waymax_v16_8_31_holdout64_ids.txt", "v16_8_31_holdout64_rvr_reference.json"),
    ]
    for manifest, ref in cases:
        ids = [x.strip() for x in (repo / "reference_manifests" / manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
        expected = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        payload = json.loads((repo / "reference_results" / ref).read_text(encoding="utf-8"))
        assert payload["scenario_ids_sha256"] == expected
        assert [str(r["scenario_id"]) for r in payload["scenario_results"]] == ids
