from __future__ import annotations


def test_dominance_hysteresis_is_strict_on_entry_weak_on_continue():
    from cowp.waymax_eval.policy_wrapper import _dominance_hysteresis_transition

    # Inactive ties do not create a new mode.
    assert _dominance_hysteresis_transition(
        False, strict_alt_dominates=False, weak_alt_dominates=True
    ) == (False, False, False, False)
    # Strict dominance enters.
    assert _dominance_hysteresis_transition(
        False, strict_alt_dominates=True, weak_alt_dominates=True
    ) == (True, True, False, False)
    # Equality/non-strict non-regression keeps an active mode, avoiding chatter.
    assert _dominance_hysteresis_transition(
        True, strict_alt_dominates=False, weak_alt_dominates=True
    ) == (True, False, True, False)
    # Any dominance violation exits instead of unconditional over-commitment.
    assert _dominance_hysteresis_transition(
        True, strict_alt_dominates=False, weak_alt_dominates=False
    ) == (False, False, False, True)


def test_option_profile_relation_is_pointwise_not_scalar_area():
    from cowp.waymax_eval.policy_wrapper import _option_profile_relation

    base = (3, 3, 2, 1)
    alt = (3, 3, 2, 2)
    strict, weak, min_margin, area_delta = _option_profile_relation(base, alt)
    assert strict and weak and min_margin == 0 and area_delta == 1

    # A larger total area cannot compensate for losing an option at one horizon.
    bad = (4, 4, 4, 0)
    strict, weak, min_margin, area_delta = _option_profile_relation(base, bad)
    assert not strict and not weak
    assert min_margin < 0
    assert area_delta > 0


def test_successor_recovery_profile_counts_distinct_macros_not_duplicate_candidates(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw

    def fake_successor(agent_state, sdc_index, emitted_target, cfg):
        return np.array(agent_state, copy=True)

    def fake_candidates(agent_state, sdc_index, roadgraph, cfg, other_future_trajs=None):
        # horizon=5. Two candidates share macro 1; only its best prefix should count.
        traj = np.zeros((5, 5, 5), dtype=np.float32)
        valid = np.asarray([True, True, True, True, True])
        conventional = np.asarray([False, False, False, False, True])
        macro = np.asarray([1, 1, 2, 3, 4], dtype=np.int64)
        utility = np.zeros((5,), dtype=np.float32)
        road = np.asarray([True, True, True, False, True])
        collision = np.asarray([False, False, False, False, True])
        prefix = np.asarray([5, 2, 3, 5, 5], dtype=np.int32)
        margin = np.zeros((5,), dtype=np.float32)
        return traj, valid, conventional, macro, utility, road, collision, prefix, margin

    monkeypatch.setattr(pw, "_counterfactual_successor_agent_state", fake_successor)
    monkeypatch.setattr(pw, "_route_lane_aware_candidates", fake_candidates)
    state = np.zeros((2, 11), dtype=np.float32)
    profile, detail = pw._successor_recovery_option_profile(
        state, 1, np.zeros((5,), dtype=np.float32), {}, {}
    )
    # Surviving semantic macros by horizon: m1(5), m2(3), m4(5).
    assert profile == (3, 3, 3, 2, 2)
    assert detail["recovery_macro_types_any"] == 3
    assert detail["recovery_macro_types_full_horizon"] == 2
    assert detail["conventional_candidates"] == 1
    assert detail["roadgraph_safe_candidates"] == 4


def test_v33_methods_keep_priority_gate_defaults():
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    from cowp.waymax_eval.rollout import _method_gate_defaults

    methods = (
        "cowp_sov_dominance_hysteresis",
        "cowp_recovery_option_spectrum_hysteresis",
    )
    for method in methods:
        assert _canonical_online_method(method, "hard") == (method, "priority")
        assert _method_gate_defaults(method, "hard") == (method, "priority")


def test_v33_fresh37_remains_unseen_by_v32_outputs_and_hash_is_fixed():
    import hashlib
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    ids = [
        x.strip()
        for x in (repo / "reference_manifests/waymax_v16_8_32_fresh37_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    assert len(ids) == 37 and len(set(ids)) == 37
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == (
        "ecce3321d8f4cd57bbd3189b3673784bec8fde185b882e9c11c38430265a1481"
    )
