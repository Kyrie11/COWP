from __future__ import annotations

import numpy as np
import pytest


def _state() -> np.ndarray:
    s = np.zeros((3, 11), dtype=np.float32)
    s[0, 0:2] = [1.0, 2.0]
    s[0, 3:5] = [5.0, 0.0]
    s[0, 5] = 5.0
    s[0, 6] = 0.0
    s[0, 7:10] = [4.8, 1.9, 1.6]
    s[0, 10] = 1.0
    s[1, 0:2] = [10.0, 3.0]
    s[1, 3:5] = [2.0, -1.0]
    s[1, 5] = float(np.hypot(2.0, -1.0))
    s[1, 7:10] = [4.8, 1.9, 1.6]
    s[1, 10] = 1.0
    # invalid agent must remain untouched
    s[2, 0:2] = [99.0, 99.0]
    s[2, 3:5] = [9.0, 9.0]
    s[2, 10] = 0.0
    return s


def test_successor_state_uses_emitted_ego_target_and_causal_cv_others():
    from cowp.waymax_eval.policy_wrapper import _counterfactual_successor_agent_state

    state = _state()
    target = np.asarray([1.55, 2.05, 0.1, 5.4, 0.2], dtype=np.float32)
    nxt = _counterfactual_successor_agent_state(state, 0, target, {"time": {"dt": 0.1}})

    np.testing.assert_allclose(nxt[0, 0:2], target[0:2])
    np.testing.assert_allclose(nxt[0, 3:5], target[3:5])
    assert nxt[0, 6] == pytest.approx(float(target[2]))
    assert nxt[0, 5] == pytest.approx(float(np.linalg.norm(target[3:5])))
    np.testing.assert_allclose(nxt[1, 0:2], state[1, 0:2] + 0.1 * state[1, 3:5])
    np.testing.assert_allclose(nxt[2], state[2])
    # Input is not mutated.
    np.testing.assert_allclose(state, _state())


def test_successor_signature_is_lexicographic_option_set_not_scalar_cost(monkeypatch):
    from cowp.waymax_eval import policy_wrapper

    K, H = 5, 8
    def fake_candidates(*_args, **_kwargs):
        traj = np.zeros((K, H, 7), dtype=np.float32)
        valid = np.asarray([1, 1, 1, 1, 0], dtype=bool)
        conv = np.asarray([1, 1, 1, 0, 0], dtype=bool)
        macro = np.asarray([1, 2, 2, 3, 0], dtype=np.int64)
        util = np.zeros(K, dtype=np.float32)
        road = np.asarray([1, 1, 1, 1, 0], dtype=bool)
        coll = np.asarray([1, 1, 1, 0, 0], dtype=bool)
        prefix = np.asarray([H, H, H, 6, 0], dtype=np.int32)
        margin = np.zeros(K, dtype=np.float32)
        return traj, valid, conv, macro, util, road, coll, prefix, margin

    monkeypatch.setattr(policy_wrapper, "_route_lane_aware_candidates", fake_candidates)
    sig, d = policy_wrapper._successor_option_signature(
        _state(), 0, np.asarray([1.5, 2.0, 0.0, 5.0, 0.0], dtype=np.float32),
        {"xy": np.zeros((0, 2), dtype=np.float32)}, {"time": {"dt": 0.1}},
    )
    assert sig == (1, 2, 3, H)
    assert d["conventional_macro_types"] == 2
    assert d["conventional_candidates"] == 3


def test_pareto_guard_rejects_prefix_gain_when_any_existing_risk_worsens():
    torch = pytest.importorskip("torch")
    from cowp.waymax_eval.policy_wrapper import _strict_no_regret_rvr_switch

    prefix = torch.tensor([2.0, 7.0])
    trans = torch.tensor([0.4, 0.3])
    rule = torch.tensor([0.5, 0.4])
    action = torch.tensor([0.2, 0.1])
    pressure = torch.tensor([0.6, 0.5])
    assert _strict_no_regret_rvr_switch(0, 1, prefix, trans, rule, action, pressure)

    action_bad = torch.tensor([0.2, 0.21])
    assert not _strict_no_regret_rvr_switch(0, 1, prefix, trans, rule, action_bad, pressure)

    prefix_same = torch.tensor([7.0, 7.0])
    assert not _strict_no_regret_rvr_switch(0, 1, prefix_same, trans, rule, action, pressure)


def test_new_methods_preserve_cowp_priority_gate():
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    from cowp.waymax_eval.rollout import _method_gate_defaults

    for m in ("cowp_rvr_pareto_guard", "cowp_successor_option_viability"):
        assert _canonical_online_method(m, "hard") == (m, "priority")
        assert _method_gate_defaults(m, "hard") == (m, "priority")
