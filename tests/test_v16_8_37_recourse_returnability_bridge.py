from __future__ import annotations


def test_returnability_relation_is_set_dominance_not_count_scalarization():
    from cowp.waymax_eval.policy_wrapper import _returnability_relation

    strict, weak = _returnability_relation(False, frozenset({1}), False, frozenset({1, 2}))
    assert strict and weak

    # Same cardinality but incomparable semantic recourse sets must be rejected.
    strict, weak = _returnability_relation(False, frozenset({1, 2}), False, frozenset({1, 3}))
    assert not strict and not weak

    # Direct restoration strictly dominates a branch that still needs recourse.
    strict, weak = _returnability_relation(False, frozenset({1, 2, 3}), True, frozenset())
    assert strict and weak

    # If both directly restore, returnability itself is only a tie and cannot be
    # used as a strict entry witness.
    strict, weak = _returnability_relation(True, frozenset(), True, frozenset())
    assert not strict and weak


def test_direct_restoring_representatives_require_actual_successor_conventional_support(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw

    # Two semantic representatives.  Only the target ending at x=2 is mocked to
    # reach a successor with a full-conventional candidate.
    cand_valid = np.asarray([1, 1], dtype=bool)
    road = np.asarray([1, 1], dtype=bool)
    prefix = np.asarray([3, 4], dtype=np.float32)
    macro = np.asarray([1, 2], dtype=np.int64)
    fallback = np.asarray([0.1, 0.2], dtype=np.float64)
    targets = np.asarray([
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    agent_state = np.zeros((2, 11), dtype=np.float32)
    agent_state[:, 10] = 1.0

    def fake_kin(current, target, cfg):
        k = np.asarray(target).reshape(-1, 5).shape[0]
        return np.ones(k, dtype=bool), np.zeros(k), np.zeros(k), {}

    def fake_bank(state, sdc_index, roadgraph, cfg, other_future_trajs=None):
        conventional = bool(float(state[sdc_index, 0]) > 1.5)
        traj = np.zeros((1, 2, 7), dtype=np.float32)
        valid = np.asarray([True])
        conv = np.asarray([conventional])
        m = np.asarray([0], dtype=np.int64)
        u = np.asarray([0.0], dtype=np.float32)
        rs = np.asarray([True])
        cs = np.asarray([conventional])
        p = np.asarray([2 if conventional else 0], dtype=np.int32)
        margin = np.asarray([1.0], dtype=np.float32)
        return traj, valid, conv, m, u, rs, cs, p, margin

    monkeypatch.setattr(pw, "_waymax_kinematic_transition_np", fake_kin)
    monkeypatch.setattr(pw, "_route_lane_aware_candidates", fake_bank)
    mask, detail = pw._direct_restoring_representatives_np(
        agent_state, 0, {}, {}, cand_valid, road, prefix, macro, fallback, targets
    )
    assert mask.tolist() == [False, True]
    assert detail["restoring"] == 1
    assert detail["evaluated"] == 2


def test_returnability_signature_uses_new_replan_action_not_original_second_waypoint(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw

    agent_state = np.zeros((2, 11), dtype=np.float32)
    agent_state[:, 10] = 1.0
    first_target = np.asarray([0.5, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    calls = {"n": 0}

    def fake_consistent(current, desired, cfg, previous_longitudinal_accel=0.0):
        k = desired.shape[0]
        # Newly replanned actions: macro 1 goes to x=1, macro 2 to x=2.
        targets = np.zeros((k, 5), dtype=np.float32)
        targets[:, 0] = np.asarray([1.0, 2.0])[:k]
        targets[:, 3] = 1.0
        return targets, np.zeros(k, dtype=np.float32), np.zeros(k, dtype=np.float32)

    def fake_kin(current, target, cfg):
        k = np.asarray(target).reshape(-1, 5).shape[0]
        return np.ones(k, dtype=bool), np.zeros(k), np.zeros(k), {}

    def fake_bank(state, sdc_index, roadgraph, cfg, other_future_trajs=None):
        calls["n"] += 1
        x = float(state[sdc_index, 0])
        if calls["n"] == 1:
            # s1 has no conventional support, but two semantic recourse reps.
            traj = np.zeros((2, 3, 7), dtype=np.float32)
            traj[:, 0, 0] = [100.0, 200.0]  # deliberately irrelevant nominal waypoints
            valid = np.asarray([True, True])
            conv = np.asarray([False, False])
            macro = np.asarray([1, 2], dtype=np.int64)
            util = np.zeros(2, dtype=np.float32)
            road = np.asarray([True, True])
            coll = np.asarray([False, False])
            prefix = np.asarray([3, 3], dtype=np.int32)
            margin = np.zeros(2, dtype=np.float32)
            return traj, valid, conv, macro, util, road, coll, prefix, margin
        # Only the replanned control action at x=2 restores conventional support.
        traj = np.zeros((1, 2, 7), dtype=np.float32)
        valid = np.asarray([True])
        conv = np.asarray([x > 1.5])
        macro = np.asarray([0], dtype=np.int64)
        util = np.zeros(1, dtype=np.float32)
        road = np.asarray([True])
        coll = conv.copy()
        prefix = np.asarray([2 if conv[0] else 0], dtype=np.int32)
        margin = np.zeros(1, dtype=np.float32)
        return traj, valid, conv, macro, util, road, coll, prefix, margin

    monkeypatch.setattr(pw, "_consistent_one_step_targets_np", fake_consistent)
    monkeypatch.setattr(pw, "_waymax_kinematic_transition_np", fake_kin)
    monkeypatch.setattr(pw, "_route_lane_aware_candidates", fake_bank)
    direct, macros, detail = pw._returnability_witness_signature(
        agent_state, 0, first_target, 0.0, {}, {}
    )
    assert not direct
    assert macros == frozenset({2})
    assert detail["recourse_action_classes_evaluated"] == 2


def test_v37_method_keeps_priority_gate_defaults():
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    from cowp.waymax_eval.rollout import _method_gate_defaults

    method = "cowp_recourse_returnability_bridge"
    assert _canonical_online_method(method, "hard") == (method, "priority")
    assert _method_gate_defaults(method, "hard") == (method, "priority")
