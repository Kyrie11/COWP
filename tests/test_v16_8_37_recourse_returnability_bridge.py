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


def test_returnability_current_edge_requires_one_safe_step_and_no_prefix_regression():
    from cowp.waymax_eval.policy_wrapper import _returnability_current_edge_admissible

    assert _returnability_current_edge_admissible(0, 1)
    assert _returnability_current_edge_admissible(3, 3)
    assert _returnability_current_edge_admissible(3, 4)
    assert not _returnability_current_edge_admissible(0, 0)
    assert not _returnability_current_edge_admissible(4, 3)
    assert not _returnability_current_edge_admissible(float("nan"), 3)


def test_returnability_signature_checks_all_action_classes_within_a_macro(monkeypatch):
    """A lower-prefix action in the same macro can be the only restoring recourse.

    The original draft kept only the max-prefix representative of each macro and
    would miss this existential witness.  V37 must deduplicate only physically
    identical emitted actions, not all candidates sharing a semantic macro.
    """
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw

    agent_state = np.zeros((2, 11), dtype=np.float32)
    agent_state[:, 10] = 1.0
    first_target = np.asarray([0.5, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    calls = {"n": 0}

    def fake_consistent(current, desired, cfg, previous_longitudinal_accel=0.0):
        targets = np.asarray([
            [1.0, 0.0, 0.0, 1.0, 0.0],  # max-prefix action, does not restore
            [2.0, 0.0, 0.0, 1.0, 0.0],  # same macro, does restore
        ], dtype=np.float32)
        k = desired.shape[0]
        return targets[:k], np.zeros(k, dtype=np.float32), np.zeros(k, dtype=np.float32)

    def fake_kin(current, target, cfg):
        k = np.asarray(target).reshape(-1, 5).shape[0]
        return np.ones(k, dtype=bool), np.zeros(k), np.zeros(k), {}

    def fake_bank(state, sdc_index, roadgraph, cfg, other_future_trajs=None):
        calls["n"] += 1
        x = float(state[sdc_index, 0])
        if calls["n"] == 1:
            traj = np.zeros((2, 3, 7), dtype=np.float32)
            valid = np.asarray([True, True])
            conv = np.asarray([False, False])
            macro = np.asarray([3, 3], dtype=np.int64)
            util = np.zeros(2, dtype=np.float32)
            road = np.asarray([True, True])
            coll = np.asarray([False, False])
            prefix = np.asarray([8, 2], dtype=np.int32)
            margin = np.zeros(2, dtype=np.float32)
            return traj, valid, conv, macro, util, road, coll, prefix, margin
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
    assert macros == frozenset({3})
    assert detail["recourse_representatives"] == 1
    assert detail["recourse_action_classes_available"] == 2
    assert detail["recourse_action_classes_evaluated"] == 2


def test_actual_bridge_respects_witnessed_macro_and_current_prefix(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw

    cand_valid = np.asarray([1, 1, 1, 1], dtype=bool)
    road = np.asarray([1, 1, 1, 1], dtype=bool)
    prefix = np.asarray([5, 4, 6, 2], dtype=np.float32)
    macro = np.asarray([1, 1, 2, 1], dtype=np.int64)
    fallback = np.asarray([0.1, 0.2, 0.0, 0.3], dtype=np.float64)
    targets = np.asarray([
        [1.0, 0.0, 0.0, 1.0, 0.0],  # witnessed macro, fails restoration
        [2.0, 0.0, 0.0, 1.0, 0.0],  # witnessed macro, restores
        [3.0, 0.0, 0.0, 1.0, 0.0],  # unwitnessed macro, would restore
        [4.0, 0.0, 0.0, 1.0, 0.0],  # witnessed but below base-prefix floor
    ], dtype=np.float32)
    agent_state = np.zeros((2, 11), dtype=np.float32)
    agent_state[:, 10] = 1.0

    def fake_kin(current, target, cfg):
        k = np.asarray(target).reshape(-1, 5).shape[0]
        return np.ones(k, dtype=bool), np.zeros(k), np.zeros(k), {}

    def fake_bank(state, sdc_index, roadgraph, cfg, other_future_trajs=None):
        x = float(state[sdc_index, 0])
        conventional = x >= 2.0
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
    mask, detail = pw._direct_restoring_candidates_np(
        agent_state, 0, {}, {}, cand_valid, road, prefix, macro, fallback, targets,
        allowed_macros=frozenset({1}), minimum_prefix_steps=4,
    )
    assert mask.tolist() == [False, True, False, False]
    assert detail["candidate_pool"] == 2
    assert detail["action_classes"] == 2
    assert detail["evaluated"] == 2
    assert detail["restoring"] == 1
    assert detail["allowed_macro_count"] == 1
    assert detail["minimum_prefix_steps"] == 4.0


def test_v37_episode_diagnostics_aggregate_action_class_and_bridge_consistency_fields():
    from cowp.waymax_eval.metrics_cowp import policy_diagnostic_scenario_rows

    rollouts = [{
        "scenario_id": "v37_diag",
        "steps": 1,
        "policy_diagnostics": [{
            "fallback_used": True,
            "fallback_reason": "no_conventional_use_recourse_returnability_bridge",
            "recourse_returnability_probe_used": True,
            "recourse_returnability_strict_dominates": True,
            "recourse_base_direct_restore": False,
            "recourse_rvr_direct_restore": False,
            "recourse_base_macro_count": 1,
            "recourse_rvr_macro_count": 2,
            "recourse_base_action_classes_available": 3,
            "recourse_rvr_action_classes_available": 5,
            "recourse_base_action_classes_evaluated": 3,
            "recourse_rvr_action_classes_evaluated": 4,
            "recourse_current_action_survives_one_step": True,
            "recovery_bridge_pending_before": True,
            "recovery_bridge_pending_after": False,
            "recovery_bridge_allowed_macro_count_before": 2,
            "recovery_bridge_entered": False,
            "recovery_bridge_direct_entry": False,
            "recovery_bridge_recourse_executed": True,
            "recovery_bridge_aborted": False,
            "recourse_direct_restoring_candidate_count": 1,
            "recourse_bridge_candidate_pool": 4,
            "recourse_bridge_action_classes_available": 3,
            "recourse_bridge_representatives_evaluated": 3,
            "recourse_bridge_minimum_prefix_steps": 2.0,
            "selected_waymax_kinematic_feasible": True,
            "recovery_switch_applied": True,
            "emergency_action_used": False,
            "zero_valid_candidate_step": False,
            "selected_candidate_valid": True,
            "selected_candidate_conventional_safe": False,
            "valid_candidates": 4,
            "conventional_candidates": 0,
        }],
        "standard_metrics": {
            "CR": 0.0,
            "CollisionRate": 0.0,
            "OffroadRate": 0.0,
            "KinematicsInfeasibilityRate": 0.0,
            "EP": 1.0,
        },
    }]
    row = policy_diagnostic_scenario_rows(rollouts)[0]
    assert row["mean_recourse_rvr_action_classes_available_on_probes"] == 5.0
    assert row["mean_recourse_rvr_action_classes_evaluated_on_probes"] == 4.0
    assert row["mean_recovery_bridge_allowed_macro_count_on_bridge_steps"] == 2.0
    assert row["mean_recourse_bridge_action_classes_available_on_bridge_steps"] == 3.0
    assert row["recovery_bridge_execution_rate_on_bridge_steps"] == 1.0
    assert row["recovery_bridge_abort_rate_on_bridge_steps"] == 0.0
