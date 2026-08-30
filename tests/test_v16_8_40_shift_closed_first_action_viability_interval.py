from __future__ import annotations


def _state(speed: float = 5.0):
    import numpy as np

    x = np.zeros((11,), dtype=np.float32)
    x[3] = speed
    x[5] = speed
    x[6] = 0.0
    x[7] = 4.5
    x[8] = 1.9
    x[10] = 1.0
    return x


def _cfg():
    return {
        "time": {"dt": 0.1},
        "limits": {"max_candidates": 16},
        "candidate": {
            "max_accel_mps2": 4.0,
            "max_decel_mps2": 6.0,
            "max_jerk_mps3": 8.0,
            "max_yaw_rate_rad_s": 1.2,
            "max_lateral_accel_mps2": 4.0,
        },
        "waymax": {"max_delta_yaw_rad": 0.12},
        "planning": {
            "online_collision_check_horizon_steps": 8,
            "online_collision_check_stride": 1,
            "online_collision_check_max_agents": 24,
            "online_collision_max_agents": 24,
            "online_collision_agent_radius_m": 60.0,
            "waymax_kinematics_max_acc_mps2": 10.4,
            "waymax_kinematics_max_steering_curvature": 0.3,
            "waymax_kinematics_dt_s": 0.1,
        },
    }


def _one_candidate(speed: float = 5.0):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw
    from cowp.label.trajectory_primitives import constant_accel_trajectory

    cfg = _cfg()
    state = _state(speed)[None, :]
    nominal = constant_accel_trajectory(state[0], 8, 0.1, accel=0.0)[None, ...]
    targets, accels, _ = pw._consistent_one_step_targets_np(
        state[0], nominal[:, 0], cfg, previous_longitudinal_accel=0.0
    )
    return cfg, state, nominal, targets, accels


def test_constraint_boundary_fractions_are_deterministic_and_interior():
    from cowp.waymax_eval.policy_wrapper import (
        _constraint_boundary_fractions_from_triplet,
    )

    first = _constraint_boundary_fractions_from_triplet(-1.0, 1.0, -1.0)
    second = _constraint_boundary_fractions_from_triplet(-1.0, 1.0, -1.0)
    assert first == second
    assert len(first) >= 2
    assert all(0.0 < frac < 1.0 for frac, _ in first)
    assert all(abs(frac - 0.5) > 1.0e-10 for frac, _ in first)
    assert any(source in {"secant_boundary", "quadratic_boundary"} for _, source in first)

    # No finite hard-margin evidence means no speculative action proposal.
    assert _constraint_boundary_fractions_from_triplet(float("nan"), 1.0, -1.0) == []


def test_first_accel_override_realizes_an_interior_existing_control_action():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import (
        _project_candidate_bank_through_controller_np,
    )

    cfg, state, nominal, _, _ = _one_candidate(5.0)
    default, _, default_accel = _project_candidate_bank_through_controller_np(
        state[0], nominal, cfg, previous_longitudinal_accel=0.0
    )
    lower, _, lower_accel = _project_candidate_bank_through_controller_np(
        state[0],
        nominal,
        cfg,
        previous_longitudinal_accel=0.0,
        first_accel_override=np.asarray([-0.8]),
    )
    interior, kin, interior_accel = _project_candidate_bank_through_controller_np(
        state[0],
        nominal,
        cfg,
        previous_longitudinal_accel=0.0,
        first_accel_override=np.asarray([-0.4]),
    )
    assert np.all(kin)
    assert np.isclose(default_accel[0, 0], 0.0, atol=1.0e-6)
    assert np.isclose(lower_accel[0, 0], -0.8, atol=1.0e-6)
    assert np.isclose(interior_accel[0, 0], -0.4, atol=1.0e-6)
    assert not np.allclose(interior[0, 0, :5], default[0, 0, :5], atol=1.0e-6)
    assert not np.allclose(interior[0, 0, :5], lower[0, 0, :5], atol=1.0e-6)


def test_v40_preserves_every_nested_v39_certified_decision_exactly(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw

    cfg, state, nominal, targets, accels = _one_candidate(5.0)
    sentinel = {
        "parent_index": 0,
        "trajectory": np.array(nominal[0], copy=True),
        "target": np.array(targets[0], copy=True),
        "accel": float(accels[0]),
        "fallback_score": 0.1,
    }
    nested_detail = {"selected": True, "selected_policy_name": "LOWER_TO_LAST_CONFLICT"}
    monkeypatch.setattr(
        pw,
        "_construct_conflict_window_control_reachable_tube_np",
        lambda *args, **kwargs: (sentinel, dict(nested_detail)),
    )
    monkeypatch.setattr(
        pw,
        "_prepare_collision_check_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("V40 must not search when nested V39 has support")
        ),
    )
    selected, detail = pw._construct_shift_closed_first_action_viability_interval_np(
        state,
        0,
        nominal,
        np.asarray([True]),
        np.asarray([True]),
        np.asarray([1], dtype=np.int64),
        np.asarray([0.1]),
        np.asarray([8]),
        targets,
        accels,
        {},
        cfg,
        0.0,
    )
    assert selected is sentinel
    assert detail["nested_v39_selected"]
    assert not detail["first_action_interval_completion_attempted"]
    assert np.array_equal(selected["trajectory"], sentinel["trajectory"])
    assert np.array_equal(selected["target"], sentinel["target"])
    assert selected["accel"] == sentinel["accel"]


def test_v40_constructs_interval_only_new_first_action_support(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw

    cfg, state, nominal, targets, accels = _one_candidate(5.0)
    monkeypatch.setattr(
        pw,
        "_construct_conflict_window_control_reachable_tube_np",
        lambda *args, **kwargs: (
            None,
            {
                "selected": False,
                "tube_hypotheses_generated": 0,
                "tube_hypotheses_unique_action": 0,
                "tube_full_physically_safe": 0,
                "tube_shift_closed": 0,
            },
        ),
    )
    monkeypatch.setattr(
        pw,
        "_collision_violation_window_against_context",
        lambda *args, **kwargs: {
            "has_violation": True,
            "first_violation_step": 2,
            "last_violation_step": 2,
            "violation_sample_count": 1,
            "source": "base_cv",
        },
    )

    def interior_only_certificate(agent_state, sdc_index, trajectory, waymax_ok, roadgraph, cfg, **kwargs):
        st = np.asarray(agent_state)
        tr = np.asarray(trajectory)
        current_speed = float(st[int(sdc_index), 5])
        first_speed = float(np.linalg.norm(tr[0, 3:5]))
        if abs(current_speed - 5.0) < 1.0e-4:
            # Nominal=5.00, lower endpoint=4.92, canonical interior=4.96.
            ok = 4.945 < first_speed < 4.975
            margin = 1.0 - abs(first_speed - 4.96) * 100.0
        else:
            # Once a current interior action has passed, certify its shifted tail.
            ok = True
            margin = 0.5
        return bool(ok), {
            "collision_min_margin_m": float(margin if ok else -abs(margin) - 0.1),
            "collision_safe": bool(ok),
            "roadgraph_safe": bool(ok),
            "kinematic_safe": bool(ok),
        }

    monkeypatch.setattr(
        pw, "_physical_recovery_tube_certificate_np", interior_only_certificate
    )
    selected, detail = pw._construct_shift_closed_first_action_viability_interval_np(
        state,
        0,
        nominal,
        np.asarray([True]),
        np.asarray([True]),
        np.asarray([1], dtype=np.int64),
        np.asarray([0.1]),
        np.asarray([0]),
        targets,
        accels,
        {},
        cfg,
        0.0,
    )
    assert selected is not None
    assert detail["first_action_interval_completion_attempted"]
    assert detail["first_action_interval_basis_count"] > 0
    assert detail["first_action_interval_seed_evaluations"] > 0
    assert detail["first_action_interval_shift_closed"] > 0
    assert detail["first_action_interval_new_actions"] > 0
    assert detail["selected_is_first_action_interval_completion"]
    assert detail["selected_is_new_first_action"]
    assert 0.0 < detail["selected_first_accel_fraction"] < 1.0
    assert not np.allclose(selected["target"], targets[0, :5], atol=1.0e-6, rtol=0.0)


def test_v40_does_not_select_a_certificate_without_a_new_first_action(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw

    cfg, state, nominal, targets, accels = _one_candidate(5.0)
    monkeypatch.setattr(
        pw,
        "_construct_conflict_window_control_reachable_tube_np",
        lambda *args, **kwargs: (None, {"selected": False}),
    )
    monkeypatch.setattr(
        pw,
        "_collision_violation_window_against_context",
        lambda *args, **kwargs: {
            "has_violation": False,
            "first_violation_step": -1,
            "last_violation_step": -1,
            "violation_sample_count": 0,
            "source": "none",
        },
    )

    stateful = {"shift_witness_due": False}

    def nominal_only(agent_state, sdc_index, trajectory, waymax_ok, roadgraph, cfg, **kwargs):
        tr = np.asarray(trajectory)
        first_speed = float(np.linalg.norm(tr[0, 3:5]))
        if stateful["shift_witness_due"]:
            ok = True
            stateful["shift_witness_due"] = False
        else:
            ok = abs(first_speed - 5.0) < 1.0e-5
            stateful["shift_witness_due"] = bool(ok)
        return bool(ok), {
            "collision_min_margin_m": 1.0 if ok else -1.0,
            "collision_safe": bool(ok),
            "roadgraph_safe": bool(ok),
            "kinematic_safe": bool(ok),
        }

    monkeypatch.setattr(pw, "_physical_recovery_tube_certificate_np", nominal_only)
    selected, detail = pw._construct_shift_closed_first_action_viability_interval_np(
        state,
        0,
        nominal,
        np.asarray([True]),
        np.asarray([True]),
        np.asarray([1], dtype=np.int64),
        np.asarray([0.1]),
        np.asarray([0]),
        targets,
        accels,
        {},
        cfg,
        0.0,
    )
    assert selected is None
    assert detail["first_action_interval_shift_closed"] > 0
    assert detail["first_action_interval_new_actions"] == 0


def test_v40_constructor_fails_closed_without_shift_certificate(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw

    cfg, state, nominal, targets, accels = _one_candidate(4.0)
    monkeypatch.setattr(
        pw,
        "_construct_conflict_window_control_reachable_tube_np",
        lambda *args, **kwargs: (None, {"selected": False}),
    )
    monkeypatch.setattr(
        pw,
        "_collision_violation_window_against_context",
        lambda *args, **kwargs: {
            "has_violation": True,
            "first_violation_step": 2,
            "last_violation_step": 2,
            "violation_sample_count": 1,
            "source": "base_cv",
        },
    )

    def current_only(agent_state, sdc_index, trajectory, waymax_ok, roadgraph, cfg, **kwargs):
        st = np.asarray(agent_state)
        # A causal successor has advanced in x even when its speed is unchanged.
        is_current = abs(float(st[int(sdc_index), 0])) < 1.0e-6
        return bool(is_current), {
            "collision_min_margin_m": 1.0 if is_current else -1.0,
            "collision_safe": bool(is_current),
            "roadgraph_safe": bool(is_current),
            "kinematic_safe": bool(is_current),
        }

    monkeypatch.setattr(pw, "_physical_recovery_tube_certificate_np", current_only)
    selected, detail = pw._construct_shift_closed_first_action_viability_interval_np(
        state,
        0,
        nominal,
        np.asarray([True]),
        np.asarray([True]),
        np.asarray([1], dtype=np.int64),
        np.asarray([0.1]),
        np.asarray([0]),
        targets,
        accels,
        {},
        cfg,
        0.0,
    )
    assert selected is None
    assert detail["first_action_interval_full_physically_safe"] > 0
    assert detail["first_action_interval_shift_closed"] == 0


def test_v40_never_requests_logged_future_and_keeps_gate_defaults(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    from cowp.waymax_eval.rollout import _method_gate_defaults

    method = "cowp_shift_closed_first_action_viability_interval"
    assert _canonical_online_method(method, "hard") == (method, "priority")
    assert _method_gate_defaults(method, "hard") == (method, "priority")

    cfg, state, nominal, targets, accels = _one_candidate(3.0)
    monkeypatch.setattr(
        pw,
        "_construct_conflict_window_control_reachable_tube_np",
        lambda *args, **kwargs: (None, {"selected": False}),
    )
    original = pw._prepare_collision_check_context
    seen = {"calls": 0}

    def guarded(*args, **kwargs):
        assert kwargs.get("other_future_trajs", None) is None
        seen["calls"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pw, "_prepare_collision_check_context", guarded)
    pw._construct_shift_closed_first_action_viability_interval_np(
        state,
        0,
        nominal,
        np.asarray([True]),
        np.asarray([True]),
        np.asarray([1], dtype=np.int64),
        np.asarray([0.1]),
        np.asarray([8]),
        targets,
        accels,
        {},
        cfg,
        0.0,
    )
    assert seen["calls"] > 0
