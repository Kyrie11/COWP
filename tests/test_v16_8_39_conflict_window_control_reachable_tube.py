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
            "online_collision_max_agents": 24,
            "online_collision_agent_radius_m": 60.0,
            "waymax_kinematics_max_acc_mps2": 10.4,
            "waymax_kinematics_max_steering_curvature": 0.3,
            "waymax_kinematics_dt_s": 0.1,
        },
    }


def test_conflict_window_family_contains_v38_and_event_release_without_duplicates():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import _conflict_window_envelope_schedule_family

    family = _conflict_window_envelope_schedule_family(8, 2, 5)
    by_name = {r["policy_name"]: r for r in family}
    assert set(by_name) == {
        "NOMINAL", "LOWER_ALL", "UPPER_ALL",
        "LOWER_TO_FIRST_CONFLICT", "UPPER_TO_FIRST_CONFLICT",
        "LOWER_TO_LAST_CONFLICT", "UPPER_TO_LAST_CONFLICT",
    }
    assert np.array_equal(by_name["NOMINAL"]["schedule"], np.zeros(8, dtype=np.int8))
    assert np.array_equal(by_name["LOWER_ALL"]["schedule"], -np.ones(8, dtype=np.int8))
    assert np.array_equal(by_name["UPPER_ALL"]["schedule"], np.ones(8, dtype=np.int8))
    assert np.array_equal(
        by_name["LOWER_TO_FIRST_CONFLICT"]["schedule"],
        np.asarray([-1, -1, -1, 0, 0, 0, 0, 0], dtype=np.int8),
    )
    assert np.array_equal(
        by_name["UPPER_TO_LAST_CONFLICT"]["schedule"],
        np.asarray([1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int8),
    )
    assert len({r["schedule"].tobytes() for r in family}) == len(family)

    # If the last sampled conflict is at the terminal edge, event-release and
    # all-horizon schedules are identical and must not fabricate extra support.
    terminal = _conflict_window_envelope_schedule_family(8, 7, 7)
    assert len(terminal) == 3


def test_schedule_projector_uses_reachable_interval_endpoints_and_preserves_default():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import _project_candidate_bank_through_controller_np

    cfg = _cfg()
    cur = _state(5.0)
    nominal = np.zeros((1, 4, 7), dtype=np.float32)
    nominal[:, :, 5] = 4.5
    nominal[:, :, 6] = 1.9
    for t in range(4):
        nominal[:, t, 0] = 0.5 * (t + 1)
        nominal[:, t, 3] = 5.0

    default, default_kin, default_acc = _project_candidate_bank_through_controller_np(
        cur, nominal, cfg, previous_longitudinal_accel=0.0
    )
    schedule = np.asarray([[-1, -1, 0, 0]], dtype=np.int8)
    projected, kin, accel = _project_candidate_bank_through_controller_np(
        cur, nominal, cfg, previous_longitudinal_accel=0.0,
        longitudinal_envelope_schedule=schedule,
    )
    assert np.all(kin)
    assert np.allclose(accel[0, :2], [-0.8, -1.6], atol=1.0e-6)
    # Releasing the endpoint policy returns to the unchanged nominal controller;
    # it is not silently held for the full horizon.
    assert accel[0, 2] > -2.4
    assert not np.allclose(projected, default, atol=1.0e-6, rtol=0.0)

    zero_schedule, zero_kin, zero_acc = _project_candidate_bank_through_controller_np(
        cur, nominal, cfg, previous_longitudinal_accel=0.0,
        longitudinal_envelope_schedule=np.zeros((1, 4), dtype=np.int8),
    )
    assert np.allclose(zero_schedule, default, atol=1.0e-6, rtol=0.0)
    assert np.array_equal(zero_kin, default_kin)
    assert np.allclose(zero_acc, default_acc, atol=1.0e-6, rtol=0.0)


def test_collision_window_matches_frozen_sampled_collision_decision():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import (
        _collision_audit_against_context,
        _collision_violation_window_against_context,
    )

    traj = np.zeros((8, 7), dtype=np.float32)
    traj[:, 0] = np.arange(8, dtype=np.float32)
    context = {
        "idx": np.arange(8, dtype=np.int64),
        "horizon_steps": 8,
        "base_xy": np.asarray([[[100.0, 0.0]] * 8], dtype=np.float32),
        "cv_xy": np.asarray([[[100.0, 0.0]] * 8], dtype=np.float32),
        "base_threshold_m": np.asarray([1.0], dtype=np.float32),
        "priority_threshold_m": np.asarray([1.0], dtype=np.float32),
        "priority_like": np.asarray([False]),
        "require_cv": True,
    }
    safe = _collision_audit_against_context(traj, context)
    window = _collision_violation_window_against_context(traj, context)
    assert safe["safe"]
    assert not window["has_violation"]

    context["base_xy"][0, 2] = traj[2, :2]
    context["base_xy"][0, 5] = traj[5, :2]
    unsafe = _collision_audit_against_context(traj, context)
    window = _collision_violation_window_against_context(traj, context)
    assert not unsafe["safe"]
    assert unsafe["safe_prefix_steps"] == 2
    assert window["has_violation"]
    assert window["first_violation_step"] == 2
    assert window["last_violation_step"] == 5
    assert window["violation_sample_count"] == 2


def test_constructor_can_add_event_release_only_shift_closed_support(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw
    from cowp.label.trajectory_primitives import constant_accel_trajectory

    cfg = _cfg()
    state = _state(5.0)[None, :]
    nominal = constant_accel_trajectory(state[0], 8, 0.1, accel=0.0)[None, ...]
    targets, accels, _ = pw._consistent_one_step_targets_np(
        state[0], nominal[:, 0], cfg, previous_longitudinal_accel=0.0
    )

    # Force a nominal conflict window ending at edge 2. The physical certificate
    # accepts only a tube that initially decelerates but releases before the end:
    # nominal has no initial lift and LOWER_ALL remains too slow at the terminal
    # edge. This isolates the support added by the event-derived schedule.
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

    def fake_certificate(agent_state, sdc_index, trajectory, waymax_ok, roadgraph, cfg, **kwargs):
        tr = np.asarray(trajectory)
        first_speed = float(np.linalg.norm(tr[0, 3:5]))
        last_speed = float(np.linalg.norm(tr[-1, 3:5]))
        ok = first_speed < 4.99 and last_speed > 3.5
        return ok, {
            "collision_min_margin_m": 1.0 if ok else -1.0,
            "collision_safe": ok,
            "roadgraph_safe": ok,
            "kinematic_safe": ok,
        }

    monkeypatch.setattr(pw, "_physical_recovery_tube_certificate_np", fake_certificate)
    selected, detail = pw._construct_conflict_window_control_reachable_tube_np(
        state, 0, nominal,
        np.asarray([True]), np.asarray([True]), np.asarray([1], dtype=np.int64),
        np.asarray([0.1]), np.asarray([0]), targets, accels, {}, cfg, 0.0,
    )
    assert selected is not None
    assert selected["event_release"]
    assert selected["policy_name"] == "LOWER_TO_FIRST_CONFLICT"
    assert detail["selected_is_event_release"]
    assert detail["tube_event_release_only_parent_count"] == 1
    assert detail["tube_shift_closed"] > 0
    assert detail["selected_nonnominal_edges"] == 3


def test_v39_tube_never_requests_logged_future(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw
    from cowp.label.trajectory_primitives import constant_accel_trajectory

    cfg = _cfg()
    state = _state(3.0)[None, :]
    nominal = constant_accel_trajectory(state[0], 8, 0.1, accel=0.0)[None, ...]
    targets, accels, _ = pw._consistent_one_step_targets_np(
        state[0], nominal[:, 0], cfg, previous_longitudinal_accel=0.0
    )
    original = pw._prepare_collision_check_context
    seen = {"calls": 0}

    def guarded(*args, **kwargs):
        assert kwargs.get("other_future_trajs", None) is None
        seen["calls"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pw, "_prepare_collision_check_context", guarded)
    selected, _ = pw._construct_conflict_window_control_reachable_tube_np(
        state, 0, nominal,
        np.asarray([True]), np.asarray([True]), np.asarray([1], dtype=np.int64),
        np.asarray([0.1]), np.asarray([8]), targets, accels, {}, cfg, 0.0,
    )
    assert selected is not None
    assert seen["calls"] > 0


def test_v39_method_keeps_priority_gate_defaults():
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    from cowp.waymax_eval.rollout import _method_gate_defaults

    method = "cowp_conflict_window_control_reachable_tube"
    assert _canonical_online_method(method, "hard") == (method, "priority")
    assert _method_gate_defaults(method, "hard") == (method, "priority")


def test_v39_constructor_fails_closed_without_shift_certificate(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw
    from cowp.label.trajectory_primitives import constant_accel_trajectory

    cfg = _cfg()
    state = _state(4.0)[None, :]
    nominal = constant_accel_trajectory(state[0], 8, 0.1, accel=0.0)[None, ...]
    targets, accels, _ = pw._consistent_one_step_targets_np(
        state[0], nominal[:, 0], cfg, previous_longitudinal_accel=0.0
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
    calls = {"n": 0}

    def current_only_certificate(*args, **kwargs):
        calls["n"] += 1
        ok = (calls["n"] % 2) == 1
        return ok, {
            "collision_min_margin_m": 1.0 if ok else -1.0,
            "collision_safe": ok,
            "roadgraph_safe": ok,
            "kinematic_safe": ok,
        }

    monkeypatch.setattr(pw, "_physical_recovery_tube_certificate_np", current_only_certificate)
    selected, detail = pw._construct_conflict_window_control_reachable_tube_np(
        state, 0, nominal,
        np.asarray([True]), np.asarray([True]), np.asarray([1], dtype=np.int64),
        np.asarray([0.1]), np.asarray([8]), targets, accels, {}, cfg, 0.0,
    )
    assert selected is None
    assert detail["tube_full_physically_safe"] > 0
    assert detail["tube_shift_closed"] == 0
