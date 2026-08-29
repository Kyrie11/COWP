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


def test_longitudinal_envelopes_are_existing_control_reachable_endpoints():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import _project_candidate_bank_through_controller_np

    cfg = _cfg()
    cur = _state(5.0)
    nominal = np.zeros((3, 3, 7), dtype=np.float32)
    nominal[:, :, 5] = 4.5
    nominal[:, :, 6] = 1.9
    for t in range(3):
        nominal[:, t, 0] = 0.5 * (t + 1)
        nominal[:, t, 3] = 5.0

    projected, kin_ok, accel = _project_candidate_bank_through_controller_np(
        cur,
        nominal,
        cfg,
        previous_longitudinal_accel=0.0,
        longitudinal_envelope_mode=np.asarray([0, -1, 1], dtype=np.int8),
    )
    assert projected.shape == nominal.shape
    assert np.all(kin_ok)
    assert np.allclose(accel[0], [0.0, 0.0, 0.0], atol=1.0e-6)
    assert np.allclose(accel[1], [-0.8, -1.6, -2.4], atol=1.0e-6)
    assert np.allclose(accel[2], [0.8, 1.6, 2.4], atol=1.0e-6)

    # Historical default path must be unchanged.
    default, default_kin, default_accel = _project_candidate_bank_through_controller_np(
        cur, nominal[:1], cfg, previous_longitudinal_accel=0.0
    )
    assert np.allclose(default, projected[:1], atol=1.0e-6, rtol=0.0)
    assert np.array_equal(default_kin, kin_ok[:1])
    assert np.allclose(default_accel, accel[:1], atol=1.0e-6, rtol=0.0)


def test_shifted_reference_appends_only_causal_constant_velocity_terminal_edge():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import _shift_append_terminal_reference_np

    tr = np.zeros((3, 7), dtype=np.float32)
    tr[:, 0] = [1.0, 2.0, 3.0]
    tr[:, 3] = 10.0
    tr[:, 5] = 4.5
    tr[:, 6] = 1.9
    shifted = _shift_append_terminal_reference_np(tr, 0.1)
    assert np.allclose(shifted[0], tr[1])
    assert np.allclose(shifted[1], tr[2])
    assert abs(float(shifted[2, 0]) - 4.0) < 1.0e-6
    assert abs(float(shifted[2, 3]) - 10.0) < 1.0e-6


def test_shift_closed_constructor_prefers_frozen_cowp_score_then_nominal_control():
    import numpy as np
    from cowp.label.trajectory_primitives import constant_accel_trajectory
    from cowp.waymax_eval.policy_wrapper import (
        _consistent_one_step_targets_np,
        _construct_shift_closed_control_reachable_tube_np,
    )

    cfg = _cfg()
    state = _state(4.0)[None, :]
    nominal = np.stack(
        [
            constant_accel_trajectory(state[0], 8, 0.1, accel=0.0),
            constant_accel_trajectory(state[0], 8, 0.1, accel=-0.5),
        ],
        axis=0,
    )
    targets, accels, _ = _consistent_one_step_targets_np(
        state[0], nominal[:, 0], cfg, previous_longitudinal_accel=0.0
    )
    selected, detail = _construct_shift_closed_control_reachable_tube_np(
        state,
        0,
        nominal,
        np.asarray([True, True]),
        np.asarray([True, True]),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([0.4, 0.1], dtype=np.float64),
        np.asarray([8, 8], dtype=np.float32),
        targets,
        accels,
        {},
        cfg,
        previous_longitudinal_accel=0.0,
    )
    assert selected is not None
    assert int(selected["parent_index"]) == 1
    assert int(selected["mode"]) == 0
    assert not detail["selected_is_lifted"]
    assert detail["tube_shift_closed"] > 0
    assert detail["tube_shift_closed"] == (
        detail["tube_nominal_shift_closed"]
        + detail["tube_lower_envelope_shift_closed"]
        + detail["tube_upper_envelope_shift_closed"]
    )
    assert detail["nominal_first_target_max_abs_error"] <= 2.0e-5


def test_constructor_can_create_lifted_support_missing_from_nominal_bank(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw
    from cowp.label.trajectory_primitives import constant_accel_trajectory

    cfg = _cfg()
    state = _state(5.0)[None, :]
    nominal = constant_accel_trajectory(state[0], 8, 0.1, accel=0.0)[None, ...]
    targets, accels, _ = pw._consistent_one_step_targets_np(
        state[0], nominal[:, 0], cfg, previous_longitudinal_accel=0.0
    )

    # Only a trajectory whose first realized speed is below the current speed is
    # physically certified.  The unchanged bank has no such tube; the lower
    # reachable acceleration envelope constructs it without relaxing limits.
    def fake_certificate(agent_state, sdc_index, trajectory, waymax_ok, roadgraph, cfg, **kwargs):
        first_speed = float(np.linalg.norm(np.asarray(trajectory)[0, 3:5]))
        ok = first_speed < 4.99
        return ok, {
            "collision_min_margin_m": 1.0 if ok else -1.0,
            "collision_safe": ok,
            "roadgraph_safe": ok,
            "kinematic_safe": ok,
        }

    monkeypatch.setattr(pw, "_physical_recovery_tube_certificate_np", fake_certificate)
    selected, detail = pw._construct_shift_closed_control_reachable_tube_np(
        state,
        0,
        nominal,
        np.asarray([True]),
        np.asarray([True]),
        np.asarray([1], dtype=np.int64),
        np.asarray([0.1], dtype=np.float64),
        np.asarray([0], dtype=np.float32),
        targets,
        accels,
        {},
        cfg,
        previous_longitudinal_accel=0.0,
    )
    assert selected is not None
    assert int(selected["mode"]) == -1
    assert detail["selected_is_lifted"]
    assert detail["tube_lifted_only_parent_count"] == 1
    assert float(selected["accel"]) < float(accels[0])
    assert not np.allclose(selected["target"], targets[0], rtol=0.0, atol=1.0e-6)


def test_tube_collision_checks_never_request_logged_future(monkeypatch):
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
    selected, _detail = pw._construct_shift_closed_control_reachable_tube_np(
        state, 0, nominal,
        np.asarray([True]), np.asarray([True]), np.asarray([1], dtype=np.int64),
        np.asarray([0.1]), np.asarray([8]), targets, accels, {}, cfg, 0.0,
    )
    assert selected is not None
    assert seen["calls"] > 0


def test_v38_method_keeps_priority_gate_defaults():
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    from cowp.waymax_eval.rollout import _method_gate_defaults

    method = "cowp_shift_closed_control_reachable_tube"
    assert _canonical_online_method(method, "hard") == (method, "priority")
    assert _method_gate_defaults(method, "hard") == (method, "priority")


def test_constructor_fails_closed_when_shift_certificate_is_missing(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw
    from cowp.label.trajectory_primitives import constant_accel_trajectory

    cfg = _cfg()
    state = _state(4.0)[None, :]
    nominal = constant_accel_trajectory(state[0], 8, 0.1, accel=0.0)[None, ...]
    targets, accels, _ = pw._consistent_one_step_targets_np(
        state[0], nominal[:, 0], cfg, previous_longitudinal_accel=0.0
    )
    calls = {"n": 0}

    def current_only_certificate(*args, **kwargs):
        calls["n"] += 1
        # Every hypothesis passes its current-tube audit but fails the immediately
        # following shifted-tube audit. No first action may therefore be emitted.
        ok = (calls["n"] % 2) == 1
        return ok, {
            "collision_min_margin_m": 1.0 if ok else -1.0,
            "collision_safe": ok,
            "roadgraph_safe": ok,
            "kinematic_safe": ok,
        }

    monkeypatch.setattr(pw, "_physical_recovery_tube_certificate_np", current_only_certificate)
    selected, detail = pw._construct_shift_closed_control_reachable_tube_np(
        state, 0, nominal,
        np.asarray([True]), np.asarray([True]), np.asarray([1], dtype=np.int64),
        np.asarray([0.1]), np.asarray([8]), targets, accels, {}, cfg, 0.0,
    )
    assert selected is None
    assert not detail["selected"]
    assert detail["tube_full_physically_safe"] > 0
    assert detail["tube_shift_closed"] == 0
