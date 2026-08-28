from __future__ import annotations


def _state(speed: float = 2.0):
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
        "candidate": {
            "max_accel_mps2": 4.0,
            "max_decel_mps2": 6.0,
            "max_jerk_mps3": 8.0,
            "max_yaw_rate_rad_s": 1.2,
            "max_lateral_accel_mps2": 4.0,
        },
        "waymax": {"max_delta_yaw_rad": 0.12},
        # Unit tests do not require Waymax to be installed; pin the documented
        # defaults used by KinematicsInfeasibilityMetric.
        "planning": {
            "waymax_kinematics_max_acc_mps2": 10.4,
            "waymax_kinematics_max_steering_curvature": 0.3,
            "waymax_kinematics_dt_s": 0.1,
        },
    }


def test_waymax_kinematic_contract_exposes_v34_yaw_rate_mismatch():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import (
        _controller_transition_feasible_np,
        _waymax_kinematic_transition_np,
    )

    cfg = _cfg()
    cur = _state(2.0)
    yaw = 0.10
    target = np.asarray(
        [0.2, 0.0, yaw, 2.0 * np.cos(yaw), 2.0 * np.sin(yaw)],
        dtype=np.float32,
    )

    # V34's internal exact-waypoint check accepts this: 0.10 rad in 0.1 s is
    # below its 1.2 rad/s yaw-rate bound and lateral acceleration is < 4 m/s^2.
    assert bool(_controller_transition_feasible_np(cur, target, cfg, 0.0)[0])

    # Waymax's evaluated contract is different: steering curvature is
    # delta_yaw / traveled_arc ~= 0.10 / 0.20 = 0.5 1/m > 0.3 1/m.
    ok, accel, steering, detail = _waymax_kinematic_transition_np(cur, target, cfg)
    assert not bool(ok[0])
    assert abs(float(accel[0])) < 1.0e-4
    assert 0.49 < abs(float(steering[0])) < 0.51
    assert detail["max_steering_curvature"] == 0.3


def test_kinematic_guard_never_accepts_infeasible_alternative():
    from cowp.waymax_eval.policy_wrapper import _kinematic_guarded_profile_relation

    # Even if both endpoints are infeasible, a better option spectrum must not
    # authorize an infeasible recovery action.
    strict, weak, delta, _margin, area = _kinematic_guarded_profile_relation(
        False, False, (1, 1, 0), (3, 3, 2)
    )
    assert not strict and not weak and delta == 0 and area > 0

    # If the alternative itself is feasible and restores feasibility relative to
    # base, an equal future spectrum is a strict product-order improvement.
    strict, weak, delta, margin, area = _kinematic_guarded_profile_relation(
        True, False, (2, 1), (2, 1)
    )
    assert strict and weak and delta == 1 and margin == 0 and area == 0


def test_control_projection_carries_stateful_jerk_memory():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import _project_candidate_bank_through_controller_np

    cfg = _cfg()
    cur = _state(10.0)
    nominal = np.zeros((2, 4, 7), dtype=np.float32)
    nominal[:, :, 0] = np.arange(1, 5, dtype=np.float32)[None, :] * 1.4
    nominal[:, :, 3] = 14.0
    nominal[:, :, 5] = 4.5
    nominal[:, :, 6] = 1.9

    projected, kim_ok, accel = _project_candidate_bank_through_controller_np(
        cur, nominal, cfg, previous_longitudinal_accel=0.0
    )
    assert projected.shape == nominal.shape
    assert kim_ok.shape == (2, 4)
    # max_jerk=8 m/s^3 at dt=.1 -> acceleration can move by .8 m/s^2 per step.
    assert np.allclose(accel[0, :4], [0.8, 1.6, 2.4, 3.2], atol=1.0e-5)
    assert np.all(kim_ok)
    # The projected speed follows controller memory rather than jumping to 14.
    assert np.linalg.norm(projected[0, 0, 3:5]) < 10.1



def test_control_projection_first_step_matches_online_controller_exactly():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import (
        _consistent_one_step_targets_np,
        _project_candidate_bank_through_controller_np,
    )

    cfg = _cfg()
    cur = _state(7.0)
    prev_accel = -0.35
    nominal = np.zeros((4, 3, 7), dtype=np.float32)
    nominal[:, :, 5] = 4.5
    nominal[:, :, 6] = 1.9
    desired_speed = np.asarray([3.0, 6.0, 9.0, 13.0], dtype=np.float32)
    desired_yaw = np.asarray([-0.18, -0.03, 0.08, 0.24], dtype=np.float32)
    for k in range(4):
        for t in range(3):
            sp = float(desired_speed[k] + 0.25 * t)
            yaw = float(desired_yaw[k] + 0.015 * t)
            nominal[k, t, 0] = 0.1 * sp * (t + 1)
            nominal[k, t, 1] = 0.03 * k
            nominal[k, t, 2] = yaw
            nominal[k, t, 3] = sp * np.cos(yaw)
            nominal[k, t, 4] = sp * np.sin(yaw)

    online_target, online_accel, _ = _consistent_one_step_targets_np(
        cur, nominal[:, 0, :5], cfg, previous_longitudinal_accel=prev_accel
    )
    projected, _kim_ok, accel_hist = _project_candidate_bank_through_controller_np(
        cur, nominal, cfg, previous_longitudinal_accel=prev_accel
    )
    assert np.allclose(projected[:, 0, :5], online_target, atol=1.0e-6, rtol=0.0)
    assert np.allclose(accel_hist[:, 0], online_accel, atol=1.0e-6, rtol=0.0)

def test_control_projected_profile_uses_min_collision_and_kinematic_survival(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw

    cfg = _cfg()
    state = np.zeros((2, 11), dtype=np.float32)
    state[1] = _state(5.0)

    def fake_successor(agent_state, sdc_index, emitted_target, cfg):
        return np.array(agent_state, copy=True)

    def fake_candidates(agent_state, sdc_index, roadgraph, cfg, other_future_trajs=None):
        traj = np.zeros((3, 5, 7), dtype=np.float32)
        for i in range(3):
            traj[i, :, 0] = float(i + 1)  # identify candidate in fake collision audit
            traj[i, :, 3] = 5.0
            traj[i, :, 5] = 4.5
            traj[i, :, 6] = 1.9
        valid = np.ones(3, dtype=bool)
        conventional = np.zeros(3, dtype=bool)
        macro = np.asarray([1, 2, 3], dtype=np.int64)
        utility = np.zeros(3, dtype=np.float32)
        road = np.ones(3, dtype=bool)
        collision = np.zeros(3, dtype=bool)
        prefix = np.asarray([5, 5, 5], dtype=np.int32)
        margin = np.zeros(3, dtype=np.float32)
        return traj, valid, conventional, macro, utility, road, collision, prefix, margin

    def fake_project(current, nominal, cfg, previous_longitudinal_accel):
        kin = np.ones((3, 5), dtype=bool)
        kin[1, 2:] = False  # macro 2 survives only two transitions
        return np.array(nominal, copy=True), kin, np.zeros((3, 5), dtype=np.float32)

    def fake_collision(traj, context):
        cid = int(round(float(traj[0, 0])))
        return {"safe_prefix_steps": {1: 5, 2: 4, 3: 3}[cid]}

    monkeypatch.setattr(pw, "_counterfactual_successor_agent_state", fake_successor)
    monkeypatch.setattr(pw, "_route_lane_aware_candidates", fake_candidates)
    monkeypatch.setattr(pw, "_project_candidate_bank_through_controller_np", fake_project)
    monkeypatch.setattr(pw, "_prepare_collision_check_context", lambda *a, **k: {})
    monkeypatch.setattr(pw, "_roadgraph_drivable_mask", lambda *a, **k: True)
    monkeypatch.setattr(pw, "_collision_audit_against_context", fake_collision)

    curve, detail = pw._successor_control_projected_option_profile(
        state, 1, np.zeros(5, dtype=np.float32), 0.0, {}, cfg
    )
    # Effective macro prefixes: m1=min(5,5)=5, m2=min(4,2)=2, m3=min(3,5)=3.
    assert curve == (3, 3, 2, 1, 1)
    assert detail["recovery_macro_types_any"] == 3
    assert detail["control_projected_max_realized_prefix_steps"] == 5


def test_v35_methods_keep_priority_gate_defaults():
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    from cowp.waymax_eval.rollout import _method_gate_defaults

    for method in (
        "cowp_waymax_kinematic_guarded_rosh",
        "cowp_control_projected_option_spectrum_hysteresis",
    ):
        assert _canonical_online_method(method, "hard") == (method, "priority")
        assert _method_gate_defaults(method, "hard") == (method, "priority")
