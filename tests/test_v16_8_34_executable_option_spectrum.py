from __future__ import annotations


def _state(speed: float = 10.0):
    import numpy as np
    x = np.zeros((11,), dtype=np.float32)
    x[3] = speed
    x[5] = speed
    x[6] = 0.0
    x[7] = 4.5
    x[8] = 1.9
    x[10] = 1.0
    return x


def _desired(speed: float):
    import numpy as np
    d = np.zeros((5,), dtype=np.float32)
    d[0] = speed * 0.1
    d[2] = 0.0
    d[3] = speed
    return d


def test_controller_transition_feasible_uses_previous_acceleration_state():
    import numpy as np
    from cowp.waymax_eval.policy_wrapper import _controller_transition_feasible_np

    cfg = {
        "time": {"dt": 0.1},
        "candidate": {
            "max_accel_mps2": 4.0,
            "max_decel_mps2": 6.0,
            "max_jerk_mps3": 8.0,
            "max_yaw_rate_rad_s": 1.2,
            "max_lateral_accel_mps2": 4.0,
        },
        "waymax": {"max_delta_yaw_rad": 0.12},
    }
    cur = _state(10.0)
    desired = np.stack([_desired(10.4), _desired(10.0)], axis=0)

    # 10 -> 10.4 m/s requests +4 m/s^2.  From a previous acceleration of 0,
    # the 0.1 s jerk envelope only permits +/-0.8 m/s^2, so the transition is
    # not immediately executable even though +4 itself is within the accel cap.
    ok0 = _controller_transition_feasible_np(cur, desired, cfg, 0.0)
    assert ok0.tolist() == [False, True]

    # Carrying controller memory changes the answer without changing agent_state.
    ok35 = _controller_transition_feasible_np(cur, desired, cfg, 3.5)
    assert ok35.tolist() == [True, False]


def test_executable_successor_profile_carries_emitted_acceleration(monkeypatch):
    import numpy as np
    import cowp.waymax_eval.policy_wrapper as pw

    cfg = {
        "time": {"dt": 0.1},
        "candidate": {
            "max_accel_mps2": 4.0,
            "max_decel_mps2": 6.0,
            "max_jerk_mps3": 8.0,
            "max_yaw_rate_rad_s": 1.2,
            "max_lateral_accel_mps2": 4.0,
        },
        "waymax": {"max_delta_yaw_rad": 0.12},
    }

    def fake_successor(agent_state, sdc_index, emitted_target, cfg):
        out = np.array(agent_state, copy=True)
        out[sdc_index] = _state(10.0)
        return out

    def fake_candidates(agent_state, sdc_index, roadgraph, cfg, other_future_trajs=None):
        traj = np.zeros((3, 5, 5), dtype=np.float32)
        for i, spd in enumerate((10.4, 10.3, 10.0)):
            traj[i, :, 2] = 0.0
            traj[i, :, 3] = spd
            traj[i, :, 0] = np.arange(1, 6) * 0.1 * spd
        valid = np.ones((3,), dtype=bool)
        conventional = np.zeros((3,), dtype=bool)
        macro = np.asarray([1, 2, 3], dtype=np.int64)
        utility = np.zeros((3,), dtype=np.float32)
        road = np.ones((3,), dtype=bool)
        collision = np.zeros((3,), dtype=bool)
        prefix = np.asarray([5, 5, 5], dtype=np.int32)
        margin = np.zeros((3,), dtype=np.float32)
        return traj, valid, conventional, macro, utility, road, collision, prefix, margin

    monkeypatch.setattr(pw, "_counterfactual_successor_agent_state", fake_successor)
    monkeypatch.setattr(pw, "_route_lane_aware_candidates", fake_candidates)
    state = np.zeros((2, 11), dtype=np.float32)
    state[1] = _state(10.0)

    p0, d0 = pw._successor_executable_recovery_option_profile(
        state, 1, np.zeros((5,), dtype=np.float32), 0.0, {}, cfg
    )
    p35, d35 = pw._successor_executable_recovery_option_profile(
        state, 1, np.zeros((5,), dtype=np.float32), 3.5, {}, cfg
    )
    # prev_accel=0 keeps only the zero-acceleration option; prev_accel=3.5 keeps
    # the +3/+4 acceleration options.  Same observable agent state, different
    # controller state -> different executable option set.
    assert p0 == (1, 1, 1, 1, 1)
    assert p35 == (2, 2, 2, 2, 2)
    assert d0["transition_rejected_roadgraph_candidates"] == 2
    assert d35["transition_rejected_roadgraph_candidates"] == 1


def test_execution_spectrum_relation_is_product_order_not_weighted_compensation():
    from cowp.waymax_eval.policy_wrapper import _execution_spectrum_relation

    # Future spectrum improvement cannot compensate for losing current hard
    # controller-transition feasibility.
    strict, weak, tdelta, min_margin, area = _execution_spectrum_relation(
        True, False, (2, 2, 1), (4, 4, 4)
    )
    assert not strict and not weak and tdelta == -1 and area > 0

    # A transition-feasibility improvement is a strict improvement when future
    # option support is non-regressive, even on an exact spectrum tie.
    strict, weak, tdelta, min_margin, area = _execution_spectrum_relation(
        False, True, (2, 2, 1), (2, 2, 1)
    )
    assert strict and weak and tdelta == 1 and min_margin == 0 and area == 0


def test_v34_methods_keep_priority_gate_defaults():
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    from cowp.waymax_eval.rollout import _method_gate_defaults

    for method in (
        "cowp_transition_guarded_rosh",
        "cowp_executable_option_spectrum_hysteresis",
    ):
        assert _canonical_online_method(method, "hard") == (method, "priority")
        assert _method_gate_defaults(method, "hard") == (method, "priority")
