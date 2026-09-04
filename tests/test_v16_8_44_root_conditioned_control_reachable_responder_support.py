from types import SimpleNamespace

import numpy as np

import cowp.waymax_eval.policy_wrapper as pw


def _traj(h=8):
    x = np.zeros((h, 7), dtype=np.float32)
    x[:, 5] = 4.5
    x[:, 6] = 1.8
    return x


def test_control_reachable_same_root_search_finds_minimum_safe_residual(monkeypatch):
    root = _traj()
    ego = _traj()
    state = np.zeros((2, 11), dtype=np.float32)
    state[:, 10] = 1.0

    # The identity root is unsafe through step 4, which deterministically sets
    # the event-conditioned control duration without introducing a grid.
    monkeypatch.setattr(
        pw,
        "unsafe_between",
        lambda *args, **kwargs: SimpleNamespace(event_mask=np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)),
    )
    monkeypatch.setattr(pw, "_shift_append_terminal_reference_np", lambda tr, dt: np.array(tr, copy=True))
    monkeypatch.setattr(pw, "_agent_state_after_future_sample_np", lambda cur, sample: np.array(cur, copy=True))
    monkeypatch.setattr(pw, "_roadgraph_drivable_mask", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        pw,
        "_trajectory_waymax_kinematic_safe_np",
        lambda *args, **kwargs: (True, {"failure_step": -1}),
    )

    def fake_build(root, cfg, *, accel_mps2, duration_s, start_delay_s=0.0):
        tr = np.array(root, copy=True)
        tr[:, 0] = float(accel_mps2)  # encode searched control for the fake safety predicate
        return tr

    monkeypatch.setattr(pw, "build_root_control_residual_trajectory", fake_build)
    monkeypatch.setattr(
        pw,
        "prepare_root_recovery_burden_bank",
        lambda root, bank, cfg, object_type, rho: [(0.1 * abs(float(bank[0][0, 0])), np.zeros(4, dtype=np.float32))],
    )

    # Safe iff the same-root residual has magnitude >= 1.0 m/s^2.
    def fake_unsafe_bool(left, right, cfg, agent_type=1, **kwargs):
        encoded = max(abs(float(np.asarray(left)[0, 0])), abs(float(np.asarray(right)[0, 0])))
        return encoded < 1.0

    monkeypatch.setattr(pw, "unsafe_between_bool", fake_unsafe_bool)

    profiles, detail = pw._root_conditioned_control_reachable_response_profiles_np(
        state,
        1,
        1,
        root,
        1.0,
        ego,
        ego,
        [],
        {},
        {"time": {"dt": 0.1}, "candidate": {"max_decel_mps2": 2.0, "max_accel_mps2": 2.0}},
    )

    assert detail["attempted"] is True
    assert detail["event_last_step"] == 4
    assert detail["duration_s"] == 0.5
    assert detail["profiles_found"] == 2
    assert len(profiles) == 2
    # Bisection should approach the first safe boundary from above rather than
    # choosing the full inherited physical bound.
    assert all(1.0 <= abs(float(p["residual_accel_mps2"])) < 1.01 for p in profiles)
    assert all(p["control_reachable_extension"] for p in profiles)


def test_certificate_extension_is_fail_closed_and_only_changes_root_unrecoverable_case(monkeypatch):
    tr = _traj(6)
    state = np.zeros((2, 11), dtype=np.float32)
    state[:, 10] = 1.0

    monkeypatch.setattr(
        pw,
        "_collision_blocking_agent_indices_against_context",
        lambda *args, **kwargs: ([1], {"blockers": [1]}),
    )
    monkeypatch.setattr(
        pw,
        "_physical_recovery_tube_certificate_np",
        lambda *args, **kwargs: (True, {"collision_min_margin_m": 1.0}),
    )
    monkeypatch.setattr(pw, "unsafe_between_bool", lambda *args, **kwargs: False)

    adaptive = {
        "profile_index": 10000,
        "trajectory": tr,
        "shifted_trajectory": tr,
        "burden": 0.2,
        "burden_components": np.zeros(4, dtype=np.float32),
        "control_reachable_extension": True,
    }
    monkeypatch.setattr(
        pw,
        "_root_conditioned_control_reachable_response_profiles_np",
        lambda *args, **kwargs: ([adaptive], {"profiles_found": 1, "profile_evaluations": 5}),
    )

    support = {
        1: {
            "ready": True,
            "object_type": 1,
            "beta": 1.0,
            "retained_mass": 0.8,
            "roots": [{"trajectory": tr, "profiles": []}],
        }
    }
    context = {"agents": []}
    object_types = np.array([1, 1], dtype=np.int32)

    ok_legacy, legacy = pw._interaction_aware_recovery_certificate_np(
        state, state, 0, tr, np.ones(1, dtype=bool), tr, np.ones(1, dtype=bool),
        {}, {}, context, context, support, object_types,
        allow_control_reachable_response_extension=False,
    )
    assert ok_legacy is False
    assert legacy["failure_reason"] == "retained_root_has_no_ego_safe_response"
    assert legacy["control_reachable_response_attempts"] == 0

    ok_v44, v44 = pw._interaction_aware_recovery_certificate_np(
        state, state, 0, tr, np.ones(1, dtype=bool), tr, np.ones(1, dtype=bool),
        {}, {}, context, context, support, object_types,
        allow_control_reachable_response_extension=True,
    )
    assert ok_v44 is True
    assert v44["control_reachable_response_attempts"] == 1
    assert v44["control_reachable_response_profiles_found"] == 1
    assert v44["control_reachable_response_selected_roots"] == 1
    assert v44["selected_responses"][0]["profile_index"] == 10000
