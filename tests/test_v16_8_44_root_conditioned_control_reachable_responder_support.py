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



def test_dynamic_profile_joint_cache_identity_is_hypothesis_specific(monkeypatch):
    """A V44 dynamic profile must never reuse a CSP result from another hypothesis.

    The pre-repair implementation keyed the shared joint cache only by
    (agent, root, profile_index).  Dynamic completions reuse profile indices, so
    a pair that was safe for one ego hypothesis could be spuriously treated as
    safe for a later hypothesis with different response trajectories.
    """
    tr = _traj(6)
    state = np.zeros((3, 11), dtype=np.float32)
    state[:, 10] = 1.0

    monkeypatch.setattr(
        pw,
        "_collision_blocking_agent_indices_against_context",
        lambda *args, **kwargs: ([1, 2], {"blockers": [1, 2]}),
    )
    monkeypatch.setattr(
        pw,
        "_physical_recovery_tube_certificate_np",
        lambda *args, **kwargs: (True, {"collision_min_margin_m": 1.0}),
    )

    # Pairwise compatibility is encoded by the first x value.  Profiles 1 and 3
    # are compatible; profiles 5 and 5.1 are not.
    def fake_unsafe_bool(left, right, cfg, agent_type=1, **kwargs):
        a = float(np.asarray(left, dtype=np.float32)[0, 0])
        b = float(np.asarray(right, dtype=np.float32)[0, 0])
        return abs(a - b) < 0.5

    monkeypatch.setattr(pw, "unsafe_between_bool", fake_unsafe_bool)

    def fake_completion(
        agent_state, agent_index, object_type, root, beta,
        current_ego_trajectory, shifted_ego_trajectory, environment,
        roadgraph, cfg, **kwargs,
    ):
        marker = float(np.asarray(current_ego_trajectory)[0, 0])
        if marker < 1.0:
            value = 1.0 if int(agent_index) == 1 else 3.0
        else:
            value = 5.0 if int(agent_index) == 1 else 5.1
        out = np.array(root, copy=True)
        out[:, 0] = value
        rec = {
            "profile_index": int(kwargs.get("profile_index_base", 10000)),
            "trajectory": out,
            "shifted_trajectory": out,
            "burden": 0.1,
            "burden_components": np.zeros(4, dtype=np.float32),
            "control_reachable_extension": True,
            # Deliberately hypothesis-specific while profile_index stays reused.
            "residual_accel_mps2": value,
            "residual_duration_s": 0.5,
        }
        return [rec], {
            "profiles_found": 1,
            "profile_evaluations": 1,
            "static_profile_cache_hits": 0,
            "environment_compatibility_cache_hits": 0,
        }

    monkeypatch.setattr(
        pw, "_root_conditioned_control_reachable_response_profiles_np", fake_completion
    )

    support = {
        1: {"ready": True, "object_type": 1, "beta": 1.0, "retained_mass": 0.8,
            "roots": [{"trajectory": tr, "profiles": []}]},
        2: {"ready": True, "object_type": 1, "beta": 1.0, "retained_mass": 0.8,
            "roots": [{"trajectory": tr, "profiles": []}]},
    }
    object_types = np.array([1, 1, 1], dtype=np.int32)
    cache = {"environment": {}, "joint": {}, "control_reachable_static": {}}
    context = {"agents": []}

    ego_a = np.array(tr, copy=True)
    ego_a[:, 0] = 0.0
    ok_a, detail_a = pw._interaction_aware_recovery_certificate_np(
        state, state, 0, ego_a, np.ones(1, dtype=bool), ego_a, np.ones(1, dtype=bool),
        {}, {}, context, context, support, object_types,
        compatibility_cache=cache, allow_control_reachable_response_extension=True,
    )
    assert ok_a is True
    assert detail_a["interaction_joint_compatibility_checks"] > 0

    ego_b = np.array(tr, copy=True)
    ego_b[:, 0] = 9.0
    ok_b, detail_b = pw._interaction_aware_recovery_certificate_np(
        state, state, 0, ego_b, np.ones(1, dtype=bool), ego_b, np.ones(1, dtype=bool),
        {}, {}, context, context, support, object_types,
        compatibility_cache=cache, allow_control_reachable_response_extension=True,
    )
    assert ok_b is False
    assert detail_b["failure_reason"] == "no_jointly_compatible_response_envelope"
    # A stale (agent, root, profile_index)-only cache would make this a hit and
    # incorrectly preserve the first hypothesis' True compatibility decision.
    assert detail_b["interaction_joint_compatibility_cache_hits"] == 0


def test_control_reachable_static_work_is_reused_without_changing_logical_evaluations(monkeypatch):
    root = _traj()
    ego = _traj()
    state = np.zeros((2, 11), dtype=np.float32)
    state[:, 10] = 1.0

    monkeypatch.setattr(
        pw,
        "unsafe_between",
        lambda *args, **kwargs: SimpleNamespace(
            event_mask=np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
        ),
    )
    monkeypatch.setattr(pw, "_shift_append_terminal_reference_np", lambda tr, dt: np.array(tr, copy=True))
    monkeypatch.setattr(pw, "_agent_state_after_future_sample_np", lambda cur, sample: np.array(cur, copy=True))
    monkeypatch.setattr(pw, "_roadgraph_drivable_mask", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        pw,
        "_trajectory_waymax_kinematic_safe_np",
        lambda *args, **kwargs: (True, {"failure_step": -1}),
    )

    build_calls = {"n": 0}

    def fake_build(root, cfg, *, accel_mps2, duration_s, start_delay_s=0.0):
        build_calls["n"] += 1
        tr = np.array(root, copy=True)
        tr[:, 0] = float(accel_mps2)
        return tr

    monkeypatch.setattr(pw, "build_root_control_residual_trajectory", fake_build)
    monkeypatch.setattr(
        pw,
        "prepare_root_recovery_burden_bank",
        lambda root, bank, cfg, object_type, rho: [
            (0.1 * abs(float(bank[0][0, 0])), np.zeros(4, dtype=np.float32))
        ],
    )
    monkeypatch.setattr(
        pw,
        "unsafe_between_bool",
        lambda left, right, cfg, agent_type=1, **kwargs: (
            max(abs(float(np.asarray(left)[0, 0])), abs(float(np.asarray(right)[0, 0]))) < 1.0
        ),
    )

    cache = {"environment": {}, "joint": {}, "control_reachable_static": {}}
    p1, d1 = pw._root_conditioned_control_reachable_response_profiles_np(
        state, 1, 1, root, 1.0, ego, ego, [], {},
        {"time": {"dt": 0.1}, "candidate": {"max_decel_mps2": 2.0, "max_accel_mps2": 2.0}},
        root_ordinal=0, compatibility_cache=cache,
    )
    first_build_calls = build_calls["n"]
    p2, d2 = pw._root_conditioned_control_reachable_response_profiles_np(
        state, 1, 1, root, 1.0, ego, ego, [], {},
        {"time": {"dt": 0.1}, "candidate": {"max_decel_mps2": 2.0, "max_accel_mps2": 2.0}},
        root_ordinal=0, compatibility_cache=cache,
    )

    assert len(p1) == len(p2) == 2
    assert d1["profile_evaluations"] == d2["profile_evaluations"]
    assert build_calls["n"] == first_build_calls
    assert d2["static_profile_cache_hits"] == d2["profile_evaluations"]


def test_control_reachable_environment_event_support_is_reused_exactly(monkeypatch):
    root = _traj()
    ego = _traj()
    state = np.zeros((3, 11), dtype=np.float32)
    state[:, 10] = 1.0
    calls = {"unsafe_between": 0}

    def fake_unsafe_between(*args, **kwargs):
        calls["unsafe_between"] += 1
        return SimpleNamespace(event_mask=np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=bool))

    monkeypatch.setattr(pw, "unsafe_between", fake_unsafe_between)
    monkeypatch.setattr(pw, "_shift_append_terminal_reference_np", lambda tr, dt: np.array(tr, copy=True))
    monkeypatch.setattr(pw, "_agent_state_after_future_sample_np", lambda cur, sample: np.array(cur, copy=True))
    monkeypatch.setattr(pw, "_roadgraph_drivable_mask", lambda *args, **kwargs: True)
    monkeypatch.setattr(pw, "_trajectory_waymax_kinematic_safe_np", lambda *args, **kwargs: (True, {}))
    monkeypatch.setattr(pw, "build_root_control_residual_trajectory", lambda root, cfg, **kwargs: np.array(root, copy=True))
    monkeypatch.setattr(
        pw, "prepare_root_recovery_burden_bank",
        lambda root, bank, cfg, object_type, rho: [(0.1, np.zeros(4, dtype=np.float32))],
    )
    monkeypatch.setattr(pw, "unsafe_between_bool", lambda *args, **kwargs: False)

    actor = {
        "agent_index": 2,
        "object_type": 1,
        "trajectory": np.array(root, copy=True),
        "shifted_trajectory": np.array(root, copy=True),
    }
    cache = {
        "environment": {}, "joint": {}, "control_reachable_static": {},
        "control_reachable_environment_events": {},
    }
    kwargs = dict(
        agent_state=state, agent_index=1, object_type=1, root=root, beta=1.0,
        current_ego_trajectory=ego, shifted_ego_trajectory=ego,
        environment=[actor], roadgraph={},
        cfg={"time": {"dt": 0.1}, "candidate": {"max_decel_mps2": 1.0, "max_accel_mps2": 1.0}},
        root_ordinal=0, compatibility_cache=cache,
    )
    _, d1 = pw._root_conditioned_control_reachable_response_profiles_np(**kwargs)
    first_calls = calls["unsafe_between"]
    _, d2 = pw._root_conditioned_control_reachable_response_profiles_np(**kwargs)
    second_calls = calls["unsafe_between"] - first_calls

    # First call: 2 ego/root event checks + 4 root/environment directions.
    assert first_calls == 6
    # Second call recomputes only ego-conditioned event support; environment is invariant.
    assert second_calls == 2
    assert d1["environment_event_cache_hits"] == 0
    assert d2["environment_event_cache_hits"] == 1
