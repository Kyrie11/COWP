from __future__ import annotations

import inspect

import numpy as np

import cowp.waymax_eval.policy_wrapper as pw
from cowp.label.audit_relevance import canonical_root_weights
from cowp.waymax_eval.metrics_cowp import policy_diagnostic_scenario_rows


def _state(n: int = 3) -> np.ndarray:
    state = np.zeros((n, 11), dtype=np.float32)
    state[:, 7] = 4.5
    state[:, 8] = 2.0
    state[:, 10] = 1.0
    return state


def _traj(marker: float, horizon: int = 4) -> np.ndarray:
    tr = np.zeros((horizon, 7), dtype=np.float32)
    tr[:, 0] = marker + np.arange(horizon, dtype=np.float32)
    tr[:, 3] = 1.0
    tr[:, 5] = 4.5
    tr[:, 6] = 2.0
    return tr


def _profile(marker: float, burden: float = 0.1, profile_index: int = 0) -> dict:
    tr = _traj(marker)
    return {
        "profile_index": profile_index,
        "trajectory": tr,
        "shifted_trajectory": tr.copy(),
        "burden": burden,
        "burden_components": np.zeros(6, dtype=np.float32),
        "current_kinematic": {},
        "shifted_kinematic": {},
    }


def _support(agent_index: int, markers: list[float]) -> dict:
    return {
        "ready": True,
        "object_type": 1,
        "retained_mass": 0.8,
        "roots": [
            {
                "weight": 0.4,
                "mode_indices": (i,),
                "profiles": [_profile(marker, profile_index=i)],
            }
            for i, marker in enumerate(markers)
        ],
    }


def test_v42_method_is_priority_hard_path() -> None:
    method, gate = pw._canonical_online_method(
        "cowp_interaction_aware_reachable_response_envelope", "hard"
    )
    assert method == "cowp_interaction_aware_reachable_response_envelope"
    assert gate == "priority"


def test_stable_softmax_is_finite_and_order_preserving() -> None:
    probs = pw._stable_softmax_np(np.asarray([1000.0, 999.0, np.nan], dtype=np.float32))
    assert np.isfinite(probs).all()
    assert probs[0] > probs[1] > probs[2]
    np.testing.assert_allclose(probs.sum(), 1.0, rtol=0.0, atol=1e-6)


def test_response_support_retains_high_mass_same_root_set(monkeypatch) -> None:
    monkeypatch.setattr(pw, "build_root_recovery_trajectory_bank", lambda root, cfg: [root.copy()])
    monkeypatch.setattr(
        pw,
        "prepare_root_recovery_burden_bank",
        lambda root, bank, cfg, object_type, rho: [(0.1, np.zeros(6, dtype=np.float32))],
    )
    monkeypatch.setattr(pw, "_roadgraph_drivable_mask", lambda tr, roadgraph: True)
    monkeypatch.setattr(
        pw,
        "_trajectory_waymax_kinematic_safe_np",
        lambda current, trajectory, cfg: (True, {"failure_step": -1}),
    )

    natural = np.stack([_traj(0.0), _traj(10.0), _traj(20.0)], axis=0)[None, ...]
    logits = np.log(np.asarray([[0.50, 0.30, 0.20]], dtype=np.float32))
    cfg = {
        "time": {"dt": 0.1},
        "planning": {
            "set_transport_probability_floor": 0.02,
            "set_transport_min_alt_weight": 0.03,
            "set_transport_cvar_tail_mass": 0.25,
        },
        "natural": {
            "certificate_min_low_burden_roots": 2,
            "root_dedup_mean_distance_m": 0.10,
        },
        "response": {"root_conditioned_transport": {"max_roots_per_agent": 3}},
        "burden": {},
    }
    support, detail = pw._prepare_interaction_response_support_np(
        _state(2), 0,
        np.asarray([1], dtype=np.int64),
        np.asarray([True]),
        natural, logits,
        np.asarray([1, 1], dtype=np.int32),
        {}, cfg,
    )
    assert detail["agents_ready"] == 1
    assert support[1]["ready"] is True
    assert support[1]["retained_root_count"] == 2
    assert support[1]["retained_mass"] >= 0.75
    assert all(root["profiles"] for root in support[1]["roots"])


def test_interaction_certificate_accepts_universal_single_blocker_envelope(monkeypatch) -> None:
    monkeypatch.setattr(
        pw,
        "_collision_blocking_agent_indices_against_context",
        lambda trajectory, context: (list(context["blockers"]), {"blocker_count": len(context["blockers"])}),
    )
    monkeypatch.setattr(
        pw,
        "_physical_recovery_tube_certificate_np",
        lambda *args, **kwargs: (True, {"collision_min_margin_m": 1.0}),
    )
    monkeypatch.setattr(pw, "unsafe_between_bool", lambda *args, **kwargs: False)

    ok, detail = pw._interaction_aware_recovery_certificate_np(
        _state(3), _state(3), 0,
        _traj(100.0), np.ones(4, dtype=bool),
        _traj(101.0), np.ones(4, dtype=bool),
        {}, {}, {"blockers": [1]}, {"blockers": [1]},
        {1: _support(1, [10.0, 20.0])},
        np.asarray([1, 1, 1], dtype=np.int32),
    )
    assert ok is True
    assert detail["supported_root_count"] == 2
    assert len(detail["selected_responses"]) == 2
    assert detail["maximum_selected_response_burden"] <= 0.1 + 1e-8


def test_interaction_certificate_rejects_nonjoint_multiagent_responses(monkeypatch) -> None:
    monkeypatch.setattr(
        pw,
        "_collision_blocking_agent_indices_against_context",
        lambda trajectory, context: (list(context["blockers"]), {"blocker_count": len(context["blockers"])}),
    )
    monkeypatch.setattr(
        pw,
        "_physical_recovery_tube_certificate_np",
        lambda *args, **kwargs: (True, {"collision_min_margin_m": 1.0}),
    )

    def fake_unsafe(a, b, cfg, agent_type=1, agent_lane_dist=None):
        ma = int(round(float(np.asarray(a)[0, 0])))
        mb = int(round(float(np.asarray(b)[0, 0])))
        # Ego markers are far away.  The only two blocker profiles conflict.
        return {ma, mb} == {10, 20}

    monkeypatch.setattr(pw, "unsafe_between_bool", fake_unsafe)
    ok, detail = pw._interaction_aware_recovery_certificate_np(
        _state(3), _state(3), 0,
        _traj(100.0), np.ones(4, dtype=bool),
        _traj(101.0), np.ones(4, dtype=bool),
        {}, {}, {"blockers": [1, 2]}, {"blockers": [1, 2]},
        {1: _support(1, [10.0]), 2: _support(2, [20.0])},
        np.asarray([1, 1, 1], dtype=np.int32),
    )
    assert ok is False
    assert detail["failure_reason"] == "no_jointly_compatible_response_envelope"
    assert detail["interaction_joint_compatibility_rejects"] > 0


def test_v42_preserves_nested_v39_certificate_exactly(monkeypatch) -> None:
    sentinel = {"parent_index": 3, "target": np.ones(5, dtype=np.float32)}
    nested_detail = {"selected": True, "selected_policy_name": "LOWER_TO_LAST_CONFLICT"}
    monkeypatch.setattr(
        pw,
        "_construct_conflict_window_control_reachable_tube_np",
        lambda *args, **kwargs: (sentinel, nested_detail),
    )
    monkeypatch.setattr(
        pw,
        "_prepare_interaction_response_support_np",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("extension must not run")),
    )
    selected, detail = pw._construct_interaction_aware_reachable_response_envelope_np(
        _state(2), 0,
        np.zeros((1, 4, 7), dtype=np.float32),
        np.asarray([True]), np.asarray([True]), np.asarray([0]),
        np.asarray([0.0]), np.asarray([0.0]),
        np.zeros((1, 5), dtype=np.float32), np.asarray([0.0]),
        {}, {}, 0.0,
        base_candidate_index=0,
        critical_track_index=np.asarray([1]), critical_valid=np.asarray([True]),
        natural_trajectories=np.zeros((1, 2, 4, 7), dtype=np.float32),
        natural_logits=np.zeros((1, 2), dtype=np.float32),
        object_types=np.asarray([1, 1]),
    )
    assert selected is sentinel
    assert detail["nested_v39_selected"] is True
    assert detail["interaction_response_attempted"] is False
    assert detail["selected_certificate_kind"] == "nested_v39"


def test_v42_constructor_is_causal_and_does_not_use_dense_response_prediction() -> None:
    source = inspect.getsource(pw._construct_interaction_aware_reachable_response_envelope_np)
    source += inspect.getsource(pw._prepare_interaction_response_support_np)
    assert "log_trajectory" not in source
    assert "ground_truth" not in source.lower()
    assert 'pred["response"]' not in source
    assert "natural_trajectories" in source



def test_online_root_weights_exactly_match_frozen_canonical_measure() -> None:
    rng = np.random.default_rng(42)
    cfg = {
        "ncf": {"min_alt_weight": 0.03, "root_probability_floor": 0.02},
        "planning": {"set_transport_min_alt_weight": 0.03, "set_transport_probability_floor": 0.02},
    }
    for _ in range(50):
        logits = rng.normal(size=8).astype(np.float32)
        valid = rng.random(8) > 0.2
        if not valid.any():
            valid[0] = True
        got, raw = pw._canonical_online_root_weights_np(logits, valid, cfg)
        expected = canonical_root_weights(
            {"valid": valid[None, :], "weight": raw[None, :]}, cfg
        )[0]
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(got.sum(), 1.0, rtol=0.0, atol=1e-6)


def test_interaction_certificate_rejects_profile_unsafe_to_nonblocker(monkeypatch) -> None:
    current_context = {
        "blockers": [1],
        "agents": [{"index": 1}, {"index": 2}],
    }
    shifted_context = {
        "blockers": [1],
        "agents": [{"index": 1}, {"index": 2}],
    }
    monkeypatch.setattr(
        pw, "_collision_blocking_agent_indices_against_context",
        lambda trajectory, context: (list(context["blockers"]), {"blocker_count": len(context["blockers"])}),
    )
    monkeypatch.setattr(
        pw, "_physical_recovery_tube_certificate_np",
        lambda *args, **kwargs: (True, {"collision_min_margin_m": 1.0}),
    )

    def fake_unsafe(a, b, cfg, agent_type=1, agent_lane_dist=None):
        ma = int(round(float(np.asarray(a)[0, 0])))
        mb = int(round(float(np.asarray(b)[0, 0])))
        # Profile marker 10 is incompatible with the non-blocking actor's CV path.
        return ma == 10 and mb <= 5

    monkeypatch.setattr(pw, "unsafe_between_bool", fake_unsafe)
    state = _state(3)
    state[2, 0] = 0.0
    ok, detail = pw._interaction_aware_recovery_certificate_np(
        state, state.copy(), 0,
        _traj(100.0), np.ones(4, dtype=bool),
        _traj(101.0), np.ones(4, dtype=bool),
        {}, {}, current_context, shifted_context,
        {1: _support(1, [10.0])},
        np.asarray([1, 1, 1], dtype=np.int32),
    )
    assert ok is False
    assert detail["failure_reason"] == "retained_root_has_no_environment_safe_response"
    assert detail["interaction_environment_compatibility_checks"] > 0
    assert detail["interaction_environment_compatibility_rejects"] > 0

def test_v42_diagnostics_aggregate_interaction_mechanism() -> None:
    rollouts = [{
        "scenario_id": "scene",
        "steps": 1,
        "standard_metrics": {"CollisionRate": 0.0, "EP": 1.0},
        "policy_diagnostics": [{
            "fallback_used": True,
            "fallback_reason": "no_conventional_use_interaction_aware_reachable_response_envelope",
            "valid_candidates": 3,
            "conventional_candidates": 0,
            "roadgraph_safe_candidates": 2,
            "collision_safe_candidates": 0,
            "recovery_tube_probe_used": True,
            "recovery_tube_selected": True,
            "recovery_tube_action_changed": True,
            "recovery_tube_nested_v39_selected": False,
            "recovery_tube_interaction_response_attempted": True,
            "recovery_tube_selected_is_interaction_response": True,
            "recovery_tube_interaction_support_agents_total": 2,
            "recovery_tube_interaction_support_agents_ready": 2,
            "recovery_tube_interaction_support_retained_roots": 4,
            "recovery_tube_interaction_support_eligible_profiles": 8,
            "recovery_tube_interaction_hypotheses_evaluated": 5,
            "recovery_tube_interaction_environment_compatibility_checks": 7,
            "recovery_tube_interaction_environment_compatibility_rejects": 1,
            "recovery_tube_interaction_selected_blocker_count": 1,
            "recovery_tube_interaction_selected_root_count": 2,
            "recovery_tube_interaction_selected_minimum_root_mass": 0.8,
            "recovery_tube_interaction_selected_maximum_response_burden": 0.2,
        }],
    }]
    row = policy_diagnostic_scenario_rows(rollouts)[0]
    assert row["no_conventional_step_rate"] == 1.0
    assert row["interaction_aware_reachable_response_envelope_step_rate"] == 1.0
    assert row["recovery_tube_interaction_attempt_step_rate"] == 1.0
    assert row["recovery_tube_interaction_selection_rate_on_certified_steps"] == 1.0
    assert row["mean_recovery_tube_interaction_selected_minimum_root_mass_on_interaction_steps"] == 0.8
    assert row["mean_recovery_tube_interaction_environment_compatibility_checks_on_attempts"] == 7.0
    assert row["mean_recovery_tube_interaction_environment_compatibility_rejects_on_attempts"] == 1.0
