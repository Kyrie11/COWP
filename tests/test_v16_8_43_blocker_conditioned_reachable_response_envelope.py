from __future__ import annotations

import inspect

import numpy as np

from cowp.models.cowp_model import COWPModel
import cowp.waymax_eval.policy_wrapper as pw
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


def _support() -> dict[int, dict]:
    return {
        1: {
            "ready": True,
            "object_type": 1,
            "retained_mass": 0.8,
            "roots": [{"weight": 0.8, "mode_indices": (0,), "profiles": [_profile(10.0)]}],
        }
    }


def _constructor_args() -> tuple:
    return (
        _state(3), 0,
        np.zeros((1, 4, 7), dtype=np.float32),
        np.asarray([True]), np.asarray([True]), np.asarray([0]),
        np.asarray([0.0]), np.asarray([0.0]),
        np.zeros((1, 5), dtype=np.float32), np.asarray([0.0]),
        {}, {}, 0.0,
    )


def _constructor_kwargs() -> dict:
    return {
        "base_candidate_index": 0,
        "critical_track_index": np.asarray([1], dtype=np.int64),
        "critical_valid": np.asarray([True]),
        "natural_trajectories": np.zeros((1, 2, 4, 7), dtype=np.float32),
        "natural_logits": np.zeros((1, 2), dtype=np.float32),
        "blocker_query_track_index": np.asarray([2], dtype=np.int64),
        "blocker_query_trajectories": np.zeros((1, 2, 4, 7), dtype=np.float32),
        "blocker_query_logits": np.zeros((1, 2), dtype=np.float32),
        "object_types": np.asarray([1, 1, 1], dtype=np.int32),
    }


def test_v43_method_is_priority_hard_path() -> None:
    method, gate = pw._canonical_online_method(
        "cowp_blocker_conditioned_interaction_aware_reachable_response_envelope", "hard"
    )
    assert method == "cowp_blocker_conditioned_interaction_aware_reachable_response_envelope"
    assert gate == "priority"


def test_v43_exactly_nests_v42_when_v42_selects(monkeypatch) -> None:
    sentinel = {"parent_index": 7, "target": np.ones(5, dtype=np.float32)}
    calls: list[int] = []

    def fake_v42(*args, **kwargs):
        calls.append(int(np.asarray(kwargs["critical_track_index"]).size))
        return sentinel, {"selected": True, "selected_certificate_kind": "interaction_aware_reachable_response_envelope"}

    monkeypatch.setattr(pw, "_construct_interaction_aware_reachable_response_envelope_np", fake_v42)
    selected, detail = pw._construct_blocker_conditioned_interaction_aware_reachable_response_envelope_np(
        *_constructor_args(), **_constructor_kwargs()
    )
    assert selected is sentinel
    assert calls == [1]
    assert detail["nested_v42_selected"] is True
    assert detail["blocker_conditioned_query_attempted"] is False


def test_v43_late_bound_query_only_expands_support_after_v42_empty(monkeypatch) -> None:
    sentinel = {"parent_index": 2, "target": np.ones(5, dtype=np.float32)}
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_v42(*args, **kwargs):
        idx = np.asarray(kwargs["critical_track_index"], dtype=np.int64).copy()
        valid = np.asarray(kwargs["critical_valid"], dtype=bool).copy()
        calls.append((idx, valid))
        if len(calls) == 1:
            return None, {
                "interaction_support_agents_ready": 1,
                "interaction_hypotheses_evaluated": 8,
                "interaction_unsupported_blocker_rejects": 6,
                "interaction_root_unrecoverable_rejects": 2,
                "interaction_failure_reason": "no_interaction_certified_action",
            }
        return sentinel, {
            "interaction_support_agents_ready": 2,
            "interaction_hypotheses_evaluated": 8,
            "interaction_unsupported_blocker_rejects": 0,
            "interaction_root_unrecoverable_rejects": 3,
            "interaction_environment_compatibility_cache_hits": 11,
            "interaction_joint_compatibility_cache_hits": 4,
            "interaction_successor_context_cache_hits": 5,
            "selected": True,
        }

    monkeypatch.setattr(pw, "_construct_interaction_aware_reachable_response_envelope_np", fake_v42)
    selected, detail = pw._construct_blocker_conditioned_interaction_aware_reachable_response_envelope_np(
        *_constructor_args(), **_constructor_kwargs()
    )
    assert selected is sentinel
    assert [x[0].tolist() for x in calls] == [[1], [1, 2]]
    assert calls[1][1].tolist() == [True, True]
    assert detail["nested_v42_selected"] is False
    assert detail["blocker_conditioned_query_attempted"] is True
    assert detail["blocker_conditioned_query_selected"] is True
    assert detail["blocker_conditioned_query_ready_agent_count"] == 1
    assert detail["blocker_conditioned_query_environment_cache_hits"] == 11


def test_v43_compatibility_cache_is_semantics_preserving(monkeypatch) -> None:
    current_context = {"blockers": [1], "agents": [{"index": 1}, {"index": 2}]}
    shifted_context = {"blockers": [1], "agents": [{"index": 1}, {"index": 2}]}
    monkeypatch.setattr(
        pw, "_collision_blocking_agent_indices_against_context",
        lambda trajectory, context: (list(context["blockers"]), {"blocker_count": len(context["blockers"])}),
    )
    monkeypatch.setattr(
        pw, "_physical_recovery_tube_certificate_np",
        lambda *args, **kwargs: (True, {"collision_min_margin_m": 1.0}),
    )
    monkeypatch.setattr(pw, "unsafe_between_bool", lambda *args, **kwargs: False)
    cache = {"environment": {}, "joint": {}}
    args = (
        _state(3), _state(3), 0,
        _traj(100.0), np.ones(4, dtype=bool),
        _traj(101.0), np.ones(4, dtype=bool),
        {}, {}, current_context, shifted_context, _support(), np.asarray([1, 1, 1], dtype=np.int32),
    )
    ok1, detail1 = pw._interaction_aware_recovery_certificate_np(*args, compatibility_cache=cache)
    ok2, detail2 = pw._interaction_aware_recovery_certificate_np(*args, compatibility_cache=cache)
    assert ok1 is True and ok2 is True
    assert detail1["failure_reason"] == detail2["failure_reason"] == "none"
    assert detail1["interaction_environment_compatibility_checks"] == detail2["interaction_environment_compatibility_checks"]
    assert detail2["interaction_environment_compatibility_cache_hits"] > detail1["interaction_environment_compatibility_cache_hits"]


def test_v43_late_bound_root_decode_uses_root_scene_latent_and_no_logged_future() -> None:
    forward_source = inspect.getsource(COWPModel.forward)
    decode_source = inspect.getsource(pw.COWPWaymaxPolicy._decode_blocker_conditioned_natural_queries_np)
    assert 'out["natural_scene_z_agent"] = enc_scene["z_agent"]' in forward_source
    assert 'pred.get("natural_scene_z_agent")' in decode_source
    assert "log_trajectory" not in decode_source
    assert "ground_truth" not in decode_source.lower()
    assert 'pred["response"]' not in decode_source


def test_v43_policy_query_scope_filters_to_model_visible_valid_agents() -> None:
    source = inspect.getsource(pw.COWPWaymaxPolicy.__call__)
    assert 'model_agent_count = int(np.asarray(batch_np["state/history"]).shape[1])' in source
    assert 'model_agent_valid = np.asarray' in source
    assert 'int(a.get("index", -1)) < model_agent_count' in source


def test_v43_diagnostics_aggregate_new_reason_and_cache_fields() -> None:
    rollouts = [{
        "scenario_id": "scene",
        "steps": 1,
        "standard_metrics": {"CollisionRate": 0.0, "EP": 1.0},
        "policy_diagnostics": [{
            "fallback_used": True,
            "fallback_reason": "no_conventional_use_blocker_conditioned_interaction_aware_reachable_response_envelope",
            "valid_candidates": 3,
            "conventional_candidates": 0,
            "roadgraph_safe_candidates": 2,
            "collision_safe_candidates": 0,
            "recovery_tube_probe_used": True,
            "recovery_tube_selected": True,
            "recovery_tube_action_changed": True,
            "recovery_tube_blocker_conditioned_query_attempted": True,
            "recovery_tube_blocker_conditioned_query_selected": True,
            "recovery_tube_blocker_conditioned_query_agent_count": 5,
            "recovery_tube_blocker_conditioned_query_ready_agent_count": 4,
            "recovery_tube_blocker_conditioned_query_hypotheses_evaluated": 10,
            "recovery_tube_blocker_conditioned_query_unsupported_blocker_rejects": 2,
            "recovery_tube_blocker_conditioned_query_root_unrecoverable_rejects": 3,
            "recovery_tube_blocker_conditioned_query_environment_cache_hits": 20,
            "recovery_tube_blocker_conditioned_query_joint_cache_hits": 4,
            "recovery_tube_blocker_conditioned_query_successor_context_cache_hits": 7,
        }],
    }]
    row = policy_diagnostic_scenario_rows(rollouts)[0]
    assert row["no_conventional_step_rate"] == 1.0
    assert row["blocker_conditioned_interaction_aware_reachable_response_envelope_step_rate"] == 1.0
    assert row["recovery_tube_blocker_query_attempt_step_rate"] == 1.0
    assert row["recovery_tube_blocker_query_selection_rate_on_certified_steps"] == 1.0
    assert row["mean_recovery_tube_blocker_query_agents_on_attempts"] == 5.0
    assert row["mean_recovery_tube_blocker_query_environment_cache_hits_on_attempts"] == 20.0


def test_v43_v42_outer_constructor_initializes_cache_diagnostics_before_aggregation(monkeypatch) -> None:
    """Regression for profile8 KeyError in the real V42 outer constructor path.

    The V43 cache counters are diagnostic-only.  The outer V42 constructor must
    initialize them before aggregating the first per-hypothesis certificate.
    """
    monkeypatch.setattr(
        pw, "_construct_conflict_window_control_reachable_tube_np",
        lambda *args, **kwargs: (None, {}),
    )
    monkeypatch.setattr(
        pw, "_prepare_interaction_response_support_np",
        lambda *args, **kwargs: ({1: {"ready": True}}, {"critical_slots": 1, "agents_ready": 1, "retained_roots": 1, "eligible_profiles": 1}),
    )
    monkeypatch.setattr(pw, "_semantic_action_class_representatives_np", lambda *args, **kwargs: [0])
    monkeypatch.setattr(pw, "_prepare_collision_check_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        pw, "_project_candidate_bank_through_controller_np",
        lambda state, traj, cfg, previous_accel, longitudinal_envelope_schedule=None: (
            np.asarray(traj, dtype=np.float32).copy(),
            np.ones(np.asarray(traj).shape[:2], dtype=bool),
            np.zeros(np.asarray(traj).shape[:2], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        pw, "_collision_violation_window_against_context",
        lambda *args, **kwargs: {"first_violation_step": 0, "last_violation_step": 0},
    )
    monkeypatch.setattr(
        pw, "_conflict_window_envelope_schedule_family",
        lambda H, first, last: [
            {
                "policy_id": 0,
                "policy_name": "NOMINAL",
                "release_edge": -1,
                "nonnominal_edges": 0,
                "schedule": np.zeros(H, dtype=np.int8),
            },
            {
                "policy_id": 2,
                "policy_name": "LOWER_TO_FIRST_CONFLICT",
                "release_edge": 0,
                "nonnominal_edges": 1,
                "schedule": np.zeros(H, dtype=np.int8),
            },
        ],
    )
    monkeypatch.setattr(pw, "_counterfactual_successor_agent_state", lambda state, *args, **kwargs: np.asarray(state).copy())
    monkeypatch.setattr(pw, "_shift_append_terminal_reference_np", lambda tr, dt: np.asarray(tr).copy())
    monkeypatch.setattr(pw, "_shift_longitudinal_envelope_schedule_np", lambda schedule, policy_id: np.asarray(schedule).copy())
    monkeypatch.setattr(
        pw, "_interaction_aware_recovery_certificate_np",
        lambda *args, **kwargs: (False, {
            "failure_reason": "unsupported_collision_blocker",
            "interaction_environment_compatibility_checks": 3,
            "interaction_environment_compatibility_rejects": 1,
            "interaction_joint_compatibility_checks": 2,
            "interaction_joint_compatibility_rejects": 0,
            "interaction_joint_assignment_backtracks": 0,
            "interaction_environment_compatibility_cache_hits": 2,
            "interaction_joint_compatibility_cache_hits": 1,
        }),
    )

    state = _state(2)
    traj = np.zeros((1, 4, 7), dtype=np.float32)
    # Make the emitted first target differ from base_target so the hypothesis is evaluated.
    traj[0, :, 0] = np.arange(4, dtype=np.float32) + 1.0
    selected, detail = pw._construct_interaction_aware_reachable_response_envelope_np(
        state, 0, traj,
        np.asarray([True]), np.asarray([True]), np.asarray([0]),
        np.asarray([0.0]), np.asarray([1.0]),
        np.zeros((1, 5), dtype=np.float32), np.asarray([0.0]),
        {}, {}, 0.0,
        base_candidate_index=0,
        critical_track_index=np.asarray([1], dtype=np.int64),
        critical_valid=np.asarray([True]),
        natural_trajectories=np.zeros((1, 2, 4, 7), dtype=np.float32),
        natural_logits=np.zeros((1, 2), dtype=np.float32),
        object_types=np.asarray([1, 1], dtype=np.int32),
    )
    assert selected is None
    assert detail["interaction_environment_compatibility_cache_hits"] == 4
    assert detail["interaction_joint_compatibility_cache_hits"] == 2
    assert detail["interaction_successor_context_cache_hits"] == 1
