from __future__ import annotations

import numpy as np

import cowp.waymax_eval.policy_wrapper as pw


def _state(n: int = 8) -> np.ndarray:
    state = np.zeros((n, 11), dtype=np.float32)
    state[:, 7] = 4.5
    state[:, 8] = 2.0
    state[:, 10] = 1.0
    return state


def _constructor_args() -> tuple:
    return (
        _state(8), 0,
        np.zeros((1, 4, 7), dtype=np.float32),
        np.asarray([True]), np.asarray([True]), np.asarray([0]),
        np.asarray([0.0]), np.asarray([0.0]),
        np.zeros((1, 5), dtype=np.float32), np.asarray([0.0]),
        {}, {}, 0.0,
    )


def test_r3_support_detail_merge_is_additive_for_disjoint_agent_builds() -> None:
    base = {
        "minimum_raw_mode_probability": 0.03,
        "probability_floor": 0.02,
        "required_root_mass": 0.75,
        "minimum_root_count": 2,
        "critical_slots": 4,
        "agents_ready": 3,
        "agents_rejected_invalid_prediction": 0,
        "agents_rejected_root_count": 1,
        "agents_rejected_root_mass": 0,
        "agents_rejected_profile_feasibility": 0,
        "retained_roots": 7,
        "eligible_profiles": 19,
        "profile_candidates": 42,
        "canonical_support_mass_sum": 3.7,
        "raw_support_mass_sum": 3.5,
    }
    late = {
        "minimum_raw_mode_probability": 0.03,
        "probability_floor": 0.02,
        "required_root_mass": 0.75,
        "minimum_root_count": 2,
        "critical_slots": 2,
        "agents_ready": 2,
        "agents_rejected_invalid_prediction": 0,
        "agents_rejected_root_count": 0,
        "agents_rejected_root_mass": 0,
        "agents_rejected_profile_feasibility": 0,
        "retained_roots": 4,
        "eligible_profiles": 11,
        "profile_candidates": 24,
        "canonical_support_mass_sum": 1.9,
        "raw_support_mass_sum": 1.8,
    }
    out = pw._merge_interaction_response_support_details_np(base, late)
    assert out["critical_slots"] == 6
    assert out["agents_ready"] == 5
    assert out["retained_roots"] == 11
    assert out["eligible_profiles"] == 30
    assert out["profile_candidates"] == 66
    assert np.isclose(out["canonical_support_mass_sum"], 5.6)
    assert np.isclose(out["raw_support_mass_sum"], 5.3)
    for key in (
        "minimum_raw_mode_probability", "probability_floor",
        "required_root_mass", "minimum_root_count",
    ):
        assert out[key] == base[key]


def test_r3_bc_iare_reuses_exact_v42_base_support_and_prepares_only_late_blockers(monkeypatch) -> None:
    sentinel = {"parent_index": 4, "target": np.ones(5, dtype=np.float32)}
    calls: list[dict] = []
    prepared_calls: list[list[int]] = []

    base_support = {1: {"ready": True, "retained_mass": 0.8, "roots": []}}
    base_detail = {
        "minimum_raw_mode_probability": 0.03,
        "probability_floor": 0.02,
        "required_root_mass": 0.75,
        "minimum_root_count": 2,
        "critical_slots": 1,
        "agents_ready": 1,
        "agents_rejected_invalid_prediction": 0,
        "agents_rejected_root_count": 0,
        "agents_rejected_root_mass": 0,
        "agents_rejected_profile_feasibility": 0,
        "retained_roots": 1,
        "eligible_profiles": 2,
        "profile_candidates": 3,
        "canonical_support_mass_sum": 0.8,
        "raw_support_mass_sum": 0.8,
    }

    def fake_v42(*args, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            trace = kwargs["internal_trace"]
            trace["unsupported_hypothesis_indices"].extend([2, 5])
            trace["unsupported_blocker_union"].update({3, 6})
            trace["prepared_response_support"] = base_support
            trace["prepared_response_support_detail"] = base_detail
            return None, {
                "interaction_support_agents_ready": 1,
                "interaction_failure_reason": "no_interaction_certified_action",
            }
        assert kwargs["prepared_response_support"] is not None
        assert set(kwargs["prepared_response_support"]) == {1, 3, 6}
        merged = kwargs["prepared_response_support_detail"]
        assert merged["critical_slots"] == 3
        assert merged["agents_ready"] == 3
        return sentinel, {
            "interaction_support_agents_ready": 3,
            "interaction_hypotheses_evaluated": 2,
            "interaction_unsupported_blocker_rejects": 0,
            "interaction_root_unrecoverable_rejects": 0,
            "selected": True,
        }

    def fake_prepare(state, sdc_index, crit_idx, crit_valid, nat, logits, object_types, roadgraph, cfg):
        idx = np.asarray(crit_idx, dtype=np.int64).reshape(-1)
        prepared_calls.append(idx.tolist())
        support = {int(i): {"ready": True, "retained_mass": 0.8, "roots": []} for i in idx}
        detail = dict(base_detail)
        detail.update({
            "critical_slots": len(idx),
            "agents_ready": len(idx),
            "retained_roots": len(idx),
            "eligible_profiles": 2 * len(idx),
            "profile_candidates": 3 * len(idx),
            "canonical_support_mass_sum": 0.8 * len(idx),
            "raw_support_mass_sum": 0.8 * len(idx),
        })
        return support, detail

    monkeypatch.setattr(pw, "_construct_interaction_aware_reachable_response_envelope_np", fake_v42)
    monkeypatch.setattr(pw, "_prepare_interaction_response_support_np", fake_prepare)

    q = np.asarray([2, 3, 4, 5, 6, 7], dtype=np.int64)
    selected, detail = pw._construct_blocker_conditioned_interaction_aware_reachable_response_envelope_np(
        *_constructor_args(),
        base_candidate_index=0,
        critical_track_index=np.asarray([1], dtype=np.int64),
        critical_valid=np.asarray([True]),
        natural_trajectories=np.zeros((1, 2, 4, 7), dtype=np.float32),
        natural_logits=np.zeros((1, 2), dtype=np.float32),
        blocker_query_track_index=q,
        blocker_query_trajectories=np.zeros((len(q), 2, 4, 7), dtype=np.float32),
        blocker_query_logits=np.zeros((len(q), 2), dtype=np.float32),
        object_types=np.ones((8,), dtype=np.int32),
    )
    assert selected is sentinel
    assert prepared_calls == [[3, 6]]
    assert detail["blocker_conditioned_query_exact_blocker_agent_count"] == 2
    assert detail["blocker_conditioned_query_ready_agent_count"] == 2
