from __future__ import annotations

from pathlib import Path
import importlib

import numpy as np
import torch
import yaml

from cowp.core.constants import NaturalSource, ObjectType, PriorityRelation
from cowp.core.types import Lane, MapData, ScenarioData
from cowp.label.critical_agents import select_critical_agents
from cowp.label.natural_alternatives import build_pair_specific_ego_neutrals, generate_natural_alternatives
from cowp.label.trajectory_primitives import constant_accel_trajectory
from cowp.models.losses import _candidate_certificate_mask, _critical_mechanism_mask
from cowp.data.dataset import mask_out_of_range_critical_agents


def _cfg() -> dict:
    cfg = yaml.safe_load(Path("configs/label_cowp_v16_8.yaml").read_text())
    cfg["limits"]["max_candidates"] = 2
    cfg["limits"]["max_critical_agents"] = 1
    cfg["limits"]["max_natural_alternatives"] = 12
    cfg["natural"]["min_natural_alternatives"] = 6
    cfg["natural"]["min_low_burden_alternatives"] = 2
    return cfg


def _lane_unresolved_scene(*, future_valid_steps: int = 80) -> ScenarioData:
    """Vehicle has a clear factual route but is far from every lane centreline."""
    H, cur, dt = 80, 10, 0.1
    T = cur + 1 + H
    states = np.zeros((2, T, 11), dtype=np.float32)

    def put(n: int, t: int, x: float, y: float, speed: float, valid: bool = True) -> None:
        states[n, t] = [x, y, 0.0, speed, 0.0, speed, 0.0, 4.8, 1.9, 1.6, float(valid)]

    for t in range(cur + 1):
        tau = (t - cur) * dt
        put(0, t, 8.0 * tau, 0.0, 8.0)
        put(1, t, -12.0 + 8.0 * tau, 15.0, 8.0)
    for j in range(H):
        tau = (j + 1) * dt
        put(0, cur + 1 + j, 8.0 * tau, 0.0, 8.0)
        valid = j < int(future_valid_steps)
        put(1, cur + 1 + j, -12.0 + 8.0 * tau, 15.0, 8.0, valid=valid)
        if not valid:
            states[1, cur + 1 + j, :10] = 0.0

    # Deliberately keep a valid HD-map lane far from the actor.  This reproduces
    # the v16.8.12 failure mode: map data exist, but centreline routing is not an
    # adequate geometry witness for this factual movement.
    far_lane = Lane(99, np.array([[-100.0, 100.0, 0.0], [200.0, 100.0, 0.0]], dtype=np.float32))
    return ScenarioData(
        scenario_id=f"v16813_lane_unresolved_{future_valid_steps}",
        timestamps=np.arange(T, dtype=np.float32) * dt,
        current_time_index=cur,
        states=states,
        object_type=np.array([ObjectType.VEHICLE, ObjectType.VEHICLE], dtype=np.int32),
        track_id=np.array([1, 2], dtype=np.int64),
        sdc_track_index=0,
        objects_of_interest=np.array([2], dtype=np.int64),
        tracks_to_predict=np.array([1], dtype=np.int32),
        map_data=MapData(lanes={99: far_lane}),
    )


def _protected_critical(*, mechanism_valid: bool = True) -> dict[str, np.ndarray]:
    return {
        "track_index": np.array([1], dtype=np.int32),
        "valid": np.array([True]),
        "mechanism_valid": np.array([mechanism_valid]),
        "base_priority": np.array([PriorityRelation.EQUAL_OR_NEGOTIATED], dtype=np.int32),
        "score": np.array([5.0], dtype=np.float32),
    }


def test_empirical_corridor_restores_typed_natural_support_without_claiming_hd_map_verification() -> None:
    cfg = _cfg()
    scene = _lane_unresolved_scene(future_valid_steps=80)
    H = int(cfg["time"]["future_steps"])
    neutral_fallback = constant_accel_trajectory(scene.ego_current, H, 0.1, accel=-2.0)
    neutral, _ = build_pair_specific_ego_neutrals(scene, _protected_critical(), neutral_fallback, cfg)
    natural = generate_natural_alternatives(scene, _protected_critical(), neutral, cfg)
    diag = natural["_diagnostics"][0]

    assert diag["reference_kind"] == "logged_geometry_neutral_timing"
    assert diag["empirical_corridor_eligible"] is True
    assert diag["root_count"] >= 6
    assert diag["low_burden_root_count"] >= 2
    assert diag["prio_root_count"] >= 1

    valid = natural["valid"][0]
    empirical = valid & (natural["map_evidence_mode"][0] == 2)
    prio = valid & (natural["source"][0] == int(NaturalSource.PRIO))
    assert int(empirical.sum()) >= 2
    assert bool(np.all(~natural["map_verified"][0][empirical]))
    assert bool(np.all(natural["priority_preserved"][0][prio]))
    # Source identity must remain non-degenerate: PRIO is no longer consumed by
    # OBS/NEU cross-source de-duplication.
    assert int(prio.sum()) >= 1
    assert int(np.sum(valid & (natural["source"][0] == int(NaturalSource.OBS)))) >= 1
    assert int(np.sum(valid & (natural["source"][0] == int(NaturalSource.NEU)))) >= 1


def test_short_future_lane_unresolved_actor_stays_critical_but_is_masked_from_mechanism_supervision() -> None:
    cfg = _cfg()
    scene = _lane_unresolved_scene(future_valid_steps=19)
    critical = select_critical_agents(scene, cfg)

    # Future availability must never change the inference-time critical universe.
    assert bool(critical["valid"][0])
    assert int(critical["track_index"][0]) == 1
    # But a 1.9 s factual path with no lane route is insufficient evidence for an
    # 8 s natural/counterfactual certificate label.
    assert not bool(critical["mechanism_valid"][0])
    assert int(critical["auditability_reason"][0]) == 3
    assert int(critical["audit_future_valid_steps"][0]) == 19

    H = int(cfg["time"]["future_steps"])
    neutral_fallback = constant_accel_trajectory(scene.ego_current, H, 0.1, accel=-2.0)
    neutral, _ = build_pair_specific_ego_neutrals(scene, critical, neutral_fallback, cfg)
    natural = generate_natural_alternatives(scene, critical, neutral, cfg)
    assert not bool(natural["valid"][0].any())
    assert natural["_diagnostics"][0]["mechanism_valid"] is False


def test_training_masks_keep_selected_critical_visible_but_exclude_unknown_mechanism_targets() -> None:
    batch = {
        "cowp/critical/valid": torch.tensor([[True, True, False]]),
        "cowp/critical/mechanism_valid": torch.tensor([[True, False, False]]),
        "cowp/candidates/valid": torch.tensor([[True, True, True]]),
        "cowp/candidates/certificate_valid": torch.tensor([[True, False, True]]),
    }
    np.testing.assert_array_equal(
        _critical_mechanism_mask(batch).cpu().numpy(),
        np.array([[True, False, False]]),
    )
    np.testing.assert_array_equal(
        _candidate_certificate_mask(batch).cpu().numpy(),
        np.array([[True, False, True]]),
    )


def test_tensor_input_invisibility_invalidates_candidate_certificate_targets() -> None:
    data = {
        "cowp/critical/track_index": np.array([0, 5], dtype=np.int32),
        "cowp/critical/input_index": np.array([0, 5], dtype=np.int64),
        "cowp/critical/valid": np.array([True, True]),
        "cowp/critical/mechanism_valid": np.array([True, True]),
        "cowp/candidates/valid": np.array([True, True, False]),
        "cowp/candidates/certificate_valid": np.array([True, True, False]),
        "dataset/mechanism_certificate_complete": np.asarray(True),
        "cowp/natural/valid": np.ones((2, 2), dtype=bool),
        "cowp/natural/traj": np.ones((2, 2, 3, 7), dtype=np.float32),
        "cowp/response/valid": np.ones((3, 2, 2), dtype=bool),
        "cowp/witness/exists": np.ones((3, 2), dtype=bool),
    }
    mask_out_of_range_critical_agents(data, num_agents=2)
    np.testing.assert_array_equal(data["cowp/critical/valid"], np.array([True, False]))
    np.testing.assert_array_equal(data["cowp/critical/mechanism_valid"], np.array([True, False]))
    np.testing.assert_array_equal(data["cowp/critical/selected_before_input_mask"], np.array([True, True]))
    np.testing.assert_array_equal(data["cowp/critical/mechanism_valid_before_input_mask"], np.array([True, True]))
    np.testing.assert_array_equal(data["cowp/critical/input_visible"], np.array([True, False]))
    # Candidate certificate targets are unknown once a selected certificate actor
    # is absent from the model input; they must not be trained as negatives.
    np.testing.assert_array_equal(data["cowp/candidates/certificate_valid"], np.array([False, False, False]))
    assert not bool(np.asarray(data["dataset/mechanism_certificate_complete"]).item())


def test_model_support_audit_loads_optional_tensor_visibility_context() -> None:
    mod = importlib.import_module("cowp.scripts.65_audit_model_support")
    assert "cowp/critical/track_index" in mod.VISIBILITY_CONTEXT
    assert "cowp/critical/track_id" in mod.VISIBILITY_CONTEXT
    assert "state/id" in mod.VISIBILITY_CONTEXT or "womd/state/id" in mod.VISIBILITY_CONTEXT
    assert "state/current/valid" in mod.VISIBILITY_CONTEXT or "womd/state/current/valid" in mod.VISIBILITY_CONTEXT


def test_waymax_label_class_selection_never_treats_unknown_certificate_as_ncf_or_false_safe() -> None:
    from cowp.waymax_eval.candidate_replay import select_candidate_indices

    arrays = {
        "cowp/candidates/valid": np.array([True, True, True]),
        "cowp/candidates/certificate_valid": np.array([False, True, True]),
        "cowp/candidates/noncoercive_feasible": np.array([True, True, False]),
        "cowp/candidates/false_safe": np.array([True, False, True]),
        "cowp/candidates/conventional_safe": np.array([True, True, True]),
        "cowp/candidates/ego_utility_prior": np.array([0.0, 1.0, 2.0], dtype=np.float32),
    }
    assert select_candidate_indices(arrays, {}, selection="noncoercive", max_candidates=8) == [1]
    assert select_candidate_indices(arrays, {}, selection="false_safe", max_candidates=8) == [2]


def test_late_auditability_finalizer_rejects_exhausted_lane_projection_without_long_future() -> None:
    cfg = _cfg()
    scene = _lane_unresolved_scene(future_valid_steps=19)
    # The actor is exactly at the endpoint of a syntactically valid lane.  A
    # cheap current-state projection therefore succeeds, but there is no
    # forward lane continuation from which an 8 s natural basis can be built.
    actor_cur = np.asarray(scene.states[1, scene.current_time_index, :2], dtype=np.float32)
    exhausted = Lane(
        77,
        np.array([[actor_cur[0] - 20.0, actor_cur[1], 0.0], [actor_cur[0], actor_cur[1], 0.0]], dtype=np.float32),
    )
    scene.map_data = MapData(lanes={77: exhausted})

    critical = select_critical_agents(scene, cfg)
    assert bool(critical["valid"][0])
    # Stage-1 auditability is deliberately cheap and can regard the endpoint as
    # lane-supported.  The natural builder must finalize against actual routable
    # lane geometry, rather than silently fabricating a straight continuation.
    assert bool(critical["mechanism_valid"][0])

    H = int(cfg["time"]["future_steps"])
    neutral_fallback = constant_accel_trajectory(scene.ego_current, H, 0.1, accel=-2.0)
    neutral, _ = build_pair_specific_ego_neutrals(scene, critical, neutral_fallback, cfg)
    natural = generate_natural_alternatives(scene, critical, neutral, cfg)

    assert bool(critical["valid"][0])
    assert not bool(critical["mechanism_valid"][0])
    assert int(critical["auditability_reason"][0]) == 3
    assert not bool(natural["valid"][0].any())
    assert natural["_diagnostics"][0]["auditability_finalizer"] == "no_routable_lane_or_substantial_factual_geometry"
