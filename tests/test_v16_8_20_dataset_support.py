from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from cowp.core.config import load_config
from cowp.core.constants import ProposalSource
from cowp.core.types import Lane, MapData
from cowp.data.cache_schema import COWP_SCHEMA
from cowp.geometry.lane_graph import build_conflict_regions
from cowp.label.critical_agents import select_critical_agents


def test_main_v16_8_20_contract_is_causal_and_dimension_aligned() -> None:
    label_cfg = load_config("configs/label_cowp_v16_8.yaml")
    model_cfg = load_config("configs/model_cowp_v16_8.yaml")
    assert label_cfg["critical"]["selection_reference_mode"] == "causal_anchor_v2"
    assert float(label_cfg["critical"].get("objects_of_interest_score_weight", 0.0)) == 0.0
    assert float(label_cfg["critical"].get("tracks_to_predict_score_weight", 0.0)) == 0.0
    assert int(label_cfg["conflict"]["candidate_pool_max_regions"]) >= 4096
    assert bool(label_cfg["candidate"]["route_joint_yield_enabled"])
    assert int(label_cfg["limits"]["max_critical_agents"]) == int(model_cfg["model"]["max_critical_agents"])
    assert int(ProposalSource.JOINT_ROUTE_NCF) == 14


def test_priority_candidate_labels_are_part_of_cache_contract() -> None:
    for key in (
        "cowp/candidates/priority_eligible",
        "cowp/candidates/priority_false_safe",
        "cowp/candidates/priority_noncoercive_feasible",
    ):
        assert key in COWP_SCHEMA


def test_critical_selection_does_not_change_when_logged_future_changes(toy_scene, cfg) -> None:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("critical", {})["selection_reference_mode"] = "causal_anchor_v2"
    cfg["critical"]["objects_of_interest_score_weight"] = 0.0
    cfg["critical"]["tracks_to_predict_score_weight"] = 0.0

    baseline = select_critical_agents(toy_scene, cfg)
    perturbed_scene = copy.deepcopy(toy_scene)
    cur = int(perturbed_scene.current_time_index)
    # Destroy both SDC and non-SDC *logged future* while keeping every
    # inference-visible history/current value unchanged.
    rng = np.random.default_rng(7)
    perturbed_scene.states[:, cur + 1 :, 0:2] += rng.normal(
        loc=0.0, scale=500.0, size=perturbed_scene.states[:, cur + 1 :, 0:2].shape
    ).astype(np.float32)
    perturbed = select_critical_agents(perturbed_scene, cfg)

    np.testing.assert_array_equal(baseline["track_index"], perturbed["track_index"])
    np.testing.assert_array_equal(baseline["valid"], perturbed["valid"])
    np.testing.assert_allclose(baseline["score"], perturbed["score"], rtol=0.0, atol=1.0e-6)
    np.testing.assert_array_equal(baseline["base_priority"], perturbed["base_priority"])
    assert baseline["_selection_diagnostics"]["logged_future_used_for_selection"] is False
    assert perturbed["_selection_diagnostics"]["logged_future_used_for_selection"] is False


def _crossing_map(order: list[int]) -> MapData:
    lanes = {
        1: Lane(1, np.asarray([[-40.0, 0.0, 0.0], [80.0, 0.0, 0.0]], dtype=np.float32)),
        2: Lane(2, np.asarray([[10.0, -50.0, 0.0], [10.0, 50.0, 0.0]], dtype=np.float32)),
        3: Lane(3, np.asarray([[20.0, -50.0, 0.0], [20.0, 50.0, 0.0]], dtype=np.float32)),
        4: Lane(4, np.asarray([[30.0, -50.0, 0.0], [30.0, 50.0, 0.0]], dtype=np.float32)),
        5: Lane(5, np.asarray([[40.0, -50.0, 0.0], [40.0, 50.0, 0.0]], dtype=np.float32)),
    }
    return MapData(lanes={k: lanes[k] for k in order})


def _region_signature(regions):
    return [
        (
            r.conflict_type,
            tuple(sorted(int(x) for x in r.involved_lane_ids)),
            tuple(np.round(np.asarray(r.center_xy, dtype=np.float32), 4).tolist()),
        )
        for r in regions
    ]


def test_conflict_top_c_is_invariant_to_map_feature_insertion_order() -> None:
    cfg = {
        "limits": {"max_conflict_regions": 2},
        "conflict": {
            "candidate_pool_max_regions": 128,
            "reference_min_lanes_considered": 2,
            "reference_max_lanes_considered": 16,
            "reference_lane_radius_m": 200.0,
            "max_intersections_per_lane_pair": 4,
        },
    }
    ref = np.asarray([0.0, 0.0], dtype=np.float32)
    a_diag, b_diag = {}, {}
    a = build_conflict_regions(
        _crossing_map([1, 2, 3, 4, 5]), cfg,
        reference_xy=ref, reference_heading=0.0, diagnostics=a_diag,
    )
    b = build_conflict_regions(
        _crossing_map([5, 4, 3, 2, 1]), cfg,
        reference_xy=ref, reference_heading=0.0, diagnostics=b_diag,
    )
    assert _region_signature(a) == _region_signature(b)
    assert a_diag["ego_reference_used"] is True
    assert b_diag["ego_reference_used"] is True
    assert a_diag["candidate_pool_saturated"] is False
    assert b_diag["candidate_pool_saturated"] is False
