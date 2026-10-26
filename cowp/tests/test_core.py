import math

import numpy as np

from ocrap.lcv import finite_sample_upper_quantile, weighted_lcvar
from ocrap.ocmero import oc_mero
from ocrap.selector import constrained_lcb_select, crisp_select
from ocrap.observation import compatibility_labels
from ocrap.schema import Observation


def test_weighted_lcvar_lower_tail_boundary_mass():
    scores = np.array([-2.0, 0.0, 10.0])
    weights = np.array([0.1, 0.4, 0.5])
    # Lowest 20% mass = all of -2 with 0.1 mass plus half of 0 with 0.1 mass, divided by 0.2.
    assert math.isclose(weighted_lcvar(scores, weights, 0.2), -1.0, rel_tol=1e-6)


def test_ocmero_detects_oracle_artifact_under_shared_observation():
    M = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    p = np.array([0.5, 0.5])
    C = np.ones((2, 2), dtype=float)
    res = oc_mero(M, p, C, alpha=0.5, beta=0.5, option_valid=np.array([True, True]))
    assert res.r_orc > 0.9
    assert res.r_dep < -0.9
    assert res.gap > 1.9


def test_crisp_lexicographic_fallback():
    utility = np.array([10.0, 12.0, 0.0])
    r_dep = np.array([-1.0, -0.5, 3.0])
    hard = np.array([2.0, 1.0, 0.5])
    harm = np.array([1.0, 5.0, 3.0])
    feasible = np.array([True, True, True])
    sel = crisp_select(utility, r_dep, hard, harm, feasible, gamma_rec=5.0, gamma_H=0.0, gamma_D=0.0)
    assert sel.selected_index == 2
    assert sel.reason == "lexicographic_fallback"


def test_finite_sample_quantile_strict_inf_when_underpowered():
    scores = np.array([0.1, 0.2])
    gamma = finite_sample_upper_quantile(scores, delta=0.01, strict=True)
    assert math.isinf(gamma)


def test_observation_compatibility_identity():
    obs = Observation(
        ego_state=np.zeros(9, dtype=np.float32),
        boxes=np.zeros((0, 9), dtype=np.float32),
        box_valid=np.zeros((0,), dtype=bool),
        occ_mask=np.zeros((7, 4, 4), dtype=np.float32),
        contact_flag=False,
        stability_proxy=np.zeros(3, dtype=np.float32),
    )
    Y, C, D = compatibility_labels([obs, obs], {"epsilon_obs": 1.0})
    assert np.allclose(Y, 1.0)
    assert np.allclose(C, 1.0)
    assert np.allclose(D, 0.0)


def test_root_margin_aggregation_is_lower_tail_by_default():
    from ocrap.root_clustering import aggregate_root_margins

    M_future = np.array([[1.0, 1.0], [-3.0, 1.0]], dtype=np.float32)
    assignments = np.array([0, 0], dtype=np.int64)
    probs = np.array([0.5, 0.5], dtype=np.float32)
    M = aggregate_root_margins(M_future, assignments, probs, K=1, cfg={"intra_root_lcvar_alpha": 0.5})
    assert M.shape == (1, 2)
    assert M[0, 0] <= -2.99
    assert math.isclose(float(M[0, 1]), 1.0, rel_tol=1e-6)


def test_womd_parser_keeps_sdc_when_truncated():
    from types import SimpleNamespace
    from ocrap.womd import parse_scenario_proto

    def state(x):
        return SimpleNamespace(center_x=x, center_y=0.0, center_z=0.0, velocity_x=0.0, velocity_y=0.0, heading=0.0, length=4.0, width=2.0, height=1.5, valid=True)

    tracks = [SimpleNamespace(id=i, object_type=1, states=[state(float(i))]) for i in range(5)]
    scenario = SimpleNamespace(
        scenario_id="sdc_truncation",
        timestamps_seconds=[0.0],
        sdc_track_index=4,
        tracks=tracks,
        map_features=[],
        dynamic_map_states=[],
    )
    raw = parse_scenario_proto(scenario, max_agents=2)
    assert raw.sdc_track_index == 0
    assert raw.object_ids[0] == "4"
    assert raw.agent_states.shape[1] == 2


def test_papercheck_importable():
    from ocrap.papercheck import papercheck_dataset

    assert callable(papercheck_dataset)


def test_constrained_lcb_selector_prefers_nominal_with_slack():
    utility = np.array([1.0, 0.6])
    r_dep = np.array([0.04, 0.20])
    hard = np.array([0.0, 0.0])
    harm = np.array([0.0, 0.0])
    feasible = np.array([True, True])
    sel = constrained_lcb_select(utility, r_dep, hard, harm, feasible, gamma_rec=0.05, nominal_slack=0.02, pred_gap=np.array([0.0, 0.0]))
    assert sel.selected_index == 0
    assert sel.reason == "nominal_slack_lcb_admitted"


def test_constrained_lcb_selector_penalizes_oracle_deployable_gap():
    utility = np.array([0.9, 0.85])
    r_dep = np.array([0.2, 0.18])
    hard = np.array([0.0, 0.0])
    harm = np.array([0.0, 0.0])
    feasible = np.array([True, True])
    sel = constrained_lcb_select(utility, r_dep, hard, harm, feasible, gamma_rec=0.1, lcb_beta=1.0, pred_gap=np.array([0.2, 0.0]), intervention_penalty=0.0, deviation_penalty=0.0)
    assert sel.selected_index == 1


def test_constrained_lcb_fallback_is_recovery_guarded_not_utility_only():
    # No candidate reaches gamma_rec. Candidate 1 has much higher utility but
    # substantially worse deployable-recovery LCB and gap; fallback must keep
    # the recovery-consistent candidate instead of chasing utility.
    utility = np.array([1.0, 10.0, 0.8])
    r_dep = np.array([0.00, -1.00, -0.02])
    hard = np.array([0.0, 0.0, 0.0])
    harm = np.array([0.0, 0.0, 0.0])
    feasible = np.array([True, True, True])
    sel = constrained_lcb_select(
        utility, r_dep, hard, harm, feasible,
        gamma_rec=0.5,
        lcb_beta=0.5,
        pred_gap=np.array([0.1, 2.0, 0.1]),
        intervention_penalty=0.0,
        deviation_penalty=0.0,
        nominal_fallback_lcb_slack=0.01,
    )
    assert sel.selected_index == 0
    assert sel.reason == "nominal_recovery_guarded_fallback"


def test_constrained_lcb_fallback_uses_utility_only_inside_near_best_recovery_set():
    utility = np.array([1.0, 2.0, 10.0])
    r_dep = np.array([-0.01, 0.00, -0.5])
    hard = np.array([0.0, 0.0, 0.0])
    harm = np.array([0.0, 0.0, 0.0])
    feasible = np.array([True, True, True])
    sel = constrained_lcb_select(
        utility, r_dep, hard, harm, feasible,
        gamma_rec=0.5,
        lcb_beta=0.0,
        pred_gap=np.array([0.1, 0.1, 0.1]),
        intervention_penalty=0.0,
        deviation_penalty=0.0,
        nominal_fallback_lcb_slack=0.0,
        fallback_lcb_margin=0.05,
    )
    assert sel.selected_index == 1
    assert sel.reason == "recovery_guarded_fallback"
