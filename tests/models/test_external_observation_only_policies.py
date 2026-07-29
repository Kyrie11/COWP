from __future__ import annotations

import copy

import numpy as np

from ocrap.external_baselines.evaluate import _yaw_rate_violation_proxy
from ocrap.external_baselines.observed_risk import observed_risk_profile
from ocrap.external_baselines.policies import select_external_policy


def _sample(*, detour: bool, nominal: bool) -> dict:
    T = 11
    x = np.linspace(0.0, 10.0, T)
    y = np.linspace(0.0, 7.0 if detour else 0.0, T)
    states = np.zeros((T, 9), dtype=np.float32)
    states[:, 0] = x
    states[:, 1] = y
    states[:, 2] = np.gradient(x, 0.1)
    states[:, 3] = np.gradient(y, 0.1)
    states[:, 4] = np.arctan2(states[:, 3], np.maximum(states[:, 2], 1e-3))
    states[:, 5] = np.gradient(states[:, 4], 0.1)
    states[:, 6] = np.hypot(states[:, 2], states[:, 3])
    states[:, 7] = 4.8
    states[:, 8] = 2.0
    controls = np.zeros((T, 2), dtype=np.float32)

    hist = np.zeros((3, 2, 16), dtype=np.float32)
    valid = np.ones((3, 2), dtype=bool)
    hist[:, 0, 0] = 0.0
    hist[:, 0, 1] = 0.0
    hist[:, 0, 3] = 10.0
    hist[:, 0, 7] = 0.0
    hist[:, 0, 10] = 4.8
    hist[:, 0, 11] = 2.0
    hist[:, 1, 0] = 5.0
    hist[:, 1, 1] = 0.0
    hist[:, 1, 3] = 0.0
    hist[:, 1, 7] = 0.0
    hist[:, 1, 10] = 4.8
    hist[:, 1, 11] = 2.0

    return {
        "prefix_states": states,
        "prefix_controls": controls,
        "agent_history": hist,
        "agent_valid": valid,
        "ego_state": states[0],
        "utility": np.float32(0.0),
        "feasible": np.int32(1),
        "is_nominal": np.int32(nominal),
        "prefix_macro_name": np.asarray("keep" if nominal else "lane_shift"),
        "m_star": np.asarray([[2.0, -1.0], [2.0, -1.0]], dtype=np.float32),
        "root_probs": np.asarray([0.5, 0.5], dtype=np.float32),
        "root_valid": np.asarray([1, 1], dtype=bool),
        "option_valid": np.asarray([1, 1], dtype=bool),
        "r_orc_star": np.float32(2.0),
        "r_dep_star": np.float32(2.0),
        "hard_violation": np.float32(0.0),
        "harm_proxy": np.float32(0.0),
    }


def _cfg() -> dict:
    return {
        "external_baselines": {
            "policy": {
                "risk_dt": 0.1,
                "expected_risk_threshold": 2.0,
                "cvar_risk_threshold": 2.0,
                "dro_cvar_threshold": 2.0,
                "marc_risk_threshold": 2.0,
                "racp_risk_threshold": 2.0,
            }
        }
    }


def test_observed_risk_detects_conflicting_candidate() -> None:
    collision = observed_risk_profile(_sample(detour=False, nominal=True), _cfg())
    detour = observed_risk_profile(_sample(detour=True, nominal=False), _cfg())
    assert collision.expected_loss > detour.expected_loss
    assert collision.min_clearance < detour.min_clearance


def test_nonoracle_policies_are_invariant_to_teacher_label_mutation() -> None:
    samples = [_sample(detour=False, nominal=True), _sample(detour=True, nominal=False)]
    mutated = copy.deepcopy(samples)
    for i, d in enumerate(mutated):
        d["m_star"] = -np.asarray(d["m_star"]) * (10.0 + i)
        d["r_orc_star"] = np.float32(-100.0 + i)
        d["r_dep_star"] = np.float32(100.0 - i)
        d["hard_violation"] = np.float32(50.0 * i)
        d["harm_proxy"] = np.float32(100.0 * (1 - i))
    methods = [
        "marc_lite", "racp_lite", "expected_risk_filter", "cvar_risk_filter",
        "dro_cvar_filter", "predictive_safety_filter", "postimpact_mpc_lite",
        "post_crash_braking", "post_collision_restoration", "severity_minimization",
    ]
    for method in methods:
        a = select_external_policy(method, samples, _cfg())
        b = select_external_policy(method, mutated, _cfg())
        assert a.selected_index == b.selected_index, method
        np.testing.assert_allclose(a.score, b.score, err_msg=method)


def test_learned_policy_selection_uses_logits_only() -> None:
    samples = [_sample(detour=False, nominal=True), _sample(detour=True, nominal=False)]
    outputs = {
        "logits": np.asarray([0.0, 3.0]),
        "utility": np.asarray([1000.0, -1000.0]),
        "hard": np.asarray([0.0, 999.0]),
        "harm": np.asarray([0.0, 999.0]),
        "r_orc": np.asarray([999.0, -999.0]),
    }
    for method in ["gameformer_lite", "wayformer_bc", "betopnet_lite"]:
        sel = select_external_policy(method, samples, _cfg(), model_outputs=outputs)
        assert sel.selected_index == 1


def test_yaw_rate_proxy_uses_schema_channel_five() -> None:
    d = _sample(detour=False, nominal=True)
    d["prefix_states"][:, 2] = 100.0  # vx must not be interpreted as heading
    d["prefix_states"][:, 5] = 0.2
    assert _yaw_rate_violation_proxy(d, yaw_rate_max=0.6) == 0.0
    d["prefix_states"][3, 5] = 0.8
    assert _yaw_rate_violation_proxy(d, yaw_rate_max=0.6) == 1.0
