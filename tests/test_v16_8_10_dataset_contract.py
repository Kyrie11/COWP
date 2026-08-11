from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from cowp.core.constants import PriorityRelation
from cowp.geometry.collision import unsafe_between
from cowp.label.burden import compute_burden
from cowp.models.set_transport_head import SetTransportCertificateHead


def _straight(x0: float, y0: float, speed: float, T: int = 20) -> np.ndarray:
    out = np.zeros((T, 7), dtype=np.float32)
    t = np.arange(1, T + 1, dtype=np.float32) * 0.1
    out[:, 0] = x0 + speed * t
    out[:, 1] = y0
    out[:, 2] = 0.0
    out[:, 3] = speed
    out[:, 5] = 4.8
    out[:, 6] = 1.9
    return out


def test_risk_known_zero_fast_path_is_exact_after_safe_predicate():
    cfg = {
        "time": {"dt": 0.1},
        "unsafe": {
            "collision_inflation_m": 0.1,
            "near_miss_distance_vehicle_m": 1.0,
            "ttc_min_vehicle_s": 1.5,
            "ttc_distance_gate_vehicle_m": 15.0,
            "rss_reaction_time_s": 0.5,
            "rss_a_max_accel": 2.0,
            "rss_b_min_comfort": 3.0,
            "rss_b_max_front": 6.0,
            "rss_min_gap_m": 2.0,
        },
        "burden": {"weights": {"acc": .25, "jerk": .10, "prog": .20, "risk": .20, "option": .15, "norm": .10}},
        "priority": {},
    }
    ego = _straight(0.0, 0.0, 5.0)
    agent = _straight(0.0, 30.0, 5.0)
    assert not unsafe_between(ego, agent, cfg, agent_type=1).unsafe
    b0, c0 = compute_burden(agent, ego, cfg, 1, natural_ref=agent, rho=PriorityRelation.UNKNOWN)
    b1, c1 = compute_burden(agent, ego, cfg, 1, natural_ref=agent, rho=PriorityRelation.UNKNOWN, risk_known_zero=True)
    np.testing.assert_allclose(c0, c1, atol=0.0, rtol=0.0)
    assert b0 == b1
    assert c1[3] == 0.0


def test_fixed_cardinality_response_gate_ignores_degenerate_valid_logits():
    head = SetTransportCertificateHead(d_model=8, hidden=8, geometry_steps=4, response_topk=2, use_response_valid_gate=False)
    assert head.use_response_valid_gate is False
    # The key invariant is configuration-level: mainline does not learn/gate on occupancy.
    cfg = yaml.safe_load(Path("configs/model_cowp_v16_8.yaml").read_text())
    train = yaml.safe_load(Path("configs/train_cowp_v16_8.yaml").read_text())
    label = yaml.safe_load(Path("configs/label_cowp_v16_8.yaml").read_text())
    assert cfg["model"]["use_response_valid_gate"] is False
    assert float(train["loss_weights"]["response_valid_bce"]) == 0.0
    assert label["critical"]["vehicle_only_main"] is True
    assert label["engineering"]["risk_known_zero_fastpath"] is True


def test_burden_uses_type_aware_ttc_and_configured_rss_contract():
    # Regression guard: burden risk must use the same type-aware/configured
    # TTC/RSS contract as unsafe_between(), otherwise the exact fast path could
    # silently diverge after a future threshold change.
    cfg = {
        "time": {"dt": 0.1},
        "unsafe": {
            "collision_inflation_m": 0.1,
            "near_miss_distance_vehicle_m": 1.0,
            "near_miss_distance_vru_m": 1.5,
            "ttc_min_vehicle_s": 0.5,
            "ttc_distance_gate_vehicle_m": 5.0,
            "ttc_min_vru_s": 3.0,
            "ttc_distance_gate_vru_m": 30.0,
            "rss_reaction_time_s": 0.8,
            "rss_a_max_accel": 1.5,
            "rss_b_min_comfort": 2.5,
            "rss_b_max_front": 5.5,
            "rss_min_gap_m": 3.0,
        },
        "burden": {"weights": {"acc": .25, "jerk": .10, "prog": .20, "risk": .20, "option": .15, "norm": .10}},
        "priority": {},
    }
    # A far safe VRU pair must remain bit-identical with the fast path under the
    # non-default contract.
    ego = _straight(0.0, 0.0, 5.0)
    vru = _straight(0.0, 40.0, 2.0)
    assert not unsafe_between(ego, vru, cfg, agent_type=2).unsafe
    b0, c0 = compute_burden(vru, ego, cfg, 2, natural_ref=vru, rho=PriorityRelation.UNKNOWN)
    b1, c1 = compute_burden(vru, ego, cfg, 2, natural_ref=vru, rho=PriorityRelation.UNKNOWN, risk_known_zero=True)
    np.testing.assert_array_equal(c0, c1)
    assert b0 == b1


def test_rss_is_laterally_gated_but_not_suppressed_by_distance_broadphase():
    cfg = {
        "time": {"dt": 0.1},
        "unsafe": {
            "collision_inflation_m": 0.1,
            "near_miss_distance_vehicle_m": 1.0,
            "ttc_min_vehicle_s": 1.5,
            "ttc_distance_gate_vehicle_m": 15.0,
            "rss_reaction_time_s": 0.5,
            "rss_a_max_accel": 2.0,
            "rss_b_min_comfort": 3.0,
            "rss_b_max_front": 6.0,
            "rss_min_gap_m": 2.0,
            "rss_heading_tolerance_deg": 35.0,
            "rss_lateral_margin_m": 0.75,
        },
    }
    # Parallel adjacent-lane vehicles are not a longitudinal RSS pair.
    front_adj = _straight(8.0, 0.0, 10.0, T=80)
    rear_adj = _straight(0.0, 3.5, 10.0, T=80)
    adj = unsafe_between(front_adj, rear_adj, cfg, agent_type=1)
    assert not adj.rss_violation

    # A same-lane high-speed pair can violate RSS beyond the 15 m TTC gate;
    # center-distance broadphase must not suppress the cheap RSS predicate.
    front = _straight(25.0, 0.0, 20.0, T=80)
    rear = _straight(0.0, 0.0, 20.0, T=80)
    same_lane = unsafe_between(front, rear, cfg, agent_type=1)
    assert same_lane.rss_violation
    assert same_lane.unsafe


def test_observational_bank_spends_budget_on_identity_first():
    from cowp.label.natural_alternatives import _ordered_observational_specs

    specs = _ordered_observational_specs({
        "obs_speed_scale": [0.85, 0.95, 1.0, 1.05, 1.15],
        "obs_time_shift_s": [-0.5, 0.0, 0.5],
        "obs_lateral_offset_m": [-0.3, 0.0, 0.3],
    })
    assert specs[0] == (1.0, 0.0, 0.0)
    assert (1.0, 0.0, 0.0) in specs[:8]
    # Regression against v16.8.9 nested-loop truncation: the first eight must
    # not all come from one 0.85 speed-scale slice.
    assert len({round(s[0], 6) for s in specs[:8]}) > 1


def test_route_geometry_neutral_proxy_follows_curved_logged_path():
    from cowp.label.natural_alternatives import _route_geometry_timed_trajectory

    H = 40
    dt = 0.1
    current = np.zeros(11, dtype=np.float32)
    current[:2] = [10.0, 0.0]
    current[3:6] = [0.0, 5.0, 5.0]
    current[6] = np.pi / 2.0
    current[7:9] = [4.8, 1.9]
    current[10] = 1.0
    theta = np.linspace(0.05, 1.4, H, dtype=np.float32)
    logged = np.zeros((H, 7), dtype=np.float32)
    logged[:, 0] = 10.0 * np.cos(theta)
    logged[:, 1] = 10.0 * np.sin(theta)
    logged[:, 2] = theta + np.pi / 2.0
    logged[:, 3] = -5.0 * np.sin(theta)
    logged[:, 4] = 5.0 * np.cos(theta)
    logged[:, 5:7] = [4.8, 1.9]

    tr = _route_geometry_timed_trajectory(logged, current, H, dt, accel=0.0)
    assert tr.shape == (H, 7)
    assert np.all(np.isfinite(tr))
    # A straight fallback from the current state would stay near x=10.  The
    # retimed route proxy must actually turn along the curved supervision path.
    assert float(np.min(tr[:, 0])) < 9.0
    assert float(np.max(tr[:, 1])) > 5.0


def test_womd_future_is_not_used_by_model_history_adapter():
    """Future ground truth may exist in train/val cache for labels/replay, but not encoder input."""
    import torch
    from cowp.data.womd_features import build_agent_history_from_womd

    B, N, Tp = 1, 2, 10
    base = {}
    for prefix, steps in (("past", Tp), ("current", 1)):
        for name in ("x", "y", "z", "length", "width", "height", "bbox_yaw", "velocity_x", "velocity_y"):
            shape = (B, N, steps) if prefix == "past" else (B, N)
            base[f"state/{prefix}/{name}"] = torch.zeros(shape)
        shape = (B, N, steps) if prefix == "past" else (B, N)
        base[f"state/{prefix}/valid"] = torch.ones(shape)
    base["state/is_sdc"] = torch.tensor([[1.0, 0.0]])

    a = dict(base)
    b = dict(base)
    a["state/future/x"] = torch.zeros(B, N, 80)
    b["state/future/x"] = torch.full((B, N, 80), 1.0e6)
    ha, ma = build_agent_history_from_womd(a, max_agents=N)
    hb, mb = build_agent_history_from_womd(b, max_agents=N)
    assert torch.equal(ha, hb)
    assert torch.equal(ma, mb)
