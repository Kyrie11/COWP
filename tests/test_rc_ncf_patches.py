from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cowp.models.coordinate import ego_centric_inputs
from importlib import import_module

merge_payloads = import_module("cowp.scripts.17_merge_waymax_shards").merge_payloads
from cowp.waymax_eval.policy_wrapper import (
    _consistent_one_step_target,
    _quintic_frenet_trajectory,
    build_online_batch,
)


def test_ego_centric_inputs_are_rigid_transform_invariant():
    hist = torch.zeros(1, 3, 2, 11)
    hist[0, 0, :, 0:2] = torch.tensor([[10.0, 5.0], [11.0, 5.0]])
    hist[0, 0, :, 6] = 0.2
    hist[0, 0, :, 7:9] = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    hist[0, 1, :, 0:2] = torch.tensor([[14.0, 7.0], [15.0, 7.0]])
    hist[0, 1, :, 6] = 0.4
    cand = torch.zeros(1, 2, 3, 7)
    cand[0, :, :, 0:2] = torch.tensor([[[12.0, 5.0], [13.0, 5.0], [14.0, 5.0]], [[11.0, 6.0], [12.0, 7.0], [13.0, 8.0]]])
    cand[..., 2] = 0.2
    conf = torch.zeros(1, 2, 8)
    conf[0, :, 1:3] = torch.tensor([[13.0, 5.0], [15.0, 7.0]])
    idx = torch.tensor([0])
    h0, c0, f0 = ego_centric_inputs(hist, cand, conf, idx)

    angle = 0.7
    ca, sa = np.cos(angle), np.sin(angle)
    R = torch.tensor([[ca, -sa], [sa, ca]], dtype=torch.float32)
    trans = torch.tensor([101.0, -37.0])
    hist2, cand2, conf2 = hist.clone(), cand.clone(), conf.clone()
    hist2[..., 0:2] = hist[..., 0:2] @ R.T + trans
    hist2[..., 7:9] = hist[..., 7:9] @ R.T
    hist2[..., 6] += angle
    cand2[..., 0:2] = cand[..., 0:2] @ R.T + trans
    cand2[..., 3:5] = cand[..., 3:5] @ R.T
    cand2[..., 2] += angle
    conf2[..., 1:3] = conf[..., 1:3] @ R.T + trans
    h1, c1, f1 = ego_centric_inputs(hist2, cand2, conf2, idx)
    assert torch.allclose(h0[..., 0:2], h1[..., 0:2], atol=2e-5)
    assert torch.allclose(h0[..., 6:9], h1[..., 6:9], atol=2e-5)
    assert torch.allclose(c0[..., 0:5], c1[..., 0:5], atol=2e-5)
    assert torch.allclose(f0[..., 1:3], f1[..., 1:3], atol=2e-5)


def test_consistent_one_step_target_limits_jerk_yaw_and_state_mismatch():
    cfg = {
        "time": {"dt": 0.1},
        "candidate": {"max_accel_mps2": 4.0, "max_decel_mps2": 6.0, "max_jerk_mps3": 6.0, "max_yaw_rate_rad_s": 1.0},
        "waymax": {"max_delta_yaw_rad": 0.12},
    }
    current = np.zeros(11, dtype=np.float32)
    current[3] = current[5] = 10.0
    current[6] = 0.0
    desired = np.asarray([4.0, 4.0, 1.5, 40.0, 40.0, 4.8, 1.9], dtype=np.float32)
    target, accel = _consistent_one_step_target(current, desired, cfg, previous_longitudinal_accel=0.0)
    assert abs(accel) <= 0.6 + 1e-6  # jerk * dt
    assert abs(target[2]) <= 0.1 + 1e-6
    displacement_velocity = 0.5 * (np.asarray([10.0, 0.0]) + target[3:5]) * 0.1
    assert np.allclose(target[:2], displacement_velocity, atol=1e-5)


def test_quintic_lane_change_has_tangent_consistent_velocity():
    current = np.zeros(11, dtype=np.float32)
    current[3] = current[5] = 8.0
    current[6] = 0.0
    tr = _quintic_frenet_trajectory(current, 50, 0.1, accel=0.0, lateral_offset=3.5, start_delay_s=0.5, lane_change_duration_s=4.0)
    heading = np.arctan2(tr[:, 4], tr[:, 3])
    assert np.max(np.abs(np.arctan2(np.sin(heading - tr[:, 2]), np.cos(heading - tr[:, 2])))) < 1e-5
    assert np.isfinite(tr).all()


def test_online_batch_exposes_sdc_and_caps_online_context():
    state = np.zeros((10, 11), dtype=np.float32)
    state[:, 7:10] = [4.8, 1.9, 1.6]
    state[:, 10] = 1.0
    state[3, 0:6] = [0, 0, 0, 8, 0, 8]
    for i in range(10):
        if i != 3:
            state[i, 0:6] = [8 + 3 * i, (-1) ** i * 2, 0, 0, 0, 0]
    road = {
        "xy": np.stack([np.linspace(-10, 80, 100), np.zeros(100)], axis=-1).astype(np.float32),
        "heading": np.zeros(100, dtype=np.float32),
        "valid": np.ones(100, dtype=bool),
        "types": np.ones(100, dtype=np.int32),
    }
    cfg = {
        "limits": {"max_candidates": 16, "max_critical_agents": 6, "max_natural_alternatives": 4, "max_safe_responses": 4, "max_conflict_regions": 64, "max_agents": 16},
        "time": {"future_steps": 20, "dt": 0.1},
        "candidate": {"max_jerk_mps3": 20.0},
        "planning": {"max_online_critical_agents": 4, "max_online_pair_conflict_tokens": 8, "max_online_map_tokens": 4},
    }
    batch = build_online_batch(state, 3, cfg, roadgraph=road)
    assert batch["state/is_sdc"].shape == (1, 16)
    assert bool(batch["state/is_sdc"][0, 3])
    assert int(batch["cowp/critical/valid"].sum()) <= 4
    assert int(batch["map/conflict_region_valid"].sum()) <= 12


def test_merge_waymax_shards_preserves_per_episode_metrics(tmp_path: Path):
    payloads = []
    for shard, collision in enumerate([0.0, 1.0]):
        p = tmp_path / f"s{shard}.json"
        payload = {
            "mode": "waymax", "method": "cowp", "num_rollouts": 1, "steps": [80],
            "standard_metrics": [{"CollisionRate": collision, "EP": 0.5 + 0.1 * shard}],
            "policy_diagnostic_summary": {"ClosedLoopPolicySteps": 80, "ClosedLoopFallbackStepRate": 0.1 * shard},
            "closed_loop_cowp_metric_summary": {"EpisodesWithDiagnostics": 1, "FallbackEpisodeRate": float(shard)},
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
        payloads.append(p)
    merged = merge_payloads(payloads)
    assert merged["num_rollouts"] == 2
    assert len(merged["standard_metrics"]) == 2
    assert np.isclose(merged["standard_metric_summary"]["CollisionRate"], 0.5)
    assert np.isclose(merged["policy_diagnostic_summary"]["ClosedLoopFallbackStepRate"], 0.05)
