from __future__ import annotations

import numpy as np
import pytest


def _state() -> np.ndarray:
    s = np.zeros((2, 11), dtype=np.float32)
    # ego
    s[0, 0:2] = [0.0, 0.0]
    s[0, 5] = 10.0
    s[0, 6] = 0.0
    s[0, 7:10] = [4.8, 1.9, 1.6]
    s[0, 10] = 1.0
    # stationary lead actor
    s[1, 0:2] = [12.0, 0.0]
    s[1, 5] = 0.0
    s[1, 6] = 0.0
    s[1, 7:10] = [4.8, 1.9, 1.6]
    s[1, 10] = 1.0
    return s


def _cfg(H: int = 8) -> dict:
    return {
        "time": {"dt": 0.1, "future_steps": H},
        "planning": {
            "online_collision_check_horizon_steps": H,
            "online_collision_check_stride": 1,
            "online_collision_agent_radius_m": 60.0,
            "online_collision_max_agents": 24,
            "online_require_cv_for_priority_agents": True,
            "online_logged_collision_buffer_m": 0.10,
            "online_priority_cv_collision_buffer_m": 0.35,
        },
    }


def test_collision_context_fast_path_preserves_full_horizon_boolean_and_exposes_prefix():
    from cowp.waymax_eval.policy_wrapper import (
        _collision_audit_against_context,
        _collision_free_against_constant_velocity,
        _prepare_collision_check_context,
    )

    H = 8
    state = _state()
    cfg = _cfg(H)
    traj = np.zeros((H, 7), dtype=np.float32)
    # Approach the stationary actor: early samples are clear, later samples violate.
    traj[:, 0] = np.linspace(1.0, 9.0, H, dtype=np.float32)
    traj[:, 2] = 0.0

    direct = _collision_free_against_constant_velocity(traj, state, 0, cfg)
    ctx = _prepare_collision_check_context(state, 0, cfg, horizon_steps=H)
    cached = _collision_free_against_constant_velocity(traj, state, 0, cfg, prepared_context=ctx)
    audit = _collision_audit_against_context(traj, ctx)

    assert direct is cached is False
    assert 0 < int(audit["safe_prefix_steps"]) < H
    assert audit["violation_source"] in {"base_cv", "priority_cv_buffer"}


def test_collision_context_is_candidate_invariant_and_reusable(monkeypatch):
    from cowp.waymax_eval import policy_wrapper

    state = _state()
    cfg = _cfg(8)
    calls = {"n": 0}
    original = policy_wrapper._agent_future_xy

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(policy_wrapper, "_agent_future_xy", counted)
    ctx = policy_wrapper._prepare_collision_check_context(state, 0, cfg, horizon_steps=8)
    after_prepare = calls["n"]
    assert after_prepare == 1
    for shift in (0.0, 1.0, 2.0, 3.0):
        traj = np.zeros((8, 7), dtype=np.float32)
        traj[:, 0] = np.linspace(shift, shift + 4.0, 8, dtype=np.float32)
        policy_wrapper._collision_audit_against_context(traj, ctx)
    assert calls["n"] == after_prepare


def test_recursive_viability_mask_is_lexicographic_not_weighted():
    torch = pytest.importorskip("torch")
    from cowp.waymax_eval.policy_wrapper import _recursive_viability_recovery_mask

    valid = torch.tensor([True, True, True, True])
    road = torch.tensor([True, True, False, True])
    prefix = torch.tensor([2.0, 6.0, 8.0, 6.0])
    mask = _recursive_viability_recovery_mask(valid, road, prefix)
    assert mask.tolist() == [False, True, False, True]

    # If the roadgraph surrogate rejects every valid option, recovery does not
    # invent road safety; it falls back to maximal collision-survival prefix.
    mask2 = _recursive_viability_recovery_mask(valid, torch.zeros_like(road), prefix)
    assert mask2.tolist() == [False, False, True, False]


def test_recursive_viability_method_keeps_primary_priority_gate():
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    from cowp.waymax_eval.rollout import _method_gate_defaults

    assert _canonical_online_method("cowp_recursive_viability", "hard") == ("cowp_recursive_viability", "priority")
    assert _method_gate_defaults("cowp_recursive_viability", "hard") == ("cowp_recursive_viability", "priority")


def _v16_8_28_collision_reference(traj, agent_state, sdc_index, cfg, other_future_trajs=None):
    """Literal-equivalent reference for the v16.8.28 causal collision boolean."""
    from cowp.waymax_eval.policy_wrapper import _agent_future_xy

    pcfg = cfg.get("planning", {})
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    H_full = int(len(traj))
    H = min(H_full, int(pcfg.get("online_collision_check_horizon_steps", H_full)))
    stride = max(1, int(pcfg.get("online_collision_check_stride", 2)))
    idx = np.arange(0, H, stride, dtype=np.int64)
    if idx.size == 0:
        idx = np.asarray([0], dtype=np.int64)
    traj_xy = np.asarray(traj[idx, :2], dtype=np.float32)
    if not np.isfinite(traj_xy).all():
        return False
    ego = agent_state[sdc_index]
    ego_radius = max(float(ego[7]), float(ego[8]), 4.0) * 0.5
    valid = agent_state[:, 10] > 0.5
    ego_xy = ego[:2].astype(np.float32)
    ego_yaw = float(ego[6])
    ego_dir = np.asarray([np.cos(ego_yaw), np.sin(ego_yaw)], dtype=np.float32)
    ego_lat = np.asarray([-ego_dir[1], ego_dir[0]], dtype=np.float32)
    ego_speed = max(float(ego[5]), float(np.linalg.norm(ego[3:5])))
    logged_buffer = float(pcfg.get("online_logged_collision_buffer_m", 0.10))
    cv_buffer = float(pcfg.get("online_priority_cv_collision_buffer_m", 0.35))
    require_cv = bool(pcfg.get("online_require_cv_for_priority_agents", True))
    max_dist = float(pcfg.get("online_collision_agent_radius_m", 60.0))
    max_agents = int(pcfg.get("online_collision_max_agents", 24))

    ranked = []
    for j in range(agent_state.shape[0]):
        if j == sdc_index or not valid[j]:
            continue
        rel = agent_state[j, :2].astype(np.float32) - ego_xy
        dist = float(np.linalg.norm(rel))
        if dist > max_dist:
            continue
        longitudinal = float(np.dot(rel, ego_dir))
        lateral = abs(float(np.dot(rel, ego_lat)))
        rel_speed = float(max(0.0, ego_speed - np.dot(agent_state[j, 3:5], ego_dir)))
        ttc = longitudinal / max(rel_speed, 1e-3) if longitudinal > 0.0 and rel_speed > 0.25 else 99.0
        priority_like = (-8.0 <= longitudinal <= 55.0 and lateral <= 7.5) or (
            ttc <= float(pcfg.get("online_priority_ttc_s", 5.0))
        )
        rank = (0 if priority_like else 1, dist)
        ranked.append((j, float(rank[0]) * 1000.0 + rank[1], priority_like))
    ranked.sort(key=lambda x: x[1])
    if max_agents > 0:
        ranked = ranked[:max_agents]

    for j, _key, priority_like in ranked:
        other_radius = max(float(agent_state[j, 7]), float(agent_state[j, 8]), 4.0) * 0.5
        radius = ego_radius + other_radius + 0.5
        logged, cv = _agent_future_xy(agent_state, j, H_full, dt, other_future_trajs)
        logged_xy = logged[idx]
        cv_xy = cv[idx]
        if not np.isfinite(logged_xy).all():
            logged_xy = cv_xy
        if not np.isfinite(logged_xy).all():
            continue
        min_logged = float(np.min(np.linalg.norm(traj_xy - logged_xy, axis=-1)))
        if min_logged < radius + logged_buffer:
            return False
        if require_cv and priority_like and np.isfinite(cv_xy).all():
            min_cv = float(np.min(np.linalg.norm(traj_xy - cv_xy, axis=-1)))
            if min_cv < radius + cv_buffer:
                return False
    return True


def test_cached_collision_boolean_matches_v16_8_28_reference_randomized():
    from cowp.waymax_eval.policy_wrapper import (
        _collision_free_against_constant_velocity,
        _prepare_collision_check_context,
    )

    rng = np.random.default_rng(16829)
    cfg = _cfg(80)
    cfg["planning"]["online_collision_check_stride"] = 2
    for _ in range(64):
        n = 1 + int(rng.integers(1, 24))
        state = np.zeros((n, 11), dtype=np.float32)
        state[:, :2] = rng.uniform(-65.0, 65.0, size=(n, 2))
        state[:, 3:5] = rng.uniform(-12.0, 12.0, size=(n, 2))
        state[:, 5] = np.linalg.norm(state[:, 3:5], axis=1)
        state[:, 6] = rng.uniform(-np.pi, np.pi, size=n)
        state[:, 7] = rng.uniform(3.8, 5.5, size=n)
        state[:, 8] = rng.uniform(1.6, 2.4, size=n)
        state[:, 9] = 1.6
        state[:, 10] = rng.random(n) > 0.08
        state[0, 10] = 1.0
        # Keep the SDC around the origin so the 60 m pruning is exercised.
        state[0, :2] = rng.uniform(-2.0, 2.0, size=2)
        state[0, 3:5] = rng.uniform(-8.0, 8.0, size=2)
        state[0, 5] = np.linalg.norm(state[0, 3:5])

        H = 80
        traj = np.zeros((H, 7), dtype=np.float32)
        p0 = state[0, :2].copy()
        v = rng.uniform(-1.0, 15.0, size=2).astype(np.float32)
        t = np.arange(1, H + 1, dtype=np.float32)[:, None] * 0.1
        traj[:, :2] = p0[None, :] + v[None, :] * t
        traj[:, 2] = np.arctan2(v[1], v[0]) if np.linalg.norm(v) > 1e-4 else state[0, 6]

        ref = _v16_8_28_collision_reference(traj, state, 0, cfg)
        ctx = _prepare_collision_check_context(state, 0, cfg, horizon_steps=H)
        got = _collision_free_against_constant_velocity(traj, state, 0, cfg, prepared_context=ctx)
        assert got == ref
