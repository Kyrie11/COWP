from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cowp.geometry.boxes import obb_distance, obb_overlap
from cowp.geometry.rss_like import same_heading_gap_violation
from cowp.geometry.ttc import dangerous_ttc


@dataclass
class UnsafeResult:
    unsafe: bool
    collision: bool
    near_miss: bool
    dangerous_ttc: bool
    rss_violation: bool
    offroad_severe: bool
    event_mask: np.ndarray
    min_distance: float


def trajectory_collision(traj_a: np.ndarray, traj_b: np.ndarray, inflation: float = 0.1) -> tuple[bool, np.ndarray]:
    t = min(len(traj_a), len(traj_b))
    mask = np.zeros(t, dtype=bool)
    for k in range(t):
        mask[k] = obb_overlap(traj_a[k], traj_b[k], inflation)
    return bool(np.any(mask)), mask


def trajectory_near_miss(traj_a: np.ndarray, traj_b: np.ndarray, threshold: float = 1.0) -> tuple[bool, np.ndarray, float]:
    t = min(len(traj_a), len(traj_b))
    mask = np.zeros(t, dtype=bool)
    dmin = float("inf")
    for k in range(t):
        d = obb_distance(traj_a[k], traj_b[k])
        dmin = min(dmin, d)
        mask[k] = d < threshold
    return bool(np.any(mask)), mask, float(dmin)


def offroad_severe_from_lane_distance(traj: np.ndarray, lane_dist: np.ndarray | None, threshold: float) -> tuple[bool, np.ndarray]:
    if lane_dist is None:
        return False, np.zeros(len(traj), dtype=bool)
    mask = np.asarray(lane_dist[: len(traj)]) > threshold
    return bool(np.any(mask)), mask


def unsafe_between(ego_traj: np.ndarray, agent_traj: np.ndarray, cfg: dict, agent_type: int = 1, agent_lane_dist: np.ndarray | None = None) -> UnsafeResult:
    unsafe_cfg = cfg.get("unsafe", cfg)
    inflation = float(unsafe_cfg.get("collision_inflation_m", 0.1))
    collision, col_mask = trajectory_collision(ego_traj, agent_traj, inflation)
    near_thresh = float(unsafe_cfg.get("near_miss_distance_vehicle_m", 1.0 if agent_type == 1 else 1.5))
    near, near_mask, dmin = trajectory_near_miss(ego_traj, agent_traj, near_thresh)
    ttc_min = float(unsafe_cfg.get("ttc_min_vehicle_s", 1.5 if agent_type == 1 else 2.0))
    dist_gate = float(unsafe_cfg.get("ttc_distance_gate_vehicle_m", 15.0 if agent_type == 1 else 20.0))
    dang, ttc_mask = dangerous_ttc(ego_traj, agent_traj, ttc_min, dist_gate)
    rss, rss_mask, _ = same_heading_gap_violation(
        rear_traj=agent_traj,
        front_traj=ego_traj,
        rho=float(unsafe_cfg.get("rss_reaction_time_s", 0.5)),
        a_max_accel=float(unsafe_cfg.get("rss_a_max_accel", 2.0)),
        b_min=float(unsafe_cfg.get("rss_b_min_comfort", 3.0)),
        b_max=float(unsafe_cfg.get("rss_b_max_front", 6.0)),
        min_gap=float(unsafe_cfg.get("rss_min_gap_m", 2.0)),
    )
    offroad, off_mask = offroad_severe_from_lane_distance(agent_traj, agent_lane_dist, float(unsafe_cfg.get("severe_offroad_distance_m", 4.0)))
    t = min(len(ego_traj), len(agent_traj))
    event_mask = np.zeros(t, dtype=bool)
    for m in (col_mask, near_mask, ttc_mask, rss_mask, off_mask):
        event_mask[: min(t, len(m))] |= m[: min(t, len(m))]
    unsafe = collision or near or dang or rss or offroad
    return UnsafeResult(unsafe, collision, near, dang, rss, offroad, event_mask, dmin)


def conventional_candidate_safe(ego_traj: np.ndarray, other_trajs: list[np.ndarray], cfg: dict) -> bool:
    for tr in other_trajs:
        result = unsafe_between(ego_traj, tr, cfg)
        if result.collision or result.offroad_severe:
            return False
    if not np.all(np.isfinite(ego_traj)):
        return False
    speed = np.linalg.norm(ego_traj[:, 3:5], axis=-1)
    acc = np.diff(speed, prepend=speed[0]) / max(float(cfg.get("time", {}).get("dt", 0.1)), 1e-3)
    if np.nanmax(np.abs(acc)) > float(cfg.get("candidate", {}).get("max_decel_mps2", 6.0)) + 1.0:
        return False
    return True
