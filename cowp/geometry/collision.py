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


def _center_distance_and_halfdiag(traj_a: np.ndarray, traj_b: np.ndarray, *, inflation: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    t = min(len(traj_a), len(traj_b))
    if t == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    a = traj_a[:t]
    b = traj_b[:t]
    dist = np.linalg.norm(a[:, :2] - b[:, :2], axis=-1)
    len_a = np.maximum(a[:, 5] + 2.0 * inflation, 0.1)
    wid_a = np.maximum(a[:, 6] + 2.0 * inflation, 0.1)
    len_b = np.maximum(b[:, 5] + 2.0 * inflation, 0.1)
    wid_b = np.maximum(b[:, 6] + 2.0 * inflation, 0.1)
    halfdiag_sum = 0.5 * np.sqrt(len_a**2 + wid_a**2) + 0.5 * np.sqrt(len_b**2 + wid_b**2)
    return dist.astype(np.float32, copy=False), halfdiag_sum.astype(np.float32, copy=False)


def trajectory_collision(traj_a: np.ndarray, traj_b: np.ndarray, inflation: float = 0.1) -> tuple[bool, np.ndarray]:
    t = min(len(traj_a), len(traj_b))
    mask = np.zeros(t, dtype=bool)
    if t == 0:
        return False, mask
    # A bounding-circle gate is exact as a negative test: if center distance is
    # larger than the sum of inflated half diagonals, OBBs cannot overlap.  This
    # avoids 80 SAT polygon checks for almost every non-contact pair.
    center_dist, halfdiag_sum = _center_distance_and_halfdiag(traj_a, traj_b, inflation=inflation)
    possible = np.where(center_dist <= halfdiag_sum + 1e-6)[0]
    for k in possible:
        mask[int(k)] = obb_overlap(traj_a[int(k)], traj_b[int(k)], inflation)
    return bool(np.any(mask)), mask


def trajectory_near_miss(traj_a: np.ndarray, traj_b: np.ndarray, threshold: float = 1.0) -> tuple[bool, np.ndarray, float]:
    t = min(len(traj_a), len(traj_b))
    mask = np.zeros(t, dtype=bool)
    if t == 0:
        return False, mask, float("inf")
    # Another exact negative gate: polygon distance is at least
    # center_distance - sum(half_diagonals).  Only the small candidate set can be
    # within the near-miss threshold, so only those timesteps need exact polygon
    # distances.
    center_dist, halfdiag_sum = _center_distance_and_halfdiag(traj_a, traj_b, inflation=0.0)
    lower_bound = np.maximum(0.0, center_dist - halfdiag_sum)
    possible = np.where(lower_bound < float(threshold))[0]
    dmin = float(np.min(lower_bound)) if len(lower_bound) else float("inf")
    for k in possible:
        d = obb_distance(traj_a[int(k)], traj_b[int(k)])
        dmin = min(dmin, d)
        mask[int(k)] = d < threshold
    return bool(np.any(mask)), mask, float(dmin)




def _center_broadphase_far(traj_a: np.ndarray, traj_b: np.ndarray, cfg: dict, agent_type: int) -> tuple[bool, float]:
    t = min(len(traj_a), len(traj_b))
    if t == 0:
        return True, float("inf")
    dist = np.linalg.norm(traj_a[:t, :2] - traj_b[:t, :2], axis=-1)
    dmin = float(np.min(dist)) if len(dist) else float("inf")
    unsafe_cfg = cfg.get("unsafe", cfg)
    near_thresh = float(unsafe_cfg.get("near_miss_distance_vehicle_m", 1.0 if agent_type == 1 else 1.5))
    dist_gate = float(unsafe_cfg.get("ttc_distance_gate_vehicle_m", 15.0 if agent_type == 1 else 20.0))
    # Conservative upper bound on vehicle half diagonals.  If the centers never
    # come within this gate, collision, near-miss, TTC and RSS tests cannot fire.
    half_a = 0.5 * np.sqrt(np.maximum(traj_a[:t, 5], 0.1) ** 2 + np.maximum(traj_a[:t, 6], 0.1) ** 2)
    half_b = 0.5 * np.sqrt(np.maximum(traj_b[:t, 5], 0.1) ** 2 + np.maximum(traj_b[:t, 6], 0.1) ** 2)
    max_half = float(np.nanmax(half_a + half_b)) if len(half_a) else 6.0
    gate = max(dist_gate, near_thresh + max_half + float(unsafe_cfg.get("collision_inflation_m", 0.1)) + 1.0)
    return dmin > gate, max(0.0, dmin - max_half)

def offroad_severe_from_lane_distance(traj: np.ndarray, lane_dist: np.ndarray | None, threshold: float) -> tuple[bool, np.ndarray]:
    if lane_dist is None:
        return False, np.zeros(len(traj), dtype=bool)
    mask = np.asarray(lane_dist[: len(traj)]) > threshold
    return bool(np.any(mask)), mask


def unsafe_between(ego_traj: np.ndarray, agent_traj: np.ndarray, cfg: dict, agent_type: int = 1, agent_lane_dist: np.ndarray | None = None) -> UnsafeResult:
    unsafe_cfg = cfg.get("unsafe", cfg)
    t = min(len(ego_traj), len(agent_traj))
    far, dmin_fast = _center_broadphase_far(ego_traj, agent_traj, cfg, agent_type)
    if far and agent_lane_dist is None:
        empty = np.zeros(t, dtype=bool)
        return UnsafeResult(False, False, False, False, False, False, empty, dmin_fast)
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
