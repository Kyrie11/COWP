from __future__ import annotations

import numpy as np


def pairwise_ttc(pos_a: np.ndarray, vel_a: np.ndarray, pos_b: np.ndarray, vel_b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pa = np.asarray(pos_a, dtype=np.float32)
    va = np.asarray(vel_a, dtype=np.float32)
    pb = np.asarray(pos_b, dtype=np.float32)
    vb = np.asarray(vel_b, dtype=np.float32)
    rel_pos = pb - pa
    rel_vel = vb - va
    dist = np.linalg.norm(rel_pos, axis=-1)
    closing_speed = -np.sum(rel_pos * rel_vel, axis=-1) / np.maximum(dist, eps)
    ttc = dist / np.maximum(closing_speed, eps)
    return np.where(closing_speed > eps, ttc, np.inf)


def dangerous_ttc(traj_a: np.ndarray, traj_b: np.ndarray, ttc_min: float, distance_gate: float) -> tuple[bool, np.ndarray]:
    t = min(len(traj_a), len(traj_b))
    if t == 0:
        return False, np.zeros(0, dtype=bool)
    pos_a = traj_a[:t, :2]
    pos_b = traj_b[:t, :2]
    vel_a = traj_a[:t, 3:5]
    vel_b = traj_b[:t, 3:5]
    ttc = pairwise_ttc(pos_a, vel_a, pos_b, vel_b)
    dist = np.linalg.norm(pos_b - pos_a, axis=-1)
    mask = (ttc < float(ttc_min)) & (dist < float(distance_gate))
    return bool(np.any(mask)), mask
