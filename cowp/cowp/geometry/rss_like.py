from __future__ import annotations

import numpy as np


def longitudinal_safety_distance(v_rear: float, v_front: float, rho: float = 0.5, a_max_accel: float = 2.0, b_min: float = 3.0, b_max: float = 6.0, min_gap: float = 2.0) -> float:
    d_safe = v_rear * rho + 0.5 * a_max_accel * rho**2
    d_safe += ((v_rear + rho * a_max_accel) ** 2) / max(2 * b_min, 1e-6)
    d_safe -= (v_front**2) / max(2 * b_max, 1e-6)
    return float(max(d_safe, min_gap))


def same_heading_gap_violation(rear_traj: np.ndarray, front_traj: np.ndarray, heading_tolerance: float = np.deg2rad(35), rho: float = 0.5, a_max_accel: float = 2.0, b_min: float = 3.0, b_max: float = 6.0, min_gap: float = 2.0) -> tuple[bool, np.ndarray, np.ndarray]:
    t = min(len(rear_traj), len(front_traj))
    violations = np.zeros(t, dtype=bool)
    d_safe_arr = np.zeros(t, dtype=np.float32)
    if t == 0:
        return False, violations, d_safe_arr
    for k in range(t):
        rear = rear_traj[k]
        front = front_traj[k]
        dh = abs(((front[2] - rear[2] + np.pi) % (2 * np.pi)) - np.pi)
        if dh > heading_tolerance:
            continue
        direction = np.array([np.cos(rear[2]), np.sin(rear[2])], dtype=np.float32)
        rel = front[:2] - rear[:2]
        longitudinal_gap = float(rel @ direction - 0.5 * rear[5] - 0.5 * front[5])
        if longitudinal_gap < -1.0:
            continue
        v_rear = float(np.linalg.norm(rear[3:5]))
        v_front = float(np.linalg.norm(front[3:5]))
        d_safe = longitudinal_safety_distance(v_rear, v_front, rho, a_max_accel, b_min, b_max, min_gap)
        d_safe_arr[k] = d_safe
        violations[k] = longitudinal_gap < d_safe
    return bool(np.any(violations)), violations, d_safe_arr
