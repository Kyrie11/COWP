from __future__ import annotations

import numpy as np


def longitudinal_safety_distance(v_rear: float, v_front: float, rho: float = 0.5, a_max_accel: float = 2.0, b_min: float = 3.0, b_max: float = 6.0, min_gap: float = 2.0) -> float:
    d_safe = v_rear * rho + 0.5 * a_max_accel * rho**2
    d_safe += ((v_rear + rho * a_max_accel) ** 2) / max(2 * b_min, 1e-6)
    d_safe -= (v_front**2) / max(2 * b_max, 1e-6)
    return float(max(d_safe, min_gap))


def same_heading_gap_violation(rear_traj: np.ndarray, front_traj: np.ndarray, heading_tolerance: float = np.deg2rad(35), rho: float = 0.5, a_max_accel: float = 2.0, b_min: float = 3.0, b_max: float = 6.0, min_gap: float = 2.0, lateral_margin: float = 0.75) -> tuple[bool, np.ndarray, np.ndarray]:
    t = min(len(rear_traj), len(front_traj))
    violations = np.zeros(t, dtype=bool)
    d_safe_arr = np.zeros(t, dtype=np.float32)
    if t == 0:
        return False, violations, d_safe_arr
    rear = np.asarray(rear_traj[:t], dtype=np.float32)
    front = np.asarray(front_traj[:t], dtype=np.float32)
    dh = np.abs((front[:, 2] - rear[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    valid_heading = dh <= heading_tolerance
    direction = np.stack([np.cos(rear[:, 2]), np.sin(rear[:, 2])], axis=-1).astype(np.float32)
    lateral_direction = np.stack([-direction[:, 1], direction[:, 0]], axis=-1).astype(np.float32)
    rel = front[:, :2] - rear[:, :2]
    longitudinal_gap = np.sum(rel * direction, axis=-1) - 0.5 * rear[:, 5] - 0.5 * front[:, 5]
    lateral_center_offset = np.abs(np.sum(rel * lateral_direction, axis=-1))
    lateral_half_extent = 0.5 * np.maximum(rear[:, 6], 0.1) + 0.5 * np.maximum(front[:, 6], 0.1)
    # Longitudinal RSS is meaningful only while the two oriented boxes occupy
    # the same/merging lateral corridor.  Without this gate, parallel vehicles
    # in adjacent lanes can be marked RSS-dangerous solely because their
    # longitudinal gap is small, creating false coercion labels.
    valid_lateral = lateral_center_offset <= lateral_half_extent + float(lateral_margin)
    valid_gap = longitudinal_gap >= -1.0
    v_rear = np.linalg.norm(rear[:, 3:5], axis=-1)
    v_front = np.linalg.norm(front[:, 3:5], axis=-1)
    d_safe = v_rear * rho + 0.5 * a_max_accel * rho**2
    d_safe += ((v_rear + rho * a_max_accel) ** 2) / max(2 * b_min, 1e-6)
    d_safe -= (v_front**2) / max(2 * b_max, 1e-6)
    d_safe = np.maximum(d_safe, min_gap).astype(np.float32)
    active = valid_heading & valid_gap & valid_lateral
    d_safe_arr[active] = d_safe[active]
    violations[:] = active & (longitudinal_gap < d_safe)
    return bool(np.any(violations)), violations, d_safe_arr
