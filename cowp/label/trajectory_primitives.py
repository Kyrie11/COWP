from __future__ import annotations

import numpy as np


def repair_planar_kinematics(traj: np.ndarray, current: np.ndarray | None = None, dt: float = 0.1, *, max_yaw_rate_rad_s: float | None = 0.8) -> np.ndarray:
    """Make [x,y,yaw,vx,vy,length,width] internally consistent.

    Lateral-offset primitives can otherwise contain sideways position changes while
    keeping the original heading/velocity.  Waymax's kinematic feasibility metric
    correctly flags those rollouts.  This repair keeps the planned positions but
    derives yaw and velocity from finite differences, with an optional yaw-rate
    limit for smoother closed-loop actions.
    """
    out = np.asarray(traj, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] < 5:
        return out
    dt = max(float(dt), 1e-3)
    if current is not None and len(current) >= 2:
        prev_xy = np.asarray(current[:2], dtype=np.float32)
        prev_yaw = float(current[6] if len(current) >= 7 else out[0, 2])
    else:
        prev_xy = out[0, :2].copy()
        prev_yaw = float(out[0, 2])
    last_yaw = prev_yaw
    for k in range(out.shape[0]):
        step = out[k, :2] - prev_xy
        vx, vy = step / dt
        speed = float(np.linalg.norm([vx, vy]))
        if speed > 1e-3:
            raw_yaw = float(np.arctan2(vy, vx))
            if max_yaw_rate_rad_s is not None:
                max_delta = abs(float(max_yaw_rate_rad_s)) * dt
                delta = (raw_yaw - last_yaw + np.pi) % (2.0 * np.pi) - np.pi
                raw_yaw = last_yaw + float(np.clip(delta, -max_delta, max_delta))
            out[k, 2] = (raw_yaw + np.pi) % (2.0 * np.pi) - np.pi
            out[k, 3] = vx
            out[k, 4] = vy
            last_yaw = float(out[k, 2])
        else:
            out[k, 2] = last_yaw
            out[k, 3:5] = 0.0
        prev_xy = out[k, :2].copy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def constant_accel_trajectory(current: np.ndarray, horizon: int, dt: float, accel: float = 0.0, lateral_offset: float = 0.0, speed_offset: float = 0.0, start_delay_s: float = 0.0, duration_s: float | None = None) -> np.ndarray:
    x, y = float(current[0]), float(current[1])
    heading = float(current[6] if len(current) >= 7 else current[2])
    speed = float(current[5] if len(current) >= 6 else np.linalg.norm(current[3:5])) + float(speed_offset)
    speed = max(speed, 0.0)
    length = float(current[7] if len(current) >= 8 and current[7] > 0 else 4.8)
    width = float(current[8] if len(current) >= 9 and current[8] > 0 else 1.9)
    out = np.zeros((horizon, 7), dtype=np.float32)
    direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
    lateral = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
    delay_steps = int(round(max(start_delay_s, 0.0) / dt))
    duration_steps = horizon if duration_s is None else max(1, int(round(duration_s / dt)))
    cur_pos = np.array([x, y], dtype=np.float32)
    cur_speed = speed
    for k in range(horizon):
        active = k >= delay_steps and k < delay_steps + duration_steps
        a = float(accel) if active else 0.0
        cur_speed = max(0.0, cur_speed + a * dt)
        cur_pos = cur_pos + direction * (cur_speed * dt)
        frac = min(1.0, (k + 1) / max(horizon, 1))
        smooth = 10 * frac**3 - 15 * frac**4 + 6 * frac**5
        pos = cur_pos + lateral * (lateral_offset * smooth)
        out[k] = [pos[0], pos[1], heading, direction[0] * cur_speed, direction[1] * cur_speed, length, width]
    return repair_planar_kinematics(out, current, dt)


def resample_logged(logged: np.ndarray, horizon: int, time_shift_steps: int = 0, speed_scale: float = 1.0, lateral_offset: float = 0.0) -> np.ndarray:
    logged = np.asarray(logged, dtype=np.float32)
    if len(logged) == 0:
        raise ValueError("logged trajectory is empty")
    idx = np.arange(horizon) + int(time_shift_steps)
    idx = np.clip(idx, 0, len(logged) - 1)
    out = logged[idx].copy()
    out[:, 3:5] *= float(speed_scale)
    if speed_scale != 1.0:
        pos = [out[0, :2].copy()]
        for k in range(1, horizon):
            pos.append(pos[-1] + out[k - 1, 3:5] * 0.1)
        out[:, :2] = np.asarray(pos, dtype=np.float32)
    if abs(lateral_offset) > 1e-6:
        lateral = np.stack([-np.sin(out[:, 2]), np.cos(out[:, 2])], axis=-1)
        out[:, :2] += lateral * float(lateral_offset)
    return out


def smooth_stop_trajectory(current: np.ndarray, horizon: int, dt: float, decel: float = -2.0, stop_after_m: float | None = None, creep_speed: float = 0.0) -> np.ndarray:
    x, y = float(current[0]), float(current[1])
    heading = float(current[6] if len(current) >= 7 else current[2])
    speed = float(current[5] if len(current) >= 6 else np.linalg.norm(current[3:5]))
    length = float(current[7] if len(current) >= 8 and current[7] > 0 else 4.8)
    width = float(current[8] if len(current) >= 9 and current[8] > 0 else 1.9)
    direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
    out = np.zeros((horizon, 7), dtype=np.float32)
    pos = np.array([x, y], dtype=np.float32)
    travelled = 0.0
    for k in range(horizon):
        if stop_after_m is not None and travelled >= stop_after_m:
            speed = min(speed, creep_speed)
        else:
            speed = max(creep_speed if stop_after_m is not None and travelled >= stop_after_m else 0.0, speed + decel * dt)
        step = speed * dt
        travelled += step
        pos = pos + direction * step
        out[k] = [pos[0], pos[1], heading, direction[0] * speed, direction[1] * speed, length, width]
    return out
