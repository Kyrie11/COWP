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


def smooth_arrival_trajectory(
    current: np.ndarray,
    horizon: int,
    dt: float,
    distance_m: float,
    target_time_s: float,
) -> np.ndarray | None:
    """Straight-path cubic timing primitive with zero acceleration at arrival.

    The old RMR-BCTE primitive solved a *constant* acceleration to the conflict
    region centre.  That construction can enter the actual conflict envelope
    seconds before its advertised pass-after time, and long delays become
    impossible once constant deceleration would stop the vehicle.

    We instead use

        s(t) = v0 t + c2 t^2 + c3 t^3,   0 <= t <= T

    with ``s(T)=distance_m`` and ``a(T)=0``.  The resulting acceleration changes
    linearly to zero, supports substantially later physically feasible arrivals,
    and has constant finite jerk.  After T, the primitive continues at the
    terminal speed.  Feasibility bounds are checked by the caller / common
    candidate validator.
    """
    cur = np.asarray(current, dtype=np.float32).reshape(-1)
    horizon = int(horizon)
    dt = max(float(dt), 1.0e-4)
    d = float(distance_m)
    T = float(target_time_s)
    if horizon <= 0 or cur.size < 7 or not np.isfinite(d) or not np.isfinite(T) or d <= 0.0 or T <= dt:
        return None

    x0, y0 = float(cur[0]), float(cur[1])
    heading = float(cur[6])
    v0 = float(max(cur[5] if cur.size > 5 else np.linalg.norm(cur[3:5]), 0.0))
    length = float(cur[7] if cur.size > 7 and cur[7] > 0 else 4.8)
    width = float(cur[8] if cur.size > 8 and cur[8] > 0 else 1.9)

    delta = v0 * T - d
    c3 = delta / max(2.0 * T**3, 1.0e-9)
    c2 = -3.0 * delta / max(2.0 * T**2, 1.0e-9)
    vT = v0 + 2.0 * c2 * T + 3.0 * c3 * T * T
    if not np.isfinite(vT) or vT < -1.0e-5:
        return None
    vT = max(float(vT), 0.0)

    direction = np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float32)
    out = np.zeros((horizon, 7), dtype=np.float32)
    for k in range(horizon):
        t = float((k + 1) * dt)
        if t <= T:
            s_t = v0 * t + c2 * t * t + c3 * t**3
            v_t = v0 + 2.0 * c2 * t + 3.0 * c3 * t * t
        else:
            s_t = d + vT * (t - T)
            v_t = vT
        if not np.isfinite(s_t) or not np.isfinite(v_t) or v_t < -1.0e-4:
            return None
        pos = np.asarray([x0, y0], dtype=np.float32) + direction * float(max(s_t, 0.0))
        out[k] = [pos[0], pos[1], heading, direction[0] * max(v_t, 0.0), direction[1] * max(v_t, 0.0), length, width]
    return repair_planar_kinematics(out, current=cur, dt=dt)


def resample_logged(
    logged: np.ndarray,
    horizon: int,
    time_shift_steps: int = 0,
    speed_scale: float = 1.0,
    lateral_offset: float = 0.0,
    *,
    current: np.ndarray | None = None,
    dt: float = 0.1,
) -> np.ndarray:
    """Create a transformed observational root without kinematic discontinuities.

    Earlier versions indexed a shifted future, changed velocity, and then moved
    positions laterally while retaining the original yaw/velocity.  That can
    introduce a first-step teleport and side-slip labels.  v15 reconstructs the
    path from the current state, applies lateral offset with a zero-at-origin
    smooth ramp, and finally repairs yaw/velocity from finite differences.
    """
    logged = np.asarray(logged, dtype=np.float32)
    if logged.ndim != 2 or logged.shape[0] == 0 or logged.shape[1] < 7:
        raise ValueError("logged trajectory must be non-empty [T,7]")
    horizon = int(horizon)
    dt = max(float(dt), 1e-3)
    idx = np.arange(horizon, dtype=np.int64) + int(time_shift_steps)
    idx = np.clip(idx, 0, len(logged) - 1)
    sampled = logged[idx].copy()

    # Determine the state immediately before the first generated point.
    if current is not None:
        current_arr = np.asarray(current, dtype=np.float32)
        prev_xy = current_arr[:2].copy()
        prev_yaw = float(current_arr[6] if current_arr.shape[0] >= 7 else sampled[0, 2])
    else:
        prev_xy = sampled[0, :2] - sampled[0, 3:5] * dt
        prev_yaw = float(sampled[0, 2])

    velocity = sampled[:, 3:5] * float(speed_scale)
    pos = np.zeros((horizon, 2), dtype=np.float32)
    cursor = prev_xy.astype(np.float32)
    for k in range(horizon):
        cursor = cursor + velocity[k] * dt
        pos[k] = cursor
    sampled[:, :2] = pos

    if abs(float(lateral_offset)) > 1e-6:
        # Start at zero offset so the first predicted state remains connected to
        # the observed current state.  A quintic ramp avoids lateral jerk.
        frac = (np.arange(1, horizon + 1, dtype=np.float32) / max(horizon, 1)).clip(0.0, 1.0)
        smooth = 10.0 * frac**3 - 15.0 * frac**4 + 6.0 * frac**5
        yaw = sampled[:, 2]
        lateral = np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1)
        sampled[:, :2] += lateral * (float(lateral_offset) * smooth[:, None])

    if current is None:
        pseudo_current = np.zeros(11, dtype=np.float32)
        pseudo_current[:2] = prev_xy
        pseudo_current[6] = prev_yaw
        pseudo_current[7:9] = sampled[0, 5:7]
        current = pseudo_current
    return repair_planar_kinematics(sampled, current=current, dt=dt)


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
