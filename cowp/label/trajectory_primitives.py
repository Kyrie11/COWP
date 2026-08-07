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



def priority_hold_release_trajectory(
    current: np.ndarray,
    horizon: int,
    dt: float,
    *,
    entry_distance_m: float,
    target_time_s: float,
    stop_margin_m: float = 4.0,
    release_speed_mps: float = 3.0,
    min_hold_s: float = 0.35,
) -> np.ndarray | None:
    """Early-commitment yield with a pre-conflict hold and timed release.

    Endpoint-only timing can still be coercive: ego may approach a protected
    agent aggressively and brake only near the conflict.  This primitive makes
    the yield commitment observable *early*: ego follows a jerk-smooth quintic
    segment to rest before the conflict boundary, holds there, then follows a
    second quintic segment that reaches the boundary no earlier than
    ``target_time_s``.

    Both moving segments match position, velocity and zero acceleration at their
    endpoints.  This removes the acceleration discontinuities of the first PCHR
    prototype and keeps the common proposal validator authoritative for the
    configured acceleration/jerk/map limits.  If the requested timing contains
    insufficient physical slack for a stop-hold-release maneuver, ``None`` is
    returned rather than fabricating a nominally yielding but dynamically
    assertive trajectory.
    """
    cur = np.asarray(current, dtype=np.float32).reshape(-1)
    H = int(horizon)
    dt = max(float(dt), 1.0e-4)
    d = float(entry_distance_m)
    T = float(target_time_s)
    margin = float(stop_margin_m)
    v_release = float(release_speed_mps)
    min_hold = max(float(min_hold_s), 0.0)
    if H <= 0 or cur.size < 7 or not all(np.isfinite(x) for x in (d, T, margin, v_release, min_hold)):
        return None
    if d <= 1.0 or T <= 2.0 * dt or margin <= 0.5 or margin >= d - 0.25 or v_release <= 0.2:
        return None

    v0 = float(max(cur[5] if cur.size > 5 else np.linalg.norm(cur[3:5]), 0.0))
    if v0 <= 0.15:
        return None
    stage_s = d - margin

    # Use conservative duration factors for zero-acceleration-endpoint quintics.
    # The release factor is deliberately longer because compressing a rest-to-
    # crossing-speed quintic creates large positive acceleration/jerk.  The
    # downstream validator still applies the exact configured limits.
    t_stop = max(0.8, 2.0 * stage_s / max(v0, 0.2))
    t_release = max(0.8, 2.5 * margin / max(v_release, 0.2))
    if t_stop + min_hold + t_release > T + 1.0e-6:
        return None
    release_start = T - t_release
    if release_start < t_stop + min_hold - 1.0e-6:
        return None

    x0, y0 = float(cur[0]), float(cur[1])
    yaw = float(cur[6])
    length = float(cur[7] if cur.size > 7 and cur[7] > 0 else 4.8)
    width = float(cur[8] if cur.size > 8 and cur[8] > 0 else 1.9)
    direction = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float32)

    def quintic_pos(t: float, duration: float, s0: float, v_start: float, s1: float, v_end: float) -> float:
        """Quintic Hermite position with zero acceleration at both endpoints."""
        duration = max(float(duration), 1.0e-6)
        u = float(np.clip(t / duration, 0.0, 1.0))
        u2, u3, u4, u5 = u * u, u**3, u**4, u**5
        h00 = 1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5
        h10 = u - 6.0 * u3 + 8.0 * u4 - 3.0 * u5
        h01 = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
        h11 = -4.0 * u3 + 7.0 * u4 - 3.0 * u5
        return float(h00 * s0 + h10 * duration * v_start + h01 * s1 + h11 * duration * v_end)

    out = np.zeros((H, 7), dtype=np.float32)
    prev_s = 0.0
    for k in range(H):
        t = float((k + 1) * dt)
        if t <= t_stop:
            s_t = quintic_pos(t, t_stop, 0.0, v0, stage_s, 0.0)
        elif t < release_start:
            s_t = stage_s
        elif t <= T:
            s_t = quintic_pos(t - release_start, t_release, stage_s, 0.0, d, v_release)
        else:
            s_t = d + v_release * (t - T)
        if not np.isfinite(s_t) or s_t < prev_s - 1.0e-3:
            # A backward segment is not a valid yielding commitment.
            return None
        prev_s = float(max(s_t, prev_s))
        pos = np.asarray([x0, y0], dtype=np.float32) + direction * prev_s
        # Velocity is repaired from finite differences below; seed it with the
        # correct direction so malformed intermediate values never leak out.
        out[k] = [pos[0], pos[1], yaw, 0.0, 0.0, length, width]
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
