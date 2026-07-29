from __future__ import annotations

import math

import numpy as np

from ocrap.data.schema import CandidatePrefix, RecoveryOption


def rollout_recovery_controller(prefix: CandidatePrefix, option: RecoveryOption, horizon_steps: int, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    dt = 1.0 / float(cfg.get("sample_rate_hz", 10.0))
    state0 = prefix.prefix_states[-1].copy()
    states = np.zeros((horizon_steps, prefix.prefix_states.shape[1]), dtype=np.float32)
    controls = np.zeros((max(0, horizon_steps - 1), prefix.prefix_controls.shape[1] if prefix.prefix_controls.size else 4), dtype=np.float32)
    states[0] = state0
    mode = option.mode
    p = np.asarray(option.params, dtype=np.float32)
    diag: dict[str, float | bool | str] = {"mode": mode}
    for t in range(1, horizon_steps):
        prev = states[t - 1].copy()
        speed = max(0.0, float(prev[6]))
        heading = float(prev[4])
        a = 0.0
        steer = 0.0
        if mode == "stop":
            a_dec, s_stop = float(p[0]), max(float(p[1]), 1.0)
            req = speed * speed / max(2.0 * abs(a_dec), 1e-3)
            a = a_dec if req >= s_stop * (1.0 - t / max(horizon_steps, 1)) else min(0.0, a_dec * 0.4)
            diag["used_s_stop"] = s_stop
        elif mode == "brake_lane":
            a_dec, T_brake = float(p[0]), max(float(p[1]), dt)
            a = a_dec if (t * dt) <= T_brake else 0.0
            diag["used_T_brake"] = T_brake
        elif mode == "lateral_escape":
            d_y, v_tar, T_lat = float(p[0]), float(p[1]), max(float(p[2]), dt)
            a = np.clip((v_tar - speed) / max(T_lat, dt), -4.0, 2.0)
            steer = np.clip(0.25 * d_y / max(T_lat, dt), -0.55, 0.55)
            diag["used_d_y"] = d_y
        elif mode == "yield_rejoin":
            a_yield, s_gap, T_rejoin = float(p[0]), float(p[1]), max(float(p[2]), dt)
            a = a_yield if t * dt < T_rejoin * 0.5 else 0.4
            steer = np.clip(-0.05 * prev[1], -0.3, 0.3)
            diag["used_s_gap"] = s_gap
        elif mode == "pull_over":
            d_y, s_shoulder, v_stop = float(p[0]), float(p[1]), float(p[2])
            a = np.clip((v_stop - speed) / 2.0, -4.0, 1.0)
            steer = np.clip(0.18 * d_y / max(s_shoulder / max(speed, 1.0), 1.0), -0.55, 0.55)
            diag["uses_shoulder"] = True
        elif mode == "mitigate_contact":
            a_dec, delta_psi, v_impact = float(p[0]), float(p[1]), float(p[2])
            a = a_dec if speed > v_impact else 0.0
            steer = np.clip(delta_psi, -0.4, 0.4)
            diag["used_v_impact"] = v_impact
        elif mode == "post_contact_stabilize":
            k_psi, k_r, a_decay = float(p[0]), float(p[1]), float(p[2])
            a = a_decay
            steer = np.clip(-k_psi * prev[4] - k_r * prev[5], -0.55, 0.55)
            diag["used_k_r"] = k_r
        elif mode == "avoid_secondary":
            d_y, a_dec, s_clear = float(p[0]), float(p[1]), float(p[2])
            a = a_dec
            steer = np.clip(0.12 * np.sign(d_y) * min(abs(d_y), s_clear / 3.0), -0.55, 0.55)
            diag["used_s_clear"] = s_clear
        speed_next = max(0.0, speed + a * dt)
        yaw_rate = speed_next * math.tan(float(steer)) / max(float(cfg.get("wheelbase_m", 2.8)), 1e-3)
        heading_next = heading + yaw_rate * dt
        cur = prev.copy()
        cur[6] = speed_next
        cur[4] = heading_next
        cur[5] = yaw_rate
        cur[2] = speed_next * math.cos(heading_next)
        cur[3] = speed_next * math.sin(heading_next)
        cur[0] = prev[0] + cur[2] * dt
        cur[1] = prev[1] + cur[3] * dt
        states[t] = cur
        if t - 1 < len(controls):
            controls[t - 1, 0] = a
            controls[t - 1, 1] = steer
            if t > 1:
                controls[t - 1, 2] = (controls[t - 1, 0] - controls[t - 2, 0]) / dt
                controls[t - 1, 3] = (controls[t - 1, 1] - controls[t - 2, 1]) / dt
    return states, controls, diag
