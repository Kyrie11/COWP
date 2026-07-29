from __future__ import annotations

import math

import numpy as np

from ocrap.data.schema import CandidatePrefix, CounterfactualFuture, SceneHistory
from ocrap.planning.route_lattice import project_to_route, offset_route_points
from ocrap.utils.seed import stable_seed

from .replay import copy_future, inject_ego_prefix


class ReactivePolicy:
    def __init__(self, cfg: dict, variant: int = 0):
        self.cfg = cfg
        self.variant = variant
        self.dt = 1.0 / float(cfg.get("sample_rate_hz", 10.0))

    def step_vehicle(self, prev: np.ndarray, route_context, ego_state: np.ndarray, traffic_light_state: np.ndarray | None) -> np.ndarray:
        out = prev.copy()
        speed = float(math.hypot(prev[3], prev[4]))
        heading = float(prev[7])
        rel = prev[:2] - ego_state[:2]
        dist = float(np.linalg.norm(rel))
        desired_gap = 4.0 + 0.7 * speed
        accel = 0.5 * (float(self.cfg.get("speed_limit_default", 13.4)) - speed) / max(float(self.cfg.get("speed_limit_default", 13.4)), 1.0)
        # IDM-like braking when the ego prefix intrudes into the route corridor.
        if dist < 30.0:
            closing = max(0.0, float(np.dot(np.array([prev[3] - ego_state[2], prev[4] - ego_state[3]]), -rel / max(dist, 1e-6))))
            accel -= (1.0 + 0.25 * self.variant) * (desired_gap / max(dist, 1.0)) ** 2
            accel -= 0.15 * closing
        if traffic_light_state is not None and len(traffic_light_state) and np.any(traffic_light_state[..., 1] > 0):
            accel = min(accel, -1.0)
        speed = max(0.0, speed + np.clip(accel, -5.0, 2.0) * self.dt)
        out[3] = speed * math.cos(heading)
        out[4] = speed * math.sin(heading)
        out[0] = prev[0] + out[3] * self.dt
        out[1] = prev[1] + out[4] * self.dt
        out[5] = (out[3] - prev[3]) / self.dt
        out[6] = (out[4] - prev[4]) / self.dt
        out[8] = math.sin(heading)
        out[9] = math.cos(heading)
        out[14] = 1.0
        out[15] = max(float(prev[15]), 0.8)
        return out

    def step_ped_cyclist(self, prev: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = prev.copy()
        heading = float(prev[7] + rng.normal(0.0, 0.03 + 0.01 * self.variant))
        speed = float(np.clip(math.hypot(prev[3], prev[4]) + rng.normal(0.0, 0.05), 0.0, 6.0))
        out[3] = speed * math.cos(heading)
        out[4] = speed * math.sin(heading)
        out[0] = prev[0] + out[3] * self.dt
        out[1] = prev[1] + out[4] * self.dt
        out[7] = heading
        out[8] = math.sin(heading)
        out[9] = math.cos(heading)
        return out


def reactive_future(history: SceneHistory, prefix: CandidatePrefix, total_steps: int, idx: int, prior: float, cfg: dict) -> CounterfactualFuture:
    states, valid = copy_future(history, total_steps)
    inject_ego_prefix(states, valid, prefix)
    rng = np.random.default_rng(stable_seed("reactive", history.scene_id, history.time_index, prefix.macro_id, idx))
    policy = ReactivePolicy(cfg, idx)
    T_p = prefix.prefix_states.shape[0]
    for a in range(1, states.shape[1]):
        # invalid current state remains absent until logged validity becomes true.
        if not bool(valid[0, a]):
            continue
        typ = int(round(float(states[0, a, 13]))) if states.shape[-1] > 13 else 1
        for t in range(1, total_steps):
            if not bool(valid[t - 1, a]):
                continue
            prev = states[t - 1, a]
            ego = states[min(t, T_p - 1), 0]
            if typ in (1, 3, 4):
                states[t, a] = policy.step_vehicle(prev, None, np.array([ego[0], ego[1], ego[3], ego[4]], dtype=np.float32), history.dynamic_map[min(t, len(history.dynamic_map)-1)] if len(history.dynamic_map) else None)
            else:
                states[t, a] = policy.step_ped_cyclist(prev, rng)
            valid[t, a] = True
    return CounterfactualFuture(idx, "reactive", prior, states, valid, {"reactive_policy": "ocrap_idm_route_following_surrogate", "reactive_variant": int(idx), "runtime_backend": "ocrap_surrogate_idm", "waymax_runtime": False})
