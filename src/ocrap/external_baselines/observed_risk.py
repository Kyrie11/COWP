from __future__ import annotations

"""Deployable, observation-only risk surrogates for external baselines.

The OC-RAP dataset contains teacher-only counterfactual tensors (``m_star`` and
recoverability labels).  Those tensors are valid *evaluation labels* but are not
available to a deployable external planner.  This module constructs a compact
multi-modal future set directly from the observable history and scores each ego
candidate against it.  The implementation is intentionally deterministic and
vectorized so it can also be used in the Waymax closed-loop hot path.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObservedRiskProfile:
    losses: np.ndarray
    weights: np.ndarray
    margins: np.ndarray
    min_clearance: float
    min_ttc: float
    collision_probability: float
    expected_loss: float
    cvar_loss: float
    worst_loss: float
    backup_margin: float
    severity_proxy: float


def _cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    try:
        return float(pcfg.get(key, default))
    except Exception:
        return float(default)


def _weighted_upper_cvar(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return 0.0
    values, weights = values[valid], weights[valid]
    weights = weights / max(float(weights.sum()), 1e-9)
    order = np.argsort(values)[::-1]
    values, weights = values[order], weights[order]
    alpha = float(np.clip(alpha, 1e-4, 1.0))
    total = 0.0
    acc = 0.0
    for value, weight in zip(values, weights):
        take = min(float(weight), alpha - total)
        if take <= 0:
            break
        acc += float(value) * take
        total += take
    return float(acc / max(total, 1e-9))


def _resample_xy(xy: np.ndarray, count: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[0] == 0 or xy.shape[1] < 2:
        return np.zeros((count, 2), dtype=float)
    if xy.shape[0] == count:
        return np.nan_to_num(xy[:, :2])
    src = np.linspace(0.0, 1.0, xy.shape[0])
    dst = np.linspace(0.0, 1.0, count)
    return np.stack([np.interp(dst, src, xy[:, 0]), np.interp(dst, src, xy[:, 1])], axis=-1)


def _ego_candidate(d: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
    states = np.asarray(d.get("prefix_states", np.zeros((0, 0))), dtype=float)
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] < 2:
        ego = np.asarray(d.get("ego_state", np.zeros((9,))), dtype=float).reshape(-1)
        states = np.zeros((2, 9), dtype=float)
        states[:, : min(ego.size, 9)] = ego[:9]
    xy = states[:, :2]
    if states.shape[1] >= 7:
        speed = np.maximum(0.0, states[:, 6])
    elif states.shape[1] >= 4:
        speed = np.hypot(states[:, 2], states[:, 3])
    else:
        speed = np.zeros((states.shape[0],), dtype=float)
    length = float(np.nanmedian(states[:, 7])) if states.shape[1] >= 8 else 4.8
    width = float(np.nanmedian(states[:, 8])) if states.shape[1] >= 9 else 2.0
    return np.nan_to_num(xy), np.nan_to_num(speed), max(length, 1.0), max(width, 0.5)


def _last_observed_agents(d: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 0))), dtype=float)
    valid = np.asarray(d.get("agent_valid", np.zeros((0, 0))), dtype=bool)
    if hist.ndim != 3 or valid.ndim != 2 or hist.shape[:2] != valid.shape:
        return []
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
    # Index 0 is the ego track in OC-RAP histories.
    for a in range(1, hist.shape[1]):
        idx = np.where(valid[:, a])[0]
        if idx.size == 0:
            continue
        j = int(idx[-1])
        s = hist[j, a]
        if s.size < 5:
            continue
        p = np.asarray(s[:2], dtype=float)
        v = np.asarray([s[3], s[4]], dtype=float)
        acc = np.asarray([s[5], s[6]], dtype=float) if s.size >= 7 else np.zeros(2, dtype=float)
        length = float(s[10]) if s.size >= 12 and np.isfinite(s[10]) and s[10] > 0 else 4.8
        width = float(s[11]) if s.size >= 12 and np.isfinite(s[11]) and s[11] > 0 else 2.0
        out.append((p, v, acc, length, width))
    return out


def _hypotheses() -> tuple[list[tuple[str, float, float, float]], np.ndarray]:
    # name, speed multiplier, longitudinal acceleration bias, lateral drift m/s
    specs = [
        ("constant_velocity", 1.00, 0.0, 0.0),
        ("yield", 0.65, -1.5, 0.0),
        ("accelerate", 1.20, 1.2, 0.0),
        ("hard_brake", 0.45, -3.5, 0.0),
        ("left_drift", 1.00, 0.0, 0.65),
        ("right_drift", 1.00, 0.0, -0.65),
        ("delay_noise", 1.08, 0.4, 0.25),
    ]
    weights = np.asarray([0.34, 0.14, 0.14, 0.10, 0.10, 0.10, 0.08], dtype=float)
    return specs, weights / weights.sum()


def observed_risk_profile(d: dict[str, Any], cfg: dict[str, Any]) -> ObservedRiskProfile:
    ego_xy, ego_speed, ego_length, ego_width = _ego_candidate(d)
    T = max(int(ego_xy.shape[0]), 2)
    ego_xy = _resample_xy(ego_xy, T)
    ego_speed = np.interp(np.linspace(0.0, 1.0, T), np.linspace(0.0, 1.0, max(ego_speed.size, 1)), ego_speed if ego_speed.size else np.zeros(1))
    dt = _cfg_float(cfg, "risk_dt", _cfg_float(cfg, "contact_dt", 0.1))
    times = np.arange(T, dtype=float) * max(dt, 1e-3)
    agents = _last_observed_agents(d)
    specs, weights = _hypotheses()
    if not agents:
        losses = np.zeros((len(specs),), dtype=float)
        margins = np.full((len(specs),), 50.0, dtype=float)
        return ObservedRiskProfile(losses, weights, margins, 50.0, float("inf"), 0.0, 0.0, 0.0, 0.0, 50.0, 0.0)

    ego_radius = 0.5 * float(np.hypot(ego_length, ego_width))
    clearance_buffer = _cfg_float(cfg, "risk_clearance_buffer_m", 0.75)
    loss_scale = max(_cfg_float(cfg, "risk_clearance_scale_m", 2.0), 1e-3)
    collision_temp = max(_cfg_float(cfg, "risk_collision_temperature_m", 0.8), 1e-3)
    severity_speed = max(_cfg_float(cfg, "risk_severity_speed_mps", 12.0), 1e-3)
    losses: list[float] = []
    margins: list[float] = []
    ttc_values: list[float] = []
    severity_values: list[float] = []

    for _, speed_mult, accel_bias, lateral_drift in specs:
        min_clear = float("inf")
        min_t = float("inf")
        rel_speed_at_min = 0.0
        for p0, v0, a0, length, width in agents:
            speed0 = float(np.linalg.norm(v0))
            direction = v0 / speed0 if speed0 > 0.3 else np.asarray([1.0, 0.0])
            normal = np.asarray([-direction[1], direction[0]])
            v = speed_mult * v0
            a = a0 + accel_bias * direction
            # Do not let hard-braking hypotheses reverse the actor.
            pred = p0[None, :] + times[:, None] * v[None, :] + 0.5 * (times[:, None] ** 2) * a[None, :] + times[:, None] * lateral_drift * normal[None, :]
            if accel_bias < 0:
                along = (pred - p0[None, :]) @ direction
                pred = p0[None, :] + np.maximum(along, 0.0)[:, None] * direction[None, :] + times[:, None] * lateral_drift * normal[None, :]
            actor_radius = 0.5 * float(np.hypot(max(length, 0.5), max(width, 0.3)))
            center_dist = np.linalg.norm(ego_xy - pred, axis=-1)
            clear = center_dist - ego_radius - actor_radius - clearance_buffer
            j = int(np.argmin(clear))
            if float(clear[j]) < min_clear:
                min_clear = float(clear[j])
                min_t = float(times[j])
                actor_vel = v + times[j] * a + lateral_drift * normal
                ego_vel_mag = float(ego_speed[min(j, ego_speed.size - 1)])
                if j + 1 < T:
                    ego_dir = ego_xy[j + 1] - ego_xy[j]
                elif j > 0:
                    ego_dir = ego_xy[j] - ego_xy[j - 1]
                else:
                    ego_dir = np.asarray([1.0, 0.0])
                norm = float(np.linalg.norm(ego_dir))
                ego_vel = ego_vel_mag * (ego_dir / norm if norm > 1e-6 else np.asarray([1.0, 0.0]))
                rel_speed_at_min = float(np.linalg.norm(ego_vel - actor_vel))
        # Smooth collision probability and a speed-dependent severity term.
        collision_prob = 1.0 / (1.0 + np.exp(np.clip(min_clear / collision_temp, -40.0, 40.0)))
        severity = min(rel_speed_at_min / severity_speed, 2.0)
        proximity = np.exp(-max(min_clear, 0.0) / loss_scale)
        penetration = max(-min_clear, 0.0) / loss_scale
        loss = collision_prob + 0.35 * proximity + 0.45 * collision_prob * severity + 0.35 * penetration
        losses.append(float(np.clip(loss, 0.0, 4.0)))
        margins.append(float(min_clear))
        ttc_values.append(min_t)
        severity_values.append(float(severity))

    losses_a = np.asarray(losses, dtype=float)
    margins_a = np.asarray(margins, dtype=float)
    expected = float(np.sum(weights * losses_a))
    alpha = _cfg_float(cfg, "cvar_alpha", 0.2)
    cvar = _weighted_upper_cvar(losses_a, weights, alpha)
    worst = float(np.max(losses_a))
    collision_probability = float(np.sum(weights * (margins_a <= 0.0)))
    min_clearance = float(np.min(margins_a))
    min_ttc = float(np.min(ttc_values)) if ttc_values else float("inf")
    severity_proxy = float(np.sum(weights * np.asarray(severity_values, dtype=float)))

    v0 = float(ego_speed[0]) if ego_speed.size else 0.0
    decel = max(_cfg_float(cfg, "backup_deceleration_mps2", 5.0), 0.5)
    reaction = max(_cfg_float(cfg, "backup_reaction_time_s", 0.25), 0.0)
    stopping_distance = v0 * reaction + v0 * v0 / (2.0 * decel)
    backup_margin = min_clearance - stopping_distance
    return ObservedRiskProfile(
        losses=losses_a,
        weights=weights,
        margins=margins_a,
        min_clearance=min_clearance,
        min_ttc=min_ttc,
        collision_probability=collision_probability,
        expected_loss=expected,
        cvar_loss=cvar,
        worst_loss=worst,
        backup_margin=float(backup_margin),
        severity_proxy=severity_proxy,
    )
