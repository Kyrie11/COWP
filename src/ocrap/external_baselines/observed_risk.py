from __future__ import annotations

"""Deployable observation-only risk models shared by external baselines.

Teacher counterfactual tensors are deliberately excluded.  The module predicts a
small deterministic multi-modal actor set from visible history, caches that set
once per candidate group, and vectorizes candidate scoring.  It exposes both
scalar risk summaries and temporal/mode-resolved curves required by contingency
planning, predictive safety filters, calibration diagnostics, and videos.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ObservedRiskContext:
    hypothesis_names: tuple[str, ...]
    weights: np.ndarray                    # [H]
    times: np.ndarray                      # [T]
    actor_xy: np.ndarray                   # [H,A,T,2]
    actor_velocity: np.ndarray             # [H,A,T,2]
    actor_radius: np.ndarray               # [A]
    clearance_buffer_m: float


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
    hypothesis_names: tuple[str, ...] = ()
    collision_probabilities: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    min_ttc_by_mode: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    closest_approach_time_by_mode: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    severity_by_mode: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    clearance_curves: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    loss_curves: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    backup_margin_curves: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))


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
    total = acc = 0.0
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
    for a in range(1, hist.shape[1]):  # index 0 is the SDC
        idx = np.where(valid[:, a])[0]
        if idx.size == 0:
            continue
        s = hist[int(idx[-1]), a]
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


def build_observed_risk_context(d: dict[str, Any], cfg: dict[str, Any], *, horizon: int | None = None) -> ObservedRiskContext:
    """Predict visible actors once for all candidates in one scene-time group."""
    ego_xy, _, _, _ = _ego_candidate(d)
    T = max(int(horizon or ego_xy.shape[0]), 2)
    dt = _cfg_float(cfg, "risk_dt", _cfg_float(cfg, "contact_dt", 0.1))
    times = np.arange(T, dtype=float) * max(dt, 1e-3)
    agents = _last_observed_agents(d)
    specs, weights = _hypotheses()
    H, A = len(specs), len(agents)
    if A == 0:
        return ObservedRiskContext(tuple(x[0] for x in specs), weights, times,
                                   np.zeros((H, 0, T, 2), dtype=float),
                                   np.zeros((H, 0, T, 2), dtype=float),
                                   np.zeros((0,), dtype=float),
                                   _cfg_float(cfg, "risk_clearance_buffer_m", 0.75))

    p0 = np.stack([x[0] for x in agents], axis=0)                     # [A,2]
    v0 = np.stack([x[1] for x in agents], axis=0)
    a0 = np.stack([x[2] for x in agents], axis=0)
    dims = np.asarray([[x[3], x[4]] for x in agents], dtype=float)
    radii = 0.5 * np.hypot(np.maximum(dims[:, 0], 0.5), np.maximum(dims[:, 1], 0.3))
    speed = np.linalg.norm(v0, axis=-1)
    direction = np.divide(v0, speed[:, None], out=np.tile(np.asarray([[1.0, 0.0]]), (A, 1)), where=speed[:, None] > 0.3)
    normal = np.stack([-direction[:, 1], direction[:, 0]], axis=-1)

    actor_xy = np.zeros((H, A, T, 2), dtype=float)
    actor_v = np.zeros_like(actor_xy)
    t = times[None, :, None]
    for h, (_, speed_mult, accel_bias, lateral_drift) in enumerate(specs):
        v = speed_mult * v0
        a = a0 + accel_bias * direction
        pred = p0[:, None, :] + t * v[:, None, :] + 0.5 * t**2 * a[:, None, :] + t * lateral_drift * normal[:, None, :]
        vel = v[:, None, :] + t * a[:, None, :] + lateral_drift * normal[:, None, :]
        if accel_bias < 0:
            along = np.einsum("atd,ad->at", pred - p0[:, None, :], direction)
            along = np.maximum(along, 0.0)
            pred = p0[:, None, :] + along[..., None] * direction[:, None, :] + t * lateral_drift * normal[:, None, :]
            stopped = along <= 1e-8
            vel = np.where(stopped[..., None], lateral_drift * normal[:, None, :], vel)
        actor_xy[h], actor_v[h] = pred, vel
    return ObservedRiskContext(tuple(x[0] for x in specs), weights, times, actor_xy, actor_v, radii,
                               _cfg_float(cfg, "risk_clearance_buffer_m", 0.75))


def _ego_velocity(xy: np.ndarray, speed: np.ndarray, dt: float) -> np.ndarray:
    tangent = np.gradient(xy, max(float(dt), 1e-3), axis=0)
    norm = np.linalg.norm(tangent, axis=-1)
    direction = np.divide(tangent, norm[:, None], out=np.tile(np.asarray([[1.0, 0.0]]), (xy.shape[0], 1)), where=norm[:, None] > 1e-6)
    return direction * speed[:, None]


def score_candidate_with_context(d: dict[str, Any], cfg: dict[str, Any], context: ObservedRiskContext) -> ObservedRiskProfile:
    ego_xy, ego_speed, ego_length, ego_width = _ego_candidate(d)
    T = int(context.times.size)
    ego_xy = _resample_xy(ego_xy, T)
    src = np.linspace(0.0, 1.0, max(ego_speed.size, 1))
    ego_speed = np.interp(np.linspace(0.0, 1.0, T), src, ego_speed if ego_speed.size else np.zeros(1))
    H = len(context.hypothesis_names)
    if context.actor_xy.shape[1] == 0:
        losses = np.zeros((H,), dtype=float)
        margins = np.full((H,), 50.0, dtype=float)
        curves = np.full((H, T), 50.0, dtype=float)
        return ObservedRiskProfile(losses, context.weights, margins, 50.0, float("inf"), 0.0, 0.0, 0.0, 0.0, 50.0, 0.0,
                                   context.hypothesis_names, np.zeros(H), np.full(H, np.inf), np.zeros(H), np.zeros(H), curves,
                                   np.zeros((H, T)), curves.copy())

    ego_radius = 0.5 * float(np.hypot(ego_length, ego_width))
    delta = context.actor_xy - ego_xy[None, None, :, :]              # [H,A,T,2]
    center = np.linalg.norm(delta, axis=-1)
    clearance = center - ego_radius - context.actor_radius[None, :, None] - context.clearance_buffer_m
    mode_clearance = np.min(clearance, axis=1)                        # [H,T]
    margins = np.min(mode_clearance, axis=1)
    closest_idx = np.argmin(mode_clearance, axis=1)
    closest_times = context.times[closest_idx]

    ttc_threshold = _cfg_float(cfg, "risk_ttc_clearance_threshold_m", 0.0)
    unsafe = mode_clearance <= ttc_threshold
    has_unsafe = np.any(unsafe, axis=1)
    first_idx = np.argmax(unsafe, axis=1)
    min_ttc_by_mode = np.where(has_unsafe, context.times[first_idx], np.inf)

    collision_temp = max(_cfg_float(cfg, "risk_collision_temperature_m", 0.8), 1e-3)
    loss_scale = max(_cfg_float(cfg, "risk_clearance_scale_m", 2.0), 1e-3)
    severity_speed = max(_cfg_float(cfg, "risk_severity_speed_mps", 12.0), 1e-3)
    collision_curve = 1.0 / (1.0 + np.exp(np.clip(mode_clearance / collision_temp, -40.0, 40.0)))
    proximity_curve = np.exp(-np.maximum(mode_clearance, 0.0) / loss_scale)
    penetration_curve = np.maximum(-mode_clearance, 0.0) / loss_scale

    # Relative speed for the actor with minimum clearance in each (mode,time).
    nearest_actor = np.argmin(clearance, axis=1)                      # [H,T]
    ego_v = _ego_velocity(ego_xy, ego_speed, context.times[1] - context.times[0] if T > 1 else 0.1)
    hidx = np.arange(H)[:, None]
    tidx = np.arange(T)[None, :]
    nearest_v = context.actor_velocity[hidx, nearest_actor, tidx]
    severity_curve = np.clip(np.linalg.norm(ego_v[None, :, :] - nearest_v, axis=-1) / severity_speed, 0.0, 2.0)
    loss_curve = collision_curve + 0.35 * proximity_curve + 0.45 * collision_curve * severity_curve + 0.35 * penetration_curve
    loss_curve = np.clip(loss_curve, 0.0, 4.0)
    aggregation = str((((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {}).get("risk_temporal_aggregation", "max"))).lower()
    if aggregation == "mean":
        losses = np.mean(loss_curve, axis=1)
    elif aggregation == "discounted_mean":
        discount = np.exp(-_cfg_float(cfg, "risk_temporal_discount", 0.15) * context.times)
        losses = np.sum(loss_curve * discount[None, :], axis=1) / max(float(discount.sum()), 1e-9)
    else:
        losses = np.max(loss_curve, axis=1)

    decel = max(_cfg_float(cfg, "backup_deceleration_mps2", 5.0), 0.5)
    reaction = max(_cfg_float(cfg, "backup_reaction_time_s", 0.25), 0.0)
    stopping = ego_speed * reaction + ego_speed**2 / (2.0 * decel)
    backup_curves = mode_clearance - stopping[None, :]
    backup_margin = float(np.min(backup_curves))

    weights = context.weights
    expected = float(np.sum(weights * losses))
    cvar = _weighted_upper_cvar(losses, weights, _cfg_float(cfg, "cvar_alpha", 0.2))
    collision_prob_by_mode = np.max(collision_curve, axis=1)
    severity_by_mode = severity_curve[np.arange(H), closest_idx]
    finite_ttc = min_ttc_by_mode[np.isfinite(min_ttc_by_mode)]
    return ObservedRiskProfile(
        losses=np.asarray(losses, dtype=float), weights=weights, margins=margins,
        min_clearance=float(np.min(margins)), min_ttc=float(np.min(finite_ttc)) if finite_ttc.size else float("inf"),
        collision_probability=float(np.sum(weights * collision_prob_by_mode)), expected_loss=expected,
        cvar_loss=float(cvar), worst_loss=float(np.max(losses)), backup_margin=backup_margin,
        severity_proxy=float(np.sum(weights * collision_prob_by_mode * severity_by_mode)),
        hypothesis_names=context.hypothesis_names, collision_probabilities=collision_prob_by_mode,
        min_ttc_by_mode=min_ttc_by_mode, closest_approach_time_by_mode=closest_times,
        severity_by_mode=severity_by_mode, clearance_curves=mode_clearance,
        loss_curves=loss_curve, backup_margin_curves=backup_curves,
    )


def observed_risk_profiles(samples: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> list[ObservedRiskProfile]:
    """Score a candidate group while reusing actor forecasts for equal horizons."""
    if not samples:
        return []
    contexts: dict[int, ObservedRiskContext] = {}
    out: list[ObservedRiskProfile] = []
    for d in samples:
        xy, _, _, _ = _ego_candidate(d)
        T = max(int(xy.shape[0]), 2)
        if T not in contexts:
            contexts[T] = build_observed_risk_context(samples[0], cfg, horizon=T)
        out.append(score_candidate_with_context(d, cfg, contexts[T]))
    return out


def observed_risk_profile(d: dict[str, Any], cfg: dict[str, Any]) -> ObservedRiskProfile:
    context = build_observed_risk_context(d, cfg)
    return score_candidate_with_context(d, cfg, context)
