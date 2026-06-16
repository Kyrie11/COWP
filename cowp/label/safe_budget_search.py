from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cowp.geometry.collision import unsafe_between
from cowp.label.burden import compute_burden


@dataclass(frozen=True)
class BudgetProfile:
    name: str
    priority: int
    schedule: tuple[tuple[float, float, float], ...]  # (start_s, duration_s, accel_mps2)
    lateral_offset_m: float = 0.0
    hard: bool = False


def rollout_accel_schedule(current: np.ndarray, horizon: int, dt: float, schedule: tuple[tuple[float, float, float], ...], lateral_offset_m: float = 0.0) -> np.ndarray:
    """Roll out a piecewise-constant longitudinal acceleration profile."""
    x, y = float(current[0]), float(current[1])
    heading = float(current[6] if len(current) >= 7 else current[2])
    speed = float(current[5] if len(current) >= 6 else np.linalg.norm(current[3:5]))
    length = float(current[7] if len(current) >= 8 and current[7] > 0 else 4.8)
    width = float(current[8] if len(current) >= 9 and current[8] > 0 else 1.9)
    direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
    lateral = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
    pos = np.array([x, y], dtype=np.float32)
    out = np.zeros((horizon, 7), dtype=np.float32)
    for t in range(horizon):
        sec = t * dt
        acc = 0.0
        for start_s, dur_s, a in schedule:
            if sec >= float(start_s) and sec < float(start_s) + float(dur_s):
                acc += float(a)
        speed = max(0.0, speed + acc * dt)
        pos = pos + direction * (speed * dt)
        frac = min(1.0, (t + 1) / max(horizon, 1))
        smooth = 10 * frac**3 - 15 * frac**4 + 6 * frac**5
        p = pos + lateral * (float(lateral_offset_m) * smooth)
        out[t] = [p[0], p[1], heading, direction[0] * speed, direction[1] * speed, length, width]
    return out


def default_budget_profiles(cfg: dict) -> list[BudgetProfile]:
    s = cfg.get("response", {}).get("safe_budget_search", {})
    comfort = [float(x) for x in s.get("comfort_decel_values_mps2", [-1.0, -1.5, -2.0])]
    hard = [float(x) for x in s.get("hard_decel_values_mps2", [-3.0, -4.0, -5.0])]
    delays = [float(x) for x in s.get("reaction_delays_s", [0.0, 0.3, 0.6])]
    recover = [float(x) for x in s.get("recover_accel_values_mps2", [0.0, 0.5])]
    profiles: list[BudgetProfile] = [BudgetProfile("preserve_speed", 0, ((0.0, 0.1, 0.0),))]
    for d in delays:
        for a in comfort:
            profiles.append(BudgetProfile("comfort_yield", 1, ((d, 1.5, a), (d + 1.5, 2.0, 0.0))))
            for r in recover:
                profiles.append(BudgetProfile("yield_then_recover", 2, ((d, 1.5, a), (d + 1.5, 1.5, r))))
        for a in hard:
            profiles.append(BudgetProfile("hard_yield", 3, ((d, 1.0, a),), hard=True))
    # Option-preserving mild lateral slack for merge/gap cases.  It remains small
    # and is filtered by safety/burden, so it does not create unrealistic evasive swerves.
    for lat in s.get("lateral_slack_offsets_m", [-0.25, 0.25]):
        profiles.append(BudgetProfile("small_lateral_slack", 2, ((0.0, 0.1, 0.0),), lateral_offset_m=float(lat)))
    return profiles


def typed_safe_budget_search(
    current: np.ndarray,
    horizon: int,
    dt: float,
    ego_candidate: np.ndarray,
    object_type: int,
    cfg: dict,
    natural_ref: np.ndarray | None = None,
) -> list[tuple[np.ndarray, str, float]]:
    """Candidate-conditioned typed safe-budget response search.

    This is a lightweight beam over typed response families rather than a blind
    acceleration primitive enumeration.  Each profile receives a lexicographic
    safe-budget score: first avoid collision/near-miss, then prefer low burden,
    then prefer lower priority/hardness cost.  Returned profiles can be merged
    into the response bank as OPT responses.
    """
    s_cfg = cfg.get("response", {}).get("safe_budget_search", {})
    beam_width = int(s_cfg.get("beam_width", 16))
    max_return = int(s_cfg.get("max_return", 16))
    profiles = default_budget_profiles(cfg)
    rows: list[tuple[float, np.ndarray, str, float]] = []
    beta_margin = float(s_cfg.get("hard_profile_penalty", 0.25))
    unsafe_penalty = float(s_cfg.get("unsafe_penalty", 100.0))
    for prof in profiles:
        tr = rollout_accel_schedule(current, horizon, dt, prof.schedule, lateral_offset_m=prof.lateral_offset_m)
        unsafe = unsafe_between(ego_candidate, tr, cfg, agent_type=object_type)
        burden, comps = compute_burden(tr, ego_candidate, cfg, object_type, natural_ref=natural_ref)
        priority_cost = 0.05 * prof.priority + (beta_margin if prof.hard else 0.0)
        # Penalize option component for hard profiles even when collision-free, so
        # the search prefers natural/comfort-preserving responses when available.
        option_cost = 0.25 * float(comps[4])
        score = (unsafe_penalty if unsafe.unsafe else 0.0) + float(burden) + priority_cost + option_cost
        rows.append((score, tr, prof.name, float(burden)))
    rows.sort(key=lambda x: x[0])
    return [(tr, name, burden) for _, tr, name, burden in rows[: max(1, min(max_return, beam_width, len(rows)))]]
