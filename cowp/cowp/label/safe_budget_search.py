from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cowp.core.constants import PriorityRelation
from cowp.geometry.collision import unsafe_between, unsafe_between_bool
from cowp.label.burden import compute_burden


@dataclass(frozen=True)
class BudgetProfile:
    name: str
    priority: int
    schedule: tuple[tuple[float, float, float], ...]  # (start_s, duration_s, accel_mps2)
    lateral_offset_m: float = 0.0
    hard: bool = False


@dataclass(frozen=True)
class PreparedBudgetTrajectory:
    trajectory: np.ndarray
    profile: BudgetProfile
    safe_burden: float
    safe_components: np.ndarray
    safe_score: float


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


def build_safe_budget_trajectory_bank(
    current: np.ndarray,
    horizon: int,
    dt: float,
    cfg: dict,
) -> list[tuple[np.ndarray, BudgetProfile]]:
    """Precompute candidate-independent safe-budget trajectories for one agent.

    The trajectory rollout depends only on the agent current state and the
    configured typed profile. Candidate-conditioned safety/burden scoring is
    deliberately *not* cached, so this is an exact engineering optimization
    with no label-semantic change.
    """
    rows: list[tuple[np.ndarray, BudgetProfile]] = []
    for prof in default_budget_profiles(cfg):
        tr = rollout_accel_schedule(
            current, horizon, dt, prof.schedule,
            lateral_offset_m=prof.lateral_offset_m,
        )
        rows.append((tr, prof))
    return rows



def prepare_safe_budget_trajectory_bank(
    trajectory_bank: list[tuple[np.ndarray, BudgetProfile]],
    *,
    object_type: int,
    cfg: dict,
    natural_ref: np.ndarray | None,
    rho: PriorityRelation,
) -> list[PreparedBudgetTrajectory]:
    """Precompute candidate-independent burden for the collision-free case.

    For a response already proven safe, ``compute_burden(...,
    risk_known_zero=True)`` does not depend on the ego candidate.  Storing this
    value once per agent/profile removes millions of duplicate kinematics and
    progress computations during a full build without changing any label.
    """
    s_cfg = cfg.get("response", {}).get("safe_budget_search", {})
    beta_margin = float(s_cfg.get("hard_profile_penalty", 0.25))
    out: list[PreparedBudgetTrajectory] = []
    for tr, prof in trajectory_bank:
        b, comps = compute_burden(
            tr, None, cfg, object_type, natural_ref=natural_ref, rho=rho,
            risk_known_zero=True,
        )
        priority_cost = 0.05 * prof.priority + (beta_margin if prof.hard else 0.0)
        option_cost = 0.25 * float(comps[4])
        out.append(PreparedBudgetTrajectory(
            trajectory=tr, profile=prof, safe_burden=float(b),
            safe_components=np.asarray(comps).copy(),
            safe_score=float(b) + priority_cost + option_cost,
        ))
    return out


def typed_safe_budget_search(
    current: np.ndarray,
    horizon: int,
    dt: float,
    ego_candidate: np.ndarray,
    object_type: int,
    cfg: dict,
    natural_ref: np.ndarray | None = None,
    rho: PriorityRelation = PriorityRelation.UNKNOWN,
) -> list[tuple[np.ndarray, str, float]]:
    """Candidate-conditioned typed safe-budget response search.

    This is a lightweight beam over typed response families rather than a blind
    acceleration primitive enumeration.  Each profile receives a lexicographic
    safe-budget score: first avoid collision/near-miss, then prefer low burden,
    then prefer lower priority/hardness cost.  Returned profiles can be merged
    into the response bank as OPT responses.
    """
    rows = typed_safe_budget_search_evaluated(current, horizon, dt, ego_candidate, object_type, cfg, natural_ref=natural_ref, rho=rho)
    return [(tr, name, burden) for tr, name, burden, _safe, _comps in rows]


def typed_safe_budget_search_evaluated(
    current: np.ndarray,
    horizon: int,
    dt: float,
    ego_candidate: np.ndarray,
    object_type: int,
    cfg: dict,
    natural_ref: np.ndarray | None = None,
    rho: PriorityRelation = PriorityRelation.UNKNOWN,
    trajectory_bank: list[tuple[np.ndarray, BudgetProfile]] | list[PreparedBudgetTrajectory] | None = None,
) -> list[tuple[np.ndarray, str, float, bool, np.ndarray]]:
    """Evaluate the typed safe-budget bank with exact semantics-preserving fast paths.

    When a prepared bank is supplied, collision-free burden is reused exactly.
    If the configured unsafe penalty dominates every possible prepared safe
    score, the search can stop once ``N`` safe profiles have been found in
    ascending safe-score order: no unseen unsafe profile can enter the top-N.
    If fewer than N safe profiles exist, every unsafe row is evaluated exactly
    as in the legacy implementation before final sorting.
    """
    s_cfg = cfg.get("response", {}).get("safe_budget_search", {})
    eng = cfg.get("engineering", {})
    beam_width = int(s_cfg.get("beam_width", 16))
    max_return = int(s_cfg.get("max_return", 16))
    n_return = max(1, min(max_return, beam_width))
    beta_margin = float(s_cfg.get("hard_profile_penalty", 0.25))
    unsafe_penalty = float(s_cfg.get("unsafe_penalty", 100.0))
    fast_bool = bool(eng.get("unsafe_bool_fastpath", True))
    early_stop = bool(eng.get("safe_budget_early_stop_fastpath", True))

    bank = trajectory_bank if trajectory_bank is not None else build_safe_budget_trajectory_bank(current, horizon, dt, cfg)
    prepared = bool(bank) and isinstance(bank[0], PreparedBudgetTrajectory)
    if prepared:
        pbank = list(bank)  # type: ignore[arg-type]
    else:
        pbank = prepare_safe_budget_trajectory_bank(
            list(bank), object_type=object_type, cfg=cfg, natural_ref=natural_ref, rho=rho
        )

    # Stable sort preserves the legacy profile order on equal scores.
    ordered = sorted(enumerate(pbank), key=lambda x: (float(x[1].safe_score), int(x[0])))
    safe_rows: list[tuple[float, np.ndarray, str, float, bool, np.ndarray]] = []
    unsafe_rows: list[PreparedBudgetTrajectory] = []
    max_safe_score = max((float(x.safe_score) for x in pbank), default=0.0)
    unsafe_dominated = unsafe_penalty > max_safe_score

    for _idx, row in ordered:
        is_unsafe = (
            unsafe_between_bool(ego_candidate, row.trajectory, cfg, agent_type=object_type)
            if fast_bool else
            bool(unsafe_between(ego_candidate, row.trajectory, cfg, agent_type=object_type).unsafe)
        )
        if not is_unsafe:
            safe_rows.append((
                float(row.safe_score), row.trajectory, row.profile.name,
                float(row.safe_burden), True, np.asarray(row.safe_components).copy(),
            ))
            if early_stop and unsafe_dominated and len(safe_rows) >= n_return:
                # Because ``ordered`` is ascending in exact safe score, these are
                # the N best safe rows. Every unsafe row has score >= unsafe_penalty
                # and therefore cannot displace them.
                safe_rows.sort(key=lambda x: x[0])
                return [(tr, name, burden, safe, comps) for _, tr, name, burden, safe, comps in safe_rows[:n_return]]
        else:
            unsafe_rows.append(row)

    rows = list(safe_rows)
    # If fewer than N safe profiles exist, unsafe profiles can be returned and
    # must retain the legacy candidate-conditioned risk burden exactly.
    if len(safe_rows) < n_return:
        for row in unsafe_rows:
            burden, comps = compute_burden(
                row.trajectory, ego_candidate, cfg, object_type,
                natural_ref=natural_ref, rho=rho, risk_known_zero=False,
            )
            priority_cost = 0.05 * row.profile.priority + (beta_margin if row.profile.hard else 0.0)
            option_cost = 0.25 * float(comps[4])
            score = unsafe_penalty + float(burden) + priority_cost + option_cost
            rows.append((score, row.trajectory, row.profile.name, float(burden), False, comps))
    rows.sort(key=lambda x: x[0])
    return [(tr, name, burden, safe, comps) for _, tr, name, burden, safe, comps in rows[: min(n_return, len(rows))]]
