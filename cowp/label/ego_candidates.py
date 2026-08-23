from __future__ import annotations

from collections import Counter

import numpy as np

from cowp.core.constants import MacroType, PriorityRelation, ProposalSource
from cowp.core.types import ScenarioData, future_states_to_traj7
from cowp.geometry.lane_graph import ConflictRegion, build_conflict_regions, build_scene_conflict_regions, trajectory_entry_to_region, tta_to_region
from cowp.label.trajectory_primitives import constant_accel_trajectory, piecewise_quintic_progress_trajectory, priority_hold_release_trajectory, repair_planar_kinematics, resample_logged, smooth_arrival_trajectory, smooth_stop_trajectory, smooth_terminal_speed_arrival_trajectory
from cowp.label.natural_alternatives import _map_route_polylines, _required_route_length, _timed_polyline_trajectory
from cowp.label.priority import determine_priority


def _candidate_map_compliant(traj: np.ndarray, lane_points: np.ndarray, cfg: dict) -> bool:
    """Cheap map-screening for generated ego primitives.

    This is deliberately a proposal filter rather than a claim of route-optimal
    kinodynamic planning.  It removes fixed lateral/terminal primitives that leave
    every nearby lane corridor, a major source of cached off-road outcomes.  The
    point cloud is local and subsampled once per scene, so the filter affects
    label/cache construction but not training-time throughput.
    """
    cand_cfg = cfg.get("candidate", {})
    if not bool(cand_cfg.get("map_filter_enabled", True)):
        return True
    min_points = int(cand_cfg.get("map_filter_min_lane_points", 64))
    # A single sparse polyline does not define a drivable corridor.  Applying a
    # nearest-point filter in that case rejects valid synthetic proposals and
    # makes toy/incomplete-map scenes protocol-dependent.
    if lane_points.size == 0 or int(lane_points.shape[0]) < min_points:
        return not bool(cand_cfg.get("map_filter_require_available", False))
    stride = max(1, int(cand_cfg.get("map_filter_stride", 4)))
    xy = np.asarray(traj, dtype=np.float32)[::stride, :2]
    if xy.size == 0 or not np.all(np.isfinite(xy)):
        return False
    # Chunk over trajectory samples to avoid a large temporary when dense WOMD
    # roadgraphs are present.  Lane points are already locally cropped below.
    d2 = ((xy[:, None, :] - lane_points[None, :, :]) ** 2).sum(axis=-1)
    dist = np.sqrt(np.min(d2, axis=1))
    threshold = float(cand_cfg.get("map_max_distance_vehicle_m", 5.0))
    min_fraction = float(cand_cfg.get("map_min_compliant_fraction", 0.80))
    hard_max = float(cand_cfg.get("map_hard_max_distance_m", 12.0))
    return bool(float(np.mean(dist <= threshold)) >= min_fraction and float(np.max(dist)) <= hard_max)


def _local_lane_point_cloud(scene: ScenarioData, center_xy: np.ndarray, cfg: dict) -> np.ndarray:
    cand_cfg = cfg.get("candidate", {})
    radius = float(cand_cfg.get("map_filter_local_radius_m", 140.0))
    sample_stride = max(1, int(cand_cfg.get("map_filter_lane_point_stride", 2)))
    chunks: list[np.ndarray] = []
    for lane in scene.map_data.lanes.values():
        xy = np.asarray(lane.xy, dtype=np.float32)
        if xy.size == 0:
            continue
        xy = xy[::sample_stride, :2]
        local = np.linalg.norm(xy - np.asarray(center_xy, dtype=np.float32)[None, :2], axis=-1) <= radius
        if local.any():
            chunks.append(xy[local])
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2), dtype=np.float32)


def _candidate_invalid_reason(traj: np.ndarray, cfg: dict, lane_points: np.ndarray | None = None) -> str | None:
    """Return a stable rejection reason for proposal diagnostics.

    v16.8.5 keeps candidate acceptance semantics unchanged; this helper only makes
    the reason observable when an entire scene loses its proposal bank.  That is
    essential for a paired proposal probe: a zero-valid scene is evidence about
    proposal feasibility, not a multiprocessing exception.
    """
    cand_cfg = cfg.get("candidate", {})
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    if len(traj) < int(cand_cfg.get("min_valid_horizon_steps", 50)):
        return "short_horizon"
    if not np.all(np.isfinite(traj)):
        return "nonfinite"
    speed = np.linalg.norm(traj[:, 3:5], axis=-1)
    acc = np.diff(speed, prepend=speed[0]) / max(dt, 1e-3)
    jerk = np.diff(acc, prepend=acc[0]) / max(dt, 1e-3)
    if np.nanmax(acc) > float(cand_cfg.get("max_accel_mps2", 4.0)) + 1e-3:
        return "max_accel"
    if -np.nanmin(acc) > float(cand_cfg.get("max_decel_mps2", 6.0)) + 1e-3:
        return "max_decel"
    ignore = max(0, int(cand_cfg.get("ignore_initial_jerk_steps", 0)))
    jerk_eval = np.abs(jerk[min(ignore, len(jerk)) :])
    if jerk_eval.size == 0 or not np.isfinite(jerk_eval).any():
        return "jerk_nonfinite"
    percentile = float(np.clip(cand_cfg.get("jerk_check_percentile", 100.0), 0.0, 100.0))
    jerk_stat = float(np.nanpercentile(jerk_eval, percentile))
    if jerk_stat > float(cand_cfg.get("max_jerk_mps3", 6.0)) + 1e-3:
        return "max_jerk"
    if lane_points is not None and not _candidate_map_compliant(traj, lane_points, cfg):
        return "map_filter"
    return None


def _candidate_valid(traj: np.ndarray, cfg: dict, lane_points: np.ndarray | None = None) -> bool:
    return _candidate_invalid_reason(traj, cfg, lane_points) is None




def _project_progress_profile_to_route(
    base_traj: np.ndarray, route_xy: np.ndarray, current: np.ndarray, dt: float, *, attach_max_m: float = 8.0
) -> np.ndarray | None:
    """Transfer a longitudinal timing profile onto a causal lane-topology route.

    Only the base profile's cumulative progress is used; no logged future geometry
    enters this transform.  This lets interaction-timing primitives follow curved
    WOMD lane topology while preserving the advertised arrival timing.
    """
    base = np.asarray(base_traj, dtype=np.float32)
    route = np.asarray(route_xy, dtype=np.float32)
    cur = np.asarray(current, dtype=np.float32).reshape(-1)
    if base.ndim != 2 or base.shape[1] < 7 or route.ndim != 2 or len(route) < 2 or cur.size < 7:
        return None
    route = route[:, :2]
    if not np.all(np.isfinite(route)) or not np.all(np.isfinite(base)):
        return None
    attach = float(np.linalg.norm(route[0] - cur[:2]))
    if attach > float(attach_max_m):
        return None
    if attach > 1.0e-3:
        route = np.concatenate([cur[None, :2], route], axis=0)
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    keep = np.ones(len(route), dtype=bool)
    keep[1:] = seg > 1.0e-3
    route = route[keep]
    if len(route) < 2:
        return None
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]).astype(np.float64)
    base_path = np.concatenate([cur[None, :2], base[:, :2]], axis=0)
    progress = np.cumsum(np.linalg.norm(np.diff(base_path, axis=0), axis=1)).astype(np.float64)
    if len(progress) != len(base) or float(np.max(progress, initial=0.0)) > float(arc[-1]) + 0.25:
        return None
    x = np.interp(progress, arc, route[:, 0])
    y = np.interp(progress, arc, route[:, 1])
    route_seg = np.diff(route, axis=0)
    route_h = np.unwrap(np.arctan2(route_seg[:, 1], route_seg[:, 0]).astype(np.float64))
    mid_arc = 0.5 * (arc[:-1] + arc[1:])
    yaw = np.interp(progress, mid_arc, route_h, left=route_h[0], right=route_h[-1])
    speed = np.linalg.norm(base[:, 3:5], axis=-1).astype(np.float64)
    out = np.zeros_like(base, dtype=np.float32)
    out[:, 0] = x.astype(np.float32)
    out[:, 1] = y.astype(np.float32)
    out[:, 2] = ((yaw + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    out[:, 3] = (np.cos(yaw) * speed).astype(np.float32)
    out[:, 4] = (np.sin(yaw) * speed).astype(np.float32)
    out[:, 5] = float(cur[7] if cur.size > 7 and cur[7] > 0 else 4.8)
    out[:, 6] = float(cur[8] if cur.size > 8 and cur[8] > 0 else 1.9)
    return repair_planar_kinematics(out, current=cur, dt=float(dt))


def _constant_accel_for_arrival(distance_m: float, speed_mps: float, target_time_s: float, cfg: dict) -> float | None:
    """Solve a bounded constant acceleration for a desired conflict arrival time.

    This produces interaction-timing proposals rather than mapping an arbitrary
    time offset directly to acceleration.  It is intentionally a proposal primitive;
    downstream conventional and RCOT certificates still decide feasibility.
    """
    t = float(target_time_s)
    d = float(distance_m)
    if not np.isfinite(t) or not np.isfinite(d) or t <= 0.35 or d <= 0.5:
        return None
    v0 = max(float(speed_mps), 0.0)
    a = 2.0 * (d - v0 * t) / max(t * t, 1.0e-3)
    cand_cfg = cfg.get("candidate", {})
    lo = float(cand_cfg.get("timing_envelope_min_accel_mps2", -3.5))
    hi = float(cand_cfg.get("timing_envelope_max_accel_mps2", 2.5))
    if a < lo - 0.75 or a > hi + 0.75:
        return None
    a = float(np.clip(a, lo, hi))
    # A constant-deceleration primitive cannot arrive at the conflict after it
    # has already stopped; clipping speed to zero would otherwise invalidate the
    # arrival-time equation and create a terminal jerk impulse.
    if v0 + a * t < -1.0e-3:
        return None
    return a


def _smooth_arrival_profile(
    distance_m: float,
    speed_mps: float,
    target_time_s: float,
    cfg: dict,
) -> tuple[float, float, float] | None:
    """Return (initial_accel, constant_jerk, terminal_speed) for cubic arrival.

    The profile is the closed-form solution used by ``smooth_arrival_trajectory``:
    it reaches the conflict-envelope entry at ``target_time_s`` while tapering
    acceleration to zero.  Bounds are applied to the physically meaningful peak
    acceleration (at t=0), terminal speed, and jerk before we spend a proposal
    slot on it.
    """
    d = float(distance_m)
    v0 = max(float(speed_mps), 0.0)
    T = float(target_time_s)
    if not np.isfinite(d) or not np.isfinite(T) or d <= 0.5 or T <= 0.35:
        return None
    a0 = 3.0 * (d - v0 * T) / max(T * T, 1.0e-9)
    jerk = -a0 / max(T, 1.0e-9)
    vT = -0.5 * v0 + 1.5 * d / max(T, 1.0e-9)
    cand_cfg = cfg.get("candidate", {})
    lo = float(cand_cfg.get("timing_envelope_min_accel_mps2", -3.5))
    hi = float(cand_cfg.get("timing_envelope_max_accel_mps2", 2.5))
    max_jerk = float(cand_cfg.get("max_jerk_mps3", 10.0))
    if a0 < lo - 1.0e-6 or a0 > hi + 1.0e-6 or vT < -1.0e-5 or abs(jerk) > max_jerk + 1.0e-6:
        return None
    return float(a0), float(jerk), float(max(vT, 0.0))


def _arrival_time_constant_accel(distance_m: float, speed_mps: float, accel_mps2: float) -> float:
    """Earliest positive solution of d = v t + 0.5 a t²."""
    d = float(distance_m)
    v = max(float(speed_mps), 0.0)
    a = float(accel_mps2)
    if not np.isfinite(d) or d <= 0.0:
        return 0.0
    if abs(a) < 1.0e-6:
        return d / max(v, 1.0e-3) if v > 1.0e-3 else float("inf")
    disc = v * v + 2.0 * a * d
    if disc < 0.0:
        return float("inf")
    root = float(np.sqrt(max(disc, 0.0)))
    roots = [(-v + root) / a, (-v - root) / a]
    positive = [t for t in roots if np.isfinite(t) and t > 1.0e-3]
    return min(positive) if positive else float("inf")


def _agent_tta_envelopes_to_region(
    scene: ScenarioData,
    region: ConflictRegion,
    cfg: dict,
) -> list[tuple[int, float, float, float]]:
    """Return causal (track-index, early, nominal, late) entry-time envelopes.

    v16.8.3 estimated time to the *region centre* as Euclidean distance divided
    by radial closing speed.  That is inconsistent with the ego certificate,
    which declares interaction when a vehicle envelope first enters the conflict
    region.  Here we intersect the agent's current velocity ray with the inflated
    circular conflict boundary and put the acceleration envelope on that entry
    distance.  Only current state is used; no logged future leaks into proposals.
    """
    cur = int(scene.current_time_index)
    states = np.asarray(scene.states[:, cur], dtype=np.float32)
    cand_cfg = cfg.get("candidate", {})
    max_agents = int(cand_cfg.get("timing_envelope_max_agents_per_region", cand_cfg.get("timing_envelope_max_agents", 6)))
    max_dist = float(cand_cfg.get("timing_envelope_agent_radius_m", 80.0))
    horizon_s = float(cfg.get("time", {}).get("future_steps", 80)) * float(cfg.get("time", {}).get("dt", 0.1))
    min_accel = float(cand_cfg.get("timing_envelope_agent_min_accel_mps2", -2.0))
    max_accel = float(cand_cfg.get("timing_envelope_agent_max_accel_mps2", 2.0))
    max_spread = float(cand_cfg.get("timing_envelope_agent_max_spread_s", 0.8))
    min_approach = float(cand_cfg.get("timing_envelope_min_approach_cos", 0.15))
    rows: list[tuple[float, float, int, float, float]] = []
    center = np.asarray(region.center_xy, dtype=np.float32)[:2]
    for j, st in enumerate(states):
        if j == int(scene.sdc_track_index) or st.shape[0] < 11 or float(st[10]) <= 0.5:
            continue
        to_center = center - st[:2]
        center_dist = float(np.linalg.norm(to_center))
        if not np.isfinite(center_dist) or center_dist > max_dist:
            continue
        vel = np.asarray(st[3:5], dtype=np.float32)
        vel_speed = float(np.linalg.norm(vel))
        speed = float(max(vel_speed, st[5] if st.shape[0] > 5 else 0.0, 0.0))
        if speed < 0.35:
            continue
        if vel_speed >= 0.35:
            direction = vel / max(vel_speed, 1.0e-6)
        else:
            heading = float(st[6])
            direction = np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float32)
        if center_dist <= 1.0e-6:
            continue
        approach = float(np.dot(direction, to_center) / center_dist)
        if approach < min_approach:
            continue

        length = float(st[7]) if st.shape[0] > 7 and st[7] > 0 else 4.8
        width = float(st[8]) if st.shape[0] > 8 and st[8] > 0 else 1.9
        inflated_radius = float(region.radius) + 0.5 * float(np.hypot(length, width))
        along = float(np.dot(to_center, direction))
        lateral_sq = max(center_dist * center_dist - along * along, 0.0)
        if along <= 0.0 or lateral_sq > inflated_radius * inflated_radius:
            continue
        entry_dist = along - float(np.sqrt(max(inflated_radius * inflated_radius - lateral_sq, 0.0)))
        entry_dist = max(entry_dist, 0.0)
        if entry_dist < 0.5:
            continue

        nominal = entry_dist / max(speed, 0.35)
        early = _arrival_time_constant_accel(entry_dist, speed, max_accel)
        late = _arrival_time_constant_accel(entry_dist, speed, min_accel)
        if not np.isfinite(early):
            early = nominal
        if not np.isfinite(late):
            late = nominal + max_spread
        early = max(0.25, min(float(early), nominal))
        late = min(horizon_s + 2.0, max(float(late), nominal))
        early = max(nominal - max_spread, early)
        late = min(nominal + max_spread, late)
        if 0.35 <= nominal <= horizon_s + 2.0:
            rows.append((nominal, entry_dist, int(j), float(early), float(late)))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return [(j, early, nominal, late) for nominal, _, j, early, late in rows[:max_agents]]


def _causal_priority_relation(scene: ScenarioData, agent_index: int, ego_keep: np.ndarray, cfg: dict) -> PriorityRelation:
    """Priority estimate for proposal allocation using current state only.

    ``determine_priority`` also supports trajectory arrival order.  Passing a
    constant-velocity extrapolation rather than logged future keeps proposal
    generation causal while retaining map-control/lane-ownership rules.
    """
    cur = int(scene.current_time_index)
    H = int(cfg.get("time", {}).get("future_steps", 80))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    st = np.asarray(scene.states[int(agent_index), cur], dtype=np.float32)
    agent_cv = constant_accel_trajectory(st, H, dt, accel=0.0)
    return determine_priority(scene, int(agent_index), ego_keep, agent_cv, cfg)


def generate_ego_candidates(scene: ScenarioData, cfg: dict, conflict_regions: list | None = None) -> dict[str, np.ndarray]:
    limits = cfg.get("limits", {})
    cand_cfg = cfg.get("candidate", {})
    horizon = int(cfg.get("time", {}).get("future_steps", 80))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    K = int(limits.get("max_candidates", 64))
    cur = scene.current_time_index
    ego_cur = scene.states[scene.sdc_track_index, cur]
    lane_points = _local_lane_point_cloud(scene, ego_cur[:2], cfg)
    logged_full = future_states_to_traj7(scene.states[scene.sdc_track_index, cur + 1 : cur + 1 + horizon, :], horizon, current_state=ego_cur)
    candidates: list[np.ndarray] = []
    macro: list[int] = []
    utility: list[float] = []
    is_logged: list[bool] = []
    is_neutral: list[bool] = []
    proposal_source: list[int] = []
    proposal_region_id: list[int] = []
    proposal_target_time_s: list[float] = []
    proposal_timing_side: list[int] = []
    proposal_target_agent_index: list[int] = []
    proposal_gap_s: list[float] = []
    proposal_accel_mps2: list[float] = []
    proposal_entry_distance_m: list[float] = []
    proposal_target_tta_error_s: list[float] = []
    rejection_counts: Counter[str] = Counter()
    attempted_by_source: Counter[str] = Counter()
    accepted_by_source: Counter[str] = Counter()

    def _near_duplicate(traj: np.ndarray, m: MacroType) -> bool:
        """Avoid wasting candidate slots on nearly identical keep-lane endings."""
        tol_m = float(cand_cfg.get("dedup_endpoint_tolerance_m", 0.35))
        tol_v = float(cand_cfg.get("dedup_speed_tolerance_mps", 0.25))
        if tol_m <= 0.0:
            return False
        # Preserve timing diversity for lane changes, stops and special macros by
        # default.  The duplicate issue is mainly the terminal keep-lane lattice.
        if not bool(cand_cfg.get("dedup_all_macro_types", False)) and m != MacroType.KEEP_LANE:
            return False
        end = traj[-1, :2]
        v_end = float(np.linalg.norm(traj[-1, 3:5]))
        for old_traj, old_macro in zip(candidates, macro):
            if int(old_macro) != int(m):
                continue
            if float(np.linalg.norm(old_traj[-1, :2] - end)) <= tol_m:
                old_v = float(np.linalg.norm(old_traj[-1, 3:5]))
                if abs(old_v - v_end) <= tol_v:
                    return True
        return False

    def add(
        traj: np.ndarray,
        m: MacroType,
        util: float = 0.0,
        logged: bool = False,
        neutral: bool = False,
        *,
        source: ProposalSource = ProposalSource.KEEP,
        region_id: int = -1,
        target_time_s: float = -1.0,
        timing_side: int = 0,
        target_agent_index: int = -1,
        gap_s: float = -1.0,
        accel_mps2: float = -99.0,
        entry_distance_m: float = -1.0,
        target_tta_error_s: float = -1.0,
    ) -> bool:
        attempted_by_source[source.name] += 1
        if len(candidates) >= K:
            rejection_counts["capacity"] += 1
            return False
        traj = traj[:horizon].astype(np.float32)
        # Logged replay is retained as an observed reference even if map geometry
        # is incomplete; all synthetic proposals are map-screened.
        proposal_lane_points = None if logged else lane_points
        reason = _candidate_invalid_reason(traj, cfg, proposal_lane_points)
        if reason is not None:
            rejection_counts[reason] += 1
            return False
        if _near_duplicate(traj, m):
            rejection_counts["near_duplicate"] += 1
            return False
        if len(traj) == horizon:
            candidates.append(traj)
            accepted_by_source[source.name] += 1
            macro.append(int(m))
            speed = np.linalg.norm(traj[:, 3:5], axis=-1)
            progress = float(np.linalg.norm(traj[-1, :2] - traj[0, :2]))
            comfort = float(np.mean(np.abs(np.diff(speed, prepend=speed[0]) / max(dt, 1e-3))))
            utility.append(float(util - 0.05 * progress + 0.2 * comfort))
            is_logged.append(logged)
            is_neutral.append(neutral)
            proposal_source.append(int(source))
            proposal_region_id.append(int(region_id))
            proposal_target_time_s.append(float(target_time_s))
            proposal_timing_side.append(int(timing_side))
            proposal_target_agent_index.append(int(target_agent_index))
            proposal_gap_s.append(float(gap_s))
            proposal_accel_mps2.append(float(accel_mps2))
            proposal_entry_distance_m.append(float(entry_distance_m))
            proposal_target_tta_error_s.append(float(target_tta_error_s))
            return True
        return False

    add(logged_full, MacroType.LOGGED_EGO, util=-1.0, logged=True, source=ProposalSource.LOGGED)
    add(
        constant_accel_trajectory(ego_cur, horizon, dt, accel=0.0),
        MacroType.KEEP_LANE,
        util=0.0,
        source=ProposalSource.KEEP,
        accel_mps2=0.0,
    )
    for acc in cand_cfg.get("accelerate_values_mps2", [0.5, 1.0, 1.5]):
        add(
            constant_accel_trajectory(ego_cur, horizon, dt, accel=float(acc)),
            MacroType.ACCELERATE_CROSS,
            util=0.2,
            source=ProposalSource.ACCELERATE,
            accel_mps2=float(acc),
        )
    for decel in cand_cfg.get("yield_decel_values_mps2", [-1.0, -2.0, -3.0]):
        add(
            constant_accel_trajectory(ego_cur, horizon, dt, accel=float(decel)),
            MacroType.YIELD,
            util=0.5,
            neutral=float(decel) in (-2.0, -3.0),
            source=ProposalSource.YIELD,
            accel_mps2=float(decel),
        )
        add(
            smooth_stop_trajectory(ego_cur, horizon, dt, decel=float(decel), creep_speed=0.0),
            MacroType.DECELERATE_CROSS,
            util=0.6,
            source=ProposalSource.YIELD,
            accel_mps2=float(decel),
        )
    # The neutral root is a core COWP mechanism and must never be displaced by
    # optional interaction-proposal families when the K-slot bank is saturated.
    add(
        smooth_stop_trajectory(ego_cur, horizon, dt, decel=-2.0, creep_speed=0.0),
        MacroType.NEUTRAL_EGO,
        util=0.8,
        neutral=True,
        source=ProposalSource.NEUTRAL,
        accel_mps2=-2.0,
    )
    # v16.8.8 base-bank preservation: semantically distinct lane-change/creep
    # actions are added before optional RMR/priority refinements.  Optional
    # proposal families may consume terminal filler slots, but must not silently
    # delete these core actions when K is saturated.
    for lateral, m in [(-3.5, MacroType.LANE_CHANGE_RIGHT), (3.5, MacroType.LANE_CHANGE_LEFT)]:
        for delay in cand_cfg.get("lane_change_start_delay_s", [0.0, 0.5, 1.0, 1.5]):
            tr = constant_accel_trajectory(ego_cur, horizon, dt, accel=0.0, lateral_offset=lateral, start_delay_s=float(delay))
            add(tr, m, util=0.3, source=ProposalSource.LANE_CHANGE)
    add(
        smooth_stop_trajectory(ego_cur, horizon, dt, decel=-1.0, creep_speed=1.0),
        MacroType.CREEP,
        util=1.0,
        source=ProposalSource.CREEP,
        accel_mps2=-1.0,
    )

    regions = conflict_regions if conflict_regions is not None else build_scene_conflict_regions(scene, cfg)
    if regions:
        keep = constant_accel_trajectory(ego_cur, horizon, dt, accel=0.0)
        ego_speed = float(max(ego_cur[5], np.linalg.norm(ego_cur[3:5]), 0.0))
        horizon_s = float(horizon * dt)
        # v16.8.4 boundary-consistent reachability.  Use the same inflated conflict
        # boundary as the interaction certificate and measure progress along the
        # actual primitive, rather than solving timing to the region centre.
        reachable: list[tuple[float, float, ConflictRegion]] = []
        for region in regions:
            entry_tta, entry_dist = trajectory_entry_to_region(keep, region, current_state=ego_cur, dt=dt)
            if np.isfinite(entry_tta) and np.isfinite(entry_dist) and entry_tta >= 0.0 and entry_dist > 0.5:
                reachable.append((float(entry_tta), float(entry_dist), region))
        reachable.sort(key=lambda row: (row[0], row[1], int(row[2].conflict_id)))

        # Do not synthesize conflict-timing actions for a map conflict that the
        # nominal primitive never reaches.  The old nearest-region fallback could
        # stop for a region behind the ego or advertise timing against a region the
        # generated trajectory never entered.
        if reachable:
            primary_tta, primary_dist, primary = reachable[0]
            for margin in cand_cfg.get("stop_margin_to_conflict_m", [2.0, 5.0, 8.0]):
                stop_after = max(0.0, primary_dist - float(margin))
                add(
                    smooth_stop_trajectory(ego_cur, horizon, dt, decel=-2.5, stop_after_m=stop_after),
                    MacroType.STOP_BEFORE_CONFLICT,
                    util=0.7,
                    neutral=True,
                    source=ProposalSource.STOP,
                    region_id=int(primary.conflict_id),
                    accel_mps2=-2.5,
                    entry_distance_m=primary_dist,
                )

            # Keep the legacy constant-acceleration lattice as an explicit ablation,
            # but make even that baseline solve to conflict *entry* rather than the
            # region centre.
            for offset in cand_cfg.get("merge_time_offsets_s", [-1.5, -0.8, 0.0, 0.8, 1.5]):
                target_time = max(0.45, float(primary_tta) + float(offset))
                target_acc = _constant_accel_for_arrival(primary_dist, ego_speed, target_time, cfg)
                if target_acc is not None:
                    legacy_traj = constant_accel_trajectory(ego_cur, horizon, dt, accel=target_acc)
                    actual_tta, _ = trajectory_entry_to_region(legacy_traj, primary, current_state=ego_cur, dt=dt)
                    tta_error = abs(float(actual_tta) - target_time) if np.isfinite(actual_tta) else -1.0
                    add(
                        legacy_traj,
                        MacroType.MERGE_AHEAD if offset <= 0 else MacroType.MERGE_BEHIND,
                        util=0.1 + max(0.0, float(offset)),
                        neutral=bool(offset > 0),
                        source=ProposalSource.LEGACY_TIMING,
                        region_id=int(primary.conflict_id),
                        target_time_s=target_time,
                        timing_side=-1 if offset <= 0 else 1,
                        gap_s=abs(float(offset)),
                        accel_mps2=target_acc,
                        entry_distance_m=primary_dist,
                        target_tta_error_s=tta_error,
                    )

            # v16.8.4 Boundary-Consistent Smooth RMR-BCTE (BCS-RMR-BCTE).
            # For each reachable conflict region, pair a causal agent entry-time
            # envelope with an ego cubic time projection that reaches the *entry
            # boundary* at the requested pass-before/pass-after time.  This fixes
            # the central semantic bug in v16.8.3 where a trajectory could be
            # tagged pass-after while entering the conflict seconds earlier.
            max_regions = max(1, int(cand_cfg.get("timing_envelope_max_regions", 3)))
            max_bcte = max(0, int(cand_cfg.get("timing_envelope_max_candidates", 24)))
            accel_dedup = max(float(cand_cfg.get("timing_envelope_accel_dedup_mps2", 0.10)), 1.0e-3)
            gap_values = [float(x) for x in cand_cfg.get("timing_envelope_gap_s", [0.8, 1.4, 2.0])]
            max_tta_error = float(cand_cfg.get("timing_envelope_max_target_tta_error_s", max(0.20, 2.0 * dt)))
            timing_profile_bins: set[tuple[int, int, int, int]] = set()
            bcte_added = 0
            for _, dist_to_conflict, region in reachable[:max_regions]:
                if bcte_added >= max_bcte or len(candidates) >= K:
                    break
                envelopes = _agent_tta_envelopes_to_region(scene, region, cfg)
                for agent_index, early_tta, _nominal_tta, late_tta in envelopes:
                    if bcte_added >= max_bcte or len(candidates) >= K:
                        break
                    for gap in gap_values:
                        if bcte_added >= max_bcte or len(candidates) >= K:
                            break
                        for side, agent_boundary in ((-1, early_tta), (1, late_tta)):
                            target_time = float(agent_boundary + side * gap)
                            # An explicit timing candidate must realize its event
                            # inside the planning horizon; long-horizon waiting is
                            # already represented by yield/stop/creep primitives.
                            if target_time <= 0.35 or target_time > horizon_s:
                                continue
                            profile = _smooth_arrival_profile(dist_to_conflict, ego_speed, target_time, cfg)
                            if profile is None:
                                continue
                            initial_accel, _profile_jerk, _terminal_speed = profile
                            accel_bin = int(round(float(initial_accel) / accel_dedup))
                            profile_key = (int(region.conflict_id), int(agent_index), int(side), accel_bin)
                            if profile_key in timing_profile_bins:
                                continue
                            timed_traj = smooth_arrival_trajectory(
                                ego_cur, horizon, dt, distance_m=dist_to_conflict, target_time_s=target_time
                            )
                            if timed_traj is None:
                                continue
                            actual_tta, actual_entry_dist = trajectory_entry_to_region(
                                timed_traj, region, current_state=ego_cur, dt=dt
                            )
                            if not np.isfinite(actual_tta):
                                continue
                            tta_error = abs(float(actual_tta) - target_time)
                            if tta_error > max_tta_error + 1.0e-6:
                                continue
                            behind = side > 0
                            accepted = add(
                                timed_traj,
                                MacroType.MERGE_BEHIND if behind else MacroType.MERGE_AHEAD,
                                util=(0.15 + 0.08 * gap if behind else 0.05 + 0.03 * gap),
                                neutral=behind,
                                source=ProposalSource.ROBUST_BCTE,
                                region_id=int(region.conflict_id),
                                target_time_s=target_time,
                                timing_side=side,
                                target_agent_index=int(agent_index),
                                gap_s=gap,
                                accel_mps2=initial_accel,
                                entry_distance_m=float(actual_entry_dist),
                                target_tta_error_s=tta_error,
                            )
                            if accepted:
                                timing_profile_bins.add(profile_key)
                                bcte_added += 1

            # v16.8.6 Priority-Commitment Hold-Release (PCHR) refinement.
            # Arrival order alone is not sufficient for non-coercion: an ego that
            # approaches assertively and brakes late can make a protected agent
            # react even if ego ultimately enters after it.  For causally inferred
            # AGENT_PRIORITY / negotiated interactions, explicitly commit to a
            # pre-conflict hold and release only after the agent's late envelope.
            if bool(cand_cfg.get("priority_hold_release_enabled", True)):
                phr_max = max(0, int(cand_cfg.get("priority_hold_release_max_candidates", 8)))
                phr_added = 0
                phr_margins = [float(x) for x in cand_cfg.get("priority_hold_release_stop_margin_m", [3.0, 5.0])]
                phr_release_speeds = [float(x) for x in cand_cfg.get("priority_hold_release_speed_mps", [2.5, 3.5, 4.0])]
                phr_min_hold_s = float(cand_cfg.get("priority_hold_release_min_hold_s", 0.35))
                phr_gap_values = [float(x) for x in cand_cfg.get("priority_hold_release_gap_s", [0.8, 1.4, 2.0])]
                protected_rel = {int(PriorityRelation.AGENT_PRIORITY), int(PriorityRelation.EQUAL_OR_NEGOTIATED)}
                phr_keys: set[tuple[int, int, int, int]] = set()
                for _, dist_to_conflict, region in reachable[:max_regions]:
                    if phr_added >= phr_max or len(candidates) >= K:
                        break
                    for agent_index, _early_tta, _nominal_tta, late_tta in _agent_tta_envelopes_to_region(scene, region, cfg):
                        if phr_added >= phr_max or len(candidates) >= K:
                            break
                        rho = _causal_priority_relation(scene, int(agent_index), keep, cfg)
                        if int(rho) not in protected_rel:
                            continue
                        for gap in phr_gap_values:
                            target_time = float(late_tta + gap)
                            if target_time <= 0.35 or target_time > horizon_s:
                                continue
                            for margin in phr_margins:
                                for release_speed in phr_release_speeds:
                                    if phr_added >= phr_max or len(candidates) >= K:
                                        break
                                    key = (
                                        int(region.conflict_id), int(agent_index),
                                        int(round(margin * 10.0)), int(round(target_time * 10.0)),
                                    )
                                    if key in phr_keys:
                                        continue
                                    tr = priority_hold_release_trajectory(
                                        ego_cur, horizon, dt,
                                        entry_distance_m=dist_to_conflict,
                                        target_time_s=target_time,
                                        stop_margin_m=margin,
                                        release_speed_mps=release_speed,
                                        min_hold_s=phr_min_hold_s,
                                    )
                                    if tr is None:
                                        continue
                                    actual_tta, actual_entry_dist = trajectory_entry_to_region(
                                        tr, region, current_state=ego_cur, dt=dt
                                    )
                                    if not np.isfinite(actual_tta):
                                        continue
                                    tta_error = abs(float(actual_tta) - target_time)
                                    if tta_error > max_tta_error + 1.0e-6:
                                        continue
                                    accepted = add(
                                        tr,
                                        MacroType.MERGE_BEHIND,
                                        util=0.20 + 0.06 * gap,
                                        neutral=True,
                                        source=ProposalSource.PRIORITY_HOLD_RELEASE,
                                        region_id=int(region.conflict_id),
                                        target_time_s=target_time,
                                        timing_side=1,
                                        target_agent_index=int(agent_index),
                                        gap_s=gap,
                                        accel_mps2=0.0,
                                        entry_distance_m=float(actual_entry_dist),
                                        target_tta_error_s=tta_error,
                                    )
                                    if accepted:
                                        phr_keys.add(key)
                                        phr_added += 1


            # v16.8.8 Priority-Smooth-Yield (PSY).  The 191-scene PCHR probe
            # showed that mandatory stop-hold-release is physically feasible in
            # too few ordinary interaction windows (17 candidates / 128 random
            # scenes, zero NCF).  PSY instead reaches the protected agent's late
            # pass-after boundary at a deliberately low entry speed, and requires
            # an observable early speed drop.  It preserves the causal protected
            # relation and common validator, but removes the full-stop bottleneck.
            if bool(cand_cfg.get("priority_smooth_yield_enabled", True)):
                psy_max = max(0, int(cand_cfg.get("priority_smooth_yield_max_candidates", 8)))
                psy_added = 0
                psy_vt = [float(x) for x in cand_cfg.get("priority_smooth_yield_terminal_speed_mps", [1.0, 2.0, 3.0])]
                psy_a0 = [float(x) for x in cand_cfg.get("priority_smooth_yield_initial_decel_mps2", [-0.8, -1.4, -2.0])]
                psy_gaps = [float(x) for x in cand_cfg.get("priority_smooth_yield_gap_s", [0.8, 1.4, 2.0])]
                commit_t = max(float(cand_cfg.get("priority_smooth_yield_commitment_check_s", 1.0)), dt)
                min_drop = max(0.0, float(cand_cfg.get("priority_smooth_yield_min_speed_drop_mps", 0.75)))
                protected_rel = {int(PriorityRelation.AGENT_PRIORITY), int(PriorityRelation.EQUAL_OR_NEGOTIATED)}
                psy_keys: set[tuple[int, int, int, int]] = set()
                for _, dist_to_conflict, region in reachable[:max_regions]:
                    if psy_added >= psy_max or len(candidates) >= K:
                        break
                    for agent_index, _early_tta, _nominal_tta, late_tta in _agent_tta_envelopes_to_region(scene, region, cfg):
                        if psy_added >= psy_max or len(candidates) >= K:
                            break
                        rho = _causal_priority_relation(scene, int(agent_index), keep, cfg)
                        if int(rho) not in protected_rel:
                            continue
                        for gap in psy_gaps:
                            target_time = float(late_tta + gap)
                            if target_time <= max(0.35, commit_t) or target_time > horizon_s:
                                continue
                            for initial_accel in psy_a0:
                                for terminal_speed in psy_vt:
                                    if psy_added >= psy_max or len(candidates) >= K:
                                        break
                                    vt = float(max(terminal_speed, 0.0))
                                    key = (
                                        int(region.conflict_id), int(agent_index),
                                        int(round(target_time * 10.0)),
                                        int(round(vt * 10.0)) * 100 + int(round(abs(initial_accel) * 10.0)),
                                    )
                                    if key in psy_keys:
                                        continue
                                    tr = smooth_terminal_speed_arrival_trajectory(
                                        ego_cur, horizon, dt, distance_m=dist_to_conflict,
                                        target_time_s=target_time, terminal_speed_mps=vt,
                                        initial_accel_mps2=initial_accel,
                                    )
                                    if tr is None:
                                        continue
                                    check_idx = min(max(int(round(commit_t / dt)) - 1, 0), horizon - 1)
                                    check_speed = float(np.linalg.norm(tr[check_idx, 3:5]))
                                    if ego_speed > 1.5 and check_speed > max(ego_speed - min_drop, 0.5) + 1.0e-6:
                                        continue
                                    actual_tta, actual_entry_dist = trajectory_entry_to_region(
                                        tr, region, current_state=ego_cur, dt=dt
                                    )
                                    if not np.isfinite(actual_tta):
                                        continue
                                    tta_error = abs(float(actual_tta) - target_time)
                                    if tta_error > max_tta_error + 1.0e-6:
                                        continue
                                    accepted = add(
                                        tr,
                                        MacroType.MERGE_BEHIND,
                                        util=0.16 + 0.05 * gap,
                                        neutral=True,
                                        source=ProposalSource.PRIORITY_SMOOTH_YIELD,
                                        region_id=int(region.conflict_id),
                                        target_time_s=target_time,
                                        timing_side=1,
                                        target_agent_index=int(agent_index),
                                        gap_s=gap,
                                        accel_mps2=initial_accel,
                                        entry_distance_m=float(actual_entry_dist),
                                        target_tta_error_s=tta_error,
                                    )
                                    if accepted:
                                        psy_keys.add(key)
                                        psy_added += 1
    # v16.8.19 Route-Topology NCF Coverage (RT-NCF).  The legacy interaction
    # timing bank is longitudinal in the current yaw; on curved approaches this
    # can be map-valid only by accident and can miss a protected pass-after plan.
    # Add a small *causal map-topology* bank before terminal fillers.  It uses only
    # current state + vector-map connectivity, never logged future timing/geometry.
    if bool(cand_cfg.get("route_topology_ncf_enabled", True)) and len(candidates) < K:
        nat_cfg = cfg.get("natural", {})
        route_max = max(1, int(cand_cfg.get("route_topology_max_routes", 2)))
        route_max_candidates = max(0, int(cand_cfg.get("route_topology_max_candidates", 10)))
        route_added = 0
        route_accels = [float(x) for x in cand_cfg.get("route_topology_base_accel_mps2", [0.0, -1.0])]
        max_pos_acc = max([0.0] + [a for a in route_accels if a > 0.0])
        required = _required_route_length(ego_cur, horizon, dt, nat_cfg, accel=max_pos_acc)
        route_paths = _map_route_polylines(scene, ego_cur, required, nat_cfg)[:route_max]
        route_attach_max = float(cand_cfg.get("route_topology_attach_max_m", nat_cfg.get("map_route_search_radius_m", 8.0)))

        # First add a minimal route-following base pair.  This repairs curved-road
        # candidate support without changing the fixed candidate cardinality.
        for ridx, route_path in enumerate(route_paths):
            for acc in route_accels:
                if route_added >= route_max_candidates or len(candidates) >= K:
                    break
                tr = _timed_polyline_trajectory(
                    route_path, ego_cur, horizon, dt, accel=acc, nat_cfg=nat_cfg
                )
                if tr is None:
                    continue
                accepted = add(
                    tr, MacroType.KEEP_LANE if acc >= -0.25 else MacroType.YIELD,
                    util=0.02 + 0.03 * max(0.0, -acc), neutral=bool(acc < -0.25),
                    source=ProposalSource.KEEP if acc >= -0.25 else ProposalSource.YIELD,
                    accel_mps2=acc,
                )
                route_added += int(bool(accepted))

        # Group protected pass-after: one ego timing must be compatible with all
        # protected agents sharing the same conflict region, not just a pairwise
        # target.  This directly addresses scenes where every pairwise candidate
        # leaves another protected root in the burden tail.
        route_group_gaps = [float(x) for x in cand_cfg.get("route_group_yield_gap_s", [0.8, 1.4, 2.0])]
        route_group_vt = [float(x) for x in cand_cfg.get("route_group_yield_terminal_speed_mps", [1.0, 2.0])]
        route_group_a0 = [float(x) for x in cand_cfg.get("route_group_yield_initial_decel_mps2", [-0.8, -1.4])]
        group_max = max(0, int(cand_cfg.get("route_group_yield_max_candidates", 8)))
        group_added = 0
        protected_rel = {int(PriorityRelation.AGENT_PRIORITY), int(PriorityRelation.EQUAL_OR_NEGOTIATED)}
        if regions and route_paths and group_max > 0:
            route_reachable: list[tuple[float, float, int, ConflictRegion, np.ndarray, np.ndarray]] = []
            for ridx, route_path in enumerate(route_paths):
                route_keep = _timed_polyline_trajectory(route_path, ego_cur, horizon, dt, accel=0.0, nat_cfg=nat_cfg)
                if route_keep is None:
                    continue
                for region in regions:
                    tta, dist = trajectory_entry_to_region(route_keep, region, current_state=ego_cur, dt=dt)
                    if np.isfinite(tta) and np.isfinite(dist) and dist > 0.5:
                        route_reachable.append((float(tta), float(dist), int(ridx), region, route_path, route_keep))
            route_reachable.sort(key=lambda row: (row[0], row[1], row[2], int(row[3].conflict_id)))
            seen_group: set[tuple[int, int, int, int]] = set()
            max_regions = max(1, int(cand_cfg.get("timing_envelope_max_regions", 3)))
            seen_regions: set[tuple[int, int]] = set()
            for _tta, dist_to_conflict, ridx, region, route_path, route_keep in route_reachable:
                if group_added >= group_max or route_added >= route_max_candidates or len(candidates) >= K:
                    break
                region_route_key = (int(region.conflict_id), int(ridx))
                if region_route_key in seen_regions:
                    continue
                if len(seen_regions) >= max_regions * max(1, len(route_paths)):
                    break
                seen_regions.add(region_route_key)
                protected_rows: list[tuple[int, float]] = []
                for agent_index, _early, _nominal, late in _agent_tta_envelopes_to_region(scene, region, cfg):
                    rho = _causal_priority_relation(scene, int(agent_index), route_keep, cfg)
                    if int(rho) in protected_rel:
                        protected_rows.append((int(agent_index), float(late)))
                if not protected_rows:
                    continue
                binding_agent, binding_late = max(protected_rows, key=lambda z: (z[1], z[0]))
                for gap in route_group_gaps:
                    target_time = float(binding_late + gap)
                    if target_time <= 0.35 or target_time > horizon_s:
                        continue
                    for initial_accel in route_group_a0:
                        for terminal_speed in route_group_vt:
                            if group_added >= group_max or route_added >= route_max_candidates or len(candidates) >= K:
                                break
                            key = (int(region.conflict_id), int(ridx), int(round(target_time * 10.0)), int(round(terminal_speed * 10.0)))
                            if key in seen_group:
                                continue
                            base = smooth_terminal_speed_arrival_trajectory(
                                ego_cur, horizon, dt, distance_m=dist_to_conflict,
                                target_time_s=target_time, terminal_speed_mps=max(terminal_speed, 0.0),
                                initial_accel_mps2=initial_accel,
                            )
                            if base is None:
                                continue
                            tr = _project_progress_profile_to_route(
                                base, route_path, ego_cur, dt, attach_max_m=route_attach_max
                            )
                            if tr is None:
                                continue
                            actual_tta, actual_entry_dist = trajectory_entry_to_region(
                                tr, region, current_state=ego_cur, dt=dt
                            )
                            if not np.isfinite(actual_tta):
                                continue
                            tta_error = abs(float(actual_tta) - target_time)
                            max_tta_error = float(cand_cfg.get("timing_envelope_max_target_tta_error_s", max(0.20, 2.0 * dt)))
                            if tta_error > max_tta_error + 1.0e-6:
                                continue
                            accepted = add(
                                tr, MacroType.MERGE_BEHIND, util=0.12 + 0.05 * gap, neutral=True,
                                source=ProposalSource.PRIORITY_SMOOTH_YIELD,
                                region_id=int(region.conflict_id), target_time_s=target_time, timing_side=1,
                                target_agent_index=int(binding_agent), gap_s=gap, accel_mps2=initial_accel,
                                entry_distance_m=float(actual_entry_dist), target_tta_error_s=tta_error,
                            )
                            if accepted:
                                seen_group.add(key)
                                group_added += 1
                                route_added += 1

    # v16.8.20 Joint Route NCF (JR-NCF).  A scene-level protected certificate
    # is universal over protected critical pairs, so a proposal that is timed
    # behind one conflict region can still coerce an actor at the next region.
    # Construct a causal longitudinal schedule on each map-topology route and
    # validate it simultaneously against several current-state TTA envelopes.
    if bool(cand_cfg.get("route_joint_yield_enabled", True)) and regions and len(candidates) < K:
        nat_cfg = cfg.get("natural", {})
        joint_route_max = max(1, int(cand_cfg.get("route_topology_max_routes", 2)))
        max_pos_acc = max(0.0, float(cand_cfg.get("route_joint_yield_max_accel_mps2", 0.5)))
        required = _required_route_length(ego_cur, horizon, dt, nat_cfg, accel=max_pos_acc)
        joint_routes = _map_route_polylines(scene, ego_cur, required, nat_cfg)[:joint_route_max]
        joint_max = max(0, int(cand_cfg.get("route_joint_yield_max_candidates", 8)))
        joint_max_regions = max(1, int(cand_cfg.get("route_joint_yield_max_regions", 4)))
        joint_gaps = [float(x) for x in cand_cfg.get("route_joint_yield_gap_s", [0.8, 1.4])]
        joint_slack = [float(x) for x in cand_cfg.get("route_joint_yield_accel_slack_mps2", [0.0, -0.4, -0.8])]
        joint_min_acc = float(cand_cfg.get("route_joint_yield_min_accel_mps2", -3.5))
        joint_max_acc = float(cand_cfg.get("route_joint_yield_max_accel_mps2", 0.5))
        joint_tol = max(float(cand_cfg.get("route_joint_yield_tta_tolerance_s", max(0.20, 2.0 * dt))), dt)
        protected_rel = {int(PriorityRelation.AGENT_PRIORITY), int(PriorityRelation.EQUAL_OR_NEGOTIATED)}
        joint_added = 0
        joint_keys: set[tuple[int, int]] = set()
        v0 = max(float(ego_cur[5] if len(ego_cur) > 5 else np.linalg.norm(ego_cur[3:5])), 0.0)

        for ridx, route_path in enumerate(joint_routes):
            if joint_added >= joint_max or len(candidates) >= K:
                break
            route_keep = _timed_polyline_trajectory(route_path, ego_cur, horizon, dt, accel=0.0, nat_cfg=nat_cfg)
            if route_keep is None:
                continue
            constraints: list[tuple[float, float, ConflictRegion, int, float]] = []
            for region in regions:
                ego_tta, dist = trajectory_entry_to_region(route_keep, region, current_state=ego_cur, dt=dt)
                if not (np.isfinite(ego_tta) and np.isfinite(dist) and dist > 0.5):
                    continue
                protected_rows: list[tuple[int, float]] = []
                for agent_index, _early, _nominal, late in _agent_tta_envelopes_to_region(scene, region, cfg):
                    rho = _causal_priority_relation(scene, int(agent_index), route_keep, cfg)
                    if int(rho) in protected_rel:
                        protected_rows.append((int(agent_index), float(late)))
                if not protected_rows:
                    continue
                binding_agent, binding_late = max(protected_rows, key=lambda z: (z[1], z[0]))
                constraints.append((float(ego_tta), float(dist), region, int(binding_agent), float(binding_late)))
            constraints.sort(key=lambda row: (row[0], row[1], int(row[2].conflict_id)))
            constraints = constraints[:joint_max_regions]
            if not constraints:
                continue

            for gap in joint_gaps:
                if joint_added >= joint_max or len(candidates) >= K:
                    break
                active: list[tuple[float, float, ConflictRegion, int, float]] = []
                accel_bounds: list[tuple[float, ConflictRegion, int, float, float]] = []
                for ego_tta, dist, region, agent_index, late in constraints:
                    target = float(late + gap)
                    if target <= 0.35:
                        continue
                    # target may be slightly beyond the rollout horizon: stopping
                    # before that region is a valid low-pressure pass-after action.
                    if target > horizon_s + 2.0:
                        continue
                    active.append((ego_tta, dist, region, agent_index, target))
                    # Constant-acceleration analytic bound: arrival no earlier than
                    # target requires a <= 2*(d-v0*t)/t^2.  The route retimer uses
                    # a ramped acceleration schedule, so every proposal is checked
                    # geometrically below rather than trusting this approximation.
                    a_req = 2.0 * (float(dist) - v0 * target) / max(target * target, 1.0e-6)
                    accel_bounds.append((float(a_req), region, int(agent_index), float(dist), target))
                if not active:
                    continue
                binding = min(accel_bounds, key=lambda row: (row[0], int(row[1].conflict_id)))
                base_acc = float(np.clip(binding[0], joint_min_acc, joint_max_acc))
                for slack in joint_slack:
                    if joint_added >= joint_max or len(candidates) >= K:
                        break
                    accel = float(np.clip(base_acc + min(float(slack), 0.0), joint_min_acc, joint_max_acc))
                    key = (int(ridx), int(round(accel * 100.0)))
                    if key in joint_keys:
                        continue
                    tr = _timed_polyline_trajectory(route_path, ego_cur, horizon, dt, accel=accel, nat_cfg=nat_cfg)
                    if tr is None:
                        continue
                    # Hard joint validation.  Infinite TTA means ego stays before
                    # the corresponding conflict region for this horizon, which is
                    # conservatively compatible with a protected pass-after bound.
                    violations: list[float] = []
                    finite_errors: list[float] = []
                    for _ego_tta, _dist, region, _agent_index, target in active:
                        actual_tta, _actual_dist = trajectory_entry_to_region(tr, region, current_state=ego_cur, dt=dt)
                        if np.isfinite(actual_tta):
                            violations.append(max(0.0, float(target - actual_tta)))
                            finite_errors.append(abs(float(actual_tta - target)))
                        else:
                            violations.append(0.0)
                    if violations and max(violations) > joint_tol + 1.0e-6:
                        continue
                    binding_region = binding[1]
                    binding_agent = int(binding[2])
                    binding_target = float(binding[4])
                    binding_entry = trajectory_entry_to_region(tr, binding_region, current_state=ego_cur, dt=dt)[1]
                    accepted = add(
                        tr,
                        MacroType.MERGE_BEHIND,
                        util=0.10 + 0.04 * gap + 0.02 * max(0.0, -accel),
                        neutral=True,
                        source=ProposalSource.JOINT_ROUTE_NCF,
                        region_id=int(binding_region.conflict_id),
                        target_time_s=binding_target,
                        timing_side=1,
                        target_agent_index=binding_agent,
                        gap_s=gap,
                        accel_mps2=accel,
                        entry_distance_m=float(binding_entry) if np.isfinite(binding_entry) else float(binding[3]),
                        target_tta_error_s=float(max(violations) if violations else 0.0),
                    )
                    if accepted:
                        joint_keys.add(key)
                        joint_added += 1

    # v16.8.25 Multi-Conflict Feasibility Corridor (MCFC).  JR-NCF above is
    # intentionally low-dimensional: a single acceleration must satisfy every
    # protected conflict and therefore tends to remain slow long after the
    # binding interaction.  MCFC keeps the same causal/map-only contract but
    # fits a piecewise zero-endpoint-acceleration progress schedule through the
    # protected pass-after corridor, then explicitly recovers speed when route
    # length and horizon permit.  It is a proposal-support repair, not a second
    # safety classifier; the downstream conventional/BCOT certificates remain
    # authoritative.
    if bool(cand_cfg.get("multi_conflict_corridor_enabled", False)) and regions and len(candidates) < K:
        nat_cfg = cfg.get("natural", {})
        corridor_route_max = max(1, int(cand_cfg.get("route_topology_max_routes", 2)))
        required = _required_route_length(
            ego_cur,
            horizon,
            dt,
            nat_cfg,
            accel=max(0.0, float(cand_cfg.get("multi_conflict_corridor_recovery_accel_mps2", [0.8])[-1])),
        )
        corridor_routes = _map_route_polylines(scene, ego_cur, required, nat_cfg)[:corridor_route_max]
        corridor_max = max(0, int(cand_cfg.get("multi_conflict_corridor_max_candidates", 8)))
        corridor_max_regions = max(1, int(cand_cfg.get("multi_conflict_corridor_max_regions", 4)))
        corridor_gaps = [float(x) for x in cand_cfg.get("multi_conflict_corridor_gap_s", [0.8, 1.4])]
        corridor_speed_scales = [float(x) for x in cand_cfg.get("multi_conflict_corridor_entry_speed_scale", [0.60, 0.85])]
        corridor_recovery_accels = [float(x) for x in cand_cfg.get("multi_conflict_corridor_recovery_accel_mps2", [0.8, 1.4])]
        corridor_recovery_window = max(float(cand_cfg.get("multi_conflict_corridor_recovery_window_s", 2.0)), 2.0 * dt)
        corridor_min_knot_dt = max(float(cand_cfg.get("multi_conflict_corridor_min_knot_dt_s", 0.6)), 2.0 * dt)
        corridor_tol = max(float(cand_cfg.get("multi_conflict_corridor_tta_tolerance_s", max(0.20, 2.0 * dt))), dt)
        corridor_commit_t = max(float(cand_cfg.get("multi_conflict_corridor_commitment_check_s", 1.0)), dt)
        corridor_min_drop = max(float(cand_cfg.get("multi_conflict_corridor_min_speed_drop_mps", 0.4)), 0.0)
        corridor_max_entry_speed = max(float(cand_cfg.get("multi_conflict_corridor_max_entry_speed_mps", 6.0)), 0.5)
        corridor_max_speed = max(float(cand_cfg.get("multi_conflict_corridor_max_recovery_speed_mps", 13.0)), corridor_max_entry_speed)
        protected_rel = {int(PriorityRelation.AGENT_PRIORITY), int(PriorityRelation.EQUAL_OR_NEGOTIATED)}
        corridor_added = 0
        corridor_keys: set[tuple[int, int, int, int]] = set()
        v0 = max(float(ego_cur[5] if len(ego_cur) > 5 else np.linalg.norm(ego_cur[3:5])), 0.0)

        for ridx, route_path in enumerate(corridor_routes):
            if corridor_added >= corridor_max or len(candidates) >= K:
                break
            route_keep = _timed_polyline_trajectory(route_path, ego_cur, horizon, dt, accel=0.0, nat_cfg=nat_cfg)
            if route_keep is None:
                continue
            constraints: list[tuple[float, float, ConflictRegion, int, float]] = []
            for region in regions:
                ego_tta, dist = trajectory_entry_to_region(route_keep, region, current_state=ego_cur, dt=dt)
                if not (np.isfinite(ego_tta) and np.isfinite(dist) and dist > 0.5):
                    continue
                protected_rows: list[tuple[int, float]] = []
                for agent_index, _early, _nominal, late in _agent_tta_envelopes_to_region(scene, region, cfg):
                    rho = _causal_priority_relation(scene, int(agent_index), route_keep, cfg)
                    if int(rho) in protected_rel:
                        protected_rows.append((int(agent_index), float(late)))
                if not protected_rows:
                    continue
                binding_agent, binding_late = max(protected_rows, key=lambda z: (z[1], z[0]))
                constraints.append((float(ego_tta), float(dist), region, int(binding_agent), float(binding_late)))
            # Progress, rather than current-time TTA, defines the knot order.
            constraints.sort(key=lambda row: (row[1], row[0], int(row[2].conflict_id)))
            dedup_constraints: list[tuple[float, float, ConflictRegion, int, float]] = []
            for row in constraints:
                if dedup_constraints and row[1] <= dedup_constraints[-1][1] + 0.25:
                    # Keep the stricter protected deadline at nearly the same
                    # route location, avoiding degenerate consecutive knots.
                    if row[4] > dedup_constraints[-1][4]:
                        dedup_constraints[-1] = row
                    continue
                dedup_constraints.append(row)
            constraints = dedup_constraints[:corridor_max_regions]
            if not constraints:
                continue

            # Route arc available to the projected progress profile.
            route_xy = np.asarray(route_path, dtype=np.float32)[:, :2]
            if len(route_xy) < 2:
                continue
            if float(np.linalg.norm(route_xy[0] - ego_cur[:2])) > 1.0e-3:
                route_xy = np.concatenate([np.asarray(ego_cur[:2], dtype=np.float32)[None, :], route_xy], axis=0)
            route_len = float(np.sum(np.linalg.norm(np.diff(route_xy, axis=0), axis=1)))

            for gap in corridor_gaps:
                if corridor_added >= corridor_max or len(candidates) >= K:
                    break
                active: list[tuple[float, float, ConflictRegion, int, float]] = []
                prev_target = 0.0
                prev_dist = 0.0
                for ego_tta, dist, region, agent_index, late in constraints:
                    raw_target = float(late + gap)
                    if raw_target <= 0.35 or raw_target > horizon_s + 2.0:
                        continue
                    # Later route conflicts cannot receive an earlier knot even
                    # if independent agent envelopes overlap in time.  A small
                    # distance-aware lower bound avoids impossible knot order.
                    min_travel = max(corridor_min_knot_dt, (float(dist) - prev_dist) / max(corridor_max_speed, 0.5))
                    target = max(raw_target, prev_target + min_travel)
                    active.append((ego_tta, float(dist), region, int(agent_index), float(target)))
                    prev_target, prev_dist = float(target), float(dist)
                if not active:
                    continue

                for speed_scale in corridor_speed_scales:
                    if corridor_added >= corridor_max or len(candidates) >= K:
                        break
                    base_times: list[float] = []
                    base_dist: list[float] = []
                    base_speeds: list[float] = []
                    last_t, last_d = 0.0, 0.0
                    last_v = v0
                    feasible_knots = True
                    for _ego_tta, dist, _region, _agent_index, target in active:
                        seg_t = max(float(target - last_t), dt)
                        seg_d = max(float(dist - last_d), 1.0e-3)
                        avg_v = seg_d / seg_t
                        # A low but nonzero conflict-entry speed avoids PCHR's
                        # full-stop feasibility collapse while preserving an
                        # observable yielding commitment.
                        entry_v = float(np.clip(
                            min(v0 * max(speed_scale, 0.05), 1.5 * avg_v),
                            0.5,
                            corridor_max_entry_speed,
                        ))
                        if not np.isfinite(entry_v):
                            feasible_knots = False
                            break
                        base_times.append(float(target))
                        base_dist.append(float(dist))
                        base_speeds.append(entry_v)
                        last_t, last_d, last_v = float(target), float(dist), entry_v
                    if not feasible_knots:
                        continue

                    for recovery_accel in corridor_recovery_accels:
                        if corridor_added >= corridor_max or len(candidates) >= K:
                            break
                        times = list(base_times)
                        distances = list(base_dist)
                        speeds = list(base_speeds)
                        recovery_used = False
                        if last_t < horizon_s - 2.0 * dt and route_len > last_d + 0.75:
                            rec_t = min(horizon_s, last_t + corridor_recovery_window)
                            rec_dt = max(rec_t - last_t, dt)
                            desired_v = float(np.clip(last_v + max(recovery_accel, 0.0) * rec_dt, last_v, min(max(v0, last_v), corridor_max_speed)))
                            desired_d = last_d + 0.5 * (last_v + desired_v) * rec_dt
                            rec_d = min(desired_d, route_len - 0.25)
                            if rec_d > last_d + 0.5:
                                # If map support truncates the recovery distance,
                                # lower the endpoint speed consistently instead
                                # of demanding an aggressive compressed segment.
                                mean_v = (rec_d - last_d) / rec_dt
                                rec_v = float(np.clip(2.0 * mean_v - last_v, last_v, desired_v))
                                times.append(float(rec_t))
                                distances.append(float(rec_d))
                                speeds.append(rec_v)
                                recovery_used = rec_v > last_v + 0.10

                        key = (
                            int(ridx), int(round(gap * 10.0)),
                            int(round(speed_scale * 100.0)), int(round(max(recovery_accel, 0.0) * 10.0)),
                        )
                        if key in corridor_keys:
                            continue
                        base = piecewise_quintic_progress_trajectory(
                            ego_cur,
                            horizon,
                            dt,
                            waypoint_times_s=times,
                            waypoint_distances_m=distances,
                            waypoint_speeds_mps=speeds,
                        )
                        if base is None:
                            continue
                        tr = _project_progress_profile_to_route(
                            base,
                            route_path,
                            ego_cur,
                            dt,
                            attach_max_m=float(cand_cfg.get("route_topology_attach_max_m", nat_cfg.get("map_route_search_radius_m", 8.0))),
                        )
                        if tr is None:
                            continue
                        # If the corridor actually delays ego relative to the
                        # current-state route, require a visible early commitment.
                        delayed = any(target > ego_tta + 0.35 for ego_tta, _d, _r, _a, target in active)
                        if delayed and ego_speed > 1.5:
                            check_idx = min(max(int(round(corridor_commit_t / dt)) - 1, 0), horizon - 1)
                            check_speed = float(np.linalg.norm(tr[check_idx, 3:5]))
                            if check_speed > max(ego_speed - corridor_min_drop, 0.5) + 1.0e-6:
                                continue
                        violations: list[float] = []
                        for _ego_tta, _dist, region, _agent_index, target in active:
                            actual_tta, _actual_dist = trajectory_entry_to_region(tr, region, current_state=ego_cur, dt=dt)
                            violations.append(0.0 if not np.isfinite(actual_tta) else max(0.0, float(target - actual_tta)))
                        if violations and max(violations) > corridor_tol + 1.0e-6:
                            continue
                        binding = max(active, key=lambda row: (row[4] - row[0], row[4], -row[1]))
                        accepted = add(
                            tr,
                            MacroType.MERGE_BEHIND,
                            util=0.08 + 0.04 * gap + 0.02 * max(0.0, -speed_scale) - (0.03 if recovery_used else 0.0),
                            neutral=True,
                            source=ProposalSource.MULTI_CONFLICT_CORRIDOR,
                            region_id=int(binding[2].conflict_id),
                            target_time_s=float(binding[4]),
                            timing_side=1,
                            target_agent_index=int(binding[3]),
                            gap_s=gap,
                            accel_mps2=float(recovery_accel if recovery_used else 0.0),
                            entry_distance_m=float(binding[1]),
                            target_tta_error_s=float(max(violations) if violations else 0.0),
                        )
                        if accepted:
                            corridor_keys.add(key)
                            corridor_added += 1

    # Fill remaining slots with terminal speed/position lattice variants.  Both
    # terminal speed and progress offsets affect the primitive geometry; otherwise
    # s_off only changes utility and the lattice collapses to duplicate paths.
    T_sec = max(horizon * dt, 1e-3)
    for v_off in cand_cfg.get("terminal_speed_offsets_mps", [-4, -2, 0, 2, 4]):
        for s_off in cand_cfg.get("terminal_s_offsets_m", [-15, -8, 0, 8, 15, 25]):
            if len(candidates) >= K:
                break
            accel_from_v = float(v_off) / T_sec
            accel_from_s = 2.0 * float(s_off) / max(T_sec * T_sec, 1e-3)
            blend = float(cand_cfg.get("terminal_lattice_speed_position_blend", 0.5))
            accel = float(np.clip(blend * accel_from_v + (1.0 - blend) * accel_from_s, -4.0, 2.5))
            util = -0.02 * float(s_off) + 0.03 * max(0.0, -float(v_off))
            add(
                constant_accel_trajectory(ego_cur, horizon, dt, accel=accel),
                MacroType.KEEP_LANE,
                util=util,
                source=ProposalSource.TERMINAL,
                accel_mps2=accel,
            )
    traj = np.zeros((K, horizon, 7), dtype=np.float32)
    valid = np.zeros(K, dtype=bool)
    macro_arr = np.full(K, int(MacroType.PAD), dtype=np.int32)
    utility_arr = np.zeros(K, dtype=np.float32)
    is_logged_arr = np.zeros(K, dtype=bool)
    is_neutral_arr = np.zeros(K, dtype=bool)
    proposal_source_arr = np.full(K, int(ProposalSource.PAD), dtype=np.int32)
    proposal_region_id_arr = np.full(K, -1, dtype=np.int32)
    proposal_target_time_arr = np.full(K, -1.0, dtype=np.float32)
    proposal_timing_side_arr = np.zeros(K, dtype=np.int8)
    proposal_target_agent_arr = np.full(K, -1, dtype=np.int32)
    proposal_gap_arr = np.full(K, -1.0, dtype=np.float32)
    proposal_accel_arr = np.full(K, -99.0, dtype=np.float32)
    proposal_entry_distance_arr = np.full(K, -1.0, dtype=np.float32)
    proposal_target_tta_error_arr = np.full(K, -1.0, dtype=np.float32)
    for k, c in enumerate(candidates[:K]):
        traj[k] = c
        valid[k] = True
        macro_arr[k] = macro[k]
        utility_arr[k] = utility[k]
        is_logged_arr[k] = is_logged[k]
        is_neutral_arr[k] = is_neutral[k]
        proposal_source_arr[k] = proposal_source[k]
        proposal_region_id_arr[k] = proposal_region_id[k]
        proposal_target_time_arr[k] = proposal_target_time_s[k]
        proposal_timing_side_arr[k] = proposal_timing_side[k]
        proposal_target_agent_arr[k] = proposal_target_agent_index[k]
        proposal_gap_arr[k] = proposal_gap_s[k]
        proposal_accel_arr[k] = proposal_accel_mps2[k]
        proposal_entry_distance_arr[k] = proposal_entry_distance_m[k]
        proposal_target_tta_error_arr[k] = proposal_target_tta_error_s[k]
    return {
        "trajectory": traj,
        "macro_type": macro_arr,
        "valid": valid,
        "ego_utility_prior": utility_arr,
        "is_logged": is_logged_arr,
        "is_neutral": is_neutral_arr,
        "proposal_source": proposal_source_arr,
        "proposal_region_id": proposal_region_id_arr,
        "proposal_target_time_s": proposal_target_time_arr,
        "proposal_timing_side": proposal_timing_side_arr,
        "proposal_target_agent_index": proposal_target_agent_arr,
        "proposal_gap_s": proposal_gap_arr,
        "proposal_accel_mps2": proposal_accel_arr,
        "proposal_entry_distance_m": proposal_entry_distance_arr,
        "proposal_target_tta_error_s": proposal_target_tta_error_arr,
        # Use macro type as a coarse topology surrogate.  Downstream diagnostics
        # should not see a degenerate single-topology candidate bank.
        "topology_id": macro_arr.copy(),
        # Private build-time diagnostics.  label_engine intentionally does not
        # serialize this object into NPZ; it is attached only to a filtered/error
        # profile row when no valid proposal survives.
        "_proposal_debug": {
            "attempted": int(sum(attempted_by_source.values())),
            "accepted": int(len(candidates)),
            "lane_point_count": int(len(lane_points)),
            "rejection_counts": dict(rejection_counts),
            "attempted_by_source": dict(attempted_by_source),
            "accepted_by_source": dict(accepted_by_source),
        },
    }
