from __future__ import annotations

from collections import Counter

import numpy as np

from cowp.core.constants import MacroType, PriorityRelation, ProposalSource
from cowp.core.types import ScenarioData, future_states_to_traj7
from cowp.geometry.lane_graph import ConflictRegion, build_conflict_regions, trajectory_entry_to_region, tta_to_region
from cowp.label.trajectory_primitives import constant_accel_trajectory, priority_hold_release_trajectory, resample_logged, smooth_arrival_trajectory, smooth_stop_trajectory
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
    regions = conflict_regions if conflict_regions is not None else build_conflict_regions(scene.map_data, cfg)
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
            timing_profile_bins: set[tuple[int, int, int]] = set()
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
                            profile_key = (int(region.conflict_id), int(side), accel_bin)
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
    for lateral, m in [(-3.5, MacroType.LANE_CHANGE_RIGHT), (3.5, MacroType.LANE_CHANGE_LEFT)]:
        for delay in cand_cfg.get("lane_change_start_delay_s", [0.0, 0.5, 1.0, 1.5]):
            # Hold the lane during delay, then apply a smoothed lateral offset over the full primitive.
            tr = constant_accel_trajectory(ego_cur, horizon, dt, accel=0.0, lateral_offset=lateral, start_delay_s=float(delay))
            add(tr, m, util=0.3, source=ProposalSource.LANE_CHANGE)
    add(
        smooth_stop_trajectory(ego_cur, horizon, dt, decel=-1.0, creep_speed=1.0),
        MacroType.CREEP,
        util=1.0,
        source=ProposalSource.CREEP,
        accel_mps2=-1.0,
    )

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
        },
    }
