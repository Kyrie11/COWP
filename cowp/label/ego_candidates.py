from __future__ import annotations

import numpy as np

from cowp.core.constants import MacroType
from cowp.core.types import ScenarioData, future_states_to_traj7
from cowp.geometry.lane_graph import build_conflict_regions, tta_to_region
from cowp.label.trajectory_primitives import constant_accel_trajectory, resample_logged, smooth_stop_trajectory


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


def _candidate_valid(traj: np.ndarray, cfg: dict, lane_points: np.ndarray | None = None) -> bool:
    cand_cfg = cfg.get("candidate", {})
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    if len(traj) < int(cand_cfg.get("min_valid_horizon_steps", 50)):
        return False
    if not np.all(np.isfinite(traj)):
        return False
    speed = np.linalg.norm(traj[:, 3:5], axis=-1)
    acc = np.diff(speed, prepend=speed[0]) / max(dt, 1e-3)
    jerk = np.diff(acc, prepend=acc[0]) / max(dt, 1e-3)
    if np.nanmax(acc) > float(cand_cfg.get("max_accel_mps2", 4.0)) + 1e-3:
        return False
    if -np.nanmin(acc) > float(cand_cfg.get("max_decel_mps2", 6.0)) + 1e-3:
        return False
    if np.nanmax(np.abs(jerk)) > float(cand_cfg.get("max_jerk_mps3", 6.0)) + 2.0:
        # Lattice primitives can have a step at the first time; allow a small numerical slack.
        return False
    if lane_points is not None and not _candidate_map_compliant(traj, lane_points, cfg):
        return False
    return True




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
    return float(np.clip(a, lo, hi))


def _agent_ttas_to_region(scene: ScenarioData, region_center: np.ndarray, cfg: dict) -> list[float]:
    """Return plausible surrounding-agent TTAs to one conflict region."""
    cur = int(scene.current_time_index)
    states = np.asarray(scene.states[:, cur], dtype=np.float32)
    max_agents = int(cfg.get("candidate", {}).get("timing_envelope_max_agents", 6))
    max_dist = float(cfg.get("candidate", {}).get("timing_envelope_agent_radius_m", 80.0))
    horizon_s = float(cfg.get("time", {}).get("future_steps", 80)) * float(cfg.get("time", {}).get("dt", 0.1))
    rows: list[tuple[float, float]] = []
    for j, st in enumerate(states):
        if j == int(scene.sdc_track_index) or st.shape[0] < 11 or float(st[10]) <= 0.5:
            continue
        to_region = np.asarray(region_center, dtype=np.float32)[:2] - st[:2]
        dist = float(np.linalg.norm(to_region))
        if not np.isfinite(dist) or dist < 1.0 or dist > max_dist:
            continue
        vel = st[3:5]
        speed = float(max(np.linalg.norm(vel), st[5] if st.shape[0] > 5 else 0.0, 0.0))
        if speed < 0.35:
            continue
        approach = float(np.dot(vel, to_region) / max(speed * dist, 1.0e-6))
        if approach < float(cfg.get("candidate", {}).get("timing_envelope_min_approach_cos", 0.15)):
            continue
        tta = dist / max(speed, 0.35)
        if 0.35 <= tta <= horizon_s + 2.0:
            rows.append((tta, dist))
    rows.sort(key=lambda x: (x[0], x[1]))
    return [float(t) for t, _ in rows[:max_agents]]

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

    def add(traj: np.ndarray, m: MacroType, util: float = 0.0, logged: bool = False, neutral: bool = False):
        if len(candidates) >= K:
            return
        traj = traj[:horizon].astype(np.float32)
        # Logged replay is retained as an observed reference even if map geometry
        # is incomplete; all synthetic proposals are map-screened.
        proposal_lane_points = None if logged else lane_points
        if len(traj) == horizon and _candidate_valid(traj, cfg, proposal_lane_points) and not _near_duplicate(traj, m):
            candidates.append(traj)
            macro.append(int(m))
            speed = np.linalg.norm(traj[:, 3:5], axis=-1)
            progress = float(np.linalg.norm(traj[-1, :2] - traj[0, :2]))
            comfort = float(np.mean(np.abs(np.diff(speed, prepend=speed[0]) / max(dt, 1e-3))))
            utility.append(float(util - 0.05 * progress + 0.2 * comfort))
            is_logged.append(logged)
            is_neutral.append(neutral)

    add(logged_full, MacroType.LOGGED_EGO, util=-1.0, logged=True)
    add(constant_accel_trajectory(ego_cur, horizon, dt, accel=0.0), MacroType.KEEP_LANE, util=0.0)
    for acc in cand_cfg.get("accelerate_values_mps2", [0.5, 1.0, 1.5]):
        add(constant_accel_trajectory(ego_cur, horizon, dt, accel=float(acc)), MacroType.ACCELERATE_CROSS, util=0.2)
    for decel in cand_cfg.get("yield_decel_values_mps2", [-1.0, -2.0, -3.0]):
        add(constant_accel_trajectory(ego_cur, horizon, dt, accel=float(decel)), MacroType.YIELD, util=0.5, neutral=float(decel) in (-2.0, -3.0))
        add(smooth_stop_trajectory(ego_cur, horizon, dt, decel=float(decel), creep_speed=0.0), MacroType.DECELERATE_CROSS, util=0.6)
    regions = conflict_regions if conflict_regions is not None else build_conflict_regions(scene.map_data, cfg)
    if regions:
        keep = constant_accel_trajectory(ego_cur, horizon, dt, accel=0.0)
        nearest = min(regions, key=lambda r: np.linalg.norm(r.center_xy - ego_cur[:2]))
        tta = tta_to_region(keep, nearest, dt=dt)
        dist_to_conflict = float(np.linalg.norm(nearest.center_xy - ego_cur[:2]))
        for margin in cand_cfg.get("stop_margin_to_conflict_m", [2.0, 5.0, 8.0]):
            stop_after = max(0.0, dist_to_conflict - float(margin))
            add(smooth_stop_trajectory(ego_cur, horizon, dt, decel=-2.5, stop_after_m=stop_after), MacroType.STOP_BEFORE_CONFLICT, util=0.7, neutral=True)
        if np.isfinite(tta):
            # Legacy ego-only timing offsets are retained for ablation/coverage.
            for offset in cand_cfg.get("merge_time_offsets_s", [-1.5, -0.8, 0.0, 0.8, 1.5]):
                target_time = max(0.45, float(tta) + float(offset))
                target_acc = _constant_accel_for_arrival(dist_to_conflict, float(max(ego_cur[5], np.linalg.norm(ego_cur[3:5]), 0.0)), target_time, cfg)
                if target_acc is not None:
                    add(
                        constant_accel_trajectory(ego_cur, horizon, dt, accel=target_acc),
                        MacroType.MERGE_AHEAD if offset <= 0 else MacroType.MERGE_BEHIND,
                        util=0.1 + max(0.0, float(offset)),
                        neutral=bool(offset > 0),
                    )

            # v16.8.2 Bidirectional Conflict-Time Envelope (BCTE): construct
            # physically solved ego arrival times immediately before/after other
            # agents' plausible arrival to the same conflict region.  This directly
            # targets the proposal-coverage ceiling observed in the mechanism gate.
            ego_speed = float(max(ego_cur[5], np.linalg.norm(ego_cur[3:5]), 0.0))
            gap_values = [float(x) for x in cand_cfg.get("timing_envelope_gap_s", [0.8, 1.4, 2.0])]
            for other_tta in _agent_ttas_to_region(scene, nearest.center_xy, cfg):
                for gap in gap_values:
                    for side in (-1.0, 1.0):
                        target_time = float(other_tta + side * gap)
                        target_acc = _constant_accel_for_arrival(dist_to_conflict, ego_speed, target_time, cfg)
                        if target_acc is None:
                            continue
                        behind = side > 0.0
                        add(
                            constant_accel_trajectory(ego_cur, horizon, dt, accel=target_acc),
                            MacroType.MERGE_BEHIND if behind else MacroType.MERGE_AHEAD,
                            util=(0.15 + 0.08 * gap if behind else 0.05 + 0.03 * gap),
                            neutral=behind,
                        )
    for lateral, m in [(-3.5, MacroType.LANE_CHANGE_RIGHT), (3.5, MacroType.LANE_CHANGE_LEFT)]:
        for delay in cand_cfg.get("lane_change_start_delay_s", [0.0, 0.5, 1.0, 1.5]):
            # Hold the lane during delay, then apply a smoothed lateral offset over the full primitive.
            tr = constant_accel_trajectory(ego_cur, horizon, dt, accel=0.0, lateral_offset=lateral, start_delay_s=float(delay))
            add(tr, m, util=0.3)
    add(smooth_stop_trajectory(ego_cur, horizon, dt, decel=-1.0, creep_speed=1.0), MacroType.CREEP, util=1.0)
    add(smooth_stop_trajectory(ego_cur, horizon, dt, decel=-2.0, creep_speed=0.0), MacroType.NEUTRAL_EGO, util=0.8, neutral=True)

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
            add(constant_accel_trajectory(ego_cur, horizon, dt, accel=accel), MacroType.KEEP_LANE, util=util)
    traj = np.zeros((K, horizon, 7), dtype=np.float32)
    valid = np.zeros(K, dtype=bool)
    macro_arr = np.full(K, int(MacroType.PAD), dtype=np.int32)
    utility_arr = np.zeros(K, dtype=np.float32)
    is_logged_arr = np.zeros(K, dtype=bool)
    is_neutral_arr = np.zeros(K, dtype=bool)
    for k, c in enumerate(candidates[:K]):
        traj[k] = c
        valid[k] = True
        macro_arr[k] = macro[k]
        utility_arr[k] = utility[k]
        is_logged_arr[k] = is_logged[k]
        is_neutral_arr[k] = is_neutral[k]
    return {
        "trajectory": traj,
        "macro_type": macro_arr,
        "valid": valid,
        "ego_utility_prior": utility_arr,
        "is_logged": is_logged_arr,
        "is_neutral": is_neutral_arr,
        # Use macro type as a coarse topology surrogate.  Downstream diagnostics
        # should not see a degenerate single-topology candidate bank.
        "topology_id": macro_arr.copy(),
    }
