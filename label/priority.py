from __future__ import annotations

import numpy as np

from cowp.core.constants import PriorityRelation
from cowp.core.types import ScenarioData, ensure_trajectory_7
from cowp.geometry.boxes import normalize_angle
from cowp.geometry.map_projection import project_state_to_lane


def _first_arrival_to_close_points(
    ego_traj: np.ndarray,
    agent_traj: np.ndarray,
    dt: float = 0.1,
    threshold: float = 8.0,
) -> tuple[float, float]:
    """Independent arrival times at the first shared conflict location.

    The previous implementation searched only synchronous trajectory samples and
    then returned the same index for both agents, so its arrival-order branches
    could never fire.  Priority is about who reaches a shared path region first,
    not whether the two logged states are close at the same timestamp.  We search
    the pairwise path geometry, select the earliest credible shared region, and
    return each trajectory's own arrival time.
    """
    ego = np.asarray(ego_traj, dtype=np.float32)
    agent = np.asarray(agent_traj, dtype=np.float32)
    if ego.ndim != 2 or agent.ndim != 2 or len(ego) == 0 or len(agent) == 0:
        return float("inf"), float("inf")
    exy = ego[:, :2]
    axy = agent[:, :2]
    finite_e = np.all(np.isfinite(exy), axis=-1)
    finite_a = np.all(np.isfinite(axy), axis=-1)
    if not finite_e.any() or not finite_a.any():
        return float("inf"), float("inf")
    dist = np.linalg.norm(exy[:, None, :] - axy[None, :, :], axis=-1)
    valid = finite_e[:, None] & finite_a[None, :] & (dist <= float(threshold))
    if not valid.any():
        return float("inf"), float("inf")
    ii, jj = np.nonzero(valid)
    # Prefer the earliest shared region; distance breaks ties.  A tiny temporal
    # term avoids selecting a later repeated crossing with the same distance.
    score = np.maximum(ii, jj).astype(np.float64) + 1.0e-3 * (ii + jj) + 1.0e-4 * dist[ii, jj]
    q = int(np.argmin(score))
    return float(ii[q] * dt), float(jj[q] * dt)


def determine_priority(scene: ScenarioData, agent_index: int, ego_traj: np.ndarray | None, agent_traj: np.ndarray | None, cfg: dict) -> PriorityRelation:
    cur = scene.current_time_index
    ego_state = scene.states[scene.sdc_track_index, cur]
    agent_state = scene.states[agent_index, cur]
    ego_proj = project_state_to_lane(ego_state, scene.map_data)
    ag_proj = project_state_to_lane(agent_state, scene.map_data)
    ego_lane = scene.map_data.lanes.get(ego_proj.lane_id)
    ag_lane = scene.map_data.lanes.get(ag_proj.lane_id)

    if ego_lane is not None and ag_lane is not None:
        if ego_lane.controlled_by_stop and not ag_lane.controlled_by_stop:
            return PriorityRelation.AGENT_PRIORITY
        if ag_lane.controlled_by_stop and not ego_lane.controlled_by_stop:
            return PriorityRelation.EGO_PRIORITY
        # Signal *presence* is not signal right-of-way.  ScenarioData does not
        # carry the live phase associated with each lane, so inferring priority
        # from controlled_by_signal alone creates systematic label noise.
        # Lane ownership: if ego is crossing laterally into the agent lane, the lane owner has priority.
        if ego_proj.lane_id != ag_proj.lane_id and abs(ag_proj.l) < 2.2:
            heading_diff = abs(float(normalize_angle(ego_state[6] - agent_state[6])))
            if heading_diff < np.deg2rad(35):
                lateral_sep = abs(float(ego_proj.l))
                if lateral_sep > 1.0 and np.linalg.norm(ego_state[:2] - agent_state[:2]) < 35.0:
                    return PriorityRelation.AGENT_PRIORITY
        # Merge: vehicle already in successor/mainline lane is preferred.
        if ag_proj.lane_id in ego_lane.exit_lanes and ego_proj.lane_id not in ag_lane.exit_lanes:
            return PriorityRelation.AGENT_PRIORITY
        if ego_proj.lane_id in ag_lane.exit_lanes and ag_proj.lane_id not in ego_lane.exit_lanes:
            return PriorityRelation.EGO_PRIORITY

    dt = float(cfg.get("time", {}).get("dt", 0.1))
    margin = float(cfg.get("priority", {}).get("arrival_order_margin_s", 0.5))
    if ego_traj is not None and agent_traj is not None and len(ego_traj) and len(agent_traj):
        ego_t, ag_t = _first_arrival_to_close_points(ego_traj, agent_traj, dt=dt, threshold=8.0)
        if np.isfinite(ego_t) and np.isfinite(ag_t):
            # If the agent is already established ahead along a common heading,
            # preserve the lane owner's claim before using path-arrival order.
            direction = np.array([np.cos(agent_state[6]), np.sin(agent_state[6])], dtype=np.float32)
            if (
                float((ego_state[:2] - agent_state[:2]) @ direction) < 0.0
                and abs(normalize_angle(ego_state[6] - agent_state[6])) < np.deg2rad(45)
            ):
                return PriorityRelation.AGENT_PRIORITY
            if ag_t + margin < ego_t:
                return PriorityRelation.AGENT_PRIORITY
            if ego_t + margin < ag_t:
                return PriorityRelation.EGO_PRIORITY
    return PriorityRelation.EQUAL_OR_NEGOTIATED


def priority_preserved(agent_traj: np.ndarray, natural_ref: np.ndarray | None, rho: PriorityRelation, cfg: dict, gap_loss_m: float = 0.0) -> bool:
    pr_cfg = cfg.get("priority", {})
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    if len(agent_traj) < 2:
        return True
    speed = np.linalg.norm(agent_traj[:, 3:5], axis=-1)
    acc = np.diff(speed, prepend=speed[0]) / max(dt, 1e-3)
    jerk = np.diff(acc, prepend=acc[0]) / max(dt, 1e-3)
    if np.min(acc) < float(pr_cfg.get("max_comfort_decel_vehicle", -3.0)) - 0.25:
        return False
    if np.max(acc) > float(pr_cfg.get("max_comfort_accel_vehicle", 2.5)) + 0.25:
        return False
    if np.max(np.abs(jerk)) > float(pr_cfg.get("max_comfort_jerk_vehicle", 3.0)) + 1.0:
        return False
    if rho == PriorityRelation.AGENT_PRIORITY and natural_ref is not None and len(natural_ref):
        progress_agent = float(np.linalg.norm(agent_traj[-1, :2] - agent_traj[0, :2]))
        progress_nat = float(np.linalg.norm(natural_ref[-1, :2] - natural_ref[0, :2]))
        if progress_nat - progress_agent > float(pr_cfg.get("priority_progress_loss_tolerance_m", 5.0)):
            return False
    if gap_loss_m > float(pr_cfg.get("gap_loss_threshold_m", 6.0)):
        return False
    return True
