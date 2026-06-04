from __future__ import annotations

import numpy as np

from cowp.core.constants import ObjectType
from cowp.core.types import ScenarioData, ensure_trajectory_7
from cowp.geometry.lane_graph import build_conflict_regions, closest_conflict_for_pair


def valid_scene_basic(scene: ScenarioData, cfg: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    cur = scene.current_time_index
    fut_req = min(int(cfg.get("time", {}).get("future_steps", 80)), scene.num_steps - cur - 1)
    if scene.sdc_track_index < 0 or scene.sdc_track_index >= scene.num_agents:
        reasons.append("invalid_sdc_track_index")
    elif scene.states[scene.sdc_track_index, cur, 10] < 0.5:
        reasons.append("ego_current_invalid")
    elif np.sum(scene.states[scene.sdc_track_index, cur + 1 : cur + 1 + fut_req, 10] > 0.5) < min(30, fut_req):
        reasons.append("ego_future_insufficient")
    vehicle_valid = (scene.object_type == int(ObjectType.VEHICLE)) & (scene.states[:, cur, 10] > 0.5)
    if int(vehicle_valid.sum()) < 2:
        reasons.append("too_few_current_vehicle_agents")
    if len(scene.map_data.lanes) == 0 and len(scene.map_data.road_edges) == 0:
        reasons.append("empty_map")
    if len(scene.timestamps) < 2 or not np.all(np.diff(scene.timestamps) > 0):
        reasons.append("timestamps_not_strictly_increasing")
    if len(scene.timestamps) >= 2:
        dt_med = float(np.median(np.diff(scene.timestamps)))
        if abs(dt_med - float(cfg.get("time", {}).get("dt", 0.1))) > 0.03:
            reasons.append("unexpected_dt")
    return len(reasons) == 0, reasons


def scene_types(scene: ScenarioData, cfg: dict) -> list[str]:
    cur = scene.current_time_index
    ego_future = scene.states[scene.sdc_track_index, cur + 1 :, :]
    ego_valid = ego_future[:, 10] > 0.5
    out: set[str] = set()
    if not np.any(ego_valid):
        return []
    ego = ensure_trajectory_7(ego_future[ego_valid])
    dy = ego[-1, 1] - ego[0, 1]
    dx = ego[-1, 0] - ego[0, 0]
    heading_change = abs(float(((ego[-1, 2] - ego[0, 2] + np.pi) % (2 * np.pi)) - np.pi))
    if abs(dy) > 2.5 and heading_change < np.deg2rad(30):
        out.add("LANE_CHANGE")
    regions = build_conflict_regions(scene.map_data, cfg)
    other_idxs = [i for i in range(scene.num_agents) if i != scene.sdc_track_index and scene.states[i, cur, 10] > 0.5]
    for i in other_idxs:
        a_future = scene.states[i, cur + 1 :, :]
        mask = a_future[:, 10] > 0.5
        if not np.any(mask):
            continue
        agent = ensure_trajectory_7(a_future[mask])
        if regions:
            region, te, ti, _ = closest_conflict_for_pair(ego, agent, regions, dt=float(cfg.get("time", {}).get("dt", 0.1)))
            if region is not None and abs(te - ti) < 3.0:
                out.add("INTERSECTION_CROSSING" if region.conflict_type == "LANE_INTERSECTION" else region.conflict_type)
        d = np.linalg.norm(ego[: min(len(ego), len(agent)), :2] - agent[: min(len(ego), len(agent)), :2], axis=-1)
        if len(d) and float(np.min(d)) < 8.0:
            out.add("DENSE_CAR_FOLLOWING")
            if abs(dy) > 2.0:
                out.add("CUT_IN")
    if any(l.controlled_by_signal for l in scene.map_data.lanes.values()):
        out.add("TRAFFIC_LIGHT_CONTROL")
    if any(l.controlled_by_stop for l in scene.map_data.lanes.values()):
        out.add("STOP_OR_YIELD_CONTROL")
    return sorted(out)


def is_interaction_heavy(scene: ScenarioData, cfg: dict) -> tuple[bool, dict[str, object]]:
    ok, reasons = valid_scene_basic(scene, cfg)
    if not ok:
        return False, {"basic_valid": False, "reasons": reasons, "scene_types": []}
    types = scene_types(scene, cfg)
    cur = scene.current_time_index
    ego = ensure_trajectory_7(scene.states[scene.sdc_track_index, cur + 1 :, :])
    min_future_dist = float("inf")
    for i in range(scene.num_agents):
        if i == scene.sdc_track_index or scene.states[i, cur, 10] < 0.5:
            continue
        tr = scene.states[i, cur + 1 :, :]
        mask = tr[:, 10] > 0.5
        if not np.any(mask):
            continue
        ag = ensure_trajectory_7(tr[mask])
        T = min(len(ego), len(ag))
        if T:
            min_future_dist = min(min_future_dist, float(np.min(np.linalg.norm(ego[:T, :2] - ag[:T, :2], axis=-1))))
    heavy = bool(types or min_future_dist < 8.0)
    return heavy, {"basic_valid": True, "reasons": [], "scene_types": types, "min_future_dist": min_future_dist}
