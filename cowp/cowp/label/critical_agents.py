from __future__ import annotations

import numpy as np

from cowp.core.constants import ObjectType, PriorityRelation
from cowp.core.types import ScenarioData, ensure_trajectory_7
from cowp.geometry.lane_graph import build_conflict_regions, closest_conflict_for_pair
from cowp.label.priority import determine_priority


def _id_in_objects_of_interest(scene: ScenarioData, track_idx: int) -> bool:
    tid = int(scene.track_id[track_idx])
    return bool(tid in set(int(x) for x in scene.objects_of_interest.tolist()) or track_idx in set(int(x) for x in scene.tracks_to_predict.tolist()))


def select_critical_agents(scene: ScenarioData, cfg: dict, ego_candidates: np.ndarray | None = None) -> dict[str, np.ndarray]:
    limits = cfg.get("limits", {})
    crit_cfg = cfg.get("critical", {})
    cur = scene.current_time_index
    max_a = int(limits.get("max_critical_agents", 8))
    ego_future = scene.states[scene.sdc_track_index, cur + 1 :, :]
    ego = ensure_trajectory_7(ego_future[: int(cfg.get("time", {}).get("future_steps", 80))])
    regions = build_conflict_regions(scene.map_data, cfg)
    scores: list[tuple[float, int, float, float, PriorityRelation]] = []
    for i in range(scene.num_agents):
        if i == scene.sdc_track_index:
            continue
        if scene.states[i, cur, 10] < 0.5:
            continue
        obj_type = int(scene.object_type[i])
        if obj_type not in (int(ObjectType.VEHICLE), int(ObjectType.CYCLIST), int(ObjectType.PEDESTRIAN)):
            continue
        if bool(crit_cfg.get("vehicle_only_main", False)) and obj_type != int(ObjectType.VEHICLE):
            continue
        cur_dist = float(np.linalg.norm(scene.states[i, cur, :2] - scene.states[scene.sdc_track_index, cur, :2]))
        fut = scene.states[i, cur + 1 : cur + 1 + len(ego), :]
        valid = fut[:, 10] > 0.5
        min_dist_future = float("inf")
        delta_tta_min = float("inf")
        shared_conflict = False
        if np.any(valid):
            ag = ensure_trajectory_7(fut[valid])
            T = min(len(ego), len(ag))
            if T:
                min_dist_future = float(np.min(np.linalg.norm(ego[:T, :2] - ag[:T, :2], axis=-1)))
            if regions:
                region, te, ti, _ = closest_conflict_for_pair(ego, ag, regions, dt=float(cfg.get("time", {}).get("dt", 0.1)))
                if region is not None:
                    delta_tta_min = float(te - ti)
                    shared_conflict = abs(delta_tta_min) < float(crit_cfg.get("tta_delta_threshold", 3.0))
        rho = determine_priority(scene, i, ego, ensure_trajectory_7(fut[valid]) if np.any(valid) else None, cfg)
        score = 0.0
        score += 2.0 if _id_in_objects_of_interest(scene, i) else 0.0
        if np.isfinite(min_dist_future):
            score += 3.0 * float(np.exp(-min_dist_future / 15.0))
        if np.isfinite(delta_tta_min):
            score += 3.0 * float(np.exp(-abs(delta_tta_min) / 2.0))
        score += 2.0 if shared_conflict else 0.0
        score += 1.5 if rho == PriorityRelation.AGENT_PRIORITY else 0.0
        score += 1.0 if cur_dist < 20.0 else 0.0
        if score >= float(crit_cfg.get("min_score", 1.5)) or cur_dist < float(crit_cfg.get("max_current_distance_m", 40.0)) or min_dist_future < float(crit_cfg.get("max_future_distance_m", 15.0)):
            scores.append((score, i, cur_dist, min_dist_future, rho))
    scores.sort(key=lambda x: (-x[0], x[2]))
    idx = np.full(max_a, -1, dtype=np.int32)
    valid = np.zeros(max_a, dtype=bool)
    score_arr = np.zeros(max_a, dtype=np.float32)
    base_priority = np.zeros(max_a, dtype=np.int32)
    for rank, item in enumerate(scores[:max_a]):
        score, i, _, _, rho = item
        idx[rank] = i
        valid[rank] = True
        score_arr[rank] = float(score)
        base_priority[rank] = int(rho)
    return {"track_index": idx, "valid": valid, "score": score_arr, "base_priority": base_priority}
