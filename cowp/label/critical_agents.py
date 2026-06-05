from __future__ import annotations

import numpy as np

from cowp.core.constants import ObjectType, PriorityRelation
from cowp.core.types import ScenarioData, future_states_to_traj7
from cowp.geometry.lane_graph import build_conflict_regions, closest_conflict_for_pair
from cowp.label.priority import determine_priority


def _id_in_objects_of_interest(scene: ScenarioData, track_idx: int) -> bool:
    tid = int(scene.track_id[track_idx])
    return bool(tid in set(int(x) for x in scene.objects_of_interest.tolist()) or track_idx in set(int(x) for x in scene.tracks_to_predict.tolist()))


def _ego_candidate_bank(scene: ScenarioData, cfg: dict, ego_candidates: dict[str, np.ndarray] | np.ndarray | None) -> list[np.ndarray]:
    cur = scene.current_time_index
    horizon = int(cfg.get("time", {}).get("future_steps", 80))
    logged = future_states_to_traj7(scene.states[scene.sdc_track_index, cur + 1 : cur + 1 + horizon, :], horizon, current_state=scene.states[scene.sdc_track_index, cur])
    bank = [logged]
    if ego_candidates is None:
        return bank
    if isinstance(ego_candidates, dict):
        arr = np.asarray(ego_candidates.get("trajectory", []), dtype=np.float32)
        valid = np.asarray(ego_candidates.get("valid", np.ones(arr.shape[0], dtype=bool)), dtype=bool)
    else:
        arr = np.asarray(ego_candidates, dtype=np.float32)
        valid = np.ones(arr.shape[0], dtype=bool) if arr.ndim >= 3 else np.zeros(0, dtype=bool)
    if arr.ndim == 3:
        for k in np.where(valid)[0]:
            tr = arr[int(k)]
            if tr.shape[0] >= 2 and np.all(np.isfinite(tr)) and not np.allclose(tr[:, :2], 0.0):
                bank.append(tr[:horizon].astype(np.float32))
    return bank


def select_critical_agents(
    scene: ScenarioData,
    cfg: dict,
    ego_candidates: dict[str, np.ndarray] | np.ndarray | None = None,
    conflict_regions: list | None = None,
) -> dict[str, np.ndarray]:
    limits = cfg.get("limits", {})
    crit_cfg = cfg.get("critical", {})
    cur = scene.current_time_index
    max_a = int(limits.get("max_critical_agents", 8))
    ego_bank = _ego_candidate_bank(scene, cfg, ego_candidates)
    ego = ego_bank[0]
    regions = conflict_regions if conflict_regions is not None else build_conflict_regions(scene.map_data, cfg)
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
            ag = future_states_to_traj7(fut, len(ego), current_state=scene.states[i, cur])
            for ego_cand in ego_bank:
                T = min(len(ego_cand), len(ag))
                if T:
                    min_dist_future = min(min_dist_future, float(np.min(np.linalg.norm(ego_cand[:T, :2] - ag[:T, :2], axis=-1))))
                if regions:
                    region, te, ti, _ = closest_conflict_for_pair(ego_cand, ag, regions, dt=float(cfg.get("time", {}).get("dt", 0.1)))
                    if region is not None and abs(float(te - ti)) < abs(delta_tta_min):
                        delta_tta_min = float(te - ti)
                    if region is not None and abs(float(te - ti)) < float(crit_cfg.get("tta_delta_threshold", 3.0)):
                        shared_conflict = True
        rho = determine_priority(scene, i, ego, future_states_to_traj7(fut, len(ego), current_state=scene.states[i, cur]) if np.any(valid) else None, cfg)
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
