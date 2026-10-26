from __future__ import annotations

import numpy as np

from cowp.core.constants import ObjectType, PriorityRelation
from cowp.core.types import ScenarioData, future_states_to_traj7
from cowp.geometry.lane_graph import build_conflict_regions, build_scene_conflict_regions, closest_conflict_for_pair
from cowp.geometry.map_projection import project_state_to_lane
from cowp.label.priority import determine_priority
from cowp.label.trajectory_primitives import constant_accel_trajectory, smooth_stop_trajectory


def _track_metadata_flags(scene: ScenarioData, track_idx: int) -> tuple[bool, bool]:
    """Return offline WOMD metadata flags for diagnostics only.

    Neither ``objects_of_interest`` nor ``tracks_to_predict`` is guaranteed to be
    available to an online planner as a causal interaction-selection signal.  The
    critical selector therefore keeps these annotations out of its score by
    default; they may still be logged for dataset stratification/auditing.
    """
    tid = int(scene.track_id[track_idx])
    ooi = bool(tid in set(int(x) for x in scene.objects_of_interest.tolist()))
    # ScenarioData normalizes tracks_to_predict to track indices in this codebase.
    ttp = bool(track_idx in set(int(x) for x in scene.tracks_to_predict.tolist()))
    return ooi, ttp


def _ego_candidate_bank(scene: ScenarioData, cfg: dict, ego_candidates: dict[str, np.ndarray] | np.ndarray | None) -> list[np.ndarray]:
    """Return the *causal screening* ego bank used to choose global critical agents.

    The default ``causal_anchor_v2`` bank is a deterministic function of current
    SDC state only.  Logged ego future and optional proposal families are excluded
    so critical selection is reproducible online and invariant to proposal-bank
    ablations.  The older modes remain only as explicit compatibility ablations.
    """
    cur = scene.current_time_index
    horizon = int(cfg.get("time", {}).get("future_steps", 80))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    current = np.asarray(scene.states[scene.sdc_track_index, cur], dtype=np.float32)
    mode = str(cfg.get("critical", {}).get("selection_reference_mode", "causal_anchor_v2"))

    def _causal_anchors() -> list[np.ndarray]:
        bank: list[np.ndarray] = []
        bank.append(constant_accel_trajectory(current, horizon, dt, accel=0.0))
        bank.append(constant_accel_trajectory(current, horizon, dt, accel=1.0))
        bank.append(constant_accel_trajectory(current, horizon, dt, accel=-1.5))
        bank.append(smooth_stop_trajectory(current, horizon, dt, decel=-2.0, creep_speed=0.0))
        lane_offset = float(cfg.get("critical", {}).get("anchor_lane_offset_m", 3.5))
        lane_delay = float(cfg.get("critical", {}).get("anchor_lane_change_delay_s", 0.5))
        bank.append(constant_accel_trajectory(current, horizon, dt, accel=0.0, lateral_offset=lane_offset, start_delay_s=lane_delay))
        bank.append(constant_accel_trajectory(current, horizon, dt, accel=0.0, lateral_offset=-lane_offset, start_delay_s=lane_delay))
        return [np.asarray(x, dtype=np.float32)[:horizon] for x in bank if x is not None and len(x) >= 2 and np.all(np.isfinite(x))]

    if mode == "causal_anchor_v2":
        return _causal_anchors()

    # v16.8.8 compatibility mode: retains logged ego future as an oracle anchor.
    # It must not pass the v16.8.20 promotion gate.
    logged = future_states_to_traj7(
        scene.states[scene.sdc_track_index, cur + 1 : cur + 1 + horizon, :],
        horizon,
        current_state=current,
    )
    if mode == "fixed_anchor_v1":
        return [logged] + _causal_anchors()
    if mode != "proposal_bank_legacy":
        raise ValueError(f"Unknown critical.selection_reference_mode={mode!r}")

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
    regions = conflict_regions if conflict_regions is not None else build_scene_conflict_regions(scene, cfg)
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
        # Causal interaction projection: use current state + constant-velocity
        # dynamics only.  Logged agent future is reserved for post-selection
        # supervision/auditability below and never influences who is selected.
        horizon = len(ego)
        dt = float(cfg.get("time", {}).get("dt", 0.1))
        ag = constant_accel_trajectory(scene.states[i, cur], horizon, dt, accel=0.0)
        min_dist_future = float("inf")  # historical key; now means causal projected distance
        delta_tta_min = float("inf")
        shared_conflict = False
        for ego_cand in ego_bank:
            T = min(len(ego_cand), len(ag))
            if T:
                min_dist_future = min(min_dist_future, float(np.min(np.linalg.norm(ego_cand[:T, :2] - ag[:T, :2], axis=-1))))
            if regions:
                region, te, ti, _ = closest_conflict_for_pair(ego_cand, ag, regions, dt=dt)
                if region is not None and abs(float(te - ti)) < abs(delta_tta_min):
                    delta_tta_min = float(te - ti)
                if region is not None and abs(float(te - ti)) < float(crit_cfg.get("tta_delta_threshold", 3.0)):
                    shared_conflict = True
        rho = determine_priority(scene, i, ego, ag, cfg)
        score = 0.0
        ooi, ttp = _track_metadata_flags(scene, i)
        score += float(crit_cfg.get("objects_of_interest_score_weight", 0.0)) if ooi else 0.0
        score += float(crit_cfg.get("tracks_to_predict_score_weight", 0.0)) if ttp else 0.0
        if np.isfinite(min_dist_future):
            score += 3.0 * float(np.exp(-min_dist_future / 15.0))
        if np.isfinite(delta_tta_min):
            score += 3.0 * float(np.exp(-abs(delta_tta_min) / 2.0))
        score += 2.0 if shared_conflict else 0.0
        score += 1.5 if rho == PriorityRelation.AGENT_PRIORITY else 0.0
        score += 1.0 if cur_dist < 20.0 else 0.0

        # Do not mark every vehicle inside a broad radius as critical.  COWP's
        # feasibility test is universal over critical agents; overly broad
        # selection both slows labeling and makes non-coercive candidates vanish
        # for reasons unrelated to the candidate's actual burden transfer.
        min_score = float(crit_cfg.get("min_score", 1.5))
        max_current = float(crit_cfg.get("max_current_distance_m", 40.0))
        max_future = float(crit_cfg.get("max_projected_distance_m", crit_cfg.get("max_future_distance_m", 15.0)))
        always_current = float(crit_cfg.get("always_keep_current_distance_m", 18.0))
        require_reason = bool(crit_cfg.get("current_distance_requires_future_or_conflict", True))
        score_hit = score >= min_score
        future_hit = min_dist_future < max_future
        current_hit = cur_dist < max_current and (not require_reason or score_hit or future_hit or shared_conflict)
        very_near_hit = cur_dist < always_current
        if score_hit or future_hit or shared_conflict or current_hit or very_near_hit:
            scores.append((score, i, cur_dist, min_dist_future, rho))
    scores.sort(key=lambda x: (-x[0], x[2]))
    idx = np.full(max_a, -1, dtype=np.int32)
    valid = np.zeros(max_a, dtype=bool)
    score_arr = np.zeros(max_a, dtype=np.float32)
    base_priority = np.zeros(max_a, dtype=np.int32)
    mechanism_valid = np.zeros(max_a, dtype=bool)
    audit_reason = np.full(max_a, 3, dtype=np.int32)  # 0 lane, 1 empirical future, 2 both, 3 unavailable
    audit_future_steps = np.zeros(max_a, dtype=np.int32)
    audit_future_fraction = np.zeros(max_a, dtype=np.float32)
    nat_cfg = cfg.get("natural", {})
    require_auditability = bool(crit_cfg.get("require_natural_auditability", True))
    audit_search_radius = float(crit_cfg.get("auditability_map_search_radius_m", nat_cfg.get("map_route_search_radius_m", 8.0)))
    audit_min_steps = max(int(nat_cfg.get("empirical_corridor_min_future_valid_steps", nat_cfg.get("obs_min_future_valid_steps", 60))), 0)
    audit_min_frac = float(np.clip(nat_cfg.get("empirical_corridor_min_future_valid_fraction", nat_cfg.get("obs_min_future_valid_fraction", 0.70)), 0.0, 1.0))
    unauditable: list[dict[str, object]] = []
    for rank, item in enumerate(scores[:max_a]):
        score, i, cur_dist, min_future_dist, rho = item
        idx[rank] = i
        valid[rank] = True
        score_arr[rank] = float(score)
        base_priority[rank] = int(rho)

        # v16.8.13 keeps the *critical-selection universe* identical to v16.8.12.
        # Auditability is a separate offline-supervision mask: it must never use
        # logged future to decide who the planner considers critical at inference.
        proj = project_state_to_lane(scene.states[i, cur], scene.map_data, search_radius=audit_search_radius)
        lane_supported = bool(int(proj.lane_id) >= 0 and int(proj.lane_id) in scene.map_data.lanes)
        fut = scene.states[i, cur + 1 : cur + 1 + len(ego), :]
        fut_valid = fut[:, 10] > 0.5 if len(fut) else np.zeros(0, dtype=bool)
        future_steps = int(np.sum(fut_valid))
        future_frac = float(np.mean(fut_valid)) if len(fut_valid) else 0.0
        empirical_supported = bool(future_steps >= audit_min_steps and future_frac >= audit_min_frac)
        mechanism_valid[rank] = bool((not require_auditability) or lane_supported or empirical_supported)
        audit_reason[rank] = 2 if lane_supported and empirical_supported else (0 if lane_supported else (1 if empirical_supported else 3))
        audit_future_steps[rank] = int(future_steps)
        audit_future_fraction[rank] = float(future_frac)
        if not mechanism_valid[rank]:
            unauditable.append({
                "slot": int(rank), "track_index": int(i), "rho": int(rho),
                "score": float(score), "current_distance_m": float(cur_dist),
                "min_future_distance_m": None if not np.isfinite(min_future_dist) else float(min_future_dist),
                "future_valid_steps": int(future_steps), "future_valid_fraction": float(future_frac),
                "lane_supported": bool(lane_supported), "empirical_supported": bool(empirical_supported),
            })
    return {
        "track_index": idx,
        "valid": valid,
        "mechanism_valid": mechanism_valid,
        "score": score_arr,
        "base_priority": base_priority,
        "auditability_reason": audit_reason,
        "audit_future_valid_steps": audit_future_steps,
        "audit_future_valid_fraction": audit_future_fraction,
        "_selection_diagnostics": {
            "selected_count": int(valid.sum()),
            "mechanism_auditable_count": int((valid & mechanism_valid).sum()),
            "mechanism_unauditable_count": int((valid & ~mechanism_valid).sum()),
            "mechanism_unauditable": unauditable[:32],
            "selection_reference_mode": str(crit_cfg.get("selection_reference_mode", "causal_anchor_v2")),
            "selection_is_causal": bool(str(crit_cfg.get("selection_reference_mode", "causal_anchor_v2")) == "causal_anchor_v2"),
            "logged_future_used_for_selection": False,
            "objects_of_interest_score_weight": float(crit_cfg.get("objects_of_interest_score_weight", 0.0)),
            "tracks_to_predict_score_weight": float(crit_cfg.get("tracks_to_predict_score_weight", 0.0)),
        },
    }
