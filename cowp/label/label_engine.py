from __future__ import annotations

import numpy as np

from cowp.core.types import ScenarioData
from cowp.geometry.lane_graph import build_conflict_regions
from cowp.label.critical_agents import select_critical_agents
from cowp.label.ego_candidates import generate_ego_candidates
from cowp.label.natural_alternatives import generate_natural_alternatives
from cowp.label.safe_responses import generate_safe_responses
from cowp.label.witness import certify_witnesses


def _make_ego_neutral(candidates: dict[str, np.ndarray]) -> np.ndarray:
    neutral = np.where(candidates.get("is_neutral", np.zeros(len(candidates["valid"]), dtype=bool)) & candidates["valid"])[0]
    if len(neutral):
        return candidates["trajectory"][int(neutral[0])]
    valid = np.where(candidates["valid"])[0]
    if len(valid):
        return candidates["trajectory"][int(valid[0])]
    raise ValueError("No valid ego candidates available for neutral intervention.")


def build_labels_for_scene(
    scene: ScenarioData,
    cfg: dict,
    ablation: dict | None = None,
    scene_meta: dict[str, object] | None = None,
    conflict_regions: list | None = None,
) -> dict[str, np.ndarray | str]:
    regions = conflict_regions if conflict_regions is not None else build_conflict_regions(scene.map_data, cfg)
    candidates = generate_ego_candidates(scene, cfg, conflict_regions=regions)
    critical = select_critical_agents(scene, cfg, candidates, conflict_regions=regions)
    ego_neutral = _make_ego_neutral(candidates)
    natural = generate_natural_alternatives(scene, critical, ego_neutral, cfg, ablation=ablation)
    response = generate_safe_responses(scene, candidates, critical, natural, cfg)
    witness = certify_witnesses(scene, candidates, critical, natural, response, cfg, ablation=ablation, conflict_regions=regions)
    max_c = int(cfg.get("limits", {}).get("max_conflict_regions", 64))
    conflict_tensor = np.zeros((max_c, 8), dtype=np.float32)
    conflict_valid = np.zeros(max_c, dtype=bool)
    for i, r in enumerate(regions[:max_c]):
        conflict_tensor[i] = [float(r.conflict_id), r.center_xy[0], r.center_xy[1], r.radius, float(r.involved_lane_ids[0]), float(r.involved_lane_ids[1]), 0.0, 0.0]
        conflict_valid[i] = True
    out: dict[str, np.ndarray | str] = {
        "scenario/id": scene.scenario_id,
        "scenario/current_time_index": np.asarray(scene.current_time_index, dtype=np.int32),
        "scenario/timestamps_seconds": scene.timestamps.astype(np.float32),
        "dataset/interaction_heavy": np.asarray(bool((scene_meta or {}).get("interaction_heavy", True)), dtype=bool),
        "dataset/scene_types": np.asarray(",".join(str(x) for x in (scene_meta or {}).get("scene_types", []))),
        "dataset/min_future_dist": np.asarray(float((scene_meta or {}).get("min_future_dist", np.inf)), dtype=np.float32),
        "cowp/candidates/trajectory": candidates["trajectory"],
        "cowp/candidates/macro_type": candidates["macro_type"],
        "cowp/candidates/valid": candidates["valid"],
        "cowp/candidates/conventional_safe": witness["candidate_conventional_safe"],
        "cowp/candidates/false_safe": witness["candidate_false_safe"],
        "cowp/candidates/noncoercive_feasible": witness["candidate_noncoercive_feasible"],
        "cowp/candidates/ego_utility_prior": candidates["ego_utility_prior"],
        "cowp/candidates/is_logged": candidates["is_logged"],
        "cowp/candidates/is_neutral": candidates["is_neutral"],
        "cowp/candidates/topology_id": candidates["topology_id"],
        "cowp/critical/track_index": critical["track_index"],
        "cowp/critical/track_id": np.asarray([scene.track_id[i] if 0 <= i < len(scene.track_id) else -1 for i in critical["track_index"]], dtype=np.int64),
        "cowp/critical/valid": critical["valid"],
        "cowp/critical/agent_type": np.asarray([scene.object_type[i] if 0 <= i < len(scene.object_type) else 0 for i in critical["track_index"]], dtype=np.int32),
        "cowp/critical/base_priority": critical["base_priority"],
        "cowp/critical/score": critical["score"],
        "cowp/natural/traj": natural["traj"],
        "cowp/natural/weight": natural["weight"],
        "cowp/natural/source": natural["source"],
        "cowp/natural/valid": natural["valid"],
        "cowp/natural/burden_neutral": natural["burden_neutral"],
        "cowp/natural/priority_preserved": natural["priority_preserved"],
        "cowp/natural/beta": natural["beta"],
        "cowp/response/traj": response["traj"],
        "cowp/response/valid": response["valid"],
        "cowp/response/source": response["source"],
        "cowp/response/is_safe": response["is_safe"],
        "cowp/response/is_low_burden": response["is_low_burden"],
        "cowp/response/burden_total": response["burden_total"],
        "cowp/response/burden_components": response["burden_components"],
        "cowp/witness/exists": witness["exists"],
        "cowp/witness/token": witness["token"],
        "cowp/witness/burden_total": witness["burden_total"],
        "cowp/witness/burden_components": witness["burden_components"],
        "cowp/witness/min_safe_burden": witness["min_safe_burden"],
        "cowp/witness/natural_conflict_mass": witness["natural_conflict_mass"],
        "cowp/witness/natural_conflict_mass_by_source": witness["natural_conflict_mass_by_source"],
        "cowp/witness/natural_mass_by_source": witness["natural_mass_by_source"],
        "cowp/witness/low_safe_mass_by_source": witness["low_safe_mass_by_source"],
        "cowp/witness/opr": witness["opr"],
        "cowp/witness/c_i": witness["c_i"],
        "cowp/witness/conflict_interval": witness["conflict_interval"],
        "cowp/witness/conflict_region_id": witness["conflict_region_id"],
        "cowp/witness/critical_agent_track_index": witness["critical_agent_track_index"],
        "cowp/witness/rho": witness["rho"],
        "map/conflict_regions": conflict_tensor,
        "map/conflict_region_valid": conflict_valid,
    }
    return out
