from __future__ import annotations

import time

import numpy as np

from cowp.core.types import ScenarioData
from cowp.geometry.lane_graph import build_conflict_regions
from cowp.label.critical_agents import select_critical_agents
from cowp.label.audit_relevance import compute_candidate_agent_audit
from cowp.label.ego_candidates import generate_ego_candidates
from cowp.label.natural_alternatives import build_pair_specific_ego_neutrals, generate_natural_alternatives
from cowp.label.safe_responses import generate_safe_responses
from cowp.label.witness import certify_witnesses


class NoValidEgoCandidatesError(ValueError):
    """Scene-local proposal failure that must not kill a dataset build."""

    def __init__(self, scenario_id: str, diagnostics: dict[str, object] | None = None):
        self.scenario_id = str(scenario_id)
        self.diagnostics = dict(diagnostics or {})
        super().__init__(
            f"No valid ego candidates available for neutral intervention: scenario_id={self.scenario_id}; "
            f"diagnostics={self.diagnostics}"
        )


def _make_ego_neutral(candidates: dict[str, object], scenario_id: str = "") -> np.ndarray:
    neutral = np.where(candidates.get("is_neutral", np.zeros(len(candidates["valid"]), dtype=bool)) & candidates["valid"])[0]
    if len(neutral):
        return candidates["trajectory"][int(neutral[0])]
    valid = np.where(candidates["valid"])[0]
    if len(valid):
        return candidates["trajectory"][int(valid[0])]
    raise NoValidEgoCandidatesError(scenario_id, candidates.get("_proposal_debug", {}))


def build_labels_for_scene(
    scene: ScenarioData,
    cfg: dict,
    ablation: dict | None = None,
    scene_meta: dict[str, object] | None = None,
    conflict_regions: list | None = None,
    profile_timings: dict[str, float] | None = None,
    profile_diagnostics: dict[str, object] | None = None,
) -> dict[str, np.ndarray | str]:
    def _timeit(name: str, fn):
        t = time.perf_counter()
        out = fn()
        if profile_timings is not None:
            profile_timings[name] = time.perf_counter() - t
        return out

    regions = conflict_regions if conflict_regions is not None else _timeit("engine_conflict_regions_s", lambda: build_conflict_regions(scene.map_data, cfg))
    candidates = _timeit("engine_candidates_s", lambda: generate_ego_candidates(scene, cfg, conflict_regions=regions))
    if profile_diagnostics is not None:
        profile_diagnostics["candidate"] = dict(candidates.get("_proposal_debug", {}))
    # Fail scene-locally before critical/natural/response work if the proposal
    # generator produced no valid intervention.  This preserves valid-scene
    # semantics while avoiding expensive downstream work on a doomed probe row.
    ego_neutral_fallback = _make_ego_neutral(candidates, scene.scenario_id)
    critical = _timeit("engine_critical_agents_s", lambda: select_critical_agents(scene, cfg, candidates, conflict_regions=regions))
    if profile_diagnostics is not None:
        profile_diagnostics["critical"] = {
            "selection_reference_mode": str(cfg.get("critical", {}).get("selection_reference_mode", "fixed_anchor_v1")),
            "count": int(np.asarray(critical.get("valid", []), dtype=bool).sum()),
            "track_indices": [int(x) for x in np.asarray(critical.get("track_index", []), dtype=np.int32)[np.asarray(critical.get("valid", []), dtype=bool)].tolist()],
            **dict(critical.get("_selection_diagnostics", {})),
        }
    # A single global stopping trajectory is not a valid neutral intervention for
    # every actor (e.g. it can itself pressure a rear follower).  Build one fixed,
    # proposal-bank-independent pressure-removing neutral per critical pair.
    ego_neutral, neutral_diag = _timeit(
        "engine_pair_neutral_s",
        lambda: build_pair_specific_ego_neutrals(scene, critical, ego_neutral_fallback, cfg),
    )
    if profile_diagnostics is not None:
        profile_diagnostics["pair_neutral"] = neutral_diag
    natural = _timeit("engine_natural_s", lambda: generate_natural_alternatives(scene, critical, ego_neutral, cfg, ablation=ablation))
    if profile_diagnostics is not None:
        profile_diagnostics["natural"] = list(natural.get("_diagnostics", []))
        # Natural generation performs the final routability audit using the exact
        # route builder.  Refresh critical coverage diagnostics so the profile does
        # not report the cheaper current-lane precheck as the final certificate
        # support state.
        selected_now = np.asarray(critical.get("valid", []), dtype=bool)
        mechanism_now = np.asarray(critical.get("mechanism_valid", selected_now), dtype=bool) & selected_now
        profile_diagnostics.setdefault("critical", {})["mechanism_auditable_count_final"] = int(mechanism_now.sum())
        profile_diagnostics.setdefault("critical", {})["mechanism_unauditable_count_final"] = int((selected_now & ~mechanism_now).sum())
    audit = _timeit("engine_audit_relevance_s", lambda: compute_candidate_agent_audit(scene, candidates, critical, natural, cfg))
    selected_critical_mask = np.asarray(critical["valid"], dtype=bool)
    mechanism_critical_mask = np.asarray(critical.get("mechanism_valid", selected_critical_mask), dtype=bool) & selected_critical_mask
    certificate_complete = bool(np.all((~selected_critical_mask) | mechanism_critical_mask))
    certificate_valid = np.asarray(candidates["valid"], dtype=bool) & certificate_complete
    if profile_diagnostics is not None:
        valid_pair = np.asarray(candidates["valid"], dtype=bool)[:, None] & mechanism_critical_mask[None, :]
        rel = np.asarray(audit["pair_relevant"], dtype=bool) & valid_pair
        profile_diagnostics["audit"] = {
            "valid_pair_count": int(valid_pair.sum()),
            "relevant_pair_count": int(rel.sum()),
            "relevant_pair_rate": float(rel.sum() / max(int(valid_pair.sum()), 1)),
            "mean_relevance_mass": float(np.asarray(audit["relevance_mass"], dtype=np.float32)[valid_pair].mean()) if np.any(valid_pair) else 0.0,
        }
    response = _timeit("engine_safe_responses_s", lambda: generate_safe_responses(scene, candidates, critical, natural, cfg, audit=audit))
    witness = _timeit("engine_witness_s", lambda: certify_witnesses(scene, candidates, critical, natural, response, cfg, ablation=ablation, conflict_regions=regions, audit=audit))
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
        "dataset/mechanism_certificate_complete": np.asarray(certificate_complete, dtype=bool),
        "cowp/candidates/trajectory": candidates["trajectory"],
        "cowp/candidates/macro_type": candidates["macro_type"],
        "cowp/candidates/valid": candidates["valid"],
        "cowp/candidates/certificate_valid": certificate_valid,
        "cowp/candidates/conventional_safe": witness["candidate_conventional_safe"],
        # Candidate-level NCF/false-safe is unknown when any selected critical
        # relation lacks an auditable natural basis.  Store a conservative false
        # value and expose certificate_valid as the explicit supervision/eval mask.
        "cowp/candidates/false_safe": np.asarray(witness["candidate_false_safe"], dtype=bool) & certificate_valid,
        "cowp/candidates/noncoercive_feasible": np.asarray(witness["candidate_noncoercive_feasible"], dtype=bool) & certificate_valid,
        "cowp/candidates/ego_utility_prior": candidates["ego_utility_prior"],
        "cowp/candidates/is_logged": candidates["is_logged"],
        "cowp/candidates/is_neutral": candidates["is_neutral"],
        "cowp/candidates/topology_id": candidates["topology_id"],
        "cowp/candidates/proposal_source": candidates.get("proposal_source", np.zeros_like(candidates["macro_type"], dtype=np.int32)),
        "cowp/candidates/proposal_region_id": candidates.get("proposal_region_id", np.full_like(candidates["macro_type"], -1, dtype=np.int32)),
        "cowp/candidates/proposal_target_time_s": candidates.get("proposal_target_time_s", np.full_like(candidates["ego_utility_prior"], -1.0, dtype=np.float32)),
        "cowp/candidates/proposal_timing_side": candidates.get("proposal_timing_side", np.zeros_like(candidates["macro_type"], dtype=np.int8)),
        "cowp/candidates/proposal_target_agent_index": candidates.get("proposal_target_agent_index", np.full_like(candidates["macro_type"], -1, dtype=np.int32)),
        "cowp/candidates/proposal_gap_s": candidates.get("proposal_gap_s", np.full_like(candidates["ego_utility_prior"], -1.0, dtype=np.float32)),
        "cowp/candidates/proposal_accel_mps2": candidates.get("proposal_accel_mps2", np.full_like(candidates["ego_utility_prior"], -99.0, dtype=np.float32)),
        "cowp/candidates/proposal_entry_distance_m": candidates.get("proposal_entry_distance_m", np.full_like(candidates["ego_utility_prior"], -1.0, dtype=np.float32)),
        "cowp/candidates/proposal_target_tta_error_s": candidates.get("proposal_target_tta_error_s", np.full_like(candidates["ego_utility_prior"], -1.0, dtype=np.float32)),
        "cowp/critical/track_index": critical["track_index"],
        "cowp/critical/track_id": np.asarray([scene.track_id[i] if 0 <= i < len(scene.track_id) else -1 for i in critical["track_index"]], dtype=np.int64),
        "cowp/critical/valid": critical["valid"],
        "cowp/critical/mechanism_valid": mechanism_critical_mask,
        "cowp/critical/auditability_reason": critical.get("auditability_reason", np.full_like(critical["track_index"], 3, dtype=np.int32)),
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
        "cowp/natural/obs_contamination": natural["obs_contamination"],
        "cowp/natural/map_compliant": natural["map_compliant"],
        "cowp/natural/map_distance_max": natural["map_distance_max"],
        "cowp/natural/map_verified": natural["map_verified"],
        "cowp/natural/map_evidence_mode": natural.get("map_evidence_mode", np.zeros_like(natural["valid"], dtype=np.int8)),
        "cowp/audit/pair_relevant": audit["pair_relevant"],
        "cowp/audit/relevance_mass": audit["relevance_mass"],
        "cowp/audit/root_affected": audit["root_affected"],
        "cowp/audit/root_unsafe": audit["root_unsafe"],
        "cowp/audit/root_event_interval": audit["root_event_interval"],
        "cowp/audit/root_direct_burden": audit["root_direct_burden"],
        "cowp/audit/root_budget_crossed": audit["root_budget_crossed"],
        "cowp/audit/root_burden_only_affected": audit["root_burden_only_affected"],
        "cowp/audit/canonical_root_weight": audit["canonical_root_weight"],
        "cowp/response/traj": response["traj"],
        "cowp/response/valid": response["valid"],
        "cowp/response/source": response["source"],
        "cowp/response/root_index": response.get("root_index", np.full_like(response["source"], -1, dtype=np.int32)),
        "cowp/response/root_affinity": response.get("root_affinity", np.zeros_like(response["burden_total"], dtype=np.float32)),
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
        "cowp/witness/causal_relevance_mass": witness["causal_relevance_mass"],
        "cowp/witness/natural_conflict_mass_by_source": witness["natural_conflict_mass_by_source"],
        "cowp/witness/natural_mass_by_source": witness["natural_mass_by_source"],
        "cowp/witness/low_safe_mass_by_source": witness["low_safe_mass_by_source"],
        "cowp/witness/opr": witness["opr"],
        "cowp/witness/c_i": witness["c_i"],
        "cowp/witness/tail_burden_excess": witness["tail_burden_excess"],
        "cowp/witness/root_min_safe_burden": witness["root_min_safe_burden"],
        "cowp/witness/conflict_interval": witness["conflict_interval"],
        "cowp/witness/conflict_region_id": witness["conflict_region_id"],
        "cowp/witness/critical_agent_track_index": witness["critical_agent_track_index"],
        "cowp/witness/rho": witness["rho"],
        "cowp/witness/pair_noncoercive_feasible": witness["pair_noncoercive_feasible"],
        "cowp/witness/blocker_code": witness["blocker_code"],
        "cowp/candidates/audited_pair_count": witness["candidate_audited_pair_count"],
        "cowp/candidates/ncf_blocker_count": witness["candidate_ncf_blocker_count"],
        "cowp/candidates/min_audited_opr": witness["candidate_min_audited_opr"],
        "cowp/candidates/max_audited_tail_burden_excess": witness["candidate_max_audited_tail_burden_excess"],
        "cowp/transport/mode_valid": witness["transport_mode_valid"],
        "cowp/transport/mode_conflict": witness["transport_mode_conflict"],
        "cowp/transport/mode_affected": witness["transport_mode_affected"],
        "cowp/transport/mode_retained_low_safe": witness["transport_mode_retained_low_safe"],
        "cowp/transport/response_root_index": witness["transport_response_root_index"],
        "cowp/transport/response_is_min_burden": witness["transport_response_is_min_burden"],
        "cowp/transport/root_recovery_mass": witness["transport_root_recovery_mass"],
        "cowp/transport/root_low_safe_score": witness["transport_root_low_safe_score"],
        "cowp/transport/root_target_confidence": witness["transport_root_target_confidence"],
        "cowp/transport/root_min_safe_burden": witness["root_min_safe_burden"],
        "cowp/transport/transported_opr": witness["transport_transported_opr"],
        "cowp/transport/canonical_root_weight": witness["transport_canonical_root_weight"],
        "map/conflict_regions": conflict_tensor,
        "map/conflict_region_valid": conflict_valid,
    }
    return out
