from __future__ import annotations

import numpy as np

from cowp.core.constants import MechanismToken, PriorityRelation
from cowp.core.types import ScenarioData, future_states_to_traj7
from cowp.geometry.collision import conventional_candidate_safe, unsafe_between
from cowp.geometry.lane_graph import build_conflict_regions, closest_conflict_for_pair
from cowp.label.burden import burden_total as weighted_burden_total
from cowp.label.burden import compute_burden


def _event_interval(mask: np.ndarray) -> tuple[int, int]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return -1, -1
    return int(idx[0]), int(idx[-1])


def _mechanism_token(comps: np.ndarray, opr: float, rho: PriorityRelation, cfg: dict) -> MechanismToken:
    ncf_cfg = cfg.get("ncf", {})
    if comps[0] >= 0.8:
        return MechanismToken.HB
    if rho == PriorityRelation.AGENT_PRIORITY and comps[5] >= 0.8:
        return MechanismToken.PA
    if comps[5] >= 0.8:
        return MechanismToken.GS
    if comps[2] >= 0.6:
        return MechanismToken.AY
    if comps[3] >= 0.6:
        return MechanismToken.SR
    if opr < float(ncf_cfg.get("alpha_opr", 0.35)):
        return MechanismToken.OR
    idx = int(np.argmax(comps))
    return [MechanismToken.HB, MechanismToken.HB, MechanismToken.AY, MechanismToken.SR, MechanismToken.OR, MechanismToken.PA][idx]


def certify_witnesses(scene: ScenarioData, candidates: dict[str, np.ndarray], critical: dict[str, np.ndarray], natural: dict[str, np.ndarray], response: dict[str, np.ndarray], cfg: dict, ablation: dict | None = None) -> dict[str, np.ndarray]:
    ablation = ablation or {}
    use_option = bool(ablation.get("use_option_preservation", True))
    limits = cfg.get("limits", {})
    K = int(limits.get("max_candidates", 64))
    A = int(limits.get("max_critical_agents", 8))
    exists = np.zeros((K, A), dtype=bool)
    token = np.zeros((K, A), dtype=np.int32)
    burden_total = np.zeros((K, A), dtype=np.float32)
    burden_components = np.zeros((K, A, 6), dtype=np.float32)
    min_safe_burden = np.full((K, A), np.inf, dtype=np.float32)
    natural_conflict_mass = np.zeros((K, A), dtype=np.float32)
    opr = np.ones((K, A), dtype=np.float32)
    c_i = np.zeros((K, A), dtype=np.float32)
    conflict_interval = np.full((K, A, 2), -1, dtype=np.int32)
    conflict_region_id = np.full((K, A), -1, dtype=np.int32)
    rho_arr = np.zeros((K, A), dtype=np.int32)
    conventional_safe = np.zeros(K, dtype=bool)
    false_safe = np.zeros(K, dtype=bool)
    ncf = np.zeros(K, dtype=bool)

    cur = scene.current_time_index
    regions = build_conflict_regions(scene.map_data, cfg)
    other_logged = []
    for j in range(scene.num_agents):
        if j == scene.sdc_track_index:
            continue
        fut = scene.states[j, cur + 1 : cur + 1 + int(cfg.get("time", {}).get("future_steps", 80)), :]
        if len(fut) and np.any(fut[:, 10] > 0.5):
            other_logged.append(future_states_to_traj7(fut, int(cfg.get("time", {}).get("future_steps", 80)), current_state=scene.states[j, cur]))
    for k in range(K):
        if not candidates["valid"][k]:
            continue
        ego = candidates["trajectory"][k]
        conventional_safe[k] = conventional_candidate_safe(ego, other_logged, cfg)
        all_ncf = conventional_safe[k]
        for a in range(A):
            if not critical["valid"][a]:
                continue
            idx = int(critical["track_index"][a])
            object_type = int(scene.object_type[idx])
            rho = PriorityRelation(int(critical.get("base_priority", np.zeros(A, dtype=np.int32))[a]))
            rho_arr[k, a] = int(rho)
            beta = float(natural.get("beta", np.full(A, 0.65))[a])
            nat_valid_idx = np.where(natural["valid"][a])[0]
            conflict_mass = 0.0
            low_safe_mass = 0.0
            interval_masks = []
            for m in nat_valid_idx:
                nat = natural["traj"][a, m]
                w = float(natural["weight"][a, m])
                if w < float(cfg.get("ncf", {}).get("min_alt_weight", 0.03)):
                    continue
                low_neu = float(natural["burden_neutral"][a, m]) <= beta
                unsafe = unsafe_between(ego, nat, cfg, agent_type=object_type)
                b_under, _ = compute_burden(nat, ego, cfg, object_type, natural_ref=nat)
                if low_neu and unsafe.unsafe:
                    conflict_mass += w
                    interval_masks.append(unsafe.event_mask)
                if low_neu and (not unsafe.unsafe) and b_under <= beta:
                    low_safe_mass += w
            natural_conflict_mass[k, a] = conflict_mass
            opr[k, a] = low_safe_mass if use_option else 1.0
            option_loss = max(0.0, 1.0 - float(opr[k, a])) if use_option else 0.0
            resp_mask = response["valid"][k, a] & response["is_safe"][k, a]
            if np.any(resp_mask):
                bvals = response["burden_total"][k, a]
                safe_indices = np.where(resp_mask)[0]
                adjusted = []
                for ridx in safe_indices:
                    comps = response["burden_components"][k, a, ridx].copy()
                    comps[4] = max(float(comps[4]), option_loss)
                    adjusted.append(weighted_burden_total(comps, cfg))
                best_local = int(np.argmin(np.asarray(adjusted, dtype=np.float32)))
                r_idx = int(safe_indices[best_local])
                min_safe_burden[k, a] = float(adjusted[best_local])
                burden_total[k, a] = float(adjusted[best_local])
                burden_components[k, a] = response["burden_components"][k, a, r_idx]
                burden_components[k, a, 4] = max(float(burden_components[k, a, 4]), option_loss)
            else:
                min_safe_burden[k, a] = np.inf
                burden_total[k, a] = 2.0
                burden_components[k, a, 3] = 2.0
            min_nat = float(np.min(natural["burden_neutral"][a, nat_valid_idx])) if len(nat_valid_idx) else 0.0
            c_i[k, a] = float(min_safe_burden[k, a] - min_nat) if np.isfinite(min_safe_burden[k, a]) else 2.0
            option_collapsed = use_option and opr[k, a] < float(cfg.get("ncf", {}).get("alpha_opr", 0.35))
            positive = conflict_mass >= float(cfg.get("ncf", {}).get("positive_min_natural_conflict_mass", 0.10)) and (min_safe_burden[k, a] > beta or option_collapsed)
            exists[k, a] = bool(positive)
            if positive:
                token[k, a] = int(_mechanism_token(burden_components[k, a], float(opr[k, a]), rho, cfg))
                if interval_masks:
                    merged = np.zeros(max(len(x) for x in interval_masks), dtype=bool)
                    for mask in interval_masks:
                        merged[: len(mask)] |= mask
                    conflict_interval[k, a] = np.asarray(_event_interval(merged), dtype=np.int32)
                if regions:
                    agent_nat = natural["traj"][a, nat_valid_idx[0]] if len(nat_valid_idx) else None
                    if agent_nat is not None:
                        region, _, _, _ = closest_conflict_for_pair(ego, agent_nat, regions, dt=float(cfg.get("time", {}).get("dt", 0.1)))
                        if region is not None:
                            conflict_region_id[k, a] = int(region.conflict_id)
            else:
                token[k, a] = int(MechanismToken.NONE)
            pair_ncf = min_safe_burden[k, a] <= beta + float(cfg.get("ncf", {}).get("gamma", 0.10)) and opr[k, a] >= float(cfg.get("ncf", {}).get("alpha_opr", 0.35)) and not positive
            all_ncf = all_ncf and pair_ncf
        false_safe[k] = conventional_safe[k] and bool(np.any(exists[k] & critical["valid"]))
        ncf[k] = bool(all_ncf)
    return {
        "exists": exists,
        "token": token,
        "burden_total": burden_total,
        "burden_components": burden_components,
        "min_safe_burden": min_safe_burden,
        "natural_conflict_mass": natural_conflict_mass,
        "opr": opr,
        "c_i": c_i,
        "conflict_interval": conflict_interval,
        "conflict_region_id": conflict_region_id,
        "critical_agent_track_index": critical["track_index"].astype(np.int32),
        "rho": rho_arr,
        "candidate_conventional_safe": conventional_safe,
        "candidate_false_safe": false_safe,
        "candidate_noncoercive_feasible": ncf,
    }
