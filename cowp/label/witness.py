from __future__ import annotations

import numpy as np

from cowp.core.constants import MechanismToken, NaturalSource, PriorityRelation
from cowp.core.types import ScenarioData, future_states_to_traj7
from cowp.geometry.collision import conventional_candidate_safe, unsafe_between
from cowp.geometry.lane_graph import build_conflict_regions, closest_conflict_for_pair
from cowp.label.burden import burden_total as weighted_burden_total
from cowp.label.burden import compute_burden
from cowp.label.safe_responses import root_conditioned_recovery_search


def _event_interval(mask: np.ndarray) -> tuple[int, int]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return -1, -1
    return int(idx[0]), int(idx[-1])


def _weighted_upper_cvar(values: np.ndarray, weights: np.ndarray, tail_mass: float) -> float:
    """Exact finite-support upper-tail CVaR used by the manuscript certificate."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    use = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(use):
        return 0.0
    v = v[use]
    w = w[use]
    w = w / max(float(np.sum(w)), 1.0e-12)
    q = float(np.clip(tail_mass, 1.0e-3, 1.0))
    order = np.argsort(-v)
    v = v[order]
    w = w[order]
    before = np.cumsum(w) - w
    take = np.minimum(w, np.maximum(q - before, 0.0))
    denom = float(np.sum(take))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.sum(take * v) / denom)



def _root_affinity(response_traj: np.ndarray, natural_roots: np.ndarray, cfg: dict) -> np.ndarray:
    """Soft multi-horizon affinity used only when a response lacks explicit identity.

    A single full-horizon ADE argmin creates brittle one-hot labels and swaps roots
    that share an endpoint.  This affinity uses the same 1/3/5/8 s semantics as the
    learned root alignment and exposes low-confidence coverage instead of treating
    an uncovered root as a confirmed negative.
    """
    roots = np.asarray(natural_roots, dtype=np.float32)
    response = np.asarray(response_traj, dtype=np.float32)
    if roots.ndim != 3 or len(roots) == 0:
        return np.zeros((0,), dtype=np.float32)
    total = min(int(response.shape[0]), int(roots.shape[1]))
    if total <= 0:
        return np.full((len(roots),), 1.0 / max(len(roots), 1), dtype=np.float32)
    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1.0e-4)
    horizons_s = cfg.get("response", {}).get("root_assignment_horizons_s", [1.0, 3.0, 5.0, 8.0])
    costs = np.zeros((len(roots),), dtype=np.float64)
    used = 0
    distance = np.linalg.norm(roots[:, :total, :2] - response[None, :total, :2], axis=-1)
    for seconds in horizons_s:
        steps = max(1, min(total, int(round(float(seconds) / dt))))
        costs += distance[:, :steps].mean(axis=-1)
        used += 1
    costs /= max(used, 1)
    temperature = max(float(cfg.get("response", {}).get("root_assignment_temperature_m", 2.0)), 1.0e-3)
    logits = -(costs - float(np.min(costs))) / temperature
    affinity = np.exp(np.clip(logits, -40.0, 0.0))
    affinity /= max(float(np.sum(affinity)), 1.0e-12)
    return affinity.astype(np.float32)


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


def certify_witnesses(
    scene: ScenarioData,
    candidates: dict[str, np.ndarray],
    critical: dict[str, np.ndarray],
    natural: dict[str, np.ndarray],
    response: dict[str, np.ndarray],
    cfg: dict,
    ablation: dict | None = None,
    conflict_regions: list | None = None,
) -> dict[str, np.ndarray]:
    ablation = ablation or {}
    use_option = bool(ablation.get("use_option_preservation", True))
    limits = cfg.get("limits", {})
    K = int(limits.get("max_candidates", 64))
    A = int(limits.get("max_critical_agents", 8))
    M = int(limits.get("max_natural_alternatives", natural["valid"].shape[-1]))
    R = int(limits.get("max_safe_responses", response["valid"].shape[-1]))
    exists = np.zeros((K, A), dtype=bool)
    token = np.zeros((K, A), dtype=np.int32)
    burden_total = np.zeros((K, A), dtype=np.float32)
    burden_components = np.zeros((K, A, 6), dtype=np.float32)
    min_safe_burden = np.full((K, A), np.inf, dtype=np.float32)
    natural_conflict_mass = np.zeros((K, A), dtype=np.float32)
    natural_conflict_mass_by_source = np.zeros((K, A, 4), dtype=np.float32)
    natural_mass_by_source = np.zeros((K, A, 4), dtype=np.float32)
    low_safe_mass_by_source = np.zeros((K, A, 4), dtype=np.float32)
    opr = np.ones((K, A), dtype=np.float32)
    c_i = np.zeros((K, A), dtype=np.float32)
    tail_burden_excess = np.zeros((K, A), dtype=np.float32)
    root_min_safe_burden = np.full((K, A, M), 2.0, dtype=np.float32)
    conflict_interval = np.full((K, A, 2), -1, dtype=np.int32)
    conflict_region_id = np.full((K, A), -1, dtype=np.int32)
    rho_arr = np.zeros((K, A), dtype=np.int32)
    conventional_safe = np.zeros(K, dtype=bool)
    false_safe = np.zeros(K, dtype=bool)
    ncf = np.zeros(K, dtype=bool)
    mode_valid = np.zeros((K, A, M), dtype=bool)
    mode_conflict = np.zeros((K, A, M), dtype=bool)
    mode_retained_low_safe = np.zeros((K, A, M), dtype=bool)
    response_root_index = np.full((K, A, R), -1, dtype=np.int32)
    response_is_min_burden = np.zeros((K, A, R), dtype=bool)
    root_recovery_mass = np.zeros((K, A), dtype=np.float32)
    root_low_safe_score = np.zeros((K, A, M), dtype=np.float32)
    root_target_confidence = np.zeros((K, A, M), dtype=np.float32)
    transported_opr = np.ones((K, A), dtype=np.float32)

    cur = scene.current_time_index
    regions = conflict_regions if conflict_regions is not None else build_conflict_regions(scene.map_data, cfg)
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
            root_weight = np.zeros(M, dtype=np.float32)
            if len(nat_valid_idx):
                raw_weight = np.asarray(natural["weight"][a, nat_valid_idx], dtype=np.float32)
                raw_weight = np.maximum(raw_weight, 0.0)
                raw_weight = raw_weight / max(float(np.sum(raw_weight)), 1e-8)
                eps_p = float(np.clip(cfg.get("ncf", {}).get("root_probability_floor", 0.02), 0.0, 0.25))
                root_weight[nat_valid_idx] = (1.0 - eps_p) * raw_weight + eps_p / len(nat_valid_idx)
            conflict_mass = 0.0
            low_safe_mass = 0.0
            low_natural_mass = 0.0
            interval_masks = []
            for m in nat_valid_idx:
                nat = natural["traj"][a, m]
                w = float(root_weight[m])
                low_neu = float(natural["burden_neutral"][a, m]) <= beta
                unsafe = unsafe_between(ego, nat, cfg, agent_type=object_type)
                b_under, _ = compute_burden(nat, ego, cfg, object_type, natural_ref=nat, rho=rho)
                if m < M and low_neu:
                    mode_valid[k, a, m] = True
                    mode_conflict[k, a, m] = bool(unsafe.unsafe)
                    mode_retained_low_safe[k, a, m] = bool((not unsafe.unsafe) and b_under <= beta)
                src = int(natural.get("source", np.full(natural["valid"].shape, int(NaturalSource.PAD), dtype=np.int32))[a, m])
                src = src if 0 <= src < 4 else int(NaturalSource.PAD)
                if low_neu:
                    low_natural_mass += w
                    natural_mass_by_source[k, a, src] += w
                if low_neu and unsafe.unsafe:
                    conflict_mass += w
                    natural_conflict_mass_by_source[k, a, src] += w
                    interval_masks.append(unsafe.event_mask)
                if low_neu and (not unsafe.unsafe) and b_under <= beta:
                    low_safe_mass += w
                    low_safe_mass_by_source[k, a, src] += w
            natural_conflict_mass[k, a] = conflict_mass
            # Eq. (option_preservation): OPR is retained floor-smoothed root
            # probability mass, not a conditional ratio over only low-burden
            # roots.  High-burden/non-retained roots therefore contribute zero.
            opr[k, a] = float(np.clip(low_safe_mass, 0.0, 1.0)) if use_option else 1.0

            # Same-root response supervision.  Fresh v16.8 labels use the explicit
            # root identity emitted by the response generator.  Legacy/global
            # responses fall back to a soft multi-horizon affinity; low-confidence
            # coverage is recorded separately and is not forced into a hard negative.
            valid_roots = np.where(natural["valid"][a])[0]
            # Candidate-conditioned root oracle: q_ikm is evaluated with an equal
            # root-preserving control budget before considering the compact/global
            # response slots.  This removes the top-R coverage bias from the core
            # coercion target while retaining response slots for auxiliary decoding.
            if bool(cfg.get("response", {}).get("root_conditioned_transport", {}).get("enabled", True)):
                for root in valid_roots:
                    root = int(root)
                    if not (mode_valid[k, a, root] and mode_conflict[k, a, root]):
                        continue
                    best_b, low_ok, _ = root_conditioned_recovery_search(
                        natural["traj"][a, root], ego, cfg,
                        object_type=object_type, beta=beta, rho=rho,
                    )
                    root_target_confidence[k, a, root] = 1.0
                    root_min_safe_burden[k, a, root] = min(
                        float(root_min_safe_burden[k, a, root]), float(best_b)
                    )
                    if low_ok:
                        root_low_safe_score[k, a, root] = 1.0

            valid_resp = np.where(response["valid"][k, a])[0]
            response_primitive_burden: dict[int, float] = {}
            explicit_root = response.get("root_index")
            explicit_affinity = response.get("root_affinity")
            min_affinity = float(cfg.get("response", {}).get("root_assignment_min_affinity", 0.35))
            for ridx in valid_resp:
                comps = np.asarray(response["burden_components"][k, a, ridx], dtype=np.float32).copy()
                # OPR is constrained separately; B_prim contains physical and
                # normative response burden only (manuscript Appendix A).
                if comps.shape[0] > 4:
                    comps[4] = 0.0
                response_primitive_burden[int(ridx)] = float(weighted_burden_total(comps, cfg))

            if len(valid_roots):
                nat_roots = natural["traj"][a, valid_roots]
                for ridx in valid_resp:
                    affin = np.zeros(M, dtype=np.float32)
                    explicit = int(explicit_root[k, a, ridx]) if explicit_root is not None else -1
                    if explicit in set(valid_roots.tolist()):
                        conf = 1.0
                        if explicit_affinity is not None:
                            conf = float(np.clip(explicit_affinity[k, a, ridx], 0.0, 1.0))
                        affin[explicit] = max(conf, min_affinity)
                    else:
                        local = _root_affinity(response["traj"][k, a, ridx], nat_roots, cfg)
                        affin[valid_roots] = local
                    root = int(np.argmax(affin))
                    response_root_index[k, a, ridx] = root
                    root_target_confidence[k, a] = np.maximum(root_target_confidence[k, a], affin)
                    is_low = response_primitive_burden[int(ridx)] <= beta
                    if response["is_safe"][k, a, ridx] and is_low:
                        root_low_safe_score[k, a] = np.maximum(root_low_safe_score[k, a], affin)

                for root in valid_roots:
                    rset = [
                        int(r) for r in valid_resp
                        if int(response_root_index[k, a, r]) == int(root)
                        and bool(response["is_safe"][k, a, r])
                        and float(root_target_confidence[k, a, root]) >= min_affinity
                    ]
                    if rset:
                        best = min(rset, key=lambda r: response_primitive_burden[r])
                        response_is_min_burden[k, a, best] = True
                        root_min_safe_burden[k, a, root] = response_primitive_burden[best]

            # Eq. (option_preservation) must include transported conflict roots:
            # s=(1-c)r+cq.  v16.7 used only the non-conflict term and therefore
            # labelled every conflicting natural root as lost even when a same-root
            # low-burden safe response existed.
            recovered = np.clip(root_low_safe_score[k, a], 0.0, 1.0)
            transported_root = mode_retained_low_safe[k, a].astype(np.float32)
            transported_root = np.where(mode_conflict[k, a], recovered, transported_root)
            transported_opr[k, a] = float(np.clip(np.sum(root_weight * transported_root), 0.0, 1.0))
            if use_option:
                opr[k, a] = transported_opr[k, a]
            for root in valid_roots:
                if mode_conflict[k, a, root] and recovered[root] > 0.0:
                    src = int(natural.get("source", np.full(natural["valid"].shape, int(NaturalSource.PAD), dtype=np.int32))[a, root])
                    src = src if 0 <= src < 4 else int(NaturalSource.PAD)
                    low_safe_mass_by_source[k, a, src] += float(root_weight[root] * recovered[root])

            # Both OPR and tail burden use the same floor-smoothed natural-root
            # measure.  CVaR is conditioned on roots that conflict with ego and
            # measures excess over each agent's adaptive burden budget beta.
            conflict_weight = root_weight * mode_valid[k, a] * mode_conflict[k, a]
            denom = float(np.sum(conflict_weight))
            if denom > 1e-8:
                root_recovery_mass[k, a] = float(np.sum(conflict_weight * recovered) / denom)
            root_excess = np.maximum(root_min_safe_burden[k, a] - beta, 0.0)
            tail_burden_excess[k, a] = _weighted_upper_cvar(
                root_excess,
                conflict_weight,
                float(cfg.get("ncf", {}).get("cvar_tail_mass", 0.25)),
            )

            resp_mask = response["valid"][k, a] & response["is_safe"][k, a]
            if np.any(resp_mask):
                safe_indices = np.where(resp_mask)[0]
                adjusted = np.asarray(
                    [response_primitive_burden[int(r)] for r in safe_indices], dtype=np.float32
                )
                best_local = int(np.argmin(adjusted))
                r_idx = int(safe_indices[best_local])
                min_safe_burden[k, a] = float(adjusted[best_local])
                burden_total[k, a] = float(adjusted[best_local])
                burden_components[k, a] = response["burden_components"][k, a, r_idx]
                if burden_components.shape[-1] > 4:
                    burden_components[k, a, 4] = 0.0
            else:
                min_safe_burden[k, a] = np.inf
                burden_total[k, a] = 2.0
                burden_components[k, a, 3] = 2.0
            c_i[k, a] = tail_burden_excess[k, a]
            option_collapsed = use_option and opr[k, a] < float(cfg.get("ncf", {}).get("alpha_opr", 0.35))
            gamma = float(cfg.get("ncf", {}).get("gamma", 0.10))
            positive = (
                conflict_mass > float(cfg.get("ncf", {}).get("positive_min_natural_conflict_mass", 0.10))
                and (tail_burden_excess[k, a] > gamma or option_collapsed)
            )
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
            pair_ncf = (
                tail_burden_excess[k, a] <= gamma
                and opr[k, a] >= float(cfg.get("ncf", {}).get("alpha_opr", 0.35))
                and not positive
            )
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
        "natural_conflict_mass_by_source": natural_conflict_mass_by_source,
        "natural_mass_by_source": natural_mass_by_source,
        "low_safe_mass_by_source": low_safe_mass_by_source,
        "opr": opr,
        "c_i": c_i,
        "tail_burden_excess": tail_burden_excess,
        "root_min_safe_burden": root_min_safe_burden,
        "conflict_interval": conflict_interval,
        "conflict_region_id": conflict_region_id,
        "critical_agent_track_index": critical["track_index"].astype(np.int32),
        "rho": rho_arr,
        "candidate_conventional_safe": conventional_safe,
        "candidate_false_safe": false_safe,
        "candidate_noncoercive_feasible": ncf,
        "transport_mode_valid": mode_valid,
        "transport_mode_conflict": mode_conflict,
        "transport_mode_retained_low_safe": mode_retained_low_safe,
        "transport_response_root_index": response_root_index,
        "transport_response_is_min_burden": response_is_min_burden,
        "transport_root_recovery_mass": root_recovery_mass,
        "transport_root_low_safe_score": root_low_safe_score,
        "transport_root_target_confidence": root_target_confidence,
        "transport_transported_opr": transported_opr,
    }
