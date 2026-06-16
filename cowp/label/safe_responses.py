from __future__ import annotations

import numpy as np

from cowp.core.constants import ResponseSource
from cowp.core.types import ScenarioData
from cowp.geometry.collision import unsafe_between
from cowp.label.burden import compute_burden
from cowp.label.trajectory_primitives import constant_accel_trajectory, resample_logged
from cowp.label.safe_budget_search import typed_safe_budget_search


def _subsample_by_source(
    primitives: list[tuple[np.ndarray, ResponseSource]],
    cfg: dict,
) -> list[tuple[np.ndarray, ResponseSource]]:
    """Keep a bounded but source-balanced response primitive bank.

    Response generation dominates label-engine time because every primitive is
    checked against every ego candidate and critical agent.  The output tensor
    stores only max_safe_responses responses, so evaluating 100+ primitives per
    pair has diminishing returns.  Keep observed/natural responses, a small hard
    emergency set, and a uniformly sampled optimized lattice.
    """
    resp_cfg = cfg.get("response", {})
    max_total = int(resp_cfg.get("max_response_primitives_per_agent", 48))
    if max_total <= 0 or len(primitives) <= max_total:
        return primitives

    pred = [p for p in primitives if p[1] == ResponseSource.PRED]
    opt = [p for p in primitives if p[1] == ResponseSource.OPT]
    emg = [p for p in primitives if p[1] == ResponseSource.EMG]

    max_pred = int(resp_cfg.get("max_predicted_responses_per_agent", 8))
    max_emg = int(resp_cfg.get("max_emergency_responses_per_agent", 4))
    kept: list[tuple[np.ndarray, ResponseSource]] = []
    kept.extend(pred[: max(0, min(max_pred, max_total))])

    emg_keep = emg[: max(0, min(max_emg, max_total - len(kept)))]
    opt_budget = max_total - len(kept) - len(emg_keep)
    if opt_budget > 0 and opt:
        if len(opt) <= opt_budget:
            kept.extend(opt)
        else:
            # Uniformly cover acceleration / delay / duration choices rather than
            # taking only the first part of the nested loops.
            idx = np.linspace(0, len(opt) - 1, opt_budget, dtype=np.int32)
            used: set[int] = set()
            for j in idx.tolist():
                if j not in used:
                    kept.append(opt[j])
                    used.add(j)
            j = 0
            while len(kept) < max_total - len(emg_keep) and j < len(opt):
                if j not in used:
                    kept.append(opt[j])
                    used.add(j)
                j += 1
    kept.extend(emg_keep)
    return kept[:max_total]


def _response_primitives_for_agent(
    scene: ScenarioData,
    agent_slot: int,
    critical: dict[str, np.ndarray],
    natural: dict[str, np.ndarray],
    cfg: dict,
) -> tuple[int, int, np.ndarray, list[tuple[np.ndarray, ResponseSource]]]:
    """Build response primitives once per critical agent.

    These trajectories do not depend on the ego candidate; only their safety and
    burden do.  The old implementation rebuilt the same primitive bank inside
    every candidate loop, multiplying trajectory-generation cost by K.
    """
    limits = cfg.get("limits", {})
    resp_cfg = cfg.get("response", {})
    R = int(limits.get("max_safe_responses", 32))
    H = int(cfg.get("time", {}).get("future_steps", 80))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    cur = scene.current_time_index
    idx = int(critical["track_index"][agent_slot])
    object_type = int(scene.object_type[idx])
    curr = scene.states[idx, cur]
    nat_ref_candidates = natural["traj"][agent_slot, natural["valid"][agent_slot]]
    nat_ref = nat_ref_candidates[0] if len(nat_ref_candidates) else constant_accel_trajectory(curr, H, dt, accel=0.0)
    primitives: list[tuple[np.ndarray, ResponseSource]] = []
    for m in np.where(natural["valid"][agent_slot])[0][: min(8, R)]:
        primitives.append((natural["traj"][agent_slot, m], ResponseSource.PRED))
    for acc in resp_cfg.get("response_acc_values_mps2", [-4.5, -3.5, -2.5, -1.5, -0.5, 0.0, 0.5, 1.0]):
        for delay in resp_cfg.get("response_start_delay_s", [0.0, 0.3, 0.6, 1.0]):
            for dur in resp_cfg.get("response_duration_s", [1.0, 2.0, 3.0]):
                primitives.append((constant_accel_trajectory(curr, H, dt, accel=float(acc), start_delay_s=float(delay), duration_s=float(dur)), ResponseSource.OPT))
    for decel in resp_cfg.get("emergency_decel_values_mps2", [-6.0, -8.0]):
        for delay in resp_cfg.get("emergency_reaction_delay_s", [0.0, 0.2, 0.5]):
            primitives.append((constant_accel_trajectory(curr, H, dt, accel=float(decel), start_delay_s=float(delay)), ResponseSource.EMG))
    primitives = _subsample_by_source(primitives, cfg)
    return idx, object_type, nat_ref, primitives


def generate_safe_responses(scene: ScenarioData, candidates: dict[str, np.ndarray], critical: dict[str, np.ndarray], natural: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    limits = cfg.get("limits", {})
    K = int(limits.get("max_candidates", 64))
    A = int(limits.get("max_critical_agents", 8))
    R = int(limits.get("max_safe_responses", 32))
    H = int(cfg.get("time", {}).get("future_steps", 80))
    traj = np.zeros((K, A, R, H, 7), dtype=np.float32)
    valid = np.zeros((K, A, R), dtype=bool)
    source = np.full((K, A, R), int(ResponseSource.PAD), dtype=np.int32)
    is_safe = np.zeros((K, A, R), dtype=bool)
    is_low = np.zeros((K, A, R), dtype=bool)
    burden_total = np.zeros((K, A, R), dtype=np.float32)
    burden_components = np.zeros((K, A, R, 6), dtype=np.float32)

    primitive_bank: dict[int, tuple[int, int, np.ndarray, list[tuple[np.ndarray, ResponseSource]]]] = {}
    for a in range(A):
        if critical["valid"][a]:
            primitive_bank[a] = _response_primitives_for_agent(scene, a, critical, natural, cfg)

    for k in range(K):
        if not candidates["valid"][k]:
            continue
        ego = candidates["trajectory"][k]
        for a, (_, object_type, nat_ref, primitives) in primitive_bank.items():
            evaluated: list[tuple[float, float, np.ndarray, ResponseSource, bool, np.ndarray]] = []
            local_primitives = list(primitives)
            if bool(cfg.get("response", {}).get("safe_budget_search", {}).get("enabled", True)):
                curr_idx = int(critical["track_index"][a])
                curr = scene.states[curr_idx, scene.current_time_index]
                for tr, _name, _budget in typed_safe_budget_search(curr, H, float(cfg.get("time", {}).get("dt", 0.1)), ego, object_type, cfg, natural_ref=nat_ref):
                    local_primitives.append((tr, ResponseSource.OPT))
            for tr, src in local_primitives:
                if not np.all(np.isfinite(tr)):
                    continue
                unsafe = unsafe_between(ego, tr, cfg, agent_type=object_type)
                b, comps = compute_burden(tr, ego, cfg, object_type, natural_ref=nat_ref)
                sort_cost = (0.0 if not unsafe.unsafe else 10.0) + b
                evaluated.append((sort_cost, b, tr, src, not unsafe.unsafe, comps))
            evaluated.sort(key=lambda x: x[0])
            beta = float(natural.get("beta", np.full(A, 0.65))[a])
            for r, (_, b, tr, src, safe, comps) in enumerate(evaluated[:R]):
                traj[k, a, r] = tr
                valid[k, a, r] = True
                source[k, a, r] = int(src)
                is_safe[k, a, r] = bool(safe)
                burden_total[k, a, r] = b
                burden_components[k, a, r] = comps
                is_low[k, a, r] = safe and b <= beta
    return {
        "traj": traj,
        "valid": valid,
        "source": source,
        "is_safe": is_safe,
        "is_low_burden": is_low,
        "burden_total": burden_total,
        "burden_components": burden_components,
    }
