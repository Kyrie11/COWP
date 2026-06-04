from __future__ import annotations

import numpy as np

from cowp.core.constants import ResponseSource
from cowp.core.types import ScenarioData
from cowp.geometry.collision import unsafe_between
from cowp.label.burden import compute_burden
from cowp.label.trajectory_primitives import constant_accel_trajectory, resample_logged


def generate_safe_responses(scene: ScenarioData, candidates: dict[str, np.ndarray], critical: dict[str, np.ndarray], natural: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    limits = cfg.get("limits", {})
    resp_cfg = cfg.get("response", {})
    K = int(limits.get("max_candidates", 64))
    A = int(limits.get("max_critical_agents", 8))
    R = int(limits.get("max_safe_responses", 32))
    H = int(cfg.get("time", {}).get("future_steps", 80))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    traj = np.zeros((K, A, R, H, 7), dtype=np.float32)
    valid = np.zeros((K, A, R), dtype=bool)
    source = np.full((K, A, R), int(ResponseSource.PAD), dtype=np.int32)
    is_safe = np.zeros((K, A, R), dtype=bool)
    is_low = np.zeros((K, A, R), dtype=bool)
    burden_total = np.zeros((K, A, R), dtype=np.float32)
    burden_components = np.zeros((K, A, R, 6), dtype=np.float32)
    cur = scene.current_time_index
    for k in range(K):
        if not candidates["valid"][k]:
            continue
        ego = candidates["trajectory"][k]
        for a in range(A):
            if not critical["valid"][a]:
                continue
            idx = int(critical["track_index"][a])
            object_type = int(scene.object_type[idx])
            curr = scene.states[idx, cur]
            nat_ref_candidates = natural["traj"][a, natural["valid"][a]]
            nat_ref = nat_ref_candidates[0] if len(nat_ref_candidates) else constant_accel_trajectory(curr, H, dt, accel=0.0)
            primitives: list[tuple[np.ndarray, ResponseSource]] = []
            # R_pred: natural/logged variants.
            for m in np.where(natural["valid"][a])[0][: min(8, R)]:
                primitives.append((natural["traj"][a, m], ResponseSource.PRED))
            # R_opt: discrete burden-minimizing longitudinal responses.
            for acc in resp_cfg.get("response_acc_values_mps2", [-4.5, -3.5, -2.5, -1.5, -0.5, 0.0, 0.5, 1.0]):
                for delay in resp_cfg.get("response_start_delay_s", [0.0, 0.3, 0.6, 1.0]):
                    for dur in resp_cfg.get("response_duration_s", [1.0, 2.0, 3.0]):
                        primitives.append((constant_accel_trajectory(curr, H, dt, accel=float(acc), start_delay_s=float(delay), duration_s=float(dur)), ResponseSource.OPT))
            # R_emg: hard braking and stop.
            for decel in resp_cfg.get("emergency_decel_values_mps2", [-6.0, -8.0]):
                for delay in resp_cfg.get("emergency_reaction_delay_s", [0.0, 0.2, 0.5]):
                    primitives.append((constant_accel_trajectory(curr, H, dt, accel=float(decel), start_delay_s=float(delay)), ResponseSource.EMG))
            evaluated: list[tuple[float, np.ndarray, ResponseSource, bool, np.ndarray]] = []
            for tr, src in primitives:
                if not np.all(np.isfinite(tr)):
                    continue
                unsafe = unsafe_between(ego, tr, cfg, agent_type=object_type)
                b, comps = compute_burden(tr, ego, cfg, object_type, natural_ref=nat_ref)
                # Sort by safe first and lower burden; keep unsafe samples too for supervised safe classifier.
                sort_cost = (0.0 if not unsafe.unsafe else 10.0) + b
                evaluated.append((sort_cost, tr, src, not unsafe.unsafe, comps))
            evaluated.sort(key=lambda x: x[0])
            beta = float(natural.get("beta", np.full(A, 0.65))[a])
            for r, (_, tr, src, safe, comps) in enumerate(evaluated[:R]):
                traj[k, a, r] = tr
                valid[k, a, r] = True
                source[k, a, r] = int(src)
                is_safe[k, a, r] = bool(safe)
                b = compute_burden(tr, ego, cfg, object_type, natural_ref=nat_ref)[0]
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
