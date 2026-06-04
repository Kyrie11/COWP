from __future__ import annotations

import numpy as np

from cowp.core.constants import NaturalSource, PriorityRelation
from cowp.core.types import ScenarioData, ensure_trajectory_7
from cowp.label.burden import adaptive_beta, compute_burden
from cowp.label.priority import determine_priority, priority_preserved
from cowp.label.trajectory_primitives import constant_accel_trajectory, resample_logged


def _normalize_weights(weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(weights, dtype=np.float32)
    total = float(np.sum(weights[valid]))
    if total > 0:
        out[valid] = weights[valid] / total
    return out


def _traj_distance(a: np.ndarray, b: np.ndarray) -> float:
    T = min(len(a), len(b))
    if T == 0:
        return float("inf")
    return float(np.mean(np.linalg.norm(a[:T, :2] - b[:T, :2], axis=-1)))


def generate_natural_alternatives(scene: ScenarioData, critical: dict[str, np.ndarray], ego_neutral_traj: np.ndarray, cfg: dict, ablation: dict | None = None) -> dict[str, np.ndarray]:
    ablation = ablation or {}
    use_obs = bool(ablation.get("use_obs_branch", True))
    use_neu = bool(ablation.get("use_neutral_branch", True))
    use_prio = bool(ablation.get("use_priority_branch", True))
    limits = cfg.get("limits", {})
    nat_cfg = cfg.get("natural", {})
    A = int(limits.get("max_critical_agents", 8))
    M = int(limits.get("max_natural_alternatives", 24))
    H = int(cfg.get("time", {}).get("future_steps", 80))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    traj = np.zeros((A, M, H, 7), dtype=np.float32)
    valid = np.zeros((A, M), dtype=bool)
    source = np.full((A, M), int(NaturalSource.PAD), dtype=np.int32)
    burden_neutral = np.zeros((A, M), dtype=np.float32)
    priority_ok = np.zeros((A, M), dtype=bool)
    weights = np.zeros((A, M), dtype=np.float32)
    beta = np.zeros(A, dtype=np.float32)

    cur = scene.current_time_index
    for a in range(A):
        if not critical["valid"][a]:
            continue
        idx = int(critical["track_index"][a])
        object_type = int(scene.object_type[idx])
        fut_states = scene.states[idx, cur + 1 : cur + 1 + H, :]
        fut_mask = fut_states[:, 10] > 0.5
        if np.any(fut_mask):
            logged = ensure_trajectory_7(fut_states)
            if len(logged) < H:
                logged = np.pad(logged, ((0, H - len(logged)), (0, 0)), mode="edge")
        else:
            logged = constant_accel_trajectory(scene.states[idx, cur], H, dt, accel=0.0)
        rho = PriorityRelation(int(critical.get("base_priority", np.zeros(A, dtype=np.int32))[a]))
        if rho == PriorityRelation.UNKNOWN:
            rho = determine_priority(scene, idx, ego_neutral_traj, logged, cfg)
        scene_current = scene.states[:, cur, :] if scene.states.ndim == 3 else None
        beta[a] = adaptive_beta(scene_current, object_type, rho, cfg, use_adaptive=True)
        candidates: list[tuple[np.ndarray, NaturalSource, float]] = []
        if use_obs:
            count = 0
            for ss in nat_cfg.get("obs_speed_scale", [0.85, 0.95, 1.0, 1.05, 1.15]):
                for shift_s in nat_cfg.get("obs_time_shift_s", [-0.5, 0.0, 0.5]):
                    for lat in nat_cfg.get("obs_lateral_offset_m", [-0.3, 0.0, 0.3]):
                        if count >= int(nat_cfg.get("max_obs_samples", 8)):
                            break
                        tr = resample_logged(logged, H, time_shift_steps=int(round(float(shift_s) / dt)), speed_scale=float(ss), lateral_offset=float(lat))
                        candidates.append((tr, NaturalSource.OBS, float(nat_cfg.get("source_weight_obs", 1.0))))
                        count += 1
                    if count >= int(nat_cfg.get("max_obs_samples", 8)):
                        break
                if count >= int(nat_cfg.get("max_obs_samples", 8)):
                    break
        if use_neu:
            count = 0
            for acc in nat_cfg.get("neutral_acc_values_mps2", [-1.0, -0.5, 0.0, 0.5, 1.0]):
                for voff in nat_cfg.get("neutral_target_speed_offsets_mps", [-2.0, 0.0, 2.0]):
                    if count >= int(nat_cfg.get("max_neutral_samples", 8)):
                        break
                    tr = constant_accel_trajectory(scene.states[idx, cur], H, dt, accel=float(acc), speed_offset=float(voff))
                    candidates.append((tr, NaturalSource.NEU, float(nat_cfg.get("source_weight_neu", 0.8))))
                    count += 1
                if count >= int(nat_cfg.get("max_neutral_samples", 8)):
                    break
        if use_prio:
            count = 0
            for acc in nat_cfg.get("prio_acc_values_mps2", [-0.5, 0.0, 0.5]):
                if count >= int(nat_cfg.get("prio_max_samples", 8)):
                    break
                tr = constant_accel_trajectory(scene.states[idx, cur], H, dt, accel=float(acc))
                candidates.append((tr, NaturalSource.PRIO, float(nat_cfg.get("source_weight_prio", 1.2))))
                count += 1
        # Fallback if ablations remove all branches.
        if not candidates:
            candidates.append((logged, NaturalSource.OBS, 1.0))
        kept = 0
        raw_w = np.zeros(M, dtype=np.float32)
        for tr, src, src_weight in candidates:
            if kept >= M:
                break
            b_total, _ = compute_burden(tr, ego_neutral_traj, cfg, object_type, natural_ref=logged, rho=rho)
            pr_ok = priority_preserved(tr, logged, rho, cfg)
            dyn_ok = bool(np.all(np.isfinite(tr)))
            map_ok = True  # Lane-distance validation is performed by diagnostics when map detail is sufficient.
            plausible = b_total <= beta[a] + 0.1
            keep = dyn_ok and map_ok and plausible and (pr_ok or rho != PriorityRelation.AGENT_PRIORITY)
            if keep:
                traj[a, kept] = tr
                valid[a, kept] = True
                source[a, kept] = int(src)
                burden_neutral[a, kept] = float(b_total)
                priority_ok[a, kept] = bool(pr_ok)
                dist = _traj_distance(tr, logged)
                raw_w[kept] = float(src_weight) * np.exp(-dist / max(float(nat_cfg.get("sigma_traj_m", 15.0)), 1e-6)) * np.exp(-b_total / max(float(nat_cfg.get("sigma_b", 0.5)), 1e-6))
                kept += 1
        if kept < int(nat_cfg.get("min_natural_alternatives", 6)):
            for acc in [-0.5, 0.0, 0.5, 1.0, -1.0, 1.5, -1.5]:
                if kept >= M:
                    break
                tr = constant_accel_trajectory(scene.states[idx, cur], H, dt, accel=acc)
                b_total, _ = compute_burden(tr, ego_neutral_traj, cfg, object_type, natural_ref=logged, rho=rho)
                traj[a, kept] = tr
                valid[a, kept] = True
                source[a, kept] = int(NaturalSource.NEU)
                burden_neutral[a, kept] = float(b_total)
                priority_ok[a, kept] = priority_preserved(tr, logged, rho, cfg)
                raw_w[kept] = np.exp(-b_total / max(float(nat_cfg.get("sigma_b", 0.5)), 1e-6))
                kept += 1
        weights[a] = _normalize_weights(raw_w, valid[a])
    return {
        "traj": traj,
        "valid": valid,
        "source": source,
        "burden_neutral": burden_neutral,
        "priority_preserved": priority_ok,
        "weight": weights,
        "beta": beta,
    }
