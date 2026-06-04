from __future__ import annotations

import numpy as np

from cowp.core.constants import ObjectType, PriorityRelation
from cowp.geometry.rss_like import same_heading_gap_violation
from cowp.geometry.ttc import pairwise_ttc


def _profile_for_type(cfg: dict, object_type: int) -> dict:
    bcfg = cfg.get("burden", {})
    if object_type == int(ObjectType.PEDESTRIAN):
        return bcfg.get("pedestrian", bcfg.get("vehicle", {}))
    if object_type == int(ObjectType.CYCLIST):
        return bcfg.get("cyclist", bcfg.get("vehicle", {}))
    return bcfg.get("vehicle", {})


def adaptive_beta(scene_states: np.ndarray | None, agent_type: int, rho: PriorityRelation, cfg: dict, use_adaptive: bool = True) -> float:
    bcfg = cfg.get("burden", {})
    if agent_type == int(ObjectType.PEDESTRIAN):
        beta = float(bcfg.get("beta0_pedestrian", 0.50))
    elif agent_type == int(ObjectType.CYCLIST):
        beta = float(bcfg.get("beta0_cyclist", 0.55))
    else:
        beta = float(bcfg.get("beta0_vehicle", 0.65))
    if use_adaptive and scene_states is not None and scene_states.size:
        cur_valid = scene_states[:, 10] > 0.5 if scene_states.ndim == 2 and scene_states.shape[1] >= 11 else np.ones(len(scene_states), dtype=bool)
        if np.any(cur_valid):
            speeds = scene_states[cur_valid, 5] if scene_states.shape[1] >= 6 else np.linalg.norm(scene_states[cur_valid, 3:5], axis=-1)
            low_speed = float(np.nanmean(speeds)) < float(bcfg.get("low_speed_threshold_mps", 5.0))
            ego_pos = scene_states[0, :2]
            dist = np.linalg.norm(scene_states[cur_valid, :2] - ego_pos[None, :], axis=-1)
            dense = int(np.sum(dist < float(bcfg.get("dense_neighbor_count_radius_m", 30.0)))) >= int(bcfg.get("dense_neighbor_count_threshold", 8))
            if low_speed and dense:
                beta += 0.10
    if rho == PriorityRelation.AGENT_PRIORITY:
        beta -= 0.10
    if agent_type != int(ObjectType.VEHICLE):
        beta -= 0.05
    return float(np.clip(beta, float(bcfg.get("beta_min", 0.45)), float(bcfg.get("beta_max", 0.85))))


def kinematics(traj: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed = np.linalg.norm(traj[:, 3:5], axis=-1)
    acc = np.diff(speed, prepend=speed[0]) / max(dt, 1e-3)
    jerk = np.diff(acc, prepend=acc[0]) / max(dt, 1e-3)
    return speed, acc, jerk


def burden_components(agent_traj: np.ndarray, ego_traj: np.ndarray | None, cfg: dict, object_type: int = 1, natural_ref: np.ndarray | None = None, option_loss: float = 0.0, rho: PriorityRelation = PriorityRelation.UNKNOWN, arrival_order_lost_s: float = 0.0, gap_loss_m: float = 0.0) -> np.ndarray:
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    prof = _profile_for_type(cfg, object_type)
    speed, acc, jerk = kinematics(agent_traj, dt)
    a_comf_pos = float(prof.get("a_comf_pos", 2.0))
    a_comf_neg = float(prof.get("a_comf_neg", -3.0))
    a_hard_pos = float(prof.get("a_hard_pos", 4.0))
    a_hard_neg = float(prof.get("a_hard_neg", -6.0))
    acc_excess = np.zeros_like(acc, dtype=np.float32)
    neg = acc < a_comf_neg
    pos = acc > a_comf_pos
    acc_excess[neg] = (np.abs(acc[neg]) - abs(a_comf_neg)) / max(abs(a_hard_neg) - abs(a_comf_neg), 1e-6)
    acc_excess[pos] = (acc[pos] - a_comf_pos) / max(a_hard_pos - a_comf_pos, 1e-6)
    b_acc = float(np.clip(np.max(acc_excess) if len(acc_excess) else 0.0, 0.0, 2.0))
    j_comf = float(prof.get("j_comf", 3.0))
    j_hard = float(prof.get("j_hard", 8.0))
    jerk_excess = np.maximum(0.0, np.abs(jerk) - j_comf) / max(j_hard - j_comf, 1e-6)
    b_jerk = float(np.clip(np.mean(jerk_excess) if len(jerk_excess) else 0.0, 0.0, 2.0))
    b_prog = 0.0
    if natural_ref is not None and len(natural_ref):
        progress_nat = float(np.linalg.norm(natural_ref[-1, :2] - natural_ref[0, :2]))
        progress_tau = float(np.linalg.norm(agent_traj[-1, :2] - agent_traj[0, :2])) if len(agent_traj) else 0.0
        loss_m = max(0.0, progress_nat - progress_tau)
        b_prog = 0.5 * loss_m / max(float(prof.get("progress_loss_norm_m", 20.0)), 1e-6)
        b_prog += 0.5 * max(0.0, arrival_order_lost_s) / max(float(prof.get("delay_norm_s", 4.0)), 1e-6)
    b_prog = float(np.clip(b_prog, 0.0, 2.0))
    b_risk = 0.0
    if ego_traj is not None and len(ego_traj) and len(agent_traj):
        T = min(len(ego_traj), len(agent_traj))
        ttc = pairwise_ttc(agent_traj[:T, :2], agent_traj[:T, 3:5], ego_traj[:T, :2], ego_traj[:T, 3:5])
        dist = np.linalg.norm(ego_traj[:T, :2] - agent_traj[:T, :2], axis=-1)
        ttc_min = float(cfg.get("unsafe", {}).get("ttc_min_vehicle_s", 1.5))
        gate = float(cfg.get("unsafe", {}).get("ttc_distance_gate_vehicle_m", 15.0))
        ttc_term = np.maximum(0.0, (ttc_min - ttc) / max(ttc_min, 1e-6)) * (dist < gate)
        rss, rss_mask, d_safe = same_heading_gap_violation(agent_traj[:T], ego_traj[:T])
        rss_term = rss_mask.astype(np.float32)
        b_risk = float(np.clip(np.mean(ttc_term + rss_term) if T else 0.0, 0.0, 2.0))
    b_option = float(np.clip(option_loss, 0.0, 2.0))
    b_norm = 0.0
    if rho == PriorityRelation.AGENT_PRIORITY and arrival_order_lost_s > float(cfg.get("priority", {}).get("priority_delay_tolerance_s", 0.8)):
        b_norm += 1.0
    if gap_loss_m > float(cfg.get("priority", {}).get("gap_loss_threshold_m", 6.0)):
        b_norm += 1.0
    b_norm = float(np.clip(b_norm, 0.0, 2.0))
    return np.asarray([b_acc, b_jerk, b_prog, b_risk, b_option, b_norm], dtype=np.float32)


def burden_total(components: np.ndarray, cfg: dict) -> float:
    weights_cfg = cfg.get("burden", {}).get("weights", {})
    w = np.asarray(
        [
            float(weights_cfg.get("acc", 0.25)),
            float(weights_cfg.get("jerk", 0.10)),
            float(weights_cfg.get("prog", 0.20)),
            float(weights_cfg.get("risk", 0.20)),
            float(weights_cfg.get("option", 0.15)),
            float(weights_cfg.get("norm", 0.10)),
        ],
        dtype=np.float32,
    )
    w = w / max(float(np.sum(w)), 1e-6)
    return float(np.clip(np.sum(np.asarray(components, dtype=np.float32) * w), 0.0, 2.0))


def compute_burden(agent_traj: np.ndarray, ego_traj: np.ndarray | None, cfg: dict, object_type: int = 1, natural_ref: np.ndarray | None = None, option_loss: float = 0.0, rho: PriorityRelation = PriorityRelation.UNKNOWN, arrival_order_lost_s: float = 0.0, gap_loss_m: float = 0.0) -> tuple[float, np.ndarray]:
    comps = burden_components(agent_traj, ego_traj, cfg, object_type, natural_ref, option_loss, rho, arrival_order_lost_s, gap_loss_m)
    return burden_total(comps, cfg), comps
