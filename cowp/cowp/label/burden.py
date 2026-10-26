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


def adaptive_beta(scene_states: np.ndarray | None, agent_type: int, rho: PriorityRelation, cfg: dict, use_adaptive: bool = True, ego_index: int = 0) -> float:
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
            ego_index = int(np.clip(ego_index, 0, len(scene_states) - 1))
            ego_pos = scene_states[ego_index, :2]
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



def _projected_progress(traj: np.ndarray, ref: np.ndarray | None = None) -> float:
    """Progress along the natural-reference direction, in meters.

    Euclidean start/end displacement underestimates stop-and-go losses on curved or
    perturbed alternatives and cannot be compared robustly across response
    profiles.  For burden we care about how much of the agent's natural forward
    option was consumed by the ego-conditioned response.
    """
    if traj is None or len(traj) < 2:
        return 0.0
    if ref is not None and len(ref) >= 2:
        direction = np.asarray(ref[-1, :2] - ref[0, :2], dtype=np.float32)
    else:
        direction = np.asarray(traj[-1, :2] - traj[0, :2], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-3:
        # Fall back to integrated displacement for nearly stationary references.
        diffs = np.diff(traj[:, :2], axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=-1))) if len(diffs) else 0.0
    direction = direction / norm
    return float(np.dot(np.asarray(traj[-1, :2] - traj[0, :2], dtype=np.float32), direction))


def _mean_speed(traj: np.ndarray) -> float:
    if traj is None or len(traj) == 0:
        return 0.0
    if traj.shape[1] >= 5:
        return float(np.nanmean(np.linalg.norm(traj[:, 3:5], axis=-1)))
    return 0.0


def _infer_progress_losses(agent_traj: np.ndarray, natural_ref: np.ndarray | None) -> tuple[float, float]:
    """Infer progress loss and equivalent delay from a natural reference.

    The label engine often knows that an agent had priority, but the response
    generator historically did not pass explicit arrival-order or gap-loss
    scalars into the burden function.  This helper recovers those quantities from
    the ego-conditioned response versus the natural trajectory, enabling PA/GS
    mechanism tokens instead of collapsing all normative violations into AY/OR.
    """
    if natural_ref is None or len(natural_ref) < 2 or agent_traj is None or len(agent_traj) < 2:
        return 0.0, 0.0
    progress_nat = _projected_progress(natural_ref, natural_ref)
    progress_tau = _projected_progress(agent_traj, natural_ref)
    progress_loss_m = max(0.0, progress_nat - progress_tau)
    delay_s = progress_loss_m / max(_mean_speed(natural_ref), 0.5)
    return float(progress_loss_m), float(delay_s)

def burden_components(agent_traj: np.ndarray, ego_traj: np.ndarray | None, cfg: dict, object_type: int = 1, natural_ref: np.ndarray | None = None, option_loss: float = 0.0, rho: PriorityRelation = PriorityRelation.UNKNOWN, arrival_order_lost_s: float = 0.0, gap_loss_m: float = 0.0, *, risk_known_zero: bool = False) -> np.ndarray:
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
    inferred_progress_loss_m, inferred_delay_s = _infer_progress_losses(agent_traj, natural_ref)
    effective_delay_s = max(float(arrival_order_lost_s), inferred_delay_s)
    effective_gap_loss_m = max(float(gap_loss_m), inferred_progress_loss_m)

    b_prog = 0.0
    if natural_ref is not None and len(natural_ref):
        b_prog = 0.5 * inferred_progress_loss_m / max(float(prof.get("progress_loss_norm_m", 20.0)), 1e-6)
        b_prog += 0.5 * max(0.0, effective_delay_s) / max(float(prof.get("delay_norm_s", 4.0)), 1e-6)
    b_prog = float(np.clip(b_prog, 0.0, 2.0))
    b_risk = 0.0
    # Exact fast path: unsafe_between() uses the same TTC/RSS predicates and
    # parameters as this risk component.  If the caller has already established
    # that the pair is safe, both TTC and RSS masks are empty and B_risk is
    # identically zero.  Skipping the duplicate computation changes no label.
    if (not risk_known_zero) and ego_traj is not None and len(ego_traj) and len(agent_traj):
        T = min(len(ego_traj), len(agent_traj))
        ttc = pairwise_ttc(agent_traj[:T, :2], agent_traj[:T, 3:5], ego_traj[:T, :2], ego_traj[:T, 3:5])
        dist = np.linalg.norm(ego_traj[:T, :2] - agent_traj[:T, :2], axis=-1)
        unsafe_cfg = cfg.get("unsafe", {})
        if int(object_type) == 1:
            ttc_min = float(unsafe_cfg.get("ttc_min_vehicle_s", 1.5))
            gate = float(unsafe_cfg.get("ttc_distance_gate_vehicle_m", 15.0))
        else:
            ttc_min = float(unsafe_cfg.get("ttc_min_vru_s", 2.0))
            gate = float(unsafe_cfg.get("ttc_distance_gate_vru_m", 20.0))
        ttc_term = np.maximum(0.0, (ttc_min - ttc) / max(ttc_min, 1e-6)) * (dist < gate)
        rss, rss_mask, d_safe = same_heading_gap_violation(
            agent_traj[:T],
            ego_traj[:T],
            heading_tolerance=np.deg2rad(float(unsafe_cfg.get("rss_heading_tolerance_deg", 35.0))),
            rho=float(unsafe_cfg.get("rss_reaction_time_s", 0.5)),
            a_max_accel=float(unsafe_cfg.get("rss_a_max_accel", 2.0)),
            b_min=float(unsafe_cfg.get("rss_b_min_comfort", 3.0)),
            b_max=float(unsafe_cfg.get("rss_b_max_front", 6.0)),
            min_gap=float(unsafe_cfg.get("rss_min_gap_m", 2.0)),
            lateral_margin=float(unsafe_cfg.get("rss_lateral_margin_m", 0.75)),
        )
        rss_term = rss_mask.astype(np.float32)
        b_risk = float(np.clip(np.mean(ttc_term + rss_term) if T else 0.0, 0.0, 2.0))
    b_option = float(np.clip(option_loss, 0.0, 2.0))
    b_norm = 0.0
    priority_cfg = cfg.get("priority", {})
    delay_tol = float(priority_cfg.get("priority_delay_tolerance_s", 0.8))
    progress_tol = float(priority_cfg.get("priority_progress_loss_tolerance_m", 5.0))
    gap_tol = float(priority_cfg.get("gap_loss_threshold_m", 6.0))
    # Priority-advantage loss: an agent with nominal priority is delayed or loses
    # material progress because the ego candidate occupies the conflict/gap.
    if rho == PriorityRelation.AGENT_PRIORITY and (effective_delay_s > delay_tol or inferred_progress_loss_m > progress_tol):
        b_norm += 1.0
    # Gap-space loss: even without a strict right-of-way relation, ego may consume
    # a merge/crossing option that would otherwise remain available.
    if effective_gap_loss_m > gap_tol:
        b_norm += 1.0
    b_norm = float(np.clip(b_norm, 0.0, 2.0))
    # Keep the local return in float64 so scalar component comparisons remain
    # numerically stable across NumPy/Python versions. Downstream cache tensors
    # are allocated as float32, so assignment still stores compact float32 labels.
    return np.asarray([b_acc, b_jerk, b_prog, b_risk, b_option, b_norm], dtype=np.float64)


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


def compute_burden(agent_traj: np.ndarray, ego_traj: np.ndarray | None, cfg: dict, object_type: int = 1, natural_ref: np.ndarray | None = None, option_loss: float = 0.0, rho: PriorityRelation = PriorityRelation.UNKNOWN, arrival_order_lost_s: float = 0.0, gap_loss_m: float = 0.0, *, risk_known_zero: bool = False) -> tuple[float, np.ndarray]:
    comps = burden_components(agent_traj, ego_traj, cfg, object_type, natural_ref, option_loss, rho, arrival_order_lost_s, gap_loss_m, risk_known_zero=risk_known_zero)
    return burden_total(comps, cfg), comps
