from __future__ import annotations

"""Reference-rule planners for COWP external baselines.

The module implements three non-learning baselines whose scoring equations follow
classic rule/trajectory-generation papers while using the already-materialized
COWP/Waymax candidate pool for fair comparison with learned planners.

Implemented methods
-------------------
- ``idm_lattice``: Treiber-Hennecke-Helbing Intelligent Driver Model (IDM)
  longitudinal acceleration target, used to select a conventional-safe lattice
  primitive whose initial acceleration best matches IDM car-following/free-road
  behavior.
- ``frenet_optimal``: Werling-Ziegler-Kammel-Thrun Frenet-frame optimal
  trajectory cost: longitudinal/lateral jerk, terminal lateral deviation,
  terminal speed error, acceleration/curvature comfort, with safety filtering.
- ``state_lattice``: Pivtoraiko-Kelly state-lattice primitive edge selection:
  each COWP candidate is treated as a feasible local motion primitive; the
  planner minimizes edge cost plus a progress-to-go heuristic under safety and
  dynamic feasibility masks.

For online Waymax, COWP's existing route/lane-aware generator builds the local
candidate lattice.  This keeps all methods on the same candidate support as the
learned baselines; this module only changes the reference-paper selection rule.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

try:  # pragma: no cover - optional import for macro bias only.
    from cowp.core.constants import MacroType
except Exception:  # pragma: no cover
    MacroType = None  # type: ignore


RULE_BASELINES = {"pdm_closed", "idm_lattice", "frenet_optimal", "state_lattice"}


@dataclass(frozen=True)
class RulePlannerParams:
    dt: float = 0.1
    # IDM parameters from the standard formulation.
    idm_desired_speed_mps: float = 13.89
    idm_time_headway_s: float = 1.5
    idm_min_spacing_m: float = 2.0
    idm_max_accel_mps2: float = 1.4
    idm_comfort_brake_mps2: float = 2.0
    idm_delta: float = 4.0
    idm_eval_steps: int = 20
    leader_lateral_gate_m: float = 4.0
    leader_max_range_m: float = 80.0
    # Frenet optimal trajectory cost weights.
    frenet_kj: float = 0.10
    frenet_kt: float = 0.05
    frenet_kd: float = 1.00
    frenet_ks: float = 0.35
    frenet_ka: float = 0.05
    frenet_kyaw: float = 0.02
    # State-lattice primitive search weights.
    lattice_progress_weight: float = 0.16
    lattice_time_weight: float = 0.02
    lattice_curvature_weight: float = 0.12
    lattice_accel_weight: float = 0.05
    lattice_jerk_weight: float = 0.02
    lattice_lane_change_penalty: float = 0.35
    lattice_stop_penalty: float = 0.20
    safety_violation_penalty: float = 1.0e6


def params_from_cfg(cfg: Mapping[str, Any] | None) -> RulePlannerParams:
    cfg = cfg or {}
    time_cfg = cfg.get("time", {}) if isinstance(cfg, Mapping) else {}
    rule_cfg = cfg.get("rule_baselines", {}) if isinstance(cfg, Mapping) else {}
    return RulePlannerParams(
        dt=float(rule_cfg.get("dt", time_cfg.get("dt", 0.1))),
        idm_desired_speed_mps=float(rule_cfg.get("idm_desired_speed_mps", 13.89)),
        idm_time_headway_s=float(rule_cfg.get("idm_time_headway_s", 1.5)),
        idm_min_spacing_m=float(rule_cfg.get("idm_min_spacing_m", 2.0)),
        idm_max_accel_mps2=float(rule_cfg.get("idm_max_accel_mps2", 1.4)),
        idm_comfort_brake_mps2=float(rule_cfg.get("idm_comfort_brake_mps2", 2.0)),
        idm_delta=float(rule_cfg.get("idm_delta", 4.0)),
        idm_eval_steps=int(rule_cfg.get("idm_eval_steps", 20)),
        leader_lateral_gate_m=float(rule_cfg.get("leader_lateral_gate_m", 4.0)),
        leader_max_range_m=float(rule_cfg.get("leader_max_range_m", 80.0)),
        frenet_kj=float(rule_cfg.get("frenet_kj", 0.10)),
        frenet_kt=float(rule_cfg.get("frenet_kt", 0.05)),
        frenet_kd=float(rule_cfg.get("frenet_kd", 1.00)),
        frenet_ks=float(rule_cfg.get("frenet_ks", 0.35)),
        frenet_ka=float(rule_cfg.get("frenet_ka", 0.05)),
        frenet_kyaw=float(rule_cfg.get("frenet_kyaw", 0.02)),
        lattice_progress_weight=float(rule_cfg.get("lattice_progress_weight", 0.16)),
        lattice_time_weight=float(rule_cfg.get("lattice_time_weight", 0.02)),
        lattice_curvature_weight=float(rule_cfg.get("lattice_curvature_weight", 0.12)),
        lattice_accel_weight=float(rule_cfg.get("lattice_accel_weight", 0.05)),
        lattice_jerk_weight=float(rule_cfg.get("lattice_jerk_weight", 0.02)),
        lattice_lane_change_penalty=float(rule_cfg.get("lattice_lane_change_penalty", 0.35)),
        lattice_stop_penalty=float(rule_cfg.get("lattice_stop_penalty", 0.20)),
        safety_violation_penalty=float(rule_cfg.get("safety_violation_penalty", 1.0e6)),
    )


def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get(batch: Mapping[str, Any], *names: str) -> Any | None:
    for name in names:
        if name in batch:
            return batch[name]
    return None


def _wrap_angle_np(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def _sdc_indices_np(batch: Mapping[str, Any], batch_size: int) -> np.ndarray:
    is_sdc = _get(batch, "state/is_sdc", "womd/state/is_sdc")
    if is_sdc is not None:
        arr = _to_numpy(is_sdc)
        if arr.ndim >= 2:
            return np.argmax(arr.astype(np.float32), axis=1).astype(np.int64)
    return np.zeros(batch_size, dtype=np.int64)


def _history_np(batch: Mapping[str, Any]) -> np.ndarray | None:
    hist = _get(batch, "state/history", "womd/state/history")
    if hist is None:
        return None
    arr = _to_numpy(hist).astype(np.float32, copy=False)
    if arr.ndim == 3:
        arr = arr[None]
    return arr


def _current_state_from_history(hist: np.ndarray) -> np.ndarray:
    """Return current states in COWP canonical [x,y,z,l,w,h,yaw,vx,vy,speed,valid]."""
    if hist.ndim != 4:
        raise ValueError(f"history must be [B,N,T,D], got {hist.shape}")
    return hist[:, :, -1, :]


def _candidate_arrays(batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    cand = _to_numpy(batch["cowp/candidates/trajectory"]).astype(np.float32, copy=False)
    valid = _to_numpy(batch["cowp/candidates/valid"]).astype(bool, copy=False)
    conventional = _get(batch, "cowp/candidates/conventional_safe")
    conventional_np = _to_numpy(conventional).astype(bool, copy=False) if conventional is not None else valid.copy()
    macro = _get(batch, "cowp/candidates/macro_type")
    macro_np = _to_numpy(macro).astype(np.int64, copy=False) if macro is not None else None
    utility = _get(batch, "cowp/candidates/ego_utility_prior")
    utility_np = _to_numpy(utility).astype(np.float32, copy=False) if utility is not None else None
    if cand.ndim == 3:
        cand = cand[None]
    if valid.ndim == 1:
        valid = valid[None]
    if conventional_np.ndim == 1:
        conventional_np = conventional_np[None]
    return cand, valid, conventional_np, macro_np, utility_np


def _speed(traj: np.ndarray) -> np.ndarray:
    return np.linalg.norm(traj[..., 3:5], axis=-1)


def _accel_from_speed(speed: np.ndarray, dt: float) -> np.ndarray:
    if speed.shape[-1] == 0:
        return np.zeros_like(speed)
    return np.diff(speed, axis=-1, prepend=speed[..., :1]) / max(float(dt), 1e-3)


def _jerk_from_accel(acc: np.ndarray, dt: float) -> np.ndarray:
    if acc.shape[-1] == 0:
        return np.zeros_like(acc)
    return np.diff(acc, axis=-1, prepend=acc[..., :1]) / max(float(dt), 1e-3)


def _curvature_like(traj: np.ndarray, dt: float) -> np.ndarray:
    yaw = np.unwrap(traj[..., 2], axis=-1)
    yaw_rate = np.diff(yaw, axis=-1, prepend=yaw[..., :1]) / max(float(dt), 1e-3)
    sp = np.maximum(_speed(traj), 0.1)
    return yaw_rate / sp


def _frenet_sd(traj: np.ndarray, ego_current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yaw = float(ego_current[6])
    e_s = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
    e_d = np.asarray([-np.sin(yaw), np.cos(yaw)], dtype=np.float32)
    rel = traj[..., :2] - ego_current[:2]
    s = rel @ e_s
    d = rel @ e_d
    return s.astype(np.float32), d.astype(np.float32)


def _find_idm_leader(cur: np.ndarray, sdc_idx: int, p: RulePlannerParams) -> tuple[float | None, float | None]:
    if sdc_idx < 0 or sdc_idx >= cur.shape[0]:
        return None, None
    ego = cur[sdc_idx]
    if ego.shape[0] < 11 or ego[10] <= 0.5:
        return None, None
    yaw = float(ego[6])
    e_s = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
    e_d = np.asarray([-np.sin(yaw), np.cos(yaw)], dtype=np.float32)
    ego_xy = ego[:2]
    ego_len = float(max(ego[3], 4.5))
    ego_speed = float(max(ego[9], np.linalg.norm(ego[7:9]), 0.0))
    best_s = np.inf
    best_v = None
    for j, other in enumerate(cur):
        if j == sdc_idx or other.shape[0] < 11 or other[10] <= 0.5:
            continue
        rel = other[:2] - ego_xy
        longitudinal = float(rel @ e_s)
        lateral = abs(float(rel @ e_d))
        if longitudinal <= 0.0 or longitudinal > p.leader_max_range_m or lateral > p.leader_lateral_gate_m:
            continue
        # Match the leader to the ego's lane and direction.  A weak cosine gate
        # avoids treating crossing traffic as a car-following leader.
        if np.cos(float(other[6]) - yaw) < 0.35:
            continue
        other_len = float(max(other[3], 4.5))
        gap = max(0.1, longitudinal - 0.5 * ego_len - 0.5 * other_len)
        if gap < best_s:
            best_s = gap
            best_v = float(other[7:9] @ e_s)
    if not np.isfinite(best_s):
        return None, None
    return float(best_s), float(max(best_v if best_v is not None else 0.0, 0.0))


def idm_reference_accel(cur: np.ndarray, sdc_idx: int, p: RulePlannerParams) -> float:
    """Treiber-Hennecke-Helbing IDM acceleration.

    a = a_max [1 - (v/v0)^delta - (s*(v,delta_v)/s)^2]
    s* = s0 + max(0, v T + v delta_v / (2 sqrt(a_max b)))
    """
    ego = cur[sdc_idx]
    v = float(max(ego[9], np.linalg.norm(ego[7:9]), 0.0))
    v0 = max(float(p.idm_desired_speed_mps), 0.1)
    gap, front_v = _find_idm_leader(cur, sdc_idx, p)
    free_term = (v / v0) ** float(p.idm_delta)
    if gap is None or front_v is None:
        return float(p.idm_max_accel_mps2 * (1.0 - free_term))
    delta_v = v - float(front_v)
    sqrt_ab = max(2.0 * np.sqrt(max(p.idm_max_accel_mps2 * p.idm_comfort_brake_mps2, 1e-6)), 1e-3)
    s_star = p.idm_min_spacing_m + max(0.0, v * p.idm_time_headway_s + v * delta_v / sqrt_ab)
    return float(p.idm_max_accel_mps2 * (1.0 - free_term - (s_star / max(float(gap), 0.1)) ** 2))


def _idm_cost_for_scene(candidates: np.ndarray, cur: np.ndarray, sdc_idx: int, p: RulePlannerParams) -> np.ndarray:
    K, H = candidates.shape[:2]
    v = float(max(cur[sdc_idx, 9], np.linalg.norm(cur[sdc_idx, 7:9]), 0.0)) if 0 <= sdc_idx < cur.shape[0] else 0.0
    a_ref = idm_reference_accel(cur, sdc_idx, p) if 0 <= sdc_idx < cur.shape[0] else 0.0
    sp = _speed(candidates)
    acc = _accel_from_speed(sp, p.dt)
    jerk = _jerk_from_accel(acc, p.dt)
    h = min(max(1, p.idm_eval_steps), H)
    # Compare the candidate's early acceleration to the IDM target, and also
    # keep the terminal speed close to the IDM one-step free/car-following target.
    target_speed = max(0.0, min(p.idm_desired_speed_mps, v + a_ref * h * p.dt))
    cost = np.mean((acc[:, :h] - a_ref) ** 2, axis=-1)
    cost += 0.05 * np.mean(np.abs(jerk[:, :h]), axis=-1)
    cost += 0.10 * ((sp[:, h - 1] - target_speed) / max(p.idm_desired_speed_mps, 1.0)) ** 2
    # Prefer progress only weakly; IDM is fundamentally a car-following rule.
    progress = np.linalg.norm(candidates[:, -1, :2] - candidates[:, 0, :2], axis=-1)
    cost -= 0.015 * progress
    return cost.astype(np.float32)


def _frenet_cost_for_scene(candidates: np.ndarray, cur: np.ndarray, sdc_idx: int, p: RulePlannerParams) -> np.ndarray:
    K, H = candidates.shape[:2]
    ego = cur[sdc_idx] if 0 <= sdc_idx < cur.shape[0] else np.zeros(11, dtype=np.float32)
    s, d = _frenet_sd(candidates, ego)
    # Longitudinal/lateral derivatives in Frenet coordinates.
    s_dot = np.diff(s, axis=-1, prepend=np.zeros((K, 1), dtype=np.float32)) / max(p.dt, 1e-3)
    s_ddot = np.diff(s_dot, axis=-1, prepend=s_dot[:, :1]) / max(p.dt, 1e-3)
    s_jerk = np.diff(s_ddot, axis=-1, prepend=s_ddot[:, :1]) / max(p.dt, 1e-3)
    d_dot = np.diff(d, axis=-1, prepend=np.zeros((K, 1), dtype=np.float32)) / max(p.dt, 1e-3)
    d_ddot = np.diff(d_dot, axis=-1, prepend=d_dot[:, :1]) / max(p.dt, 1e-3)
    d_jerk = np.diff(d_ddot, axis=-1, prepend=d_ddot[:, :1]) / max(p.dt, 1e-3)
    sp = _speed(candidates)
    acc = _accel_from_speed(sp, p.dt)
    curv = _curvature_like(candidates, p.dt)
    target_speed = max(float(ego[9]) if ego.shape[0] > 9 else 0.0, min(float(p.idm_desired_speed_mps), 13.89))
    T = float(H) * p.dt
    # Werling's cost family: C_d = k_j J_d + k_t T + k_d d_1^2,
    # C_s = k_j J_s + k_t T + k_s (s_dot_1 - s_dot_target)^2.
    jd = np.sum(d_jerk ** 2, axis=-1) * p.dt
    js = np.sum(s_jerk ** 2, axis=-1) * p.dt
    terminal_lateral = d[:, -1] ** 2
    terminal_speed_error = (sp[:, -1] - target_speed) ** 2
    comfort = np.mean(np.abs(acc), axis=-1) + np.mean(np.abs(curv), axis=-1)
    yaw_rate = np.diff(np.unwrap(candidates[:, :, 2], axis=-1), axis=-1, prepend=candidates[:, :1, 2]) / max(p.dt, 1e-3)
    cost = (
        p.frenet_kj * (jd + js)
        + 2.0 * p.frenet_kt * T
        + p.frenet_kd * terminal_lateral
        + p.frenet_ks * terminal_speed_error
        + p.frenet_ka * comfort
        + p.frenet_kyaw * np.mean(np.abs(yaw_rate), axis=-1)
    )
    return cost.astype(np.float32)


def _macro_penalty(macro_row: np.ndarray | None, K: int, p: RulePlannerParams) -> np.ndarray:
    out = np.zeros(K, dtype=np.float32)
    if macro_row is None or MacroType is None:
        return out
    for i in range(min(K, len(macro_row))):
        m = int(macro_row[i])
        if m in {int(MacroType.LANE_CHANGE_LEFT), int(MacroType.LANE_CHANGE_RIGHT)}:
            out[i] += p.lattice_lane_change_penalty
        if m in {int(MacroType.STOP_BEFORE_CONFLICT), int(MacroType.YIELD), int(MacroType.CREEP), int(MacroType.NEUTRAL_EGO)}:
            out[i] += p.lattice_stop_penalty
    return out


def _state_lattice_cost_for_scene(candidates: np.ndarray, cur: np.ndarray, sdc_idx: int, p: RulePlannerParams, macro_row: np.ndarray | None, utility_row: np.ndarray | None) -> np.ndarray:
    K, H = candidates.shape[:2]
    sp = _speed(candidates)
    acc = _accel_from_speed(sp, p.dt)
    jerk = _jerk_from_accel(acc, p.dt)
    curv = _curvature_like(candidates, p.dt)
    ds = np.linalg.norm(np.diff(candidates[:, :, :2], axis=1, prepend=candidates[:, :1, :2]), axis=-1)
    path_length = np.sum(ds, axis=-1)
    progress = np.linalg.norm(candidates[:, -1, :2] - candidates[:, 0, :2], axis=-1)
    edge_cost = (
        p.lattice_time_weight * H * p.dt
        + p.lattice_curvature_weight * np.mean(np.abs(curv), axis=-1)
        + p.lattice_accel_weight * np.mean(np.abs(acc), axis=-1)
        + p.lattice_jerk_weight * np.mean(np.abs(jerk), axis=-1)
        - p.lattice_progress_weight * progress
        + 0.005 * path_length
    )
    edge_cost += _macro_penalty(macro_row, K, p)
    if utility_row is not None and len(utility_row) >= K:
        # COWP stores lower ego utility prior as better in the online builder.
        edge_cost += 0.25 * np.nan_to_num(utility_row[:K], nan=0.0, posinf=10.0, neginf=-10.0)
    return edge_cost.astype(np.float32)



def _pdm_closed_cost_for_scene(candidates: np.ndarray, cur: np.ndarray, sdc_idx: int, p: RulePlannerParams) -> np.ndarray:
    """PDM-Closed-style proposal score on the COWP local proposal lattice.

    PDM-Closed is rule based: a centerline/local-lateral proposal family is
    rolled against a simple predictive observation and ranked by safety,
    progress and comfort.  COWP already materializes a richer route-aware local
    proposal family, so this adaptation preserves the PDM scoring/prediction
    logic while replacing only nuPlan's proposal constructor.
    """
    K, H = candidates.shape[:2]
    ego = cur[sdc_idx] if 0 <= sdc_idx < cur.shape[0] else np.zeros(11, dtype=np.float32)
    sp = _speed(candidates)
    acc = _accel_from_speed(sp, p.dt)
    jerk = _jerk_from_accel(acc, p.dt)
    curv = _curvature_like(candidates, p.dt)
    rel = candidates[..., :2] - ego[:2]
    heading = float(ego[6]) if ego.shape[0] > 6 else 0.0
    e_s = np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float32)
    e_d = np.asarray([-np.sin(heading), np.cos(heading)], dtype=np.float32)
    progress = rel[:, -1] @ e_s
    lateral_terminal = np.abs(rel[:, -1] @ e_d)

    # Predict non-ego vehicles with the constant-velocity observation model that
    # underlies the lightweight PDM proposal scorer.  This stays independent of
    # COWP witness/burden labels.
    t = (np.arange(H, dtype=np.float32) + 1.0) * float(p.dt)
    min_clear = np.full(K, 1e3, dtype=np.float32)
    ego_radius = max(float(ego[3] if ego.shape[0] > 3 else 4.5), float(ego[4] if ego.shape[0] > 4 else 2.0)) * 0.45
    for j, other in enumerate(cur):
        if j == sdc_idx or other.shape[0] < 11 or other[10] <= 0.5:
            continue
        op = other[:2][None, :] + t[:, None] * other[7:9][None, :]
        d = np.linalg.norm(candidates[:, :, :2] - op[None, :, :], axis=-1)
        other_radius = max(float(other[3]), float(other[4])) * 0.45
        min_clear = np.minimum(min_clear, d.min(axis=-1) - ego_radius - other_radius)
    safety_pen = np.square(np.maximum(0.0, 2.0 - min_clear)) * 5.0
    comfort = 0.10 * np.mean(np.abs(acc), axis=-1) + 0.03 * np.mean(np.abs(jerk), axis=-1) + 0.20 * np.mean(np.abs(curv), axis=-1)
    centerline = 0.08 * lateral_terminal
    target_speed = min(max(float(ego[9]) + 2.0, 5.0), p.idm_desired_speed_mps)
    speed_cost = 0.05 * np.square(sp[:, -1] - target_speed)
    return (safety_pen + comfort + centerline + speed_cost - 0.18 * progress).astype(np.float32)

def rule_costs_for_batch(batch: Mapping[str, Any], cfg: Mapping[str, Any] | None, method: str, *, require_conventional_safe: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cost, accept_mask, valid_mask)`` for a rule method.

    Lower cost is better.  ``accept_mask`` is the method's safety/conventional
    feasible set and is used both for selection and learned-offline acceptance
    metrics.
    """
    method = str(method).lower()
    if method not in RULE_BASELINES:
        raise ValueError(f"Unsupported rule baseline: {method}. Expected one of {sorted(RULE_BASELINES)}")
    p = params_from_cfg(cfg)
    candidates, valid, conventional, macro, utility = _candidate_arrays(batch)
    B, K = valid.shape[:2]
    hist = _history_np(batch)
    cur = _current_state_from_history(hist) if hist is not None else np.zeros((B, 1, 11), dtype=np.float32)
    sdc = _sdc_indices_np(batch, B)
    accept = valid.copy()
    if require_conventional_safe:
        accept &= conventional[:, :K]
    cost = np.full((B, K), p.safety_violation_penalty, dtype=np.float32)
    for b in range(B):
        cand_b = candidates[b, :K]
        cur_b = cur[b]
        sdc_b = int(np.clip(sdc[b], 0, max(cur_b.shape[0] - 1, 0)))
        macro_b = macro[b, :K] if macro is not None and macro.ndim >= 2 else None
        utility_b = utility[b, :K] if utility is not None and utility.ndim >= 2 else None
        if method == "pdm_closed":
            c = _pdm_closed_cost_for_scene(cand_b, cur_b, sdc_b, p)
        elif method == "idm_lattice":
            c = _idm_cost_for_scene(cand_b, cur_b, sdc_b, p)
        elif method == "frenet_optimal":
            c = _frenet_cost_for_scene(cand_b, cur_b, sdc_b, p)
        else:
            c = _state_lattice_cost_for_scene(cand_b, cur_b, sdc_b, p, macro_b, utility_b)
        c = np.nan_to_num(c, nan=p.safety_violation_penalty, posinf=p.safety_violation_penalty, neginf=-p.safety_violation_penalty)
        cost[b, : min(K, len(c))] = c[:K]
        cost[b, ~accept[b]] = p.safety_violation_penalty
        if not np.any(accept[b]) and np.any(valid[b]):
            # If the conventional filter is empty in a rare scene, fall back to
            # the valid set rather than returning no action.  Keep the costs.
            accept[b] = valid[b]
    return cost, accept, valid


def rule_scores_for_batch(batch: Mapping[str, Any], cfg: Mapping[str, Any] | None, method: str, *, require_conventional_safe: bool = True) -> tuple[np.ndarray, np.ndarray]:
    cost, accept, _ = rule_costs_for_batch(batch, cfg, method, require_conventional_safe=require_conventional_safe)
    return (-cost).astype(np.float32), accept.astype(bool)


def select_rule_indices(batch: Mapping[str, Any], cfg: Mapping[str, Any] | None, method: str, *, require_conventional_safe: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, accept = rule_scores_for_batch(batch, cfg, method, require_conventional_safe=require_conventional_safe)
    masked = np.where(accept, scores, -np.inf)
    selected = np.full(scores.shape[0], -1, dtype=np.int64)
    for b in range(scores.shape[0]):
        if np.any(accept[b]):
            selected[b] = int(np.argmax(masked[b]))
    return selected, accept, scores
