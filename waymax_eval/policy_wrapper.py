from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable
import importlib
import math
import time

import numpy as np

from cowp.core.constants import MacroType, ObjectType, PriorityRelation
from cowp.planning.set_preservation_selector import select_set_preservation_frontier_1d
from cowp.geometry.collision import unsafe_between_bool
from cowp.label.audit_relevance import canonical_root_weights
from cowp.label.burden import adaptive_beta
from cowp.label.safe_responses import build_root_recovery_trajectory_bank, prepare_root_recovery_burden_bank
from cowp.label.trajectory_primitives import constant_accel_trajectory, priority_hold_release_trajectory, smooth_arrival_trajectory, smooth_stop_trajectory, smooth_terminal_speed_arrival_trajectory, repair_planar_kinematics
from cowp.utils.checkpoint_compat import compatible_state_dict, strip_compiled_prefix


def _load_state_dict_compatible(model, state: dict) -> None:
    """Load old checkpoints after adding optional heads."""
    try:
        model.load_state_dict(state)
        return
    except Exception:
        pass
    state = strip_compiled_prefix(state)
    try:
        model.load_state_dict(state)
        return
    except Exception:
        pass
    model_state = model.state_dict()
    compatible, _, _ = compatible_state_dict(model_state, state)
    model.load_state_dict(compatible, strict=False)


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import jax  # type: ignore

        x = jax.device_get(x)
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception as exc:
        raise TypeError(f"Cannot convert object of type {type(x)!r} to numpy array") from exc


def _device_get_many(values: list[Any]) -> list[Any]:
    """Fetch a group of JAX leaves behind one host synchronization when possible."""
    try:
        import jax  # type: ignore

        return list(jax.device_get(tuple(values)))
    except Exception:
        return values


def _get_field(obj: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        if obj is None:
            return None
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict) and name in obj:
            return obj[name]
    return None


def _unwrap_batch_dim(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def _macro_name(value: int) -> str:
    try:
        return MacroType(int(value)).name
    except Exception:
        return f"UNKNOWN_{int(value)}"


def _canonical_online_method(method: str | None, gate_mode: str | None = None) -> tuple[str, str]:
    """Normalize online method aliases before expensive policy branches run."""
    m = str(method or "cowp").lower()
    alias = {
        "cowp_priority": "cowp",
        "priority_ncf": "cowp",
        "p_ncf": "cowp",
        "cowp_universal": "universal_ncf",
        "hard_ncf": "universal_ncf",
        "ego_utility": "idm_lattice",
        "utility_lattice": "idm_lattice",
        "planner_only": "planner_score_only",
        "no_ncf": "planner_score_only",
        "safety_only": "conventional_safety",
    }
    m = alias.get(m, m)
    unsupported_shared_forward_ablations = {
        "cowp_wo_counterfactual",
        "cowp_wo_neutral_branch",
        "cowp_wo_priority_branch",
        "cowp_wo_option_preservation",
        "cowp_wo_witness_rejection",
        "cowp_wo_dual_edge",
        "cowp_wo_conflict_query",
    }
    if m in unsupported_shared_forward_ablations:
        raise ValueError(
            f"{m} is not a valid shared-forward online ablation. "
            "Use a separately retrained ablation checkpoint/config for Waymax."
        )
    g = str(gate_mode or "priority").lower()
    if m in {"cowp", "cowp_cert_utility", "cowp_fallback_outcome", "cowp_recursive_viability", "cowp_rvr_pareto_guard", "cowp_successor_option_viability", "cowp_bihorizon_option_viability", "cowp_successor_restore_only", "cowp_trihorizon_option_persistence", "cowp_sov_recovery_commitment", "cowp_sov_dominance_hysteresis", "cowp_recovery_option_spectrum_hysteresis", "cowp_transition_guarded_rosh", "cowp_executable_option_spectrum_hysteresis", "cowp_waymax_kinematic_guarded_rosh", "cowp_control_projected_option_spectrum_hysteresis", "cowp_control_projected_recovery_frontier", "cowp_recourse_returnability_bridge", "cowp_shift_closed_control_reachable_tube", "cowp_conflict_window_control_reachable_tube", "cowp_shift_closed_first_action_viability_interval", "cowp_interaction_aware_reachable_response_envelope", "cowp_blocker_conditioned_interaction_aware_reachable_response_envelope"} and g == "hard":
        g = "priority"
    if m == "universal_ncf":
        g = "hard"
    elif m == "soft_burden_cost_only":
        g = "soft"
    elif m in {"idm_lattice", "conventional_safety", "planner_score_only"}:
        g = "none"
    return m, g


def _recursive_viability_recovery_mask(cand_valid, roadgraph_safe, collision_prefix_steps):
    """Lexicographic recovery set used only when full conventional feasibility is empty.

    The function intentionally has no tunable weights: preserve the drivable pool
    when it is non-empty, then retain candidates that maximize the causal
    collision-free prefix.  The caller applies the existing fallback score only
    inside this set.  It works with torch boolean/float tensors without importing
    torch at module import time.
    """
    road_pool = cand_valid & roadgraph_safe
    pool = road_pool if bool(road_pool.any().detach().cpu().item()) else cand_valid
    if not bool(pool.any().detach().cpu().item()):
        return cand_valid
    max_prefix = collision_prefix_steps[pool].max()
    return pool & (collision_prefix_steps >= max_prefix)



def _macro_recovery_representatives_np(
    cand_valid: np.ndarray,
    roadgraph_safe: np.ndarray,
    collision_prefix_steps: np.ndarray,
    macro_types: np.ndarray,
    fallback_scores: np.ndarray,
    pad_macro: int = int(MacroType.PAD),
) -> list[int]:
    """Build one deterministic recovery representative per semantic macro.

    V16.8.29--35 compared the original COWP fallback with only one global RVR
    endpoint.  V16.8.36 keeps the *same fixed candidate bank* but exposes its
    semantic support: use the same roadgraph-first recovery pool as RVR, then for
    every non-PAD macro retain the candidate with the longest current causal
    collision-safe prefix, breaking ties by the already-frozen COWP fallback
    score and finally by candidate index.

    This is support construction, not proposal expansion: no trajectory is added
    and no learned score/threshold is introduced.
    """
    valid = np.asarray(cand_valid, dtype=bool).reshape(-1)
    road = np.asarray(roadgraph_safe, dtype=bool).reshape(-1)
    prefix = np.asarray(collision_prefix_steps, dtype=np.float64).reshape(-1)
    macro = np.asarray(macro_types, dtype=np.int64).reshape(-1)
    score = np.asarray(fallback_scores, dtype=np.float64).reshape(-1)
    n = min(valid.size, road.size, prefix.size, macro.size, score.size)
    if n <= 0:
        return []
    valid, road, prefix, macro, score = valid[:n], road[:n], prefix[:n], macro[:n], score[:n]
    road_pool = valid & road
    pool = road_pool if bool(road_pool.any()) else valid
    reps: list[int] = []
    for m in sorted(int(x) for x in np.unique(macro[pool]) if int(x) != int(pad_macro)):
        idx = np.flatnonzero(pool & (macro == int(m)))
        if idx.size == 0:
            continue
        pmax = float(np.nanmax(prefix[idx]))
        tied = idx[np.isclose(prefix[idx], pmax, rtol=0.0, atol=1.0e-9)]
        finite = tied[np.isfinite(score[tied])]
        if finite.size:
            best_score = float(np.min(score[finite]))
            tied = finite[np.isclose(score[finite], best_score, rtol=0.0, atol=1.0e-9)]
        reps.append(int(np.min(tied)))
    return reps


def _recovery_frontier_mode_choice_np(
    base_idx: int,
    representative_indices: list[int] | tuple[int, ...],
    macro_types: np.ndarray,
    fallback_scores: np.ndarray,
    strict_dominance: dict[int, bool],
    weak_dominance: dict[int, bool],
    active_macro: int,
) -> tuple[int, int, bool, bool, bool]:
    """Choose a candidate from the physical recovery frontier without scalarizing it.

    Entry: any representative that *strictly* dominates the base control-projected
    option spectrum is physically admissible; select the least COWP-fallback-cost
    member of that hard frontier.

    Continuation: never jump directly between recovery macros.  Continue only the
    currently active semantic macro while one of its representatives weakly
    dominates the base.  On dominance loss, exit to the base and clear mode state.

    Returns ``(chosen_idx, new_active_macro, entered, continued, exited)``.
    """
    reps = [int(i) for i in representative_indices]
    macro = np.asarray(macro_types, dtype=np.int64).reshape(-1)
    scores = np.asarray(fallback_scores, dtype=np.float64).reshape(-1)

    def _best(cands: list[int]) -> int:
        if not cands:
            return int(base_idx)
        return int(min(cands, key=lambda i: (float(scores[i]) if np.isfinite(scores[i]) else float('inf'), int(i))))

    if int(active_macro) >= 0:
        same = [i for i in reps if 0 <= i < macro.size and int(macro[i]) == int(active_macro) and bool(weak_dominance.get(i, False))]
        if same:
            return _best(same), int(active_macro), False, True, False
        return int(base_idx), -1, False, False, True

    strict = [i for i in reps if i != int(base_idx) and bool(strict_dominance.get(i, False))]
    if not strict:
        return int(base_idx), -1, False, False, False
    chosen = _best(strict)
    return chosen, int(macro[chosen]), True, False, False


def _counterfactual_successor_agent_state(
    agent_state: np.ndarray,
    sdc_index: int,
    emitted_target: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Causal one-step successor surrogate used by the v16.8.30 mechanism probe.

    Other valid agents advance with the same constant-velocity assumption used by
    the online conventional collision screen.  Ego is advanced with the *actual
    jerk/yaw-rate-limited target* that would be emitted to Waymax, rather than the
    raw candidate waypoint.  This is deliberately a model-relative successor, not
    a claim of formal controlled invariance.
    """
    nxt = np.array(agent_state, dtype=np.float32, copy=True)
    if nxt.ndim != 2 or not (0 <= int(sdc_index) < int(nxt.shape[0])):
        return nxt
    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1.0e-6)
    valid = nxt[:, 10] > 0.5 if nxt.shape[1] > 10 else np.ones(nxt.shape[0], dtype=bool)
    other = valid.copy()
    other[int(sdc_index)] = False
    if nxt.shape[1] >= 5:
        nxt[other, 0:2] = nxt[other, 0:2] + nxt[other, 3:5] * dt
    target = np.asarray(emitted_target, dtype=np.float32).reshape(-1)
    if target.size >= 5:
        ego = nxt[int(sdc_index)]
        ego[0:2] = target[0:2]
        ego[6] = target[2]
        ego[3:5] = target[3:5]
        if ego.shape[0] > 5:
            ego[5] = float(np.linalg.norm(target[3:5]))
        nxt[int(sdc_index)] = ego
    return nxt


def _successor_option_signature(
    agent_state: np.ndarray,
    sdc_index: int,
    emitted_target: np.ndarray,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
) -> tuple[tuple[int, int, int, int], dict[str, int]]:
    """Evaluate option-set richness at the next replanning state.

    The lexicographic signature is intentionally parameter-free:
      1) any full conventional option exists;
      2) number of distinct conventional macro types;
      3) number of conventional candidates;
      4) best causal collision-safe prefix among drivable valid candidates.

    Counts are *diagnostic support statistics* over the fixed online proposal
    generator.  They are not relabeled as a safety certificate.
    """
    nxt = _counterfactual_successor_agent_state(agent_state, sdc_index, emitted_target, cfg)
    (
        _traj, valid, conventional, macro, _utility, road_safe, _collision_safe,
        prefix, _margin,
    ) = _route_lane_aware_candidates(
        nxt, int(sdc_index), roadgraph, cfg, other_future_trajs=None
    )
    valid = np.asarray(valid, dtype=bool)
    conventional = np.asarray(conventional, dtype=bool) & valid
    road_valid = np.asarray(road_safe, dtype=bool) & valid
    macro = np.asarray(macro, dtype=np.int64)
    prefix = np.asarray(prefix, dtype=np.int32)
    conv_count = int(conventional.sum())
    conv_macro_count = int(np.unique(macro[conventional]).size) if conv_count else 0
    if bool(road_valid.any()):
        max_prefix = int(prefix[road_valid].max())
    elif bool(valid.any()):
        max_prefix = int(prefix[valid].max())
    else:
        max_prefix = 0
    sig = (int(conv_count > 0), conv_macro_count, conv_count, max_prefix)
    detail = {
        "conventional_exists": int(conv_count > 0),
        "conventional_macro_types": conv_macro_count,
        "conventional_candidates": conv_count,
        "max_collision_safe_prefix_steps": max_prefix,
        "valid_candidates": int(valid.sum()),
        "roadgraph_safe_candidates": int(road_valid.sum()),
    }
    return sig, detail


def _strict_no_regret_rvr_switch(
    base_idx: int,
    rvr_idx: int,
    collision_prefix_steps,
    fallback_transport_ucb,
    rule_decision_risk,
    action_decision_risk,
    pressure_decision_risk,
    *,
    eps: float = 1.0e-7,
) -> bool:
    """Parameter-free diagnostic: permit RVR only under strict Pareto no-regret.

    This branch is not the proposed paper mechanism.  It tests whether v16.8.29
    failed mainly because max-prefix lexicography overrode already-available
    transport/rule/action/pressure evidence.
    """
    if int(base_idx) == int(rvr_idx):
        return False
    bp = float(collision_prefix_steps[int(base_idx)].detach().cpu().item())
    rp = float(collision_prefix_steps[int(rvr_idx)].detach().cpu().item())
    if rp <= bp + eps:
        return False
    tensors = (fallback_transport_ucb, rule_decision_risk, action_decision_risk, pressure_decision_risk)
    for t in tensors:
        b = float(t[int(base_idx)].detach().cpu().item())
        r = float(t[int(rvr_idx)].detach().cpu().item())
        if r > b + eps:
            return False
    return True


def _bihorizon_option_dominates(
    base_sig: tuple[int, int, int, int],
    alt_sig: tuple[int, int, int, int],
    base_prefix: int,
    alt_prefix: int,
) -> bool:
    """Parameter-free two-horizon viability dominance.

    The alternative must not reduce either the current causal survival prefix or
    the lexicographic successor option-set signature, and must strictly improve at
    least one.  This is intentionally a product partial order rather than a weighted
    scalarization.
    """
    return bool(
        alt_sig >= base_sig
        and int(alt_prefix) >= int(base_prefix)
        and (alt_sig > base_sig or int(alt_prefix) > int(base_prefix))
    )


def _successor_restoration_dominates(
    base_detail: dict[str, int], alt_detail: dict[str, int]
) -> bool:
    """Diagnostic 0->1 restoration test for a full conventional successor option."""
    return bool(
        int(alt_detail.get("conventional_exists", 0))
        > int(base_detail.get("conventional_exists", 0))
    )


def _second_successor_option_signature(
    agent_state: np.ndarray,
    sdc_index: int,
    first_emitted_target: np.ndarray,
    first_emitted_accel: float,
    candidate_traj: np.ndarray,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
) -> tuple[tuple[int, int, int, int], dict[str, int]]:
    """Evaluate option support after two *causal emitted* control intervals.

    The first transition is exactly the emitted action already used by the
    one-step successor probe.  The second transition follows waypoint 1 of the
    same selected recovery trajectory, but is projected again through the same
    jerk/yaw-rate-limited controller using the first-step acceleration as state.
    Other agents advance with the unchanged constant-velocity causal model.

    This is a deliberately minimal persistence probe.  It uses no logged future
    and does not claim a formal viability-kernel guarantee.
    """
    traj = np.asarray(candidate_traj, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] < 2:
        # Degenerate trajectory: a second causal horizon cannot be evaluated.
        return (0, 0, 0, 0), {
            "conventional_exists": 0,
            "conventional_macro_types": 0,
            "conventional_candidates": 0,
            "max_collision_safe_prefix_steps": 0,
            "valid_candidates": 0,
            "roadgraph_safe_candidates": 0,
        }
    s1 = _counterfactual_successor_agent_state(
        agent_state, sdc_index, first_emitted_target, cfg
    )
    second_target, _second_accel = _consistent_one_step_target(
        s1[int(sdc_index)], traj[1], cfg, float(first_emitted_accel)
    )
    return _successor_option_signature(
        s1, int(sdc_index), second_target, roadgraph, cfg
    )


def _trihorizon_option_persistence_dominates(
    base_sig1: tuple[int, int, int, int],
    alt_sig1: tuple[int, int, int, int],
    base_sig2: tuple[int, int, int, int],
    alt_sig2: tuple[int, int, int, int],
    base_prefix: int,
    alt_prefix: int,
) -> bool:
    """Three-horizon product-order gate for temporal option persistence.

    v16.8.31 BHOV established high recovery recall but failed the disjoint
    non-harmfulness gate.  The minimal next hypothesis is that a one-successor
    option comparison is too myopic.  The RVR alternative is therefore accepted
    only when it is non-worse at all three causal horizons:

      H0: current collision-safe prefix,
      V1: option-set signature after the first emitted action,
      V2: option-set signature after a second emitted action along the same
          recovery trajectory.

    At least one horizon must improve strictly.  No weights/tolerances are
    introduced, so the test remains a set-preservation partial order.
    """
    return bool(
        int(alt_prefix) >= int(base_prefix)
        and alt_sig1 >= base_sig1
        and alt_sig2 >= base_sig2
        and (
            int(alt_prefix) > int(base_prefix)
            or alt_sig1 > base_sig1
            or alt_sig2 > base_sig2
        )
    )




def _dominance_hysteresis_transition(
    active: bool,
    *,
    strict_alt_dominates: bool,
    weak_alt_dominates: bool,
) -> tuple[bool, bool, bool, bool]:
    """Parameter-free recovery-mode transition under a dominance partial order.

    Entry requires strict dominance.  Once active, equality is allowed so the
    controller does not chatter back to COWP on an exact tie; the mode exits as
    soon as the alternative becomes worse under the same relation.

    Returns ``(active_after, entered, continued, exited)``.
    """
    if bool(active):
        if bool(weak_alt_dominates):
            return True, False, True, False
        return False, False, False, True
    if bool(strict_alt_dominates):
        return True, True, False, False
    return False, False, False, False


def _successor_recovery_option_profile(
    agent_state: np.ndarray,
    sdc_index: int,
    emitted_target: np.ndarray,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
) -> tuple[tuple[int, ...], dict[str, int | float]]:
    """Causal successor recovery-option persistence profile.

    In zero-conventional states, the earlier successor signature often collapses
    to a single ``max_prefix`` statistic because conventional existence/counts are
    all zero.  That is structurally brittle: one long-lived trajectory can mask
    the loss of all alternative recovery modes.

    This profile keeps the *entire semantic recovery-option support curve*.
    For each causal horizon h, it counts how many distinct non-PAD macro types
    have at least one valid, roadgraph-safe candidate whose collision-safe prefix
    survives h steps.  No future Waymax state is read; other agents advance only
    through the same constant-velocity causal model used by the conventional
    screen, and the candidate bank is regenerated by the unchanged online
    physical proposal generator.

    Pointwise dominance of the curve is equivalent to saying that the alternative
    successor never has fewer surviving semantic recovery modes at any horizon.
    """
    nxt = _counterfactual_successor_agent_state(agent_state, sdc_index, emitted_target, cfg)
    (
        traj, valid, conventional, macro, _utility, road_safe, _collision_safe,
        prefix, _margin,
    ) = _route_lane_aware_candidates(
        nxt, int(sdc_index), roadgraph, cfg, other_future_trajs=None
    )
    traj = np.asarray(traj)
    valid = np.asarray(valid, dtype=bool)
    road_valid = np.asarray(road_safe, dtype=bool) & valid
    conventional = np.asarray(conventional, dtype=bool) & valid
    macro = np.asarray(macro, dtype=np.int64)
    prefix = np.asarray(prefix, dtype=np.int32)

    horizon = int(traj.shape[1]) if traj.ndim >= 3 and traj.shape[1] > 0 else int(max(prefix.max(initial=0), 1))
    best_by_macro: dict[int, int] = {}
    for idx in np.flatnonzero(road_valid):
        m = int(macro[idx])
        if m == int(MacroType.PAD):
            continue
        p = int(max(prefix[idx], 0))
        if p <= 0:
            continue
        if p > best_by_macro.get(m, 0):
            best_by_macro[m] = p

    curve = tuple(
        int(sum(int(p) >= h for p in best_by_macro.values()))
        for h in range(1, horizon + 1)
    )
    area = int(sum(curve))
    detail: dict[str, int | float] = {
        "profile_horizon_steps": int(horizon),
        "recovery_macro_types_h1": int(curve[0]) if curve else 0,
        "recovery_macro_types_full_horizon": int(curve[-1]) if curve else 0,
        "recovery_profile_area": int(area),
        "recovery_macro_types_any": int(len(best_by_macro)),
        "valid_candidates": int(valid.sum()),
        "roadgraph_safe_candidates": int(road_valid.sum()),
        "conventional_candidates": int(conventional.sum()),
        "max_collision_safe_prefix_steps": int(prefix[road_valid].max()) if bool(road_valid.any()) else (int(prefix[valid].max()) if bool(valid.any()) else 0),
    }
    return curve, detail



def _controller_transition_feasible_np(
    current: np.ndarray,
    desired: np.ndarray,
    cfg: dict,
    previous_longitudinal_accel: float = 0.0,
) -> np.ndarray:
    """Hard, parameter-free compatibility of a nominal first step with the real controller state.

    Online candidate validity intentionally ignores the initialization transient of
    discrete trajectory jerk so useful acceleration/yield primitives are not erased
    from the proposal bank.  That contract is preserved.  For *physical option-set
    representation*, however, an option should only count as immediately executable
    if its first desired step can be reached from the current controller memory
    without violating the already-existing acceleration, jerk, yaw-rate and lateral
    acceleration limits.

    This function introduces no new threshold: it reuses the same hard limits and
    dt already used by the candidate validator / emitted-action controller.  It is
    diagnostic/selection state for v16.8.34 recovery only; it does not relabel a
    candidate as conventional-safe or NCF.
    """
    cur = np.asarray(current, dtype=np.float64).reshape(-1)
    des = np.asarray(desired, dtype=np.float64)
    if des.ndim == 1:
        des = des[None, :]
    k = int(des.shape[0]) if des.ndim == 2 else 0
    if k <= 0 or cur.size < 7 or des.shape[1] < 3:
        return np.zeros((max(k, 0),), dtype=bool)

    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1.0e-6)
    cand_cfg = cfg.get("candidate", {})
    wm_cfg = cfg.get("waymax", {})
    max_accel = max(float(cand_cfg.get("max_accel_mps2", 4.0)), 1.0e-6)
    max_decel = max(float(cand_cfg.get("max_decel_mps2", 6.0)), 1.0e-6)
    max_jerk = max(float(cand_cfg.get("max_jerk_mps3", 8.0)), 1.0e-6)
    max_yaw_rate = max(float(cand_cfg.get("max_yaw_rate_rad_s", 1.2)), 1.0e-6)
    max_lateral_accel = max(float(cand_cfg.get("max_lateral_accel_mps2", 4.0)), 1.0e-6)

    cur_xy = cur[:2]
    cur_vel = np.asarray(cur[3:5], dtype=np.float64) if cur.size >= 5 else np.zeros((2,), dtype=np.float64)
    cur_speed = float(max(np.linalg.norm(cur_vel), float(cur[5]) if cur.size > 5 else 0.0, 0.0))
    cur_yaw = float(cur[6])

    if des.shape[1] >= 5:
        desired_vel = des[:, 3:5]
        desired_speed = np.linalg.norm(desired_vel, axis=-1)
    else:
        desired_vel = np.zeros((k, 2), dtype=np.float64)
        desired_speed = np.zeros((k,), dtype=np.float64)
    position_speed = np.linalg.norm(des[:, :2] - cur_xy[None, :], axis=-1) / dt
    desired_speed = np.where(desired_speed < 1.0e-3, position_speed, desired_speed)

    raw_accel = (desired_speed - cur_speed) / dt
    accel_ok = (raw_accel <= max_accel + 1.0e-3) & (raw_accel >= -max_decel - 1.0e-3)
    jerk_ok = np.abs(raw_accel - float(previous_longitudinal_accel)) <= max_jerk * dt + 1.0e-3

    desired_yaw_from_vel = np.arctan2(desired_vel[:, 1], desired_vel[:, 0])
    desired_yaw = np.where(
        desired_speed > 0.25,
        desired_yaw_from_vel,
        des[:, 2] if des.shape[1] > 2 else cur_yaw,
    )
    requested_dyaw = np.asarray(_wrap_angle(desired_yaw - cur_yaw), dtype=np.float64)
    max_dyaw = min(float(wm_cfg.get("max_delta_yaw_rad", 0.12)), max_yaw_rate * dt)
    yaw_ok = np.abs(requested_dyaw) <= max(max_dyaw, 1.0e-6) + 1.0e-3

    # The emitted controller realizes a speed/yaw pair.  Check the resulting
    # one-step lateral acceleration against the same hard candidate limit so a
    # nominal macro is not counted as an independent executable option when the
    # interface itself would need to distort that transition.
    desired_v_realized = desired_speed[:, None] * np.stack(
        [np.cos(desired_yaw), np.sin(desired_yaw)], axis=-1
    )
    acc_vec = (desired_v_realized - cur_vel[None, :]) / dt
    lat_axis = np.stack([-np.sin(desired_yaw), np.cos(desired_yaw)], axis=-1)
    lateral_accel = np.abs(np.sum(acc_vec * lat_axis, axis=-1))
    lateral_ok = lateral_accel <= max_lateral_accel + 1.0e-3

    finite = np.isfinite(raw_accel) & np.isfinite(requested_dyaw) & np.isfinite(lateral_accel)
    return np.asarray(finite & accel_ok & jerk_ok & yaw_ok & lateral_ok, dtype=bool)


def _successor_executable_recovery_option_profile(
    agent_state: np.ndarray,
    sdc_index: int,
    emitted_target: np.ndarray,
    emitted_accel: float,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
) -> tuple[tuple[int, ...], dict[str, int | float]]:
    """Recovery-option survival curve over *controller-realizable* successor options.

    v16.8.33 counts semantic macro modes in a causal successor state, but the
    online policy controller is not Markov in ``agent_state`` alone: its next
    emitted action also depends on the previous longitudinal acceleration.  This
    v16.8.34 profile therefore carries the acceleration produced by the current
    emitted action into the successor and filters the unchanged proposal bank by
    hard first-step controller-transition feasibility before an option contributes
    to the spectrum.

    The candidate bank, roadgraph screen and collision-prefix calculation remain
    exactly unchanged.  No future Waymax/logged state is read.
    """
    nxt = _counterfactual_successor_agent_state(agent_state, sdc_index, emitted_target, cfg)
    (
        traj, valid, conventional, macro, _utility, road_safe, _collision_safe,
        prefix, _margin,
    ) = _route_lane_aware_candidates(
        nxt, int(sdc_index), roadgraph, cfg, other_future_trajs=None
    )
    traj = np.asarray(traj)
    valid = np.asarray(valid, dtype=bool)
    road_valid = np.asarray(road_safe, dtype=bool) & valid
    conventional = np.asarray(conventional, dtype=bool) & valid
    macro = np.asarray(macro, dtype=np.int64)
    prefix = np.asarray(prefix, dtype=np.int32)

    if traj.ndim >= 3 and traj.shape[0] == valid.shape[0] and traj.shape[1] > 0:
        transition_ok = _controller_transition_feasible_np(
            nxt[int(sdc_index)], traj[:, 0], cfg, float(emitted_accel)
        )
    else:
        transition_ok = np.zeros_like(valid, dtype=bool)
    executable_road = road_valid & transition_ok

    horizon = int(traj.shape[1]) if traj.ndim >= 3 and traj.shape[1] > 0 else int(max(prefix.max(initial=0), 1))
    best_by_macro: dict[int, int] = {}
    for idx in np.flatnonzero(executable_road):
        m = int(macro[idx])
        if m == int(MacroType.PAD):
            continue
        p = int(max(prefix[idx], 0))
        if p <= 0:
            continue
        if p > best_by_macro.get(m, 0):
            best_by_macro[m] = p

    curve = tuple(
        int(sum(int(p) >= h for p in best_by_macro.values()))
        for h in range(1, horizon + 1)
    )
    detail: dict[str, int | float] = {
        "profile_horizon_steps": int(horizon),
        "recovery_macro_types_h1": int(curve[0]) if curve else 0,
        "recovery_macro_types_full_horizon": int(curve[-1]) if curve else 0,
        "recovery_profile_area": int(sum(curve)),
        "recovery_macro_types_any": int(len(best_by_macro)),
        "valid_candidates": int(valid.sum()),
        "roadgraph_safe_candidates": int(road_valid.sum()),
        "controller_transition_feasible_candidates": int((valid & transition_ok).sum()),
        "executable_roadgraph_candidates": int(executable_road.sum()),
        "transition_rejected_roadgraph_candidates": int((road_valid & ~transition_ok).sum()),
        "conventional_candidates": int(conventional.sum()),
        "max_collision_safe_prefix_steps": int(prefix[executable_road].max()) if bool(executable_road.any()) else (int(prefix[road_valid].max()) if bool(road_valid.any()) else 0),
    }
    return curve, detail


def _execution_spectrum_relation(
    base_transition_ok: bool,
    alt_transition_ok: bool,
    base_profile: tuple[int, ...],
    alt_profile: tuple[int, ...],
) -> tuple[bool, bool, int, int, int]:
    """Product-order dominance over current execution and future option support.

    A recovery alternative is weakly admissible only if it does not regress the
    current hard controller-transition feasibility *and* does not lose any future
    option-spectrum support.  Strict dominance requires improvement in at least one
    of those components.  No scalar weighting or tunable margin is used.
    """
    profile_strict, profile_weak, min_margin, area_delta = _option_profile_relation(
        base_profile, alt_profile
    )
    transition_delta = int(bool(alt_transition_ok)) - int(bool(base_transition_ok))
    transition_weak = transition_delta >= 0
    transition_strict = transition_delta > 0
    weak = bool(transition_weak and profile_weak)
    strict = bool(weak and (transition_strict or profile_strict))
    return strict, weak, int(transition_delta), int(min_margin), int(area_delta)




_WAYMAX_KINEMATICS_CONTRACT_CACHE: tuple[float, float, float, str] | None = None


def _waymax_kinematics_contract(cfg: dict) -> tuple[float, float, float, str]:
    """Resolve the *actual evaluation* kinematics contract used by Waymax.

    V16.8.34 filtered nominal waypoints with COWP's internal acceleration/jerk/
    yaw/lateral-acceleration limits.  That predicate is useful for controller
    projection diagnostics, but it is not the contract used by Waymax's
    ``KinematicsInfeasibilityMetric``.  Current public Waymax evaluates the
    inverse transition with acceleration magnitude and steering curvature.

    The rollout constructs that metric with its default constructor, so we try to
    introspect the installed class and use the exact private values carried by the
    instantiated metric.  The documented public defaults are a fail-safe fallback
    for unit-test environments where Waymax is not installed.  No value here is a
    tunable COWP threshold.
    """
    pcfg = cfg.get("planning", {})
    # Explicit overrides exist only for reproducibility across a deliberately
    # pinned Waymax fork; the V16.8.35 launcher does not set them.
    override_acc = pcfg.get("waymax_kinematics_max_acc_mps2")
    override_steer = pcfg.get("waymax_kinematics_max_steering_curvature")
    override_dt = pcfg.get("waymax_kinematics_dt_s")
    if override_acc is not None or override_steer is not None or override_dt is not None:
        return (
            float(10.4 if override_acc is None else override_acc),
            float(0.3 if override_steer is None else override_steer),
            float(cfg.get("time", {}).get("dt", 0.1) if override_dt is None else override_dt),
            "config_override",
        )

    global _WAYMAX_KINEMATICS_CONTRACT_CACHE
    if _WAYMAX_KINEMATICS_CONTRACT_CACHE is not None:
        return _WAYMAX_KINEMATICS_CONTRACT_CACHE

    candidates = (
        ("waymax.metrics", "KinematicsInfeasibilityMetric"),
        ("waymax.metrics.comfort", "KinematicsInfeasibilityMetric"),
    )
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            metric = cls()
            max_acc = float(getattr(metric, "_max_acc"))
            max_steering = float(getattr(metric, "_max_steering"))
            dt = float(getattr(metric, "_dt"))
            if all(np.isfinite([max_acc, max_steering, dt])) and max_acc > 0.0 and max_steering > 0.0 and dt > 0.0:
                _WAYMAX_KINEMATICS_CONTRACT_CACHE = (max_acc, max_steering, dt, f"{module_name}.{class_name}")
                return _WAYMAX_KINEMATICS_CONTRACT_CACHE
        except Exception:
            continue
    _WAYMAX_KINEMATICS_CONTRACT_CACHE = (10.4, 0.3, 0.1, "public_waymax_default_fallback")
    return _WAYMAX_KINEMATICS_CONTRACT_CACHE


def _waymax_kinematic_transition_np(
    current: np.ndarray,
    target: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    """Reproduce Waymax KinematicsInfeasibilityMetric inverse-transition logic.

    ``current`` can be shape ``(D,)`` or ``(K,D)``; ``target`` is ``(K,5)`` or
    ``(5,)`` with ``[x,y,yaw,vx,vy]``.  The calculation intentionally mirrors
    public Waymax ``bicycle_model.compute_inverse``: acceleration comes from
    speed change, steering is yaw change divided by traveled arc length, and
    steering is zeroed when either endpoint speed is below 0.6 m/s.

    Position is not part of Waymax's kinematics-infeasibility test.  This helper
    therefore checks the evaluator contract rather than inventing a COWP-specific
    penalty.
    """
    cur = np.asarray(current, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if cur.ndim == 1:
        cur = cur[None, :]
    if tgt.ndim == 1:
        tgt = tgt[None, :]
    k = int(tgt.shape[0]) if tgt.ndim == 2 else 0
    if k <= 0 or cur.ndim != 2 or cur.shape[1] < 7 or tgt.shape[1] < 5:
        z = np.zeros((max(k, 0),), dtype=np.float64)
        return np.zeros((max(k, 0),), dtype=bool), z, z, {
            "max_acc_mps2": 10.4, "max_steering_curvature": 0.3, "metric_dt_s": 0.1,
            "contract_source": "invalid_input",
        }
    if cur.shape[0] == 1 and k > 1:
        cur = np.repeat(cur, k, axis=0)
    if cur.shape[0] != k:
        raise ValueError(f"current/target batch mismatch: {cur.shape[0]} vs {k}")

    max_acc, max_steering, metric_dt, source = _waymax_kinematics_contract(cfg)
    metric_dt = max(float(metric_dt), 1.0e-6)
    old_speed = np.linalg.norm(cur[:, 3:5], axis=-1)
    new_speed = np.linalg.norm(tgt[:, 3:5], axis=-1)
    accel = (new_speed - old_speed) / metric_dt

    new_yaw_recorded = np.asarray(_wrap_angle(tgt[:, 2]), dtype=np.float64)
    new_yaw_from_velocity = np.arctan2(tgt[:, 4], tgt[:, 3])
    # Waymax uses velocity yaw when the *new* speed is not tiny, but keeps the
    # recorded target yaw at very low new speed.  The old yaw is the trajectory
    # state's recorded yaw.  It then zeros steering if either endpoint is slow.
    speed_limit = 0.6
    real_new_yaw = np.where(np.abs(new_speed) <= speed_limit, new_yaw_recorded, new_yaw_from_velocity)
    delta_yaw = np.asarray(_wrap_angle(real_new_yaw - cur[:, 6]), dtype=np.float64)
    denom = old_speed * metric_dt + 0.5 * accel * metric_dt * metric_dt
    steering = np.divide(
        delta_yaw, denom, out=np.full_like(delta_yaw, np.inf), where=np.abs(denom) > 1.0e-12
    )
    low_speed = (np.abs(old_speed) < speed_limit) | (np.abs(new_speed) < speed_limit)
    steering = np.where(low_speed, 0.0, steering)
    eps = 1.0e-3
    finite = np.isfinite(accel) & np.isfinite(steering)
    feasible = finite & (np.abs(accel) <= float(max_acc) + eps) & (np.abs(steering) <= float(max_steering) + eps)
    return (
        np.asarray(feasible, dtype=bool),
        np.asarray(accel, dtype=np.float32),
        np.asarray(steering, dtype=np.float32),
        {
            "max_acc_mps2": float(max_acc),
            "max_steering_curvature": float(max_steering),
            "metric_dt_s": float(metric_dt),
            "contract_source": str(source),
        },
    )


def _project_candidate_bank_through_controller_np(
    current: np.ndarray,
    nominal_traj: np.ndarray,
    cfg: dict,
    previous_longitudinal_accel: float,
    first_accel_override: np.ndarray | None = None,
    longitudinal_envelope_mode: np.ndarray | None = None,
    longitudinal_envelope_schedule: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project an entire candidate bank through the *same stateful controller*.

    V16.8.34 asked whether the nominal first waypoint could be reached without
    projection.  That rejects precisely the trajectories for which the online
    controller is designed to apply a bounded correction.  V16.8.35 instead asks
    what trajectory will actually be realized if that controller is applied
    repeatedly, carrying longitudinal-acceleration memory at every step.

    The loop is over horizon only; all candidates are propagated in parallel.
    Returns ``(projected_traj, waymax_kinematic_ok[K,H], accel[K,H])``.

    ``first_accel_override`` and ``longitudinal_envelope_mode`` are v16.8.38
    support-construction hooks.  V16.8.39 adds
    ``longitudinal_envelope_schedule[K,H]`` so a causal, event-derived recovery
    policy can use an existing reachable-interval endpoint only through the
    predicted conflict window and then return to the nominal controller.  Every
    schedule entry is still one of {-1,0,+1}; it never widens acceleration or
    jerk limits.  Historical callers leave all hooks as ``None`` and retain the
    original path byte-for-byte.
    """
    nominal = np.asarray(nominal_traj, dtype=np.float64)
    cur0 = np.asarray(current, dtype=np.float64).reshape(-1)
    if nominal.ndim != 3 or nominal.shape[2] < 5 or cur0.size < 7:
        k = int(nominal.shape[0]) if nominal.ndim >= 1 else 0
        h = int(nominal.shape[1]) if nominal.ndim >= 2 else 0
        return np.asarray(nominal, dtype=np.float32), np.zeros((k, h), dtype=bool), np.zeros((k, h), dtype=np.float32)
    K, H, D = nominal.shape
    projected = np.asarray(nominal, dtype=np.float32).copy()
    state = np.repeat(cur0[None, :], K, axis=0)
    prev_accel = np.full((K,), float(previous_longitudinal_accel), dtype=np.float64)
    kin_ok = np.zeros((K, H), dtype=bool)
    accel_hist = np.zeros((K, H), dtype=np.float32)

    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1.0e-6)
    cand_cfg = cfg.get("candidate", {})
    wm_cfg = cfg.get("waymax", {})
    max_accel = max(float(cand_cfg.get("max_accel_mps2", 4.0)), 1.0e-6)
    max_decel = max(float(cand_cfg.get("max_decel_mps2", 6.0)), 1.0e-6)
    max_jerk = max(float(cand_cfg.get("max_jerk_mps3", 8.0)), 1.0e-6)
    max_yaw_rate = max(float(cand_cfg.get("max_yaw_rate_rad_s", 1.2)), 1.0e-6)
    max_dyaw = min(float(wm_cfg.get("max_delta_yaw_rad", 0.12)), max_yaw_rate * dt)
    override = None
    if first_accel_override is not None:
        override = np.asarray(first_accel_override, dtype=np.float64).reshape(-1)
        if override.size != K:
            raise ValueError(
                "first_accel_override must contain one value per candidate: "
                f"got {override.size}, expected {K}"
            )
    envelope_mode = None
    if longitudinal_envelope_mode is not None:
        envelope_mode = np.asarray(longitudinal_envelope_mode, dtype=np.int8).reshape(-1)
        if envelope_mode.size != K:
            raise ValueError(
                "longitudinal_envelope_mode must contain one value per candidate: "
                f"got {envelope_mode.size}, expected {K}"
            )
        if bool(np.any(~np.isin(envelope_mode, np.asarray([-1, 0, 1], dtype=np.int8)))):
            raise ValueError("longitudinal_envelope_mode values must be in {-1,0,+1}")
    envelope_schedule = None
    if longitudinal_envelope_schedule is not None:
        envelope_schedule = np.asarray(longitudinal_envelope_schedule, dtype=np.int8)
        if envelope_schedule.ndim == 1:
            if K != 1 or envelope_schedule.size != H:
                raise ValueError(
                    "one-dimensional longitudinal_envelope_schedule is only "
                    f"valid for K=1 and must have H={H} entries"
                )
            envelope_schedule = envelope_schedule[None, :]
        if envelope_schedule.shape != (K, H):
            raise ValueError(
                "longitudinal_envelope_schedule must have shape [K,H]: "
                f"got {envelope_schedule.shape}, expected {(K, H)}"
            )
        if bool(np.any(~np.isin(envelope_schedule, np.asarray([-1, 0, 1], dtype=np.int8)))):
            raise ValueError("longitudinal_envelope_schedule values must be in {-1,0,+1}")
        if envelope_mode is not None:
            raise ValueError(
                "provide either longitudinal_envelope_mode or "
                "longitudinal_envelope_schedule, not both"
            )

    for t in range(H):
        desired = nominal[:, t, :]
        cur_xy = state[:, :2]
        cur_vel = state[:, 3:5]
        cur_yaw = state[:, 6]
        cur_speed = np.maximum(np.linalg.norm(cur_vel, axis=-1), np.maximum(state[:, 5] if state.shape[1] > 5 else 0.0, 0.0))
        desired_vel = desired[:, 3:5]
        desired_speed = np.linalg.norm(desired_vel, axis=-1)
        position_speed = np.linalg.norm(desired[:, :2] - cur_xy, axis=-1) / dt
        desired_speed = np.where(desired_speed < 1.0e-3, position_speed, desired_speed)
        raw_accel = np.clip((desired_speed - cur_speed) / dt, -max_decel, max_accel)
        lo = np.maximum(-max_decel, prev_accel - max_jerk * dt)
        hi = np.minimum(max_accel, prev_accel + max_jerk * dt)
        accel = np.clip(raw_accel, lo, hi)
        active_envelope = envelope_mode
        if envelope_schedule is not None:
            active_envelope = envelope_schedule[:, t]
        if active_envelope is not None:
            accel = np.where(active_envelope < 0, lo, np.where(active_envelope > 0, hi, accel))
        if t == 0 and override is not None:
            finite_override = np.isfinite(override)
            bounded_override = np.clip(override, lo, hi)
            accel = np.where(finite_override, bounded_override, accel)
        next_speed = np.maximum(0.0, cur_speed + accel * dt)

        yaw_from_vel = np.arctan2(desired_vel[:, 1], desired_vel[:, 0])
        desired_yaw = np.where(desired_speed > 0.25, yaw_from_vel, desired[:, 2])
        requested_dyaw = np.asarray(_wrap_angle(desired_yaw - cur_yaw), dtype=np.float64)
        dyaw = np.clip(requested_dyaw, -max_dyaw, max_dyaw)
        next_yaw = np.asarray(_wrap_angle(cur_yaw + dyaw), dtype=np.float64)
        v0 = cur_speed[:, None] * np.stack([np.cos(cur_yaw), np.sin(cur_yaw)], axis=-1)
        v1 = next_speed[:, None] * np.stack([np.cos(next_yaw), np.sin(next_yaw)], axis=-1)
        next_xy = cur_xy + 0.5 * (v0 + v1) * dt
        target = np.concatenate([next_xy, next_yaw[:, None], v1], axis=-1)

        feasible, _inv_acc, _steer, _contract = _waymax_kinematic_transition_np(state, target, cfg)
        kin_ok[:, t] = feasible
        accel_hist[:, t] = accel.astype(np.float32)
        projected[:, t, 0:2] = next_xy.astype(np.float32)
        projected[:, t, 2] = next_yaw.astype(np.float32)
        projected[:, t, 3:5] = v1.astype(np.float32)
        if D > 5:
            projected[:, t, 5] = nominal[:, t, 5].astype(np.float32)
        if D > 6:
            projected[:, t, 6] = nominal[:, t, 6].astype(np.float32)

        state[:, 0:2] = next_xy
        state[:, 3:5] = v1
        state[:, 6] = next_yaw
        if state.shape[1] > 5:
            state[:, 5] = next_speed
        prev_accel = accel
    return projected, kin_ok, accel_hist


def _successor_control_projected_option_profile(
    agent_state: np.ndarray,
    sdc_index: int,
    emitted_target: np.ndarray,
    emitted_accel: float,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
) -> tuple[tuple[int, ...], dict[str, int | float | str]]:
    """Causal successor spectrum over *control-realized* recovery trajectories.

    The candidate generator is unchanged.  Each nominal successor candidate is
    propagated through the already-existing jerk/acceleration/yaw controller, with
    the current emitted acceleration carried as controller memory.  Its semantic
    macro survives horizon ``h`` only while the projected trajectory remains:

      1) nominally valid under the frozen proposal contract;
      2) drivable under the same roadgraph screen;
      3) collision-free under the same causal constant-velocity screen; and
      4) feasible under Waymax's own inverse acceleration/steering-curvature
         kinematics metric for every realized transition up to ``h``.

    This changes the *observable used by recovery selection*, not the proposal,
    conventional-safe label, RCOT/BCOT certificate, or actual controller.
    """
    nxt = _counterfactual_successor_agent_state(agent_state, sdc_index, emitted_target, cfg)
    (
        traj, valid, conventional, macro, _utility, _road_safe_nominal, _collision_safe_nominal,
        _prefix_nominal, _margin_nominal,
    ) = _route_lane_aware_candidates(nxt, int(sdc_index), roadgraph, cfg, other_future_trajs=None)
    traj = np.asarray(traj)
    valid = np.asarray(valid, dtype=bool)
    conventional = np.asarray(conventional, dtype=bool) & valid
    macro = np.asarray(macro, dtype=np.int64)
    if traj.ndim != 3 or traj.shape[0] != valid.shape[0] or traj.shape[1] <= 0:
        return tuple(), {
            "profile_horizon_steps": 0, "valid_candidates": int(valid.sum()),
            "conventional_candidates": int(conventional.sum()), "control_projected_candidates": 0,
        }

    projected, kin_ok, _accel_hist = _project_candidate_bank_through_controller_np(
        nxt[int(sdc_index)], traj, cfg, float(emitted_accel)
    )
    K, H = int(projected.shape[0]), int(projected.shape[1])
    collision_ctx = _prepare_collision_check_context(
        nxt, int(sdc_index), cfg, horizon_steps=H, other_future_trajs=None
    )
    road_ok = np.zeros((K,), dtype=bool)
    collision_prefix = np.zeros((K,), dtype=np.int32)
    kinematic_prefix = np.zeros((K,), dtype=np.int32)
    realized_prefix = np.zeros((K,), dtype=np.int32)
    for i in np.flatnonzero(valid):
        road_ok[i] = bool(_roadgraph_drivable_mask(projected[i], roadgraph))
        if road_ok[i]:
            collision_prefix[i] = int(_collision_audit_against_context(projected[i], collision_ctx)["safe_prefix_steps"])
        # Prefix is the number of consecutive evaluator-feasible realized
        # transitions.  A later recovery option cannot be counted through an
        # earlier kinematic contract violation.
        bad = np.flatnonzero(~kin_ok[i])
        kinematic_prefix[i] = int(bad[0]) if bad.size else H
        realized_prefix[i] = int(min(collision_prefix[i], kinematic_prefix[i])) if road_ok[i] else 0

    best_by_macro: dict[int, int] = {}
    for i in np.flatnonzero(valid & road_ok):
        m = int(macro[i])
        if m == int(MacroType.PAD):
            continue
        p = int(max(realized_prefix[i], 0))
        if p > best_by_macro.get(m, 0):
            best_by_macro[m] = p
    curve = tuple(
        int(sum(int(p) >= h for p in best_by_macro.values())) for h in range(1, H + 1)
    )
    max_acc, max_steer, metric_dt, source = _waymax_kinematics_contract(cfg)
    return curve, {
        "profile_horizon_steps": int(H),
        "recovery_macro_types_h1": int(curve[0]) if curve else 0,
        "recovery_macro_types_full_horizon": int(curve[-1]) if curve else 0,
        "recovery_profile_area": int(sum(curve)),
        "recovery_macro_types_any": int(len(best_by_macro)),
        "valid_candidates": int(valid.sum()),
        "conventional_candidates": int(conventional.sum()),
        "control_projected_candidates": int(valid.sum()),
        "control_projected_roadgraph_safe_candidates": int((valid & road_ok).sum()),
        "control_projected_h1_kinematic_feasible_candidates": int((valid & kin_ok[:, 0]).sum()),
        "control_projected_full_kinematic_feasible_candidates": int((valid & np.all(kin_ok, axis=1)).sum()),
        "control_projected_mean_kinematic_prefix_steps": float(np.mean(kinematic_prefix[valid])) if bool(valid.any()) else 0.0,
        "control_projected_mean_collision_prefix_steps": float(np.mean(collision_prefix[valid & road_ok])) if bool((valid & road_ok).any()) else 0.0,
        "control_projected_max_realized_prefix_steps": int(realized_prefix[valid & road_ok].max()) if bool((valid & road_ok).any()) else 0,
        "waymax_kinematics_max_acc_mps2": float(max_acc),
        "waymax_kinematics_max_steering_curvature": float(max_steer),
        "waymax_kinematics_metric_dt_s": float(metric_dt),
        "waymax_kinematics_contract_source": str(source),
    }


def _kinematic_guarded_profile_relation(
    alt_transition_feasible: bool,
    base_transition_feasible: bool,
    base_profile: tuple[int, ...],
    alt_profile: tuple[int, ...],
) -> tuple[bool, bool, int, int, int]:
    """Partial order with a hard *alternative itself is executable* invariant.

    V16.8.34 allowed ``base=False, alt=False`` to tie on the current transition.
    That is unsuitable once the predicate is the evaluator's actual kinematics
    contract: a recovery action that is itself infeasible must never be entered or
    continued.  Otherwise, profile support remains a pointwise partial order.
    """
    p_strict, p_weak, min_margin, area_delta = _option_profile_relation(base_profile, alt_profile)
    transition_delta = int(bool(alt_transition_feasible)) - int(bool(base_transition_feasible))
    weak = bool(alt_transition_feasible and p_weak)
    strict = bool(weak and (p_strict or transition_delta > 0))
    return strict, weak, int(transition_delta), int(min_margin), int(area_delta)


def _option_profile_relation(
    base_profile: tuple[int, ...],
    alt_profile: tuple[int, ...],
) -> tuple[bool, bool, int, int]:
    """Return strict/weak pointwise dominance and diagnostic margins.

    ``strict`` means alt >= base at every horizon and > at least one horizon.
    ``weak`` permits exact equality and is used only for continuation of an
    already-entered recovery mode.  ``min_margin`` exposes the worst horizon
    difference; ``area_delta`` is diagnostic only and never used for selection.
    """
    n = max(len(base_profile), len(alt_profile))
    b = np.zeros((n,), dtype=np.int32)
    a = np.zeros((n,), dtype=np.int32)
    if base_profile:
        b[: len(base_profile)] = np.asarray(base_profile, dtype=np.int32)
    if alt_profile:
        a[: len(alt_profile)] = np.asarray(alt_profile, dtype=np.int32)
    diff = a - b
    weak = bool(np.all(diff >= 0))
    strict = bool(weak and np.any(diff > 0))
    min_margin = int(diff.min()) if diff.size else 0
    area_delta = int(diff.sum()) if diff.size else 0
    return strict, weak, min_margin, area_delta



def _returnability_current_edge_admissible(
    base_prefix_steps: float,
    alt_prefix_steps: float,
) -> bool:
    """Require the alternative to survive the next real replanning edge.

    Returnability is meaningless if the selected recovery control is already
    collision-unsafe before the next policy invocation.  The alternative must
    therefore retain at least one causal collision-free step and may not reduce
    the current prefix relative to the unchanged COWP fallback.  This is a hard
    feasibility relation, not a tuned prefix margin.
    """
    bp = float(base_prefix_steps)
    ap = float(alt_prefix_steps)
    return bool(
        math.isfinite(bp)
        and math.isfinite(ap)
        and ap >= 1.0 - 1.0e-9
        and ap >= bp - 1.0e-9
    )


def _returnability_relation(
    base_direct_restore: bool,
    base_recourse_macros: frozenset[int],
    alt_direct_restore: bool,
    alt_recourse_macros: frozenset[int],
) -> tuple[bool, bool]:
    """Partial order for the v16.8.37 recourse-returnability witness.

    Direct one-step restoration of the frozen full-conventional set strictly
    dominates a state that still needs a recourse action.  If neither branch
    restores immediately, the alternative must preserve a *superset* of
    witnessed semantic recourse macros, with strict set inclusion for entry.
    Counts are never scalarized; incomparable sets are rejected.
    """
    bd = bool(base_direct_restore)
    ad = bool(alt_direct_restore)
    if bd != ad:
        return bool(ad and not bd), bool(ad and not bd)
    if bd and ad:
        return False, True
    b = frozenset(int(x) for x in base_recourse_macros)
    a = frozenset(int(x) for x in alt_recourse_macros)
    weak = bool(b.issubset(a))
    strict = bool(weak and a != b)
    return strict, weak


def _semantic_action_class_representatives_np(
    pool: np.ndarray,
    macro_types: np.ndarray,
    action_targets: np.ndarray,
    collision_prefix_steps: np.ndarray,
    fallback_scores: np.ndarray | None = None,
    *,
    prefer_fallback: bool = False,
    atol: float = 1.0e-6,
) -> list[int]:
    """Return deterministic representatives of distinct emitted-action classes.

    A semantic macro can contain several physically different emitted controls.  A
    one-representative-per-macro probe is therefore not an existential recourse
    test: its chosen representative may fail even though another action in the
    same macro restores conventional support.  V16.8.37 keeps every distinct
    ``(macro, emitted target)`` class and removes only controls that are physically
    identical up to ``atol``.

    ``prefer_fallback=False`` orders classes by longer current causal prefix and
    then index, which is used for the counterfactual witness.  On the actual bridge
    step, ``prefer_fallback=True`` keeps the frozen COWP fallback preference among
    physically equivalent candidates after all hard gates have passed.
    """
    keep = np.asarray(pool, dtype=bool).reshape(-1)
    macro = np.asarray(macro_types, dtype=np.int64).reshape(-1)
    targets = np.asarray(action_targets, dtype=np.float32)
    prefix = np.asarray(collision_prefix_steps, dtype=np.float64).reshape(-1)
    scores = None if fallback_scores is None else np.asarray(fallback_scores, dtype=np.float64).reshape(-1)
    n = min(keep.size, macro.size, targets.shape[0] if targets.ndim >= 2 else 0, prefix.size)
    if scores is not None:
        n = min(n, scores.size)
    if n <= 0:
        return []
    keep, macro, targets, prefix = keep[:n], macro[:n], targets[:n], prefix[:n]
    if scores is not None:
        scores = scores[:n]

    reps: list[int] = []
    for m in sorted(int(x) for x in np.unique(macro[keep])):
        idx = np.flatnonzero(keep & (macro == int(m))).tolist()
        if prefer_fallback and scores is not None:
            idx.sort(key=lambda i: (
                float(scores[i]) if np.isfinite(scores[i]) else float("inf"),
                -float(prefix[i]) if np.isfinite(prefix[i]) else float("inf"),
                int(i),
            ))
        else:
            idx.sort(key=lambda i: (
                -float(prefix[i]) if np.isfinite(prefix[i]) else float("inf"),
                int(i),
            ))
        chosen_for_macro: list[int] = []
        for i in idx:
            target = np.asarray(targets[int(i)], dtype=np.float32)
            if any(np.allclose(target, targets[int(j)], rtol=0.0, atol=float(atol)) for j in chosen_for_macro):
                continue
            chosen_for_macro.append(int(i))
            reps.append(int(i))
    return reps


def _returnability_witness_signature(
    agent_state: np.ndarray,
    sdc_index: int,
    emitted_target: np.ndarray,
    emitted_accel: float,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
) -> tuple[bool, frozenset[int], dict[str, int | float]]:
    """One-real-replan causal recourse witness used by v16.8.37.

    The first successor uses the *actually emitted* control target.  If the
    unchanged online bank already contains a full-conventional option, the branch
    directly restores the physical feasible region.  Otherwise every distinct
    emitted-action class in each non-PAD semantic macro is tested.  A macro enters
    the witnessed recourse set when at least one of its newly replanned actions,
    projected with the carried controller acceleration, reaches a second causal
    state with a non-empty unchanged full-conventional bank.

    The second edge is a newly generated action at ``s_{t+1}``, never waypoint
    ``t+2`` of the original candidate.  Surrounding agents use the same frozen
    causal propagation as the conventional audit; no logged future is consumed.
    """
    s1 = _counterfactual_successor_agent_state(agent_state, sdc_index, emitted_target, cfg)
    (
        traj1, valid1, conv1, macro1, _utility1, road1, _coll1, prefix1, _margin1,
    ) = _route_lane_aware_candidates(s1, int(sdc_index), roadgraph, cfg, other_future_trajs=None)
    traj1 = np.asarray(traj1)
    valid1 = np.asarray(valid1, dtype=bool)
    conv1 = np.asarray(conv1, dtype=bool) & valid1
    macro1 = np.asarray(macro1, dtype=np.int64)
    road1 = np.asarray(road1, dtype=bool)
    prefix1 = np.asarray(prefix1, dtype=np.float64)
    direct = bool(conv1.any())
    detail: dict[str, int | float] = {
        "successor_valid_candidates": int(valid1.sum()),
        "successor_conventional_candidates": int(conv1.sum()),
        "recourse_candidate_pool": 0,
        "recourse_representatives": 0,
        "recourse_action_classes_available": 0,
        "recourse_action_classes_evaluated": 0,
        "recourse_macros_restoring": 0,
    }
    if direct or traj1.ndim != 3 or traj1.shape[0] == 0:
        return direct, frozenset(), detail

    action_targets1, _action_accels1, _projection_risk1 = _consistent_one_step_targets_np(
        s1[int(sdc_index)], traj1[:, 0, :], cfg, float(emitted_accel)
    )
    kin1, _ia1, _st1, _contract1 = _waymax_kinematic_transition_np(
        s1[int(sdc_index)], action_targets1, cfg
    )
    pool = valid1 & road1 & (prefix1 > 0.0) & np.asarray(kin1, dtype=bool) & (macro1 != int(MacroType.PAD))
    detail["recourse_candidate_pool"] = int(pool.sum())
    detail["recourse_representatives"] = int(np.unique(macro1[pool]).size) if bool(pool.any()) else 0
    reps = _semantic_action_class_representatives_np(
        pool, macro1, action_targets1, prefix1, fallback_scores=None, prefer_fallback=False
    )
    detail["recourse_action_classes_available"] = int(len(reps))
    restoring: set[int] = set()
    for rep in reps:
        rep = int(rep)
        m = int(macro1[rep])
        # Once existence is witnessed for a macro, remaining controls in that
        # macro cannot change the semantic recourse set and are skipped.
        if m in restoring:
            continue
        detail["recourse_action_classes_evaluated"] = int(detail["recourse_action_classes_evaluated"]) + 1
        target = np.asarray(action_targets1[rep], dtype=np.float32)
        s2 = _counterfactual_successor_agent_state(s1, int(sdc_index), target, cfg)
        _t2, v2, c2, _m2, _u2, _r2, _cs2, _p2, _mg2 = _route_lane_aware_candidates(
            s2, int(sdc_index), roadgraph, cfg, other_future_trajs=None
        )
        if bool((np.asarray(v2, dtype=bool) & np.asarray(c2, dtype=bool)).any()):
            restoring.add(m)
    detail["recourse_macros_restoring"] = int(len(restoring))
    return False, frozenset(restoring), detail


def _direct_restoring_candidates_np(
    agent_state: np.ndarray,
    sdc_index: int,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
    cand_valid: np.ndarray,
    roadgraph_safe: np.ndarray,
    collision_prefix_steps: np.ndarray,
    macro: np.ndarray,
    fallback_score: np.ndarray,
    action_targets: np.ndarray,
    *,
    allowed_macros: frozenset[int] | set[int] | None = None,
    minimum_prefix_steps: float | None = None,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Find same-bank action classes that directly restore conventional support.

    This helper runs only on the single actual replanning step after a non-direct
    returnability entry.  It fixes two important witness-consistency requirements:

    1. search every distinct emitted-action class, not only one max-prefix member
       per semantic macro;
    2. never trade away current causal survival relative to the ordinary COWP
       fallback on that actual state.

    When ``allowed_macros`` is supplied, the executed bridge must belong to the
    semantic recourse set witnessed at entry; an arbitrary newly discovered macro
    cannot be substituted post hoc.
    """
    valid = np.asarray(cand_valid, dtype=bool)
    road = np.asarray(roadgraph_safe, dtype=bool)
    prefix = np.asarray(collision_prefix_steps, dtype=np.float64)
    macro = np.asarray(macro, dtype=np.int64)
    scores = np.asarray(fallback_score, dtype=np.float64)
    targets = np.asarray(action_targets, dtype=np.float32)
    kin, _ia, _st, _contract = _waymax_kinematic_transition_np(
        agent_state[int(sdc_index)], targets, cfg
    )
    pool = valid & road & (prefix > 0.0) & np.asarray(kin, dtype=bool) & (macro != int(MacroType.PAD))
    if minimum_prefix_steps is not None and math.isfinite(float(minimum_prefix_steps)):
        pool &= prefix >= (float(minimum_prefix_steps) - 1.0e-9)
    allowed = None if allowed_macros is None else frozenset(int(x) for x in allowed_macros)
    if allowed is not None:
        pool &= np.asarray([int(m) in allowed for m in macro], dtype=bool)

    reps = _semantic_action_class_representatives_np(
        pool, macro, targets, prefix, scores, prefer_fallback=True
    )
    mask = np.zeros_like(valid, dtype=bool)
    evaluated = 0
    for rep in reps:
        rep = int(rep)
        evaluated += 1
        s1 = _counterfactual_successor_agent_state(
            agent_state, int(sdc_index), np.asarray(targets[rep], dtype=np.float32), cfg
        )
        _t1, v1, c1, _m1, _u1, _r1, _cs1, _p1, _mg1 = _route_lane_aware_candidates(
            s1, int(sdc_index), roadgraph, cfg, other_future_trajs=None
        )
        if bool((np.asarray(v1, dtype=bool) & np.asarray(c1, dtype=bool)).any()):
            mask[rep] = True
    return mask, {
        "candidate_pool": int(pool.sum()),
        "representatives": int(np.unique(macro[pool]).size) if bool(pool.any()) else 0,
        "action_classes": int(len(reps)),
        "evaluated": int(evaluated),
        "restoring": int(mask.sum()),
        "minimum_prefix_steps": float(minimum_prefix_steps) if minimum_prefix_steps is not None else -1.0,
        "allowed_macro_count": int(len(allowed)) if allowed is not None else -1,
    }


def _direct_restoring_representatives_np(
    agent_state: np.ndarray,
    sdc_index: int,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
    cand_valid: np.ndarray,
    roadgraph_safe: np.ndarray,
    collision_prefix_steps: np.ndarray,
    macro: np.ndarray,
    fallback_score: np.ndarray,
    action_targets: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Backward-compatible alias for historical focused tests."""
    return _direct_restoring_candidates_np(
        agent_state, sdc_index, roadgraph, cfg, cand_valid, roadgraph_safe,
        collision_prefix_steps, macro, fallback_score, action_targets, **kwargs,
    )



def _shift_append_terminal_reference_np(trajectory: np.ndarray, dt: float) -> np.ndarray:
    """Shift one realized recovery tube and append a causal terminal edge.

    The appended state is a constant-velocity continuation of the final realized
    state.  It is not a logged future and introduces no learned/tuned terminal
    target.  The shifted reference is re-projected from the causal successor before
    it can certify a first action.
    """
    tr = np.asarray(trajectory, dtype=np.float32)
    if tr.ndim != 2 or tr.shape[0] <= 0 or tr.shape[1] < 5:
        return np.asarray(tr, dtype=np.float32).copy()
    out = np.asarray(tr, dtype=np.float32).copy()
    if tr.shape[0] > 1:
        out[:-1] = tr[1:]
    last = np.asarray(tr[-1], dtype=np.float32).copy()
    step_dt = max(float(dt), 1.0e-6)
    last[0:2] = last[0:2] + last[3:5] * step_dt
    speed = float(np.linalg.norm(last[3:5]))
    if speed > 0.25:
        last[2] = np.float32(np.arctan2(float(last[4]), float(last[3])))
    out[-1] = last
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _shift_longitudinal_envelope_schedule_np(
    schedule: np.ndarray,
    policy_id: int,
) -> np.ndarray:
    """Shift one envelope schedule with the same terminal semantics as V39.

    A finite event-release policy returns to the nominal controller after its
    release event, so the newly appended terminal edge is zero.  In contrast,
    ``LOWER_ALL``/``UPPER_ALL`` are all-horizon policies: after executing one
    edge, their shifted witness must append the same reachable endpoint.

    Centralising this rule prevents a hard-certificate mismatch between the V39
    nested constructor and V40 interval completion.  The helper changes no
    controller limit and consumes no future simulator state.
    """
    sch = np.asarray(schedule, dtype=np.int8).reshape(-1)
    shifted = np.zeros_like(sch, dtype=np.int8)
    if sch.size > 1:
        shifted[:-1] = sch[1:]
    pid = int(policy_id)
    if sch.size and pid in {-1, 1}:
        shifted[-1] = np.int8(-1 if pid < 0 else 1)
    return shifted


def _physical_recovery_tube_certificate_np(
    agent_state: np.ndarray,
    sdc_index: int,
    trajectory: np.ndarray,
    waymax_kinematic_ok: np.ndarray,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
    *,
    collision_context: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, int | float | bool | str]]:
    """Audit one realized tube against the frozen physical contracts.

    This certificate is intentionally orthogonal to COWP's social certificate.  It
    never relabels the parent candidate as conventional-safe or NCF.  It checks the
    realized control tube under the unchanged roadgraph screen, causal CV collision
    model, and Waymax-aligned inverse-dynamics feasibility.
    """
    tr = np.asarray(trajectory, dtype=np.float32)
    kin = np.asarray(waymax_kinematic_ok, dtype=bool).reshape(-1)
    if tr.ndim != 2 or tr.shape[0] <= 0 or tr.shape[1] < 5:
        return False, {
            "finite": False,
            "roadgraph_safe": False,
            "collision_safe": False,
            "kinematic_safe": False,
            "collision_prefix_steps": 0,
            "collision_min_margin_m": float("-inf"),
            "collision_violation_source": "invalid_trajectory",
        }
    finite = bool(np.isfinite(tr[:, :5]).all())
    road_safe = bool(finite and _roadgraph_drivable_mask(tr, roadgraph))
    ctx = collision_context
    if ctx is None:
        ctx = _prepare_collision_check_context(
            np.asarray(agent_state, dtype=np.float32), int(sdc_index), cfg,
            horizon_steps=int(tr.shape[0]), other_future_trajs=None,
        )
    coll = _collision_audit_against_context(tr, ctx) if finite else {
        "safe": False,
        "safe_prefix_steps": 0,
        "min_clearance_margin_m": float("-inf"),
        "violation_source": "nonfinite_ego",
    }
    kin_safe = bool(kin.size >= tr.shape[0] and np.all(kin[: tr.shape[0]]))
    certified = bool(finite and road_safe and bool(coll.get("safe", False)) and kin_safe)
    return certified, {
        "finite": bool(finite),
        "roadgraph_safe": bool(road_safe),
        "collision_safe": bool(coll.get("safe", False)),
        "kinematic_safe": bool(kin_safe),
        "collision_prefix_steps": int(coll.get("safe_prefix_steps", 0)),
        "collision_min_margin_m": float(coll.get("min_clearance_margin_m", float("-inf"))),
        "collision_violation_source": str(coll.get("violation_source", "unknown")),
    }


def _construct_shift_closed_control_reachable_tube_np(
    agent_state: np.ndarray,
    sdc_index: int,
    nominal_trajectories: np.ndarray,
    cand_valid: np.ndarray,
    nominal_roadgraph_safe: np.ndarray,
    macro_types: np.ndarray,
    fallback_scores: np.ndarray,
    collision_prefix_steps: np.ndarray,
    action_targets: np.ndarray,
    action_accels: np.ndarray,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
    previous_longitudinal_accel: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Construct a shift-closed, control-reachable recovery tube.

    V16.8.37 asked whether an existing action returns to a non-empty conventional
    bank within at most one newly replanned edge.  The witness was precise but
    almost always empty.  V16.8.38 instead constructs explicit backup support in
    the *current control-reachable set*:

      * one representative is retained for every distinct (semantic macro,
        actually emitted target) class in the unchanged fixed bank;
      * each parent geometry is realized under three parameter-free longitudinal
        policies: unchanged controller, lower acceleration-reachable envelope, and
        upper acceleration-reachable envelope;
      * a first action is admissible only if its full realized tube is physically
        certified and the one-step-shifted tube can be re-projected and certified
        again from the causal successor with carried controller memory.

    The lower/upper envelopes are endpoints of the existing acceleration/jerk
    limits, not new proposal hyperparameters.  Selection is hard-certificate first,
    then the frozen COWP fallback preference, then minimum deviation from the
    nominal emitted acceleration.  No future Waymax state, scalar risk trade-off,
    relaxed conventional threshold, or dwell-time state is used.
    """
    state = np.asarray(agent_state, dtype=np.float32)
    traj = np.asarray(nominal_trajectories, dtype=np.float32)
    valid = np.asarray(cand_valid, dtype=bool).reshape(-1)
    nominal_road = np.asarray(nominal_roadgraph_safe, dtype=bool).reshape(-1)
    macro = np.asarray(macro_types, dtype=np.int64).reshape(-1)
    scores = np.asarray(fallback_scores, dtype=np.float64).reshape(-1)
    prefix = np.asarray(collision_prefix_steps, dtype=np.float64).reshape(-1)
    targets = np.asarray(action_targets, dtype=np.float32)
    nominal_accels = np.asarray(action_accels, dtype=np.float64).reshape(-1)

    detail: dict[str, Any] = {
        "probe_used": True,
        "parent_pool": 0,
        "parent_action_classes": 0,
        "tube_hypotheses_generated": 0,
        "tube_hypotheses_unique_action": 0,
        "tube_full_physically_safe": 0,
        "tube_shift_closed": 0,
        "tube_nominal_shift_closed": 0,
        "tube_lower_envelope_shift_closed": 0,
        "tube_upper_envelope_shift_closed": 0,
        "tube_lifted_only_parent_count": 0,
        "nominal_first_target_max_abs_error": 0.0,
        "selected": False,
        "selected_is_lifted": False,
        "selected_envelope_mode": 0,
        "selected_parent_candidate": -1,
        "selected_parent_macro": -1,
        "selected_parent_macro_name": "NONE",
        "selected_first_accel_delta": 0.0,
        "selected_collision_min_margin_m": -999.0,
        "selected_shift_collision_min_margin_m": -999.0,
        "selected_fallback_score": 0.0,
    }
    if (
        state.ndim != 2 or not (0 <= int(sdc_index) < state.shape[0])
        or traj.ndim != 3 or traj.shape[0] <= 0 or traj.shape[1] <= 0 or traj.shape[2] < 5
    ):
        detail["invalid_input"] = True
        return None, detail
    n = min(
        traj.shape[0], valid.size, nominal_road.size, macro.size, scores.size,
        prefix.size, targets.shape[0] if targets.ndim == 2 else 0, nominal_accels.size,
    )
    if n <= 0:
        detail["invalid_input"] = True
        return None, detail
    traj, valid, nominal_road, macro = traj[:n], valid[:n], nominal_road[:n], macro[:n]
    scores, prefix, targets, nominal_accels = scores[:n], prefix[:n], targets[:n], nominal_accels[:n]

    # Preserve the historical roadgraph-first recovery pool.  If no nominal parent
    # survives it, fall back to all valid parents and re-audit every realized tube.
    pool = valid & nominal_road
    if not bool(pool.any()):
        pool = valid.copy()
    pad_value = int(MacroType.PAD)
    pool &= macro != pad_value
    detail["parent_pool"] = int(pool.sum())
    if not bool(pool.any()):
        return None, detail

    reps = _semantic_action_class_representatives_np(
        pool, macro, targets, prefix, scores, prefer_fallback=True,
    )
    detail["parent_action_classes"] = int(len(reps))
    if not reps:
        return None, detail

    parent_indices: list[int] = []
    modes: list[int] = []
    for idx in reps:
        for mode in (0, -1, 1):
            parent_indices.append(int(idx))
            modes.append(int(mode))
    parent_arr = np.asarray(parent_indices, dtype=np.int64)
    mode_arr = np.asarray(modes, dtype=np.int8)
    expanded_nominal = np.asarray(traj[parent_arr], dtype=np.float32)
    projected, kin_ok, accel_hist = _project_candidate_bank_through_controller_np(
        state[int(sdc_index)], expanded_nominal, cfg,
        float(previous_longitudinal_accel),
        longitudinal_envelope_mode=mode_arr,
    )
    detail["tube_hypotheses_generated"] = int(projected.shape[0])

    # The mode-0 first edge must remain exactly aligned with the online controller.
    mode0 = np.flatnonzero(mode_arr == 0)
    if mode0.size:
        err = np.max(np.abs(projected[mode0, 0, :5] - targets[parent_arr[mode0], :5]), axis=1)
        detail["nominal_first_target_max_abs_error"] = float(np.max(err)) if err.size else 0.0
        if bool(np.any(err > 2.0e-5)):
            raise RuntimeError(
                "Shift-closed tube projector mismatch: nominal first edge differs "
                "from the unchanged online controller target"
            )

    # Several semantic parents can emit the same physical first action.  Keep the
    # best certified witness per actual action so duplicated macro labels cannot
    # fabricate support.
    action_groups: list[list[int]] = []
    for j in range(projected.shape[0]):
        first = np.asarray(projected[j, 0, :5], dtype=np.float32)
        placed = False
        for group in action_groups:
            if np.allclose(first, projected[group[0], 0, :5], rtol=0.0, atol=1.0e-6):
                group.append(int(j))
                placed = True
                break
        if not placed:
            action_groups.append([int(j)])
    detail["tube_hypotheses_unique_action"] = int(len(action_groups))

    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1.0e-6)
    certified_records: list[dict[str, Any]] = []
    current_collision_context = _prepare_collision_check_context(
        state, int(sdc_index), cfg,
        horizon_steps=int(projected.shape[1]), other_future_trajs=None,
    )
    full_safe_count = 0
    shift_closed_count = 0
    shift_closed_by_mode = {-1: 0, 0: 0, 1: 0}
    nominal_closed_parents: set[int] = set()
    lifted_closed_parents: set[int] = set()

    for group in action_groups:
        group_records: list[dict[str, Any]] = []
        for j in group:
            current_ok, current_detail = _physical_recovery_tube_certificate_np(
                state, int(sdc_index), projected[j], kin_ok[j], roadgraph, cfg,
                collision_context=current_collision_context,
            )
            if not current_ok:
                continue
            full_safe_count += 1
            first_target = np.asarray(projected[j, 0, :5], dtype=np.float32)
            first_accel = float(accel_hist[j, 0])
            successor = _counterfactual_successor_agent_state(
                state, int(sdc_index), first_target, cfg,
            )
            shifted_reference = _shift_append_terminal_reference_np(projected[j], dt)
            shifted_projected, shifted_kin_ok, shifted_accel_hist = _project_candidate_bank_through_controller_np(
                successor[int(sdc_index)], shifted_reference[None, ...], cfg,
                first_accel,
                longitudinal_envelope_mode=np.asarray([int(mode_arr[j])], dtype=np.int8),
            )
            shifted_ok, shifted_detail = _physical_recovery_tube_certificate_np(
                successor, int(sdc_index), shifted_projected[0], shifted_kin_ok[0],
                roadgraph, cfg,
            )
            if not shifted_ok:
                continue
            shift_closed_count += 1
            parent = int(parent_arr[j])
            mode = int(mode_arr[j])
            shift_closed_by_mode[mode] = int(shift_closed_by_mode.get(mode, 0)) + 1
            if mode == 0:
                nominal_closed_parents.add(parent)
            else:
                lifted_closed_parents.add(parent)
            nominal_accel = float(nominal_accels[parent])
            record = {
                "expanded_index": int(j),
                "parent_index": parent,
                "macro": int(macro[parent]),
                "mode": mode,
                "trajectory": np.asarray(projected[j], dtype=np.float32),
                "target": first_target,
                "accel": first_accel,
                "accel_history": np.asarray(accel_hist[j], dtype=np.float32),
                "shifted_trajectory": np.asarray(shifted_projected[0], dtype=np.float32),
                "shifted_accel_history": np.asarray(shifted_accel_hist[0], dtype=np.float32),
                "fallback_score": float(scores[parent]) if np.isfinite(scores[parent]) else float("inf"),
                "first_accel_delta": float(first_accel - nominal_accel),
                "current_certificate": current_detail,
                "shifted_certificate": shifted_detail,
            }
            group_records.append(record)

        if group_records:
            # One physical first action may have several backup witnesses.  Keep the
            # mature COWP preference and the least-distorting control lift.
            chosen_group = min(
                group_records,
                key=lambda r: (
                    float(r["fallback_score"]),
                    abs(float(r["first_accel_delta"])),
                    0 if int(r["mode"]) == 0 else 1,
                    int(r["parent_index"]),
                    int(r["mode"]),
                ),
            )
            certified_records.append(chosen_group)

    detail["tube_full_physically_safe"] = int(full_safe_count)
    detail["tube_shift_closed"] = int(shift_closed_count)
    detail["tube_nominal_shift_closed"] = int(shift_closed_by_mode.get(0, 0))
    detail["tube_lower_envelope_shift_closed"] = int(shift_closed_by_mode.get(-1, 0))
    detail["tube_upper_envelope_shift_closed"] = int(shift_closed_by_mode.get(1, 0))
    detail["tube_lifted_only_parent_count"] = int(len(lifted_closed_parents - nominal_closed_parents))
    if not certified_records:
        return None, detail

    selected = min(
        certified_records,
        key=lambda r: (
            float(r["fallback_score"]),
            abs(float(r["first_accel_delta"])),
            0 if int(r["mode"]) == 0 else 1,
            int(r["parent_index"]),
            int(r["mode"]),
        ),
    )
    detail.update({
        "selected": True,
        "selected_is_lifted": bool(int(selected["mode"]) != 0),
        "selected_envelope_mode": int(selected["mode"]),
        "selected_parent_candidate": int(selected["parent_index"]),
        "selected_parent_macro": int(selected["macro"]),
        "selected_parent_macro_name": _macro_name(int(selected["macro"])),
        "selected_first_accel_delta": float(selected["first_accel_delta"]),
        "selected_collision_min_margin_m": float(selected["current_certificate"].get("collision_min_margin_m", -999.0)),
        "selected_shift_collision_min_margin_m": float(selected["shifted_certificate"].get("collision_min_margin_m", -999.0)),
        "selected_fallback_score": float(selected["fallback_score"]),
    })
    return selected, detail


def _conflict_window_envelope_schedule_family(
    horizon_steps: int,
    first_violation_step: int,
    last_violation_step: int,
) -> list[dict[str, Any]]:
    """Parameter-free control schedules anchored to causal conflict events.

    The V38 constant policies are retained exactly.  If the nominal realized tube
    violates the frozen collision screen, V39 additionally applies the lower or
    upper reachable acceleration endpoint from the current edge through either the
    first or the last sampled violation, then releases to the unchanged nominal
    controller.  Duplicate schedules (for example a one-sample conflict window)
    are removed exactly rather than inflated as separate support.
    """
    H = max(int(horizon_steps), 0)
    if H <= 0:
        return []
    records: list[dict[str, Any]] = []
    seen: set[bytes] = set()

    def add(policy_id: int, policy_name: str, schedule: np.ndarray, release_edge: int) -> None:
        sch = np.asarray(schedule, dtype=np.int8).reshape(-1)
        if sch.size != H:
            raise ValueError(f"conflict-window schedule has {sch.size} edges, expected {H}")
        if bool(np.any(~np.isin(sch, np.asarray([-1, 0, 1], dtype=np.int8)))):
            raise ValueError("conflict-window schedule values must be in {-1,0,+1}")
        key = sch.tobytes()
        if key in seen:
            return
        seen.add(key)
        records.append({
            "policy_id": int(policy_id),
            "policy_name": str(policy_name),
            "schedule": sch,
            "release_edge": int(release_edge),
            "nonnominal_edges": int(np.count_nonzero(sch)),
            "event_release": bool(abs(int(policy_id)) >= 2),
        })

    zero = np.zeros((H,), dtype=np.int8)
    add(0, "NOMINAL", zero, 0)
    add(-1, "LOWER_ALL", -np.ones((H,), dtype=np.int8), H)
    add(1, "UPPER_ALL", np.ones((H,), dtype=np.int8), H)

    if int(first_violation_step) >= 0:
        event_specs = (
            ("FIRST_CONFLICT", -2, 2, int(first_violation_step) + 1),
            ("LAST_CONFLICT", -3, 3, int(last_violation_step) + 1),
        )
        for tag, lower_id, upper_id, raw_release in event_specs:
            release = int(min(max(raw_release, 1), H))
            lower = np.zeros((H,), dtype=np.int8)
            upper = np.zeros((H,), dtype=np.int8)
            lower[:release] = -1
            upper[:release] = 1
            add(lower_id, f"LOWER_TO_{tag}", lower, release)
            add(upper_id, f"UPPER_TO_{tag}", upper, release)
    return records


def _construct_conflict_window_control_reachable_tube_np(
    agent_state: np.ndarray,
    sdc_index: int,
    nominal_trajectories: np.ndarray,
    cand_valid: np.ndarray,
    nominal_roadgraph_safe: np.ndarray,
    macro_types: np.ndarray,
    fallback_scores: np.ndarray,
    collision_prefix_steps: np.ndarray,
    action_targets: np.ndarray,
    action_accels: np.ndarray,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
    previous_longitudinal_accel: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Construct a full-horizon, shift-closed conflict-window recovery tube.

    V38 established that controller-lifted support is real but extremely sparse:
    its three longitudinal policies hold nominal/lower/upper behavior for the
    entire frozen horizon.  V39 retains those exact controls as a nested baseline
    and adds a causal one-switch family.  The switch time is not tuned: it is the
    first or last violation of the nominal realized tube under the unchanged
    sampled CV collision contract.  Endpoint control is applied pre-emptively from
    the current edge through that event and then released to the nominal
    controller.  Every emitted action still requires both the unchanged full
    physical tube certificate and one-step shift closure.
    """
    state = np.asarray(agent_state, dtype=np.float32)
    traj = np.asarray(nominal_trajectories, dtype=np.float32)
    valid = np.asarray(cand_valid, dtype=bool).reshape(-1)
    nominal_road = np.asarray(nominal_roadgraph_safe, dtype=bool).reshape(-1)
    macro = np.asarray(macro_types, dtype=np.int64).reshape(-1)
    scores = np.asarray(fallback_scores, dtype=np.float64).reshape(-1)
    prefix = np.asarray(collision_prefix_steps, dtype=np.float64).reshape(-1)
    targets = np.asarray(action_targets, dtype=np.float32)
    nominal_accels = np.asarray(action_accels, dtype=np.float64).reshape(-1)

    detail: dict[str, Any] = {
        "probe_used": True,
        "parent_pool": 0,
        "parent_action_classes": 0,
        "parents_with_nominal_conflict": 0,
        "mean_parent_first_conflict_step": 0.0,
        "mean_parent_last_conflict_step": 0.0,
        "tube_hypotheses_generated": 0,
        "tube_hypotheses_unique_action": 0,
        "tube_full_physically_safe": 0,
        "tube_shift_closed": 0,
        "tube_nominal_shift_closed": 0,
        "tube_lower_envelope_shift_closed": 0,
        "tube_upper_envelope_shift_closed": 0,
        "tube_event_release_shift_closed": 0,
        "tube_lower_event_release_shift_closed": 0,
        "tube_upper_event_release_shift_closed": 0,
        "tube_lifted_only_parent_count": 0,
        "tube_event_release_only_parent_count": 0,
        "nominal_first_target_max_abs_error": 0.0,
        "selected": False,
        "selected_is_lifted": False,
        "selected_is_event_release": False,
        "selected_policy_id": 0,
        "selected_policy_name": "NONE",
        "selected_envelope_mode": 0,
        "selected_release_edge": 0,
        "selected_nonnominal_edges": 0,
        "selected_parent_candidate": -1,
        "selected_parent_macro": -1,
        "selected_parent_macro_name": "NONE",
        "selected_first_accel_delta": 0.0,
        "selected_collision_min_margin_m": -999.0,
        "selected_shift_collision_min_margin_m": -999.0,
        "selected_fallback_score": 0.0,
    }
    if (
        state.ndim != 2 or not (0 <= int(sdc_index) < state.shape[0])
        or traj.ndim != 3 or traj.shape[0] <= 0 or traj.shape[1] <= 0 or traj.shape[2] < 5
    ):
        detail["invalid_input"] = True
        return None, detail
    n = min(
        traj.shape[0], valid.size, nominal_road.size, macro.size, scores.size,
        prefix.size, targets.shape[0] if targets.ndim == 2 else 0, nominal_accels.size,
    )
    if n <= 0:
        detail["invalid_input"] = True
        return None, detail
    traj, valid, nominal_road, macro = traj[:n], valid[:n], nominal_road[:n], macro[:n]
    scores, prefix, targets, nominal_accels = scores[:n], prefix[:n], targets[:n], nominal_accels[:n]

    pool = valid & nominal_road
    if not bool(pool.any()):
        pool = valid.copy()
    pool &= macro != int(MacroType.PAD)
    detail["parent_pool"] = int(pool.sum())
    if not bool(pool.any()):
        return None, detail

    reps = _semantic_action_class_representatives_np(
        pool, macro, targets, prefix, scores, prefer_fallback=True,
    )
    detail["parent_action_classes"] = int(len(reps))
    if not reps:
        return None, detail

    H = int(traj.shape[1])
    current_collision_context = _prepare_collision_check_context(
        state, int(sdc_index), cfg, horizon_steps=H, other_future_trajs=None,
    )
    rep_arr = np.asarray(reps, dtype=np.int64)
    nominal_projected, _nominal_kin, _nominal_accel_hist = _project_candidate_bank_through_controller_np(
        state[int(sdc_index)], traj[rep_arr], cfg, float(previous_longitudinal_accel),
    )
    nominal_err = np.max(
        np.abs(nominal_projected[:, 0, :5] - targets[rep_arr, :5]), axis=1,
    )
    detail["nominal_first_target_max_abs_error"] = float(np.max(nominal_err)) if nominal_err.size else 0.0
    if bool(np.any(nominal_err > 2.0e-5)):
        raise RuntimeError(
            "Conflict-window tube projector mismatch: nominal first edge differs "
            "from the unchanged online controller target"
        )

    parent_indices: list[int] = []
    policy_ids: list[int] = []
    policy_names: list[str] = []
    release_edges: list[int] = []
    nonnominal_edges: list[int] = []
    schedules: list[np.ndarray] = []
    first_events: list[int] = []
    last_events: list[int] = []
    for local_i, parent in enumerate(reps):
        window = _collision_violation_window_against_context(
            nominal_projected[local_i], current_collision_context,
        )
        first_event = int(window.get("first_violation_step", -1))
        last_event = int(window.get("last_violation_step", -1))
        if bool(window.get("has_violation", False)):
            first_events.append(first_event)
            last_events.append(last_event)
        family = _conflict_window_envelope_schedule_family(H, first_event, last_event)
        for rec in family:
            parent_indices.append(int(parent))
            policy_ids.append(int(rec["policy_id"]))
            policy_names.append(str(rec["policy_name"]))
            release_edges.append(int(rec["release_edge"]))
            nonnominal_edges.append(int(rec["nonnominal_edges"]))
            schedules.append(np.asarray(rec["schedule"], dtype=np.int8))
    detail["parents_with_nominal_conflict"] = int(len(first_events))
    if first_events:
        detail["mean_parent_first_conflict_step"] = float(np.mean(first_events))
        detail["mean_parent_last_conflict_step"] = float(np.mean(last_events))

    parent_arr = np.asarray(parent_indices, dtype=np.int64)
    policy_arr = np.asarray(policy_ids, dtype=np.int8)
    schedule_arr = np.stack(schedules, axis=0).astype(np.int8)
    release_arr = np.asarray(release_edges, dtype=np.int32)
    nonnominal_arr = np.asarray(nonnominal_edges, dtype=np.int32)
    expanded_nominal = np.asarray(traj[parent_arr], dtype=np.float32)
    projected, kin_ok, accel_hist = _project_candidate_bank_through_controller_np(
        state[int(sdc_index)], expanded_nominal, cfg,
        float(previous_longitudinal_accel),
        longitudinal_envelope_schedule=schedule_arr,
    )
    detail["tube_hypotheses_generated"] = int(projected.shape[0])

    action_groups: list[list[int]] = []
    for j in range(projected.shape[0]):
        first = np.asarray(projected[j, 0, :5], dtype=np.float32)
        placed = False
        for group in action_groups:
            if np.allclose(first, projected[group[0], 0, :5], rtol=0.0, atol=1.0e-6):
                group.append(int(j))
                placed = True
                break
        if not placed:
            action_groups.append([int(j)])
    detail["tube_hypotheses_unique_action"] = int(len(action_groups))

    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1.0e-6)
    certified_records: list[dict[str, Any]] = []
    full_safe_count = 0
    shift_closed_count = 0
    closed_by_policy: Counter[str] = Counter()
    nominal_closed_parents: set[int] = set()
    v38_closed_parents: set[int] = set()
    lifted_closed_parents: set[int] = set()
    event_closed_parents: set[int] = set()

    for group in action_groups:
        group_records: list[dict[str, Any]] = []
        for j in group:
            current_ok, current_detail = _physical_recovery_tube_certificate_np(
                state, int(sdc_index), projected[j], kin_ok[j], roadgraph, cfg,
                collision_context=current_collision_context,
            )
            if not current_ok:
                continue
            full_safe_count += 1
            first_target = np.asarray(projected[j, 0, :5], dtype=np.float32)
            first_accel = float(accel_hist[j, 0])
            successor = _counterfactual_successor_agent_state(
                state, int(sdc_index), first_target, cfg,
            )
            shifted_reference = _shift_append_terminal_reference_np(projected[j], dt)
            shifted_schedule = _shift_longitudinal_envelope_schedule_np(
                schedule_arr[j], int(policy_arr[j])
            )
            shifted_projected, shifted_kin_ok, shifted_accel_hist = _project_candidate_bank_through_controller_np(
                successor[int(sdc_index)], shifted_reference[None, ...], cfg,
                first_accel,
                longitudinal_envelope_schedule=shifted_schedule[None, :],
            )
            shifted_ok, shifted_detail = _physical_recovery_tube_certificate_np(
                successor, int(sdc_index), shifted_projected[0], shifted_kin_ok[0],
                roadgraph, cfg,
            )
            if not shifted_ok:
                continue
            shift_closed_count += 1
            parent = int(parent_arr[j])
            policy_id = int(policy_arr[j])
            policy_name = str(policy_names[j])
            closed_by_policy[policy_name] += 1
            if policy_id == 0:
                nominal_closed_parents.add(parent)
                v38_closed_parents.add(parent)
            elif abs(policy_id) == 1:
                v38_closed_parents.add(parent)
                lifted_closed_parents.add(parent)
            else:
                lifted_closed_parents.add(parent)
                event_closed_parents.add(parent)
            nominal_accel = float(nominal_accels[parent])
            record = {
                "expanded_index": int(j),
                "parent_index": parent,
                "macro": int(macro[parent]),
                "policy_id": policy_id,
                "policy_name": policy_name,
                "mode": int(schedule_arr[j, 0]),
                "schedule": np.asarray(schedule_arr[j], dtype=np.int8),
                "release_edge": int(release_arr[j]),
                "nonnominal_edges": int(nonnominal_arr[j]),
                "event_release": bool(abs(policy_id) >= 2),
                "trajectory": np.asarray(projected[j], dtype=np.float32),
                "target": first_target,
                "accel": first_accel,
                "accel_history": np.asarray(accel_hist[j], dtype=np.float32),
                "shifted_trajectory": np.asarray(shifted_projected[0], dtype=np.float32),
                "shifted_accel_history": np.asarray(shifted_accel_hist[0], dtype=np.float32),
                "fallback_score": float(scores[parent]) if np.isfinite(scores[parent]) else float("inf"),
                "first_accel_delta": float(first_accel - nominal_accel),
                "current_certificate": current_detail,
                "shifted_certificate": shifted_detail,
            }
            group_records.append(record)

        if group_records:
            # Same actual first action can own several terminal witnesses.  Frozen
            # COWP preference remains primary; event-release duration only resolves
            # an otherwise identical action/witness tie.
            chosen_group = min(
                group_records,
                key=lambda r: (
                    float(r["fallback_score"]),
                    abs(float(r["first_accel_delta"])),
                    int(r["nonnominal_edges"]),
                    0 if int(r["policy_id"]) == 0 else (1 if bool(r["event_release"]) else 2),
                    int(r["parent_index"]),
                    int(r["policy_id"]),
                ),
            )
            certified_records.append(chosen_group)

    detail["tube_full_physically_safe"] = int(full_safe_count)
    detail["tube_shift_closed"] = int(shift_closed_count)
    detail["tube_nominal_shift_closed"] = int(closed_by_policy.get("NOMINAL", 0))
    detail["tube_lower_envelope_shift_closed"] = int(closed_by_policy.get("LOWER_ALL", 0))
    detail["tube_upper_envelope_shift_closed"] = int(closed_by_policy.get("UPPER_ALL", 0))
    lower_event = int(sum(v for k, v in closed_by_policy.items() if k.startswith("LOWER_TO_")))
    upper_event = int(sum(v for k, v in closed_by_policy.items() if k.startswith("UPPER_TO_")))
    detail["tube_lower_event_release_shift_closed"] = lower_event
    detail["tube_upper_event_release_shift_closed"] = upper_event
    detail["tube_event_release_shift_closed"] = int(lower_event + upper_event)
    detail["tube_lifted_only_parent_count"] = int(
        len(lifted_closed_parents - nominal_closed_parents)
    )
    # V38-support means nominal plus all-horizon lower/upper.  This diagnostic
    # isolates parent geometries rescued only by the new event-release family.
    detail["tube_event_release_only_parent_count"] = int(
        len(event_closed_parents - v38_closed_parents)
    )

    if not certified_records:
        return None, detail
    selected = min(
        certified_records,
        key=lambda r: (
            float(r["fallback_score"]),
            abs(float(r["first_accel_delta"])),
            int(r["nonnominal_edges"]),
            0 if int(r["policy_id"]) == 0 else (1 if bool(r["event_release"]) else 2),
            int(r["parent_index"]),
            int(r["policy_id"]),
        ),
    )
    detail.update({
        "selected": True,
        "selected_is_lifted": bool(int(selected["policy_id"]) != 0),
        "selected_is_event_release": bool(selected["event_release"]),
        "selected_policy_id": int(selected["policy_id"]),
        "selected_policy_name": str(selected["policy_name"]),
        "selected_envelope_mode": int(selected["mode"]),
        "selected_release_edge": int(selected["release_edge"]),
        "selected_nonnominal_edges": int(selected["nonnominal_edges"]),
        "selected_parent_candidate": int(selected["parent_index"]),
        "selected_parent_macro": int(selected["macro"]),
        "selected_parent_macro_name": _macro_name(int(selected["macro"])),
        "selected_first_accel_delta": float(selected["first_accel_delta"]),
        "selected_collision_min_margin_m": float(selected["current_certificate"].get("collision_min_margin_m", -999.0)),
        "selected_shift_collision_min_margin_m": float(selected["shifted_certificate"].get("collision_min_margin_m", -999.0)),
        "selected_fallback_score": float(selected["fallback_score"]),
    })
    return selected, detail


def _constraint_boundary_fractions_from_triplet(
    margin_at_zero: float,
    margin_at_half: float,
    margin_at_one: float,
) -> list[tuple[float, str]]:
    """Return deterministic interior control fractions implied by hard margins.

    The three seed points are the two ends and the canonical midpoint of the
    *existing* first-action acceleration interval.  No outcome or learned score is
    used.  Candidate fractions are only algebraic collision-boundary estimates:

      * secant roots on [0, 1/2] and [1/2, 1] when a hard-margin sign changes;
      * the maximizer and real roots of the unique quadratic interpolant.

    Every returned fraction is subsequently re-projected through the unchanged
    controller and must pass the full current + one-step-shifted physical
    certificate.  The interpolation therefore proposes support but never certifies
    it.
    """
    vals = np.asarray(
        [margin_at_zero, margin_at_half, margin_at_one], dtype=np.float64
    )
    if not bool(np.isfinite(vals).all()):
        return []

    candidates: list[tuple[float, str]] = []
    seeds = ((0.0, float(vals[0])), (0.5, float(vals[1])), (1.0, float(vals[2])))

    for (x0, y0), (x1, y1) in ((seeds[0], seeds[1]), (seeds[1], seeds[2])):
        if y0 == 0.0:
            candidates.append((x0, "secant_exact"))
        if y1 == 0.0:
            candidates.append((x1, "secant_exact"))
        if y0 * y1 < 0.0 and y1 != y0:
            root = x0 - y0 * (x1 - x0) / (y1 - y0)
            candidates.append((float(root), "secant_boundary"))

    m0, mh, m1 = (float(vals[0]), float(vals[1]), float(vals[2]))
    qa = 2.0 * (m0 + m1 - 2.0 * mh)
    qb = (m1 - m0) - qa
    qc = m0
    eps = 64.0 * np.finfo(np.float64).eps

    if abs(qa) > eps:
        vertex = -qb / (2.0 * qa)
        if qa < 0.0 and 0.0 < vertex < 1.0:
            candidates.append((float(vertex), "quadratic_margin_max"))
        disc = qb * qb - 4.0 * qa * qc
        if disc >= 0.0 and np.isfinite(disc):
            root_disc = float(np.sqrt(max(disc, 0.0)))
            for root in ((-qb - root_disc) / (2.0 * qa), (-qb + root_disc) / (2.0 * qa)):
                if 0.0 < root < 1.0:
                    candidates.append((float(root), "quadratic_boundary"))
    elif abs(qb) > eps:
        root = -qc / qb
        if 0.0 < root < 1.0:
            candidates.append((float(root), "linear_boundary"))

    # The endpoints and canonical midpoint have already been evaluated.  Exact
    # deduplication at 1e-10 only removes algebraically duplicate proposals; it is
    # not a learned or outcome-selected control resolution.
    out: list[tuple[float, str]] = []
    for frac, source in sorted(candidates, key=lambda x: (float(x[0]), str(x[1]))):
        f = float(min(max(frac, 0.0), 1.0))
        if min(abs(f - 0.0), abs(f - 0.5), abs(f - 1.0)) <= 1.0e-10:
            continue
        if any(abs(f - old_f) <= 1.0e-10 for old_f, _ in out):
            continue
        out.append((f, str(source)))
    return out


def _construct_shift_closed_first_action_viability_interval_np(
    agent_state: np.ndarray,
    sdc_index: int,
    nominal_trajectories: np.ndarray,
    cand_valid: np.ndarray,
    nominal_roadgraph_safe: np.ndarray,
    macro_types: np.ndarray,
    fallback_scores: np.ndarray,
    collision_prefix_steps: np.ndarray,
    action_targets: np.ndarray,
    action_accels: np.ndarray,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
    previous_longitudinal_accel: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Complete V39 with a continuous first-action viability interval.

    V39 greatly expanded terminal control-sequence witnesses but its actual first
    action remained ternary for every parent: nominal, reachable lower endpoint, or
    reachable upper endpoint.  This constructor preserves every V39-certified
    decision *exactly*.  Only when the nested V39 hard set is empty does it search
    the interior of the already-existing first-step acceleration/jerk interval.

    For each unchanged V39 lifted future witness, the current first acceleration is
    treated as a continuous interval between the parent's nominal emitted
    acceleration and the corresponding lower/upper reachable endpoint.  The two
    ends plus the canonical midpoint are projected first.  Their causal collision
    margins algebraically propose secant/quadratic boundary fractions; every
    proposal is then re-run through the unchanged controller and must satisfy the
    unchanged full physical certificate and one-step shift closure.

    This is support completion, not threshold relaxation: controller limits,
    candidate geometries, collision horizon, roadgraph audit, Waymax kinematics,
    social certificate, and COWP fallback preference are untouched.  No logged
    future, outcome label, scalar risk trade-off, release-time search, or dwell
    state is used.
    """
    nested_selected, nested_detail = _construct_conflict_window_control_reachable_tube_np(
        agent_state,
        sdc_index,
        nominal_trajectories,
        cand_valid,
        nominal_roadgraph_safe,
        macro_types,
        fallback_scores,
        collision_prefix_steps,
        action_targets,
        action_accels,
        roadgraph,
        cfg,
        previous_longitudinal_accel,
    )
    detail = dict(nested_detail)
    detail.update({
        "nested_v39_selected": bool(nested_selected is not None),
        "first_action_interval_completion_attempted": False,
        "first_action_interval_basis_count": 0,
        "first_action_interval_seed_evaluations": 0,
        "first_action_interval_boundary_proposals": 0,
        "first_action_interval_hypotheses_evaluated": 0,
        "first_action_interval_unique_actions": 0,
        "first_action_interval_full_physically_safe": 0,
        "first_action_interval_shift_closed": 0,
        "first_action_interval_only_parent_count": 0,
        "first_action_interval_new_actions": 0,
        "selected_is_first_action_interval_completion": False,
        "selected_is_new_first_action": False,
        "selected_first_accel_fraction": 0.0,
        "selected_boundary_source": "NONE",
    })
    if nested_selected is not None:
        # Strong preservation invariant: V40 cannot perturb any V39-certified
        # decision.  This protects the high-precision V39 layer while testing only
        # the support-empty regime that caused the recall failure.
        return nested_selected, detail

    state = np.asarray(agent_state, dtype=np.float32)
    traj = np.asarray(nominal_trajectories, dtype=np.float32)
    valid = np.asarray(cand_valid, dtype=bool).reshape(-1)
    nominal_road = np.asarray(nominal_roadgraph_safe, dtype=bool).reshape(-1)
    macro = np.asarray(macro_types, dtype=np.int64).reshape(-1)
    scores = np.asarray(fallback_scores, dtype=np.float64).reshape(-1)
    prefix = np.asarray(collision_prefix_steps, dtype=np.float64).reshape(-1)
    targets = np.asarray(action_targets, dtype=np.float32)
    nominal_accels = np.asarray(action_accels, dtype=np.float64).reshape(-1)

    if (
        state.ndim != 2
        or not (0 <= int(sdc_index) < state.shape[0])
        or traj.ndim != 3
        or traj.shape[0] <= 0
        or traj.shape[1] <= 0
        or traj.shape[2] < 5
    ):
        detail["invalid_input"] = True
        return None, detail
    n = min(
        traj.shape[0],
        valid.size,
        nominal_road.size,
        macro.size,
        scores.size,
        prefix.size,
        targets.shape[0] if targets.ndim == 2 else 0,
        nominal_accels.size,
    )
    if n <= 0:
        detail["invalid_input"] = True
        return None, detail
    traj = traj[:n]
    valid = valid[:n]
    nominal_road = nominal_road[:n]
    macro = macro[:n]
    scores = scores[:n]
    prefix = prefix[:n]
    targets = targets[:n]
    nominal_accels = nominal_accels[:n]

    pool = valid & nominal_road
    if not bool(pool.any()):
        pool = valid.copy()
    pool &= macro != int(MacroType.PAD)
    reps = _semantic_action_class_representatives_np(
        pool, macro, targets, prefix, scores, prefer_fallback=True
    )
    if not reps:
        return None, detail

    H = int(traj.shape[1])
    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1.0e-6)
    cand_cfg = cfg.get("candidate", {})
    max_accel = max(float(cand_cfg.get("max_accel_mps2", 4.0)), 1.0e-6)
    max_decel = max(float(cand_cfg.get("max_decel_mps2", 6.0)), 1.0e-6)
    max_jerk = max(float(cand_cfg.get("max_jerk_mps3", 8.0)), 1.0e-6)
    first_lo = float(max(-max_decel, float(previous_longitudinal_accel) - max_jerk * dt))
    first_hi = float(min(max_accel, float(previous_longitudinal_accel) + max_jerk * dt))

    current_collision_context = _prepare_collision_check_context(
        state,
        int(sdc_index),
        cfg,
        horizon_steps=H,
        other_future_trajs=None,
    )
    rep_arr = np.asarray(reps, dtype=np.int64)
    nominal_projected, _, _ = _project_candidate_bank_through_controller_np(
        state[int(sdc_index)],
        traj[rep_arr],
        cfg,
        float(previous_longitudinal_accel),
    )

    bases: list[dict[str, Any]] = []
    seen_basis: set[tuple[int, int, bytes]] = set()
    for local_i, parent in enumerate(reps):
        window = _collision_violation_window_against_context(
            nominal_projected[local_i], current_collision_context
        )
        family = _conflict_window_envelope_schedule_family(
            H,
            int(window.get("first_violation_step", -1)),
            int(window.get("last_violation_step", -1)),
        )
        for rec in family:
            policy_id = int(rec["policy_id"])
            if policy_id == 0:
                continue
            schedule = np.asarray(rec["schedule"], dtype=np.int8)
            direction = -1 if policy_id < 0 else 1
            # The first acceleration is overridden continuously, so schedules that
            # differ only at edge zero are the same future witness.  Deduplicate
            # them exactly.
            key = (int(parent), int(direction), schedule[1:].tobytes())
            if key in seen_basis:
                continue
            seen_basis.add(key)
            bases.append({
                "parent_index": int(parent),
                "macro": int(macro[parent]),
                "policy_id": policy_id,
                "policy_name": str(rec["policy_name"]),
                "direction": int(direction),
                "schedule": schedule,
                "release_edge": int(rec["release_edge"]),
                "nonnominal_edges": int(rec["nonnominal_edges"]),
                "event_release": bool(rec["event_release"]),
                "nominal_accel": float(nominal_accels[parent]),
                "endpoint_accel": float(first_lo if direction < 0 else first_hi),
            })

    detail["first_action_interval_completion_attempted"] = True
    detail["first_action_interval_basis_count"] = int(len(bases))
    if not bases:
        return None, detail

    def build_hypotheses(
        fractions_and_sources: list[list[tuple[float, str]]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        parents: list[int] = []
        schedules: list[np.ndarray] = []
        overrides: list[float] = []
        meta: list[dict[str, Any]] = []
        for basis_i, items in enumerate(fractions_and_sources):
            basis = bases[basis_i]
            for frac, source in items:
                f = float(min(max(float(frac), 0.0), 1.0))
                nominal_a = float(basis["nominal_accel"])
                endpoint_a = float(basis["endpoint_accel"])
                override = nominal_a + f * (endpoint_a - nominal_a)
                parents.append(int(basis["parent_index"]))
                schedules.append(np.asarray(basis["schedule"], dtype=np.int8))
                overrides.append(float(override))
                rec = dict(basis)
                rec.update({
                    "basis_index": int(basis_i),
                    "first_accel_fraction": f,
                    "boundary_source": str(source),
                    "first_accel_override": float(override),
                })
                meta.append(rec)
        if not meta:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, H), dtype=np.int8),
                np.zeros((0,), dtype=np.float64),
                [],
            )
        return (
            np.asarray(parents, dtype=np.int64),
            np.stack(schedules, axis=0).astype(np.int8),
            np.asarray(overrides, dtype=np.float64),
            meta,
        )

    seed_specs = [[(0.0, "interval_endpoint_nominal"), (0.5, "interval_midpoint"), (1.0, "interval_endpoint_reachable")] for _ in bases]
    seed_parent, seed_schedule, seed_override, seed_meta = build_hypotheses(seed_specs)
    detail["first_action_interval_seed_evaluations"] = int(len(seed_meta))

    def project_bank(
        parent_arr: np.ndarray,
        schedule_arr: np.ndarray,
        override_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _project_candidate_bank_through_controller_np(
            state[int(sdc_index)],
            np.asarray(traj[parent_arr], dtype=np.float32),
            cfg,
            float(previous_longitudinal_accel),
            first_accel_override=override_arr,
            longitudinal_envelope_schedule=schedule_arr,
        )

    seed_projected, seed_kin, seed_accel = project_bank(
        seed_parent, seed_schedule, seed_override
    )

    certified_records: list[dict[str, Any]] = []
    current_full_safe = 0
    shift_closed = 0
    collision_margins: list[float] = []
    baseline_first_targets: list[np.ndarray] = [
        np.asarray(targets[int(parent), :5], dtype=np.float32) for parent in reps
    ]

    def evaluate_one(
        j: int,
        parent_arr: np.ndarray,
        schedule_arr: np.ndarray,
        projected: np.ndarray,
        kin_ok: np.ndarray,
        accel_hist: np.ndarray,
        meta: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, float]:
        nonlocal current_full_safe, shift_closed
        current_ok, current_detail = _physical_recovery_tube_certificate_np(
            state,
            int(sdc_index),
            projected[j],
            kin_ok[j],
            roadgraph,
            cfg,
            collision_context=current_collision_context,
        )
        margin = float(current_detail.get("collision_min_margin_m", float("-inf")))
        if not current_ok:
            return None, margin
        current_full_safe += 1
        first_target = np.asarray(projected[j, 0, :5], dtype=np.float32)
        first_accel = float(accel_hist[j, 0])
        successor = _counterfactual_successor_agent_state(
            state, int(sdc_index), first_target, cfg
        )
        shifted_reference = _shift_append_terminal_reference_np(projected[j], dt)
        shifted_schedule = _shift_longitudinal_envelope_schedule_np(
            schedule_arr[j], int(meta[j]["policy_id"])
        )
        shifted_projected, shifted_kin_ok, shifted_accel_hist = _project_candidate_bank_through_controller_np(
            successor[int(sdc_index)],
            shifted_reference[None, ...],
            cfg,
            first_accel,
            longitudinal_envelope_schedule=shifted_schedule[None, :],
        )
        shifted_ok, shifted_detail = _physical_recovery_tube_certificate_np(
            successor,
            int(sdc_index),
            shifted_projected[0],
            shifted_kin_ok[0],
            roadgraph,
            cfg,
        )
        if not shifted_ok:
            shifted_margin = float(
                shifted_detail.get("collision_min_margin_m", float("-inf"))
            )
            return None, float(min(margin, shifted_margin))
        shift_closed += 1
        parent = int(parent_arr[j])
        rec = dict(meta[j])
        nominal_accel = float(nominal_accels[parent])
        is_new = not any(
            np.allclose(first_target, old, rtol=0.0, atol=1.0e-6)
            for old in baseline_first_targets
        )
        rec.update({
            "trajectory": np.asarray(projected[j], dtype=np.float32),
            "target": first_target,
            "accel": first_accel,
            "accel_history": np.asarray(accel_hist[j], dtype=np.float32),
            "shifted_trajectory": np.asarray(shifted_projected[0], dtype=np.float32),
            "shifted_accel_history": np.asarray(shifted_accel_hist[0], dtype=np.float32),
            "fallback_score": float(scores[parent]) if np.isfinite(scores[parent]) else float("inf"),
            "first_accel_delta": float(first_accel - nominal_accel),
            "current_certificate": current_detail,
            "shifted_certificate": shifted_detail,
            "is_new_first_action": bool(is_new),
        })
        return rec, float(
            min(
                margin,
                float(shifted_detail.get("collision_min_margin_m", float("-inf"))),
            )
        )

    seed_records_by_basis: list[list[tuple[float, float]]] = [[] for _ in bases]
    for j, meta_j in enumerate(seed_meta):
        if float(meta_j["first_accel_fraction"]) in {0.0, 1.0}:
            baseline_first_targets.append(
                np.asarray(seed_projected[j, 0, :5], dtype=np.float32)
            )
        rec, margin = evaluate_one(
            j,
            seed_parent,
            seed_schedule,
            seed_projected,
            seed_kin,
            seed_accel,
            seed_meta,
        )
        collision_margins.append(float(margin))
        seed_records_by_basis[int(meta_j["basis_index"])].append(
            (float(meta_j["first_accel_fraction"]), float(margin))
        )
        if rec is not None:
            certified_records.append(rec)

    boundary_specs: list[list[tuple[float, str]]] = []
    for items in seed_records_by_basis:
        by_frac = {round(float(f), 12): float(m) for f, m in items}
        if not all(k in by_frac for k in (0.0, 0.5, 1.0)):
            boundary_specs.append([])
            continue
        boundary_specs.append(
            _constraint_boundary_fractions_from_triplet(
                by_frac[0.0], by_frac[0.5], by_frac[1.0]
            )
        )
    boundary_parent, boundary_schedule, boundary_override, boundary_meta = build_hypotheses(
        boundary_specs
    )
    detail["first_action_interval_boundary_proposals"] = int(len(boundary_meta))
    if boundary_meta:
        boundary_projected, boundary_kin, boundary_accel = project_bank(
            boundary_parent, boundary_schedule, boundary_override
        )
        for j in range(len(boundary_meta)):
            rec, _ = evaluate_one(
                j,
                boundary_parent,
                boundary_schedule,
                boundary_projected,
                boundary_kin,
                boundary_accel,
                boundary_meta,
            )
            if rec is not None:
                certified_records.append(rec)

    detail["first_action_interval_hypotheses_evaluated"] = int(
        len(seed_meta) + len(boundary_meta)
    )
    detail["first_action_interval_full_physically_safe"] = int(current_full_safe)
    detail["first_action_interval_shift_closed"] = int(shift_closed)

    if not certified_records:
        detail["tube_hypotheses_generated"] = int(
            detail.get("tube_hypotheses_generated", 0)
            + detail["first_action_interval_hypotheses_evaluated"]
        )
        detail["tube_full_physically_safe"] = int(
            detail.get("tube_full_physically_safe", 0) + current_full_safe
        )
        detail["tube_shift_closed"] = int(
            detail.get("tube_shift_closed", 0) + shift_closed
        )
        return None, detail

    # Deduplicate by actual emitted first action.  A continuous interval must not
    # fabricate support by attaching many certificates to the same control.
    action_groups: list[list[dict[str, Any]]] = []
    for rec in certified_records:
        placed = False
        for group in action_groups:
            if np.allclose(
                rec["target"], group[0]["target"], rtol=0.0, atol=1.0e-6
            ):
                group.append(rec)
                placed = True
                break
        if not placed:
            action_groups.append([rec])
    detail["first_action_interval_unique_actions"] = int(len(action_groups))

    def witness_key(r: dict[str, Any]) -> tuple[Any, ...]:
        return (
            float(r["fallback_score"]),
            abs(float(r["first_accel_delta"])),
            int(r["nonnominal_edges"]),
            0 if bool(r["event_release"]) else 1,
            abs(float(r["first_accel_fraction"]) - 0.0),
            int(r["parent_index"]),
            int(r["policy_id"]),
            str(r["boundary_source"]),
        )

    unique_records = [min(group, key=witness_key) for group in action_groups]
    new_action_records = [r for r in unique_records if bool(r["is_new_first_action"])]
    interval_parents = {int(r["parent_index"]) for r in new_action_records}
    detail["first_action_interval_only_parent_count"] = int(len(interval_parents))
    detail["first_action_interval_new_actions"] = int(len(new_action_records))

    # V40 is an actual first-action support-completion mechanism.  A certificate
    # attached to an already available nominal/endpoint first action is useful as a
    # diagnostic, but executing it cannot alter the closed-loop transition and must
    # not outrank a genuinely new admissible action through the minimum-distortion
    # tie-break.  V39 has already had the exclusive opportunity to select every
    # endpoint action above.  Therefore V40 fails closed unless the interval adds a
    # new emitted first action.
    if not new_action_records:
        detail["tube_hypotheses_generated"] = int(
            detail.get("tube_hypotheses_generated", 0)
            + detail["first_action_interval_hypotheses_evaluated"]
        )
        detail["tube_hypotheses_unique_action"] = int(
            detail.get("tube_hypotheses_unique_action", 0)
        )
        detail["tube_full_physically_safe"] = int(
            detail.get("tube_full_physically_safe", 0) + current_full_safe
        )
        detail["tube_shift_closed"] = int(
            detail.get("tube_shift_closed", 0) + shift_closed
        )
        return None, detail

    selected = min(new_action_records, key=witness_key)

    detail["tube_hypotheses_generated"] = int(
        detail.get("tube_hypotheses_generated", 0)
        + detail["first_action_interval_hypotheses_evaluated"]
    )
    detail["tube_hypotheses_unique_action"] = int(
        detail.get("tube_hypotheses_unique_action", 0)
        + sum(bool(r["is_new_first_action"]) for r in unique_records)
    )
    detail["tube_full_physically_safe"] = int(
        detail.get("tube_full_physically_safe", 0) + current_full_safe
    )
    detail["tube_shift_closed"] = int(
        detail.get("tube_shift_closed", 0) + shift_closed
    )
    detail.update({
        "selected": True,
        "selected_is_lifted": True,
        "selected_is_event_release": bool(selected["event_release"]),
        "selected_policy_id": int(selected["policy_id"]),
        "selected_policy_name": str(selected["policy_name"]),
        "selected_envelope_mode": int(selected["direction"]),
        "selected_release_edge": int(selected["release_edge"]),
        "selected_nonnominal_edges": int(selected["nonnominal_edges"]),
        "selected_parent_candidate": int(selected["parent_index"]),
        "selected_parent_macro": int(selected["macro"]),
        "selected_parent_macro_name": _macro_name(int(selected["macro"])),
        "selected_first_accel_delta": float(selected["first_accel_delta"]),
        "selected_collision_min_margin_m": float(
            selected["current_certificate"].get("collision_min_margin_m", -999.0)
        ),
        "selected_shift_collision_min_margin_m": float(
            selected["shifted_certificate"].get("collision_min_margin_m", -999.0)
        ),
        "selected_fallback_score": float(selected["fallback_score"]),
        "selected_is_first_action_interval_completion": True,
        "selected_is_new_first_action": bool(selected["is_new_first_action"]),
        "selected_first_accel_fraction": float(selected["first_accel_fraction"]),
        "selected_boundary_source": str(selected["boundary_source"]),
    })
    return selected, detail


def _stable_logistic_np(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.clip(np.asarray(x, dtype=np.float32), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(x_arr))


def _traj_arrays(
    state: Any, *, allow_logged_fallback: bool = False, prefer_logged: bool = False
) -> tuple[Any, int]:
    """Return the causal simulator trajectory unless an oracle explicitly opts in.

    ``log_trajectory`` contains privileged future states in Waymax.  Falling back
    to it implicitly makes a supposedly closed-loop policy non-causal, so the
    main path now refuses that fallback.
    """
    if prefer_logged:
        traj = _get_field(state, ("log_trajectory",))
        if traj is None and allow_logged_fallback:
            traj = _get_field(state, ("sim_trajectory", "trajectory"))
    else:
        traj = _get_field(state, ("sim_trajectory", "trajectory"))
        if traj is None and allow_logged_fallback:
            traj = _get_field(state, ("log_trajectory",))
    timestep = _get_field(state, ("timestep", "time_index", "current_timestep"))
    t = int(_to_numpy(timestep).reshape(-1)[0]) if timestep is not None else 0
    if traj is None:
        raise ValueError(
            "SimulatorState has no causal sim_trajectory/trajectory attribute. "
            "log_trajectory fallback is disabled outside an explicit oracle ablation."
        )
    return traj, t


def _extract_sdc_index(state: Any, default: int = 0) -> int:
    is_sdc = _get_field(state, ("is_sdc", "sdc_mask"))
    if is_sdc is None:
        meta = _get_field(state, ("object_metadata", "metadata"))
        is_sdc = _get_field(meta, ("is_sdc",)) if meta is not None else None
    if is_sdc is not None:
        sdc_arr = _to_numpy(is_sdc)
        while sdc_arr.ndim > 1:
            sdc_arr = sdc_arr[0]
        if sdc_arr.size:
            return int(np.argmax(sdc_arr.astype(float)))
    return int(default)


def _extract_traj_components(
    state: Any, *, allow_logged_fallback: bool = False, prefer_logged: bool = False, traj: Any | None = None
) -> dict[str, np.ndarray]:
    if traj is None:
        traj, _ = _traj_arrays(
            state, allow_logged_fallback=allow_logged_fallback, prefer_logged=prefer_logged
        )
    raw = [
        _get_field(traj, ("x", "center_x")),
        _get_field(traj, ("y", "center_y")),
        _get_field(traj, ("yaw", "heading", "bbox_yaw")),
        _get_field(traj, ("vel_x", "velocity_x", "vx")),
        _get_field(traj, ("vel_y", "velocity_y", "vy")),
        _get_field(traj, ("length",)),
        _get_field(traj, ("width",)),
        _get_field(traj, ("height",)),
        _get_field(traj, ("valid",)),
    ]
    if raw[0] is None or raw[1] is None:
        raise ValueError("trajectory must expose x/y components")
    host = _device_get_many(raw)
    x = _unwrap_batch_dim(np.asarray(host[0]))
    y = _unwrap_batch_dim(np.asarray(host[1]))
    zeros = np.zeros_like(x, dtype=np.float32)

    def arr(i: int, default: np.ndarray, *, boolean: bool = False) -> np.ndarray:
        if raw[i] is None:
            return default.copy()
        a = _unwrap_batch_dim(np.asarray(host[i]))
        return a.astype(bool) if boolean else a.astype(np.float32)

    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "yaw": arr(2, zeros),
        "vx": arr(3, zeros),
        "vy": arr(4, zeros),
        "length": arr(5, np.full_like(x, 4.8, dtype=np.float32)),
        "width": arr(6, np.full_like(x, 1.9, dtype=np.float32)),
        "height": arr(7, np.full_like(x, 1.6, dtype=np.float32)),
        "valid": arr(8, np.ones_like(x, dtype=bool), boolean=True),
    }


def _state11_at(components: dict[str, np.ndarray], t: int) -> np.ndarray:
    x = components["x"]
    if x.ndim == 2:
        tt = int(np.clip(t, 0, x.shape[1] - 1))
        cols = [components[k][:, tt] for k in ("x", "y", "yaw", "vx", "vy", "length", "width", "height", "valid")]
    else:
        cols = [components[k] for k in ("x", "y", "yaw", "vx", "vy", "length", "width", "height", "valid")]
    x_c, y_c, yaw_c, vx_c, vy_c, l_c, w_c, h_c, v_c = cols
    N = int(np.asarray(x_c).shape[0])
    out = np.zeros((N, 11), dtype=np.float32)
    out[:, 0] = np.nan_to_num(x_c, nan=0.0)
    out[:, 1] = np.nan_to_num(y_c, nan=0.0)
    out[:, 3] = np.nan_to_num(vx_c, nan=0.0)
    out[:, 4] = np.nan_to_num(vy_c, nan=0.0)
    out[:, 5] = np.linalg.norm(out[:, 3:5], axis=-1)
    out[:, 6] = np.nan_to_num(yaw_c, nan=0.0)
    out[:, 7] = np.where(np.asarray(l_c) > 0, l_c, 4.8)
    out[:, 8] = np.where(np.asarray(w_c) > 0, w_c, 1.9)
    out[:, 9] = np.where(np.asarray(h_c) > 0, h_c, 1.6)
    out[:, 10] = np.asarray(v_c).astype(bool).astype(np.float32)
    return out


def extract_current_agent_state(state: Any) -> tuple[np.ndarray, int]:
    """Best-effort extraction of [N,11] current states from a Waymax SimulatorState."""
    _, t = _traj_arrays(state)
    comps = _extract_traj_components(state)
    return _state11_at(comps, t), _extract_sdc_index(state)


def _history_from_components(
    comps: dict[str, np.ndarray],
    t: int,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build current and history tensors from one host copy of the trajectory tree.

    The v6 online path copied the full Waymax trajectory tree once for history and
    a second time for logged-future checks.  It also called ``_state11_at`` in a
    Python loop for every history frame.  This vectorized helper preserves exactly
    the same history values while doing one trajectory extraction and one indexed
    gather per field.
    """
    cur11 = _state11_at(comps, t)
    hist_steps = int(cfg.get("model", cfg).get("history_steps", cfg.get("time", {}).get("history_steps", 11)))
    max_agents = int(cfg.get("limits", {}).get("max_agents", cfg.get("model", cfg).get("max_agents", 128)))
    d_state = int(cfg.get("model", cfg).get("d_state", 11))
    n = min(max_agents, cur11.shape[0])
    hist = np.zeros((max_agents, hist_steps, d_state), dtype=np.float32)
    temporal = np.asarray(comps["x"]).ndim == 2
    if temporal:
        total_t = int(comps["x"].shape[1])
        idx = np.clip(np.arange(t - hist_steps + 1, t + 1), 0, total_t - 1).astype(np.int64)
        def gather(name: str, default: float = 0.0) -> np.ndarray:
            arr = np.asarray(comps.get(name))
            if arr.ndim == 2:
                return np.asarray(arr[:n, idx], dtype=np.float32)
            if arr.ndim == 1:
                return np.repeat(np.asarray(arr[:n, None], dtype=np.float32), hist_steps, axis=1)
            return np.full((n, hist_steps), default, dtype=np.float32)
        x = gather("x"); y = gather("y"); yaw = gather("yaw")
        vx = gather("vx"); vy = gather("vy")
        length = gather("length", 4.8); width = gather("width", 1.9); height = gather("height", 1.6)
        valid = gather("valid", 1.0) > 0.5
    else:
        s11 = cur11[:n]
        x = np.repeat(s11[:, 0:1], hist_steps, axis=1)
        y = np.repeat(s11[:, 1:2], hist_steps, axis=1)
        yaw = np.repeat(s11[:, 6:7], hist_steps, axis=1)
        vx = np.repeat(s11[:, 3:4], hist_steps, axis=1)
        vy = np.repeat(s11[:, 4:5], hist_steps, axis=1)
        length = np.repeat(s11[:, 7:8], hist_steps, axis=1)
        width = np.repeat(s11[:, 8:9], hist_steps, axis=1)
        height = np.repeat(s11[:, 9:10], hist_steps, axis=1)
        valid = np.repeat((s11[:, 10:11] > 0.5), hist_steps, axis=1)
    hist[:n, :, 0] = np.nan_to_num(x, nan=0.0)
    hist[:n, :, 1] = np.nan_to_num(y, nan=0.0)
    hist[:n, :, 2] = 0.0
    hist[:n, :, 3] = np.where(length > 0.0, length, 4.8)
    hist[:n, :, 4] = np.where(width > 0.0, width, 1.9)
    hist[:n, :, 5] = np.where(height > 0.0, height, 1.6)
    hist[:n, :, 6] = np.nan_to_num(yaw, nan=0.0)
    hist[:n, :, 7] = np.nan_to_num(vx, nan=0.0)
    hist[:n, :, 8] = np.nan_to_num(vy, nan=0.0)
    hist[:n, :, 9] = np.sqrt(hist[:n, :, 7] ** 2 + hist[:n, :, 8] ** 2)
    hist[:n, :, 10] = valid.astype(np.float32)
    return hist, cur11


def _extract_logged_future_from_components(
    comps: dict[str, np.ndarray], t: int, sdc_index: int, cfg: dict
) -> np.ndarray | None:
    """Extract privileged logged future from an already materialized state tree.

    This is intentionally an *oracle ablation* only.  Main closed-loop evaluation
    must be causal and therefore uses current/history state plus constant-velocity
    fallback or the learned response model, never future simulator/log states.
    """
    H = int(cfg.get("time", {}).get("future_steps", cfg.get("eval", {}).get("rollout_horizon_steps", 80)))
    cur = _state11_at(comps, t)
    n = int(cur.shape[0])
    out = np.zeros((n, H, 7), dtype=np.float32)
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    temporal = np.asarray(comps.get("x", np.zeros(0))).ndim == 2
    total_t = int(comps["x"].shape[1]) if temporal else 0
    for h in range(H):
        if temporal and t + 1 + h < total_t:
            s11 = _state11_at(comps, t + 1 + h)
        else:
            s11 = cur.copy()
            tau = float(h + 1) * dt
            s11[:, :2] = cur[:, :2] + cur[:, 3:5] * tau
        out[:, h, 0:2] = s11[:, 0:2]
        out[:, h, 2] = s11[:, 6]
        out[:, h, 3:5] = s11[:, 3:5]
        out[:, h, 5] = s11[:, 7]
        out[:, h, 6] = s11[:, 8]
    if 0 <= int(sdc_index) < n:
        out[int(sdc_index), :, :] = 0.0
    return out


def _extract_logged_future_agent_trajs(state: Any, sdc_index: int, cfg: dict) -> np.ndarray | None:
    """Compatibility wrapper for explicit ``logged_oracle`` ablations only."""
    source = str(cfg.get("planning", {}).get("online_other_future_source", "constant_velocity")).lower()
    if source not in {"logged", "logged_oracle", "oracle"}:
        return None
    try:
        _, t = _traj_arrays(state, allow_logged_fallback=True, prefer_logged=True)
        comps = _extract_traj_components(
            state, allow_logged_fallback=True, prefer_logged=True
        )
    except Exception:
        return None
    return _extract_logged_future_from_components(comps, t, sdc_index, cfg)


def extract_online_state_bundle(
    state: Any, cfg: dict, *, cached_sdc_index: int | None = None
) -> tuple[np.ndarray, np.ndarray, int, dict[str, np.ndarray], int]:
    """Extract history/current state once for the causal online policy.

    Trajectory leaves are copied from JAX to host as one pytree.  SDC identity is
    invariant within a scenario and may be supplied by the policy cache.
    """
    traj, t = _traj_arrays(state)
    comps = _extract_traj_components(state, traj=traj)
    hist, cur11 = _history_from_components(comps, t, cfg)
    sdc = int(cached_sdc_index) if cached_sdc_index is not None else _extract_sdc_index(state)
    return hist, cur11, sdc, comps, t


def extract_agent_history_model_state(state: Any, cfg: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Backward-compatible history extractor implemented through one vectorized pass."""
    hist, cur11, sdc, _, _ = extract_online_state_bundle(state, cfg)
    return hist, cur11, sdc

def _extract_roadgraph_tokens(state: Any, cfg: dict) -> dict[str, np.ndarray]:
    """Return best-effort WOMD/Waymax roadgraph point tokens.

    Keep feature ids, types and directions when Waymax exposes them; the external
    baseline adapters need those fields to reconstruct lane/crosswalk polylines
    instead of flattening unrelated map elements into one fake feature.
    """
    rg = _get_field(state, ("roadgraph_points", "roadgraph", "roadgraph_static_points"))
    empty = {
        "xy": np.zeros((0, 2), dtype=np.float32),
        "heading": np.zeros(0, dtype=np.float32),
        "valid": np.zeros(0, dtype=bool),
        "types": np.zeros(0, dtype=np.int32),
        "ids": np.zeros(0, dtype=np.int64),
        "dir_xy": np.zeros((0, 2), dtype=np.float32),
    }
    if rg is None:
        return dict(empty)
    x_field = _get_field(rg, ("x", "center_x"))
    y_field = _get_field(rg, ("y", "center_y"))
    xy_field = _get_field(rg, ("xy", "points", "xyz"))
    if x_field is not None and y_field is not None:
        x = _to_numpy(x_field)
        y = _to_numpy(y_field)
        while x.ndim > 1:
            x = x[0]
            y = y[0]
        xy = np.stack([x, y], axis=-1).astype(np.float32)
    elif xy_field is not None:
        arr = _to_numpy(xy_field)
        while arr.ndim > 2:
            arr = arr[0]
        xy = arr[..., :2].reshape(-1, 2).astype(np.float32)
    else:
        return dict(empty)
    dir_x = _get_field(rg, ("dir_x", "direction_x", "dx"))
    dir_y = _get_field(rg, ("dir_y", "direction_y", "dy"))
    dir_xy = np.zeros((len(xy), 2), dtype=np.float32)
    if dir_x is not None and dir_y is not None:
        dx = _to_numpy(dir_x)
        dy = _to_numpy(dir_y)
        while dx.ndim > 1:
            dx = dx[0]
            dy = dy[0]
        dx = dx.reshape(-1)[: len(xy)]
        dy = dy.reshape(-1)[: len(xy)]
        dir_xy[: len(dx), 0] = dx
        dir_xy[: len(dy), 1] = dy
        heading = np.arctan2(dir_xy[:, 1], dir_xy[:, 0]).astype(np.float32)
    else:
        diff = np.gradient(xy, axis=0) if len(xy) > 1 else np.zeros_like(xy)
        dir_xy = diff.astype(np.float32)
        heading = np.arctan2(diff[:, 1], diff[:, 0]).astype(np.float32)
    valid_field = _get_field(rg, ("valid",))
    if valid_field is not None:
        valid = _to_numpy(valid_field)
        while valid.ndim > 1:
            valid = valid[0]
        valid = valid.reshape(-1).astype(bool)[: len(xy)]
    else:
        valid = np.isfinite(xy).all(axis=-1)
    type_field = _get_field(rg, ("types", "type", "map_element_type"))
    if type_field is not None:
        types = _to_numpy(type_field)
        while types.ndim > 1:
            types = types[0]
        types = types.reshape(-1).astype(np.int32)[: len(xy)]
    else:
        types = np.zeros(0, dtype=np.int32)
    id_field = _get_field(rg, ("ids", "id", "roadgraph_id", "map_element_id"))
    if id_field is not None:
        ids = _to_numpy(id_field)
        while ids.ndim > 1:
            ids = ids[0]
        ids = ids.reshape(-1).astype(np.int64)[: len(xy)]
    else:
        ids = np.zeros(0, dtype=np.int64)
    finite = np.isfinite(xy).all(axis=-1) & np.isfinite(heading)
    valid = valid & finite
    max_points = int(cfg.get("limits", {}).get("max_roadgraph_points", 20000))
    if len(xy) > max_points:
        idx = np.linspace(0, len(xy) - 1, max_points, dtype=np.int64)
        xy, heading, valid, dir_xy = xy[idx], heading[idx], valid[idx], dir_xy[idx]
        if len(types) == len(finite):
            types = types[idx]
        if len(ids) == len(finite):
            ids = ids[idx]
    return {"xy": xy, "heading": heading, "valid": valid, "types": types, "ids": ids, "dir_xy": dir_xy}


def _roadgraph_womd_batch_fields(roadgraph: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    """Convert extracted roadgraph tokens to the WOMD tensor-cache key layout."""
    if roadgraph is None:
        return {}
    xy = np.asarray(roadgraph.get("xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    n = int(xy.shape[0])
    if n <= 0:
        return {}
    valid = np.asarray(roadgraph.get("valid", np.ones(n, dtype=bool)), dtype=bool).reshape(-1)[:n]
    out: dict[str, np.ndarray] = {
        "roadgraph_samples/xyz": np.concatenate([xy, np.zeros((n, 1), dtype=np.float32)], axis=-1)[None],
        "roadgraph_samples/valid": valid[None],
    }
    dir_xy = np.asarray(roadgraph.get("dir_xy", np.zeros((n, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    if dir_xy.shape[0] >= n:
        out["roadgraph_samples/dir"] = np.concatenate([dir_xy[:n], np.zeros((n, 1), dtype=np.float32)], axis=-1)[None]
    types = np.asarray(roadgraph.get("types", np.zeros(0, dtype=np.int32))).reshape(-1)
    if types.shape[0] >= n:
        out["roadgraph_samples/type"] = types[:n].astype(np.int64, copy=False)[None]
    ids = np.asarray(roadgraph.get("ids", np.zeros(0, dtype=np.int64))).reshape(-1)
    if ids.shape[0] >= n:
        out["roadgraph_samples/id"] = ids[:n].astype(np.int64, copy=False)[None]
    return out


def _lane_centerline_mask(roadgraph: dict[str, np.ndarray]) -> np.ndarray:
    """Waymax/WOMD lane centerline points only (exclude edges/crosswalks)."""
    valid = roadgraph.get("valid", np.zeros(0, dtype=bool)).astype(bool, copy=False)
    types = roadgraph.get("types", np.zeros(len(valid), dtype=np.int32))
    if len(types) != len(valid) or not np.any(types):
        return valid
    # WOMD RoadGraphSamples: 1 freeway, 2 surface street, 3 bike lane.
    # Vehicle planning intentionally excludes bike-lane centerlines.
    return valid & np.isin(types.astype(np.int32, copy=False), np.asarray([1, 2], dtype=np.int32))


def _nearest_lane_heading(current: np.ndarray, roadgraph: dict[str, np.ndarray], *, search_radius: float = 8.0) -> float:
    xy = roadgraph.get("xy", np.zeros((0, 2), dtype=np.float32))
    valid = _lane_centerline_mask(roadgraph)
    heading = roadgraph.get("heading", np.zeros(0, dtype=np.float32))
    if len(xy) == 0 or not np.any(valid):
        return float(current[6])
    d = np.linalg.norm(xy - current[:2][None], axis=-1)
    mask = valid & (d < float(search_radius))
    if not np.any(mask):
        return float(current[6])
    idx = np.where(mask)[0][np.argmin(d[mask])]
    h = float(heading[idx])
    # Prefer a lane direction aligned with the current vehicle heading.
    if np.cos(h - float(current[6])) < 0.0:
        h = float(_wrap_angle(h + np.pi))
    return h


def _quintic_frenet_trajectory(
    current: np.ndarray,
    horizon: int,
    dt: float,
    *,
    accel: float = 0.0,
    lateral_offset: float = 0.0,
    start_delay_s: float = 0.0,
    lane_change_duration_s: float = 4.0,
) -> np.ndarray:
    """Tangent-consistent Frenet primitive with quintic lateral motion.

    Unlike adding an offset to a longitudinal trajectory and repairing yaw later,
    this primitive derives position, velocity, and yaw from the same differentiable
    curve.  It therefore satisfies the StateDynamics action contract much more
    closely and avoids the dominant kinematic-infeasibility failure mode.
    """
    H = int(horizon)
    dt = float(dt)
    t = (np.arange(H, dtype=np.float32) + 1.0) * dt
    yaw0 = float(current[6])
    v0 = max(float(current[5]), float(np.linalg.norm(current[3:5])), 0.0)
    a = float(accel)
    v = np.maximum(v0 + a * t, 0.0)
    # Exact integral with a stop clamp; cumulative trapezoid is robust when v hits zero.
    v_prev = np.concatenate([np.asarray([v0], dtype=np.float32), v[:-1]])
    s_long = np.cumsum(0.5 * (v_prev + v) * dt).astype(np.float32)

    delay = max(float(start_delay_s), 0.0)
    duration = max(float(lane_change_duration_s), 1.5)
    u = np.clip((t - delay) / duration, 0.0, 1.0)
    q = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    dq_du = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    d_lat = float(lateral_offset) * q
    d_dot = float(lateral_offset) * dq_du / duration
    d_dot[(t <= delay) | (t >= delay + duration)] = 0.0

    e_s = np.asarray([np.cos(yaw0), np.sin(yaw0)], dtype=np.float32)
    e_d = np.asarray([-np.sin(yaw0), np.cos(yaw0)], dtype=np.float32)
    xy = current[:2][None, :] + s_long[:, None] * e_s[None, :] + d_lat[:, None] * e_d[None, :]
    vel = v[:, None] * e_s[None, :] + d_dot[:, None] * e_d[None, :]
    speed = np.linalg.norm(vel, axis=-1)
    yaw = np.where(speed > 0.15, np.arctan2(vel[:, 1], vel[:, 0]), yaw0).astype(np.float32)
    out = np.zeros((H, 7), dtype=np.float32)
    out[:, 0:2] = xy
    out[:, 2] = yaw
    out[:, 3:5] = vel
    out[:, 5] = float(current[7]) if current.shape[0] > 7 else 4.8
    out[:, 6] = float(current[8]) if current.shape[0] > 8 else 1.9
    return out


def _terminal_frenet_trajectory(
    current: np.ndarray,
    H: int,
    dt: float,
    *,
    target_s: float,
    target_speed: float,
    lateral_offset: float = 0.0,
    lane_change_duration_s: float = 4.0,
) -> np.ndarray:
    """Cubic-longitudinal / quintic-lateral terminal primitive."""
    T = max(float(H) * float(dt), float(dt))
    tau = (np.arange(1, H + 1, dtype=np.float32) * dt).astype(np.float32)
    v0 = float(max(current[5], np.linalg.norm(current[3:5]), 0.0))
    vT = float(max(target_speed, 0.0))
    sT = float(max(target_s, 0.0))
    A = np.asarray([[T**3, T**2], [3.0 * T**2, 2.0 * T]], dtype=np.float64)
    b = np.asarray([sT - v0 * T, vT - v0], dtype=np.float64)
    try:
        a3, a2 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        a3, a2 = 0.0, (vT - v0) / max(2.0 * T, 1e-3)
    s_long = (a3 * tau**3 + a2 * tau**2 + v0 * tau).astype(np.float32)
    v = np.maximum((3.0 * a3 * tau**2 + 2.0 * a2 * tau + v0).astype(np.float32), 0.0)
    yaw0 = float(current[6])
    e_s = np.asarray([np.cos(yaw0), np.sin(yaw0)], dtype=np.float32)
    e_d = np.asarray([-np.sin(yaw0), np.cos(yaw0)], dtype=np.float32)
    u = np.clip(tau / max(float(lane_change_duration_s), 1.5), 0.0, 1.0)
    q = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    dq_du = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    d_lat = float(lateral_offset) * q
    d_dot = float(lateral_offset) * dq_du / max(float(lane_change_duration_s), 1.5)
    vel = v[:, None] * e_s[None, :] + d_dot[:, None] * e_d[None, :]
    xy = current[:2][None, :] + s_long[:, None] * e_s[None, :] + d_lat[:, None] * e_d[None, :]
    speed = np.linalg.norm(vel, axis=-1)
    yaw = np.where(speed > 0.15, np.arctan2(vel[:, 1], vel[:, 0]), yaw0).astype(np.float32)
    out = np.zeros((H, 7), dtype=np.float32)
    out[:, 0:2] = xy
    out[:, 2] = yaw
    out[:, 3:5] = vel
    out[:, 5] = float(current[7]) if current.shape[0] > 7 else 4.8
    out[:, 6] = float(current[8]) if current.shape[0] > 8 else 1.9
    return out


def _candidate_dyn_ok(traj: np.ndarray, cfg: dict) -> bool:
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    cand_cfg = cfg.get("candidate", {})
    if traj.ndim != 2 or traj.shape[0] < 2 or traj.shape[1] < 5 or not np.all(np.isfinite(traj)):
        return False
    vel = traj[:, 3:5]
    speed = np.linalg.norm(vel, axis=-1)
    acc_vec = np.diff(vel, axis=0, prepend=vel[:1]) / max(dt, 1e-3)
    acc_long = np.diff(speed, prepend=speed[0]) / max(dt, 1e-3)
    jerk_vec = np.diff(acc_vec, axis=0, prepend=acc_vec[:1]) / max(dt, 1e-3)
    # A constant-acceleration primitive has a one-sample acceleration jump at the
    # beginning when acceleration is estimated from discrete velocities.  Counting
    # that initialization transient as jerk rejected nearly all yield/accelerate
    # online primitives, leaving only ~8 valid candidates and forcing frequent
    # fallback.  Ignore a small configurable prefix and evaluate the true steady
    # trajectory jerk.
    ignore = int(cand_cfg.get("ignore_initial_jerk_steps", cfg.get("planning", {}).get("online_ignore_initial_jerk_steps", 2)))
    jerk_eval = jerk_vec[min(max(ignore, 0), max(len(jerk_vec) - 1, 0)) :] if len(jerk_vec) else jerk_vec
    if jerk_eval.size == 0:
        jerk_eval = jerk_vec
    yaw = np.unwrap(traj[:, 2])
    yaw_rate = np.diff(yaw, prepend=yaw[0]) / max(dt, 1e-3)
    moving = speed > 0.5
    vel_heading = np.arctan2(vel[:, 1], vel[:, 0])
    slip = np.abs(_wrap_angle(vel_heading - traj[:, 2]))
    lateral_acc = np.abs(acc_vec[:, 0] * (-np.sin(traj[:, 2])) + acc_vec[:, 1] * np.cos(traj[:, 2]))
    jerk_norm = np.linalg.norm(jerk_eval, axis=-1) if jerk_eval.size else np.asarray([0.0], dtype=np.float32)
    jerk_stat = float(np.nanpercentile(jerk_norm, float(cand_cfg.get("jerk_check_percentile", 99.0))))
    return bool(
        np.nanmax(acc_long) <= float(cand_cfg.get("max_accel_mps2", 4.0)) + 1e-3
        and -np.nanmin(acc_long) <= float(cand_cfg.get("max_decel_mps2", 6.0)) + 1e-3
        and jerk_stat <= float(cand_cfg.get("max_jerk_mps3", 8.0)) + 1e-3
        and np.nanmax(np.abs(yaw_rate)) <= float(cand_cfg.get("max_yaw_rate_rad_s", 1.2)) + 1e-3
        and np.nanmax(lateral_acc) <= float(cand_cfg.get("max_lateral_accel_mps2", 4.0)) + 1e-3
        and (not np.any(moving) or np.nanmax(slip[moving]) <= float(cand_cfg.get("max_sideslip_rad", 0.20)))
    )


def _roadgraph_drivable_mask(traj: np.ndarray, roadgraph: dict[str, np.ndarray], max_dist: float = 5.5) -> bool:
    xy = roadgraph.get("xy", np.zeros((0, 2), dtype=np.float32))
    valid = _lane_centerline_mask(roadgraph)
    if len(xy) == 0 or not np.any(valid):
        return True
    sample = traj[np.linspace(0, len(traj) - 1, min(8, len(traj)), dtype=np.int64), :2]
    lo = np.nanmin(sample, axis=0) - float(max_dist) - 3.0
    hi = np.nanmax(sample, axis=0) + float(max_dist) + 3.0
    local = valid & (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
    if not np.any(local):
        d0 = np.linalg.norm(xy - sample[0][None, :], axis=-1)
        near = valid & (d0 < 60.0)
        pts = xy[near] if np.any(near) else xy[valid]
    else:
        pts = xy[local]
    if len(pts) > 2048:
        pts = pts[np.linspace(0, len(pts) - 1, 2048, dtype=np.int64)]
    d2 = ((sample[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
    return bool(np.mean(np.sqrt(np.min(d2, axis=1)) <= float(max_dist)) >= 0.75)

def _agent_future_xy(
    agent_state: np.ndarray,
    j: int,
    H: int,
    dt: float,
    other_future_trajs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    ts = (np.arange(1, H + 1, dtype=np.float32)[:, None]) * max(float(dt), 1e-3)
    cv = agent_state[j, :2][None, :].astype(np.float32) + agent_state[j, 3:5][None, :].astype(np.float32) * ts
    logged = cv
    if other_future_trajs is not None and j < int(other_future_trajs.shape[0]) and other_future_trajs.shape[1] > 0:
        logged = np.asarray(other_future_trajs[j, :H, :2], dtype=np.float32)
        if logged.shape[0] < H:
            logged = np.concatenate([logged, cv[logged.shape[0]:]], axis=0)
        bad = ~np.isfinite(logged).all(axis=-1)
        if bad.any():
            logged[bad] = cv[bad]
    return logged, cv


def _prepare_collision_check_context(
    agent_state: np.ndarray,
    sdc_index: int,
    cfg: dict,
    *,
    horizon_steps: int,
    other_future_trajs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Precompute candidate-invariant data for the causal online collision audit.

    v16.8.28 rebuilt the same nearby-agent ranking and constant-velocity future for
    every candidate.  Those quantities depend only on the current simulator state,
    so caching them once per policy step is execution-equivalent and materially
    reduces the CPU candidate-build hot path.  The returned object is deliberately
    plain NumPy/Python data so it never changes device placement or model semantics.
    """
    pcfg = cfg.get("planning", {})
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    H_full = int(horizon_steps)
    H = min(H_full, int(pcfg.get("online_collision_check_horizon_steps", H_full)))
    stride = max(1, int(pcfg.get("online_collision_check_stride", 2)))
    idx = np.arange(0, H, stride, dtype=np.int64)
    if idx.size == 0:
        idx = np.asarray([0], dtype=np.int64)

    ego = agent_state[sdc_index]
    ego_radius = max(float(ego[7]), float(ego[8]), 4.0) * 0.5
    valid = agent_state[:, 10] > 0.5
    ego_xy = ego[:2].astype(np.float32)
    ego_yaw = float(ego[6])
    ego_dir = np.asarray([np.cos(ego_yaw), np.sin(ego_yaw)], dtype=np.float32)
    ego_lat = np.asarray([-ego_dir[1], ego_dir[0]], dtype=np.float32)
    ego_speed = max(float(ego[5]), float(np.linalg.norm(ego[3:5])))
    logged_buffer = float(pcfg.get("online_logged_collision_buffer_m", 0.10))
    cv_buffer = float(pcfg.get("online_priority_cv_collision_buffer_m", 0.35))
    require_cv = bool(pcfg.get("online_require_cv_for_priority_agents", True))
    max_dist = float(pcfg.get("online_collision_agent_radius_m", 60.0))
    max_agents = int(pcfg.get("online_collision_max_agents", 24))

    ranked: list[tuple[int, float, bool]] = []
    for j in range(agent_state.shape[0]):
        if j == sdc_index or not valid[j]:
            continue
        rel = agent_state[j, :2].astype(np.float32) - ego_xy
        dist = float(np.linalg.norm(rel))
        if dist > max_dist:
            continue
        longitudinal = float(np.dot(rel, ego_dir))
        lateral = abs(float(np.dot(rel, ego_lat)))
        rel_speed = float(max(0.0, ego_speed - np.dot(agent_state[j, 3:5], ego_dir)))
        ttc = longitudinal / max(rel_speed, 1e-3) if longitudinal > 0.0 and rel_speed > 0.25 else 99.0
        priority_like = (-8.0 <= longitudinal <= 55.0 and lateral <= 7.5) or (
            ttc <= float(pcfg.get("online_priority_ttc_s", 5.0))
        )
        rank = (0 if priority_like else 1, dist)
        ranked.append((j, float(rank[0]) * 1000.0 + rank[1], priority_like))
    ranked.sort(key=lambda x: x[1])
    if max_agents > 0:
        ranked = ranked[:max_agents]

    agents: list[dict[str, Any]] = []
    for j, _key, priority_like in ranked:
        other_radius = max(float(agent_state[j, 7]), float(agent_state[j, 8]), 4.0) * 0.5
        radius = ego_radius + other_radius + 0.5
        logged, cv = _agent_future_xy(agent_state, j, H_full, dt, other_future_trajs)
        logged_xy = logged[idx]
        cv_xy = cv[idx]
        if not np.isfinite(logged_xy).all():
            logged_xy = cv_xy
        if not np.isfinite(logged_xy).all():
            continue
        agents.append({
            "index": int(j),
            "priority_like": bool(priority_like),
            "logged_xy": np.asarray(logged_xy, dtype=np.float32),
            "cv_xy": np.asarray(cv_xy, dtype=np.float32),
            "base_threshold_m": float(radius + logged_buffer),
            "priority_threshold_m": float(radius + cv_buffer),
        })
    if agents:
        base_xy = np.stack([np.asarray(a["logged_xy"], dtype=np.float32) for a in agents], axis=0)
        cv_xy = np.stack([np.asarray(a["cv_xy"], dtype=np.float32) for a in agents], axis=0)
        base_threshold_m = np.asarray([float(a["base_threshold_m"]) for a in agents], dtype=np.float32)
        priority_threshold_m = np.asarray([float(a["priority_threshold_m"]) for a in agents], dtype=np.float32)
        priority_like = np.asarray([bool(a["priority_like"]) for a in agents], dtype=bool)
    else:
        S = int(idx.size)
        base_xy = np.zeros((0, S, 2), dtype=np.float32)
        cv_xy = np.zeros((0, S, 2), dtype=np.float32)
        base_threshold_m = np.zeros((0,), dtype=np.float32)
        priority_threshold_m = np.zeros((0,), dtype=np.float32)
        priority_like = np.zeros((0,), dtype=bool)
    return {
        "idx": idx,
        "horizon_steps": int(H),
        "full_horizon_steps": int(H_full),
        "require_cv": bool(require_cv),
        "agents": agents,
        # Stacked views remove the Python agent loop from every candidate audit.
        # They are derived from the exact same v16.8.28 agent futures/thresholds.
        "base_xy": base_xy,
        "cv_xy": cv_xy,
        "base_threshold_m": base_threshold_m,
        "priority_threshold_m": priority_threshold_m,
        "priority_like": priority_like,
    }


def _collision_audit_against_context(traj: np.ndarray, context: dict[str, Any]) -> dict[str, Any]:
    """Return the exact old full-horizon decision plus a causal survival prefix.

    ``safe_prefix_steps`` is diagnostic/recovery information only: it is the first
    raw trajectory index at which the existing causal circle/CV screen is violated,
    or the configured collision-check horizon when no violation occurs.  It does
    not promote a candidate to ``conventional_safe`` and therefore cannot weaken
    the v16.8.27 conventional-safety integrity contract.

    v16.8.29 evaluates all preselected nearby agents in one NumPy broadcast.  This
    removes the per-candidate Python agent loop while preserving the v16.8.28
    boolean inequalities and the same sampled time indices.
    """
    idx = np.asarray(context.get("idx", np.asarray([0], dtype=np.int64)), dtype=np.int64)
    H = int(context.get("horizon_steps", len(traj)))
    if idx.size == 0:
        idx = np.asarray([0], dtype=np.int64)
    traj_xy = np.asarray(traj[idx, :2], dtype=np.float32)
    if not np.isfinite(traj_xy).all():
        return {
            "safe": False,
            "safe_prefix_steps": 0,
            "min_clearance_margin_m": float("-inf"),
            "violation_source": "nonfinite_ego",
        }

    base_xy = np.asarray(context.get("base_xy", np.zeros((0, idx.size, 2), dtype=np.float32)), dtype=np.float32)
    if base_xy.shape[0] == 0:
        return {
            "safe": True,
            "safe_prefix_steps": int(H),
            "min_clearance_margin_m": 999.0,
            "violation_source": "none",
        }
    base_thr = np.asarray(context["base_threshold_m"], dtype=np.float32)[:, None]
    d_base = np.linalg.norm(traj_xy[None, :, :] - base_xy, axis=-1)
    base_margin = d_base - base_thr
    min_margin = float(np.min(base_margin))
    base_bad = base_margin < 0.0
    earliest_base = H
    if bool(np.any(base_bad)):
        first_cols = np.flatnonzero(np.any(base_bad, axis=0))
        if first_cols.size:
            earliest_base = int(idx[int(first_cols[0])])

    earliest_priority = H
    require_cv = bool(context.get("require_cv", True))
    priority_like = np.asarray(context.get("priority_like", np.zeros((base_xy.shape[0],), dtype=bool)), dtype=bool)
    if require_cv and bool(np.any(priority_like)):
        cv_xy = np.asarray(context["cv_xy"], dtype=np.float32)[priority_like]
        cv_thr = np.asarray(context["priority_threshold_m"], dtype=np.float32)[priority_like, None]
        d_cv = np.linalg.norm(traj_xy[None, :, :] - cv_xy, axis=-1)
        cv_margin = d_cv - cv_thr
        min_margin = min(min_margin, float(np.min(cv_margin)))
        cv_bad = cv_margin < 0.0
        if bool(np.any(cv_bad)):
            first_cols = np.flatnonzero(np.any(cv_bad, axis=0))
            if first_cols.size:
                earliest_priority = int(idx[int(first_cols[0])])

    earliest = min(earliest_base, earliest_priority)
    if earliest >= H:
        source = "none"
    elif earliest_base <= earliest_priority:
        source = "base_cv"
    else:
        source = "priority_cv_buffer"
    return {
        "safe": bool(earliest >= H),
        "safe_prefix_steps": int(min(max(earliest, 0), H)),
        "min_clearance_margin_m": float(min_margin),
        "violation_source": str(source),
    }


def _collision_violation_window_against_context(
    traj: np.ndarray,
    context: dict[str, Any],
) -> dict[str, int | bool | str]:
    """Return the first/last sampled violation of the frozen collision screen.

    V16.8.39 uses this only to derive *when* an endpoint control envelope may be
    released back to the unchanged nominal controller.  The inequalities,
    sampled indices, priority-agent CV buffer, and causal predictions are exactly
    those used by :func:`_collision_audit_against_context`.  No outcome label or
    logged future enters this event detector.
    """
    tr = np.asarray(traj, dtype=np.float32)
    idx = np.asarray(context.get("idx", np.asarray([0], dtype=np.int64)), dtype=np.int64)
    H = int(context.get("horizon_steps", len(tr)))
    if idx.size == 0:
        idx = np.asarray([0], dtype=np.int64)
    if tr.ndim != 2 or tr.shape[0] <= int(np.max(idx, initial=0)) or tr.shape[1] < 2:
        return {
            "has_violation": True,
            "first_violation_step": 0,
            "last_violation_step": 0,
            "violation_sample_count": 1,
            "source": "invalid_trajectory",
        }
    traj_xy = np.asarray(tr[idx, :2], dtype=np.float32)
    if not np.isfinite(traj_xy).all():
        bad = ~np.isfinite(traj_xy).all(axis=-1)
        cols = np.flatnonzero(bad)
        first = int(idx[int(cols[0])]) if cols.size else 0
        last = int(idx[int(cols[-1])]) if cols.size else first
        return {
            "has_violation": True,
            "first_violation_step": first,
            "last_violation_step": last,
            "violation_sample_count": int(max(cols.size, 1)),
            "source": "nonfinite_ego",
        }

    base_xy = np.asarray(
        context.get("base_xy", np.zeros((0, idx.size, 2), dtype=np.float32)),
        dtype=np.float32,
    )
    if base_xy.shape[0] == 0:
        return {
            "has_violation": False,
            "first_violation_step": -1,
            "last_violation_step": -1,
            "violation_sample_count": 0,
            "source": "none",
        }

    base_thr = np.asarray(context["base_threshold_m"], dtype=np.float32)[:, None]
    base_bad_by_time = np.any(
        np.linalg.norm(traj_xy[None, :, :] - base_xy, axis=-1) - base_thr < 0.0,
        axis=0,
    )
    priority_bad_by_time = np.zeros((idx.size,), dtype=bool)
    require_cv = bool(context.get("require_cv", True))
    priority_like = np.asarray(
        context.get("priority_like", np.zeros((base_xy.shape[0],), dtype=bool)),
        dtype=bool,
    )
    if require_cv and bool(np.any(priority_like)):
        cv_xy = np.asarray(context["cv_xy"], dtype=np.float32)[priority_like]
        cv_thr = np.asarray(context["priority_threshold_m"], dtype=np.float32)[priority_like, None]
        priority_bad_by_time = np.any(
            np.linalg.norm(traj_xy[None, :, :] - cv_xy, axis=-1) - cv_thr < 0.0,
            axis=0,
        )
    bad_by_time = np.asarray(base_bad_by_time | priority_bad_by_time, dtype=bool)
    cols = np.flatnonzero(bad_by_time)
    if cols.size == 0:
        return {
            "has_violation": False,
            "first_violation_step": -1,
            "last_violation_step": -1,
            "violation_sample_count": 0,
            "source": "none",
        }
    first_col = int(cols[0])
    last_col = int(cols[-1])
    source = "priority_cv_buffer" if bool(priority_bad_by_time[first_col]) and not bool(base_bad_by_time[first_col]) else "base_cv"
    return {
        "has_violation": True,
        "first_violation_step": int(min(max(int(idx[first_col]), 0), max(H - 1, 0))),
        "last_violation_step": int(min(max(int(idx[last_col]), 0), max(H - 1, 0))),
        "violation_sample_count": int(cols.size),
        "source": str(source),
    }

def _stable_softmax_np(logits: np.ndarray) -> np.ndarray:
    """Finite, deterministic softmax for online natural-root probabilities."""
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    out = np.zeros_like(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not bool(np.any(finite)):
        return out.astype(np.float32)
    z = np.clip(x[finite] - float(np.max(x[finite])), -80.0, 0.0)
    e = np.exp(z)
    denom = float(np.sum(e))
    if not np.isfinite(denom) or denom <= 0.0:
        return out.astype(np.float32)
    out[finite] = e / denom
    return out.astype(np.float32)


def _canonical_online_root_weights_np(
    logits: np.ndarray,
    valid: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the exact frozen label/SetTransport natural-root probability measure.

    Returns ``(canonical_weight, raw_softmax_weight)``.  The canonical support
    first applies the frozen ``p_min`` threshold with the all-modes fallback,
    then renormalizes and applies the independent probability-floor smoothing.
    Calling the shared label helper prevents online mechanism drift.
    """
    raw = _stable_softmax_np(logits)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    M = min(int(raw.size), int(mask.size))
    if M <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    raw = np.asarray(raw[:M], dtype=np.float32)
    mask = np.asarray(mask[:M], dtype=bool)
    canonical = canonical_root_weights(
        {"valid": mask[None, :], "weight": raw[None, :]}, cfg,
    )[0]
    return np.asarray(canonical, dtype=np.float32), raw


def _extract_object_types_np(state: Any, num_agents: int) -> np.ndarray:
    """Extract Waymax object types without consulting future simulator state.

    Waymax releases have exposed the field as either ``object_types`` or
    ``object_type`` under ``object_metadata``.  Unknown/missing values are mapped
    to ``ObjectType.UNKNOWN``; downstream response checks then use the stricter
    non-vehicle safety profile rather than guessing a vehicle label.
    """
    n = max(int(num_agents), 0)
    out = np.full((n,), int(ObjectType.UNKNOWN), dtype=np.int32)
    meta = _get_field(state, ("object_metadata", "metadata"))
    raw = _get_field(meta, ("object_types", "object_type", "types", "type")) if meta is not None else None
    if raw is None:
        raw = _get_field(state, ("object_types", "object_type"))
    if raw is None or n <= 0:
        return out
    try:
        arr = _to_numpy(raw)
    except Exception:
        return out
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[0] >= n and 2 <= arr.shape[1] <= 8:
        # Be tolerant of one-hot metadata adapters used by focused tests.
        arr = np.argmax(arr, axis=-1)
    else:
        arr = arr.reshape(-1)
    known = {
        int(ObjectType.UNKNOWN), int(ObjectType.VEHICLE), int(ObjectType.PEDESTRIAN),
        int(ObjectType.CYCLIST), int(ObjectType.OTHER),
    }
    for i in range(min(n, int(arr.size))):
        try:
            value = int(arr[i])
        except Exception:
            continue
        out[i] = value if value in known else int(ObjectType.UNKNOWN)
    return out


def _agent_state_after_future_sample_np(current: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Lift one [x,y,yaw,vx,vy,length,width] sample into the online state layout."""
    cur = np.asarray(current, dtype=np.float32).reshape(-1)
    width = max(int(cur.size), 11)
    nxt = np.zeros((width,), dtype=np.float32)
    nxt[: cur.size] = cur
    s = np.asarray(sample, dtype=np.float32).reshape(-1)
    if s.size >= 5:
        nxt[0:2] = s[0:2]
        nxt[3:5] = s[3:5]
        nxt[5] = float(np.linalg.norm(s[3:5]))
        nxt[6] = s[2]
    if s.size >= 7:
        nxt[7] = max(float(s[5]), 0.1)
        nxt[8] = max(float(s[6]), 0.1)
    nxt[10] = 1.0
    return nxt


def _constant_velocity_trajectory_from_state_np(
    agent_state: np.ndarray,
    agent_index: int,
    horizon_steps: int,
    cfg: dict,
) -> np.ndarray | None:
    """Construct the same causal constant-velocity future used by online audit.

    The trajectory is full [x,y,yaw,vx,vy,length,width] geometry so response
    profiles can be checked against non-blocking actors with the unchanged COWP
    unsafe predicate.  No simulator/log future is read.
    """
    state = np.asarray(agent_state, dtype=np.float32)
    j = int(agent_index)
    H = max(int(horizon_steps), 0)
    if (
        state.ndim != 2 or H <= 0 or not (0 <= j < state.shape[0])
        or state.shape[1] < 11 or not bool(state[j, 10] > 0.5)
        or not np.isfinite(state[j, :9]).all()
    ):
        return None
    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1.0e-6)
    ts = np.arange(1, H + 1, dtype=np.float32) * dt
    tr = np.zeros((H, 7), dtype=np.float32)
    tr[:, :2] = state[j, :2][None, :] + state[j, 3:5][None, :] * ts[:, None]
    tr[:, 2] = float(_wrap_angle(float(state[j, 6])))
    tr[:, 3:5] = state[j, 3:5][None, :]
    tr[:, 5] = max(float(state[j, 7]), 0.1)
    tr[:, 6] = max(float(state[j, 8]), 0.1)
    return tr


def _trajectory_waymax_kinematic_safe_np(
    current: np.ndarray,
    trajectory: np.ndarray,
    cfg: dict,
) -> tuple[bool, dict[str, int | float | str]]:
    """Require every response edge to satisfy Waymax's inverse-dynamics metric."""
    tr = np.asarray(trajectory, dtype=np.float32)
    if tr.ndim != 2 or tr.shape[0] <= 0 or tr.shape[1] < 7 or not np.isfinite(tr[:, :7]).all():
        return False, {
            "failure_step": 0,
            "max_abs_accel_mps2": float("inf"),
            "max_abs_steering_curvature": float("inf"),
            "contract_source": "invalid_response_trajectory",
        }
    cur = np.asarray(current, dtype=np.float32).reshape(-1)
    max_acc = 0.0
    max_steer = 0.0
    contract_source = "unknown"
    for t in range(int(tr.shape[0])):
        target = np.asarray(tr[t, [0, 1, 2, 3, 4]], dtype=np.float32)
        feasible, accel, steering, contract = _waymax_kinematic_transition_np(cur, target, cfg)
        contract_source = str(contract.get("contract_source", contract_source))
        if accel.size:
            max_acc = max(max_acc, abs(float(accel[0])))
        if steering.size:
            max_steer = max(max_steer, abs(float(steering[0])))
        if feasible.size == 0 or not bool(feasible[0]):
            return False, {
                "failure_step": int(t),
                "max_abs_accel_mps2": float(max_acc),
                "max_abs_steering_curvature": float(max_steer),
                "contract_source": contract_source,
            }
        cur = _agent_state_after_future_sample_np(cur, tr[t])
    return True, {
        "failure_step": -1,
        "max_abs_accel_mps2": float(max_acc),
        "max_abs_steering_curvature": float(max_steer),
        "contract_source": contract_source,
    }


def _collision_blocking_agent_indices_against_context(
    trajectory: np.ndarray,
    context: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    """Return the exact agents responsible for the frozen circle/CV rejection."""
    tr = np.asarray(trajectory, dtype=np.float32)
    idx = np.asarray(context.get("idx", np.asarray([0], dtype=np.int64)), dtype=np.int64)
    agents = list(context.get("agents", []))
    detail: dict[str, Any] = {
        "nonfinite_ego": False,
        "base_blocker_count": 0,
        "priority_blocker_count": 0,
        "blocker_count": 0,
    }
    if idx.size == 0:
        idx = np.asarray([0], dtype=np.int64)
    if (
        tr.ndim != 2 or tr.shape[1] < 2 or tr.shape[0] <= int(np.max(idx, initial=0))
        or not np.isfinite(tr[idx, :2]).all()
    ):
        detail["nonfinite_ego"] = True
        return [], detail
    base_xy = np.asarray(
        context.get("base_xy", np.zeros((0, idx.size, 2), dtype=np.float32)),
        dtype=np.float32,
    )
    rows = min(len(agents), int(base_xy.shape[0]))
    if rows <= 0:
        return [], detail
    ego_xy = tr[idx, :2]
    base_thr = np.asarray(context.get("base_threshold_m", np.zeros((rows,), dtype=np.float32)), dtype=np.float32)[:rows, None]
    base_bad = np.any(
        np.linalg.norm(ego_xy[None, :, :] - base_xy[:rows], axis=-1) - base_thr < 0.0,
        axis=1,
    )
    priority_bad = np.zeros((rows,), dtype=bool)
    priority_like = np.asarray(context.get("priority_like", np.zeros((rows,), dtype=bool)), dtype=bool)[:rows]
    if bool(context.get("require_cv", True)) and bool(np.any(priority_like)):
        cv_xy = np.asarray(context.get("cv_xy", base_xy), dtype=np.float32)[:rows]
        cv_thr = np.asarray(context.get("priority_threshold_m", np.zeros((rows,), dtype=np.float32)), dtype=np.float32)[:rows, None]
        all_cv_bad = np.any(
            np.linalg.norm(ego_xy[None, :, :] - cv_xy, axis=-1) - cv_thr < 0.0,
            axis=1,
        )
        priority_bad = priority_like & all_cv_bad
    blockers = sorted({
        int(agents[i].get("index", -1))
        for i in range(rows)
        if bool(base_bad[i] or priority_bad[i]) and int(agents[i].get("index", -1)) >= 0
    })
    detail.update({
        "base_blocker_count": int(np.sum(base_bad)),
        "priority_blocker_count": int(np.sum(priority_bad)),
        "blocker_count": int(len(blockers)),
        "blocker_indices": blockers,
    })
    return blockers, detail


def _collision_context_without_agent_indices(
    context: dict[str, Any],
    excluded_agent_indices: set[int] | frozenset[int] | list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """Filter supported responders while retaining every other frozen collision check."""
    excluded = {int(x) for x in excluded_agent_indices}
    agents = list(context.get("agents", []))
    keep = [i for i, a in enumerate(agents) if int(a.get("index", -1)) not in excluded]
    out = dict(context)
    out["agents"] = [dict(agents[i]) for i in keep]
    for key in ("base_xy", "cv_xy", "base_threshold_m", "priority_threshold_m", "priority_like"):
        arr = np.asarray(context.get(key))
        if arr.ndim >= 1 and arr.shape[0] >= len(agents):
            out[key] = np.asarray(arr[keep]).copy()
    return out


def _prepare_interaction_response_support_np(
    agent_state: np.ndarray,
    sdc_index: int,
    critical_track_index: np.ndarray,
    critical_valid: np.ndarray,
    natural_trajectories: np.ndarray,
    natural_logits: np.ndarray,
    object_types: np.ndarray,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Build a causal, root-conditioned response envelope once per planning step.

    The envelope uses only the trained natural decoder and the frozen deterministic
    same-root response bank.  Root support is measured by the *exact* canonical
    label/SetTransport probability distribution: p_min filtering, support
    renormalization and probability-floor smoothing remain distinct operations.
    Geometric deduplication then merges canonical mass without changing its sum.
    Every retained root must own at least one low-burden, drivable and
    Waymax-kinematic profile in both current and one-step-shifted form.
    """
    state = np.asarray(agent_state, dtype=np.float32)
    crit_idx = np.asarray(critical_track_index, dtype=np.int64).reshape(-1)
    crit_valid = np.asarray(critical_valid, dtype=bool).reshape(-1)
    nat = np.asarray(natural_trajectories, dtype=np.float32)
    logits = np.asarray(natural_logits, dtype=np.float32)
    obj = np.asarray(object_types, dtype=np.int32).reshape(-1)
    pcfg = cfg.get("planning", {})
    ncfg = cfg.get("natural", {})
    ncf_cfg = cfg.get("ncf", {})
    rcfg = cfg.get("response", {}).get("root_conditioned_transport", {})
    min_alt_weight = max(float(ncf_cfg.get(
        "min_alt_weight", pcfg.get("set_transport_min_alt_weight", 0.03)
    )), 0.0)
    probability_floor = float(np.clip(ncf_cfg.get(
        "root_probability_floor", pcfg.get("set_transport_probability_floor", 0.02)
    ), 0.0, 0.25))
    required_mass = float(np.clip(
        1.0 - float(pcfg.get(
            "set_transport_cvar_tail_mass", ncf_cfg.get("cvar_tail_mass", 0.25)
        )), 0.0, 1.0,
    ))
    min_roots = max(int(ncfg.get("certificate_min_low_burden_roots", 2)), 1)
    dedup_m = max(float(ncfg.get("root_dedup_mean_distance_m", 0.10)), 0.0)
    max_roots = max(int(rcfg.get("max_roots_per_agent", nat.shape[1] if nat.ndim >= 2 else 0)), 0)
    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1.0e-6)

    aggregate: dict[str, Any] = {
        "minimum_raw_mode_probability": float(min_alt_weight),
        "probability_floor": float(probability_floor),
        "required_root_mass": float(required_mass),
        "minimum_root_count": int(min_roots),
        "critical_slots": 0,
        "agents_ready": 0,
        "agents_rejected_invalid_prediction": 0,
        "agents_rejected_root_count": 0,
        "agents_rejected_root_mass": 0,
        "agents_rejected_profile_feasibility": 0,
        "retained_roots": 0,
        "eligible_profiles": 0,
        "profile_candidates": 0,
        "canonical_support_mass_sum": 0.0,
        "raw_support_mass_sum": 0.0,
    }
    support: dict[int, dict[str, Any]] = {}
    if state.ndim != 2 or nat.ndim != 4 or logits.ndim != 2:
        aggregate["invalid_input"] = True
        return support, aggregate
    A = min(crit_idx.size, crit_valid.size, nat.shape[0], logits.shape[0])
    for slot in range(A):
        if not bool(crit_valid[slot]):
            continue
        agent_index = int(crit_idx[slot])
        if (
            agent_index == int(sdc_index) or not (0 <= agent_index < state.shape[0])
            or state.shape[1] <= 10 or not bool(state[agent_index, 10] > 0.5)
        ):
            continue
        aggregate["critical_slots"] = int(aggregate["critical_slots"]) + 1
        record: dict[str, Any] = {
            "ready": False,
            "reason": "invalid_prediction",
            "slot": int(slot),
            "agent_index": int(agent_index),
            "object_type": int(obj[agent_index]) if agent_index < obj.size else int(ObjectType.UNKNOWN),
            "beta": 0.0,
            "raw_support_mass": 0.0,
            "canonical_support_mass": 0.0,
            "retained_raw_mass": 0.0,
            "retained_mass": 0.0,
            "retained_root_count": 0,
            "roots": [],
        }
        M = min(int(logits.shape[1]), int(nat.shape[1]))
        valid_modes = np.zeros((M,), dtype=bool)
        roots_by_mode: dict[int, np.ndarray] = {}
        for mode in range(M):
            root = np.asarray(nat[slot, mode], dtype=np.float32).copy()
            if root.ndim != 2 or root.shape[0] <= 0 or root.shape[1] < 7 or not np.isfinite(root[:, :5]).all():
                continue
            root[:, 2] = np.asarray(_wrap_angle(root[:, 2]), dtype=np.float32)
            root[:, 5] = max(float(state[agent_index, 7]), 0.1)
            root[:, 6] = max(float(state[agent_index, 8]), 0.1)
            valid_modes[mode] = True
            roots_by_mode[int(mode)] = root
        canonical, raw_probs = _canonical_online_root_weights_np(logits[slot, :M], valid_modes, cfg)
        canonical_support = canonical > 0.0
        if not bool(np.any(canonical_support)):
            aggregate["agents_rejected_invalid_prediction"] = int(aggregate["agents_rejected_invalid_prediction"]) + 1
            support[agent_index] = record
            continue
        raw_support_mass = float(np.sum(raw_probs[canonical_support]))
        canonical_support_mass = float(np.sum(canonical[canonical_support]))
        record["raw_support_mass"] = raw_support_mass
        record["canonical_support_mass"] = canonical_support_mass
        aggregate["raw_support_mass_sum"] = float(aggregate["raw_support_mass_sum"]) + raw_support_mass
        aggregate["canonical_support_mass_sum"] = float(aggregate["canonical_support_mass_sum"]) + canonical_support_mass
        mode_records: list[dict[str, Any]] = []
        for mode in np.flatnonzero(canonical_support):
            mode_records.append({
                "weight": float(canonical[mode]),
                "raw_weight": float(raw_probs[mode]),
                "mode_indices": [int(mode)],
                "trajectory": roots_by_mode[int(mode)],
            })
        mode_records.sort(key=lambda r: (-float(r["weight"]), int(r["mode_indices"][0])))
        deduped: list[dict[str, Any]] = []
        for candidate in mode_records:
            placed = False
            for old in deduped:
                h = min(int(candidate["trajectory"].shape[0]), int(old["trajectory"].shape[0]))
                mean_xy = float(np.mean(np.linalg.norm(
                    candidate["trajectory"][:h, :2] - old["trajectory"][:h, :2], axis=-1,
                ))) if h > 0 else float("inf")
                if mean_xy <= dedup_m:
                    old["weight"] = float(old["weight"]) + float(candidate["weight"])
                    old["raw_weight"] = float(old["raw_weight"]) + float(candidate["raw_weight"])
                    old["mode_indices"].extend(int(x) for x in candidate["mode_indices"])
                    placed = True
                    break
            if not placed:
                deduped.append(candidate)
        eligible = sorted(
            deduped, key=lambda r: (-float(r["weight"]), int(min(r["mode_indices"]))),
        )
        if max_roots > 0:
            eligible = eligible[:max_roots]
        total_eligible_mass = float(sum(float(r["weight"]) for r in eligible))
        total_eligible_raw_mass = float(sum(float(r["raw_weight"]) for r in eligible))
        if len(eligible) < min_roots:
            record["reason"] = "insufficient_root_count"
            record["available_root_count"] = int(len(eligible))
            record["available_root_mass"] = float(total_eligible_mass)
            aggregate["agents_rejected_root_count"] = int(aggregate["agents_rejected_root_count"]) + 1
            support[agent_index] = record
            continue
        if total_eligible_mass + 1.0e-9 < required_mass:
            record["reason"] = "insufficient_root_mass"
            record["available_root_count"] = int(len(eligible))
            record["available_root_mass"] = float(total_eligible_mass)
            aggregate["agents_rejected_root_mass"] = int(aggregate["agents_rejected_root_mass"]) + 1
            support[agent_index] = record
            continue
        selected_roots: list[dict[str, Any]] = []
        cumulative_mass = 0.0
        cumulative_raw_mass = 0.0
        for root_rec in eligible:
            selected_roots.append(root_rec)
            cumulative_mass += float(root_rec["weight"])
            cumulative_raw_mass += float(root_rec["raw_weight"])
            if len(selected_roots) >= min_roots and cumulative_mass + 1.0e-9 >= required_mass:
                break
        object_type = int(record["object_type"])
        beta = adaptive_beta(
            state, object_type, PriorityRelation.AGENT_PRIORITY, cfg,
            use_adaptive=True, ego_index=int(sdc_index),
        )
        record["beta"] = float(beta)
        prepared_roots: list[dict[str, Any]] = []
        all_roots_ready = True
        for root_rec in selected_roots:
            root = np.asarray(root_rec["trajectory"], dtype=np.float32)
            try:
                bank = build_root_recovery_trajectory_bank(root, cfg)
                burdens = prepare_root_recovery_burden_bank(
                    root, bank, cfg, object_type=object_type,
                    rho=PriorityRelation.AGENT_PRIORITY,
                )
            except Exception:
                bank, burdens = [], []
            profile_records: list[dict[str, Any]] = []
            for profile_index, (profile, burden_entry) in enumerate(zip(bank, burdens)):
                aggregate["profile_candidates"] = int(aggregate["profile_candidates"]) + 1
                burden = float(burden_entry[0])
                tr = np.asarray(profile, dtype=np.float32)
                if burden > float(beta) + 1.0e-9 or tr.ndim != 2 or tr.shape[0] <= 0 or tr.shape[1] < 7:
                    continue
                if not bool(_roadgraph_drivable_mask(tr, roadgraph)):
                    continue
                current_kin_ok, current_kin = _trajectory_waymax_kinematic_safe_np(state[agent_index], tr, cfg)
                shifted = _shift_append_terminal_reference_np(tr, dt)
                response_successor = _agent_state_after_future_sample_np(state[agent_index], tr[0])
                shifted_kin_ok, shifted_kin = _trajectory_waymax_kinematic_safe_np(response_successor, shifted, cfg)
                if not (current_kin_ok and shifted_kin_ok and bool(_roadgraph_drivable_mask(shifted, roadgraph))):
                    continue
                profile_records.append({
                    "profile_index": int(profile_index),
                    "trajectory": tr,
                    "shifted_trajectory": shifted,
                    "burden": float(burden),
                    "burden_components": np.asarray(burden_entry[1], dtype=np.float32),
                    "current_kinematic": current_kin,
                    "shifted_kinematic": shifted_kin,
                })
                aggregate["eligible_profiles"] = int(aggregate["eligible_profiles"]) + 1
            profile_records.sort(key=lambda p: (float(p["burden"]), int(p["profile_index"])))
            prepared_roots.append({
                "weight": float(root_rec["weight"]),
                "raw_weight": float(root_rec["raw_weight"]),
                "mode_indices": tuple(sorted(int(x) for x in root_rec["mode_indices"])),
                "trajectory": root,
                "profiles": profile_records,
            })
            if not profile_records:
                all_roots_ready = False
        record.update({
            "retained_raw_mass": float(cumulative_raw_mass),
            "retained_mass": float(cumulative_mass),
            "retained_root_count": int(len(prepared_roots)),
            "available_root_mass": float(total_eligible_mass),
            "available_raw_root_mass": float(total_eligible_raw_mass),
            "roots": prepared_roots,
        })
        aggregate["retained_roots"] = int(aggregate["retained_roots"]) + int(len(prepared_roots))
        if not all_roots_ready:
            record["reason"] = "root_has_no_low_burden_shift_closed_profile"
            aggregate["agents_rejected_profile_feasibility"] = int(aggregate["agents_rejected_profile_feasibility"]) + 1
            support[agent_index] = record
            continue
        record["ready"] = True
        record["reason"] = "ready"
        aggregate["agents_ready"] = int(aggregate["agents_ready"]) + 1
        old = support.get(agent_index)
        if old is None or (not bool(old.get("ready", False))) or float(record["retained_mass"]) > float(old.get("retained_mass", -1.0)):
            support[agent_index] = record
    return support, aggregate


def _merge_interaction_response_support_details_np(
    base: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Exact aggregate merge for disjoint per-agent response-support builds.

    ``_prepare_interaction_response_support_np`` has no cross-agent coupling:
    every count/mass field is a sum over critical slots, while the probability
    and root-policy fields are frozen constants.  BC-IARE online late-bound
    queries explicitly exclude the original critical agents, so preparing only
    the new exact blockers and adding these aggregates is identical to preparing
    the concatenated disjoint slot list.
    """
    out = dict(base)
    additive = {
        "critical_slots",
        "agents_ready",
        "agents_rejected_invalid_prediction",
        "agents_rejected_root_count",
        "agents_rejected_root_mass",
        "agents_rejected_profile_feasibility",
        "retained_roots",
        "eligible_profiles",
        "profile_candidates",
        "canonical_support_mass_sum",
        "raw_support_mass_sum",
    }
    for key in additive:
        av = base.get(key, 0)
        bv = extra.get(key, 0)
        out[key] = av + bv
    if bool(base.get("invalid_input", False)) or bool(extra.get("invalid_input", False)):
        out["invalid_input"] = True
    for key in (
        "minimum_raw_mode_probability",
        "probability_floor",
        "required_root_mass",
        "minimum_root_count",
    ):
        if key not in out and key in extra:
            out[key] = extra[key]
    return out


def _interaction_aware_recovery_certificate_np(
    agent_state: np.ndarray,
    successor_state: np.ndarray,
    sdc_index: int,
    current_ego_trajectory: np.ndarray,
    current_ego_kinematic_ok: np.ndarray,
    shifted_ego_trajectory: np.ndarray,
    shifted_ego_kinematic_ok: np.ndarray,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
    current_collision_context: dict[str, Any],
    shifted_collision_context: dict[str, Any],
    response_support: dict[int, dict[str, Any]],
    object_types: np.ndarray,
    compatibility_cache: dict[str, dict[Any, bool]] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Certify one ego tube under universal low-burden same-root response support."""
    current_blockers, current_blocker_detail = _collision_blocking_agent_indices_against_context(
        current_ego_trajectory, current_collision_context,
    )
    shifted_blockers, shifted_blocker_detail = _collision_blocking_agent_indices_against_context(
        shifted_ego_trajectory, shifted_collision_context,
    )
    blockers = sorted(set(current_blockers) | set(shifted_blockers))
    detail: dict[str, Any] = {
        "failure_reason": "none",
        "current_blockers": current_blockers,
        "shifted_blockers": shifted_blockers,
        "blocker_count": int(len(blockers)),
        "current_blocker_detail": current_blocker_detail,
        "shifted_blocker_detail": shifted_blocker_detail,
        "response_root_checks": 0,
        "response_profile_evaluations": 0,
        "interaction_environment_compatibility_checks": 0,
        "interaction_environment_compatibility_rejects": 0,
        "interaction_joint_compatibility_checks": 0,
        "interaction_joint_compatibility_rejects": 0,
        "interaction_joint_assignment_backtracks": 0,
        "interaction_environment_compatibility_cache_hits": 0,
        "interaction_joint_compatibility_cache_hits": 0,
        "interaction_successor_context_cache_hits": 0,
        "supported_root_count": 0,
        "minimum_retained_root_mass": 0.0,
        "maximum_selected_response_burden": 0.0,
        "current_residual_certificate": {},
        "shifted_residual_certificate": {},
    }
    if compatibility_cache is None:
        compatibility_cache = {}
    environment_cache = compatibility_cache.setdefault("environment", {})
    joint_cache = compatibility_cache.setdefault("joint", {})
    if not blockers:
        detail["failure_reason"] = "no_collision_blocker"
        return False, detail
    unsupported = [j for j in blockers if not bool(response_support.get(int(j), {}).get("ready", False))]
    if unsupported:
        detail["failure_reason"] = "unsupported_collision_blocker"
        detail["unsupported_blockers"] = unsupported
        return False, detail

    excluded = set(blockers)
    current_residual_context = _collision_context_without_agent_indices(current_collision_context, excluded)
    shifted_residual_context = _collision_context_without_agent_indices(shifted_collision_context, excluded)
    current_ok, current_physical = _physical_recovery_tube_certificate_np(
        np.asarray(agent_state, dtype=np.float32), int(sdc_index),
        current_ego_trajectory, current_ego_kinematic_ok,
        roadgraph, cfg, collision_context=current_residual_context,
    )
    shifted_ok, shifted_physical = _physical_recovery_tube_certificate_np(
        np.asarray(successor_state, dtype=np.float32), int(sdc_index),
        shifted_ego_trajectory, shifted_ego_kinematic_ok,
        roadgraph, cfg, collision_context=shifted_residual_context,
    )
    detail["current_residual_certificate"] = current_physical
    detail["shifted_residual_certificate"] = shifted_physical
    if not (current_ok and shifted_ok):
        detail["failure_reason"] = "residual_physical_certificate_failed"
        return False, detail

    obj = np.asarray(object_types, dtype=np.int32).reshape(-1)
    environment_indices = sorted({
        int(agent.get("index", -1))
        for context in (current_collision_context, shifted_collision_context)
        for agent in context.get("agents", [])
        if int(agent.get("index", -1)) >= 0
        and int(agent.get("index", -1)) != int(sdc_index)
        and int(agent.get("index", -1)) not in excluded
    })
    environment: list[dict[str, Any]] = []
    current_horizon = int(np.asarray(current_ego_trajectory).shape[0])
    shifted_horizon = int(np.asarray(shifted_ego_trajectory).shape[0])
    for environment_index in environment_indices:
        current_cv = _constant_velocity_trajectory_from_state_np(
            agent_state, environment_index, current_horizon, cfg,
        )
        shifted_cv = _constant_velocity_trajectory_from_state_np(
            successor_state, environment_index, shifted_horizon, cfg,
        )
        if current_cv is None or shifted_cv is None:
            detail["failure_reason"] = "invalid_environment_actor_prediction"
            detail["failed_environment_agent_index"] = int(environment_index)
            return False, detail
        environment.append({
            "agent_index": int(environment_index),
            "object_type": int(obj[environment_index]) if environment_index < obj.size else int(ObjectType.UNKNOWN),
            "trajectory": current_cv,
            "shifted_trajectory": shifted_cv,
        })
    detail["environment_agent_count"] = int(len(environment))

    # Build a separable robust response envelope.  One profile is selected for
    # every retained root.  Profiles belonging to different blockers must be
    # mutually safe in both the current and shifted tubes.  Because roots of the
    # same agent are alternatives rather than simultaneous states, they need no
    # pairwise check.  Cross-agent pairwise compatibility makes every Cartesian
    # combination of retained roots jointly realizable without using a learned
    # correlation model or future simulator state.
    root_nodes: list[dict[str, Any]] = []
    minimum_mass = 1.0
    root_count = 0
    for agent_index in blockers:
        agent_support = response_support[int(agent_index)]
        minimum_mass = min(minimum_mass, float(agent_support.get("retained_mass", 0.0)))
        object_type = int(agent_support.get("object_type", int(ObjectType.UNKNOWN)))
        for root_ordinal, root in enumerate(agent_support.get("roots", [])):
            root_count += 1
            detail["response_root_checks"] = int(detail["response_root_checks"]) + 1
            ego_safe_count = 0
            environment_safe_profiles: list[dict[str, Any]] = []
            for profile in root.get("profiles", []):
                detail["response_profile_evaluations"] = int(detail["response_profile_evaluations"]) + 1
                profile_current = np.asarray(profile["trajectory"], dtype=np.float32)
                profile_shifted = np.asarray(profile["shifted_trajectory"], dtype=np.float32)
                try:
                    current_unsafe = unsafe_between_bool(
                        np.asarray(current_ego_trajectory, dtype=np.float32),
                        profile_current, cfg, agent_type=object_type,
                    )
                    shifted_unsafe = unsafe_between_bool(
                        np.asarray(shifted_ego_trajectory, dtype=np.float32),
                        profile_shifted, cfg, agent_type=object_type,
                    )
                except Exception:
                    current_unsafe = shifted_unsafe = True
                if current_unsafe or shifted_unsafe:
                    continue
                ego_safe_count += 1
                environment_safe = True
                for actor in environment:
                    detail["interaction_environment_compatibility_checks"] = int(
                        detail["interaction_environment_compatibility_checks"]
                    ) + 1
                    cache_key = (
                        int(agent_index), int(root_ordinal), int(profile.get("profile_index", -1)),
                        int(actor["agent_index"]),
                    )
                    cached = environment_cache.get(cache_key)
                    if cached is None:
                        try:
                            current_bad = (
                                unsafe_between_bool(
                                    profile_current, np.asarray(actor["trajectory"], dtype=np.float32),
                                    cfg, agent_type=int(actor["object_type"]),
                                )
                                or unsafe_between_bool(
                                    np.asarray(actor["trajectory"], dtype=np.float32), profile_current,
                                    cfg, agent_type=object_type,
                                )
                            )
                            shifted_bad = (
                                unsafe_between_bool(
                                    profile_shifted, np.asarray(actor["shifted_trajectory"], dtype=np.float32),
                                    cfg, agent_type=int(actor["object_type"]),
                                )
                                or unsafe_between_bool(
                                    np.asarray(actor["shifted_trajectory"], dtype=np.float32), profile_shifted,
                                    cfg, agent_type=object_type,
                                )
                            )
                            cached = bool(not (current_bad or shifted_bad))
                        except Exception:
                            cached = False
                        environment_cache[cache_key] = bool(cached)
                    else:
                        detail["interaction_environment_compatibility_cache_hits"] = int(
                            detail["interaction_environment_compatibility_cache_hits"]
                        ) + 1
                    if not bool(cached):
                        detail["interaction_environment_compatibility_rejects"] = int(
                            detail["interaction_environment_compatibility_rejects"]
                        ) + 1
                        environment_safe = False
                        break
                if environment_safe:
                    environment_safe_profiles.append(profile)
            environment_safe_profiles.sort(key=lambda p: (float(p["burden"]), int(p["profile_index"])))
            if not environment_safe_profiles:
                detail["failure_reason"] = (
                    "retained_root_has_no_ego_safe_response"
                    if ego_safe_count <= 0 else "retained_root_has_no_environment_safe_response"
                )
                detail["failed_agent_index"] = int(agent_index)
                detail["failed_root_ordinal"] = int(root_ordinal)
                detail["failed_root_stage"] = "ego" if ego_safe_count <= 0 else "environment"
                return False, detail
            root_nodes.append({
                "agent_index": int(agent_index),
                "object_type": object_type,
                "root_ordinal": int(root_ordinal),
                "root": root,
                "profiles": environment_safe_profiles,
            })

    # Minimum-domain-first deterministic backtracking.  This is an exact CSP over
    # the bounded frozen root/profile bank, not a top-k heuristic.  A solution
    # selects one response per retained root whose cross-agent pairs are all safe;
    # therefore every possible multi-agent retained-root realization has a valid
    # simultaneous response tuple.
    ordered_nodes = sorted(
        root_nodes,
        key=lambda n: (
            len(n["profiles"]), int(n["agent_index"]), int(n["root_ordinal"]),
        ),
    )
    assignments: list[tuple[dict[str, Any], dict[str, Any]]] = []
    compatibility_checks = 0
    compatibility_rejects = 0
    backtracks = 0

    def pair_compatible(
        node: dict[str, Any], profile: dict[str, Any],
        other_node: dict[str, Any], other_profile: dict[str, Any],
    ) -> bool:
        nonlocal compatibility_checks, compatibility_rejects
        if int(node["agent_index"]) == int(other_node["agent_index"]):
            return True
        compatibility_checks += 1
        left = (
            int(node["agent_index"]), int(node["root_ordinal"]), int(profile.get("profile_index", -1)),
        )
        right = (
            int(other_node["agent_index"]), int(other_node["root_ordinal"]), int(other_profile.get("profile_index", -1)),
        )
        cache_key = tuple(sorted((left, right)))
        cached = joint_cache.get(cache_key)
        if cached is None:
            try:
                current_bad = (
                    unsafe_between_bool(
                        np.asarray(profile["trajectory"], dtype=np.float32),
                        np.asarray(other_profile["trajectory"], dtype=np.float32),
                        cfg, agent_type=int(other_node["object_type"]),
                    )
                    or unsafe_between_bool(
                        np.asarray(other_profile["trajectory"], dtype=np.float32),
                        np.asarray(profile["trajectory"], dtype=np.float32),
                        cfg, agent_type=int(node["object_type"]),
                    )
                )
                shifted_bad = (
                    unsafe_between_bool(
                        np.asarray(profile["shifted_trajectory"], dtype=np.float32),
                        np.asarray(other_profile["shifted_trajectory"], dtype=np.float32),
                        cfg, agent_type=int(other_node["object_type"]),
                    )
                    or unsafe_between_bool(
                        np.asarray(other_profile["shifted_trajectory"], dtype=np.float32),
                        np.asarray(profile["shifted_trajectory"], dtype=np.float32),
                        cfg, agent_type=int(node["object_type"]),
                    )
                )
                cached = bool(not (current_bad or shifted_bad))
            except Exception:
                cached = False
            joint_cache[cache_key] = bool(cached)
        else:
            detail["interaction_joint_compatibility_cache_hits"] = int(
                detail["interaction_joint_compatibility_cache_hits"]
            ) + 1
        if not bool(cached):
            compatibility_rejects += 1
            return False
        return True

    def assign_node(pos: int) -> bool:
        nonlocal backtracks
        if pos >= len(ordered_nodes):
            return True
        node = ordered_nodes[pos]
        for profile in node["profiles"]:
            if all(pair_compatible(node, profile, old_node, old_profile) for old_node, old_profile in assignments):
                assignments.append((node, profile))
                if assign_node(pos + 1):
                    return True
                assignments.pop()
                backtracks += 1
        return False

    if not assign_node(0):
        detail.update({
            "failure_reason": "no_jointly_compatible_response_envelope",
            "interaction_joint_compatibility_checks": int(compatibility_checks),
            "interaction_joint_compatibility_rejects": int(compatibility_rejects),
            "interaction_joint_assignment_backtracks": int(backtracks),
        })
        return False, detail

    selected_responses: list[dict[str, Any]] = []
    maximum_burden = 0.0
    for node, accepted_profile in sorted(
        assignments,
        key=lambda x: (int(x[0]["agent_index"]), int(x[0]["root_ordinal"])),
    ):
        root = node["root"]
        burden = float(accepted_profile["burden"])
        maximum_burden = max(maximum_burden, burden)
        selected_responses.append({
            "agent_index": int(node["agent_index"]),
            "root_ordinal": int(node["root_ordinal"]),
            "mode_indices": tuple(int(x) for x in root.get("mode_indices", ())),
            "root_weight": float(root.get("weight", 0.0)),
            "profile_index": int(accepted_profile["profile_index"]),
            "burden": burden,
        })
    detail.update({
        "interaction_joint_compatibility_checks": int(compatibility_checks),
        "interaction_joint_compatibility_rejects": int(compatibility_rejects),
        "interaction_joint_assignment_backtracks": int(backtracks),
    })
    detail.update({
        "supported_root_count": int(root_count),
        "minimum_retained_root_mass": float(minimum_mass if blockers else 0.0),
        "maximum_selected_response_burden": float(maximum_burden),
        "selected_responses": selected_responses,
    })
    return True, detail


def _construct_interaction_aware_reachable_response_envelope_np(
    agent_state: np.ndarray,
    sdc_index: int,
    nominal_trajectories: np.ndarray,
    cand_valid: np.ndarray,
    nominal_roadgraph_safe: np.ndarray,
    macro_types: np.ndarray,
    fallback_scores: np.ndarray,
    collision_prefix_steps: np.ndarray,
    action_targets: np.ndarray,
    action_accels: np.ndarray,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
    previous_longitudinal_accel: float,
    *,
    base_candidate_index: int,
    critical_track_index: np.ndarray,
    critical_valid: np.ndarray,
    natural_trajectories: np.ndarray,
    natural_logits: np.ndarray,
    object_types: np.ndarray,
    shared_compatibility_cache: dict[str, dict[Any, bool]] | None = None,
    shared_successor_context_cache: dict[bytes, tuple[np.ndarray, dict[str, Any]]] | None = None,
    prepared_response_support: dict[int, dict[str, Any]] | None = None,
    prepared_response_support_detail: dict[str, Any] | None = None,
    prepared_hypothesis_workspace: dict[str, Any] | None = None,
    prepared_unsupported_replay_cache: dict[int, dict[str, Any]] | None = None,
    known_nested_v39_empty: bool = False,
    hypothesis_indices: np.ndarray | None = None,
    internal_trace: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """V16.8.42 root-conditioned interaction-aware reachable-response envelope.

    The exact V39 conflict-window tube remains the nested first branch.  Only when
    it has no physical certificate does this extension revisit the same V39
    controller-reachable hypotheses.  A static CV blocker may be replaced by an
    interaction contingency only when every retained high-mass natural root of
    that exact blocker owns the same low-burden, drivable, Waymax-kinematic
    response profile for both the current and one-step-shifted ego tubes.  All
    non-blocking agents, ego roadgraph checks and ego inverse dynamics remain hard.
    """
    if bool(known_nested_v39_empty):
        # Engineering-only fast path used by V43 after the exact V42 stage has
        # already proved the nested V39 hard set empty for the same policy step.
        # Re-running V39 cannot change the expanded-support result.
        nested_selected, nested_detail = None, {}
    else:
        nested_selected, nested_detail = _construct_conflict_window_control_reachable_tube_np(
            agent_state, sdc_index, nominal_trajectories, cand_valid,
            nominal_roadgraph_safe, macro_types, fallback_scores,
            collision_prefix_steps, action_targets, action_accels,
            roadgraph, cfg, previous_longitudinal_accel,
        )
    detail = dict(nested_detail)
    detail.update({
        "nested_v39_selected": bool(nested_selected is not None),
        "interaction_response_attempted": False,
        "interaction_response_selected": False,
        "interaction_support_agents_total": 0,
        "interaction_support_agents_ready": 0,
        "interaction_support_retained_roots": 0,
        "interaction_support_eligible_profiles": 0,
        "interaction_hypotheses_evaluated": 0,
        "interaction_noop_hypotheses_skipped": 0,
        "interaction_no_blocker_rejects": 0,
        "interaction_unsupported_blocker_rejects": 0,
        "interaction_residual_physical_rejects": 0,
        "interaction_root_unrecoverable_rejects": 0,
        "interaction_joint_incompatibility_rejects": 0,
        "interaction_environment_compatibility_checks": 0,
        "interaction_environment_compatibility_rejects": 0,
        "interaction_joint_compatibility_checks": 0,
        "interaction_joint_compatibility_rejects": 0,
        "interaction_joint_assignment_backtracks": 0,
        # V16.8.43 performance diagnostics must exist on the outer constructor
        # detail before the first interaction hypothesis is aggregated.  These
        # counters do not participate in certificate admission or selection.
        "interaction_environment_compatibility_cache_hits": 0,
        "interaction_joint_compatibility_cache_hits": 0,
        "interaction_successor_context_cache_hits": 0,
        "interaction_selected_blocker_count": 0,
        "interaction_selected_root_count": 0,
        "interaction_selected_minimum_root_mass": 0.0,
        "interaction_selected_maximum_response_burden": 0.0,
        "interaction_selected_profile_evaluations": 0,
        "selected_is_interaction_response": False,
        "selected_certificate_kind": "nested_v39" if nested_selected is not None else "none",
    })
    if nested_selected is not None:
        return nested_selected, detail

    detail["interaction_response_attempted"] = True
    state = np.asarray(agent_state, dtype=np.float32)
    traj = np.asarray(nominal_trajectories, dtype=np.float32)
    valid = np.asarray(cand_valid, dtype=bool).reshape(-1)
    nominal_road = np.asarray(nominal_roadgraph_safe, dtype=bool).reshape(-1)
    macro = np.asarray(macro_types, dtype=np.int64).reshape(-1)
    scores = np.asarray(fallback_scores, dtype=np.float64).reshape(-1)
    prefix = np.asarray(collision_prefix_steps, dtype=np.float64).reshape(-1)
    targets = np.asarray(action_targets, dtype=np.float32)
    nominal_accels = np.asarray(action_accels, dtype=np.float64).reshape(-1)
    if (
        state.ndim != 2 or not (0 <= int(sdc_index) < state.shape[0])
        or traj.ndim != 3 or traj.shape[0] <= 0 or traj.shape[1] <= 0 or traj.shape[2] < 5
    ):
        detail["interaction_invalid_input"] = True
        return None, detail
    n = min(
        traj.shape[0], valid.size, nominal_road.size, macro.size, scores.size,
        prefix.size, targets.shape[0] if targets.ndim == 2 else 0, nominal_accels.size,
    )
    base_idx = int(base_candidate_index)
    if n <= 0 or not (0 <= base_idx < n):
        detail["interaction_invalid_input"] = True
        return None, detail
    traj, valid, nominal_road, macro = traj[:n], valid[:n], nominal_road[:n], macro[:n]
    scores, prefix, targets, nominal_accels = scores[:n], prefix[:n], targets[:n], nominal_accels[:n]

    if prepared_response_support is None:
        response_support, support_detail = _prepare_interaction_response_support_np(
            state, int(sdc_index), critical_track_index, critical_valid,
            natural_trajectories, natural_logits, object_types, roadgraph, cfg,
        )
    else:
        # V16.8.43R3 engineering-only fast path: response support is a
        # per-agent function of the current state, frozen natural roots and
        # frozen response bank; it is independent of the ego hypothesis being
        # replayed.  Reusing the exact first-pass support therefore removes
        # duplicate work without changing any certificate predicate.
        response_support = prepared_response_support
        support_detail = dict(prepared_response_support_detail or {})
    detail["interaction_support_detail"] = support_detail
    if internal_trace is not None:
        # Internal-only objects; never serialized into rollout diagnostics.
        # BC-IARE R3 uses them only when the late-bound query set is disjoint
        # from the original critical set, which is the online query contract.
        internal_trace["prepared_response_support"] = response_support
        internal_trace["prepared_response_support_detail"] = dict(support_detail)
    detail["interaction_support_agents_total"] = int(support_detail.get("critical_slots", 0))
    detail["interaction_support_agents_ready"] = int(support_detail.get("agents_ready", 0))
    detail["interaction_support_retained_roots"] = int(support_detail.get("retained_roots", 0))
    detail["interaction_support_eligible_profiles"] = int(support_detail.get("eligible_profiles", 0))
    if int(support_detail.get("agents_ready", 0)) <= 0:
        detail["interaction_failure_reason"] = "no_response_ready_critical_agent"
        return None, detail

    if prepared_hypothesis_workspace is None:
        pool = valid & nominal_road
        if not bool(pool.any()):
            pool = valid.copy()
        pool &= macro != int(MacroType.PAD)
        reps = _semantic_action_class_representatives_np(
            pool, macro, targets, prefix, scores, prefer_fallback=True,
        )
        if not reps:
            detail["interaction_failure_reason"] = "no_parent_action_class"
            return None, detail

        H = int(traj.shape[1])
        current_collision_context = _prepare_collision_check_context(
            state, int(sdc_index), cfg, horizon_steps=H, other_future_trajs=None,
        )
        rep_arr = np.asarray(reps, dtype=np.int64)
        nominal_projected, _nominal_kin, _nominal_accel_hist = _project_candidate_bank_through_controller_np(
            state[int(sdc_index)], traj[rep_arr], cfg, float(previous_longitudinal_accel),
        )
        parent_indices: list[int] = []
        policy_ids: list[int] = []
        policy_names: list[str] = []
        release_edges: list[int] = []
        nonnominal_edges: list[int] = []
        schedules: list[np.ndarray] = []
        for local_i, parent in enumerate(reps):
            window = _collision_violation_window_against_context(
                nominal_projected[local_i], current_collision_context,
            )
            family = _conflict_window_envelope_schedule_family(
                H,
                int(window.get("first_violation_step", -1)),
                int(window.get("last_violation_step", -1)),
            )
            for rec in family:
                parent_indices.append(int(parent))
                policy_ids.append(int(rec["policy_id"]))
                policy_names.append(str(rec["policy_name"]))
                release_edges.append(int(rec["release_edge"]))
                nonnominal_edges.append(int(rec["nonnominal_edges"]))
                schedules.append(np.asarray(rec["schedule"], dtype=np.int8))
        if not schedules:
            detail["interaction_failure_reason"] = "no_conflict_window_hypothesis"
            return None, detail
        parent_arr = np.asarray(parent_indices, dtype=np.int64)
        policy_arr = np.asarray(policy_ids, dtype=np.int8)
        schedule_arr = np.stack(schedules, axis=0).astype(np.int8)
        release_arr = np.asarray(release_edges, dtype=np.int32)
        nonnominal_arr = np.asarray(nonnominal_edges, dtype=np.int32)
        projected, kin_ok, accel_hist = _project_candidate_bank_through_controller_np(
            state[int(sdc_index)], np.asarray(traj[parent_arr], dtype=np.float32), cfg,
            float(previous_longitudinal_accel),
            longitudinal_envelope_schedule=schedule_arr,
        )
        base_target = np.asarray(targets[base_idx, :5], dtype=np.float32)
        hypothesis_workspace = {
            "H": int(H),
            "current_collision_context": current_collision_context,
            "parent_arr": parent_arr,
            "policy_arr": policy_arr,
            "policy_names": tuple(policy_names),
            "schedule_arr": schedule_arr,
            "release_arr": release_arr,
            "nonnominal_arr": nonnominal_arr,
            "projected": projected,
            "kin_ok": kin_ok,
            "accel_hist": accel_hist,
            "base_target": base_target,
        }
        if internal_trace is not None:
            internal_trace["prepared_hypothesis_workspace"] = hypothesis_workspace
    else:
        # V16.8.43R3 engineering-only fast path.  The V43 second pass uses the
        # exact same state, candidate bank, controller limits and V39 schedule
        # family as the V42 first pass; only response support changes.  Reusing
        # this immutable workspace therefore preserves hypothesis identity and
        # hard physical semantics exactly.
        ws = prepared_hypothesis_workspace
        try:
            H = int(ws["H"])
            current_collision_context = ws["current_collision_context"]
            parent_arr = np.asarray(ws["parent_arr"], dtype=np.int64)
            policy_arr = np.asarray(ws["policy_arr"], dtype=np.int8)
            policy_names = list(ws["policy_names"])
            schedule_arr = np.asarray(ws["schedule_arr"], dtype=np.int8)
            release_arr = np.asarray(ws["release_arr"], dtype=np.int32)
            nonnominal_arr = np.asarray(ws["nonnominal_arr"], dtype=np.int32)
            projected = np.asarray(ws["projected"], dtype=np.float32)
            kin_ok = np.asarray(ws["kin_ok"], dtype=bool)
            accel_hist = np.asarray(ws["accel_hist"], dtype=np.float32)
            base_target = np.asarray(ws["base_target"], dtype=np.float32)
        except Exception:
            detail["interaction_invalid_prepared_workspace"] = True
            return None, detail

    def hypothesis_key(j: int) -> tuple[Any, ...]:
        parent = int(parent_arr[j])
        pid = int(policy_arr[j])
        return (
            float(scores[parent]) if np.isfinite(scores[parent]) else float("inf"),
            abs(float(accel_hist[j, 0]) - float(nominal_accels[parent])),
            int(nonnominal_arr[j]),
            0 if pid == 0 else (1 if abs(pid) >= 2 else 2),
            parent,
            pid,
        )

    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1.0e-6)
    # V16.8.43 performance-only caches.  They memoize predicates whose inputs are
    # invariant across hypotheses in one policy step.  Logical compatibility
    # checks/rejects remain counted exactly as before so mechanism statistics are
    # comparable to V42; cache-hit counters are additional diagnostics only.
    compatibility_cache = shared_compatibility_cache
    if compatibility_cache is None:
        compatibility_cache = {"environment": {}, "joint": {}}
    else:
        compatibility_cache.setdefault("environment", {})
        compatibility_cache.setdefault("joint", {})
    successor_context_cache = shared_successor_context_cache
    if successor_context_cache is None:
        successor_context_cache = {}
    if hypothesis_indices is None:
        candidate_hypotheses = list(range(int(projected.shape[0])))
    else:
        raw_hypotheses = np.asarray(hypothesis_indices, dtype=np.int64).reshape(-1)
        candidate_hypotheses = sorted({
            int(j) for j in raw_hypotheses.tolist()
            if 0 <= int(j) < int(projected.shape[0])
        })
    for j in sorted(candidate_hypotheses, key=hypothesis_key):
        first_target = np.asarray(projected[j, 0, :5], dtype=np.float32)
        if np.allclose(first_target, base_target, rtol=0.0, atol=1.0e-6):
            detail["interaction_noop_hypotheses_skipped"] = int(detail["interaction_noop_hypotheses_skipped"]) + 1
            continue
        detail["interaction_hypotheses_evaluated"] = int(detail["interaction_hypotheses_evaluated"]) + 1
        first_accel = float(accel_hist[j, 0])
        prepared_replay = (
            prepared_unsupported_replay_cache.get(int(j))
            if isinstance(prepared_unsupported_replay_cache, dict) else None
        )
        if prepared_replay is not None:
            # This record was computed by the exact V42 first pass for the same
            # hypothesis before it failed solely on unsupported blocker support.
            # Reusing it removes duplicate successor/shift projection work.
            successor = prepared_replay["successor"]
            shifted_collision_context = prepared_replay["shifted_collision_context"]
            shifted_projected = prepared_replay["shifted_projected"]
            shifted_kin_ok = prepared_replay["shifted_kin_ok"]
            shifted_accel_hist = prepared_replay["shifted_accel_hist"]
            # R2 would have hit the shared successor-context cache here; keep the
            # logical diagnostic identical even though R3 reuses the full record.
            detail["interaction_successor_context_cache_hits"] = int(
                detail["interaction_successor_context_cache_hits"]
            ) + 1
        else:
            successor_key = np.asarray(first_target, dtype=np.float32).tobytes() + np.float32(first_accel).tobytes()
            cached_successor = successor_context_cache.get(successor_key)
            if cached_successor is None:
                successor = _counterfactual_successor_agent_state(
                    state, int(sdc_index), first_target, cfg,
                )
                shifted_collision_context = _prepare_collision_check_context(
                    successor, int(sdc_index), cfg, horizon_steps=H, other_future_trajs=None,
                )
                successor_context_cache[successor_key] = (successor, shifted_collision_context)
            else:
                successor, shifted_collision_context = cached_successor
                detail["interaction_successor_context_cache_hits"] = int(
                    detail["interaction_successor_context_cache_hits"]
                ) + 1
            shifted_reference = _shift_append_terminal_reference_np(projected[j], dt)
            shifted_schedule = _shift_longitudinal_envelope_schedule_np(
                schedule_arr[j], int(policy_arr[j]),
            )
            shifted_projected, shifted_kin_ok, shifted_accel_hist = _project_candidate_bank_through_controller_np(
                successor[int(sdc_index)], shifted_reference[None, ...], cfg,
                first_accel, longitudinal_envelope_schedule=shifted_schedule[None, :],
            )
        interaction_ok, interaction_detail = _interaction_aware_recovery_certificate_np(
            state, successor, int(sdc_index),
            projected[j], kin_ok[j], shifted_projected[0], shifted_kin_ok[0],
            roadgraph, cfg, current_collision_context, shifted_collision_context,
            response_support, object_types, compatibility_cache=compatibility_cache,
        )
        reason = str(interaction_detail.get("failure_reason", "unknown"))
        detail["interaction_environment_compatibility_checks"] = int(
            detail["interaction_environment_compatibility_checks"]
        ) + int(interaction_detail.get("interaction_environment_compatibility_checks", 0))
        detail["interaction_environment_compatibility_rejects"] = int(
            detail["interaction_environment_compatibility_rejects"]
        ) + int(interaction_detail.get("interaction_environment_compatibility_rejects", 0))
        detail["interaction_joint_compatibility_checks"] = int(detail["interaction_joint_compatibility_checks"]) + int(
            interaction_detail.get("interaction_joint_compatibility_checks", 0)
        )
        detail["interaction_joint_compatibility_rejects"] = int(detail["interaction_joint_compatibility_rejects"]) + int(
            interaction_detail.get("interaction_joint_compatibility_rejects", 0)
        )
        detail["interaction_joint_assignment_backtracks"] = int(detail["interaction_joint_assignment_backtracks"]) + int(
            interaction_detail.get("interaction_joint_assignment_backtracks", 0)
        )
        detail["interaction_environment_compatibility_cache_hits"] = int(
            detail["interaction_environment_compatibility_cache_hits"]
        ) + int(interaction_detail.get("interaction_environment_compatibility_cache_hits", 0))
        detail["interaction_joint_compatibility_cache_hits"] = int(
            detail["interaction_joint_compatibility_cache_hits"]
        ) + int(interaction_detail.get("interaction_joint_compatibility_cache_hits", 0))
        if not interaction_ok:
            if reason == "no_collision_blocker":
                detail["interaction_no_blocker_rejects"] = int(detail["interaction_no_blocker_rejects"]) + 1
            elif reason == "unsupported_collision_blocker":
                detail["interaction_unsupported_blocker_rejects"] = int(detail["interaction_unsupported_blocker_rejects"]) + 1
                if internal_trace is not None:
                    unsupported_h = internal_trace.setdefault("unsupported_hypothesis_indices", [])
                    unsupported_h.append(int(j))
                    blocker_union = internal_trace.setdefault("unsupported_blocker_union", set())
                    blocker_union.update(int(x) for x in interaction_detail.get("unsupported_blockers", []))
                    replay_cache = internal_trace.setdefault("unsupported_replay_cache", {})
                    replay_cache[int(j)] = {
                        "successor": successor,
                        "shifted_collision_context": shifted_collision_context,
                        "shifted_projected": shifted_projected,
                        "shifted_kin_ok": shifted_kin_ok,
                        "shifted_accel_hist": shifted_accel_hist,
                    }
            elif reason == "residual_physical_certificate_failed":
                detail["interaction_residual_physical_rejects"] = int(detail["interaction_residual_physical_rejects"]) + 1
            elif reason in {
                "retained_root_has_no_ego_safe_response",
                "retained_root_has_no_environment_safe_response",
                "invalid_environment_actor_prediction",
            }:
                detail["interaction_root_unrecoverable_rejects"] = int(detail["interaction_root_unrecoverable_rejects"]) + 1
            elif reason == "no_jointly_compatible_response_envelope":
                detail["interaction_joint_incompatibility_rejects"] = int(detail["interaction_joint_incompatibility_rejects"]) + 1
            continue

        parent = int(parent_arr[j])
        pid = int(policy_arr[j])
        current_certificate = dict(interaction_detail.get("current_residual_certificate", {}))
        shifted_certificate = dict(interaction_detail.get("shifted_residual_certificate", {}))
        selected = {
            "expanded_index": int(j),
            "parent_index": parent,
            "macro": int(macro[parent]),
            "policy_id": pid,
            "policy_name": str(policy_names[j]),
            "mode": int(schedule_arr[j, 0]),
            "schedule": np.asarray(schedule_arr[j], dtype=np.int8),
            "release_edge": int(release_arr[j]),
            "nonnominal_edges": int(nonnominal_arr[j]),
            "event_release": bool(abs(pid) >= 2),
            "trajectory": np.asarray(projected[j], dtype=np.float32),
            "target": first_target,
            "accel": first_accel,
            "accel_history": np.asarray(accel_hist[j], dtype=np.float32),
            "shifted_trajectory": np.asarray(shifted_projected[0], dtype=np.float32),
            "shifted_accel_history": np.asarray(shifted_accel_hist[0], dtype=np.float32),
            "fallback_score": float(scores[parent]) if np.isfinite(scores[parent]) else float("inf"),
            "first_accel_delta": float(first_accel - float(nominal_accels[parent])),
            "current_certificate": current_certificate,
            "shifted_certificate": shifted_certificate,
            "interaction_certificate": interaction_detail,
        }
        detail.update({
            "selected": True,
            "interaction_response_selected": True,
            "selected_is_interaction_response": True,
            "selected_certificate_kind": "interaction_aware_reachable_response_envelope",
            "selected_is_lifted": bool(pid != 0),
            "selected_is_event_release": bool(abs(pid) >= 2),
            "selected_policy_id": pid,
            "selected_policy_name": str(policy_names[j]),
            "selected_envelope_mode": int(schedule_arr[j, 0]),
            "selected_release_edge": int(release_arr[j]),
            "selected_nonnominal_edges": int(nonnominal_arr[j]),
            "selected_parent_candidate": parent,
            "selected_parent_macro": int(macro[parent]),
            "selected_parent_macro_name": _macro_name(int(macro[parent])),
            "selected_first_accel_delta": float(first_accel - float(nominal_accels[parent])),
            "selected_collision_min_margin_m": float(current_certificate.get("collision_min_margin_m", -999.0)),
            "selected_shift_collision_min_margin_m": float(shifted_certificate.get("collision_min_margin_m", -999.0)),
            "selected_fallback_score": float(scores[parent]) if np.isfinite(scores[parent]) else float("inf"),
            "selected_is_new_first_action": True,
            "interaction_selected_blocker_count": int(interaction_detail.get("blocker_count", 0)),
            "interaction_selected_root_count": int(interaction_detail.get("supported_root_count", 0)),
            "interaction_selected_minimum_root_mass": float(interaction_detail.get("minimum_retained_root_mass", 0.0)),
            "interaction_selected_maximum_response_burden": float(interaction_detail.get("maximum_selected_response_burden", 0.0)),
            "interaction_selected_profile_evaluations": int(interaction_detail.get("response_profile_evaluations", 0)),
            "interaction_selected_environment_agent_count": int(interaction_detail.get("environment_agent_count", 0)),
            "interaction_selected_environment_compatibility_checks": int(interaction_detail.get("interaction_environment_compatibility_checks", 0)),
            "interaction_selected_responses": interaction_detail.get("selected_responses", []),
        })
        return selected, detail

    detail["interaction_failure_reason"] = "no_interaction_certified_action"
    return None, detail


def _construct_blocker_conditioned_interaction_aware_reachable_response_envelope_np(
    agent_state: np.ndarray,
    sdc_index: int,
    nominal_trajectories: np.ndarray,
    cand_valid: np.ndarray,
    nominal_roadgraph_safe: np.ndarray,
    macro_types: np.ndarray,
    fallback_scores: np.ndarray,
    collision_prefix_steps: np.ndarray,
    action_targets: np.ndarray,
    action_accels: np.ndarray,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
    previous_longitudinal_accel: float,
    *,
    base_candidate_index: int,
    critical_track_index: np.ndarray,
    critical_valid: np.ndarray,
    natural_trajectories: np.ndarray,
    natural_logits: np.ndarray,
    blocker_query_track_index: np.ndarray,
    blocker_query_trajectories: np.ndarray | None,
    blocker_query_logits: np.ndarray | None,
    object_types: np.ndarray,
    blocker_query_decoder: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """V16.8.43 late-bound blocker-conditioned support completion.

    Stage A runs the *exact* V42 RC-IARE constructor first.  If it returns a
    certificate, that action is returned unchanged.  Only if the full V42 hard
    set is empty do we extend the natural-option query domain with agents drawn
    from the frozen collision context.  The decoder, canonical root measure,
    burden budget, response bank, environment checks, joint CSP, physical tube
    certificate, shift closure, and deterministic action ordering are unchanged.

    This isolates a single hypothesis exposed by the V42 attribution: the online
    scene-level critical set (default active cap 4) can omit exact blockers seen
    by the wider collision audit (up to 24), creating a support-indexing false
    negative.  The extension does *not* enlarge the social NCF critical set.
    """
    shared_compatibility_cache: dict[str, dict[Any, bool]] = {
        "environment": {}, "joint": {},
    }
    shared_successor_context_cache: dict[bytes, tuple[np.ndarray, dict[str, Any]]] = {}
    base_trace: dict[str, Any] = {
        "unsupported_hypothesis_indices": [],
        "unsupported_blocker_union": set(),
    }
    base_selected, base_detail = _construct_interaction_aware_reachable_response_envelope_np(
        agent_state, sdc_index, nominal_trajectories, cand_valid,
        nominal_roadgraph_safe, macro_types, fallback_scores,
        collision_prefix_steps, action_targets, action_accels,
        roadgraph, cfg, previous_longitudinal_accel,
        base_candidate_index=int(base_candidate_index),
        critical_track_index=critical_track_index,
        critical_valid=critical_valid,
        natural_trajectories=natural_trajectories,
        natural_logits=natural_logits,
        object_types=object_types,
        shared_compatibility_cache=shared_compatibility_cache,
        shared_successor_context_cache=shared_successor_context_cache,
        internal_trace=base_trace,
    )
    detail = dict(base_detail)
    detail.update({
        "nested_v42_selected": bool(base_selected is not None),
        "blocker_conditioned_query_attempted": False,
        "blocker_conditioned_query_selected": False,
        "blocker_conditioned_query_agent_count": 0,
        "blocker_conditioned_query_ready_agent_count": 0,
        "blocker_conditioned_query_hypotheses_evaluated": 0,
        "blocker_conditioned_query_unsupported_blocker_rejects": 0,
        "blocker_conditioned_query_root_unrecoverable_rejects": 0,
        "blocker_conditioned_query_environment_cache_hits": 0,
        "blocker_conditioned_query_joint_cache_hits": 0,
        "blocker_conditioned_query_successor_context_cache_hits": 0,
        "blocker_conditioned_query_candidate_agents_before_exact_filter": 0,
        "blocker_conditioned_query_exact_blocker_agent_count": 0,
        "blocker_conditioned_query_replayed_hypothesis_count": 0,
    })
    if base_selected is not None:
        detail["selected_certificate_kind"] = str(
            base_detail.get("selected_certificate_kind", "nested_v42")
        )
        return base_selected, detail

    query_idx = np.asarray(blocker_query_track_index, dtype=np.int64).reshape(-1)
    if query_idx.size <= 0:
        detail["blocker_conditioned_query_failure_reason"] = "no_late_bound_blocker_query"
        return None, detail
    query_traj_all: np.ndarray | None = None
    query_logits_all: np.ndarray | None = None
    if blocker_query_decoder is None:
        if blocker_query_trajectories is None or blocker_query_logits is None:
            detail["blocker_conditioned_query_failure_reason"] = "no_late_bound_blocker_query"
            return None, detail
        query_traj_all = np.asarray(blocker_query_trajectories, dtype=np.float32)
        query_logits_all = np.asarray(blocker_query_logits, dtype=np.float32)
        if query_traj_all.ndim != 4 or query_logits_all.ndim != 2:
            detail["blocker_conditioned_query_failure_reason"] = "no_late_bound_blocker_query"
            return None, detail
        q_all = min(query_idx.size, query_traj_all.shape[0], query_logits_all.shape[0])
        query_idx = query_idx[:q_all]
        query_traj_all = query_traj_all[:q_all]
        query_logits_all = query_logits_all[:q_all]
    else:
        q_all = int(query_idx.size)
    if q_all <= 0:
        detail["blocker_conditioned_query_failure_reason"] = "no_late_bound_blocker_query"
        return None, detail
    detail["blocker_conditioned_query_candidate_agents_before_exact_filter"] = int(q_all)

    # Runtime-fidelity repair: V43 late-bound support can only change a V42
    # hypothesis that failed specifically because an exact collision blocker had
    # no response support.  Hypotheses that already failed residual physical,
    # root feasibility, joint compatibility, or no-blocker checks are invariant
    # to adding support for previously unsupported agents.  Restricting the
    # second pass to that repairable subset is therefore logically exact.
    unsupported_hypotheses = np.asarray(
        base_trace.get("unsupported_hypothesis_indices", []), dtype=np.int64
    ).reshape(-1)
    unsupported_blockers = {
        int(x) for x in base_trace.get("unsupported_blocker_union", set())
    }
    if unsupported_hypotheses.size <= 0 or not unsupported_blockers:
        detail["blocker_conditioned_query_failure_reason"] = "no_unsupported_blocker_hypothesis"
        return None, detail
    keep = np.asarray([int(x) in unsupported_blockers for x in query_idx.tolist()], dtype=bool)
    filtered_idx = query_idx[keep]
    if filtered_idx.size <= 0:
        detail["blocker_conditioned_query_failure_reason"] = "no_model_visible_unsupported_blocker_query"
        return None, detail
    if blocker_query_decoder is not None:
        query_traj, query_logits = blocker_query_decoder(filtered_idx)
        query_traj = np.asarray(query_traj, dtype=np.float32)
        query_logits = np.asarray(query_logits, dtype=np.float32)
        q = min(filtered_idx.size, query_traj.shape[0] if query_traj.ndim == 4 else 0,
                query_logits.shape[0] if query_logits.ndim == 2 else 0)
        query_idx = filtered_idx[:q]
        query_traj = query_traj[:q] if query_traj.ndim == 4 else np.zeros((0, 0, 0, 7), dtype=np.float32)
        query_logits = query_logits[:q] if query_logits.ndim == 2 else np.zeros((0, 0), dtype=np.float32)
    else:
        assert query_traj_all is not None and query_logits_all is not None
        query_idx = filtered_idx
        query_traj = query_traj_all[keep]
        query_logits = query_logits_all[keep]
        q = int(query_idx.size)
    detail["blocker_conditioned_query_exact_blocker_agent_count"] = int(q)
    detail["blocker_conditioned_query_replayed_hypothesis_count"] = int(
        np.unique(unsupported_hypotheses).size
    )
    if q <= 0:
        detail["blocker_conditioned_query_failure_reason"] = "no_ready_exact_blocker_decode"
        return None, detail
    detail["blocker_conditioned_query_attempted"] = True
    detail["blocker_conditioned_query_agent_count"] = int(q)

    base_idx = np.asarray(critical_track_index, dtype=np.int64).reshape(-1)
    base_valid = np.asarray(critical_valid, dtype=bool).reshape(-1)
    base_traj = np.asarray(natural_trajectories, dtype=np.float32)
    base_logits = np.asarray(natural_logits, dtype=np.float32)
    a = min(base_idx.size, base_valid.size, base_traj.shape[0] if base_traj.ndim == 4 else 0,
            base_logits.shape[0] if base_logits.ndim == 2 else 0)
    if a > 0:
        combined_idx = np.concatenate([base_idx[:a], query_idx], axis=0)
        combined_valid = np.concatenate([base_valid[:a], np.ones(q, dtype=bool)], axis=0)
        combined_traj = np.concatenate([base_traj[:a], query_traj], axis=0)
        combined_logits = np.concatenate([base_logits[:a], query_logits], axis=0)
    else:
        combined_idx = query_idx.copy()
        combined_valid = np.ones(q, dtype=bool)
        combined_traj = query_traj.copy()
        combined_logits = query_logits.copy()

    prepared_support: dict[int, dict[str, Any]] | None = None
    prepared_support_detail: dict[str, Any] | None = None
    base_prepared_support = base_trace.get("prepared_response_support")
    base_prepared_detail = base_trace.get("prepared_response_support_detail")
    base_agents = {int(x) for x in base_idx[:a].tolist()} if a > 0 else set()
    query_agents = {int(x) for x in query_idx.tolist()}
    if (
        isinstance(base_prepared_support, dict)
        and isinstance(base_prepared_detail, dict)
        and base_agents.isdisjoint(query_agents)
    ):
        # R3 runtime-only support reuse.  The online blocker-query contract makes
        # these agent sets disjoint.  Prepare response banks only for the newly
        # decoded exact blockers, then merge the per-agent support with the exact
        # V42 first-pass support.
        late_support, late_support_detail = _prepare_interaction_response_support_np(
            np.asarray(agent_state, dtype=np.float32), int(sdc_index),
            query_idx, np.ones(q, dtype=bool), query_traj, query_logits,
            object_types, roadgraph, cfg,
        )
        prepared_support = dict(base_prepared_support)
        prepared_support.update(late_support)
        prepared_support_detail = _merge_interaction_response_support_details_np(
            base_prepared_detail, late_support_detail,
        )

    expanded_selected, expanded_detail = _construct_interaction_aware_reachable_response_envelope_np(
        agent_state, sdc_index, nominal_trajectories, cand_valid,
        nominal_roadgraph_safe, macro_types, fallback_scores,
        collision_prefix_steps, action_targets, action_accels,
        roadgraph, cfg, previous_longitudinal_accel,
        base_candidate_index=int(base_candidate_index),
        critical_track_index=combined_idx,
        critical_valid=combined_valid,
        natural_trajectories=combined_traj,
        natural_logits=combined_logits,
        object_types=object_types,
        shared_compatibility_cache=shared_compatibility_cache,
        shared_successor_context_cache=shared_successor_context_cache,
        prepared_response_support=prepared_support,
        prepared_response_support_detail=prepared_support_detail,
        prepared_hypothesis_workspace=(
            base_trace.get("prepared_hypothesis_workspace")
            if isinstance(base_trace.get("prepared_hypothesis_workspace"), dict) else None
        ),
        prepared_unsupported_replay_cache=(
            base_trace.get("unsupported_replay_cache")
            if isinstance(base_trace.get("unsupported_replay_cache"), dict) else None
        ),
        known_nested_v39_empty=True,
        hypothesis_indices=unsupported_hypotheses,
    )
    support_detail = expanded_detail.get("interaction_support_detail", {})
    base_ready = int(base_detail.get("interaction_support_agents_ready", 0))
    expanded_ready = int(expanded_detail.get("interaction_support_agents_ready", 0))
    # Query indices explicitly exclude original critical agents, so the ready
    # count increment is an exact late-bound readiness count.
    late_ready = max(expanded_ready - base_ready, 0)
    detail["blocker_conditioned_query_ready_agent_count"] = int(late_ready)
    detail["blocker_conditioned_query_hypotheses_evaluated"] = int(
        expanded_detail.get("interaction_hypotheses_evaluated", 0)
    )
    detail["blocker_conditioned_query_unsupported_blocker_rejects"] = int(
        expanded_detail.get("interaction_unsupported_blocker_rejects", 0)
    )
    detail["blocker_conditioned_query_root_unrecoverable_rejects"] = int(
        expanded_detail.get("interaction_root_unrecoverable_rejects", 0)
    )
    detail["blocker_conditioned_query_environment_cache_hits"] = int(
        expanded_detail.get("interaction_environment_compatibility_cache_hits", 0)
    )
    detail["blocker_conditioned_query_joint_cache_hits"] = int(
        expanded_detail.get("interaction_joint_compatibility_cache_hits", 0)
    )
    detail["blocker_conditioned_query_successor_context_cache_hits"] = int(
        expanded_detail.get("interaction_successor_context_cache_hits", 0)
    )
    if expanded_selected is None:
        detail["blocker_conditioned_query_failure_reason"] = str(
            expanded_detail.get("interaction_failure_reason", "no_blocker_conditioned_certificate")
        )
        return None, detail

    # Preserve the expanded certificate diagnostics while explicitly marking
    # that the selected action required late-bound blocker support.
    selected_detail = dict(expanded_detail)
    selected_detail.update(detail)
    selected_detail.update({
        "selected": True,
        "nested_v42_selected": False,
        "blocker_conditioned_query_attempted": True,
        "blocker_conditioned_query_selected": True,
        "selected_certificate_kind": "blocker_conditioned_interaction_aware_reachable_response_envelope",
        "selected_is_interaction_response": True,
        "interaction_response_selected": True,
    })
    return expanded_selected, selected_detail

def _collision_free_against_constant_velocity(
    traj: np.ndarray,
    agent_state: np.ndarray,
    sdc_index: int,
    cfg: dict,
    other_future_trajs: np.ndarray | None = None,
    *,
    prepared_context: dict[str, Any] | None = None,
) -> bool:
    """Causal conventional collision screen, with an exact-equivalent fast path."""
    context = prepared_context
    if context is None:
        context = _prepare_collision_check_context(
            agent_state,
            sdc_index,
            cfg,
            horizon_steps=int(len(traj)),
            other_future_trajs=other_future_trajs,
        )
    return bool(_collision_audit_against_context(traj, context)["safe"])


def _add_candidate(
    out: list[np.ndarray],
    macros: list[int],
    utils: list[float],
    valids: list[bool],
    traj: np.ndarray,
    macro: MacroType,
    utility: float,
    agent_state: np.ndarray,
    sdc_index: int,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
    other_future_trajs: np.ndarray | None = None,
    *,
    collision_ctx: dict[str, Any] | None = None,
    audits: list[dict[str, Any]] | None = None,
) -> None:
    if len(out) >= int(cfg.get("limits", {}).get("max_candidates", 64)):
        return
    traj = repair_planar_kinematics(traj, agent_state[sdc_index], float(cfg.get("time", {}).get("dt", 0.1)))
    if not _candidate_dyn_ok(traj, cfg):
        return
    # De-duplicate by endpoint to keep the limited candidate tensor useful.  The
    # original 0.35 m threshold removed most short-horizon smoke-test candidates
    # (especially low-speed accel/yield primitives), leaving the online planner
    # with too few alternatives and causing brittle fallback.
    end = traj[-1, :2]
    dedup_eps = float(cfg.get("planning", {}).get("online_candidate_dedup_endpoint_m", cfg.get("candidate", {}).get("dedup_endpoint_tolerance_m", 0.25)))
    dedup_eps = float(np.clip(dedup_eps, 0.05, 0.35))
    end_speed = float(np.linalg.norm(traj[-1, 3:5]))
    speed_eps = float(cfg.get("planning", {}).get("online_candidate_dedup_speed_mps", cfg.get("candidate", {}).get("dedup_speed_tolerance_mps", 0.15)))
    same_macro_only = bool(cfg.get("planning", {}).get("online_dedup_same_macro_only", True))
    for old_traj, old_macro in zip(out, macros):
        if same_macro_only and int(old_macro) != int(macro):
            continue
        old_speed = float(np.linalg.norm(old_traj[-1, 3:5]))
        if np.linalg.norm(old_traj[-1, :2] - end) < dedup_eps and abs(old_speed - end_speed) < speed_eps:
            return

    # Conventional-safe remains the exact v16.8.27 contract: roadgraph audit AND
    # the full configured causal collision screen.  v16.8.29 records the two
    # components separately and exposes the collision-safe prefix only for an
    # explicitly uncertified recovery branch.
    road_ok = bool(_roadgraph_drivable_mask(traj, roadgraph))
    if collision_ctx is not None:
        collision_audit = _collision_audit_against_context(traj, collision_ctx)
        collision_ok = bool(collision_audit["safe"])
    elif road_ok:
        collision_ok = bool(_collision_free_against_constant_velocity(
            traj, agent_state, sdc_index, cfg, other_future_trajs=other_future_trajs
        ))
        collision_audit = {
            "safe": collision_ok,
            "safe_prefix_steps": int(len(traj)) if collision_ok else 0,
            "min_clearance_margin_m": 0.0,
            "violation_source": "unknown",
        }
    else:
        # Preserve the historical short-circuit for direct helper/unit-test calls.
        collision_ok = False
        collision_audit = {
            "safe": False,
            "safe_prefix_steps": 0,
            "min_clearance_margin_m": 0.0,
            "violation_source": "not_evaluated_roadgraph_failed",
        }
    conv = bool(road_ok and collision_ok)
    out.append(traj.astype(np.float32))
    macros.append(int(macro))
    utils.append(float(utility))
    valids.append(conv)
    if audits is not None:
        audits.append({
            "roadgraph_safe": bool(road_ok),
            "collision_safe": bool(collision_ok),
            "collision_safe_prefix_steps": int(collision_audit["safe_prefix_steps"]),
            "collision_min_clearance_margin_m": float(collision_audit["min_clearance_margin_m"]),
            "collision_violation_source": str(collision_audit["violation_source"]),
        })

def _online_arrival_accel(distance_m: float, speed_mps: float, target_time_s: float, cfg: dict) -> float | None:
    """Bounded constant-acceleration solution for an online conflict arrival time."""
    d = float(distance_m)
    t = float(target_time_s)
    if not np.isfinite(d) or not np.isfinite(t) or d <= 0.5 or t <= 0.35:
        return None
    v0 = max(float(speed_mps), 0.0)
    a = 2.0 * (d - v0 * t) / max(t * t, 1.0e-3)
    pcfg = cfg.get("planning", {})
    lo = float(pcfg.get("online_timing_envelope_min_accel_mps2", cfg.get("candidate", {}).get("timing_envelope_min_accel_mps2", -3.5)))
    hi = float(pcfg.get("online_timing_envelope_max_accel_mps2", cfg.get("candidate", {}).get("timing_envelope_max_accel_mps2", 2.5)))
    if a < lo - 0.75 or a > hi + 0.75:
        return None
    a = float(np.clip(a, lo, hi))
    if v0 + a * t < -1.0e-3:
        return None
    return a


def _online_smooth_arrival_profile(distance_m: float, speed_mps: float, target_time_s: float, cfg: dict) -> tuple[float, float, float] | None:
    """Feasibility precheck for the online cubic timing projection.

    Mirrors the offline BCS-RMR-BCTE profile: reach the fixed conflict-path
    distance at T while tapering longitudinal acceleration to zero.  The
    profile avoids the stop-before-arrival pathology of constant deceleration
    and makes offline/online timing semantics materially closer.
    """
    d = float(distance_m)
    T = float(target_time_s)
    v0 = max(float(speed_mps), 0.0)
    if not np.isfinite(d) or not np.isfinite(T) or d <= 0.5 or T <= 0.35:
        return None
    a0 = 3.0 * (d - v0 * T) / max(T * T, 1.0e-9)
    jerk = -a0 / max(T, 1.0e-9)
    vT = -0.5 * v0 + 1.5 * d / max(T, 1.0e-9)
    pcfg = cfg.get("planning", {})
    ccfg = cfg.get("candidate", {})
    lo = float(pcfg.get("online_timing_envelope_min_accel_mps2", ccfg.get("timing_envelope_min_accel_mps2", -3.5)))
    hi = float(pcfg.get("online_timing_envelope_max_accel_mps2", ccfg.get("timing_envelope_max_accel_mps2", 2.5)))
    max_jerk = float(ccfg.get("max_jerk_mps3", 8.0))
    if a0 < lo - 1.0e-6 or a0 > hi + 1.0e-6 or vT < -1.0e-5 or abs(jerk) > max_jerk + 1.0e-6:
        return None
    return float(a0), float(jerk), float(max(vT, 0.0))


def _route_lane_aware_candidates(
    agent_state: np.ndarray,
    sdc_index: int,
    roadgraph: dict[str, np.ndarray],
    cfg: dict,
    *,
    other_future_trajs: np.ndarray | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    limits = cfg.get("limits", {})
    K = int(limits.get("max_candidates", 64))
    H = int(cfg.get("time", {}).get("future_steps", cfg.get("eval", {}).get("rollout_horizon_steps", 80)))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    current = agent_state[sdc_index].copy()
    current[6] = _nearest_lane_heading(current, roadgraph)
    speed = max(float(current[5]), 0.0)
    candidates: list[np.ndarray] = []
    macros: list[int] = []
    utils: list[float] = []
    conventional: list[bool] = []
    audits: list[dict[str, Any]] = []
    collision_ctx = _prepare_collision_check_context(
        agent_state, sdc_index, cfg, horizon_steps=H, other_future_trajs=other_future_trajs
    )

    cand_cfg = cfg.get("candidate", {})
    acc_bank = [0.0]
    acc_bank.extend(float(x) for x in cand_cfg.get("yield_decel_values_mps2", [-1.0, -2.0, -3.0]))
    acc_bank.extend(float(x) for x in cand_cfg.get("accelerate_values_mps2", [0.5, 1.0, 1.5]))
    # Online closed-loop states are more diverse than the offline root cache.
    # Add a denser longitudinal bank so COWP can trade progress against safety
    # instead of being forced into fallback whenever the few root primitives fail.
    acc_bank.extend(float(x) for x in cfg.get("planning", {}).get("online_extra_accel_values_mps2", [-4.0, -2.5, -1.5, -0.5, 0.25, 0.75, 1.25, 2.0]))
    acc_bank = sorted(set(round(float(a), 3) for a in acc_bank))
    # Route/keep-lane timing lattice.
    for acc in acc_bank:
        macro = MacroType.KEEP_LANE if abs(acc) < 1e-6 else (MacroType.ACCELERATE_CROSS if acc > 0 else MacroType.YIELD)
        tr = constant_accel_trajectory(current, H, dt, accel=acc)
        progress = float(np.linalg.norm(tr[-1, :2] - tr[0, :2]))
        # Lower score is better. Prefer progress, penalize aggressive accel.
        util = -0.03 * progress + 0.08 * abs(acc)
        _add_candidate(candidates, macros, utils, conventional, tr, macro, util, agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs, collision_ctx=collision_ctx, audits=audits)

    # Reserve the core neutral option before optional timing/lane-change banks can
    # saturate K.  Offline v16.8.6 uses the same reservation principle.
    neutral_tr = smooth_stop_trajectory(current, H, dt, decel=float(cfg.get("planning", {}).get("fallback_decel_mps2", -2.0)))
    _add_candidate(candidates, macros, utils, conventional, neutral_tr, MacroType.NEUTRAL_EGO, 0.8, agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs, collision_ctx=collision_ctx, audits=audits)

    # v16.8.8 online base-bank reservation.  Optional RMR/PSY refinements are
    # allowed to use spare slots, but they must not erase both lateral escape
    # directions when K becomes saturated.  Reserve only a compact canonical
    # subset here; the richer lane-change bank is still expanded later if space
    # remains.
    reserve_delays = [float(x) for x in cfg.get("planning", {}).get("online_lane_change_reserve_delays_s", [0.5, 1.5])]
    lane_widths = cfg.get("planning", {}).get("online_lane_change_offsets_m", [-3.5, 3.5])
    for lateral_offset in lane_widths:
        macro = MacroType.LANE_CHANGE_LEFT if float(lateral_offset) > 0 else MacroType.LANE_CHANGE_RIGHT
        for delay in reserve_delays:
            tr = _quintic_frenet_trajectory(
                current, H, dt, accel=0.0, lateral_offset=float(lateral_offset),
                start_delay_s=float(delay),
                lane_change_duration_s=float(cfg.get("planning", {}).get("online_lane_change_duration_s", 4.0)),
            )
            progress = float(np.linalg.norm(tr[-1, :2] - tr[0, :2]))
            _add_candidate(
                candidates, macros, utils, conventional, tr, macro,
                -0.02 * progress + 0.15 + 0.03 * float(delay),
                agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs, collision_ctx=collision_ctx, audits=audits,
            )

    # Stop / yield before likely conflicts.  In online mode we do not have proto
    # conflict regions, so estimate conflict distance from nearest forward agent
    # or route distance.
    ego_xy = current[:2]
    dir_vec = np.array([np.cos(current[6]), np.sin(current[6])], dtype=np.float32)
    rel = agent_state[:, :2] - ego_xy[None]
    along = rel @ dir_vec
    lateral = np.abs(rel @ np.array([-dir_vec[1], dir_vec[0]], dtype=np.float32))
    valid_other = (agent_state[:, 10] > 0.5)
    valid_other[sdc_index] = False
    forward = valid_other & (along > 3.0) & (along < 60.0) & (lateral < 8.0)
    conflict_dists = [float(x) for x in along[forward][:4]] if np.any(forward) else [max(12.0, speed * 2.5), max(22.0, speed * 4.0)]
    for dist in sorted(conflict_dists)[:4]:
        for margin in cand_cfg.get("stop_margin_to_conflict_m", [2.0, 5.0, 8.0]):
            tr = smooth_stop_trajectory(current, H, dt, decel=-2.0, stop_after_m=max(0.0, float(dist) - float(margin)))
            _add_candidate(candidates, macros, utils, conventional, tr, MacroType.STOP_BEFORE_CONFLICT, 0.4 + 0.02 * max(0.0, 20.0 - dist), agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs, collision_ctx=collision_ctx, audits=audits)

    # Merge-ahead/behind timing around nearby agents: vary speed to pass before or
    # after their projected conflict time.  This is still primitive-based, but it
    # creates the same root aggressive-vs-yield contrast as the label lattice.
    if np.any(valid_other):
        near_order = np.argsort(np.linalg.norm(agent_state[:, :2] - ego_xy[None], axis=-1))
        for j in near_order[: min(8, len(near_order))]:
            if j == sdc_index or not valid_other[j]:
                continue
            rel_j = agent_state[j, :2] - ego_xy
            s = float(rel_j @ dir_vec)
            if -10.0 <= s <= 55.0:
                for offset in cand_cfg.get("merge_time_offsets_s", [-1.2, -0.4, 0.4, 1.2]):
                    acc = float(np.clip(-0.9 * float(offset), -3.0, 2.0))
                    tr = constant_accel_trajectory(current, H, dt, accel=acc)
                    m = MacroType.MERGE_AHEAD if offset <= 0 else MacroType.MERGE_BEHIND
                    _add_candidate(candidates, macros, utils, conventional, tr, m, 0.05 + max(0.0, offset), agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs, collision_ctx=collision_ctx, audits=audits)

    # v16.8.4 online BCS timing projection.  Waymax does not expose the offline
    # proto conflict-region graph at every policy step, so we retain the causal
    # closest-approach event locator, but project ego timing with the same smooth
    # arrival family used offline.  This removes the constant-deceleration
    # stop-before-arrival pathology and includes current->first-step arc length.
    nominal = constant_accel_trajectory(current, H, dt, accel=0.0)
    nominal_xy = nominal[:, :2]
    nominal_xy_from_current = np.concatenate([current[None, :2], nominal_xy], axis=0)
    nominal_step = np.linalg.norm(np.diff(nominal_xy_from_current, axis=0), axis=-1)
    nominal_s = np.cumsum(nominal_step)
    max_bcte_agents = int(cfg.get("planning", {}).get("online_timing_envelope_max_agents", 6))
    max_bcte_candidates = int(cfg.get("planning", {}).get("online_timing_envelope_max_candidates", 24))
    max_ca_dist = float(cfg.get("planning", {}).get("online_timing_envelope_max_closest_m", 14.0))
    time_uncertainty = max(0.0, float(cfg.get("planning", {}).get("online_timing_envelope_uncertainty_s", 0.8)))
    accel_dedup = max(
        1.0e-3,
        float(cfg.get("planning", {}).get(
            "online_timing_envelope_accel_dedup_mps2",
            cfg.get("candidate", {}).get("timing_envelope_accel_dedup_mps2", 0.10),
        )),
    )
    gap_values = [float(x) for x in cfg.get("planning", {}).get(
        "online_timing_envelope_gap_s", cfg.get("candidate", {}).get("timing_envelope_gap_s", [0.8, 1.4, 2.0])
    )]
    if np.any(valid_other):
        near_order = np.argsort(np.linalg.norm(agent_state[:, :2] - ego_xy[None], axis=-1))
        used = 0
        bcte_added = 0
        timing_profile_bins: set[tuple[int, int, int]] = set()
        for j in near_order:
            if used >= max_bcte_agents or bcte_added >= max_bcte_candidates or len(candidates) >= K:
                break
            if j == sdc_index or not valid_other[j]:
                continue
            pred_j, cv_j = _agent_future_xy(agent_state, int(j), H, dt, other_future_trajs)
            agent_xy = pred_j if np.isfinite(pred_j).all() else cv_j
            if not np.isfinite(agent_xy).all():
                continue
            d = np.linalg.norm(nominal_xy - agent_xy[:H, :2], axis=-1)
            q = int(np.argmin(d))
            if float(d[q]) > max_ca_dist or q < 2:
                continue
            conflict_distance = float(nominal_s[q])
            event_time = float((q + 1) * dt)
            for gap in gap_values:
                if bcte_added >= max_bcte_candidates or len(candidates) >= K:
                    break
                for side in (-1.0, 1.0):
                    # Before uses the early boundary; after uses the late
                    # boundary. The symmetric uncertainty is deliberately
                    # conservative until a learned arrival distribution is
                    # introduced in the distributional-RCOT extension.
                    target_time = event_time + side * (gap + time_uncertainty)
                    if target_time <= 0.35 or target_time > float(H * dt):
                        continue
                    profile = _online_smooth_arrival_profile(conflict_distance, speed, target_time, cfg)
                    if profile is None:
                        continue
                    initial_accel, _profile_jerk, _terminal_speed = profile
                    accel_bin = int(round(float(initial_accel) / accel_dedup))
                    profile_key = (int(j), int(np.sign(side)), accel_bin)
                    if profile_key in timing_profile_bins:
                        continue
                    behind = side > 0.0
                    tr = smooth_arrival_trajectory(
                        current, H, dt, distance_m=conflict_distance, target_time_s=target_time
                    )
                    if tr is None:
                        continue
                    before_count = len(candidates)
                    _add_candidate(
                        candidates, macros, utils, conventional, tr,
                        MacroType.MERGE_BEHIND if behind else MacroType.MERGE_AHEAD,
                        0.12 + (0.08 * gap if behind else 0.03 * gap),
                        agent_state, sdc_index, roadgraph, cfg,
                        other_future_trajs=other_future_trajs, collision_ctx=collision_ctx, audits=audits,
                    )
                    if len(candidates) > before_count:
                        timing_profile_bins.add(profile_key)
                        bcte_added += 1
            used += 1

        # v16.8.6 online Priority-Commitment Hold-Release (PCHR).  Use the same
        # causal closest-approach event approximation as online BCS-RMR, but only
        # allocate hold-release candidates to priority-like interactions.
        if bool(cfg.get("planning", {}).get("online_priority_hold_release_enabled", cand_cfg.get("priority_hold_release_enabled", True))):
            phr_max = int(cfg.get("planning", {}).get("online_priority_hold_release_max_candidates", cand_cfg.get("priority_hold_release_max_candidates", 8)))
            phr_added = 0
            margins = [float(x) for x in cfg.get("planning", {}).get("online_priority_hold_release_stop_margin_m", cand_cfg.get("priority_hold_release_stop_margin_m", [3.0, 5.0]))]
            release_speeds = [float(x) for x in cfg.get("planning", {}).get("online_priority_hold_release_speed_mps", cand_cfg.get("priority_hold_release_speed_mps", [2.5, 3.5, 4.0]))]
            hold_s = float(cfg.get("planning", {}).get("online_priority_hold_release_min_hold_s", cand_cfg.get("priority_hold_release_min_hold_s", 0.35)))
            phr_gaps = [float(x) for x in cfg.get("planning", {}).get("online_priority_hold_release_gap_s", cand_cfg.get("priority_hold_release_gap_s", [0.8, 1.4, 2.0]))]
            ego_lat_vec = np.asarray([-dir_vec[1], dir_vec[0]], dtype=np.float32)
            for j in near_order:
                if phr_added >= phr_max or len(candidates) >= K:
                    break
                if j == sdc_index or not valid_other[j]:
                    continue
                rel_j = agent_state[j, :2] - ego_xy
                longitudinal = float(np.dot(rel_j, dir_vec))
                lateral = abs(float(np.dot(rel_j, ego_lat_vec)))
                closing = float(max(0.0, speed - np.dot(agent_state[j, 3:5], dir_vec)))
                ttc = longitudinal / max(closing, 1.0e-3) if longitudinal > 0.0 and closing > 0.25 else 99.0
                priority_like = (-8.0 <= longitudinal <= 55.0 and lateral <= 7.5) or ttc <= float(cfg.get("planning", {}).get("online_priority_ttc_s", 5.0))
                if not priority_like:
                    continue
                pred_j, cv_j = _agent_future_xy(agent_state, int(j), H, dt, other_future_trajs)
                agent_xy = pred_j if np.isfinite(pred_j).all() else cv_j
                if not np.isfinite(agent_xy).all():
                    continue
                dca = np.linalg.norm(nominal_xy - agent_xy[:H, :2], axis=-1)
                q = int(np.argmin(dca))
                if q < 2 or float(dca[q]) > max_ca_dist:
                    continue
                conflict_distance = float(nominal_s[q])
                event_time = float((q + 1) * dt)
                for gap in phr_gaps:
                    target_time = event_time + time_uncertainty + gap
                    if target_time <= 0.35 or target_time > float(H * dt):
                        continue
                    for margin in margins:
                        for release_speed in release_speeds:
                            if phr_added >= phr_max or len(candidates) >= K:
                                break
                            tr = priority_hold_release_trajectory(
                                current, H, dt, entry_distance_m=conflict_distance,
                                target_time_s=target_time, stop_margin_m=margin,
                                release_speed_mps=release_speed, min_hold_s=hold_s,
                            )
                            if tr is None:
                                continue
                            before = len(candidates)
                            _add_candidate(
                                candidates, macros, utils, conventional, tr, MacroType.MERGE_BEHIND,
                                0.18 + 0.06 * gap, agent_state, sdc_index, roadgraph, cfg,
                                other_future_trajs=other_future_trajs, collision_ctx=collision_ctx, audits=audits,
                            )
                            if len(candidates) > before:
                                phr_added += 1
                                break

        # v16.8.8 online Priority-Smooth-Yield (PSY).  Use the same causal
        # closest-approach event approximation as online RMR but replace the
        # rarely feasible full-stop PCHR with a low-entry-speed smooth arrival.
        if bool(cfg.get("planning", {}).get("online_priority_smooth_yield_enabled", cand_cfg.get("priority_smooth_yield_enabled", True))):
            psy_max = int(cfg.get("planning", {}).get("online_priority_smooth_yield_max_candidates", cand_cfg.get("priority_smooth_yield_max_candidates", 8)))
            psy_added = 0
            psy_vt = [float(x) for x in cfg.get("planning", {}).get("online_priority_smooth_yield_terminal_speed_mps", cand_cfg.get("priority_smooth_yield_terminal_speed_mps", [1.0, 2.0, 3.0]))]
            psy_a0 = [float(x) for x in cfg.get("planning", {}).get("online_priority_smooth_yield_initial_decel_mps2", cand_cfg.get("priority_smooth_yield_initial_decel_mps2", [-0.8, -1.4, -2.0]))]
            psy_gaps = [float(x) for x in cfg.get("planning", {}).get("online_priority_smooth_yield_gap_s", cand_cfg.get("priority_smooth_yield_gap_s", [0.8, 1.4, 2.0]))]
            commit_t = max(float(cfg.get("planning", {}).get("online_priority_smooth_yield_commitment_check_s", cand_cfg.get("priority_smooth_yield_commitment_check_s", 1.0))), dt)
            min_drop = max(0.0, float(cfg.get("planning", {}).get("online_priority_smooth_yield_min_speed_drop_mps", cand_cfg.get("priority_smooth_yield_min_speed_drop_mps", 0.75))))
            ego_lat_vec = np.asarray([-dir_vec[1], dir_vec[0]], dtype=np.float32)
            for j in near_order:
                if psy_added >= psy_max or len(candidates) >= K:
                    break
                if j == sdc_index or not valid_other[j]:
                    continue
                rel_j = agent_state[j, :2] - ego_xy
                longitudinal = float(np.dot(rel_j, dir_vec))
                lateral = abs(float(np.dot(rel_j, ego_lat_vec)))
                closing = float(max(0.0, speed - np.dot(agent_state[j, 3:5], dir_vec)))
                ttc = longitudinal / max(closing, 1.0e-3) if longitudinal > 0.0 and closing > 0.25 else 99.0
                priority_like = (-8.0 <= longitudinal <= 55.0 and lateral <= 7.5) or ttc <= float(cfg.get("planning", {}).get("online_priority_ttc_s", 5.0))
                if not priority_like:
                    continue
                pred_j, cv_j = _agent_future_xy(agent_state, int(j), H, dt, other_future_trajs)
                agent_xy = pred_j if np.isfinite(pred_j).all() else cv_j
                if not np.isfinite(agent_xy).all():
                    continue
                dca = np.linalg.norm(nominal_xy - agent_xy[:H, :2], axis=-1)
                q = int(np.argmin(dca))
                if q < 2 or float(dca[q]) > max_ca_dist:
                    continue
                conflict_distance = float(nominal_s[q])
                event_time = float((q + 1) * dt)
                for gap in psy_gaps:
                    target_time = event_time + time_uncertainty + gap
                    if target_time <= commit_t or target_time > float(H * dt):
                        continue
                    for initial_accel in psy_a0:
                        for terminal_speed in psy_vt:
                            if psy_added >= psy_max or len(candidates) >= K:
                                break
                            tr = smooth_terminal_speed_arrival_trajectory(
                                current, H, dt, distance_m=conflict_distance,
                                target_time_s=target_time, terminal_speed_mps=terminal_speed,
                                initial_accel_mps2=initial_accel,
                            )
                            if tr is None:
                                continue
                            idx = min(max(int(round(commit_t / dt)) - 1, 0), H - 1)
                            check_speed = float(np.linalg.norm(tr[idx, 3:5]))
                            if speed > 1.5 and check_speed > max(speed - min_drop, 0.5) + 1.0e-6:
                                continue
                            before = len(candidates)
                            _add_candidate(
                                candidates, macros, utils, conventional, tr, MacroType.MERGE_BEHIND,
                                0.16 + 0.05 * gap, agent_state, sdc_index, roadgraph, cfg,
                                other_future_trajs=other_future_trajs, collision_ctx=collision_ctx, audits=audits,
                            )
                            if len(candidates) > before:
                                psy_added += 1

    # Lane-change/cut-in proposals.  Use ±lane_width relative to the local lane
    # heading; the conventional mask filters obvious offroad/collision cases.
    lane_widths = cfg.get("planning", {}).get("online_lane_change_offsets_m", [-3.5, 3.5])
    for lateral_offset in lane_widths:
        macro = MacroType.LANE_CHANGE_LEFT if lateral_offset > 0 else MacroType.LANE_CHANGE_RIGHT
        for delay in cand_cfg.get("lane_change_start_delay_s", [0.0, 0.5, 1.0, 1.5, 2.0]):
            for acc in [0.0, -0.5, 0.5]:
                tr = _quintic_frenet_trajectory(
                    current, H, dt, accel=acc, lateral_offset=float(lateral_offset),
                    start_delay_s=float(delay),
                    lane_change_duration_s=float(cfg.get("planning", {}).get("online_lane_change_duration_s", 4.0)),
                )
                progress = float(np.linalg.norm(tr[-1, :2] - tr[0, :2]))
                _add_candidate(candidates, macros, utils, conventional, tr, macro, -0.02 * progress + 0.15 + 0.03 * float(delay), agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs, collision_ctx=collision_ctx, audits=audits)

    # Terminal position-speed frontier fill.  This exposes distinct progress/yield
    # alternatives after closed-loop states have drifted away from the root cache.
    target_online = int(cfg.get("planning", {}).get("target_online_candidates", min(32, K)))
    if len(candidates) < min(target_online, K):
        speed_offsets = [float(x) for x in cfg.get("planning", {}).get(
            "online_terminal_speed_offsets_mps", [-5.0, -3.0, -1.5, 0.0, 1.5, 3.0]
        )]
        s_offsets = [float(x) for x in cfg.get("planning", {}).get(
            "online_terminal_s_offsets_m", [-16.0, -8.0, 0.0, 8.0, 16.0]
        )]
        lat_offsets = [0.0] + [float(x) for x in cfg.get("planning", {}).get("online_terminal_lateral_offsets_m", [])]
        nominal_s = max(float(speed) * float(H) * float(dt), 4.0)
        for lat in lat_offsets:
            for ds in s_offsets:
                for dv in speed_offsets:
                    if len(candidates) >= min(target_online, K) or len(candidates) >= K:
                        break
                    target_speed = max(0.0, speed + dv)
                    target_s = max(1.0, nominal_s + ds)
                    tr = _terminal_frenet_trajectory(
                        current, H, dt, target_s=target_s, target_speed=target_speed,
                        lateral_offset=float(lat),
                        lane_change_duration_s=float(cfg.get("planning", {}).get("online_lane_change_duration_s", 4.0)),
                    )
                    if abs(lat) > 1.0:
                        macro = MacroType.LANE_CHANGE_LEFT if lat > 0 else MacroType.LANE_CHANGE_RIGHT
                    elif target_speed > speed + 0.75 or target_s > nominal_s + 4.0:
                        macro = MacroType.ACCELERATE_CROSS
                    elif target_speed < max(speed - 0.75, 0.0) or target_s < nominal_s - 4.0:
                        macro = MacroType.YIELD
                    else:
                        macro = MacroType.KEEP_LANE
                    util = -0.025 * float(target_s) + 0.08 * abs(float(dv)) + 0.10 * abs(float(lat))
                    _add_candidate(
                        candidates, macros, utils, conventional, tr, macro, util,
                        agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs, collision_ctx=collision_ctx, audits=audits
                    )

    # Ensure the online batch contains a minimally useful ego-motion set even in
    # low-speed scenes where endpoint de-duplication and dynamics checks collapse
    # many primitives.  Every supplemental/neutral candidate is still subjected
    # to the same conventional-safety audit; failed candidates remain valid only
    # for the explicitly uncertified last-resort pool.
    min_online = int(cfg.get("planning", {}).get("min_online_candidates", min(8, K)))
    if len(candidates) < min_online:
        supplemental_acc = [0.25, -0.25, 0.75, -0.75, 1.25, -1.25]
        for acc in supplemental_acc:
            if len(candidates) >= min_online or len(candidates) >= K:
                break
            macro = MacroType.ACCELERATE_CROSS if acc > 0 else MacroType.YIELD
            tr = constant_accel_trajectory(current, H, dt, accel=float(acc))
            progress = float(np.linalg.norm(tr[-1, :2] - tr[0, :2]))
            _add_candidate(candidates, macros, utils, conventional, tr, macro, -0.02 * progress + 0.10 * abs(acc), agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs, collision_ctx=collision_ctx, audits=audits)

    # Final neutral retry (normally de-duplicated against the reserved neutral slot).
    if len(candidates) < K:
        tr = smooth_stop_trajectory(current, H, dt, decel=float(cfg.get("planning", {}).get("fallback_decel_mps2", -2.0)))
        _add_candidate(candidates, macros, utils, conventional, tr, MacroType.NEUTRAL_EGO, 0.8, agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs, collision_ctx=collision_ctx, audits=audits)

    traj = np.zeros((K, H, 7), dtype=np.float32)
    valid = np.zeros(K, dtype=bool)
    conventional_safe = np.zeros(K, dtype=bool)
    macro = np.full(K, int(MacroType.PAD), dtype=np.int64)
    utility = np.zeros(K, dtype=np.float32)
    roadgraph_safe = np.zeros(K, dtype=bool)
    collision_safe = np.zeros(K, dtype=bool)
    collision_safe_prefix_steps = np.zeros(K, dtype=np.int32)
    collision_min_clearance_margin_m = np.full(K, -999.0, dtype=np.float32)
    for i, tr in enumerate(candidates[:K]):
        traj[i] = tr
        valid[i] = True
        conventional_safe[i] = bool(conventional[i])
        macro[i] = int(macros[i])
        utility[i] = float(utils[i])
        if i < len(audits):
            audit = audits[i]
            roadgraph_safe[i] = bool(audit.get("roadgraph_safe", False))
            collision_safe[i] = bool(audit.get("collision_safe", False))
            collision_safe_prefix_steps[i] = int(audit.get("collision_safe_prefix_steps", 0))
            collision_min_clearance_margin_m[i] = np.float32(audit.get("collision_min_clearance_margin_m", -999.0))
    return (
        traj, valid, conventional_safe, macro, utility,
        roadgraph_safe, collision_safe, collision_safe_prefix_steps, collision_min_clearance_margin_m,
    )


def _critical_interaction_rank(agent_state: np.ndarray, sdc_index: int, candidates: np.ndarray, cand_valid: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Select a small, risk-focused critical-agent set.

    The original online builder filled almost every available slot, including
    weakly related agents.  That creates a severe train/online distribution shift
    and lets the max-over-agents witness gate reject nearly every candidate.
    """
    A = int(cfg.get("limits", {}).get("max_critical_agents", 8))
    plan_cfg = cfg.get("planning", {})
    active_cap = min(A, int(plan_cfg.get("max_online_critical_agents", 4)))
    min_score = float(plan_cfg.get("online_critical_score_threshold", 1.20))
    max_now = float(plan_cfg.get("online_critical_max_distance_m", 55.0))
    max_closest = float(plan_cfg.get("online_critical_max_closest_m", 18.0))
    valid_agents = agent_state[:, 10] > 0.5
    ego_xy = agent_state[sdc_index, :2]
    dist_now = np.linalg.norm(agent_state[:, :2] - ego_xy[None], axis=-1)
    scores = np.full(agent_state.shape[0], -1e9, dtype=np.float32)
    closest = np.full(agent_state.shape[0], np.inf, dtype=np.float32)
    cand_xy = candidates[cand_valid, :, :2] if np.any(cand_valid) else candidates[:1, :, :2]
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    t = np.arange(1, cand_xy.shape[1] + 1, dtype=np.float32)[None, :, None] * dt
    ego_vel = agent_state[sdc_index, 3:5]
    for j in range(agent_state.shape[0]):
        if j == sdc_index or not valid_agents[j] or dist_now[j] > max_now:
            continue
        pred = agent_state[j, :2][None, None, :] + agent_state[j, 3:5][None, None, :] * t
        min_dist = float(np.min(np.linalg.norm(cand_xy - pred, axis=-1))) if cand_xy.size else float(dist_now[j])
        closest[j] = min_dist
        rel = agent_state[j, :2] - ego_xy
        closing = -float(np.dot(rel, agent_state[j, 3:5] - ego_vel)) / max(float(np.linalg.norm(rel)), 1e-3)
        time_risk = np.exp(-max(0.0, min_dist) / 8.0)
        proximity = np.exp(-float(dist_now[j]) / 22.0)
        closing_bonus = 0.75 if closing > 1.0 else (0.35 if closing > 0.25 else 0.0)
        scores[j] = 3.5 * time_risk + 2.0 * proximity + closing_bonus
    order = [
        i for i in np.argsort(-scores).tolist()
        if i != sdc_index and valid_agents[i] and scores[i] >= min_score and closest[i] <= max_closest
    ]
    idx = np.full(A, 0, dtype=np.int64)
    mask = np.zeros(A, dtype=bool)
    for a, j in enumerate(order[:active_cap]):
        idx[a] = int(j)
        mask[a] = True
    return idx, mask


def _online_conflict_tokens(agent_state: np.ndarray, sdc_index: int, candidates: np.ndarray, cand_valid: np.ndarray, critical_idx: np.ndarray, critical_valid: np.ndarray, roadgraph: dict[str, np.ndarray], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    C = int(cfg.get("limits", {}).get("max_conflict_regions", 64))
    plan_cfg = cfg.get("planning", {})
    pair_cap = min(C, int(plan_cfg.get("max_online_pair_conflict_tokens", 24)))
    map_cap = min(max(C - pair_cap, 0), int(plan_cfg.get("max_online_map_tokens", 12)))
    tokens = np.zeros((C, 8), dtype=np.float32)
    valid = np.zeros(C, dtype=bool)
    rows: list[np.ndarray] = []
    ego = agent_state[sdc_index]
    active = candidates[cand_valid] if np.any(cand_valid) else candidates[:1]
    # Preserve macro/candidate diversity but avoid filling all 64 tokens with
    # near-duplicates.  Rank candidate-agent closest approaches globally.
    proposals: list[tuple[float, np.ndarray]] = []
    for a, j in enumerate(critical_idx):
        if not bool(critical_valid[a]):
            continue
        j = int(j)
        t = np.arange(1, active.shape[1] + 1, dtype=np.float32)[:, None] * float(cfg.get("time", {}).get("dt", 0.1))
        pred = agent_state[j, :2][None, :] + agent_state[j, 3:5][None, :] * t
        stride = max(1, len(active) // max(1, pair_cap // max(1, int(critical_valid.sum()))))
        for tr in active[::stride]:
            d = np.linalg.norm(tr[:, :2] - pred, axis=-1)
            m = int(np.argmin(d))
            center = 0.5 * (tr[m, :2] + pred[m])
            radius = max(4.0, 0.5 * (float(ego[7]) + float(agent_state[j, 7])) + 1.0)
            row = np.asarray([1.0, center[0], center[1], radius, float(d[m]), m / max(len(tr) - 1, 1), float(a), 1.0], dtype=np.float32)
            proposals.append((float(d[m]), row))
    for _, row in sorted(proposals, key=lambda x: x[0])[:pair_cap]:
        rows.append(row)

    xy = roadgraph.get("xy", np.zeros((0, 2), dtype=np.float32))
    lane_valid = _lane_centerline_mask(roadgraph)
    if len(xy) and np.any(lane_valid) and map_cap > 0:
        d = np.linalg.norm(xy - ego[:2][None], axis=-1)
        order = np.argsort(d)
        last_xy = None
        added = 0
        min_spacing = float(plan_cfg.get("online_map_token_spacing_m", 6.0))
        for q in order:
            if added >= map_cap or len(rows) >= C:
                break
            if not lane_valid[q] or d[q] > 60.0:
                continue
            if last_xy is not None and np.linalg.norm(xy[q] - last_xy) < min_spacing:
                continue
            rows.append(np.asarray([2.0, xy[q, 0], xy[q, 1], 4.0, d[q], 0.0, 0.0, 1.0], dtype=np.float32))
            last_xy = xy[q]
            added += 1
    for i, row in enumerate(rows[:C]):
        tokens[i] = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        valid[i] = True
    return tokens, valid



def _priority_claim_weights(
    agent_state: np.ndarray,
    sdc_index: int,
    candidates: np.ndarray,
    cand_valid: np.ndarray,
    critical_idx: np.ndarray,
    critical_valid: np.ndarray,
    macro: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Heuristic right-of-way / priority proxy for online P-NCF gating.

    It is intentionally conservative: nearby same-direction lead/adjacent agents
    and agents whose constant-velocity path is close to an ego candidate receive
    higher priority.  The witness gate is hard only for high-priority claims;
    lower-priority conflicts become a soft ranking penalty.
    """
    K = int(candidates.shape[0])
    A = int(critical_idx.shape[0])
    out = np.zeros((K, A), dtype=np.float32)
    if not (0 <= sdc_index < agent_state.shape[0]):
        return out
    ego = agent_state[sdc_index]
    ego_xy = ego[:2].astype(np.float32)
    ego_yaw = float(ego[6])
    ego_dir = np.asarray([np.cos(ego_yaw), np.sin(ego_yaw)], dtype=np.float32)
    ego_lat = np.asarray([-np.sin(ego_yaw), np.cos(ego_yaw)], dtype=np.float32)
    ego_speed = float(max(ego[5], np.linalg.norm(ego[3:5])))
    H = candidates.shape[1] if candidates.ndim >= 3 else int(cfg.get("time", {}).get("future_steps", 80))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    ts = (np.arange(H, dtype=np.float32) + 1.0)[:, None] * dt
    lane_change_macros = {
        int(MacroType.LANE_CHANGE_LEFT),
        int(MacroType.LANE_CHANGE_RIGHT),
        int(MacroType.MERGE_AHEAD),
        int(MacroType.ACCELERATE_CROSS),
    }
    macro_is_interactive = np.isin(macro.astype(np.int64, copy=False), list(lane_change_macros))
    for a, raw_j in enumerate(critical_idx):
        if not bool(critical_valid[a]):
            continue
        j = int(raw_j)
        if j < 0 or j >= agent_state.shape[0] or j == sdc_index:
            continue
        aj = agent_state[j]
        if aj.shape[0] < 11 or aj[10] <= 0.5:
            continue
        rel = aj[:2].astype(np.float32) - ego_xy
        longitudinal = float(np.dot(rel, ego_dir))
        lateral = abs(float(np.dot(rel, ego_lat)))
        dist = float(np.linalg.norm(rel))
        same_dir = float(np.cos(float(aj[6]) - ego_yaw)) > 0.4
        rel_speed_long = ego_speed - float(np.dot(aj[3:5], ego_dir))
        ttc = 99.0
        if longitudinal > 0.0 and rel_speed_long > 0.25:
            ttc = longitudinal / rel_speed_long
        base = 0.10
        base += 0.35 if (-6.0 <= longitudinal <= 45.0 and lateral <= 5.0 and same_dir) else 0.0
        base += 0.20 if dist <= 25.0 else 0.0
        base += 0.20 if ttc <= 5.0 else 0.0
        agent_pred = aj[:2][None, :] + aj[3:5][None, :] * ts
        for k in range(K):
            if not bool(cand_valid[k]):
                continue
            cand_xy = candidates[k, :, :2]
            finite = np.isfinite(cand_xy).all(axis=-1)
            if not finite.any():
                continue
            min_d = float(np.min(np.linalg.norm(cand_xy[finite] - agent_pred[finite], axis=-1)))
            risk = float(np.exp(-max(min_d, 0.0) / 9.0))
            w = base + 0.35 * risk + (0.10 if bool(macro_is_interactive[k]) else 0.0)
            out[k, a] = np.float32(np.clip(w, 0.0, 1.0))
    return out

def _candidate_pressure_prior_np(
    agent_state: np.ndarray,
    sdc_index: int,
    candidates: np.ndarray,
    cand_valid: np.ndarray,
    critical_idx: np.ndarray,
    critical_valid: np.ndarray,
    macro: np.ndarray,
    cfg: dict,
    other_future_trajs: np.ndarray | None = None,
) -> np.ndarray:
    """Mechanism-aware coercion pressure prior for online P-NCF selection.

    Vectorized over candidates.  The previous K x A Python loop was not the main
    theory issue, but it made COWP Waymax rollout too slow for diagnosis.
    """
    K = int(candidates.shape[0])
    out = np.zeros(K, dtype=np.float32)
    if K == 0 or not (0 <= int(sdc_index) < int(agent_state.shape[0])):
        return out
    pcfg = cfg.get("planning", {})
    H_full = int(candidates.shape[1]) if candidates.ndim >= 3 else int(cfg.get("time", {}).get("future_steps", 80))
    H = min(H_full, int(pcfg.get("online_pressure_prior_horizon_steps", H_full)))
    stride = max(1, int(pcfg.get("online_pressure_prior_stride", pcfg.get("online_rule_risk_stride", 2))))
    idx = np.arange(0, H, stride, dtype=np.int64)
    if idx.size == 0:
        idx = np.asarray([0], dtype=np.int64)
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    cand_xy = np.asarray(candidates[:, idx, :2], dtype=np.float32)
    finite_k = cand_valid.astype(bool) & np.isfinite(cand_xy).all(axis=(1, 2))
    pressure_macros = {
        int(MacroType.ACCELERATE_CROSS): 0.75,
        int(MacroType.MERGE_AHEAD): 0.90,
        int(MacroType.LANE_CHANGE_LEFT): 0.70,
        int(MacroType.LANE_CHANGE_RIGHT): 0.70,
    }
    relief_macros = {
        int(MacroType.YIELD): -0.25,
        int(MacroType.STOP_BEFORE_CONFLICT): -0.35,
        int(MacroType.NEUTRAL_EGO): -0.20,
        int(MacroType.CREEP): -0.10,
        int(MacroType.MERGE_BEHIND): -0.15,
    }
    macro_bias = np.zeros(K, dtype=np.float32)
    for k in range(K):
        mk = int(macro[k]) if k < len(macro) else int(MacroType.KEEP_LANE)
        macro_bias[k] = float(pressure_macros.get(mk, 0.05) + relief_macros.get(mk, 0.0))
    worst = np.maximum(macro_bias, 0.0).astype(np.float32)
    ego = agent_state[int(sdc_index)]
    ego_xy = ego[:2].astype(np.float32)
    ego_yaw = float(ego[6])
    ego_dir = np.asarray([np.cos(ego_yaw), np.sin(ego_yaw)], dtype=np.float32)
    ego_lat = np.asarray([-np.sin(ego_yaw), np.cos(ego_yaw)], dtype=np.float32)
    ego_speed = float(max(ego[5], np.linalg.norm(ego[3:5]), 0.0))
    max_agents = int(pcfg.get("online_pressure_prior_max_agents", 8))
    active: list[tuple[int, float]] = []
    for a, raw_j in enumerate(critical_idx):
        if not bool(critical_valid[a]):
            continue
        j = int(raw_j)
        if j < 0 or j >= agent_state.shape[0] or j == sdc_index or agent_state[j, 10] <= 0.5:
            continue
        active.append((j, float(np.linalg.norm(agent_state[j, :2].astype(np.float32) - ego_xy))))
    active.sort(key=lambda x: x[1])
    if max_agents > 0:
        active = active[:max_agents]
    for j, _dist in active:
        aj = agent_state[j]
        pred, _cv = _agent_future_xy(agent_state, j, H_full, dt, other_future_trajs)
        pred = np.asarray(pred[idx, :2], dtype=np.float32)
        tmask = np.isfinite(pred).all(axis=-1)
        if not bool(tmask.any()):
            continue
        d = np.full(K, 99.0, dtype=np.float32)
        diff = cand_xy[:, tmask, :] - pred[None, tmask, :]
        d[finite_k] = np.sqrt(np.min(np.sum(diff[finite_k] * diff[finite_k], axis=-1), axis=-1))
        close = np.exp(-np.maximum(d, 0.0) / 6.5)
        rel = aj[:2].astype(np.float32) - ego_xy
        longitudinal = float(np.dot(rel, ego_dir))
        lateral = abs(float(np.dot(rel, ego_lat)))
        rel_speed = ego_speed - float(np.dot(aj[3:5], ego_dir))
        ttc = longitudinal / max(rel_speed, 1e-3) if longitudinal > 0.0 and rel_speed > 0.25 else 99.0
        priority = 0.0
        priority += 0.35 if -6.0 <= longitudinal <= 45.0 and lateral <= 5.5 else 0.0
        priority += 0.25 if ttc <= 5.0 else 0.0
        priority += 0.20 if (bool(finite_k.any()) and float(np.min(d[finite_k])) <= 6.0) else 0.0
        worst = np.maximum(worst, macro_bias + 0.55 * close.astype(np.float32) + float(priority))
    out[finite_k] = np.clip(worst[finite_k], 0.0, 1.0)
    return out



def _candidate_rule_risk_np(
    agent_state: np.ndarray,
    sdc_index: int,
    candidates: np.ndarray,
    cand_valid: np.ndarray,
    conventional_safe: np.ndarray,
    critical_idx: np.ndarray,
    critical_valid: np.ndarray,
    cfg: dict,
    other_future_trajs: np.ndarray | None = None,
) -> np.ndarray:
    """Fast counterfactual rule risk for online candidate selection.

    This is intentionally low-capacity: it should calibrate obvious logged-future
    false-safety without becoming a hand-coded closed-loop planner.  The
    implementation is vectorized over candidates and clips the sigmoid argument
    to avoid log-spam from exp overflow.
    """
    K = int(candidates.shape[0])
    out = np.ones(K, dtype=np.float32)
    if K == 0 or not (0 <= int(sdc_index) < int(agent_state.shape[0])):
        return out
    pcfg = cfg.get("planning", {})
    H_full = int(candidates.shape[1]) if candidates.ndim >= 3 else int(cfg.get("time", {}).get("future_steps", 80))
    H = min(H_full, int(pcfg.get("online_rule_risk_horizon_steps", pcfg.get("online_collision_check_horizon_steps", H_full))))
    stride = max(1, int(pcfg.get("online_rule_risk_stride", pcfg.get("online_collision_check_stride", 2))))
    idx = np.arange(0, H, stride, dtype=np.int64)
    if idx.size == 0:
        idx = np.asarray([0], dtype=np.int64)
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    ego = agent_state[int(sdc_index)]
    ego_radius = max(float(ego[7]), float(ego[8]), 4.0) * 0.5
    ego_xy = ego[:2].astype(np.float32)
    ego_yaw = float(ego[6])
    ego_dir = np.asarray([np.cos(ego_yaw), np.sin(ego_yaw)], dtype=np.float32)
    ego_lat = np.asarray([-ego_dir[1], ego_dir[0]], dtype=np.float32)
    ego_speed = float(max(ego[5], np.linalg.norm(ego[3:5]), 0.0))
    valid_agents = agent_state[:, 10] > 0.5
    active_agents: list[tuple[int, float]] = []
    for raw_j, ok in zip(critical_idx, critical_valid):
        if not bool(ok):
            continue
        j = int(raw_j)
        if j != sdc_index and 0 <= j < agent_state.shape[0] and bool(valid_agents[j]):
            active_agents.append((j, float(np.linalg.norm(agent_state[j, :2].astype(np.float32) - ego_xy))))
    if not active_agents:
        near = np.argsort(np.linalg.norm(agent_state[:, :2] - ego_xy[None, :], axis=-1))[:8]
        active_agents = [(int(j), float(np.linalg.norm(agent_state[int(j), :2] - ego_xy))) for j in near if int(j) != int(sdc_index) and bool(valid_agents[int(j)])]
    active_agents.sort(key=lambda x: x[1])
    max_agents = int(pcfg.get("online_rule_risk_max_agents", 8))
    if max_agents > 0:
        active_agents = active_agents[:max_agents]

    cand_xy = np.asarray(candidates[:, idx, :2], dtype=np.float32)
    finite_k = cand_valid.astype(bool) & np.isfinite(cand_xy).all(axis=(1, 2))
    worst = np.where(conventional_safe.astype(bool), 0.0, 0.25).astype(np.float32)
    for j, dist0 in active_agents:
        if dist0 > float(pcfg.get("online_rule_risk_agent_radius_m", 60.0)):
            continue
        aj = agent_state[j]
        rel = aj[:2].astype(np.float32) - ego_xy
        other_radius = max(float(aj[7]), float(aj[8]), 4.0) * 0.5
        radius = ego_radius + other_radius + 0.5
        logged, cv = _agent_future_xy(agent_state, j, H_full, dt, other_future_trajs)
        logged_xy = np.asarray(logged[idx, :2], dtype=np.float32)
        cv_xy = np.asarray(cv[idx, :2], dtype=np.float32)
        logged_mask = np.isfinite(logged_xy).all(axis=-1)
        cv_mask = np.isfinite(cv_xy).all(axis=-1)
        d_logged = np.full(K, 99.0, dtype=np.float32)
        d_cv = np.full(K, 99.0, dtype=np.float32)
        if bool(logged_mask.any()):
            diff = cand_xy[:, logged_mask, :] - logged_xy[None, logged_mask, :]
            d_logged[finite_k] = np.sqrt(np.min(np.sum(diff[finite_k] * diff[finite_k], axis=-1), axis=-1))
        if bool(cv_mask.any()):
            diff = cand_xy[:, cv_mask, :] - cv_xy[None, cv_mask, :]
            d_cv[finite_k] = np.sqrt(np.min(np.sum(diff[finite_k] * diff[finite_k], axis=-1), axis=-1))
        longitudinal = float(np.dot(rel, ego_dir))
        lateral = abs(float(np.dot(rel, ego_lat)))
        rel_speed = ego_speed - float(np.dot(aj[3:5], ego_dir))
        ttc = longitudinal / max(rel_speed, 1e-3) if longitudinal > 0.0 and rel_speed > 0.25 else 99.0
        priority = 1.0 if (-8.0 <= longitudinal <= 55.0 and lateral <= 7.5) or ttc <= 5.0 else 0.0
        clearance = np.minimum(d_logged, d_cv if priority > 0.0 else np.maximum(d_cv, d_logged)) - float(radius)
        close_risk = _stable_logistic_np((clearance - 1.0) / 1.5).astype(np.float32)
        false_safe_risk = np.minimum(1.0, np.maximum(0.0, d_logged - d_cv) / 8.0).astype(np.float32) * float(priority)
        worst = np.maximum(worst, 0.70 * close_risk + 0.30 * false_safe_risk)
    out[finite_k] = np.clip(worst[finite_k], 0.0, 1.0)
    return out



def _consistent_one_step_targets_np(
    current: np.ndarray,
    desired: np.ndarray,
    cfg: dict,
    previous_longitudinal_accel: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized counterpart of ``_consistent_one_step_target`` for all K candidates.

    Returns dynamically consistent emitted targets, clipped accelerations, and a
    projection-risk score measuring how strongly each raw candidate had to be
    corrected before it could be issued to Waymax.
    """
    desired = np.asarray(desired, dtype=np.float64)
    k = int(desired.shape[0])
    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1e-6)
    cand_cfg = cfg.get("candidate", {})
    wm_cfg = cfg.get("waymax", {})
    cur_xy = np.asarray(current[:2], dtype=np.float64)
    cur_yaw = float(current[6])
    cur_vel = np.asarray(current[3:5], dtype=np.float64)
    cur_speed = float(max(np.linalg.norm(cur_vel), float(current[5]) if current.shape[0] > 5 else 0.0, 0.0))
    desired_vel = desired[:, 3:5] if desired.shape[1] >= 5 else np.zeros((k, 2), dtype=np.float64)
    desired_speed = np.linalg.norm(desired_vel, axis=-1)
    position_speed = np.linalg.norm(desired[:, :2] - cur_xy[None, :], axis=-1) / dt
    desired_speed = np.where(desired_speed < 1e-3, position_speed, desired_speed)
    raw_accel_unclipped = (desired_speed - cur_speed) / dt
    max_accel = max(float(cand_cfg.get("max_accel_mps2", 4.0)), 1e-3)
    max_decel = max(float(cand_cfg.get("max_decel_mps2", 6.0)), 1e-3)
    max_jerk = max(float(cand_cfg.get("max_jerk_mps3", 8.0)), 1e-3)
    raw_accel = np.clip(raw_accel_unclipped, -max_decel, max_accel)
    accel = np.clip(
        raw_accel,
        float(previous_longitudinal_accel) - max_jerk * dt,
        float(previous_longitudinal_accel) + max_jerk * dt,
    )
    next_speed = np.maximum(0.0, cur_speed + accel * dt)
    desired_yaw_from_vel = np.arctan2(desired_vel[:, 1], desired_vel[:, 0])
    desired_yaw = np.where(desired_speed > 0.25, desired_yaw_from_vel, desired[:, 2] if desired.shape[1] > 2 else cur_yaw)
    max_yaw_rate = max(float(cand_cfg.get("max_yaw_rate_rad_s", 1.2)), 1e-3)
    max_dyaw = min(float(wm_cfg.get("max_delta_yaw_rad", 0.12)), max_yaw_rate * dt)
    requested_dyaw = np.asarray(_wrap_angle(desired_yaw - cur_yaw), dtype=np.float64)
    dyaw = np.clip(requested_dyaw, -max_dyaw, max_dyaw)
    next_yaw = np.asarray(_wrap_angle(cur_yaw + dyaw), dtype=np.float64)
    v0 = cur_speed * np.asarray([np.cos(cur_yaw), np.sin(cur_yaw)], dtype=np.float64)
    v1 = next_speed[:, None] * np.stack([np.cos(next_yaw), np.sin(next_yaw)], axis=-1)
    next_xy = cur_xy[None, :] + 0.5 * (v0[None, :] + v1) * dt
    target = np.concatenate([next_xy, next_yaw[:, None], v1], axis=-1).astype(np.float32)

    accel_clip = np.abs(raw_accel_unclipped - accel) / max(max_accel, max_decel)
    yaw_clip = np.abs(requested_dyaw - dyaw) / max(max_dyaw, 1e-6)
    desired_xy = desired[:, :2]
    projection_error = np.linalg.norm(desired_xy - next_xy, axis=-1) / max(float(cfg.get("planning", {}).get("action_projection_error_scale_m", 1.5)), 1e-3)
    projection_risk = np.clip(0.40 * accel_clip + 0.30 * yaw_clip + 0.30 * projection_error, 0.0, 1.0).astype(np.float32)
    return target, accel.astype(np.float32), projection_risk


def _one_step_action_risk_np(
    agent_state: np.ndarray,
    sdc_index: int,
    candidates: np.ndarray,
    cand_valid: np.ndarray,
    cfg: dict,
    previous_longitudinal_accel: float = 0.0,
    *,
    return_targets: bool = False,
):
    """Vectorized candidate/action consistency risk.

    The v6 implementation executed nested Python loops over K candidates and the
    short horizon, then recomputed the selected candidate's one-step controller.
    This version performs the same horizon acceleration/jerk/yaw checks with
    NumPy broadcasting, adds the exact emitted-action projection risk, and can
    return all precomputed action targets for reuse by the selected candidate.
    """
    k = int(candidates.shape[0])
    out = np.ones(k, dtype=np.float32)
    empty_targets = np.zeros((k, 5), dtype=np.float32)
    empty_accel = np.zeros(k, dtype=np.float32)
    if not (0 <= int(sdc_index) < int(agent_state.shape[0])) or candidates.ndim < 3:
        return (out, empty_targets, empty_accel) if return_targets else out
    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1e-3)
    cand_cfg = cfg.get("candidate", {})
    pcfg = cfg.get("planning", {})
    cur = np.asarray(agent_state[int(sdc_index)], dtype=np.float64)
    cur_speed = float(max(np.linalg.norm(cur[3:5]), cur[5] if cur.shape[0] > 5 else 0.0, 0.0))
    cur_yaw = float(cur[6])
    max_accel = max(float(cand_cfg.get("max_accel_mps2", 4.0)), 1e-3)
    max_decel = max(float(cand_cfg.get("max_decel_mps2", 6.0)), 1e-3)
    max_jerk = max(float(cand_cfg.get("max_jerk_mps3", 8.0)), 1e-3)
    max_yaw_rate = max(float(cand_cfg.get("max_yaw_rate_rad_s", 1.2)), 1e-3)
    horizon = max(1, min(int(pcfg.get("online_action_risk_horizon_steps", 8)), int(candidates.shape[1])))
    traj = np.asarray(candidates[:, :horizon], dtype=np.float64)
    vel = traj[..., 3:5]
    speed = np.linalg.norm(vel, axis=-1)
    prev_xy = np.concatenate([
        np.broadcast_to(cur[:2], (k, 1, 2)), traj[:, :-1, :2]
    ], axis=1)
    pos_speed = np.linalg.norm(traj[..., :2] - prev_xy, axis=-1) / dt
    speed = np.where(speed < 1e-3, pos_speed, speed)
    prev_speed = np.concatenate([
        np.full((k, 1), cur_speed, dtype=np.float64), speed[:, :-1]
    ], axis=1)
    raw_accel = (speed - prev_speed) / dt
    prev_accel = np.concatenate([
        np.full((k, 1), float(previous_longitudinal_accel), dtype=np.float64), raw_accel[:, :-1]
    ], axis=1)
    jerk_need = np.abs(raw_accel - prev_accel) / (max_jerk * dt)
    accel_excess = np.maximum(0.0, raw_accel - max_accel) / max_accel + np.maximum(0.0, -raw_accel - max_decel) / max_decel
    yaw_from_vel = np.arctan2(vel[..., 1], vel[..., 0])
    desired_yaw = np.where(speed > 0.25, yaw_from_vel, traj[..., 2])
    prev_yaw = np.concatenate([
        np.full((k, 1), cur_yaw, dtype=np.float64), desired_yaw[:, :-1]
    ], axis=1)
    yaw_need = np.abs(np.asarray(_wrap_angle(desired_yaw - prev_yaw), dtype=np.float64)) / (max_yaw_rate * dt)
    decay = 1.0 / (1.0 + 0.25 * np.arange(horizon, dtype=np.float64))[None, :]
    step_risk = decay * (
        0.42 * np.maximum(0.0, jerk_need - 1.0)
        + 0.34 * np.maximum(0.0, yaw_need - 1.0)
        + 0.24 * accel_excess
    )
    horizon_risk = np.max(step_risk, axis=1)
    targets, clipped_accel, projection_risk = _consistent_one_step_targets_np(
        cur, traj[:, 0], cfg, previous_longitudinal_accel
    )
    projection_mix = float(pcfg.get("candidate_action_projection_risk_mix", 0.65))
    out = np.clip((1.0 - projection_mix) * horizon_risk + projection_mix * projection_risk, 0.0, 1.0).astype(np.float32)
    out[~np.asarray(cand_valid, dtype=bool)] = 1.0
    if return_targets:
        return out, targets, clipped_accel
    return out

def build_online_batch(
    agent_state: np.ndarray,
    sdc_index: int,
    cfg: dict,
    *,
    history_model_state: np.ndarray | None = None,
    roadgraph: dict[str, np.ndarray] | None = None,
    other_future_trajs: np.ndarray | None = None,
    compute_rule_risk: bool = True,
    include_training_targets: bool = False,
) -> dict[str, Any]:
    K = int(cfg.get("limits", {}).get("max_candidates", 64))
    A = int(cfg.get("limits", {}).get("max_critical_agents", 8))
    M = int(cfg.get("limits", {}).get("max_natural_alternatives", 24))
    R = int(cfg.get("limits", {}).get("max_safe_responses", 32))
    H = int(cfg.get("time", {}).get("future_steps", cfg.get("eval", {}).get("rollout_horizon_steps", 80)))
    d_state = int(cfg.get("model", cfg).get("d_state", 11))
    max_agents = int(cfg.get("limits", {}).get("max_agents", cfg.get("model", cfg).get("max_agents", 128)))
    if roadgraph is None:
        roadgraph = {"xy": np.zeros((0, 2), dtype=np.float32), "heading": np.zeros(0, dtype=np.float32), "valid": np.zeros(0, dtype=bool), "types": np.zeros(0, dtype=np.int32)}
    if history_model_state is None:
        hist = np.zeros((max_agents, 1, d_state), dtype=np.float32)
        n = min(max_agents, agent_state.shape[0])
        hist[:n, 0, 0:3] = agent_state[:n, 0:3]
        hist[:n, 0, 3:6] = agent_state[:n, 7:10]
        hist[:n, 0, 6] = agent_state[:n, 6]
        hist[:n, 0, 7:9] = agent_state[:n, 3:5]
        hist[:n, 0, 9] = agent_state[:n, 5]
        hist[:n, 0, 10] = agent_state[:n, 10]
    else:
        hist = history_model_state.astype(np.float32, copy=False)
        n = min(max_agents, agent_state.shape[0])
    agent_mask = np.zeros(max_agents, dtype=bool)
    agent_mask[:n] = agent_state[:n, 10] > 0.5
    if 0 <= sdc_index < max_agents:
        agent_mask[sdc_index] = True

    (
        cand_traj, cand_valid, conventional_safe, macro, utility,
        roadgraph_safe, collision_safe, collision_safe_prefix_steps, collision_min_clearance_margin_m,
    ) = _route_lane_aware_candidates(
        agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs
    )
    crit_idx, crit_valid = _critical_interaction_rank(agent_state, sdc_index, cand_traj, cand_valid, cfg)
    if compute_rule_risk:
        rule_risk = _candidate_rule_risk_np(
            agent_state, sdc_index, cand_traj, cand_valid, conventional_safe, crit_idx, crit_valid, cfg,
            other_future_trajs=other_future_trajs,
        )
    else:
        rule_risk = np.zeros(K, dtype=np.float32)
    conflict, conflict_valid = _online_conflict_tokens(agent_state, sdc_index, cand_traj, cand_valid, crit_idx, crit_valid, roadgraph, cfg)
    batch = {
        "state/history": hist[None],
        "state/agent_valid": agent_mask[None],
        "state/is_sdc": np.eye(max_agents, dtype=bool)[sdc_index][None] if 0 <= sdc_index < max_agents else np.zeros((1, max_agents), dtype=bool),
        "cowp/candidates/trajectory": cand_traj[None],
        "cowp/candidates/valid": cand_valid[None],
        "cowp/candidates/macro_type": macro[None],
        "cowp/candidates/ego_utility_prior": utility[None],
        "cowp/candidates/conventional_safe": conventional_safe[None],
        "cowp/candidates/roadgraph_safe": roadgraph_safe[None],
        "cowp/candidates/collision_safe": collision_safe[None],
        "cowp/candidates/collision_safe_prefix_steps": collision_safe_prefix_steps[None],
        "cowp/candidates/collision_min_clearance_margin_m": collision_min_clearance_margin_m[None],
        "cowp/candidates/rule_risk": rule_risk[None],
        "cowp/critical/track_index": crit_idx[None],
        "cowp/critical/input_index": crit_idx[None],
        "cowp/critical/valid": crit_valid[None],
        "map/conflict_regions": conflict[None],
        "map/conflict_region_valid": conflict_valid[None],
    }
    batch.update(_roadgraph_womd_batch_fields(roadgraph))
    if include_training_targets:
        # Compatibility/debug-only targets.  They are intentionally omitted in the
        # default online path because the model inference keys above do not consume
        # them and allocating them at every Waymax step adds avoidable CPU work.
        batch.update({
            "cowp/natural/traj": np.zeros((1, A, M, H, 7), dtype=np.float32),
            "cowp/natural/valid": np.zeros((1, A, M), dtype=bool),
            "cowp/natural/weight": np.zeros((1, A, M), dtype=np.float32),
            "cowp/natural/source": np.zeros((1, A, M), dtype=np.int64),
            "cowp/natural/priority_preserved": np.zeros((1, A, M), dtype=bool),
            "cowp/response/valid": np.zeros((1, K, A, R), dtype=bool),
            "cowp/response/is_safe": np.zeros((1, K, A, R), dtype=bool),
            "cowp/response/is_low_burden": np.zeros((1, K, A, R), dtype=bool),
            "cowp/response/burden_total": np.zeros((1, K, A, R), dtype=np.float32),
            "cowp/response/burden_components": np.zeros((1, K, A, R, 6), dtype=np.float32),
            "cowp/witness/exists": np.zeros((1, K, A), dtype=bool),
            "cowp/witness/token": np.zeros((1, K, A), dtype=np.int64),
            "cowp/witness/burden_total": np.zeros((1, K, A), dtype=np.float32),
            "cowp/witness/burden_components": np.zeros((1, K, A, 6), dtype=np.float32),
            "cowp/witness/opr": np.ones((1, K, A), dtype=np.float32),
            "cowp/witness/c_i": np.zeros((1, K, A), dtype=np.float32),
            "cowp/witness/conflict_interval": np.zeros((1, K, A, 2), dtype=np.int64),
        })
    return batch


def _consistent_one_step_target(
    current: np.ndarray,
    desired: np.ndarray,
    cfg: dict,
    previous_longitudinal_accel: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Convert a trajectory waypoint into a dynamically consistent 10 Hz state.

    Direct-state Waymax actions must not independently command position, yaw and
    velocity that contradict one another.  This controller integrates a jerk-
    limited longitudinal acceleration and yaw-rate-limited heading for one step.
    """
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    cand_cfg = cfg.get("candidate", {})
    wm_cfg = cfg.get("waymax", {})
    cur_xy = np.asarray(current[:2], dtype=np.float64)
    cur_yaw = float(current[6])
    cur_vel = np.asarray(current[3:5], dtype=np.float64)
    cur_speed = float(max(np.linalg.norm(cur_vel), float(current[5]) if current.shape[0] > 5 else 0.0, 0.0))
    desired_vel = np.asarray(desired[3:5], dtype=np.float64) if desired.shape[0] >= 5 else np.zeros(2)
    desired_speed = float(np.linalg.norm(desired_vel))
    if desired_speed < 1e-3:
        desired_speed = float(np.linalg.norm(np.asarray(desired[:2], dtype=np.float64) - cur_xy) / max(dt, 1e-6))
    raw_accel = (desired_speed - cur_speed) / max(dt, 1e-6)
    max_accel = float(cand_cfg.get("max_accel_mps2", 4.0))
    max_decel = float(cand_cfg.get("max_decel_mps2", 6.0))
    max_jerk = float(cand_cfg.get("max_jerk_mps3", 8.0))
    raw_accel = float(np.clip(raw_accel, -max_decel, max_accel))
    accel = float(np.clip(raw_accel, previous_longitudinal_accel - max_jerk * dt, previous_longitudinal_accel + max_jerk * dt))
    next_speed = float(max(0.0, cur_speed + accel * dt))

    if desired_speed > 0.25:
        desired_yaw = float(np.arctan2(desired_vel[1], desired_vel[0]))
    else:
        desired_yaw = float(desired[2]) if desired.shape[0] > 2 else cur_yaw
    max_yaw_rate = float(cand_cfg.get("max_yaw_rate_rad_s", 1.2))
    max_dyaw = min(float(wm_cfg.get("max_delta_yaw_rad", 0.12)), max_yaw_rate * dt)
    dyaw = float(np.clip(_wrap_angle(desired_yaw - cur_yaw), -max_dyaw, max_dyaw))
    next_yaw = float(_wrap_angle(cur_yaw + dyaw))

    # Trapezoidal integration makes displacement and reported velocity mutually
    # consistent, reducing Waymax kinematic-infeasibility flags.
    v0 = cur_speed * np.asarray([np.cos(cur_yaw), np.sin(cur_yaw)], dtype=np.float64)
    v1 = next_speed * np.asarray([np.cos(next_yaw), np.sin(next_yaw)], dtype=np.float64)
    next_xy = cur_xy + 0.5 * (v0 + v1) * dt
    target = np.asarray([next_xy[0], next_xy[1], next_yaw, v1[0], v1[1]], dtype=np.float32)
    return target, accel



def _topk_frontier_mask_1d(base_mask, risk, tie_breaker=None, *, keep_frac=0.40, keep_min=1, keep_max=4, eps=1.0e-3):
    """Exact top-k frontier for one online scene.

    A quantile threshold keeps every tied candidate when certificate risk is
    flat, so COWP becomes conventional_safety.  Exact top-k preserves the
    intended set-valued feasibility layer.
    """
    import torch

    base = base_mask.bool()
    frontier = torch.zeros_like(base)
    if not bool(base.any().detach().cpu().item()):
        return frontier
    idx = torch.where(base)[0]
    n = int(idx.numel())
    k = max(int(keep_min), int(torch.ceil(torch.tensor(float(n) * float(keep_frac), device=risk.device)).item()))
    k = min(max(k, 1), n, max(int(keep_max), 1))
    r = torch.nan_to_num(risk[idx].float(), nan=1.0, posinf=1.0, neginf=0.0)
    if tie_breaker is None:
        tb = torch.arange(n, device=risk.device, dtype=r.dtype) / max(float(n), 1.0)
    else:
        tb = torch.nan_to_num(tie_breaker[idx].float(), nan=0.0, posinf=1.0, neginf=0.0)
        if tb.numel() > 1:
            lo = tb.min(); hi = tb.max(); span = (hi - lo).clamp_min(1.0e-6)
            tb = ((tb - lo) / span).clamp(0.0, 1.0)
        else:
            tb = torch.zeros_like(tb)
    order = torch.argsort(r + float(eps) * tb, stable=True)
    frontier[idx[order[:k]]] = True
    return frontier

def _guard_frontier_base_1d(base, score_risk, progress, action_risk=None, *, keep_min=2, pcfg=None):
    import torch
    pcfg = pcfg or {}
    base = base.bool()
    if not bool(base.any().detach().cpu().item()):
        return base
    idx = torch.where(base)[0]
    need = min(max(int(keep_min), 1), int(idx.numel()))
    guarded = base.clone()
    if torch.is_tensor(progress) and progress.numel() == base.numel():
        vals = torch.nan_to_num(progress.float(), nan=0.0, posinf=0.0, neginf=0.0)
        p_ref = vals[idx].max() if idx.numel() else vals.new_tensor(0.0)
        min_abs = float(pcfg.get("candidate_frontier_min_progress_m", 1.0))
        ratio = float(pcfg.get("candidate_frontier_min_progress_ratio", 0.12))
        if float(p_ref.detach().cpu().item()) > min_abs:
            pg = base & (vals >= max(min_abs, ratio * float(p_ref.detach().cpu().item())))
            if int(pg.sum().detach().cpu().item()) >= need:
                guarded = pg
    if torch.is_tensor(score_risk) and score_risk.numel() == base.numel():
        sr = torch.nan_to_num(score_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)
        best = sr[idx].min() if idx.numel() else sr.new_tensor(0.0)
        sg = base & (sr <= best + float(pcfg.get("candidate_frontier_score_slack", 0.85)))
        joint = guarded & sg
        if int(joint.sum().detach().cpu().item()) >= need:
            guarded = joint
    if torch.is_tensor(action_risk) and action_risk.numel() == base.numel():
        ar = torch.nan_to_num(action_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)
        ag = guarded & (ar <= float(pcfg.get("candidate_frontier_max_action_risk", 0.90)))
        if int(ag.sum().detach().cpu().item()) >= need:
            guarded = ag
    return guarded



def _risk_budgeted_selection_scores_1d(
    scores,
    base_mask,
    frontier_mask,
    noncoercive_risk,
    score_risk,
    progress_shortfall,
    action_risk=None,
    rule_risk=None,
    outcome_risk=None,
    *,
    pcfg=None,
):
    """Online counterpart of the utility-regret bounded P-NCF selector.

    COWP should not be a raw-score planner after the frontier is built.  The
    final choice is primarily the least-coercive candidate inside a progress- and
    utility-guarded frontier; score, progress, action and outcome terms only break
    ties or shield obviously unstable actions.
    """
    import torch

    pcfg = pcfg or {}
    inf = torch.full_like(scores, float("inf"))
    select = frontier_mask.bool()
    if not bool(select.any().detach().cpu().item()):
        select = base_mask.bool()
    if not bool(select.any().detach().cpu().item()):
        return torch.where(base_mask.bool(), scores, inf)
    nr = torch.nan_to_num(noncoercive_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)
    sr = torch.nan_to_num(score_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)
    ps = torch.nan_to_num(progress_shortfall.float(), nan=1.0, posinf=1.0, neginf=0.0)
    ar = torch.zeros_like(nr) if action_risk is None else torch.nan_to_num(action_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)
    rr = torch.zeros_like(nr) if rule_risk is None else torch.nan_to_num(rule_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)
    orr = torch.zeros_like(nr) if outcome_risk is None else torch.nan_to_num(outcome_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)

    budget = float(pcfg.get("candidate_selection_risk_budget", 0.18))
    min_keep = max(int(pcfg.get("candidate_selection_min_keep", 1)), 1)
    vals = nr[select]
    low = vals.min() if vals.numel() else nr.new_tensor(0.0)
    budget_mask = select & (nr <= low + budget)
    if int(budget_mask.sum().detach().cpu().item()) >= min_keep:
        select = budget_mask

    obj = (
        nr
        + float(pcfg.get("candidate_selection_score_weight", 0.18)) * sr
        + float(pcfg.get("candidate_selection_progress_weight", 0.10)) * ps
        + float(pcfg.get("candidate_selection_action_weight", 1.15)) * ar
        + float(pcfg.get("candidate_selection_rule_weight", 0.25)) * rr
        + float(pcfg.get("candidate_selection_outcome_weight", 0.45)) * orr
    )
    return torch.where(select, obj, inf)

def _plan_continuity_risk_np(
    candidates: np.ndarray,
    cand_valid: np.ndarray,
    previous_traj: np.ndarray | None,
    cfg: dict,
) -> np.ndarray:
    """Distance to the previous plan shifted by one simulation step.

    This is a selection regularizer, not a safety certificate.  It suppresses
    one-step candidate oscillation while preserving fresh replanning and all hard
    COWP/physical gates.
    """
    K = int(candidates.shape[0])
    out = np.zeros(K, dtype=np.float32)
    if previous_traj is None or candidates.ndim < 3 or previous_traj.ndim < 2:
        return out
    pcfg = cfg.get("planning", {})
    h = min(
        int(pcfg.get("online_plan_continuity_horizon_steps", 12)),
        int(candidates.shape[1]),
        max(int(previous_traj.shape[0]) - 1, 0),
    )
    if h <= 0:
        return out
    ref = np.asarray(previous_traj[1 : 1 + h, :2], dtype=np.float32)
    cur = np.asarray(candidates[:, :h, :2], dtype=np.float32)
    finite = cand_valid.astype(bool) & np.isfinite(cur).all(axis=(1, 2)) & np.isfinite(ref).all()
    if not finite.any():
        return out
    d = np.linalg.norm(cur[finite] - ref[None, :, :], axis=-1).mean(axis=-1)
    scale = max(float(pcfg.get("online_plan_continuity_scale_m", 3.0)), 1.0e-3)
    out[finite] = np.clip(d / scale, 0.0, 1.0).astype(np.float32)
    return out


def _resolve_execution_trajectory(
    candidate_trajectories: np.ndarray,
    selected: int,
    has_valid_candidate: bool,
    current_ego_state: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, bool, str]:
    """Resolve the trajectory that is actually sent to Waymax.

    Candidate tensors are zero padded.  A no-valid-candidate state therefore must
    never execute a padded slot as if it were a real trajectory.  In that state we
    synthesize a bounded smooth-stop trajectory from the *current ego state*.  This
    is an execution-integrity fallback, not a selectable proposal and not a safety
    certificate.
    """
    cands = np.asarray(candidate_trajectories, dtype=np.float32)
    if bool(has_valid_candidate):
        return np.asarray(cands[int(selected)], dtype=np.float32), False, "candidate"

    if cands.ndim != 3 or cands.shape[1] <= 0:
        raise RuntimeError(f"Invalid candidate trajectory tensor shape for emergency execution: {cands.shape!r}")
    horizon = int(cands.shape[1])
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    decel = float(cfg.get("planning", {}).get("fallback_decel_mps2", -2.0))
    emergency = smooth_stop_trajectory(
        np.asarray(current_ego_state, dtype=np.float32), horizon, dt, decel=decel
    )
    if emergency.shape != cands.shape[1:] or not np.isfinite(emergency).all():
        raise RuntimeError(
            "Emergency no-valid execution trajectory is malformed; refusing to execute padding. "
            f"shape={emergency.shape!r}, expected={cands.shape[1:]!r}"
        )
    return np.asarray(emergency, dtype=np.float32), True, "bounded_smooth_stop"


@dataclass
class COWPWaymaxPolicy:
    checkpoint: str
    cfg: dict
    device: str = "auto"
    witness_threshold: float = 0.5
    bcot_risk_budget: float | None = None
    action_mode: str = "absolute_xy_yaw"
    ncf_gate_mode: str = "hard"
    priority_hard_threshold: float = 0.55
    secondary_witness_threshold: float = 0.85
    secondary_opr_alpha: float = 0.10
    soft_ncf_penalty: float = 1.5
    method: str = "cowp"
    adaptive_frontier_margin: float = 0.20
    outcome_risk_penalty: float = 0.0
    outcome_risk_threshold: float = 1.10
    profile_policy_runtime: bool = False
    profile_policy_runtime_sync: bool = False

    def __post_init__(self) -> None:
        import torch

        from cowp.models.cowp_model import COWPModel

        dev = torch.device("cuda" if self.device == "auto" and torch.cuda.is_available() else ("cpu" if self.device == "auto" else self.device))
        ckpt = torch.load(self.checkpoint, map_location="cpu")
        model_cfg = ckpt.get("cfg", self.cfg)
        self.model = COWPModel(model_cfg)
        _load_state_dict_compatible(self.model, ckpt["model"])
        del ckpt
        self.model.to(dev)
        self.model.eval()
        self.torch = torch
        self.dev = dev
        if dev.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        self._last_diagnostics: dict[str, Any] | None = None
        self._diagnostics_log: list[dict[str, Any]] = []
        self._previous_longitudinal_accel: float = 0.0
        self._previous_scenario_index: int | None = None
        self._previous_selected_traj: np.ndarray | None = None
        # v16.8.32 diagnostic state.
        self._recovery_commitment_active: bool = False
        # v16.8.33 parameter-free recovery-mode hysteresis state.  The two new
        # methods share state semantics but use different successor viability
        # relations (legacy SOV signature vs recovery-option persistence profile).
        self._recovery_hysteresis_active: bool = False
        # v16.8.36: semantic recovery mode for the control-projected recovery frontier.
        self._recovery_frontier_macro: int = -1
        # v16.8.37: at most one pending real-replan bridge action.  The
        # witnessed semantic recourse set is persisted so the actual bridge
        # cannot substitute an unrelated macro after observing the next state.
        self._recovery_bridge_pending: bool = False
        self._recovery_bridge_allowed_macros: frozenset[int] = frozenset()
        self._cached_roadgraph_scenario_index: int | None = None
        self._cached_roadgraph: dict[str, np.ndarray] | None = None
        self._cached_sdc_scenario_index: int | None = None
        self._cached_sdc_index: int | None = None

    def _profile_sync(self) -> None:
        if not bool(self.profile_policy_runtime_sync):
            return
        if getattr(self.dev, "type", "cpu") == "cuda":
            try:
                self.torch.cuda.synchronize(self.dev)
            except Exception:
                pass

    def _profile_stamp(self) -> float:
        self._profile_sync()
        return time.perf_counter()

    def _decode_blocker_conditioned_natural_queries_np(
        self,
        batch: dict[str, Any],
        pred: dict[str, Any],
        query_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode natural roots for late-bound exact-blocker candidates.

        V16.8.43 deliberately does not modify the scene-level critical set used
        by RCOT/BCOT/NCF.  Instead, the physical-recovery branch may ask the same
        frozen natural decoder about additional agents already present in the
        frozen collision context.  Natural decoding must use the root-scene graph
        latent, never the candidate-conditioned planner graph.
        """
        q = np.asarray(query_indices, dtype=np.int64).reshape(-1)
        if q.size <= 0:
            return (
                np.zeros((0, 0, 0, 7), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
            )
        z_scene = pred.get("natural_scene_z_agent")
        if not self.torch.is_tensor(z_scene):
            return (
                np.zeros((0, 0, 0, 7), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
            )
        idx_t = self.torch.as_tensor(q[None, :], device=self.dev, dtype=self.torch.long)
        anchor7 = self.model._critical_anchor7(batch["state/history"], idx_t)
        device_type = z_scene.device.type
        with self.torch.autocast(device_type=device_type, enabled=False):
            natural = self.model.natural_decoder(
                z_scene.float(), idx_t, decode_traj=True,
                anchor7=anchor7.float(),
                dt=float(self.model.cfg.get("time", {}).get("dt", 0.1)),
            )
        natural = self.model._add_natural_anchor(natural, anchor7)
        traj_t = natural.get("traj")
        logits_t = natural.get("logits")
        if not (self.torch.is_tensor(traj_t) and self.torch.is_tensor(logits_t)):
            return (
                np.zeros((0, 0, 0, 7), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
            )
        return (
            traj_t[0].detach().cpu().numpy().astype(np.float32, copy=False),
            logits_t[0].detach().cpu().numpy().astype(np.float32, copy=False),
        )

    def _trajectory_to_action(
        self,
        state: Any,
        agent_state: np.ndarray,
        sdc_index: int,
        traj: np.ndarray,
        *,
        precomputed_target: np.ndarray | None = None,
        precomputed_accel: float | None = None,
    ) -> Any:
        try:
            from waymax import datatypes  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("waymax.datatypes is required to convert a selected COWP trajectory to a Waymax action.") from exc
        N = agent_state.shape[0]
        data_dim = int(self.cfg.get("waymax", {}).get("action_dim", 3))
        if self.action_mode == "absolute_xy_yaw":
            data_dim = max(data_dim, 5)
        data = np.zeros((N, data_dim), dtype=np.float32)
        valid = np.zeros((N, 1), dtype=bool)
        valid[sdc_index, 0] = True
        desired = traj[0]
        if precomputed_target is None or precomputed_accel is None:
            target, accel = _consistent_one_step_target(
                agent_state[sdc_index], desired, self.cfg, self._previous_longitudinal_accel
            )
        else:
            target = np.asarray(precomputed_target, dtype=np.float32)
            accel = float(precomputed_accel)
        self._previous_longitudinal_accel = float(accel)
        if self.action_mode == "absolute_xy_yaw":
            data[sdc_index, :5] = target
        else:
            dx = float(target[0] - agent_state[sdc_index, 0])
            dy = float(target[1] - agent_state[sdc_index, 1])
            dyaw = float(_wrap_angle(float(target[2] - agent_state[sdc_index, 6])))
            data[sdc_index, : min(data_dim, 3)] = np.asarray([dx, dy, dyaw], dtype=np.float32)[:data_dim]
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        try:
            import jax.numpy as jnp  # type: ignore
            return datatypes.Action(data=jnp.asarray(data), valid=jnp.asarray(valid))
        except Exception:
            return datatypes.Action(data=data, valid=valid)

    def __call__(self, state: Any, *, step: int | None = None, scenario_index: int | None = None) -> Any:
        profile_enabled = bool(self.profile_policy_runtime)
        profile_t0 = self._profile_stamp() if profile_enabled else 0.0
        if step == 0 or (scenario_index is not None and scenario_index != self._previous_scenario_index):
            self._previous_longitudinal_accel = 0.0
            self._previous_selected_traj = None
            self._recovery_commitment_active = False
            self._recovery_hysteresis_active = False
            self._recovery_frontier_macro = -1
            self._recovery_bridge_pending = False
            self._recovery_bridge_allowed_macros = frozenset()
        self._previous_scenario_index = scenario_index
        method, gate_mode = _canonical_online_method(getattr(self, "method", "cowp"), self.ncf_gate_mode)
        needs_cowp_risk = method not in {"planner_score_only", "conventional_safety", "idm_lattice"}
        cached_sdc = self._cached_sdc_index if (scenario_index is not None and self._cached_sdc_scenario_index == int(scenario_index)) else None
        history, agent_state, sdc_index, traj_components, current_t = extract_online_state_bundle(
            state, self.cfg, cached_sdc_index=cached_sdc
        )
        if scenario_index is not None and cached_sdc is None:
            self._cached_sdc_scenario_index = int(scenario_index)
            self._cached_sdc_index = int(sdc_index)
        if scenario_index is not None and self._cached_roadgraph_scenario_index == int(scenario_index) and self._cached_roadgraph is not None:
            roadgraph = self._cached_roadgraph
        else:
            roadgraph = _extract_roadgraph_tokens(state, self.cfg)
            if scenario_index is not None:
                self._cached_roadgraph_scenario_index = int(scenario_index)
                self._cached_roadgraph = roadgraph
        future_source = str(self.cfg.get("planning", {}).get("online_other_future_source", "constant_velocity")).lower()
        if future_source in {"logged", "logged_oracle", "oracle"}:
            other_future_trajs = _extract_logged_future_agent_trajs(
                state, sdc_index, self.cfg
            )
            if other_future_trajs is None:
                raise RuntimeError(
                    "Explicit logged-oracle evaluation requested, but privileged "
                    "log_trajectory could not be extracted."
                )
        else:
            # Causal main evaluation: no future state from sim/log trajectory is read.
            # Candidate checks use the existing constant-velocity fallback and the
            # learned response/certificate branch.  This also removes a second full
            # device-to-host trajectory copy at every simulator step.
            other_future_trajs = None
        profile_t_state = self._profile_stamp() if profile_enabled else 0.0
        batch_np = build_online_batch(
            agent_state, sdc_index, self.cfg, history_model_state=history, roadgraph=roadgraph,
            other_future_trajs=other_future_trajs, compute_rule_risk=needs_cowp_risk,
        )
        profile_t_candidate = self._profile_stamp() if profile_enabled else 0.0
        online_keys = (
            "state/history",
            "state/agent_valid",
            "state/is_sdc",
            "cowp/candidates/trajectory",
            "cowp/candidates/valid",
            "cowp/candidates/macro_type",
            "cowp/candidates/ego_utility_prior",
            "cowp/candidates/conventional_safe",
            "cowp/candidates/roadgraph_safe",
            "cowp/candidates/collision_safe",
            "cowp/candidates/collision_safe_prefix_steps",
            "cowp/candidates/collision_min_clearance_margin_m",
            "cowp/candidates/rule_risk",
            "cowp/critical/track_index",
            "cowp/critical/input_index",
            "cowp/critical/valid",
            "map/conflict_regions",
            "map/conflict_region_valid",
        )
        batch = {k: self.torch.as_tensor(batch_np[k], device=self.dev) for k in online_keys if k in batch_np}
        profile_t_h2d = self._profile_stamp() if profile_enabled else 0.0
        # Execution overrides are populated only by the v16.8.38/v16.8.39
        # physically certified recovery branches.
        # Define them before the shared inference block so historical baselines keep
        # their unchanged path without unbound local state.
        recovery_tube_trajectory_override: np.ndarray | None = None
        recovery_tube_target_override: np.ndarray | None = None
        recovery_tube_accel_override: float | None = None
        with self.torch.inference_mode():
            pred = self.model(batch, stage="planner")
            profile_t_model = self._profile_stamp() if profile_enabled else 0.0
            scores = self.torch.nan_to_num(pred["planner_score"][0].float(), nan=1e6, posinf=1e6, neginf=-1e6)
            cand_valid = batch["cowp/candidates/valid"][0].bool()
            # One host synchronization for a predicate reused throughout the
            # decision path.  The previous implementation queried any(valid)
            # repeatedly inside normalization and diagnostics, forcing the CUDA
            # stream to synchronize many times per simulator step.
            has_valid = bool(cand_valid.any().detach().cpu().item())
            conventional = batch.get("cowp/candidates/conventional_safe", batch["cowp/candidates/valid"])[0].bool()
            roadgraph_safe = batch.get("cowp/candidates/roadgraph_safe", batch["cowp/candidates/valid"])[0].bool()
            collision_safe = batch.get("cowp/candidates/collision_safe", batch["cowp/candidates/conventional_safe"])[0].bool()
            collision_prefix_steps = batch.get(
                "cowp/candidates/collision_safe_prefix_steps",
                self.torch.zeros_like(batch["cowp/candidates/valid"], dtype=self.torch.int32),
            )[0].float()
            collision_margin = batch.get(
                "cowp/candidates/collision_min_clearance_margin_m",
                self.torch.zeros_like(batch["cowp/candidates/valid"], dtype=self.torch.float32),
            )[0].float()
            utility = batch.get("cowp/candidates/ego_utility_prior", None)
            utility_scores = self.torch.nan_to_num(utility[0].float(), nan=1e6, posinf=1e6, neginf=-1e6) if utility is not None else scores
            continuity_np = _plan_continuity_risk_np(
                batch_np["cowp/candidates/trajectory"][0],
                batch_np["cowp/candidates/valid"][0].astype(bool),
                self._previous_selected_traj,
                self.cfg,
            )
            continuity_risk = self.torch.as_tensor(continuity_np, device=self.dev, dtype=scores.dtype)
            continuity_weight = float(self.cfg.get("planning", {}).get("online_plan_continuity_weight", 0.20))

            if not needs_cowp_risk:
                # Fast exact-equivalent path for internal baselines.  These methods
                # do not use witness, priority, pressure, rule, action, or outcome
                # risk in selection, so skip those expensive online computations.
                if method == "idm_lattice":
                    select_mask = cand_valid & conventional
                    adjusted_scores = utility_scores
                elif method == "conventional_safety":
                    select_mask = cand_valid & conventional
                    adjusted_scores = scores
                else:  # planner_score_only
                    select_mask = cand_valid
                    adjusted_scores = scores
                adjusted_scores = adjusted_scores + continuity_weight * continuity_risk
                fallback_used = False
                fallback_reason = "accepted_baseline"
                macro_t = batch["cowp/candidates/macro_type"][0].long()
                stop_ids = self.torch.as_tensor(
                    [int(MacroType.STOP_BEFORE_CONFLICT), int(MacroType.YIELD), int(MacroType.CREEP), int(MacroType.NEUTRAL_EGO)],
                    device=self.dev, dtype=macro_t.dtype,
                )
                stop_like = (macro_t[:, None] == stop_ids[None, :]).any(dim=-1)
                has_select = bool(select_mask.any().detach().cpu().item())
                if not has_select:
                    fallback_used = True
                    if bool((cand_valid & stop_like).any().detach().cpu().item()):
                        select_mask = cand_valid & stop_like
                        fallback_reason = "baseline_use_stop_like"
                    else:
                        select_mask = cand_valid
                        fallback_reason = "baseline_use_valid" if has_valid else "baseline_no_valid_emergency_stop"
                    has_select = has_valid
                selected = int(self.torch.argmin(self.torch.where(select_mask, adjusted_scores, self.torch.full_like(adjusted_scores, float("inf")))).item()) if has_select else 0
                selected_report = int(selected) if has_valid else -1
                selected_macro_type = int(macro_t[selected].detach().cpu().item()) if has_valid else int(MacroType.NEUTRAL_EGO)
                selected_macro_name = _macro_name(selected_macro_type) if has_valid else "EMERGENCY_BOUNDED_STOP"
                selected_candidate_valid = bool(cand_valid[selected].detach().cpu().item()) if has_valid else False
                selected_candidate_conventional_safe = bool(conventional[selected].detach().cpu().item()) if has_valid else False
                valid_prefix = collision_prefix_steps[cand_valid] if has_valid else collision_prefix_steps[:0]
                baseline_host = self.torch.stack([
                    select_mask.sum().float(),
                    cand_valid.sum().float(),
                    (cand_valid & conventional).sum().float(),
                    batch["cowp/critical/valid"][0].bool().sum().float(),
                    batch["map/conflict_region_valid"][0].bool().sum().float(),
                    scores[selected] if has_valid else scores.new_tensor(0.0),
                    (cand_valid & roadgraph_safe).sum().float(),
                    (cand_valid & collision_safe).sum().float(),
                    valid_prefix.max() if valid_prefix.numel() else scores.new_tensor(0.0),
                    collision_prefix_steps[selected] if has_valid else scores.new_tensor(0.0),
                    roadgraph_safe[selected].float() if has_valid else scores.new_tensor(0.0),
                    collision_safe[selected].float() if has_valid else scores.new_tensor(0.0),
                    collision_margin[selected] if has_valid else scores.new_tensor(0.0),
                ]).detach().cpu().tolist()
                conv_n, road_n, coll_n = int(baseline_host[2]), int(baseline_host[6]), int(baseline_host[7])
                if conv_n > 0:
                    zero_conv_reason = "none"
                elif road_n == 0 and coll_n == 0:
                    zero_conv_reason = "road_and_collision_empty"
                elif road_n == 0:
                    zero_conv_reason = "roadgraph_empty"
                elif coll_n == 0:
                    zero_conv_reason = "collision_empty"
                else:
                    zero_conv_reason = "intersection_empty"
                diag = {
                    "scenario_index": int(scenario_index) if scenario_index is not None else -1,
                    "step": int(step) if step is not None else -1,
                    "selected_candidate": int(selected_report),
                    "selected_macro_type": int(selected_macro_type),
                    "selected_macro_name": str(selected_macro_name),
                    "selected_candidate_valid": bool(selected_candidate_valid),
                    "selected_candidate_conventional_safe": bool(selected_candidate_conventional_safe),
                    "accepted_candidates": int(baseline_host[0]),
                    "frontier_candidates": -1,
                    "valid_candidates": int(baseline_host[1]),
                    "conventional_candidates": int(baseline_host[2]),
                    "roadgraph_safe_candidates": int(baseline_host[6]),
                    "collision_safe_candidates": int(baseline_host[7]),
                    "max_collision_safe_prefix_steps": int(round(baseline_host[8])),
                    "selected_collision_safe_prefix_steps": int(round(baseline_host[9])),
                    "selected_candidate_roadgraph_safe": bool(baseline_host[10] > 0.5),
                    "selected_candidate_collision_safe": bool(baseline_host[11] > 0.5),
                    "selected_collision_min_clearance_margin_m": float(baseline_host[12]),
                    "zero_conventional_reason": str(zero_conv_reason),
                    "critical_agents": int(baseline_host[3]),
                    "conflict_tokens": int(baseline_host[4]),
                    "fallback_used": bool(fallback_used),
                    "fallback_reason": fallback_reason,
                    "max_witness_prob": 0.0,
                    "mean_witness_prob": 0.0,
                    "mean_witness_uncertainty": 0.0,
                    "max_witness_certificate": 0.0,
                    "min_opr": 1.0,
                    "mean_opr": 1.0,
                    "score": float(baseline_host[5]),
                    "witness_threshold": float(self.witness_threshold),
                    "bcot_risk_budget": float(
                        self.cfg.get("planning", {}).get("candidate_transport_budget", 0.35)
                        if self.bcot_risk_budget is None else self.bcot_risk_budget
                    ),
                    "alpha_opr": float(self.cfg.get("planning", {}).get("alpha_opr_infer", self.cfg.get("ncf", {}).get("alpha_opr", 0.35))),
                    "gate_mode": str(gate_mode),
                    "method": str(method),
                    "priority_hard_threshold": float(self.priority_hard_threshold),
                    "accepted_primary_bad_candidates": 0,
                    "severe_bad_candidates": 0,
                    "option_bad_candidates": 0,
                    "selected_priority_max": 0.0,
                    "selected_priority_mean": 0.0,
                    "selected_outcome_risk": 0.0,
                    "selected_outcome_decision_risk": 0.0,
                    "selected_candidate_ncf_prob": 0.0,
                    "selected_candidate_false_safe_prob": 0.0,
                    "selected_candidate_quality_prob": 0.0,
                    "selected_candidate_cert_risk": 0.0,
                    "selected_candidate_pressure_prior": 0.0,
                    "selected_candidate_rule_risk": 0.0,
                    "selected_candidate_action_risk": 0.0,
                    "min_candidate_cert_risk": 0.0,
                    "mean_candidate_cert_risk": 0.0,
                    "mean_candidate_pressure_prior": 0.0,
                    "mean_candidate_rule_risk": 0.0,
                    "mean_candidate_action_risk": 0.0,
                    "beta_threshold": float(self.cfg.get("burden", {}).get("beta0_vehicle", 0.65)),
                }
                diag["selected_plan_continuity_risk"] = float(continuity_np[selected]) if has_valid else 0.0
                traj, emergency_action_used, execution_source = _resolve_execution_trajectory(
                    batch_np["cowp/candidates/trajectory"][0], selected, has_valid, agent_state[sdc_index], self.cfg
                )
                diag["emergency_action_used"] = bool(emergency_action_used)
                diag["execution_trajectory_source"] = str(execution_source)
                self._last_diagnostics = diag
                self._diagnostics_log.append(diag)
                self._previous_selected_traj = np.array(traj, copy=True)
                if profile_enabled:
                    profile_t_selection = self._profile_stamp()
                    action = self._trajectory_to_action(state, agent_state, sdc_index, traj)
                    profile_t_action = self._profile_stamp()
                    diag.update({
                        "runtime_state_extract_map_s": float(profile_t_state - profile_t0),
                        "runtime_candidate_build_cpu_s": float(profile_t_candidate - profile_t_state),
                        "runtime_h2d_s": float(profile_t_h2d - profile_t_candidate),
                        "runtime_model_forward_s": float(profile_t_model - profile_t_h2d),
                        "runtime_selection_s": float(profile_t_selection - profile_t_model),
                        "runtime_action_projection_s": float(profile_t_action - profile_t_selection),
                        "runtime_policy_total_s": float(profile_t_action - profile_t0),
                    })
                    self._diagnostics_log[-1] = dict(diag)
                    self._last_diagnostics = diag
                    return action
                return self._trajectory_to_action(state, agent_state, sdc_index, traj)

            pcfg = self.cfg.get("planning", {})
            temp = max(float(pcfg.get("witness_temperature", 1.0)), 1e-3)
            bias = float(pcfg.get("witness_logit_bias", 0.0))
            logit_witness = self.torch.sigmoid((pred["witness"]["exist_logits"][0].float() - bias) / temp)
            evidence_witness = pred["witness"].get("evidential_prob")
            uncertainty = pred["witness"].get("epistemic_uncertainty")
            source = str(pcfg.get("witness_probability_source", "mixed")).lower()
            if source == "logit" or not self.torch.is_tensor(evidence_witness):
                witness = logit_witness
            elif source == "evidential":
                witness = evidence_witness[0].float()
            else:
                mix = float(pcfg.get("evidential_probability_mix", 0.5))
                witness = (1.0 - mix) * logit_witness + mix * evidence_witness[0].float()
            if self.torch.is_tensor(uncertainty):
                uncertainty = uncertainty[0].float().clamp(0.0, 1.0)
            else:
                uncertainty = self.torch.zeros_like(witness)
            witness = self.torch.nan_to_num(witness, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            uncertainty = self.torch.nan_to_num(uncertainty, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            ucb_scale = float(pcfg.get("evidential_ucb_scale", 0.0 if source == "logit" else 0.15))
            witness_cert = (witness + ucb_scale * uncertainty).clamp(0.0, 1.0)
            opr = self.torch.nan_to_num(pred["witness"]["opr"][0].float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            burden = pred["witness"].get("burden_total")
            c_i = pred["witness"].get("c_i")
            outcome = pred.get("outcome", {})
            cand_ncf_logit = pred.get("candidate_ncf_logit")
            cand_fs_logit = pred.get("candidate_false_safe_logit")
            cand_quality_logit = pred.get("candidate_quality_logit")
            if self.torch.is_tensor(cand_ncf_logit):
                cand_ncf_prob = self.torch.sigmoid(self.torch.nan_to_num(cand_ncf_logit[0].float(), nan=0.0, posinf=20.0, neginf=-20.0))
            else:
                cand_ncf_prob = self.torch.sigmoid(self.torch.nan_to_num(-scores, nan=0.0, posinf=20.0, neginf=-20.0))
            if self.torch.is_tensor(cand_fs_logit):
                cand_false_safe_prob = self.torch.sigmoid(self.torch.nan_to_num(cand_fs_logit[0].float(), nan=0.0, posinf=20.0, neginf=-20.0))
            else:
                cand_false_safe_prob = self.torch.sigmoid(self.torch.nan_to_num(scores, nan=0.0, posinf=20.0, neginf=-20.0))
            if self.torch.is_tensor(cand_quality_logit):
                cand_quality_prob = self.torch.sigmoid(self.torch.nan_to_num(cand_quality_logit[0].float(), nan=0.0, posinf=20.0, neginf=-20.0))
            else:
                cand_quality_prob = cand_ncf_prob * (1.0 - cand_false_safe_prob)
            pcfg_runtime = self.cfg.get("planning", {})
            candidate_cert_risk = (
                float(pcfg_runtime.get("candidate_risk_ncf_weight", 1.0)) * (1.0 - cand_ncf_prob.clamp(0.0, 1.0))
                + float(pcfg_runtime.get("candidate_risk_false_safe_weight", 2.0)) * cand_false_safe_prob.clamp(0.0, 1.0)
                + float(pcfg_runtime.get("candidate_risk_quality_weight", 0.75)) * (1.0 - cand_quality_prob.clamp(0.0, 1.0))
            )
            pressure_np = _candidate_pressure_prior_np(
                agent_state,
                sdc_index,
                batch_np["cowp/candidates/trajectory"][0],
                batch_np["cowp/candidates/valid"][0].astype(bool),
                batch_np["cowp/critical/track_index"][0],
                batch_np["cowp/critical/valid"][0].astype(bool),
                batch_np["cowp/candidates/macro_type"][0],
                self.cfg,
                other_future_trajs=other_future_trajs,
            )
            pressure_prior = self.torch.as_tensor(pressure_np, device=self.dev, dtype=scores.dtype).clamp(0.0, 1.0)
            rule_risk_t = batch.get("cowp/candidates/rule_risk")
            if self.torch.is_tensor(rule_risk_t):
                rule_risk = self.torch.nan_to_num(rule_risk_t[0].float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            else:
                rule_risk = self.torch.zeros_like(scores)
            action_risk_np, action_targets_np, action_accels_np = _one_step_action_risk_np(
                agent_state,
                sdc_index,
                batch_np["cowp/candidates/trajectory"][0],
                batch_np["cowp/candidates/valid"][0].astype(bool),
                self.cfg,
                self._previous_longitudinal_accel,
                return_targets=True,
            )
            action_risk = self.torch.as_tensor(action_risk_np, device=self.dev, dtype=scores.dtype).clamp(0.0, 1.0)
            traj_np_runtime = np.asarray(batch_np["cowp/candidates/trajectory"][0])
            controller_transition_feasible_np = (
                _controller_transition_feasible_np(
                    agent_state[int(sdc_index)], traj_np_runtime[:, 0], self.cfg,
                    self._previous_longitudinal_accel,
                )
                if traj_np_runtime.ndim >= 3 and traj_np_runtime.shape[1] > 0
                else np.zeros((int(traj_np_runtime.shape[0]) if traj_np_runtime.ndim else 0,), dtype=bool)
            )
            controller_transition_feasible_np &= np.asarray(
                batch_np["cowp/candidates/valid"][0], dtype=bool
            )

            def _scene_norm(x):
                x = self.torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
                if not has_valid:
                    return self.torch.zeros_like(x)
                vals = x[cand_valid]
                lo = vals.min()
                hi = vals.quantile(0.90) if vals.numel() > 1 else vals.max()
                span = (hi - lo).clamp_min(float(self.cfg.get("planning", {}).get("decision_risk_min_spread", 1e-3)))
                y = ((x - lo) / span).clamp(0.0, 1.0)
                spread = vals.std(unbiased=False) if vals.numel() > 1 else self.torch.tensor(0.0, device=self.dev, dtype=x.dtype)
                return self.torch.where(spread >= float(self.cfg.get("planning", {}).get("decision_risk_min_std", 1e-3)), y, self.torch.zeros_like(y))

            cert_decision_risk = _scene_norm(candidate_cert_risk)
            pressure_decision_risk = _scene_norm(pressure_prior)
            rule_decision_risk = _scene_norm(rule_risk)
            action_decision_risk = _scene_norm(action_risk)
            score_decision_risk = _scene_norm(scores)
            traj_t = batch.get("cowp/candidates/trajectory")
            if self.torch.is_tensor(traj_t) and traj_t.ndim >= 4:
                xy0 = traj_t[0, :, 0, :2].float()
                xy1 = traj_t[0, :, -1, :2].float()
                delta_xy = self.torch.nan_to_num(xy1 - xy0, nan=0.0, posinf=0.0, neginf=0.0)
                if traj_t.shape[-1] >= 3:
                    yaw0 = self.torch.nan_to_num(traj_t[0, :, 0, 2].float(), nan=0.0)
                    heading0 = self.torch.stack([self.torch.cos(yaw0), self.torch.sin(yaw0)], dim=-1)
                    candidate_progress = (delta_xy * heading0).sum(dim=-1).clamp_min(0.0)
                else:
                    candidate_progress = self.torch.linalg.norm(delta_xy, dim=-1)
                candidate_progress = self.torch.where(cand_valid, candidate_progress, self.torch.zeros_like(candidate_progress))
            else:
                candidate_progress = self.torch.zeros_like(scores)
            progress_ref = candidate_progress[cand_valid].max().clamp_min(1.0e-6) if has_valid else self.torch.tensor(1.0, device=self.dev, dtype=scores.dtype)
            progress_shortfall = (1.0 - (candidate_progress / progress_ref).clamp(0.0, 1.0)).clamp(0.0, 1.0)
            if isinstance(outcome, dict) and (float(self.outcome_risk_penalty) > 0.0 or method == "cowp_fallback_outcome"):
                col_r = self.torch.sigmoid(outcome.get("collision_logit", self.torch.zeros_like(scores))[0].float()).clamp(0.0, 1.0)
                off_r = self.torch.sigmoid(outcome.get("offroad_logit", self.torch.zeros_like(scores))[0].float()).clamp(0.0, 1.0)
                outcome_risk = self.torch.nan_to_num(1.0 - (1.0 - col_r) * (1.0 - off_r), nan=1.0, posinf=1.0, neginf=0.0)
            else:
                outcome_risk = self.torch.zeros_like(scores)
            outcome_decision_risk = _scene_norm(outcome_risk)

            # The main COWP result must remain a non-coercion certificate.  v5
            # silently replaced a flat certificate with a 90% blend of outcome,
            # action, rule, and pressure heuristics.  That can improve conventional
            # closed-loop metrics, but it invalidates the paper's mechanism claim.
            structured = pred.get("candidate_structured_coercion_risk")
            if self.torch.is_tensor(structured):
                structured_risk = self.torch.nan_to_num(structured[0].float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                structured_decision_risk = _scene_norm(structured_risk)
            else:
                structured_risk = (1.0 - cand_ncf_prob).clamp(0.0, 1.0)
                structured_decision_risk = cert_decision_risk
            structured_mix = float(pcfg_runtime.get("candidate_structured_decision_mix", 0.35))
            cert_decision_risk = ((1.0 - structured_mix) * cert_decision_risk + structured_mix * structured_decision_risk).clamp(0.0, 1.0)
            raw_vals = candidate_cert_risk[cand_valid] if has_valid else candidate_cert_risk[:0]
            raw_spread = raw_vals.float().std(unbiased=False) if raw_vals.numel() > 1 else self.torch.tensor(0.0, device=self.dev, dtype=scores.dtype)
            flat_cert = bool((raw_spread < float(pcfg_runtime.get("candidate_cert_fallback_min_std", 2.0e-3))).detach().cpu().item())
            allow_hybrid = bool(pcfg_runtime.get("candidate_cert_allow_hybrid_fallback", False))
            if flat_cert and not allow_hybrid:
                # A deterministic, paper-aligned fallback: use the analytic response-
                # set preservation certificate, never planner/outcome score.
                cert_decision_risk = structured_decision_risk
                cand_ncf_prob = (1.0 - structured_risk).clamp(0.0, 1.0)
                cand_false_safe_prob = structured_risk
                cand_quality_prob = (1.0 - structured_risk).clamp(0.0, 1.0)
            elif allow_hybrid:
                fallback_cert_risk = (
                    float(pcfg_runtime.get("candidate_cert_fallback_outcome_mix", 0.55)) * outcome_decision_risk
                    + float(pcfg_runtime.get("candidate_cert_fallback_action_mix", 0.20)) * action_decision_risk
                    + float(pcfg_runtime.get("candidate_cert_fallback_rule_mix", 0.15)) * rule_decision_risk
                    + float(pcfg_runtime.get("candidate_cert_fallback_pressure_mix", 0.10)) * pressure_decision_risk
                ).clamp(0.0, 1.0)
                mix_value = float(pcfg_runtime.get("candidate_cert_flat_fallback_mix", 0.90) if flat_cert else pcfg_runtime.get("candidate_cert_hybrid_fallback_mix", 0.0))
                cert_decision_risk = ((1.0 - mix_value) * cert_decision_risk + mix_value * fallback_cert_risk).clamp(0.0, 1.0)
                cand_ncf_prob = ((1.0 - mix_value) * cand_ncf_prob + mix_value * (1.0 - fallback_cert_risk)).clamp(0.0, 1.0)
                cand_false_safe_prob = ((1.0 - mix_value) * cand_false_safe_prob + mix_value * fallback_cert_risk).clamp(0.0, 1.0)
                cand_quality_prob = ((1.0 - mix_value) * cand_quality_prob + mix_value * (1.0 - fallback_cert_risk)).clamp(0.0, 1.0)
            candidate_cert_risk = cert_decision_risk

            transport_risk_t = pred.get("candidate_transport_risk")
            if self.torch.is_tensor(transport_risk_t):
                transport_risk = self.torch.nan_to_num(
                    transport_risk_t[0].float(), nan=1.0, posinf=1.0, neginf=0.0
                ).clamp(0.0, 1.0)
            else:
                transport_risk = structured_risk
            transport_unc_t = pred.get("candidate_transport_uncertainty")
            if self.torch.is_tensor(transport_unc_t):
                transport_uncertainty = self.torch.nan_to_num(
                    transport_unc_t[0].float(), nan=1.0, posinf=1.0, neginf=0.0
                ).clamp(0.0, 1.0)
            else:
                transport_uncertainty = self.torch.zeros_like(scores)
            transport_severe_t = pred.get("candidate_transport_severe_prob")
            if self.torch.is_tensor(transport_severe_t):
                transport_severe = self.torch.nan_to_num(
                    transport_severe_t[0].float(), nan=1.0, posinf=1.0, neginf=0.0
                ).clamp(0.0, 1.0)
            else:
                transport_severe = self.torch.zeros_like(scores)

            crit_mask = batch["cowp/critical/valid"][0].bool()
            witness = self.torch.where(crit_mask[None, :], witness, self.torch.zeros_like(witness))
            witness_cert = self.torch.where(crit_mask[None, :], witness_cert, self.torch.zeros_like(witness_cert))
            uncertainty = self.torch.where(crit_mask[None, :], uncertainty, self.torch.zeros_like(uncertainty))
            opr = self.torch.where(crit_mask[None, :], opr, self.torch.ones_like(opr))
            alpha = float(self.cfg.get("planning", {}).get("alpha_opr_infer", self.cfg.get("ncf", {}).get("alpha_opr", 0.35)))
            adjusted_scores = scores
            priority = self.torch.zeros_like(witness)
            primary_bad = self.torch.zeros_like(cand_valid)
            severe_bad = self.torch.zeros_like(cand_valid)
            option_bad = self.torch.zeros_like(cand_valid)
            if method == "idm_lattice":
                accepted = cand_valid & conventional
                adjusted_scores = utility_scores
            elif method == "conventional_safety":
                accepted = cand_valid & conventional
            elif method == "planner_score_only":
                accepted = cand_valid
            elif gate_mode == "hard":
                predicted_bad = (witness_cert >= float(self.witness_threshold)).any(dim=-1)
                accepted = cand_valid & conventional & ~predicted_bad
                accepted = accepted & (opr.min(dim=-1).values >= alpha)
            else:
                priority_np = _priority_claim_weights(
                    agent_state,
                    sdc_index,
                    batch_np["cowp/candidates/trajectory"][0],
                    batch_np["cowp/candidates/valid"][0].astype(bool),
                    batch_np["cowp/critical/track_index"][0],
                    batch_np["cowp/critical/valid"][0].astype(bool),
                    batch_np["cowp/candidates/macro_type"][0],
                    self.cfg,
                )
                heuristic_priority = self.torch.as_tensor(priority_np, device=self.dev, dtype=witness.dtype)
                learned_priority = pred.get("priority_claim_logits")
                if self.torch.is_tensor(learned_priority):
                    learned_priority = self.torch.sigmoid(learned_priority[0].float())
                    # Before the new head is trained it may be uncalibrated; blend with
                    # the physically grounded heuristic rather than replacing it.
                    blended_priority = 0.5 * heuristic_priority + 0.5 * learned_priority
                    # A high-confidence physical priority claim is a hard semantic
                    # anchor; the learned head may refine unknown relations but may
                    # not dilute an already protected relation below the gate.
                    priority = self.torch.where(
                        heuristic_priority >= float(self.priority_hard_threshold),
                        heuristic_priority,
                        blended_priority,
                    )
                else:
                    priority = heuristic_priority
                primary_claim = priority >= float(self.priority_hard_threshold)
                hard_max_unc = float(pcfg_runtime.get("set_transport_hard_max_uncertainty", 0.40))
                opr_ucb_scale = float(pcfg_runtime.get("set_transport_opr_ucb_scale", 0.50))
                confident_pair = uncertainty <= hard_max_unc
                opr_upper = (opr + opr_ucb_scale * uncertainty).clamp(0.0, 1.0)
                severe_pair_bad = ((witness_cert >= float(self.secondary_witness_threshold))
                                   & (opr_upper <= float(self.secondary_opr_alpha))
                                   & primary_claim & confident_pair).any(dim=-1)
                transport_gate_mode = str(pcfg_runtime.get("candidate_transport_gate_mode", "budget")).lower()
                if transport_gate_mode in {"pairmax", "pair_max", "legacy"}:
                    primary_bad = ((witness_cert >= float(self.witness_threshold)) & primary_claim & confident_pair).any(dim=-1)
                    option_bad = ((opr_upper < alpha) & primary_claim & confident_pair).any(dim=-1)
                    severe_bad = severe_pair_bad
                else:
                    configured_budget = float(pcfg_runtime.get("candidate_transport_budget", 0.35))
                    transport_budget = configured_budget if self.bcot_risk_budget is None else float(self.bcot_risk_budget)
                    transport_budget = min(max(transport_budget, 0.02), 0.98)
                    transport_ucb = (
                        transport_risk
                        + float(pcfg_runtime.get("candidate_transport_ucb_scale", 0.25)) * transport_uncertainty
                    ).clamp(0.0, 1.0)
                    primary_bad = transport_ucb >= transport_budget
                    option_bad = self.torch.zeros_like(primary_bad)
                    aggregate_severe_bad = (
                        transport_severe >= float(pcfg_runtime.get("candidate_transport_severe_threshold", 0.80))
                    )
                    # The aggregate tail head is useful for ranking and audit, but it
                    # compresses all protected relations into one scalar.  Making it
                    # a second unconditional veto duplicates the localized pair
                    # certificate and is especially brittle when root-recovery
                    # labels are sparse.  Keep the pair-localized high-confidence
                    # veto hard; enable the aggregate veto only as an explicit
                    # ablation/safety setting.
                    if bool(pcfg_runtime.get("candidate_transport_aggregate_severe_hard_veto", False)):
                        severe_bad = severe_pair_bad | aggregate_severe_bad
                    else:
                        severe_bad = severe_pair_bad
                uncertain_mix = float(pcfg_runtime.get("set_transport_uncertain_penalty", 0.25))
                burden_penalty = transport_risk
                option_penalty = float(pcfg_runtime.get("candidate_transport_uncertainty_penalty", 0.20)) * transport_uncertainty
                cert_penalty = float(self.cfg.get("planning", {}).get("candidate_certificate_penalty", 1.0))
                pressure_penalty = float(self.cfg.get("planning", {}).get("candidate_pressure_prior_penalty", 0.75))
                rule_penalty = float(self.cfg.get("planning", {}).get("candidate_rule_risk_penalty", 1.25))
                action_penalty = float(self.cfg.get("planning", {}).get("candidate_action_risk_penalty", 1.0))
                transport_pure = bool(pcfg_runtime.get("candidate_transport_pure_selector", True))
                if transport_pure:
                    adjusted_scores = (
                        scores
                        + float(self.soft_ncf_penalty) * (burden_penalty + option_penalty)
                        + rule_penalty * rule_decision_risk
                        + action_penalty * action_decision_risk
                        + float(self.outcome_risk_penalty) * outcome_decision_risk
                    )
                else:
                    adjusted_scores = (
                        scores
                        + float(self.soft_ncf_penalty) * (burden_penalty + option_penalty)
                        + cert_penalty * cert_decision_risk
                        + pressure_penalty * pressure_decision_risk
                        + rule_penalty * rule_decision_risk
                        + action_penalty * action_decision_risk
                        + float(self.outcome_risk_penalty) * outcome_decision_risk
                    )
                if gate_mode == "soft":
                    accepted = cand_valid & conventional
                elif gate_mode in {"none", "off"}:
                    accepted = cand_valid & conventional
                else:
                    accepted = cand_valid & conventional & ~primary_bad & ~option_bad & ~severe_bad & (outcome_risk <= float(self.outcome_risk_threshold))
            # Separate the semantic certificate set from the shortlist used for
            # one-plan selection.  Online diagnostics and offline gates must refer
            # to the former, not an arbitrary top-k frontier.
            certificate_accepted = accepted.clone()
            selection_mask = accepted

            # Scene-adaptive feasibility frontier.  If the absolute witness/OPR
            # calibration is imperfect, restrict COWP to the least-coercive
            # conventional frontier in the current scene.  This makes the online
            # controller consistent with the set-valued NCF certificate used in
            # learned-offline evaluation.
            if method == "cowp_cert_utility" and gate_mode in {"priority", "soft"}:
                # Same semantic certificate and physical shield as COWP, but no
                # second BCOT/frontier ranking among already accepted trajectories.
                # This isolates the post-certificate ranking question while keeping
                # the online physical-safety guard unchanged.
                pcfg_ctu = self.cfg.get("planning", {})
                physical_ok_ctu = (
                    (action_risk <= float(pcfg_ctu.get("candidate_hard_max_action_risk", 0.45)))
                    & (rule_risk <= float(pcfg_ctu.get("candidate_hard_max_rule_risk", 0.70)))
                    & (outcome_risk <= float(self.outcome_risk_threshold))
                )
                shielded = certificate_accepted & physical_ok_ctu
                if bool(shielded.any().detach().cpu().item()):
                    selection_mask = shielded
                else:
                    selection_mask = certificate_accepted
                adjusted_scores = scores

            if method in {"cowp", "cowp_fallback_outcome", "cowp_recursive_viability", "cowp_rvr_pareto_guard", "cowp_successor_option_viability", "cowp_bihorizon_option_viability", "cowp_successor_restore_only", "cowp_trihorizon_option_persistence", "cowp_sov_recovery_commitment", "cowp_sov_dominance_hysteresis", "cowp_recovery_option_spectrum_hysteresis", "cowp_transition_guarded_rosh", "cowp_executable_option_spectrum_hysteresis", "cowp_waymax_kinematic_guarded_rosh", "cowp_control_projected_option_spectrum_hysteresis", "cowp_control_projected_recovery_frontier", "cowp_recourse_returnability_bridge", "cowp_shift_closed_control_reachable_tube", "cowp_conflict_window_control_reachable_tube", "cowp_shift_closed_first_action_viability_interval", "cowp_interaction_aware_reachable_response_envelope", "cowp_blocker_conditioned_interaction_aware_reachable_response_envelope"} and gate_mode in {"priority", "soft"}:
                pcfg_selector = self.cfg.get("planning", {})
                physical_ok = (
                    (action_risk <= float(pcfg_selector.get("candidate_hard_max_action_risk", 0.45)))
                    & (rule_risk <= float(pcfg_selector.get("candidate_hard_max_rule_risk", 0.70)))
                    & (outcome_risk <= float(self.outcome_risk_threshold))
                )
                # Preserve hard semantic feasibility; do not rebuild from generic
                # conventional safety and silently resurrect rejected witnesses.
                frontier_base = certificate_accepted & physical_ok
                if frontier_base.any():
                    pair_risk = transport_risk
                    pair_mix = float(self.cfg.get("planning", {}).get("candidate_pair_risk_mix", 0.20))
                    pressure_mix = float(self.cfg.get("planning", {}).get("candidate_pressure_prior_mix", 0.35))
                    rule_mix = float(self.cfg.get("planning", {}).get("candidate_rule_risk_mix", 1.0))
                    action_mix = float(self.cfg.get("planning", {}).get("candidate_action_risk_mix", 1.5))
                    outcome_mix = float(self.cfg.get("planning", {}).get("candidate_outcome_risk_mix", float(self.outcome_risk_penalty)))
                    shield_tie_mix = float(self.cfg.get("planning", {}).get("candidate_frontier_shield_tie_mix", 0.08))
                    # P-NCF is the primary frontier variable; rule/action/outcome
                    # risks are feasibility shields.  In v4 they were mixed directly
                    # into the frontier with large weights, which improved standard
                    # CR/EP but let the selected candidate become more coercive.
                    transport_pure = bool(pcfg_selector.get("candidate_transport_pure_selector", True))
                    if transport_pure:
                        noncoercive_risk = (
                            transport_risk
                            + float(pcfg_selector.get("candidate_transport_uncertainty_penalty", 0.15)) * transport_uncertainty
                        ).clamp(0.0, 1.0)
                        frontier_ncf_prob = (1.0 - transport_risk).clamp(0.0, 1.0)
                        frontier_false_safe_prob = transport_risk
                    else:
                        noncoercive_risk = cert_decision_risk + pair_mix * pair_risk + pressure_mix * pressure_decision_risk
                        frontier_ncf_prob = cand_ncf_prob
                        frontier_false_safe_prob = cand_false_safe_prob
                    shield_risk = rule_mix * rule_decision_risk + action_mix * action_decision_risk + outcome_mix * outcome_decision_risk
                    frontier_risk = noncoercive_risk + shield_tie_mix * shield_risk
                    frontier_outcome_risk = self.torch.zeros_like(outcome_decision_risk) if method == "cowp_fallback_outcome" else outcome_decision_risk
                    result = select_set_preservation_frontier_1d(
                        scores=scores,
                        base_mask=frontier_base,
                        noncoercive_risk=noncoercive_risk,
                        score_risk=score_decision_risk,
                        progress=candidate_progress,
                        progress_shortfall=progress_shortfall,
                        action_risk=action_decision_risk,
                        rule_risk=rule_decision_risk,
                        outcome_risk=frontier_outcome_risk,
                        ncf_probability=frontier_ncf_prob,
                        false_safe_probability=frontier_false_safe_prob,
                        cfg=pcfg_selector,
                    )
                    frontier = result.frontier
                    if bool(frontier.any().detach().cpu().item()):
                        selection_mask = frontier
                        adjusted_scores = result.adjusted_scores
            # Plan continuity is a tie/selection regularizer only.  It is added
            # after hard feasibility/frontier construction and therefore cannot
            # make a rejected candidate feasible.
            adjusted_scores = adjusted_scores + continuity_weight * continuity_risk
            # Conservative fallback hierarchy: first accepted P-NCF/NCF, then
            # conventionally screened candidates, then any dynamically valid
            # candidate.  If the valid pool itself is empty, selection remains
            # explicitly uncertified and execution uses a bounded current-state
            # smooth stop; zero-padded proposal slots are never executable.
            macro_t = batch["cowp/candidates/macro_type"][0].long()
            stop_ids = self.torch.as_tensor(
                [int(MacroType.STOP_BEFORE_CONFLICT), int(MacroType.YIELD), int(MacroType.CREEP), int(MacroType.NEUTRAL_EGO)],
                device=self.dev,
                dtype=macro_t.dtype,
            )
            stop_like = (macro_t[:, None] == stop_ids[None, :]).any(dim=-1)
            # The candidate-valid branch subsumes every valid stop-like candidate.
            # Do not keep a dead "emergency_stop_like" branch after cand_valid.any().
            fallback_flags = self.torch.stack([
                selection_mask.any(),
                (cand_valid & conventional).any(),
                cand_valid.any(),
            ]).detach().cpu().tolist()
            fallback_transport_ucb = (
                transport_risk
                + float(pcfg_runtime.get("fallback_transport_ucb_scale", pcfg_runtime.get("candidate_transport_ucb_scale", 0.25))) * transport_uncertainty
            ).clamp(0.0, 1.0)
            fallback_score = (
                float(pcfg_runtime.get("fallback_transport_weight", 2.5)) * fallback_transport_ucb
                + float(pcfg_runtime.get("fallback_rule_weight", 1.0)) * rule_decision_risk
                + float(pcfg_runtime.get("fallback_action_weight", 0.75)) * action_decision_risk
                + float(pcfg_runtime.get("fallback_pressure_weight", 0.75)) * pressure_decision_risk
                + float(pcfg_runtime.get("fallback_outcome_weight", 0.50)) * outcome_decision_risk
                + float(pcfg_runtime.get("fallback_utility_weight", 0.05)) * score_decision_risk
                - float(pcfg_runtime.get("fallback_stop_like_bonus", 0.05)) * stop_like.float()
            )
            recovery_base_candidate = -1
            recovery_rvr_candidate = -1
            recovery_switch_applied = False
            recovery_action_target_equal = False
            successor_probe_used = False
            second_successor_probe_used = False
            successor_base_detail = {}
            successor_rvr_detail = {}
            second_successor_base_detail = {}
            second_successor_rvr_detail = {}
            successor_signature_cmp = 0
            second_successor_signature_cmp = 0
            recovery_commitment_active_before = bool(self._recovery_commitment_active) if method == "cowp_sov_recovery_commitment" else False
            recovery_commitment_entered = False
            recovery_commitment_continued = False
            recovery_commitment_cleared = False
            hysteresis_method = method in {"cowp_sov_dominance_hysteresis", "cowp_recovery_option_spectrum_hysteresis", "cowp_transition_guarded_rosh", "cowp_executable_option_spectrum_hysteresis", "cowp_waymax_kinematic_guarded_rosh", "cowp_control_projected_option_spectrum_hysteresis", "cowp_control_projected_recovery_frontier"}
            recovery_hysteresis_active_before = bool(self._recovery_hysteresis_active) if hysteresis_method else False
            recovery_hysteresis_entered = False
            recovery_hysteresis_continued = False
            recovery_hysteresis_exited = False
            recovery_hysteresis_cleared = False
            option_profile_probe_used = False
            option_profile_strict_dominates = False
            option_profile_weak_dominates = False
            option_profile_min_margin = 0
            option_profile_area_delta = 0
            option_profile_base_detail = {}
            option_profile_rvr_detail = {}
            recovery_base_transition_feasible = False
            recovery_rvr_transition_feasible = False
            recovery_transition_delta = 0
            executable_option_profile_used = False
            waymax_kinematic_guard_used = False
            control_projected_option_profile_used = False
            recovery_base_waymax_kinematic_feasible = False
            recovery_rvr_waymax_kinematic_feasible = False
            recovery_base_waymax_inverse_accel = 0.0
            recovery_rvr_waymax_inverse_accel = 0.0
            recovery_base_waymax_steering = 0.0
            recovery_rvr_waymax_steering = 0.0
            recovery_waymax_contract_detail = {}
            recovery_frontier_probe_used = False
            recovery_frontier_representative_count = 0
            recovery_frontier_profiles_evaluated = 0
            recovery_frontier_strict_admissible_count = 0
            recovery_frontier_weak_admissible_count = 0
            recovery_frontier_current_prefix_admissible_count = 0
            recovery_frontier_selected_prefix_delta_steps = 0
            recovery_frontier_active_macro_before = int(self._recovery_frontier_macro) if method == "cowp_control_projected_recovery_frontier" else -1
            recovery_frontier_selected_macro = -1
            recovery_frontier_selected_candidate = -1
            recovery_frontier_selected_is_historical_rvr = False
            recovery_frontier_selected_is_non_rvr = False
            recovery_frontier_selected_fallback_score_delta = 0.0
            recovery_bridge_pending_before = bool(self._recovery_bridge_pending) if method == "cowp_recourse_returnability_bridge" else False
            recovery_bridge_allowed_macros_before = (
                frozenset(self._recovery_bridge_allowed_macros)
                if method == "cowp_recourse_returnability_bridge" else frozenset()
            )
            recovery_bridge_entered = False
            recovery_bridge_direct_entry = False
            recovery_bridge_recourse_executed = False
            recovery_bridge_aborted = False
            recourse_returnability_probe_used = False
            recourse_base_direct_restore = False
            recourse_rvr_direct_restore = False
            recourse_base_macros: frozenset[int] = frozenset()
            recourse_rvr_macros: frozenset[int] = frozenset()
            recourse_returnability_strict_dominates = False
            recourse_returnability_weak_dominates = False
            recourse_base_detail: dict[str, int | float] = {}
            recourse_rvr_detail: dict[str, int | float] = {}
            recourse_direct_restoring_candidate_count = 0
            recourse_bridge_representatives_evaluated = 0
            recourse_bridge_action_classes_available = 0
            recourse_bridge_candidate_pool = 0
            recourse_bridge_selected_macro = -1
            recourse_bridge_minimum_prefix_steps = -1.0
            recourse_current_prefix_nonregressive = False
            recourse_current_action_survives_one_step = False
            recovery_tube_probe_used = False
            recovery_tube_selected = False
            recovery_tube_action_changed = False
            recovery_tube_detail: dict[str, Any] = {}
            recovery_tube_trajectory_override: np.ndarray | None = None
            recovery_tube_target_override: np.ndarray | None = None
            recovery_tube_accel_override: float | None = None
            recovery_tube_selected_fallback_score_delta = 0.0
            if bool(fallback_flags[0]):
                if method == "cowp_sov_recovery_commitment" and self._recovery_commitment_active:
                    self._recovery_commitment_active = False
                    recovery_commitment_cleared = True
                if hysteresis_method and self._recovery_hysteresis_active:
                    self._recovery_hysteresis_active = False
                    recovery_hysteresis_cleared = True
                if method == "cowp_control_projected_recovery_frontier":
                    self._recovery_frontier_macro = -1
                if method == "cowp_recourse_returnability_bridge":
                    self._recovery_bridge_pending = False
                    self._recovery_bridge_allowed_macros = frozenset()
                select_mask = selection_mask
                select_score = adjusted_scores
                fallback_used = False
                fallback_reason = "accepted_ncf" if gate_mode == "hard" else ("accepted_baseline" if gate_mode in {"none", "off"} else "accepted_priority_ncf")
            elif bool(fallback_flags[1]):
                if method == "cowp_sov_recovery_commitment" and self._recovery_commitment_active:
                    # Recovery terminates by *state restoration*, not by a fixed
                    # dwell time: once any full conventional option exists, hand
                    # control back to the unchanged COWP path.
                    self._recovery_commitment_active = False
                    recovery_commitment_cleared = True
                if hysteresis_method and self._recovery_hysteresis_active:
                    self._recovery_hysteresis_active = False
                    recovery_hysteresis_cleared = True
                if method == "cowp_control_projected_recovery_frontier":
                    self._recovery_frontier_macro = -1
                if method == "cowp_recourse_returnability_bridge":
                    self._recovery_bridge_pending = False
                    self._recovery_bridge_allowed_macros = frozenset()
                select_mask = cand_valid & conventional
                select_score = fallback_score
                fallback_used = True
                fallback_reason = "no_certificate_use_least_coercive_conventional"
            elif bool(fallback_flags[2]):
                fallback_used = True
                if method in {"cowp_recursive_viability", "cowp_rvr_pareto_guard", "cowp_successor_option_viability", "cowp_bihorizon_option_viability", "cowp_successor_restore_only", "cowp_trihorizon_option_persistence", "cowp_sov_recovery_commitment", "cowp_sov_dominance_hysteresis", "cowp_recovery_option_spectrum_hysteresis", "cowp_transition_guarded_rosh", "cowp_executable_option_spectrum_hysteresis", "cowp_waymax_kinematic_guarded_rosh", "cowp_control_projected_option_spectrum_hysteresis", "cowp_control_projected_recovery_frontier", "cowp_recourse_returnability_bridge", "cowp_shift_closed_control_reachable_tube", "cowp_conflict_window_control_reachable_tube", "cowp_shift_closed_first_action_viability_interval", "cowp_interaction_aware_reachable_response_envelope", "cowp_blocker_conditioned_interaction_aware_reachable_response_envelope"}:
                    # Historical v16.8.29--35 recovery probes use the same controlled
                    # base-vs-global-RVR pair. V16.8.36 deliberately keeps both as
                    # references but expands the *current* support to one existing-bank
                    # representative per semantic macro.
                    rvr_mask = _recursive_viability_recovery_mask(
                        cand_valid, roadgraph_safe, collision_prefix_steps
                    )
                    recovery_base_candidate = int(self.torch.argmin(
                        self.torch.where(cand_valid, fallback_score, self.torch.full_like(fallback_score, float("inf")))
                    ).item())
                    recovery_rvr_candidate = int(self.torch.argmin(
                        self.torch.where(rvr_mask, fallback_score, self.torch.full_like(fallback_score, float("inf")))
                    ).item())
                    if method == "cowp_recursive_viability":
                        select_mask = rvr_mask
                        select_score = fallback_score
                        recovery_switch_applied = bool(recovery_rvr_candidate != recovery_base_candidate)
                        fallback_reason = "no_conventional_use_recursive_viability"
                    elif method == "cowp_rvr_pareto_guard":
                        recovery_switch_applied = _strict_no_regret_rvr_switch(
                            recovery_base_candidate, recovery_rvr_candidate,
                            collision_prefix_steps, fallback_transport_ucb,
                            rule_decision_risk, action_decision_risk, pressure_decision_risk,
                        )
                        chosen = recovery_rvr_candidate if recovery_switch_applied else recovery_base_candidate
                        select_mask = self.torch.zeros_like(cand_valid)
                        select_mask[chosen] = True
                        select_score = fallback_score
                        fallback_reason = "no_conventional_use_rvr_pareto_guard"
                    elif method in {
                        "cowp_shift_closed_control_reachable_tube",
                        "cowp_conflict_window_control_reachable_tube",
                        "cowp_shift_closed_first_action_viability_interval",
                        "cowp_interaction_aware_reachable_response_envelope",
                        "cowp_blocker_conditioned_interaction_aware_reachable_response_envelope",
                    }:
                        # V38 constructs nominal/all-horizon lower/all-horizon upper
                        # controller tubes. V39 keeps that support nested and adds
                        # causal first/last-conflict envelope-release schedules.
                        # Both methods retain the same full physical certificate and
                        # one-step shift closure.
                        recovery_tube_probe_used = True
                        valid_np = cand_valid.detach().cpu().numpy().astype(bool)
                        road_np = roadgraph_safe.detach().cpu().numpy().astype(bool)
                        prefix_np = collision_prefix_steps.detach().cpu().numpy().astype(np.float64)
                        macro_np_tube = macro_t.detach().cpu().numpy().astype(np.int64)
                        fallback_np = fallback_score.detach().cpu().numpy().astype(np.float64)
                        common_tube_args = (
                            agent_state,
                            int(sdc_index),
                            np.asarray(batch_np["cowp/candidates/trajectory"][0], dtype=np.float32),
                            valid_np,
                            road_np,
                            macro_np_tube,
                            fallback_np,
                            prefix_np,
                            np.asarray(action_targets_np, dtype=np.float32),
                            np.asarray(action_accels_np, dtype=np.float32),
                            roadgraph,
                            self.cfg,
                            float(self._previous_longitudinal_accel),
                        )
                        if method in {
                            "cowp_interaction_aware_reachable_response_envelope",
                            "cowp_blocker_conditioned_interaction_aware_reachable_response_envelope",
                        }:
                            natural_out = pred.get("natural", {})
                            natural_traj_t = natural_out.get("traj") if isinstance(natural_out, dict) else None
                            natural_logits_t = natural_out.get("logits") if isinstance(natural_out, dict) else None
                            if self.torch.is_tensor(natural_traj_t) and self.torch.is_tensor(natural_logits_t):
                                natural_traj_np = natural_traj_t[0].detach().cpu().numpy().astype(np.float32, copy=False)
                                natural_logits_np = natural_logits_t[0].detach().cpu().numpy().astype(np.float32, copy=False)
                            else:
                                natural_traj_np = np.zeros((0, 0, 0, 7), dtype=np.float32)
                                natural_logits_np = np.zeros((0, 0), dtype=np.float32)
                            critical_idx_np = np.asarray(
                                batch_np["cowp/critical/track_index"][0], dtype=np.int64
                            )
                            critical_valid_np = np.asarray(
                                batch_np["cowp/critical/valid"][0], dtype=bool
                            )
                            object_types_np = _extract_object_types_np(state, int(agent_state.shape[0]))
                            if method == "cowp_blocker_conditioned_interaction_aware_reachable_response_envelope":
                                # Query scope is inherited from the frozen causal
                                # collision context, not from an enlarged social
                                # critical set.  Only non-critical nearby actors are
                                # decoded; exact blockers are still identified by the
                                # hard tube certificate inside the constructor.
                                H_query = int(np.asarray(batch_np["cowp/candidates/trajectory"][0]).shape[1])
                                query_context = _prepare_collision_check_context(
                                    agent_state, int(sdc_index), self.cfg,
                                    horizon_steps=H_query, other_future_trajs=None,
                                )
                                original_critical = {
                                    int(x) for x, ok in zip(critical_idx_np.tolist(), critical_valid_np.tolist())
                                    if bool(ok)
                                }
                                model_agent_count = int(np.asarray(batch_np["state/history"]).shape[1])
                                model_agent_valid = np.asarray(
                                    batch_np["state/agent_valid"][0], dtype=bool
                                ).reshape(-1)
                                query_indices = np.asarray([
                                    int(a.get("index", -1))
                                    for a in query_context.get("agents", [])
                                    if 0 <= int(a.get("index", -1)) < model_agent_count
                                    and int(a.get("index", -1)) < model_agent_valid.shape[0]
                                    and bool(model_agent_valid[int(a.get("index", -1))])
                                    and int(a.get("index", -1)) != int(sdc_index)
                                    and int(a.get("index", -1)) not in original_critical
                                ], dtype=np.int64)
                                # Exact de-duplication preserves the collision-context
                                # order and therefore introduces no new ranking rule.
                                if query_indices.size:
                                    _, first_pos = np.unique(query_indices, return_index=True)
                                    query_indices = query_indices[np.sort(first_pos)]
                                # Runtime-fidelity repair: V42 can only be changed by
                                # hypotheses rejected for an unsupported exact collision blocker.
                                # Defer NaturalDecoder work until that frozen V42 trace identifies
                                # the blocker subset; this preserves certificate semantics while
                                # avoiding queries for irrelevant nearby actors.
                                selected_tube, recovery_tube_detail = _construct_blocker_conditioned_interaction_aware_reachable_response_envelope_np(
                                    *common_tube_args,
                                    base_candidate_index=int(recovery_base_candidate),
                                    critical_track_index=critical_idx_np,
                                    critical_valid=critical_valid_np,
                                    natural_trajectories=natural_traj_np,
                                    natural_logits=natural_logits_np,
                                    blocker_query_track_index=query_indices,
                                    blocker_query_trajectories=None,
                                    blocker_query_logits=None,
                                    object_types=object_types_np,
                                    blocker_query_decoder=lambda exact_idx: self._decode_blocker_conditioned_natural_queries_np(
                                        batch, pred, exact_idx,
                                    ),
                                )
                            else:
                                selected_tube, recovery_tube_detail = _construct_interaction_aware_reachable_response_envelope_np(
                                    *common_tube_args,
                                    base_candidate_index=int(recovery_base_candidate),
                                    critical_track_index=critical_idx_np,
                                    critical_valid=critical_valid_np,
                                    natural_trajectories=natural_traj_np,
                                    natural_logits=natural_logits_np,
                                    object_types=object_types_np,
                                )
                        else:
                            if method == "cowp_shift_closed_first_action_viability_interval":
                                tube_constructor = _construct_shift_closed_first_action_viability_interval_np
                            elif method == "cowp_conflict_window_control_reachable_tube":
                                tube_constructor = _construct_conflict_window_control_reachable_tube_np
                            else:
                                tube_constructor = _construct_shift_closed_control_reachable_tube_np
                            selected_tube, recovery_tube_detail = tube_constructor(*common_tube_args)
                        chosen = int(recovery_base_candidate)
                        if selected_tube is not None:
                            chosen = int(selected_tube["parent_index"])
                            recovery_tube_selected = True
                            recovery_tube_trajectory_override = np.asarray(
                                selected_tube["trajectory"], dtype=np.float32
                            )
                            recovery_tube_target_override = np.asarray(
                                selected_tube["target"], dtype=np.float32
                            )
                            recovery_tube_accel_override = float(selected_tube["accel"])
                            base_target = np.asarray(
                                action_targets_np[int(recovery_base_candidate)], dtype=np.float32
                            )
                            recovery_tube_action_changed = bool(
                                not np.allclose(
                                    recovery_tube_target_override,
                                    base_target,
                                    rtol=0.0,
                                    atol=1.0e-6,
                                )
                            )
                            recovery_switch_applied = bool(recovery_tube_action_changed)
                            recovery_tube_selected_fallback_score_delta = float(
                                fallback_np[int(chosen)]
                                - fallback_np[int(recovery_base_candidate)]
                            )
                        select_mask = self.torch.zeros_like(cand_valid)
                        select_mask[int(chosen)] = True
                        select_score = fallback_score
                        if method == "cowp_blocker_conditioned_interaction_aware_reachable_response_envelope":
                            fallback_reason = "no_conventional_use_blocker_conditioned_interaction_aware_reachable_response_envelope"
                        elif method == "cowp_interaction_aware_reachable_response_envelope":
                            fallback_reason = "no_conventional_use_interaction_aware_reachable_response_envelope"
                        elif method == "cowp_shift_closed_first_action_viability_interval":
                            fallback_reason = "no_conventional_use_shift_closed_first_action_viability_interval"
                        elif method == "cowp_conflict_window_control_reachable_tube":
                            fallback_reason = "no_conventional_use_conflict_window_control_reachable_tube"
                        else:
                            fallback_reason = "no_conventional_use_shift_closed_control_reachable_tube"
                    elif method == "cowp_recourse_returnability_bridge":
                        # V16.8.37 follows the preregistered V36 failure branch:
                        # stop enlarging/tuning spectrum selectors and test an
                        # explicit *return to the unchanged full-conventional set*.
                        # The bridge can occupy at most one real replanning edge.
                        chosen = int(recovery_base_candidate)
                        valid_np = cand_valid.detach().cpu().numpy().astype(bool)
                        road_np = roadgraph_safe.detach().cpu().numpy().astype(bool)
                        prefix_np = collision_prefix_steps.detach().cpu().numpy().astype(np.float64)
                        macro_np_bridge = macro_t.detach().cpu().numpy().astype(np.int64)
                        fallback_np = fallback_score.detach().cpu().numpy().astype(np.float64)

                        if bool(self._recovery_bridge_pending):
                            # The actual bridge must be consistent with the semantic
                            # recourse set witnessed at entry and may not reduce the
                            # current causal survival prefix below ordinary COWP.
                            recourse_bridge_minimum_prefix_steps = float(prefix_np[int(recovery_base_candidate)])
                            restore_mask, bridge_detail = _direct_restoring_candidates_np(
                                agent_state, int(sdc_index), roadgraph, self.cfg,
                                valid_np, road_np, prefix_np, macro_np_bridge, fallback_np,
                                np.asarray(action_targets_np, dtype=np.float32),
                                allowed_macros=recovery_bridge_allowed_macros_before,
                                minimum_prefix_steps=recourse_bridge_minimum_prefix_steps,
                            )
                            recourse_direct_restoring_candidate_count = int(np.asarray(restore_mask, dtype=bool).sum())
                            recourse_bridge_representatives_evaluated = int(bridge_detail.get("evaluated", 0))
                            recourse_bridge_action_classes_available = int(bridge_detail.get("action_classes", 0))
                            recourse_bridge_candidate_pool = int(bridge_detail.get("candidate_pool", 0))
                            restoring_idx = np.flatnonzero(restore_mask)
                            if restoring_idx.size:
                                chosen = int(min(
                                    restoring_idx.tolist(),
                                    key=lambda i: (float(fallback_np[int(i)]) if np.isfinite(fallback_np[int(i)]) else float("inf"), int(i)),
                                ))
                                recourse_bridge_selected_macro = int(macro_np_bridge[int(chosen)])
                                if recovery_bridge_allowed_macros_before and recourse_bridge_selected_macro not in recovery_bridge_allowed_macros_before:
                                    raise RuntimeError("Recourse bridge witness-consistency violation: selected macro was not witnessed at entry")
                                recovery_bridge_recourse_executed = True
                                recovery_switch_applied = bool(chosen != int(recovery_base_candidate))
                            else:
                                recovery_bridge_aborted = True
                            # One-replan bridge is structural, not a dwell-time
                            # hyperparameter: it always terminates after this edge.
                            self._recovery_bridge_pending = False
                            self._recovery_bridge_allowed_macros = frozenset()
                        else:
                            bp = int(round(float(prefix_np[int(recovery_base_candidate)])))
                            rp = int(round(float(prefix_np[int(recovery_rvr_candidate)])))
                            recourse_current_prefix_nonregressive = bool(rp >= bp)
                            recourse_current_action_survives_one_step = _returnability_current_edge_admissible(bp, rp)
                            if recovery_rvr_candidate != recovery_base_candidate and recourse_current_action_survives_one_step:
                                bt = np.asarray(action_targets_np[recovery_base_candidate], dtype=np.float32)
                                rt = np.asarray(action_targets_np[recovery_rvr_candidate], dtype=np.float32)
                                recovery_action_target_equal = bool(np.allclose(bt, rt, rtol=0.0, atol=1.0e-6))
                                if not recovery_action_target_equal:
                                    # Preserve the positive V35 signal as a coarse
                                    # pre-gate.  Returnability is only evaluated for
                                    # an RVR branch that already strictly dominates
                                    # under the frozen control-projected observable.
                                    option_profile_probe_used = True
                                    waymax_kinematic_guard_used = True
                                    control_projected_option_profile_used = True
                                    current_pair = np.stack([bt, rt], axis=0)
                                    kim_ok_pair, kim_acc_pair, kim_steer_pair, recovery_waymax_contract_detail = _waymax_kinematic_transition_np(
                                        agent_state[int(sdc_index)], current_pair, self.cfg
                                    )
                                    recovery_base_waymax_kinematic_feasible = bool(kim_ok_pair[0])
                                    recovery_rvr_waymax_kinematic_feasible = bool(kim_ok_pair[1])
                                    recovery_base_waymax_inverse_accel = float(kim_acc_pair[0])
                                    recovery_rvr_waymax_inverse_accel = float(kim_acc_pair[1])
                                    recovery_base_waymax_steering = float(kim_steer_pair[0])
                                    recovery_rvr_waymax_steering = float(kim_steer_pair[1])
                                    base_profile, option_profile_base_detail = _successor_control_projected_option_profile(
                                        agent_state, sdc_index, bt,
                                        float(action_accels_np[recovery_base_candidate]), roadgraph, self.cfg
                                    )
                                    rvr_profile, option_profile_rvr_detail = _successor_control_projected_option_profile(
                                        agent_state, sdc_index, rt,
                                        float(action_accels_np[recovery_rvr_candidate]), roadgraph, self.cfg
                                    )
                                    (
                                        option_profile_strict_dominates,
                                        option_profile_weak_dominates,
                                        recovery_transition_delta,
                                        option_profile_min_margin,
                                        option_profile_area_delta,
                                    ) = _kinematic_guarded_profile_relation(
                                        recovery_rvr_waymax_kinematic_feasible,
                                        recovery_base_waymax_kinematic_feasible,
                                        base_profile, rvr_profile,
                                    )
                                    if option_profile_strict_dominates:
                                        recourse_returnability_probe_used = True
                                        (
                                            recourse_base_direct_restore,
                                            recourse_base_macros,
                                            recourse_base_detail,
                                        ) = _returnability_witness_signature(
                                            agent_state, int(sdc_index), bt,
                                            float(action_accels_np[recovery_base_candidate]),
                                            roadgraph, self.cfg,
                                        )
                                        (
                                            recourse_rvr_direct_restore,
                                            recourse_rvr_macros,
                                            recourse_rvr_detail,
                                        ) = _returnability_witness_signature(
                                            agent_state, int(sdc_index), rt,
                                            float(action_accels_np[recovery_rvr_candidate]),
                                            roadgraph, self.cfg,
                                        )
                                        (
                                            recourse_returnability_strict_dominates,
                                            recourse_returnability_weak_dominates,
                                        ) = _returnability_relation(
                                            recourse_base_direct_restore, recourse_base_macros,
                                            recourse_rvr_direct_restore, recourse_rvr_macros,
                                        )
                                        if recourse_returnability_strict_dominates:
                                            chosen = int(recovery_rvr_candidate)
                                            recovery_bridge_entered = True
                                            recovery_bridge_direct_entry = bool(recourse_rvr_direct_restore)
                                            self._recovery_bridge_pending = bool(not recourse_rvr_direct_restore)
                                            self._recovery_bridge_allowed_macros = (
                                                frozenset(recourse_rvr_macros)
                                                if self._recovery_bridge_pending else frozenset()
                                            )
                                            recovery_switch_applied = bool(chosen != int(recovery_base_candidate))

                        select_mask = self.torch.zeros_like(cand_valid)
                        select_mask[int(chosen)] = True
                        select_score = fallback_score
                        fallback_reason = "no_conventional_use_recourse_returnability_bridge"
                    else:
                        chosen = recovery_base_candidate
                        base_sig = None
                        rvr_sig = None
                        bp = int(round(float(collision_prefix_steps[recovery_base_candidate].detach().cpu().item())))
                        rp = int(round(float(collision_prefix_steps[recovery_rvr_candidate].detach().cpu().item())))

                        if method == "cowp_control_projected_recovery_frontier":
                            # V16.8.36 changes one factor relative to V35 CPOSH:
                            # instead of comparing only base vs one global-RVR
                            # endpoint, expose one RVR-style representative per
                            # semantic macro already present in the fixed bank.
                            # Physical admissibility is still the unchanged V35
                            # control-projected pointwise option-spectrum dominance.
                            recovery_frontier_probe_used = True
                            option_profile_probe_used = True
                            control_projected_option_profile_used = True
                            waymax_kinematic_guard_used = True

                            valid_np = cand_valid.detach().cpu().numpy().astype(bool)
                            road_np = roadgraph_safe.detach().cpu().numpy().astype(bool)
                            prefix_np = collision_prefix_steps.detach().cpu().numpy().astype(np.float64)
                            macro_np_frontier = macro_t.detach().cpu().numpy().astype(np.int64)
                            fallback_np = fallback_score.detach().cpu().numpy().astype(np.float64)
                            reps = _macro_recovery_representatives_np(
                                valid_np, road_np, prefix_np, macro_np_frontier, fallback_np
                            )
                            if recovery_base_candidate not in reps:
                                reps = [int(recovery_base_candidate), *reps]
                            # Preserve deterministic order while removing duplicates.
                            reps = list(dict.fromkeys(int(i) for i in reps))
                            recovery_frontier_representative_count = int(len(reps))

                            base_target = np.asarray(action_targets_np[recovery_base_candidate], dtype=np.float32)
                            base_ok_arr, _base_inv_acc, _base_steer, recovery_waymax_contract_detail = _waymax_kinematic_transition_np(
                                agent_state[int(sdc_index)], base_target, self.cfg
                            )
                            recovery_base_waymax_kinematic_feasible = bool(base_ok_arr[0])
                            base_profile, option_profile_base_detail = _successor_control_projected_option_profile(
                                agent_state, sdc_index, base_target,
                                float(action_accels_np[recovery_base_candidate]), roadgraph, self.cfg
                            )

                            strict_map: dict[int, bool] = {int(recovery_base_candidate): False}
                            weak_map: dict[int, bool] = {int(recovery_base_candidate): True}
                            relation_detail: dict[int, tuple[int, int]] = {}
                            base_prefix_frontier = int(round(float(prefix_np[int(recovery_base_candidate)])))
                            for rep in reps:
                                rep = int(rep)
                                if rep == int(recovery_base_candidate):
                                    continue
                                alt_target = np.asarray(action_targets_np[rep], dtype=np.float32)
                                rep_prefix_frontier = int(round(float(prefix_np[rep])))
                                # The frontier never trades away immediate causal survival
                                # for a richer successor set. This is a hard product-order
                                # component, not a weighted prefix reward.
                                prefix_weak = bool(rep_prefix_frontier >= base_prefix_frontier)
                                prefix_strict = bool(rep_prefix_frontier > base_prefix_frontier)
                                if prefix_weak:
                                    recovery_frontier_current_prefix_admissible_count += 1
                                else:
                                    # Hard current-survival pre-gate: avoid the expensive
                                    # successor projection when the representative already
                                    # loses against base at H0.
                                    strict_map[rep] = False
                                    weak_map[rep] = False
                                    relation_detail[rep] = (int(rep_prefix_frontier - base_prefix_frontier), 0)
                                    continue
                                # Exact emitted-action equality cannot create a new
                                # physical branch.  It is weakly equivalent for
                                # continuation but never a strict entry witness.
                                if np.allclose(base_target, alt_target, rtol=0.0, atol=1.0e-6):
                                    strict_map[rep] = bool(prefix_strict)
                                    weak_map[rep] = bool(prefix_weak)
                                    relation_detail[rep] = (0, 0)
                                    continue
                                alt_ok_arr, _alt_inv_acc, _alt_steer, _ = _waymax_kinematic_transition_np(
                                    agent_state[int(sdc_index)], alt_target, self.cfg
                                )
                                alt_profile, alt_detail = _successor_control_projected_option_profile(
                                    agent_state, sdc_index, alt_target,
                                    float(action_accels_np[rep]), roadgraph, self.cfg
                                )
                                recovery_frontier_profiles_evaluated += 1
                                strict_i, weak_i, _td, min_margin_i, area_i = _kinematic_guarded_profile_relation(
                                    bool(alt_ok_arr[0]), bool(recovery_base_waymax_kinematic_feasible),
                                    base_profile, alt_profile,
                                )
                                # Product order: immediate current safe-prefix and
                                # successor control-projected option spectrum must both
                                # be non-regressive; at least one component must improve.
                                strict_map[rep] = bool(prefix_weak and weak_i and (prefix_strict or strict_i))
                                weak_map[rep] = bool(prefix_weak and weak_i)
                                relation_detail[rep] = (int(min_margin_i), int(area_i))

                            recovery_frontier_strict_admissible_count = int(sum(bool(strict_map.get(i, False)) for i in reps if i != recovery_base_candidate))
                            recovery_frontier_weak_admissible_count = int(sum(bool(weak_map.get(i, False)) for i in reps if i != recovery_base_candidate))
                            chosen, new_macro, entered, continued, exited = _recovery_frontier_mode_choice_np(
                                recovery_base_candidate, reps, macro_np_frontier, fallback_np,
                                strict_map, weak_map, int(self._recovery_frontier_macro),
                            )
                            self._recovery_frontier_macro = int(new_macro)
                            self._recovery_hysteresis_active = bool(new_macro >= 0)
                            recovery_hysteresis_entered = bool(entered)
                            recovery_hysteresis_continued = bool(continued)
                            recovery_hysteresis_exited = bool(exited)
                            recovery_switch_applied = bool(int(chosen) != int(recovery_base_candidate))
                            recovery_frontier_selected_candidate = int(chosen)
                            recovery_frontier_selected_macro = int(macro_np_frontier[int(chosen)]) if 0 <= int(chosen) < len(macro_np_frontier) else -1
                            recovery_frontier_selected_is_historical_rvr = bool(int(chosen) == int(recovery_rvr_candidate))
                            recovery_frontier_selected_is_non_rvr = bool(recovery_switch_applied and int(chosen) != int(recovery_rvr_candidate))
                            recovery_frontier_selected_fallback_score_delta = float(fallback_np[int(chosen)] - fallback_np[int(recovery_base_candidate)]) if 0 <= int(chosen) < len(fallback_np) else 0.0
                            recovery_frontier_selected_prefix_delta_steps = int(round(float(prefix_np[int(chosen)] - prefix_np[int(recovery_base_candidate)]))) if 0 <= int(chosen) < len(prefix_np) else 0
                            if int(chosen) in relation_detail:
                                option_profile_min_margin, option_profile_area_delta = relation_detail[int(chosen)]
                                option_profile_strict_dominates = bool(strict_map.get(int(chosen), False))
                                option_profile_weak_dominates = bool(weak_map.get(int(chosen), False))
                            chosen = int(chosen)
                        else:
                            # v16.8.32 commitment diagnostic: strict SOV is an
                            # entry trigger, followed by unconditional RVR until state
                            # restoration.  v16.8.33 keeps this historical branch intact
                            # as a reference, but adds a parameter-free dominance
                            # hysteresis to separate chattering from over-commitment.
                            commitment_already_active = bool(
                                method == "cowp_sov_recovery_commitment"
                                and self._recovery_commitment_active
                            )
                            hysteresis_already_active = bool(
                                hysteresis_method and self._recovery_hysteresis_active
                            )
                            if commitment_already_active:
                                recovery_switch_applied = bool(recovery_rvr_candidate != recovery_base_candidate)
                                recovery_commitment_continued = True
                                chosen = recovery_rvr_candidate
                            elif hysteresis_already_active and recovery_rvr_candidate == recovery_base_candidate:
                                # Exact candidate equality cannot create a hybrid action;
                                # keep the mode through the tie without an unnecessary
                                # counterfactual probe.
                                recovery_hysteresis_continued = True
                                chosen = recovery_rvr_candidate
                            elif recovery_rvr_candidate != recovery_base_candidate:
                                bt = np.asarray(action_targets_np[recovery_base_candidate], dtype=np.float32)
                                rt = np.asarray(action_targets_np[recovery_rvr_candidate], dtype=np.float32)
                                recovery_action_target_equal = bool(np.allclose(bt, rt, rtol=0.0, atol=1.0e-6))
                                if recovery_action_target_equal:
                                    if hysteresis_already_active:
                                        recovery_hysteresis_continued = True
                                        chosen = recovery_rvr_candidate
                                elif method in {
                                    "cowp_waymax_kinematic_guarded_rosh",
                                    "cowp_control_projected_option_spectrum_hysteresis",
                                }:
                                    # v16.8.35 fixes the representation mismatch exposed
                                    # by v16.8.34.  The current-action guard now mirrors
                                    # Waymax's evaluated inverse acceleration/steering-
                                    # curvature contract.  The main branch additionally
                                    # builds the successor spectrum from trajectories
                                    # repeatedly projected through the same stateful
                                    # controller that emits online actions.
                                    option_profile_probe_used = True
                                    waymax_kinematic_guard_used = True
                                    current_pair = np.stack([bt, rt], axis=0)
                                    (
                                        kim_ok_pair, kim_acc_pair, kim_steer_pair,
                                        recovery_waymax_contract_detail,
                                    ) = _waymax_kinematic_transition_np(
                                        agent_state[int(sdc_index)], current_pair, self.cfg
                                    )
                                    recovery_base_waymax_kinematic_feasible = bool(kim_ok_pair[0])
                                    recovery_rvr_waymax_kinematic_feasible = bool(kim_ok_pair[1])
                                    recovery_base_waymax_inverse_accel = float(kim_acc_pair[0])
                                    recovery_rvr_waymax_inverse_accel = float(kim_acc_pair[1])
                                    recovery_base_waymax_steering = float(kim_steer_pair[0])
                                    recovery_rvr_waymax_steering = float(kim_steer_pair[1])

                                    if method == "cowp_control_projected_option_spectrum_hysteresis":
                                        control_projected_option_profile_used = True
                                        base_profile, option_profile_base_detail = _successor_control_projected_option_profile(
                                            agent_state, sdc_index, bt,
                                            float(action_accels_np[recovery_base_candidate]),
                                            roadgraph, self.cfg
                                        )
                                        rvr_profile, option_profile_rvr_detail = _successor_control_projected_option_profile(
                                            agent_state, sdc_index, rt,
                                            float(action_accels_np[recovery_rvr_candidate]),
                                            roadgraph, self.cfg
                                        )
                                    else:
                                        base_profile, option_profile_base_detail = _successor_recovery_option_profile(
                                            agent_state, sdc_index, bt, roadgraph, self.cfg
                                        )
                                        rvr_profile, option_profile_rvr_detail = _successor_recovery_option_profile(
                                            agent_state, sdc_index, rt, roadgraph, self.cfg
                                        )
                                    (
                                        option_profile_strict_dominates,
                                        option_profile_weak_dominates,
                                        recovery_transition_delta,
                                        option_profile_min_margin,
                                        option_profile_area_delta,
                                    ) = _kinematic_guarded_profile_relation(
                                        recovery_rvr_waymax_kinematic_feasible,
                                        recovery_base_waymax_kinematic_feasible,
                                        base_profile, rvr_profile,
                                    )
                                    (active_after, entered, continued, exited) = _dominance_hysteresis_transition(
                                        hysteresis_already_active,
                                        strict_alt_dominates=option_profile_strict_dominates,
                                        weak_alt_dominates=option_profile_weak_dominates,
                                    )
                                    self._recovery_hysteresis_active = bool(active_after)
                                    recovery_hysteresis_entered = bool(entered)
                                    recovery_hysteresis_continued = bool(continued)
                                    recovery_hysteresis_exited = bool(exited)
                                    recovery_switch_applied = bool(active_after)
                                    chosen = recovery_rvr_candidate if active_after else recovery_base_candidate
                                elif method in {
                                    "cowp_recovery_option_spectrum_hysteresis",
                                    "cowp_transition_guarded_rosh",
                                    "cowp_executable_option_spectrum_hysteresis",
                                }:
                                    # v16.8.33 ROSH keeps the nominal semantic option
                                    # spectrum.  v16.8.34 adds two controlled probes:
                                    #   TG-ROSH: same future spectrum + current hard
                                    #            controller-transition non-regression;
                                    #   EOSH:    additionally counts only successor
                                    #            options executable from the carried
                                    #            controller acceleration state.
                                    option_profile_probe_used = True
                                    recovery_base_transition_feasible = bool(
                                        controller_transition_feasible_np[recovery_base_candidate]
                                    )
                                    recovery_rvr_transition_feasible = bool(
                                        controller_transition_feasible_np[recovery_rvr_candidate]
                                    )
                                    if method == "cowp_executable_option_spectrum_hysteresis":
                                        executable_option_profile_used = True
                                        base_profile, option_profile_base_detail = _successor_executable_recovery_option_profile(
                                            agent_state, sdc_index, bt,
                                            float(action_accels_np[recovery_base_candidate]),
                                            roadgraph, self.cfg
                                        )
                                        rvr_profile, option_profile_rvr_detail = _successor_executable_recovery_option_profile(
                                            agent_state, sdc_index, rt,
                                            float(action_accels_np[recovery_rvr_candidate]),
                                            roadgraph, self.cfg
                                        )
                                    else:
                                        base_profile, option_profile_base_detail = _successor_recovery_option_profile(
                                            agent_state, sdc_index, bt, roadgraph, self.cfg
                                        )
                                        rvr_profile, option_profile_rvr_detail = _successor_recovery_option_profile(
                                            agent_state, sdc_index, rt, roadgraph, self.cfg
                                        )

                                    if method == "cowp_recovery_option_spectrum_hysteresis":
                                        (
                                            option_profile_strict_dominates,
                                            option_profile_weak_dominates,
                                            option_profile_min_margin,
                                            option_profile_area_delta,
                                        ) = _option_profile_relation(base_profile, rvr_profile)
                                    else:
                                        (
                                            option_profile_strict_dominates,
                                            option_profile_weak_dominates,
                                            recovery_transition_delta,
                                            option_profile_min_margin,
                                            option_profile_area_delta,
                                        ) = _execution_spectrum_relation(
                                            recovery_base_transition_feasible,
                                            recovery_rvr_transition_feasible,
                                            base_profile,
                                            rvr_profile,
                                        )
                                    (
                                        active_after, entered, continued, exited,
                                    ) = _dominance_hysteresis_transition(
                                        hysteresis_already_active,
                                        strict_alt_dominates=option_profile_strict_dominates,
                                        weak_alt_dominates=option_profile_weak_dominates,
                                    )
                                    self._recovery_hysteresis_active = bool(active_after)
                                    recovery_hysteresis_entered = bool(entered)
                                    recovery_hysteresis_continued = bool(continued)
                                    recovery_hysteresis_exited = bool(exited)
                                    recovery_switch_applied = bool(active_after)
                                    chosen = recovery_rvr_candidate if active_after else recovery_base_candidate
                                else:
                                    # Legacy successor signature is still evaluated for
                                    # historical SOV/BHOV/THOP/commitment and for the
                                    # v16.8.33 state-machine-only diagnostic.
                                    successor_probe_used = True
                                    base_sig, successor_base_detail = _successor_option_signature(
                                        agent_state, sdc_index, bt, roadgraph, self.cfg
                                    )
                                    rvr_sig, successor_rvr_detail = _successor_option_signature(
                                        agent_state, sdc_index, rt, roadgraph, self.cfg
                                    )
                                    successor_signature_cmp = 1 if rvr_sig > base_sig else (-1 if rvr_sig < base_sig else 0)
                                    if method == "cowp_successor_option_viability":
                                        recovery_switch_applied = bool(rvr_sig > base_sig)
                                    elif method == "cowp_bihorizon_option_viability":
                                        recovery_switch_applied = _bihorizon_option_dominates(
                                            base_sig, rvr_sig, bp, rp
                                        )
                                    elif method == "cowp_successor_restore_only":
                                        recovery_switch_applied = _successor_restoration_dominates(
                                            successor_base_detail, successor_rvr_detail
                                        )
                                    elif method == "cowp_trihorizon_option_persistence":
                                        # Historical v16.8.32 branch kept unchanged for
                                        # reference/regression purposes.
                                        bhov_pre = _bihorizon_option_dominates(
                                            base_sig, rvr_sig, bp, rp
                                        )
                                        if bhov_pre:
                                            second_successor_probe_used = True
                                            traj_np = np.asarray(batch_np["cowp/candidates/trajectory"][0])
                                            base_sig2, second_successor_base_detail = _second_successor_option_signature(
                                                agent_state, sdc_index, bt,
                                                float(action_accels_np[recovery_base_candidate]),
                                                traj_np[recovery_base_candidate], roadgraph, self.cfg
                                            )
                                            rvr_sig2, second_successor_rvr_detail = _second_successor_option_signature(
                                                agent_state, sdc_index, rt,
                                                float(action_accels_np[recovery_rvr_candidate]),
                                                traj_np[recovery_rvr_candidate], roadgraph, self.cfg
                                            )
                                            second_successor_signature_cmp = 1 if rvr_sig2 > base_sig2 else (-1 if rvr_sig2 < base_sig2 else 0)
                                            recovery_switch_applied = _trihorizon_option_persistence_dominates(
                                                base_sig, rvr_sig, base_sig2, rvr_sig2, bp, rp
                                            )
                                    elif method == "cowp_sov_dominance_hysteresis":
                                        strict_dom = bool(rvr_sig > base_sig)
                                        weak_dom = bool(rvr_sig >= base_sig)
                                        (
                                            active_after, entered, continued, exited,
                                        ) = _dominance_hysteresis_transition(
                                            hysteresis_already_active,
                                            strict_alt_dominates=strict_dom,
                                            weak_alt_dominates=weak_dom,
                                        )
                                        self._recovery_hysteresis_active = bool(active_after)
                                        recovery_hysteresis_entered = bool(entered)
                                        recovery_hysteresis_continued = bool(continued)
                                        recovery_hysteresis_exited = bool(exited)
                                        recovery_switch_applied = bool(active_after)
                                        chosen = recovery_rvr_candidate if active_after else recovery_base_candidate
                                    else:
                                        # Historical SOV-triggered unconditional
                                        # commitment entry.
                                        recovery_switch_applied = bool(rvr_sig > base_sig)
                                        if recovery_switch_applied:
                                            self._recovery_commitment_active = True
                                            recovery_commitment_entered = True
                                    if recovery_switch_applied and method not in {
                                        "cowp_sov_dominance_hysteresis",
                                        "cowp_recovery_option_spectrum_hysteresis",
                                        "cowp_transition_guarded_rosh",
                                        "cowp_executable_option_spectrum_hysteresis",
                                        "cowp_waymax_kinematic_guarded_rosh",
                                        "cowp_control_projected_option_spectrum_hysteresis",
                                        "cowp_control_projected_recovery_frontier",
                                    }:
                                        chosen = recovery_rvr_candidate
                        select_mask = self.torch.zeros_like(cand_valid)
                        select_mask[chosen] = True
                        select_score = fallback_score
                        if method == "cowp_successor_option_viability":
                            fallback_reason = "no_conventional_use_successor_option_viability"
                        elif method == "cowp_bihorizon_option_viability":
                            fallback_reason = "no_conventional_use_bihorizon_option_viability"
                        elif method == "cowp_successor_restore_only":
                            fallback_reason = "no_conventional_use_successor_restore_only"
                        elif method == "cowp_trihorizon_option_persistence":
                            fallback_reason = "no_conventional_use_trihorizon_option_persistence"
                        elif method == "cowp_sov_recovery_commitment":
                            fallback_reason = "no_conventional_use_sov_recovery_commitment"
                        elif method == "cowp_sov_dominance_hysteresis":
                            fallback_reason = "no_conventional_use_sov_dominance_hysteresis"
                        elif method == "cowp_transition_guarded_rosh":
                            fallback_reason = "no_conventional_use_transition_guarded_rosh"
                        elif method == "cowp_executable_option_spectrum_hysteresis":
                            fallback_reason = "no_conventional_use_executable_option_spectrum_hysteresis"
                        elif method == "cowp_waymax_kinematic_guarded_rosh":
                            fallback_reason = "no_conventional_use_waymax_kinematic_guarded_rosh"
                        elif method == "cowp_control_projected_option_spectrum_hysteresis":
                            fallback_reason = "no_conventional_use_control_projected_option_spectrum_hysteresis"
                        elif method == "cowp_control_projected_recovery_frontier":
                            fallback_reason = "no_conventional_use_control_projected_recovery_frontier"
                        else:
                            fallback_reason = "no_conventional_use_recovery_option_spectrum_hysteresis"
                else:
                    select_mask = cand_valid
                    select_score = fallback_score
                    fallback_reason = "no_conventional_use_least_coercive_valid"
            else:
                # No selectable candidate exists.  Selection diagnostics remain
                # explicitly uncertified; execution is resolved later from the
                # current ego state and must never consume a zero-padded slot.
                if method == "cowp_sov_recovery_commitment" and self._recovery_commitment_active:
                    self._recovery_commitment_active = False
                    recovery_commitment_cleared = True
                if hysteresis_method and self._recovery_hysteresis_active:
                    self._recovery_hysteresis_active = False
                    recovery_hysteresis_cleared = True
                if method == "cowp_control_projected_recovery_frontier":
                    self._recovery_frontier_macro = -1
                if method == "cowp_recourse_returnability_bridge":
                    self._recovery_bridge_pending = False
                    self._recovery_bridge_allowed_macros = frozenset()
                select_mask = cand_valid
                select_score = fallback_score
                fallback_used = True
                fallback_reason = "no_valid_candidate"
            selected = int(self.torch.argmin(self.torch.where(select_mask, select_score, self.torch.full_like(select_score, float("inf")))).item()) if has_valid else 0
            selected_report = int(selected) if has_valid else -1
            selected_macro_type = int(macro_t[selected].detach().cpu().item()) if has_valid else int(MacroType.NEUTRAL_EGO)
            selected_macro_name = _macro_name(selected_macro_type) if has_valid else "EMERGENCY_BOUNDED_STOP"
            selected_candidate_valid = bool(cand_valid[selected].detach().cpu().item()) if has_valid else False
            selected_candidate_conventional_safe = bool(conventional[selected].detach().cpu().item()) if has_valid else False
            selected_waymax_kinematic_feasible = False
            selected_waymax_inverse_accel = 0.0
            selected_waymax_steering = 0.0
            if has_valid and method in {
                "cowp_waymax_kinematic_guarded_rosh",
                "cowp_control_projected_option_spectrum_hysteresis",
                "cowp_control_projected_recovery_frontier",
                "cowp_recourse_returnability_bridge",
                "cowp_shift_closed_control_reachable_tube",
                "cowp_conflict_window_control_reachable_tube",
                "cowp_shift_closed_first_action_viability_interval",
                "cowp_interaction_aware_reachable_response_envelope",
                "cowp_blocker_conditioned_interaction_aware_reachable_response_envelope",
            }:
                selected_contract_target = (
                    np.asarray(recovery_tube_target_override, dtype=np.float32)
                    if recovery_tube_target_override is not None
                    else np.asarray(action_targets_np[selected], dtype=np.float32)
                )
                _sel_ok, _sel_acc, _sel_steer, _sel_contract = _waymax_kinematic_transition_np(
                    agent_state[int(sdc_index)], selected_contract_target, self.cfg
                )
                selected_waymax_kinematic_feasible = bool(_sel_ok[0])
                selected_waymax_inverse_accel = float(_sel_acc[0])
                selected_waymax_steering = float(_sel_steer[0])
            if fallback_reason == "no_certificate_use_least_coercive_conventional" and not selected_candidate_conventional_safe:
                raise RuntimeError(
                    "Conventional fallback integrity violation: selected candidate did not pass the "
                    "online conventional-safety audit. This indicates candidate-pool semantic corruption."
                )
            selected_witness = witness[selected] if has_valid else witness[:0]
            selected_opr = opr[selected] if has_valid else opr[:0]
            has_crit = bool(crit_mask.any().detach().cpu().item())
            zero = scores.new_tensor(0.0)
            one = scores.new_tensor(1.0)
            frontier_count = frontier.sum().float() if "frontier" in locals() and self.torch.is_tensor(frontier) else scores.new_tensor(-1.0)
            valid_cert = candidate_cert_risk[cand_valid] if has_valid else candidate_cert_risk[:0]
            valid_pressure = pressure_prior[cand_valid] if has_valid else pressure_prior[:0]
            valid_rule = rule_risk[cand_valid] if has_valid else rule_risk[:0]
            valid_action = action_risk[cand_valid] if has_valid else action_risk[:0]
            bsel = self.torch.nan_to_num(burden[0, selected].float(), nan=0.0, posinf=2.0, neginf=0.0) if burden is not None and has_valid else None
            csel = self.torch.nan_to_num(c_i[0, selected].float(), nan=0.0, posinf=2.0, neginf=0.0) if c_i is not None and has_valid else None
            diagnostic_tensors = [
                certificate_accepted.sum().float(),
                selection_mask.sum().float(),
                cand_valid.sum().float(),
                (cand_valid & conventional).sum().float(),
                crit_mask.sum().float(),
                batch["map/conflict_region_valid"][0].bool().sum().float(),
                selected_witness.max() if selected_witness.numel() else zero,
                selected_witness.mean() if selected_witness.numel() else zero,
                (uncertainty[selected].mean() if uncertainty[selected].numel() else zero) if has_valid else zero,
                (witness_cert[selected].max() if witness_cert[selected].numel() else zero) if has_valid else zero,
                selected_opr.min() if selected_opr.numel() else one,
                selected_opr.mean() if selected_opr.numel() else one,
                scores[selected] if has_valid else zero,
                primary_bad.sum().float() if primary_bad.numel() else zero,
                severe_bad.sum().float() if severe_bad.numel() else zero,
                option_bad.sum().float() if option_bad.numel() else zero,
                (priority[selected].max() if priority.numel() else zero) if has_valid else zero,
                (priority[selected].mean() if priority.numel() else zero) if has_valid else zero,
                (outcome_risk[selected] if outcome_risk.numel() else zero) if has_valid else zero,
                (outcome_decision_risk[selected] if outcome_decision_risk.numel() else zero) if has_valid else zero,
                (cand_ncf_prob[selected] if cand_ncf_prob.numel() else zero) if has_valid else zero,
                (cand_false_safe_prob[selected] if cand_false_safe_prob.numel() else zero) if has_valid else zero,
                (cand_quality_prob[selected] if cand_quality_prob.numel() else zero) if has_valid else zero,
                (candidate_cert_risk[selected] if candidate_cert_risk.numel() else zero) if has_valid else zero,
                (pressure_prior[selected] if pressure_prior.numel() else zero) if has_valid else zero,
                (rule_risk[selected] if rule_risk.numel() else zero) if has_valid else zero,
                (action_risk[selected] if action_risk.numel() else zero) if has_valid else zero,
                valid_cert.min() if valid_cert.numel() else zero,
                valid_cert.mean() if valid_cert.numel() else zero,
                valid_pressure.mean() if valid_pressure.numel() else zero,
                valid_rule.mean() if valid_rule.numel() else zero,
                valid_action.mean() if valid_action.numel() else zero,
                bsel[crit_mask].max() if bsel is not None and has_crit else zero,
                csel[crit_mask].max() if csel is not None and has_crit else zero,
                (cand_valid & roadgraph_safe).sum().float(),
                (cand_valid & collision_safe).sum().float(),
                collision_prefix_steps[cand_valid].max() if has_valid else zero,
                collision_prefix_steps[selected] if has_valid else zero,
                roadgraph_safe[selected].float() if has_valid else zero,
                collision_safe[selected].float() if has_valid else zero,
                collision_margin[selected] if has_valid else zero,
            ]
            host = self.torch.stack([x.float() for x in diagnostic_tensors]).detach().cpu().tolist()
            conv_n, road_n, coll_n = int(host[3]), int(host[34]), int(host[35])
            if conv_n > 0:
                zero_conv_reason = "none"
            elif road_n == 0 and coll_n == 0:
                zero_conv_reason = "road_and_collision_empty"
            elif road_n == 0:
                zero_conv_reason = "roadgraph_empty"
            elif coll_n == 0:
                zero_conv_reason = "collision_empty"
            else:
                zero_conv_reason = "intersection_empty"
            diag = {
                "scenario_index": int(scenario_index) if scenario_index is not None else -1,
                "step": int(step) if step is not None else -1,
                "selected_candidate": int(selected_report),
                "selected_macro_type": int(selected_macro_type),
                "selected_macro_name": str(selected_macro_name),
                "selected_candidate_valid": bool(selected_candidate_valid),
                "selected_candidate_conventional_safe": bool(selected_candidate_conventional_safe),
                "certificate_accepted_candidates": int(host[0]),
                "accepted_candidates": int(host[0]),
                "frontier_candidates": int(host[1]),
                "valid_candidates": int(host[2]),
                "conventional_candidates": int(host[3]),
                "roadgraph_safe_candidates": int(host[34]),
                "collision_safe_candidates": int(host[35]),
                "max_collision_safe_prefix_steps": int(round(host[36])),
                "selected_collision_safe_prefix_steps": int(round(host[37])),
                "selected_candidate_roadgraph_safe": bool(host[38] > 0.5),
                "selected_candidate_collision_safe": bool(host[39] > 0.5),
                "selected_collision_min_clearance_margin_m": float(host[40]),
                "zero_conventional_reason": str(zero_conv_reason),
                "critical_agents": int(host[4]),
                "conflict_tokens": int(host[5]),
                "fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason,
                "max_witness_prob": float(host[6]),
                "mean_witness_prob": float(host[7]),
                "mean_witness_uncertainty": float(host[8]),
                "max_witness_certificate": float(host[9]),
                "min_opr": float(host[10]),
                "mean_opr": float(host[11]),
                "score": float(host[12]),
                "witness_threshold": float(self.witness_threshold),
                "alpha_opr": float(alpha),
                "gate_mode": str(gate_mode),
                "method": str(method),
                "priority_hard_threshold": float(self.priority_hard_threshold),
                "accepted_primary_bad_candidates": int(host[13]),
                "severe_bad_candidates": int(host[14]),
                "option_bad_candidates": int(host[15]),
                "selected_priority_max": float(host[16]),
                "selected_priority_mean": float(host[17]),
                "selected_outcome_risk": float(host[18]),
                "selected_outcome_decision_risk": float(host[19]),
                "selected_candidate_ncf_prob": float(host[20]),
                "selected_candidate_false_safe_prob": float(host[21]),
                "selected_candidate_quality_prob": float(host[22]),
                "selected_candidate_cert_risk": float(host[23]),
                "selected_candidate_pressure_prior": float(host[24]),
                "selected_candidate_rule_risk": float(host[25]),
                "selected_candidate_action_risk": float(host[26]),
                "min_candidate_cert_risk": float(host[27]),
                "mean_candidate_cert_risk": float(host[28]),
                "mean_candidate_pressure_prior": float(host[29]),
                "mean_candidate_rule_risk": float(host[30]),
                "mean_candidate_action_risk": float(host[31]),
                "beta_threshold": float(self.cfg.get("burden", {}).get("beta0_vehicle", 0.65)),
            }
            diag.update({
                "recovery_base_candidate": int(recovery_base_candidate),
                "recovery_rvr_candidate": int(recovery_rvr_candidate),
                "recovery_switch_applied": bool(recovery_switch_applied),
                "recovery_action_target_equal": bool(recovery_action_target_equal),
                "successor_option_probe_used": bool(successor_probe_used),
                "successor_signature_compare": int(successor_signature_cmp),
                "successor_base_conventional_exists": int(successor_base_detail.get("conventional_exists", -1)),
                "successor_base_conventional_macro_types": int(successor_base_detail.get("conventional_macro_types", -1)),
                "successor_base_conventional_candidates": int(successor_base_detail.get("conventional_candidates", -1)),
                "successor_base_max_collision_safe_prefix_steps": int(successor_base_detail.get("max_collision_safe_prefix_steps", -1)),
                "successor_rvr_conventional_exists": int(successor_rvr_detail.get("conventional_exists", -1)),
                "successor_rvr_conventional_macro_types": int(successor_rvr_detail.get("conventional_macro_types", -1)),
                "successor_rvr_conventional_candidates": int(successor_rvr_detail.get("conventional_candidates", -1)),
                "successor_rvr_max_collision_safe_prefix_steps": int(successor_rvr_detail.get("max_collision_safe_prefix_steps", -1)),
                "second_successor_option_probe_used": bool(second_successor_probe_used),
                "second_successor_signature_compare": int(second_successor_signature_cmp),
                "second_successor_base_conventional_exists": int(second_successor_base_detail.get("conventional_exists", -1)),
                "second_successor_base_conventional_macro_types": int(second_successor_base_detail.get("conventional_macro_types", -1)),
                "second_successor_base_conventional_candidates": int(second_successor_base_detail.get("conventional_candidates", -1)),
                "second_successor_base_max_collision_safe_prefix_steps": int(second_successor_base_detail.get("max_collision_safe_prefix_steps", -1)),
                "second_successor_rvr_conventional_exists": int(second_successor_rvr_detail.get("conventional_exists", -1)),
                "second_successor_rvr_conventional_macro_types": int(second_successor_rvr_detail.get("conventional_macro_types", -1)),
                "second_successor_rvr_conventional_candidates": int(second_successor_rvr_detail.get("conventional_candidates", -1)),
                "second_successor_rvr_max_collision_safe_prefix_steps": int(second_successor_rvr_detail.get("max_collision_safe_prefix_steps", -1)),
                "recovery_commitment_active_before": bool(recovery_commitment_active_before),
                "recovery_commitment_active_after": bool(self._recovery_commitment_active) if method == "cowp_sov_recovery_commitment" else False,
                "recovery_commitment_entered": bool(recovery_commitment_entered),
                "recovery_commitment_continued": bool(recovery_commitment_continued),
                "recovery_commitment_cleared": bool(recovery_commitment_cleared),
                "recovery_hysteresis_active_before": bool(recovery_hysteresis_active_before),
                "recovery_hysteresis_active_after": bool(self._recovery_hysteresis_active) if hysteresis_method else False,
                "recovery_hysteresis_entered": bool(recovery_hysteresis_entered),
                "recovery_hysteresis_continued": bool(recovery_hysteresis_continued),
                "recovery_hysteresis_exited": bool(recovery_hysteresis_exited),
                "recovery_hysteresis_cleared": bool(recovery_hysteresis_cleared),
                "recovery_option_profile_probe_used": bool(option_profile_probe_used),
                "recovery_option_profile_strict_dominates": bool(option_profile_strict_dominates),
                "recovery_option_profile_weak_dominates": bool(option_profile_weak_dominates),
                "recovery_option_profile_min_margin": int(option_profile_min_margin),
                "recovery_option_profile_area_delta": int(option_profile_area_delta),
                "recovery_option_profile_base_h1_macros": int(option_profile_base_detail.get("recovery_macro_types_h1", -1)),
                "recovery_option_profile_rvr_h1_macros": int(option_profile_rvr_detail.get("recovery_macro_types_h1", -1)),
                "recovery_option_profile_base_full_horizon_macros": int(option_profile_base_detail.get("recovery_macro_types_full_horizon", -1)),
                "recovery_option_profile_rvr_full_horizon_macros": int(option_profile_rvr_detail.get("recovery_macro_types_full_horizon", -1)),
                "recovery_option_profile_base_area": int(option_profile_base_detail.get("recovery_profile_area", -1)),
                "recovery_option_profile_rvr_area": int(option_profile_rvr_detail.get("recovery_profile_area", -1)),
                "recovery_option_profile_base_valid_candidates": int(option_profile_base_detail.get("valid_candidates", -1)),
                "recovery_option_profile_rvr_valid_candidates": int(option_profile_rvr_detail.get("valid_candidates", -1)),
                "recovery_base_controller_transition_feasible": bool(recovery_base_transition_feasible),
                "recovery_rvr_controller_transition_feasible": bool(recovery_rvr_transition_feasible),
                "recovery_controller_transition_delta": int(recovery_transition_delta),
                "recovery_executable_option_profile_used": bool(executable_option_profile_used),
                "recovery_option_profile_base_transition_feasible_candidates": int(option_profile_base_detail.get("controller_transition_feasible_candidates", -1)),
                "recovery_option_profile_rvr_transition_feasible_candidates": int(option_profile_rvr_detail.get("controller_transition_feasible_candidates", -1)),
                "recovery_option_profile_base_executable_roadgraph_candidates": int(option_profile_base_detail.get("executable_roadgraph_candidates", -1)),
                "recovery_option_profile_rvr_executable_roadgraph_candidates": int(option_profile_rvr_detail.get("executable_roadgraph_candidates", -1)),
                "recovery_option_profile_base_transition_rejected_roadgraph_candidates": int(option_profile_base_detail.get("transition_rejected_roadgraph_candidates", -1)),
                "recovery_option_profile_rvr_transition_rejected_roadgraph_candidates": int(option_profile_rvr_detail.get("transition_rejected_roadgraph_candidates", -1)),
                "selected_controller_transition_feasible": bool(controller_transition_feasible_np[selected]) if has_valid and int(selected) < len(controller_transition_feasible_np) else False,
                "recovery_waymax_kinematic_guard_used": bool(waymax_kinematic_guard_used),
                "recovery_control_projected_option_profile_used": bool(control_projected_option_profile_used),
                "recovery_base_waymax_kinematic_feasible": bool(recovery_base_waymax_kinematic_feasible),
                "recovery_rvr_waymax_kinematic_feasible": bool(recovery_rvr_waymax_kinematic_feasible),
                "recovery_waymax_kinematic_transition_delta": int(bool(recovery_rvr_waymax_kinematic_feasible)) - int(bool(recovery_base_waymax_kinematic_feasible)),
                "recovery_base_waymax_inverse_accel": float(recovery_base_waymax_inverse_accel),
                "recovery_rvr_waymax_inverse_accel": float(recovery_rvr_waymax_inverse_accel),
                "recovery_base_waymax_steering_curvature": float(recovery_base_waymax_steering),
                "recovery_rvr_waymax_steering_curvature": float(recovery_rvr_waymax_steering),
                "waymax_kinematics_contract_max_acc_mps2": float(recovery_waymax_contract_detail.get("max_acc_mps2", -1.0)),
                "waymax_kinematics_contract_max_steering_curvature": float(recovery_waymax_contract_detail.get("max_steering_curvature", -1.0)),
                "waymax_kinematics_contract_dt_s": float(recovery_waymax_contract_detail.get("metric_dt_s", -1.0)),
                "waymax_kinematics_contract_source": str(recovery_waymax_contract_detail.get("contract_source", "not_used")),
                "recovery_option_profile_base_control_projected_h1_kinematic_feasible_candidates": int(option_profile_base_detail.get("control_projected_h1_kinematic_feasible_candidates", -1)),
                "recovery_option_profile_rvr_control_projected_h1_kinematic_feasible_candidates": int(option_profile_rvr_detail.get("control_projected_h1_kinematic_feasible_candidates", -1)),
                "recovery_option_profile_base_control_projected_full_kinematic_feasible_candidates": int(option_profile_base_detail.get("control_projected_full_kinematic_feasible_candidates", -1)),
                "recovery_option_profile_rvr_control_projected_full_kinematic_feasible_candidates": int(option_profile_rvr_detail.get("control_projected_full_kinematic_feasible_candidates", -1)),
                "recovery_option_profile_base_control_projected_max_realized_prefix_steps": int(option_profile_base_detail.get("control_projected_max_realized_prefix_steps", -1)),
                "recovery_option_profile_rvr_control_projected_max_realized_prefix_steps": int(option_profile_rvr_detail.get("control_projected_max_realized_prefix_steps", -1)),
                "recovery_frontier_probe_used": bool(recovery_frontier_probe_used),
                "recovery_frontier_representative_count": int(recovery_frontier_representative_count),
                "recovery_frontier_profiles_evaluated": int(recovery_frontier_profiles_evaluated),
                "recovery_frontier_strict_admissible_count": int(recovery_frontier_strict_admissible_count),
                "recovery_frontier_weak_admissible_count": int(recovery_frontier_weak_admissible_count),
                "recovery_frontier_current_prefix_admissible_count": int(recovery_frontier_current_prefix_admissible_count),
                "recovery_frontier_selected_prefix_delta_steps": int(recovery_frontier_selected_prefix_delta_steps),
                "recovery_frontier_active_macro_before": int(recovery_frontier_active_macro_before),
                "recovery_frontier_active_macro_after": int(self._recovery_frontier_macro) if method == "cowp_control_projected_recovery_frontier" else -1,
                "recovery_frontier_selected_macro": int(recovery_frontier_selected_macro),
                "recovery_frontier_selected_candidate": int(recovery_frontier_selected_candidate),
                "recovery_frontier_selected_is_historical_rvr": bool(recovery_frontier_selected_is_historical_rvr),
                "recovery_frontier_selected_is_non_rvr": bool(recovery_frontier_selected_is_non_rvr),
                "recovery_frontier_selected_fallback_score_delta": float(recovery_frontier_selected_fallback_score_delta),
                "recovery_bridge_pending_before": bool(recovery_bridge_pending_before),
                "recovery_bridge_pending_after": bool(self._recovery_bridge_pending) if method == "cowp_recourse_returnability_bridge" else False,
                "recovery_bridge_allowed_macro_count_before": int(len(recovery_bridge_allowed_macros_before)),
                "recovery_bridge_allowed_macro_count_after": int(len(self._recovery_bridge_allowed_macros)) if method == "cowp_recourse_returnability_bridge" else 0,
                "recovery_bridge_entered": bool(recovery_bridge_entered),
                "recovery_bridge_direct_entry": bool(recovery_bridge_direct_entry),
                "recovery_bridge_recourse_executed": bool(recovery_bridge_recourse_executed),
                "recovery_bridge_aborted": bool(recovery_bridge_aborted),
                "recourse_returnability_probe_used": bool(recourse_returnability_probe_used),
                "recourse_base_direct_restore": bool(recourse_base_direct_restore),
                "recourse_rvr_direct_restore": bool(recourse_rvr_direct_restore),
                "recourse_base_macro_count": int(len(recourse_base_macros)),
                "recourse_rvr_macro_count": int(len(recourse_rvr_macros)),
                "recourse_returnability_strict_dominates": bool(recourse_returnability_strict_dominates),
                "recourse_returnability_weak_dominates": bool(recourse_returnability_weak_dominates),
                "recourse_base_action_classes_available": int(recourse_base_detail.get("recourse_action_classes_available", 0)),
                "recourse_rvr_action_classes_available": int(recourse_rvr_detail.get("recourse_action_classes_available", 0)),
                "recourse_base_action_classes_evaluated": int(recourse_base_detail.get("recourse_action_classes_evaluated", 0)),
                "recourse_rvr_action_classes_evaluated": int(recourse_rvr_detail.get("recourse_action_classes_evaluated", 0)),
                "recourse_base_successor_conventional_candidates": int(recourse_base_detail.get("successor_conventional_candidates", -1)),
                "recourse_rvr_successor_conventional_candidates": int(recourse_rvr_detail.get("successor_conventional_candidates", -1)),
                "recourse_direct_restoring_candidate_count": int(recourse_direct_restoring_candidate_count),
                "recourse_bridge_representatives_evaluated": int(recourse_bridge_representatives_evaluated),
                "recourse_bridge_action_classes_available": int(recourse_bridge_action_classes_available),
                "recourse_bridge_candidate_pool": int(recourse_bridge_candidate_pool),
                "recourse_bridge_selected_macro": int(recourse_bridge_selected_macro),
                "recourse_bridge_minimum_prefix_steps": float(recourse_bridge_minimum_prefix_steps),
                "recourse_current_prefix_nonregressive": bool(recourse_current_prefix_nonregressive),
                "recourse_current_action_survives_one_step": bool(recourse_current_action_survives_one_step),
                "recovery_tube_probe_used": bool(recovery_tube_probe_used),
                "recovery_tube_selected": bool(recovery_tube_selected),
                "recovery_tube_action_changed": bool(recovery_tube_action_changed),
                "recovery_tube_parent_pool": int(recovery_tube_detail.get("parent_pool", 0)),
                "recovery_tube_parent_action_classes": int(recovery_tube_detail.get("parent_action_classes", 0)),
                "recovery_tube_hypotheses_generated": int(recovery_tube_detail.get("tube_hypotheses_generated", 0)),
                "recovery_tube_unique_action_hypotheses": int(recovery_tube_detail.get("tube_hypotheses_unique_action", 0)),
                "recovery_tube_full_physically_safe": int(recovery_tube_detail.get("tube_full_physically_safe", 0)),
                "recovery_tube_shift_closed": int(recovery_tube_detail.get("tube_shift_closed", 0)),
                "recovery_tube_nominal_shift_closed": int(recovery_tube_detail.get("tube_nominal_shift_closed", 0)),
                "recovery_tube_lower_envelope_shift_closed": int(recovery_tube_detail.get("tube_lower_envelope_shift_closed", 0)),
                "recovery_tube_upper_envelope_shift_closed": int(recovery_tube_detail.get("tube_upper_envelope_shift_closed", 0)),
                "recovery_tube_parents_with_nominal_conflict": int(recovery_tube_detail.get("parents_with_nominal_conflict", 0)),
                "recovery_tube_mean_parent_first_conflict_step": float(recovery_tube_detail.get("mean_parent_first_conflict_step", 0.0)),
                "recovery_tube_mean_parent_last_conflict_step": float(recovery_tube_detail.get("mean_parent_last_conflict_step", 0.0)),
                "recovery_tube_event_release_shift_closed": int(recovery_tube_detail.get("tube_event_release_shift_closed", 0)),
                "recovery_tube_lower_event_release_shift_closed": int(recovery_tube_detail.get("tube_lower_event_release_shift_closed", 0)),
                "recovery_tube_upper_event_release_shift_closed": int(recovery_tube_detail.get("tube_upper_event_release_shift_closed", 0)),
                "recovery_tube_lifted_only_parent_count": int(recovery_tube_detail.get("tube_lifted_only_parent_count", 0)),
                "recovery_tube_event_release_only_parent_count": int(recovery_tube_detail.get("tube_event_release_only_parent_count", 0)),
                "recovery_tube_nominal_first_target_max_abs_error": float(recovery_tube_detail.get("nominal_first_target_max_abs_error", 0.0)),
                "recovery_tube_selected_is_lifted": bool(recovery_tube_detail.get("selected_is_lifted", False)),
                "recovery_tube_selected_is_event_release": bool(recovery_tube_detail.get("selected_is_event_release", False)),
                "recovery_tube_selected_policy_id": int(recovery_tube_detail.get("selected_policy_id", 0)),
                "recovery_tube_selected_policy_name": str(recovery_tube_detail.get("selected_policy_name", "NONE")),
                "recovery_tube_selected_envelope_mode": int(recovery_tube_detail.get("selected_envelope_mode", 0)),
                "recovery_tube_selected_release_edge": int(recovery_tube_detail.get("selected_release_edge", 0)),
                "recovery_tube_selected_nonnominal_edges": int(recovery_tube_detail.get("selected_nonnominal_edges", 0)),
                "recovery_tube_selected_parent_candidate": int(recovery_tube_detail.get("selected_parent_candidate", -1)),
                "recovery_tube_selected_parent_macro": int(recovery_tube_detail.get("selected_parent_macro", -1)),
                "recovery_tube_selected_parent_macro_name": str(recovery_tube_detail.get("selected_parent_macro_name", "NONE")),
                "recovery_tube_selected_first_accel_delta": float(recovery_tube_detail.get("selected_first_accel_delta", 0.0)),
                "recovery_tube_selected_collision_min_margin_m": float(recovery_tube_detail.get("selected_collision_min_margin_m", -999.0)),
                "recovery_tube_selected_shift_collision_min_margin_m": float(recovery_tube_detail.get("selected_shift_collision_min_margin_m", -999.0)),
                "recovery_tube_selected_fallback_score_delta": float(recovery_tube_selected_fallback_score_delta),
                "recovery_tube_nested_v39_selected": bool(recovery_tube_detail.get("nested_v39_selected", False)),
                "recovery_tube_first_action_interval_completion_attempted": bool(recovery_tube_detail.get("first_action_interval_completion_attempted", False)),
                "recovery_tube_first_action_interval_basis_count": int(recovery_tube_detail.get("first_action_interval_basis_count", 0)),
                "recovery_tube_first_action_interval_seed_evaluations": int(recovery_tube_detail.get("first_action_interval_seed_evaluations", 0)),
                "recovery_tube_first_action_interval_boundary_proposals": int(recovery_tube_detail.get("first_action_interval_boundary_proposals", 0)),
                "recovery_tube_first_action_interval_hypotheses_evaluated": int(recovery_tube_detail.get("first_action_interval_hypotheses_evaluated", 0)),
                "recovery_tube_first_action_interval_unique_actions": int(recovery_tube_detail.get("first_action_interval_unique_actions", 0)),
                "recovery_tube_first_action_interval_full_physically_safe": int(recovery_tube_detail.get("first_action_interval_full_physically_safe", 0)),
                "recovery_tube_first_action_interval_shift_closed": int(recovery_tube_detail.get("first_action_interval_shift_closed", 0)),
                "recovery_tube_first_action_interval_only_parent_count": int(recovery_tube_detail.get("first_action_interval_only_parent_count", 0)),
                "recovery_tube_first_action_interval_new_actions": int(recovery_tube_detail.get("first_action_interval_new_actions", 0)),
                "recovery_tube_selected_is_first_action_interval_completion": bool(recovery_tube_detail.get("selected_is_first_action_interval_completion", False)),
                "recovery_tube_selected_is_new_first_action": bool(recovery_tube_detail.get("selected_is_new_first_action", False)),
                "recovery_tube_selected_first_accel_fraction": float(recovery_tube_detail.get("selected_first_accel_fraction", 0.0)),
                "recovery_tube_selected_boundary_source": str(recovery_tube_detail.get("selected_boundary_source", "NONE")),
                "recovery_tube_interaction_response_attempted": bool(recovery_tube_detail.get("interaction_response_attempted", False)),
                "recovery_tube_interaction_response_selected": bool(recovery_tube_detail.get("interaction_response_selected", False)),
                "recovery_tube_selected_is_interaction_response": bool(recovery_tube_detail.get("selected_is_interaction_response", False)),
                "recovery_tube_selected_certificate_kind": str(recovery_tube_detail.get("selected_certificate_kind", "none")),
                "recovery_tube_interaction_failure_reason": str(recovery_tube_detail.get("interaction_failure_reason", "none")),
                "recovery_tube_interaction_support_agents_total": int(recovery_tube_detail.get("interaction_support_agents_total", 0)),
                "recovery_tube_interaction_support_agents_ready": int(recovery_tube_detail.get("interaction_support_agents_ready", 0)),
                "recovery_tube_interaction_support_retained_roots": int(recovery_tube_detail.get("interaction_support_retained_roots", 0)),
                "recovery_tube_interaction_support_eligible_profiles": int(recovery_tube_detail.get("interaction_support_eligible_profiles", 0)),
                "recovery_tube_interaction_hypotheses_evaluated": int(recovery_tube_detail.get("interaction_hypotheses_evaluated", 0)),
                "recovery_tube_interaction_noop_hypotheses_skipped": int(recovery_tube_detail.get("interaction_noop_hypotheses_skipped", 0)),
                "recovery_tube_interaction_no_blocker_rejects": int(recovery_tube_detail.get("interaction_no_blocker_rejects", 0)),
                "recovery_tube_interaction_unsupported_blocker_rejects": int(recovery_tube_detail.get("interaction_unsupported_blocker_rejects", 0)),
                "recovery_tube_interaction_residual_physical_rejects": int(recovery_tube_detail.get("interaction_residual_physical_rejects", 0)),
                "recovery_tube_interaction_root_unrecoverable_rejects": int(recovery_tube_detail.get("interaction_root_unrecoverable_rejects", 0)),
                "recovery_tube_interaction_joint_incompatibility_rejects": int(recovery_tube_detail.get("interaction_joint_incompatibility_rejects", 0)),
                "recovery_tube_interaction_environment_compatibility_checks": int(recovery_tube_detail.get("interaction_environment_compatibility_checks", 0)),
                "recovery_tube_interaction_environment_compatibility_rejects": int(recovery_tube_detail.get("interaction_environment_compatibility_rejects", 0)),
                "recovery_tube_interaction_environment_compatibility_cache_hits": int(recovery_tube_detail.get("interaction_environment_compatibility_cache_hits", 0)),
                "recovery_tube_interaction_joint_compatibility_checks": int(recovery_tube_detail.get("interaction_joint_compatibility_checks", 0)),
                "recovery_tube_interaction_joint_compatibility_rejects": int(recovery_tube_detail.get("interaction_joint_compatibility_rejects", 0)),
                "recovery_tube_interaction_joint_compatibility_cache_hits": int(recovery_tube_detail.get("interaction_joint_compatibility_cache_hits", 0)),
                "recovery_tube_interaction_joint_assignment_backtracks": int(recovery_tube_detail.get("interaction_joint_assignment_backtracks", 0)),
                "recovery_tube_interaction_successor_context_cache_hits": int(recovery_tube_detail.get("interaction_successor_context_cache_hits", 0)),
                "recovery_tube_interaction_selected_blocker_count": int(recovery_tube_detail.get("interaction_selected_blocker_count", 0)),
                "recovery_tube_interaction_selected_root_count": int(recovery_tube_detail.get("interaction_selected_root_count", 0)),
                "recovery_tube_interaction_selected_minimum_root_mass": float(recovery_tube_detail.get("interaction_selected_minimum_root_mass", 0.0)),
                "recovery_tube_interaction_selected_maximum_response_burden": float(recovery_tube_detail.get("interaction_selected_maximum_response_burden", 0.0)),
                "recovery_tube_interaction_selected_profile_evaluations": int(recovery_tube_detail.get("interaction_selected_profile_evaluations", 0)),
                "recovery_tube_interaction_selected_environment_agent_count": int(recovery_tube_detail.get("interaction_selected_environment_agent_count", 0)),
                "recovery_tube_interaction_selected_environment_compatibility_checks": int(recovery_tube_detail.get("interaction_selected_environment_compatibility_checks", 0)),
                "recovery_tube_nested_v42_selected": bool(recovery_tube_detail.get("nested_v42_selected", False)),
                "recovery_tube_blocker_conditioned_query_attempted": bool(recovery_tube_detail.get("blocker_conditioned_query_attempted", False)),
                "recovery_tube_blocker_conditioned_query_selected": bool(recovery_tube_detail.get("blocker_conditioned_query_selected", False)),
                "recovery_tube_blocker_conditioned_query_agent_count": int(recovery_tube_detail.get("blocker_conditioned_query_agent_count", 0)),
                "recovery_tube_blocker_conditioned_query_candidate_agents_before_exact_filter": int(recovery_tube_detail.get("blocker_conditioned_query_candidate_agents_before_exact_filter", 0)),
                "recovery_tube_blocker_conditioned_query_exact_blocker_agent_count": int(recovery_tube_detail.get("blocker_conditioned_query_exact_blocker_agent_count", 0)),
                "recovery_tube_blocker_conditioned_query_replayed_hypothesis_count": int(recovery_tube_detail.get("blocker_conditioned_query_replayed_hypothesis_count", 0)),
                "recovery_tube_blocker_conditioned_query_ready_agent_count": int(recovery_tube_detail.get("blocker_conditioned_query_ready_agent_count", 0)),
                "recovery_tube_blocker_conditioned_query_hypotheses_evaluated": int(recovery_tube_detail.get("blocker_conditioned_query_hypotheses_evaluated", 0)),
                "recovery_tube_blocker_conditioned_query_unsupported_blocker_rejects": int(recovery_tube_detail.get("blocker_conditioned_query_unsupported_blocker_rejects", 0)),
                "recovery_tube_blocker_conditioned_query_root_unrecoverable_rejects": int(recovery_tube_detail.get("blocker_conditioned_query_root_unrecoverable_rejects", 0)),
                "recovery_tube_blocker_conditioned_query_environment_cache_hits": int(recovery_tube_detail.get("blocker_conditioned_query_environment_cache_hits", 0)),
                "recovery_tube_blocker_conditioned_query_joint_cache_hits": int(recovery_tube_detail.get("blocker_conditioned_query_joint_cache_hits", 0)),
                "recovery_tube_blocker_conditioned_query_successor_context_cache_hits": int(recovery_tube_detail.get("blocker_conditioned_query_successor_context_cache_hits", 0)),
                "selected_waymax_kinematic_feasible": bool(selected_waymax_kinematic_feasible),
                "selected_waymax_inverse_accel": float(selected_waymax_inverse_accel),
                "selected_waymax_steering_curvature": float(selected_waymax_steering),
            })
            if recovery_base_candidate >= 0 and recovery_rvr_candidate >= 0:
                diag.update({
                    "recovery_prefix_gain_steps": int(round(float(
                        collision_prefix_steps[recovery_rvr_candidate].detach().cpu().item()
                        - collision_prefix_steps[recovery_base_candidate].detach().cpu().item()
                    ))),
                    "recovery_action_risk_delta": float(
                        action_decision_risk[recovery_rvr_candidate].detach().cpu().item()
                        - action_decision_risk[recovery_base_candidate].detach().cpu().item()
                    ),
                    "recovery_rule_risk_delta": float(
                        rule_decision_risk[recovery_rvr_candidate].detach().cpu().item()
                        - rule_decision_risk[recovery_base_candidate].detach().cpu().item()
                    ),
                    "recovery_pressure_risk_delta": float(
                        pressure_decision_risk[recovery_rvr_candidate].detach().cpu().item()
                        - pressure_decision_risk[recovery_base_candidate].detach().cpu().item()
                    ),
                })
            if burden is not None:
                diag["max_predicted_burden"] = float(host[32])
            if c_i is not None:
                diag["max_predicted_c_i"] = float(host[33])
            diag["selected_plan_continuity_risk"] = float(continuity_np[selected]) if has_valid else 0.0
        traj, emergency_action_used, execution_source = _resolve_execution_trajectory(
            batch_np["cowp/candidates/trajectory"][0], selected, has_valid, agent_state[sdc_index], self.cfg
        )
        execution_target = None if emergency_action_used else np.asarray(action_targets_np[selected], dtype=np.float32)
        execution_accel = None if emergency_action_used else float(action_accels_np[selected])
        if (
            not emergency_action_used
            and recovery_tube_trajectory_override is not None
            and recovery_tube_target_override is not None
            and recovery_tube_accel_override is not None
        ):
            traj = np.asarray(recovery_tube_trajectory_override, dtype=np.float32)
            execution_target = np.asarray(recovery_tube_target_override, dtype=np.float32)
            execution_accel = float(recovery_tube_accel_override)
            if method == "cowp_blocker_conditioned_interaction_aware_reachable_response_envelope":
                execution_source = "blocker_conditioned_interaction_aware_reachable_response_envelope"
            elif method == "cowp_interaction_aware_reachable_response_envelope":
                execution_source = "interaction_aware_reachable_response_envelope"
            elif method == "cowp_shift_closed_first_action_viability_interval":
                execution_source = "shift_closed_first_action_viability_interval"
            elif method == "cowp_conflict_window_control_reachable_tube":
                execution_source = "conflict_window_control_reachable_tube"
            else:
                execution_source = "shift_closed_control_reachable_tube"
        diag["emergency_action_used"] = bool(emergency_action_used)
        diag["execution_trajectory_source"] = str(execution_source)
        self._last_diagnostics = diag
        self._diagnostics_log.append(diag)
        self._previous_selected_traj = np.array(traj, copy=True)
        if profile_enabled:
            profile_t_selection = self._profile_stamp()
            action = self._trajectory_to_action(
                state,
                agent_state,
                sdc_index,
                traj,
                precomputed_target=execution_target,
                precomputed_accel=execution_accel,
            )
            profile_t_action = self._profile_stamp()
            diag.update({
                "runtime_state_extract_map_s": float(profile_t_state - profile_t0),
                "runtime_candidate_build_cpu_s": float(profile_t_candidate - profile_t_state),
                "runtime_h2d_s": float(profile_t_h2d - profile_t_candidate),
                "runtime_model_forward_s": float(profile_t_model - profile_t_h2d),
                "runtime_selection_s": float(profile_t_selection - profile_t_model),
                "runtime_action_projection_s": float(profile_t_action - profile_t_selection),
                "runtime_policy_total_s": float(profile_t_action - profile_t0),
            })
            self._diagnostics_log[-1] = dict(diag)
            self._last_diagnostics = diag
            return action
        return self._trajectory_to_action(
            state,
            agent_state,
            sdc_index,
            traj,
            precomputed_target=execution_target,
            precomputed_accel=execution_accel,
        )

    def consume_diagnostics(self) -> dict[str, Any] | None:
        row = self._last_diagnostics
        self._last_diagnostics = None
        return row

    def diagnostics_log(self) -> list[dict[str, Any]]:
        return list(self._diagnostics_log)


def make_cowp_policy(
    checkpoint: str,
    cfg: dict,
    *,
    device: str = "auto",
    witness_threshold: float = 0.5,
    bcot_risk_budget: float | None = None,
    action_mode: str = "absolute_xy_yaw",
    ncf_gate_mode: str = "hard",
    priority_hard_threshold: float = 0.55,
    secondary_witness_threshold: float = 0.85,
    secondary_opr_alpha: float = 0.10,
    soft_ncf_penalty: float = 1.5,
    method: str = "cowp",
    adaptive_frontier_margin: float = 0.20,
    outcome_risk_penalty: float = 0.0,
    outcome_risk_threshold: float = 1.10,
    profile_policy_runtime: bool = False,
    profile_policy_runtime_sync: bool = False,
) -> COWPWaymaxPolicy:
    return COWPWaymaxPolicy(
        checkpoint=checkpoint,
        cfg=cfg,
        device=device,
        witness_threshold=witness_threshold,
        bcot_risk_budget=bcot_risk_budget,
        action_mode=action_mode,
        ncf_gate_mode=ncf_gate_mode,
        priority_hard_threshold=priority_hard_threshold,
        secondary_witness_threshold=secondary_witness_threshold,
        secondary_opr_alpha=secondary_opr_alpha,
        soft_ncf_penalty=soft_ncf_penalty,
        method=method,
        adaptive_frontier_margin=adaptive_frontier_margin,
        outcome_risk_penalty=outcome_risk_penalty,
        outcome_risk_threshold=outcome_risk_threshold,
        profile_policy_runtime=profile_policy_runtime,
        profile_policy_runtime_sync=profile_policy_runtime_sync,
    )
