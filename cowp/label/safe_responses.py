from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cowp.core.constants import PriorityRelation, ResponseSource
from cowp.core.types import ScenarioData
from cowp.geometry.collision import unsafe_between, unsafe_between_bool
from cowp.label.burden import compute_burden
from cowp.label.trajectory_primitives import constant_accel_trajectory
from cowp.label.safe_budget_search import (
    build_safe_budget_trajectory_bank,
    prepare_safe_budget_trajectory_bank,
    typed_safe_budget_search_evaluated,
)


@dataclass(frozen=True)
class _ResponsePrimitive:
    """One candidate-independent response primitive with an explicit root identity.

    ``root_index`` is -1 only for legacy/global emergency primitives.  Root-indexed
    primitives are generated as bounded residuals around exactly one natural root,
    so same-root recovery is supervised by construction rather than inferred after
    a globally truncated response search.
    """

    traj: np.ndarray
    source: ResponseSource
    root_index: int
    natural_ref: np.ndarray
    root_weight: float = 0.0
    is_identity: bool = False


def _subsample_by_source(
    primitives: list[_ResponsePrimitive],
    cfg: dict,
) -> list[_ResponsePrimitive]:
    """Keep a bounded, source-balanced bank without deleting root coverage.

    The previous implementation uniformly subsampled a global primitive lattice.
    A response for a low-mass natural root could therefore disappear before label
    construction, turning missing search coverage into a negative recoverability
    label.  v16.8 first keeps one primitive per explicit root, then fills the
    remaining budget by source.
    """
    resp_cfg = cfg.get("response", {})
    max_total = int(resp_cfg.get("max_response_primitives_per_agent", 48))
    if max_total <= 0 or len(primitives) <= max_total:
        return primitives

    rooted = [p for p in primitives if p.root_index >= 0]
    generic = [p for p in primitives if p.root_index < 0]

    # Preserve at least one trajectory for every represented root, ordered by the
    # probability mass that enters OPR.  max_safe_responses is normally >= M.
    best_by_root: dict[int, _ResponsePrimitive] = {}
    for p in rooted:
        old = best_by_root.get(int(p.root_index))
        if old is None or (p.source == ResponseSource.PRED and old.source != ResponseSource.PRED):
            best_by_root[int(p.root_index)] = p
    root_cover = sorted(best_by_root.values(), key=lambda p: (-float(p.root_weight), int(p.root_index)))
    kept = root_cover[:max_total]
    if len(kept) >= max_total:
        return kept

    chosen_ids = {id(p) for p in kept}
    remaining_rooted = [p for p in rooted if id(p) not in chosen_ids]
    pred = [p for p in generic if p.source == ResponseSource.PRED]
    opt = [p for p in generic if p.source == ResponseSource.OPT]
    emg = [p for p in generic if p.source == ResponseSource.EMG]

    # Root-conditioned alternatives are more informative than unassigned generic
    # responses because they directly test the paper's same-root predicate.
    ordered = remaining_rooted + pred + opt + emg
    budget = max_total - len(kept)
    if len(ordered) <= budget:
        kept.extend(ordered)
    elif budget > 0:
        idx = np.linspace(0, len(ordered) - 1, budget, dtype=np.int32)
        used: set[int] = set()
        for j in idx.tolist():
            if j not in used:
                kept.append(ordered[j])
                used.add(j)
        j = 0
        while len(kept) < max_total and j < len(ordered):
            if j not in used:
                kept.append(ordered[j])
                used.add(j)
            j += 1
    return kept[:max_total]


def _root_residual_trajectory(
    root: np.ndarray,
    *,
    dt: float,
    accel: float,
    start_delay_s: float,
    duration_s: float,
) -> np.ndarray:
    """Apply a bounded longitudinal residual while preserving root geometry.

    The residual follows each root's own heading and therefore cannot silently
    change a turn/merge/priority-preserving root into another maneuver.  Position
    is shifted by the integrated speed residual and velocity is reconstructed from
    the root heading.  The identity profile (accel=0) is exact.
    """
    out = np.asarray(root, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] < 7 or len(out) == 0:
        raise ValueError(f"root trajectory must be [T,7], got {out.shape}")
    if abs(float(accel)) < 1.0e-9:
        return out

    delay_step = max(int(round(float(start_delay_s) / max(float(dt), 1.0e-4))), 0)
    duration_steps = max(int(round(float(duration_s) / max(float(dt), 1.0e-4))), 1)
    root_speed = np.linalg.norm(out[:, 3:5], axis=-1)
    delta_v = np.zeros(len(out), dtype=np.float32)
    dv = 0.0
    for t in range(len(out)):
        if delay_step <= t < delay_step + duration_steps:
            dv += float(accel) * float(dt)
        # Do not reverse a root; same-root responses may slow/accelerate but retain
        # the root's passage direction.
        dv = max(dv, -float(root_speed[t]))
        delta_v[t] = dv

    # Re-parameterize time along the *original root polyline* rather than
    # translating every point along its local tangent.  Tangent translation can
    # cut corners on curved roots and silently turn a same-root response into a
    # different spatial maneuver.  Arc-length warping preserves the root's
    # topological/geometric identity exactly within its support.
    xy = out[:, :2].copy()
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    root_s = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(seg, dtype=np.float32)])
    delta_s = np.cumsum(delta_v * float(dt), dtype=np.float32)
    query_s = np.maximum(root_s + delta_s, 0.0)
    heading_unwrapped = np.unwrap(out[:, 2].astype(np.float64))
    inside_s = np.minimum(query_s, root_s[-1])
    x_new = np.interp(inside_s, root_s, xy[:, 0]).astype(np.float32)
    y_new = np.interp(inside_s, root_s, xy[:, 1]).astype(np.float32)
    yaw_new = np.interp(inside_s, root_s, heading_unwrapped).astype(np.float32)
    beyond = query_s > root_s[-1]
    if np.any(beyond):
        terminal_tangent = np.array([np.cos(heading_unwrapped[-1]), np.sin(heading_unwrapped[-1])], dtype=np.float32)
        extension = (query_s[beyond] - root_s[-1])[:, None]
        xy_ext = xy[-1][None, :] + extension * terminal_tangent[None, :]
        x_new[beyond] = xy_ext[:, 0]
        y_new[beyond] = xy_ext[:, 1]
        yaw_new[beyond] = float(heading_unwrapped[-1])
    out[:, 0] = x_new
    out[:, 1] = y_new
    out[:, 2] = ((yaw_new + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    new_speed = np.maximum(root_speed + delta_v, 0.0)
    tangent = np.stack([np.cos(out[:, 2]), np.sin(out[:, 2])], axis=-1).astype(np.float32)
    out[:, 3:5] = tangent * new_speed[:, None]
    return out.astype(np.float32)


def build_root_recovery_trajectory_bank(root: np.ndarray, cfg: dict) -> list[np.ndarray]:
    """Precompute the candidate-independent same-root recovery tube.

    Safety and burden remain candidate-conditioned; only deterministic root
    time-warp construction is reused across ego candidates.
    """
    rcfg = cfg.get("response", {}).get("root_conditioned_transport", {})
    profiles = rcfg.get("label_search_profiles", rcfg.get("profiles", []))
    if not profiles:
        profiles = [
            {"accel_mps2": 0.0, "start_delay_s": 0.0, "duration_s": 1.0},
            {"accel_mps2": -1.5, "start_delay_s": 0.2, "duration_s": 2.0},
            {"accel_mps2": 0.75, "start_delay_s": 0.2, "duration_s": 1.5},
        ]
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    return [
        _root_residual_trajectory(
            root,
            dt=dt,
            accel=float(profile.get("accel_mps2", 0.0)),
            start_delay_s=float(profile.get("start_delay_s", 0.0)),
            duration_s=float(profile.get("duration_s", 2.0)),
        )
        for profile in profiles
    ]


def prepare_root_recovery_burden_bank(
    root: np.ndarray,
    trajectory_bank: list[np.ndarray],
    cfg: dict,
    *,
    object_type: int,
    rho: PriorityRelation,
) -> list[tuple[float, np.ndarray]]:
    """Candidate-independent burden for a safe same-root recovery trajectory."""
    out: list[tuple[float, np.ndarray]] = []
    for tr in trajectory_bank:
        b, comps = compute_burden(
            tr, None, cfg, int(object_type), natural_ref=root, rho=rho, risk_known_zero=True
        )
        out.append((float(b), np.asarray(comps).copy()))
    return out


def root_conditioned_recovery_search(
    root: np.ndarray,
    ego: np.ndarray,
    cfg: dict,
    *,
    object_type: int,
    beta: float,
    rho: PriorityRelation,
    trajectory_bank: list[np.ndarray] | None = None,
    identity_cache: tuple[float, bool] | None = None,
    static_burden_bank: list[tuple[float, np.ndarray]] | None = None,
    min_only: bool = False,
) -> tuple[float, bool, int]:
    """Search a bounded time-warp tube around one natural root.

    ``min_only=True`` is an exact fast path for witness/transport labels: safe
    burden has zero pair-risk and is therefore candidate-independent.  Profiles
    are tested in ascending static burden and the first safe profile is the exact
    minimum-burden safe response.  The legacy full scan remains the default for
    callers that need the total number of safe profiles.
    """
    bank = trajectory_bank if trajectory_bank is not None else build_root_recovery_trajectory_bank(root, cfg)
    static = static_burden_bank
    if static is None:
        static = prepare_root_recovery_burden_bank(
            root, bank, cfg, object_type=int(object_type), rho=rho
        )
    fast_bool = bool(cfg.get("engineering", {}).get("unsafe_bool_fastpath", True))

    if min_only:
        order = sorted(range(len(bank)), key=lambda j: (float(static[j][0]), int(j)))
        for j in order:
            tr = bank[j]
            if j == 0 and identity_cache is not None and np.array_equal(
                np.asarray(tr, dtype=np.float32), np.asarray(root, dtype=np.float32)
            ):
                b_cached, safe_cached = identity_cache
                if bool(safe_cached):
                    best = float(np.clip(float(b_cached), 0.0, 2.0))
                    return best, bool(best <= float(beta)), 1
                continue
            unsafe = (
                unsafe_between_bool(ego, tr, cfg, agent_type=int(object_type))
                if fast_bool else bool(unsafe_between(ego, tr, cfg, agent_type=int(object_type)).unsafe)
            )
            if not unsafe:
                best = float(np.clip(float(static[j][0]), 0.0, 2.0))
                return best, bool(best <= float(beta)), 1
        return 2.0, False, 0

    best = float("inf")
    safe_count = 0
    for j, tr in enumerate(bank):
        if j == 0 and identity_cache is not None and np.array_equal(np.asarray(tr, dtype=np.float32), np.asarray(root, dtype=np.float32)):
            b_cached, safe_cached = identity_cache
            if bool(safe_cached):
                safe_count += 1
                best = min(best, float(b_cached))
            continue
        unsafe = (
            unsafe_between_bool(ego, tr, cfg, agent_type=int(object_type))
            if fast_bool else bool(unsafe_between(ego, tr, cfg, agent_type=int(object_type)).unsafe)
        )
        if unsafe:
            continue
        safe_count += 1
        best = min(best, float(static[j][0]))
    if not np.isfinite(best):
        return 2.0, False, safe_count
    best = float(np.clip(best, 0.0, 2.0))
    return best, bool(best <= float(beta)), safe_count


def _response_primitives_for_agent(
    scene: ScenarioData,
    agent_slot: int,
    critical: dict[str, np.ndarray],
    natural: dict[str, np.ndarray],
    cfg: dict,
) -> tuple[int, int, PriorityRelation, np.ndarray, list[_ResponsePrimitive]]:
    """Build an explicit root-conditioned response bank once per critical agent."""
    limits = cfg.get("limits", {})
    resp_cfg = cfg.get("response", {})
    rcfg = resp_cfg.get("root_conditioned_transport", {})
    R = int(limits.get("max_safe_responses", 32))
    H = int(cfg.get("time", {}).get("future_steps", 80))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    cur = scene.current_time_index
    idx = int(critical["track_index"][agent_slot])
    object_type = int(scene.object_type[idx])
    rho = PriorityRelation(int(critical.get("base_priority", np.zeros(len(critical["valid"]), dtype=np.int32))[agent_slot]))
    curr = scene.states[idx, cur]

    valid_roots = np.where(natural["valid"][agent_slot])[0]
    nat_ref_candidates = natural["traj"][agent_slot, valid_roots]
    nat_ref = nat_ref_candidates[0] if len(nat_ref_candidates) else constant_accel_trajectory(curr, H, dt, accel=0.0)
    primitives: list[_ResponsePrimitive] = []

    root_enabled = bool(rcfg.get("enabled", True))
    max_roots = max(int(rcfg.get("max_roots_per_agent", R)), 0)
    root_order = sorted(
        (int(m) for m in valid_roots),
        key=lambda m: -float(natural.get("weight", np.zeros_like(natural["valid"], dtype=np.float32))[agent_slot, m]),
    )[:max_roots]
    if root_enabled:
        # Identity is indispensable: it proves whether the natural option itself
        # remains safe.  Mild residuals test recoverability without abandoning the
        # root.  Keep the profile count small enough that every root remains covered.
        profiles = rcfg.get(
            "profiles",
            [
                {"accel_mps2": 0.0, "start_delay_s": 0.0, "duration_s": 1.0},
                {"accel_mps2": -1.5, "start_delay_s": 0.2, "duration_s": 2.0},
            ],
        )
        for m in root_order:
            root = np.asarray(natural["traj"][agent_slot, m], dtype=np.float32)
            weight = float(natural.get("weight", np.zeros_like(natural["valid"], dtype=np.float32))[agent_slot, m])
            for j, profile in enumerate(profiles):
                tr = _root_residual_trajectory(
                    root,
                    dt=dt,
                    accel=float(profile.get("accel_mps2", 0.0)),
                    start_delay_s=float(profile.get("start_delay_s", 0.0)),
                    duration_s=float(profile.get("duration_s", 2.0)),
                )
                primitives.append(
                    _ResponsePrimitive(
                        traj=tr,
                        source=ResponseSource.PRED if j == 0 else ResponseSource.OPT,
                        root_index=m,
                        natural_ref=root,
                        root_weight=weight,
                        is_identity=(j == 0),
                    )
                )
    else:
        for m in root_order[: min(8, R)]:
            root = np.asarray(natural["traj"][agent_slot, m], dtype=np.float32)
            primitives.append(_ResponsePrimitive(root, ResponseSource.PRED, m, root, float(natural["weight"][agent_slot, m]), True))

    # Generic primitives are retained as an all-response safety fallback, but they
    # do not receive a hard root identity at generation time.
    for acc in resp_cfg.get("response_acc_values_mps2", [-4.5, -3.5, -2.5, -1.5, -0.5, 0.0, 0.5, 1.0]):
        for delay in resp_cfg.get("response_start_delay_s", [0.0, 0.3, 0.6, 1.0]):
            for dur in resp_cfg.get("response_duration_s", [1.0, 2.0, 3.0]):
                primitives.append(
                    _ResponsePrimitive(
                        constant_accel_trajectory(curr, H, dt, accel=float(acc), start_delay_s=float(delay), duration_s=float(dur)),
                        ResponseSource.OPT,
                        -1,
                        nat_ref,
                    )
                )
    for decel in resp_cfg.get("emergency_decel_values_mps2", [-6.0, -8.0]):
        for delay in resp_cfg.get("emergency_reaction_delay_s", [0.0, 0.2, 0.5]):
            primitives.append(
                _ResponsePrimitive(
                    constant_accel_trajectory(curr, H, dt, accel=float(decel), start_delay_s=float(delay)),
                    ResponseSource.EMG,
                    -1,
                    nat_ref,
                )
            )
    primitives = _subsample_by_source(primitives, cfg)
    return idx, object_type, rho, nat_ref, primitives


def _select_root_balanced(
    evaluated: list[tuple[float, float, np.ndarray, ResponseSource, bool, np.ndarray, int, float]],
    R: int,
) -> list[tuple[float, float, np.ndarray, ResponseSource, bool, np.ndarray, int, float]]:
    """Select responses while guaranteeing one best item per explicit root."""
    if len(evaluated) <= int(R):
        return sorted(evaluated, key=lambda x: x[0])
    best_by_root: dict[int, tuple[float, float, np.ndarray, ResponseSource, bool, np.ndarray, int, float]] = {}
    for item in evaluated:
        root = int(item[6])
        if root < 0:
            continue
        old = best_by_root.get(root)
        if old is None or item[0] < old[0]:
            best_by_root[root] = item
    cover = sorted(best_by_root.values(), key=lambda x: (-float(x[7]), int(x[6])))[: int(R)]
    selected_ids = {id(x) for x in cover}
    # Tuples are recreated only once above; identity is sufficient and avoids a
    # costly trajectory comparison.
    rest = [x for x in sorted(evaluated, key=lambda x: x[0]) if id(x) not in selected_ids]
    return (cover + rest)[: int(R)]


def generate_safe_responses(
    scene: ScenarioData,
    candidates: dict[str, np.ndarray],
    critical: dict[str, np.ndarray],
    natural: dict[str, np.ndarray],
    cfg: dict,
    audit: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    limits = cfg.get("limits", {})
    K = int(limits.get("max_candidates", 64))
    A = int(limits.get("max_critical_agents", 8))
    R = int(limits.get("max_safe_responses", 32))
    H = int(cfg.get("time", {}).get("future_steps", 80))
    traj = np.zeros((K, A, R, H, 7), dtype=np.float32)
    valid = np.zeros((K, A, R), dtype=bool)
    source = np.full((K, A, R), int(ResponseSource.PAD), dtype=np.int32)
    root_index = np.full((K, A, R), -1, dtype=np.int32)
    root_affinity = np.zeros((K, A, R), dtype=np.float32)
    is_safe = np.zeros((K, A, R), dtype=bool)
    is_low = np.zeros((K, A, R), dtype=bool)
    burden_total = np.zeros((K, A, R), dtype=np.float32)
    burden_components = np.zeros((K, A, R, 6), dtype=np.float32)

    primitive_bank: dict[int, tuple[int, int, PriorityRelation, np.ndarray, list[_ResponsePrimitive]]] = {}
    primitive_static: dict[int, list[tuple[float, np.ndarray]]] = {}
    budget_bank: dict[int, list] = {}
    budget_enabled = bool(cfg.get("response", {}).get("safe_budget_search", {}).get("enabled", True))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    mechanism_mask = np.asarray(critical.get("mechanism_valid", critical["valid"]), dtype=bool)
    for a in range(A):
        if critical["valid"][a] and a < len(mechanism_mask) and mechanism_mask[a]:
            primitive_bank[a] = _response_primitives_for_agent(scene, a, critical, natural, cfg)
            _idx, object_type, rho, nat_ref, primitives = primitive_bank[a]
            primitive_static[a] = [
                compute_burden(
                    p.traj, None, cfg, object_type, natural_ref=p.natural_ref, rho=rho, risk_known_zero=True
                )
                for p in primitives
            ]
            if budget_enabled:
                curr_idx = int(critical["track_index"][a])
                curr = scene.states[curr_idx, scene.current_time_index]
                raw_budget = build_safe_budget_trajectory_bank(curr, H, dt, cfg)
                budget_bank[a] = prepare_safe_budget_trajectory_bank(
                    raw_budget, object_type=object_type, cfg=cfg, natural_ref=nat_ref, rho=rho
                )

    for k in range(K):
        if not candidates["valid"][k]:
            continue
        ego = candidates["trajectory"][k]
        for a, (_, object_type, rho, nat_ref, primitives) in primitive_bank.items():
            if audit is not None and not bool(np.asarray(audit.get("pair_relevant"))[k, a]):
                # No factual low-burden natural root is affected by this candidate.
                # Response search is unnecessary and would only create padded
                # negatives for an unaudited pair.
                continue
            evaluated: list[tuple[float, float, np.ndarray, ResponseSource, bool, np.ndarray, int, float]] = []
            if budget_enabled:
                curr_idx = int(critical["track_index"][a])
                curr = scene.states[curr_idx, scene.current_time_index]
                for tr, _name, b, safe, comps in typed_safe_budget_search_evaluated(
                    curr,
                    H,
                    dt,
                    ego,
                    object_type,
                    cfg,
                    natural_ref=nat_ref,
                    rho=rho,
                    trajectory_bank=budget_bank[a],
                ):
                    sort_cost = (0.0 if safe else 10.0) + float(b)
                    evaluated.append((sort_cost, float(b), tr, ResponseSource.OPT, bool(safe), comps, -1, 0.0))
            for primitive_i, primitive in enumerate(primitives):
                tr = primitive.traj
                if not np.all(np.isfinite(tr)):
                    continue
                reused_identity = False
                if audit is not None and primitive.is_identity and int(primitive.root_index) >= 0:
                    evaluated_mask = audit.get("_root_direct_evaluated")
                    if evaluated_mask is not None and bool(np.asarray(evaluated_mask, dtype=bool)[k, a, int(primitive.root_index)]):
                        unsafe_flag = bool(np.asarray(audit["root_unsafe"], dtype=bool)[k, a, int(primitive.root_index)])
                        b_exact = audit.get("_root_direct_burden_exact", audit.get("root_direct_burden"))
                        c_exact = audit.get("_root_direct_burden_components_exact")
                        b = float(np.asarray(b_exact)[k, a, int(primitive.root_index)])
                        if c_exact is not None:
                            comps = np.asarray(c_exact)[k, a, int(primitive.root_index)].astype(np.float32)
                        else:
                            # Compatibility path for an older audit object; fresh
                            # v16.8.11 audits always expose exact components.
                            _, comps = compute_burden(
                                tr, ego, cfg, object_type, natural_ref=primitive.natural_ref, rho=rho,
                                risk_known_zero=bool(cfg.get("engineering", {}).get("risk_known_zero_fastpath", True)) and not unsafe_flag,
                            )
                        reused_identity = True
                if not reused_identity:
                    unsafe_flag = (
                        unsafe_between_bool(ego, tr, cfg, agent_type=object_type)
                        if bool(cfg.get("engineering", {}).get("unsafe_bool_fastpath", True))
                        else bool(unsafe_between(ego, tr, cfg, agent_type=object_type).unsafe)
                    )
                    if not unsafe_flag:
                        b, comps = primitive_static[a][primitive_i]
                        comps = np.asarray(comps).copy()
                    else:
                        b, comps = compute_burden(
                            tr, ego, cfg, object_type, natural_ref=primitive.natural_ref, rho=rho,
                            risk_known_zero=False,
                        )
                sort_cost = (0.0 if not unsafe_flag else 10.0) + float(b)
                evaluated.append(
                    (
                        sort_cost,
                        float(b),
                        tr,
                        primitive.source,
                        not unsafe_flag,
                        comps,
                        int(primitive.root_index),
                        float(primitive.root_weight),
                    )
                )
            selected = _select_root_balanced(evaluated, R)
            beta = float(natural.get("beta", np.full(A, 0.65))[a])
            for r, (_, b, tr, src, safe, comps, root, _root_weight) in enumerate(selected):
                traj[k, a, r] = tr
                valid[k, a, r] = True
                source[k, a, r] = int(src)
                root_index[k, a, r] = int(root)
                root_affinity[k, a, r] = 1.0 if int(root) >= 0 else 0.0
                is_safe[k, a, r] = bool(safe)
                burden_total[k, a, r] = float(b)
                burden_components[k, a, r] = comps
                is_low[k, a, r] = bool(safe and b <= beta)
    return {
        "traj": traj,
        "valid": valid,
        "source": source,
        "root_index": root_index,
        "root_affinity": root_affinity,
        "is_safe": is_safe,
        "is_low_burden": is_low,
        "burden_total": burden_total,
        "burden_components": burden_components,
    }
