from __future__ import annotations

import math

import numpy as np

from ocrap.data.schema import CandidatePrefix, F_CTRL, F_EGO, SceneHistory
from ocrap.planning.route_lattice import offset_route_points, project_to_route
from ocrap.planning.utility import nominal_utility
from ocrap.utils.seed import stable_seed

MACROS = ["nominal", "keep", "brake", "yield", "lane_shift", "merge", "pull_over", "stabilize", "perturb_nominal"]


def _list_cfg(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        return [x.strip().strip("'\"") for x in raw.split(",") if x.strip().strip("'\"")]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def _macro_bank_from_cfg(cfg: dict) -> list[str]:
    """Return the prefix macro bank, optionally narrowed by prefix_macro_whitelist.

    Clean safe/normal shards can use this whitelist to retain nominal-like
    candidates without changing the global MACROS ids used by saved datasets.
    """
    allowed = _list_cfg(cfg.get("prefix_macro_whitelist", None))
    if not allowed:
        return list(MACROS)
    seen: set[str] = set()
    bank: list[str] = []
    for macro in ["nominal", *allowed]:
        if macro in MACROS and macro not in seen:
            bank.append(macro)
            seen.add(macro)
    return bank or ["nominal"]


def _macro_sequence_from_cfg(cfg: dict, n: int) -> list[tuple[str, int]]:
    """Build a candidate macro/variant schedule without hidden duplicates.

    An explicit ``prefix_macro_schedule`` can front-load several distinct
    variants of important recovery macros. Variant ids are counted per macro.
    Without a schedule, preserve the historical round-robin behavior.
    """
    n = max(1, int(n))
    bank = _macro_bank_from_cfg(cfg)
    requested = _list_cfg(cfg.get("prefix_macro_schedule", None))
    if not requested:
        seq: list[tuple[str, int]] = [("nominal", 0)]
        for rep in range(max(1, n)):
            for macro in bank[1:]:
                seq.append((macro, rep))
                if len(seq) >= n:
                    return seq[:n]
        pad_macro = "perturb_nominal" if "perturb_nominal" in bank else bank[-1]
        while len(seq) < n:
            seq.append((pad_macro, len(seq)))
        return seq[:n]

    schedule = [m for m in requested if m in bank and m != "nominal"]
    if not schedule:
        schedule = [m for m in bank if m != "nominal"]
    if not schedule:
        return [("nominal", 0)] * n
    counts: dict[str, int] = {}
    seq = [("nominal", 0)]
    idx = 0
    while len(seq) < n:
        macro = schedule[idx % len(schedule)]
        variant = counts.get(macro, 0)
        counts[macro] = variant + 1
        seq.append((macro, variant))
        idx += 1
    return seq[:n]


def _macro_params(macro: str, variant: int, ego_speed: float, cfg: dict) -> np.ndarray:
    """Generate a finite, deliberately diverse recovery frontier.

    Older versions changed ``variant`` for several macro names without changing
    their parameters, so candidate slots were consumed by exact duplicates.
    v48 uses bounded variant banks whose geometry/control meaning is distinct.
    """
    v = int(max(0, variant))
    if macro == "nominal":
        return np.array([ego_speed, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if macro == "keep":
        dv = (-0.6, 0.0, 0.6)[v % 3]
        return np.array([max(0.0, ego_speed + dv), 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if macro == "brake":
        decel = (1.5, 2.5, 3.5, 4.5)[v % 4]
        stop_s = (8.0, 12.0, 18.0, 25.0)[(v // 4) % 4]
        return np.array([max(0.0, ego_speed - decel), 0.0, -decel, stop_s, 0.0], dtype=np.float32)
    if macro == "yield":
        decel = (1.5, 2.5, 3.5)[v % 3]
        stop_s = (8.0, 12.0, 18.0, 24.0)[(v // 3) % 4]
        return np.array([max(0.0, ego_speed - decel), 0.0, -decel, stop_s, 0.0], dtype=np.float32)
    if macro == "lane_shift":
        side = -1.0 if v % 2 else 1.0
        magnitude = (1.5, 2.5, 3.5)[(v // 2) % 3]
        speed_delta = (-0.5, 0.0)[(v // 6) % 2]
        return np.array([max(0.0, ego_speed + speed_delta), side * magnitude, 0.0, 0.0, 1.0], dtype=np.float32)
    if macro == "merge":
        side = -1.0 if v % 2 else 1.0
        magnitude = (1.2, 2.0, 2.8)[(v // 2) % 3]
        speed_delta = (-0.5, 0.5, 1.2)[(v // 6) % 3]
        return np.array([max(0.0, ego_speed + speed_delta), side * magnitude, 0.4 * speed_delta, 0.0, 1.0], dtype=np.float32)
    if macro == "pull_over":
        side = -1.0 if v % 2 == 0 else 1.0
        magnitude = (3.5, 4.5)[(v // 2) % 2]
        decel = (2.0, 3.0)[(v // 4) % 2]
        return np.array([max(0.0, ego_speed - decel), side * magnitude, -decel, 18.0 + 6.0 * ((v // 8) % 2), 1.0], dtype=np.float32)
    if macro == "stabilize":
        decel = (0.5, 1.0, 1.8, 2.5)[v % 4]
        lateral = (-0.6, 0.0, 0.6)[(v // 4) % 3]
        return np.array([max(0.0, ego_speed - decel), lateral, -0.5 * decel, 0.0, 0.0], dtype=np.float32)
    # perturb nominal
    speed_delta = (-1.0, -0.5, 0.5, 1.0)[v % 4]
    lateral = (-0.6, -0.3, 0.3, 0.6)[(v // 4) % 4]
    return np.array([max(0.0, ego_speed + speed_delta), lateral, 0.0, 0.0, 0.0], dtype=np.float32)


def _rollout(history: SceneHistory, macro: str, params: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    sr = float(cfg.get("sample_rate_hz", 10.0))
    dt = 1.0 / sr
    T_p = max(2, int(round(float(cfg.get("prefix_horizon_s", 1.0)) * sr)))
    ego = history.ego_state.astype(np.float32)
    route = history.route
    proj = project_to_route(ego[:2], route)
    v0 = max(0.0, float(ego[6]))
    target_v, d_lat, a_bias, stop_s, topology_required = [float(x) for x in params]
    a_nom = np.clip((target_v - v0) / max((T_p - 1) * dt, 1e-3) + 0.3 * a_bias, float(cfg.get("control_limits", {}).get("a_min", -6.0)), float(cfg.get("control_limits", {}).get("a_max", 3.0)))
    speeds = np.maximum(0.0, v0 + a_nom * np.arange(T_p, dtype=np.float32) * dt)
    ds = np.cumsum(speeds * dt)
    if macro in ("brake", "yield", "pull_over") and stop_s > 1e-3:
        ds = np.minimum(ds, stop_s)
        speeds = np.where(ds >= stop_s - 1e-3, 0.0, speeds)
    s_query = proj.s + ds
    ramp = np.linspace(0.0, 1.0, T_p, dtype=np.float32)
    lat = proj.d + d_lat * (3 * ramp**2 - 2 * ramp**3)
    pts, headings = offset_route_points(route, s_query, lat)
    states = np.zeros((T_p, F_EGO), dtype=np.float32)
    states[:, 0:2] = pts
    states[:, 4] = headings
    states[:, 6] = speeds
    states[:, 2] = speeds * np.cos(headings)
    states[:, 3] = speeds * np.sin(headings)
    states[:, 5] = np.gradient(headings, dt) if T_p > 2 else 0.0
    states[:, 7] = ego[7] if ego.shape[0] > 7 else 4.8
    states[:, 8] = ego[8] if ego.shape[0] > 8 else 2.0
    controls = np.zeros((T_p - 1, F_CTRL), dtype=np.float32)
    if T_p > 1:
        controls[:, 0] = np.diff(speeds) / dt
        controls[:, 1] = np.clip(np.arctan(float(cfg.get("wheelbase_m", 2.8)) * np.gradient(headings, dt)[:-1] / np.maximum(speeds[:-1], 0.5)), -0.7, 0.7)
        controls[1:, 2] = np.diff(controls[:, 0]) / dt
        controls[1:, 3] = np.diff(controls[:, 1]) / dt
    limits = cfg.get("control_limits", {})
    control_ok = bool(
        np.all(controls[:, 0] <= float(limits.get("a_max", 3.0)) + 1e-4)
        and np.all(controls[:, 0] >= float(limits.get("a_min", -6.0)) - 1e-4)
        and np.all(np.abs(controls[:, 1]) <= float(limits.get("delta_max", 0.55)) + 0.15)
        and np.all(np.abs(controls[:, 2]) <= float(limits.get("j_max", 6.0)) + 8.0)
        and np.all(np.abs(controls[:, 3]) <= float(limits.get("steer_rate_max", 0.5)) + 1.0)
    )
    max_dev = float(np.max(np.abs(lat)))
    route_topology_valid = bool(max_dev <= 4.5 or macro == "pull_over")
    wrong_way_hard = bool(np.any(np.cos(headings - proj.heading) < -0.2))
    offroad_hard = bool(max_dev > 6.0)
    finite = bool(np.isfinite(states).all() and np.isfinite(controls).all())
    # Prefix-level collision/contact labels: keep them as labels, not blanket filters.
    prefix_collision = False
    min_other = 99.0
    if history.agent_history.shape[1] > 1:
        cur = history.agent_history[-1, 1:, :2]
        val = history.agent_valid[-1, 1:].astype(bool)
        if val.any():
            d = np.linalg.norm(cur[val][None, :, :] - states[:, None, :2], axis=-1)
            min_other = float(np.min(d))
            prefix_collision = bool(min_other < 1.8)
    hard_violation = float(prefix_collision) * 1.0 + float(offroad_hard) * 2.0 + float(wrong_way_hard) * 2.0
    diagnostics = {
        "macro_type_id": int(MACROS.index(macro)) if macro in MACROS else -1,
        "control_limits_satisfied": control_ok,
        "route_topology_valid": route_topology_valid,
        "wrong_way_hard": wrong_way_hard,
        "offroad_hard": offroad_hard,
        "prefix_collision": prefix_collision,
        "prefix_contact": bool(min_other < 0.8),
        "max_route_deviation": max_dev,
        "log_divergence": float(np.linalg.norm(states[-1, :2] - history.future_agent_states[min(T_p - 1, len(history.future_agent_states)-1), 0, :2])) if len(history.future_agent_states) else 0.0,
        "topology_required": bool(topology_required),
    }
    feasible = finite and control_ok and route_topology_valid and not wrong_way_hard and not offroad_hard
    diagnostics["feasible"] = feasible
    diagnostics["hard_violation"] = hard_violation
    return states, controls, diagnostics


def generate_candidate_prefixes(history: SceneHistory, cfg: dict) -> list[CandidatePrefix]:
    n = int(cfg.get("num_candidate_prefixes", 24))
    ego_speed = float(history.ego_state[6])
    prefixes: list[CandidatePrefix] = []
    macro_seq = _macro_sequence_from_cfg(cfg, n)
    seen_signatures: set[tuple[str, tuple[float, ...]]] = set()
    for i, (macro, variant) in enumerate(macro_seq[:n]):
        params = _macro_params(macro, variant, ego_speed, cfg)
        states, controls, diag = _rollout(history, macro, params, cfg)
        signature = (macro, tuple(float(x) for x in np.round(params, 4)))
        diag["padded_duplicate"] = signature in seen_signatures
        seen_signatures.add(signature)
        util = nominal_utility(states, controls, diag, cfg)
        harm_proxy = float(diag.get("prefix_contact", False)) + 0.2 * float(diag.get("prefix_collision", False))
        prefixes.append(CandidatePrefix(i, macro, params, states, controls, util, bool(diag["feasible"]), float(diag["hard_violation"]), harm_proxy, diag))
    return prefixes
