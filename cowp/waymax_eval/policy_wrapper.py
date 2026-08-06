from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cowp.core.constants import MacroType
from cowp.planning.set_preservation_selector import select_set_preservation_frontier_1d
from cowp.label.trajectory_primitives import constant_accel_trajectory, smooth_arrival_trajectory, smooth_stop_trajectory, repair_planar_kinematics
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
    g = str(gate_mode or "priority").lower()
    if m == "cowp" and g == "hard":
        g = "priority"
    if m == "universal_ncf":
        g = "hard"
    elif m == "soft_burden_cost_only":
        g = "soft"
    elif m in {"idm_lattice", "conventional_safety", "planner_score_only"}:
        g = "none"
    return m, g


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
    state: Any, *, allow_logged_fallback: bool = False, prefer_logged: bool = False
) -> dict[str, np.ndarray]:
    traj, _ = _traj_arrays(
        state, allow_logged_fallback=allow_logged_fallback, prefer_logged=prefer_logged
    )
    x = _unwrap_batch_dim(_to_numpy(_get_field(traj, ("x", "center_x"))))
    y = _unwrap_batch_dim(_to_numpy(_get_field(traj, ("y", "center_y"))))
    yaw_field = _get_field(traj, ("yaw", "heading", "bbox_yaw"))
    vx_field = _get_field(traj, ("vel_x", "velocity_x", "vx"))
    vy_field = _get_field(traj, ("vel_y", "velocity_y", "vy"))
    length_field = _get_field(traj, ("length",))
    width_field = _get_field(traj, ("width",))
    height_field = _get_field(traj, ("height",))
    valid_field = _get_field(traj, ("valid",))
    zeros = np.zeros_like(x, dtype=np.float32)
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "yaw": _unwrap_batch_dim(_to_numpy(yaw_field)).astype(np.float32) if yaw_field is not None else zeros.copy(),
        "vx": _unwrap_batch_dim(_to_numpy(vx_field)).astype(np.float32) if vx_field is not None else zeros.copy(),
        "vy": _unwrap_batch_dim(_to_numpy(vy_field)).astype(np.float32) if vy_field is not None else zeros.copy(),
        "length": _unwrap_batch_dim(_to_numpy(length_field)).astype(np.float32) if length_field is not None else np.full_like(x, 4.8, dtype=np.float32),
        "width": _unwrap_batch_dim(_to_numpy(width_field)).astype(np.float32) if width_field is not None else np.full_like(x, 1.9, dtype=np.float32),
        "height": _unwrap_batch_dim(_to_numpy(height_field)).astype(np.float32) if height_field is not None else np.full_like(x, 1.6, dtype=np.float32),
        "valid": _unwrap_batch_dim(_to_numpy(valid_field)).astype(bool) if valid_field is not None else np.ones_like(x, dtype=bool),
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
    state: Any, cfg: dict
) -> tuple[np.ndarray, np.ndarray, int, dict[str, np.ndarray], int]:
    """Extract history/current state once for the causal online policy."""
    _, t = _traj_arrays(state)
    comps = _extract_traj_components(state)
    hist, cur11 = _history_from_components(comps, t, cfg)
    sdc = _extract_sdc_index(state)
    return hist, cur11, sdc, comps, t


def extract_agent_history_model_state(state: Any, cfg: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Backward-compatible history extractor implemented through one vectorized pass."""
    hist, cur11, sdc, _, _ = extract_online_state_bundle(state, cfg)
    return hist, cur11, sdc

def _extract_roadgraph_tokens(state: Any, cfg: dict) -> dict[str, np.ndarray]:
    """Return best-effort roadgraph point tokens from Waymax state.

    Public/local Waymax builds expose roadgraph fields with slightly different
    names.  The policy should not fail if roadgraph is unavailable; it simply
    falls back to ego-heading proposals.
    """
    rg = _get_field(state, ("roadgraph_points", "roadgraph", "roadgraph_static_points"))
    if rg is None:
        return {"xy": np.zeros((0, 2), dtype=np.float32), "heading": np.zeros(0, dtype=np.float32), "valid": np.zeros(0, dtype=bool), "types": np.zeros(0, dtype=np.int32)}
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
        return {"xy": np.zeros((0, 2), dtype=np.float32), "heading": np.zeros(0, dtype=np.float32), "valid": np.zeros(0, dtype=bool), "types": np.zeros(0, dtype=np.int32)}
    dir_x = _get_field(rg, ("dir_x", "direction_x", "dx"))
    dir_y = _get_field(rg, ("dir_y", "direction_y", "dy"))
    if dir_x is not None and dir_y is not None:
        dx = _to_numpy(dir_x)
        dy = _to_numpy(dir_y)
        while dx.ndim > 1:
            dx = dx[0]
            dy = dy[0]
        heading = np.arctan2(dy.reshape(-1), dx.reshape(-1)).astype(np.float32)
    else:
        # Local finite-difference heading.  It is noisy across polyline breaks but
        # still better than all-zero map tokens for conflict-query conditioning.
        diff = np.gradient(xy, axis=0) if len(xy) > 1 else np.zeros_like(xy)
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
        types = np.zeros(len(xy), dtype=np.int32)
    finite = np.isfinite(xy).all(axis=-1) & np.isfinite(heading)
    valid = valid & finite
    max_points = int(cfg.get("limits", {}).get("max_roadgraph_points", 20000))
    if len(xy) > max_points:
        # Keep a deterministic spread of points; local filtering below selects
        # only points near ego, so this mostly protects pathological states.
        idx = np.linspace(0, len(xy) - 1, max_points, dtype=np.int64)
        xy, heading, valid, types = xy[idx], heading[idx], valid[idx], types[idx]
    return {"xy": xy, "heading": heading, "valid": valid, "types": types}


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


def _collision_free_against_constant_velocity(
    traj: np.ndarray,
    agent_state: np.ndarray,
    sdc_index: int,
    cfg: dict,
    other_future_trajs: np.ndarray | None = None,
) -> bool:
    pcfg = cfg.get("planning", {})
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    H_full = int(len(traj))
    H = min(H_full, int(pcfg.get("online_collision_check_horizon_steps", H_full)))
    stride = max(1, int(pcfg.get("online_collision_check_stride", 2)))
    idx = np.arange(0, H, stride, dtype=np.int64)
    if idx.size == 0:
        idx = np.asarray([0], dtype=np.int64)
    traj_xy = np.asarray(traj[idx, :2], dtype=np.float32)
    if not np.isfinite(traj_xy).all():
        return False
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

    ranked: list[tuple[int, float, bool, float, float, float]] = []
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
        priority_like = (-8.0 <= longitudinal <= 55.0 and lateral <= 7.5) or (ttc <= float(pcfg.get("online_priority_ttc_s", 5.0)))
        # Check priority/front agents first, then nearest agents.  Limiting the
        # tail prevents pathological 128-agent scenes from dominating online COWP.
        rank = (0 if priority_like else 1, dist)
        ranked.append((j, float(rank[0]) * 1000.0 + rank[1], priority_like, longitudinal, lateral, ttc))
    ranked.sort(key=lambda x: x[1])
    if max_agents > 0:
        ranked = ranked[:max_agents]

    for j, _key, priority_like, _longitudinal, _lateral, _ttc in ranked:
        other_radius = max(float(agent_state[j, 7]), float(agent_state[j, 8]), 4.0) * 0.5
        radius = ego_radius + other_radius + 0.5
        logged, cv = _agent_future_xy(agent_state, j, H_full, dt, other_future_trajs)
        logged_xy = logged[idx]
        cv_xy = cv[idx]
        if not np.isfinite(logged_xy).all():
            logged_xy = cv_xy
        if not np.isfinite(logged_xy).all():
            continue
        min_logged = float(np.min(np.linalg.norm(traj_xy - logged_xy, axis=-1)))
        if min_logged < radius + logged_buffer:
            return False
        if require_cv and priority_like and np.isfinite(cv_xy).all():
            min_cv = float(np.min(np.linalg.norm(traj_xy - cv_xy, axis=-1)))
            if min_cv < radius + cv_buffer:
                return False
    return True

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
    conventional_check: bool = True,
    other_future_trajs: np.ndarray | None = None,
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
    conv = True
    if conventional_check:
        conv = _roadgraph_drivable_mask(traj, roadgraph) and _collision_free_against_constant_velocity(
            traj, agent_state, sdc_index, cfg, other_future_trajs=other_future_trajs
        )
    out.append(traj.astype(np.float32))
    macros.append(int(macro))
    utils.append(float(utility))
    valids.append(bool(conv))


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        _add_candidate(candidates, macros, utils, conventional, tr, macro, util, agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs)

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
            _add_candidate(candidates, macros, utils, conventional, tr, MacroType.STOP_BEFORE_CONFLICT, 0.4 + 0.02 * max(0.0, 20.0 - dist), agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs)

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
                    _add_candidate(candidates, macros, utils, conventional, tr, m, 0.05 + max(0.0, offset), agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs)

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
                        other_future_trajs=other_future_trajs,
                    )
                    if len(candidates) > before_count:
                        timing_profile_bins.add(profile_key)
                        bcte_added += 1
            used += 1

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
                _add_candidate(candidates, macros, utils, conventional, tr, macro, -0.02 * progress + 0.15 + 0.03 * float(delay), agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs)

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
                        agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs
                    )

    # Ensure the online batch contains a minimally useful ego-motion set even in
    # low-speed scenes where endpoint de-duplication and dynamics checks collapse
    # many primitives.  These are still kinematically repaired and checked first;
    # only the final neutral fallback bypasses the conservative mask.
    min_online = int(cfg.get("planning", {}).get("min_online_candidates", min(8, K)))
    if len(candidates) < min_online:
        supplemental_acc = [0.25, -0.25, 0.75, -0.75, 1.25, -1.25]
        for acc in supplemental_acc:
            if len(candidates) >= min_online or len(candidates) >= K:
                break
            macro = MacroType.ACCELERATE_CROSS if acc > 0 else MacroType.YIELD
            tr = constant_accel_trajectory(current, H, dt, accel=float(acc))
            progress = float(np.linalg.norm(tr[-1, :2] - tr[0, :2]))
            _add_candidate(candidates, macros, utils, conventional, tr, macro, -0.02 * progress + 0.10 * abs(acc), agent_state, sdc_index, roadgraph, cfg, other_future_trajs=other_future_trajs)

    # Guaranteed conservative option, even if marked not conventional-safe.
    if len(candidates) < K:
        tr = smooth_stop_trajectory(current, H, dt, decel=float(cfg.get("planning", {}).get("fallback_decel_mps2", -2.0)))
        _add_candidate(candidates, macros, utils, conventional, tr, MacroType.NEUTRAL_EGO, 0.8, agent_state, sdc_index, roadgraph, cfg, conventional_check=False, other_future_trajs=other_future_trajs)

    traj = np.zeros((K, H, 7), dtype=np.float32)
    valid = np.zeros(K, dtype=bool)
    conventional_safe = np.zeros(K, dtype=bool)
    macro = np.full(K, int(MacroType.PAD), dtype=np.int64)
    utility = np.zeros(K, dtype=np.float32)
    for i, tr in enumerate(candidates[:K]):
        traj[i] = tr
        valid[i] = True
        conventional_safe[i] = bool(conventional[i])
        macro[i] = int(macros[i])
        utility[i] = float(utils[i])
    return traj, valid, conventional_safe, macro, utility


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

    cand_traj, cand_valid, conventional_safe, macro, utility = _route_lane_aware_candidates(
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
        "cowp/candidates/rule_risk": rule_risk[None],
        "cowp/critical/track_index": crit_idx[None],
        "cowp/critical/input_index": crit_idx[None],
        "cowp/critical/valid": crit_valid[None],
        "map/conflict_regions": conflict[None],
        "map/conflict_region_valid": conflict_valid[None],
    }
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
        self._cached_roadgraph_scenario_index: int | None = None
        self._cached_roadgraph: dict[str, np.ndarray] | None = None

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
        if step == 0 or (scenario_index is not None and scenario_index != self._previous_scenario_index):
            self._previous_longitudinal_accel = 0.0
            self._previous_selected_traj = None
        self._previous_scenario_index = scenario_index
        method, gate_mode = _canonical_online_method(getattr(self, "method", "cowp"), self.ncf_gate_mode)
        needs_cowp_risk = method not in {"planner_score_only", "conventional_safety", "idm_lattice"}
        history, agent_state, sdc_index, traj_components, current_t = extract_online_state_bundle(state, self.cfg)
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
        batch_np = build_online_batch(
            agent_state, sdc_index, self.cfg, history_model_state=history, roadgraph=roadgraph,
            other_future_trajs=other_future_trajs, compute_rule_risk=needs_cowp_risk,
        )
        online_keys = (
            "state/history",
            "state/agent_valid",
            "state/is_sdc",
            "cowp/candidates/trajectory",
            "cowp/candidates/valid",
            "cowp/candidates/macro_type",
            "cowp/candidates/ego_utility_prior",
            "cowp/candidates/conventional_safe",
            "cowp/candidates/rule_risk",
            "cowp/critical/track_index",
            "cowp/critical/input_index",
            "cowp/critical/valid",
            "map/conflict_regions",
            "map/conflict_region_valid",
        )
        batch = {k: self.torch.as_tensor(batch_np[k], device=self.dev) for k in online_keys if k in batch_np}
        with self.torch.inference_mode():
            pred = self.model(batch, stage="planner")
            scores = self.torch.nan_to_num(pred["planner_score"][0].float(), nan=1e6, posinf=1e6, neginf=-1e6)
            cand_valid = batch["cowp/candidates/valid"][0].bool()
            # One host synchronization for a predicate reused throughout the
            # decision path.  The previous implementation queried any(valid)
            # repeatedly inside normalization and diagnostics, forcing the CUDA
            # stream to synchronize many times per simulator step.
            has_valid = bool(cand_valid.any().detach().cpu().item())
            conventional = batch.get("cowp/candidates/conventional_safe", batch["cowp/candidates/valid"])[0].bool()
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
                        fallback_reason = "baseline_use_valid"
                    has_select = has_valid
                selected = int(self.torch.argmin(self.torch.where(select_mask, adjusted_scores, self.torch.full_like(adjusted_scores, float("inf")))).item()) if has_select else 0
                baseline_host = self.torch.stack([
                    select_mask.sum().float(),
                    cand_valid.sum().float(),
                    (cand_valid & conventional).sum().float(),
                    batch["cowp/critical/valid"][0].bool().sum().float(),
                    batch["map/conflict_region_valid"][0].bool().sum().float(),
                    scores[selected].float(),
                ]).detach().cpu().tolist()
                diag = {
                    "scenario_index": int(scenario_index) if scenario_index is not None else -1,
                    "step": int(step) if step is not None else -1,
                    "selected_candidate": int(selected),
                    "accepted_candidates": int(baseline_host[0]),
                    "frontier_candidates": -1,
                    "valid_candidates": int(baseline_host[1]),
                    "conventional_candidates": int(baseline_host[2]),
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
                diag["selected_plan_continuity_risk"] = float(continuity_np[selected])
                self._last_diagnostics = diag
                self._diagnostics_log.append(diag)
                traj = np.asarray(batch_np["cowp/candidates/trajectory"][0, selected], dtype=np.float32)
                self._previous_selected_traj = np.array(traj, copy=True)
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
            if isinstance(outcome, dict) and float(self.outcome_risk_penalty) > 0.0:
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
            if method == "cowp" and gate_mode in {"priority", "soft"}:
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
                    result = select_set_preservation_frontier_1d(
                        scores=scores,
                        base_mask=frontier_base,
                        noncoercive_risk=noncoercive_risk,
                        score_risk=score_decision_risk,
                        progress=candidate_progress,
                        progress_shortfall=progress_shortfall,
                        action_risk=action_decision_risk,
                        rule_risk=rule_decision_risk,
                        outcome_risk=outcome_decision_risk,
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
            # Conservative fallback hierarchy: first accepted P-NCF/NCF; then a
            # neutral/stop-like conventional candidate; finally the guaranteed
            # neutral candidate.  Avoid falling back to an arbitrary conventional
            # false-safe plan, which hid the effect of NCF rejection in the smoke test.
            macro_t = batch["cowp/candidates/macro_type"][0].long()
            stop_ids = self.torch.as_tensor(
                [int(MacroType.STOP_BEFORE_CONFLICT), int(MacroType.YIELD), int(MacroType.CREEP), int(MacroType.NEUTRAL_EGO)],
                device=self.dev,
                dtype=macro_t.dtype,
            )
            stop_like = (macro_t[:, None] == stop_ids[None, :]).any(dim=-1)
            fallback_flags = self.torch.stack([
                selection_mask.any(),
                (cand_valid & conventional).any(),
                cand_valid.any(),
                (cand_valid & stop_like).any(),
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
            if bool(fallback_flags[0]):
                select_mask = selection_mask
                select_score = adjusted_scores
                fallback_used = False
                fallback_reason = "accepted_ncf" if gate_mode == "hard" else ("accepted_baseline" if gate_mode in {"none", "off"} else "accepted_priority_ncf")
            elif bool(fallback_flags[1]):
                select_mask = cand_valid & conventional
                select_score = fallback_score
                fallback_used = True
                fallback_reason = "no_certificate_use_least_coercive_conventional"
            elif bool(fallback_flags[2]):
                select_mask = cand_valid
                select_score = fallback_score
                fallback_used = True
                fallback_reason = "no_conventional_use_least_coercive_valid"
            elif bool(fallback_flags[3]):
                select_mask = cand_valid & stop_like
                select_score = fallback_score
                fallback_used = True
                fallback_reason = "emergency_stop_like"
            else:
                select_mask = cand_valid
                select_score = fallback_score
                fallback_used = True
                fallback_reason = "no_valid_candidate"
            selected = int(self.torch.argmin(self.torch.where(select_mask, select_score, self.torch.full_like(select_score, float("inf")))).item()) if has_valid else 0
            selected_witness = witness[selected]
            selected_opr = opr[selected]
            has_crit = bool(crit_mask.any().detach().cpu().item())
            zero = scores.new_tensor(0.0)
            one = scores.new_tensor(1.0)
            frontier_count = frontier.sum().float() if "frontier" in locals() and self.torch.is_tensor(frontier) else scores.new_tensor(-1.0)
            valid_cert = candidate_cert_risk[cand_valid] if has_valid else candidate_cert_risk[:0]
            valid_pressure = pressure_prior[cand_valid] if has_valid else pressure_prior[:0]
            valid_rule = rule_risk[cand_valid] if has_valid else rule_risk[:0]
            valid_action = action_risk[cand_valid] if has_valid else action_risk[:0]
            bsel = self.torch.nan_to_num(burden[0, selected].float(), nan=0.0, posinf=2.0, neginf=0.0) if burden is not None else None
            csel = self.torch.nan_to_num(c_i[0, selected].float(), nan=0.0, posinf=2.0, neginf=0.0) if c_i is not None else None
            diagnostic_tensors = [
                certificate_accepted.sum().float(),
                selection_mask.sum().float(),
                cand_valid.sum().float(),
                (cand_valid & conventional).sum().float(),
                crit_mask.sum().float(),
                batch["map/conflict_region_valid"][0].bool().sum().float(),
                selected_witness.max() if selected_witness.numel() else zero,
                selected_witness.mean() if selected_witness.numel() else zero,
                uncertainty[selected].mean() if uncertainty[selected].numel() else zero,
                witness_cert[selected].max() if witness_cert[selected].numel() else zero,
                selected_opr.min() if selected_opr.numel() else one,
                selected_opr.mean() if selected_opr.numel() else one,
                scores[selected],
                primary_bad.sum().float() if primary_bad.numel() else zero,
                severe_bad.sum().float() if severe_bad.numel() else zero,
                option_bad.sum().float() if option_bad.numel() else zero,
                priority[selected].max() if priority.numel() else zero,
                priority[selected].mean() if priority.numel() else zero,
                outcome_risk[selected] if outcome_risk.numel() else zero,
                outcome_decision_risk[selected] if outcome_decision_risk.numel() else zero,
                cand_ncf_prob[selected] if cand_ncf_prob.numel() else zero,
                cand_false_safe_prob[selected] if cand_false_safe_prob.numel() else zero,
                cand_quality_prob[selected] if cand_quality_prob.numel() else zero,
                candidate_cert_risk[selected] if candidate_cert_risk.numel() else zero,
                pressure_prior[selected] if pressure_prior.numel() else zero,
                rule_risk[selected] if rule_risk.numel() else zero,
                action_risk[selected] if action_risk.numel() else zero,
                valid_cert.min() if valid_cert.numel() else zero,
                valid_cert.mean() if valid_cert.numel() else zero,
                valid_pressure.mean() if valid_pressure.numel() else zero,
                valid_rule.mean() if valid_rule.numel() else zero,
                valid_action.mean() if valid_action.numel() else zero,
                bsel[crit_mask].max() if bsel is not None and has_crit else zero,
                csel[crit_mask].max() if csel is not None and has_crit else zero,
            ]
            host = self.torch.stack([x.float() for x in diagnostic_tensors]).detach().cpu().tolist()
            diag = {
                "scenario_index": int(scenario_index) if scenario_index is not None else -1,
                "step": int(step) if step is not None else -1,
                "selected_candidate": int(selected),
                "certificate_accepted_candidates": int(host[0]),
                "accepted_candidates": int(host[0]),
                "frontier_candidates": int(host[1]),
                "valid_candidates": int(host[2]),
                "conventional_candidates": int(host[3]),
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
            if burden is not None:
                diag["max_predicted_burden"] = float(host[32])
            if c_i is not None:
                diag["max_predicted_c_i"] = float(host[33])
            diag["selected_plan_continuity_risk"] = float(continuity_np[selected])
        self._last_diagnostics = diag
        self._diagnostics_log.append(diag)
        traj = np.asarray(batch_np["cowp/candidates/trajectory"][0, selected], dtype=np.float32)
        self._previous_selected_traj = np.array(traj, copy=True)
        return self._trajectory_to_action(
            state,
            agent_state,
            sdc_index,
            traj,
            precomputed_target=action_targets_np[selected],
            precomputed_accel=float(action_accels_np[selected]),
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
    )
