from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cowp.core.constants import MacroType
from cowp.label.trajectory_primitives import constant_accel_trajectory, smooth_stop_trajectory, repair_planar_kinematics


def _load_state_dict_compatible(model, state: dict) -> None:
    """Load old checkpoints after adding optional heads."""
    try:
        model.load_state_dict(state)
        return
    except Exception:
        pass
    if state and all(str(k).startswith("_orig_mod.") for k in state.keys()):
        state = {k[len("_orig_mod."):]: v for k, v in state.items()}
        try:
            model.load_state_dict(state)
            return
        except Exception:
            pass
    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
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


def _traj_arrays(state: Any) -> tuple[Any, int]:
    traj = _get_field(state, ("sim_trajectory", "trajectory", "log_trajectory"))
    timestep = _get_field(state, ("timestep", "time_index", "current_timestep"))
    t = int(_to_numpy(timestep).reshape(-1)[0]) if timestep is not None else 0
    if traj is None:
        raise ValueError("SimulatorState has no sim_trajectory/trajectory/log_trajectory attribute.")
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


def _extract_traj_components(state: Any) -> dict[str, np.ndarray]:
    traj, _ = _traj_arrays(state)
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


def extract_agent_history_model_state(state: Any, cfg: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract model-format history and current ScenarioData-format states.

    Training cache uses [x,y,z,length,width,height,heading,vx,vy,speed,valid].
    The first online wrapper only supplied a single current frame, which made the
    encoder distribution much narrower than training.  This helper pads the last
    ``history_steps`` frames from Waymax, reusing the earliest available frame at
    the beginning of an episode.
    """
    _, t = _traj_arrays(state)
    comps = _extract_traj_components(state)
    cur11 = _state11_at(comps, t)
    sdc = _extract_sdc_index(state)
    hist_steps = int(cfg.get("model", cfg).get("history_steps", cfg.get("time", {}).get("history_steps", 11)))
    max_agents = int(cfg.get("limits", {}).get("max_agents", cfg.get("model", cfg).get("max_agents", 128)))
    d_state = int(cfg.get("model", cfg).get("d_state", 11))
    n = min(max_agents, cur11.shape[0])
    hist = np.zeros((max_agents, hist_steps, d_state), dtype=np.float32)
    if comps["x"].ndim == 2:
        indices = [int(np.clip(t - hist_steps + 1 + h, 0, comps["x"].shape[1] - 1)) for h in range(hist_steps)]
    else:
        indices = [t for _ in range(hist_steps)]
    for h, tt in enumerate(indices):
        s11 = _state11_at(comps, tt)[:n]
        hist[:n, h, 0:3] = s11[:, 0:3]
        hist[:n, h, 3:6] = s11[:, 7:10]
        hist[:n, h, 6] = s11[:, 6]
        hist[:n, h, 7:9] = s11[:, 3:5]
        hist[:n, h, 9] = s11[:, 5]
        hist[:n, h, 10] = s11[:, 10]
    mask = np.zeros(max_agents, dtype=bool)
    mask[:n] = cur11[:n, 10] > 0.5
    if 0 <= sdc < max_agents:
        mask[sdc] = True
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
    yaw = np.unwrap(traj[:, 2])
    yaw_rate = np.diff(yaw, prepend=yaw[0]) / max(dt, 1e-3)
    moving = speed > 0.5
    vel_heading = np.arctan2(vel[:, 1], vel[:, 0])
    slip = np.abs(_wrap_angle(vel_heading - traj[:, 2]))
    lateral_acc = np.abs(acc_vec[:, 0] * (-np.sin(traj[:, 2])) + acc_vec[:, 1] * np.cos(traj[:, 2]))
    return bool(
        np.nanmax(acc_long) <= float(cand_cfg.get("max_accel_mps2", 4.0)) + 1e-3
        and -np.nanmin(acc_long) <= float(cand_cfg.get("max_decel_mps2", 6.0)) + 1e-3
        and np.nanmax(np.linalg.norm(jerk_vec, axis=-1)) <= float(cand_cfg.get("max_jerk_mps3", 8.0)) + 1e-3
        and np.nanmax(np.abs(yaw_rate)) <= float(cand_cfg.get("max_yaw_rate_rad_s", 1.2)) + 1e-3
        and np.nanmax(lateral_acc) <= float(cand_cfg.get("max_lateral_accel_mps2", 4.0)) + 1e-3
        and (not np.any(moving) or np.nanmax(slip[moving]) <= float(cand_cfg.get("max_sideslip_rad", 0.20)))
    )


def _roadgraph_drivable_mask(traj: np.ndarray, roadgraph: dict[str, np.ndarray], max_dist: float = 5.5) -> bool:
    xy = roadgraph.get("xy", np.zeros((0, 2), dtype=np.float32))
    valid = _lane_centerline_mask(roadgraph)
    if len(xy) == 0 or not np.any(valid):
        return True
    pts = xy[valid]
    # Sample only a few points to keep online inference cheap.
    sample = traj[np.linspace(0, len(traj) - 1, min(12, len(traj)), dtype=np.int64), :2]
    d2 = ((sample[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
    return bool(np.mean(np.sqrt(np.min(d2, axis=1)) <= float(max_dist)) >= 0.75)


def _collision_free_against_constant_velocity(traj: np.ndarray, agent_state: np.ndarray, sdc_index: int, cfg: dict) -> bool:
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    ego_radius = max(float(agent_state[sdc_index, 7]), float(agent_state[sdc_index, 8]), 4.0) * 0.5 + 0.4
    valid = agent_state[:, 10] > 0.5
    for j in range(agent_state.shape[0]):
        if j == sdc_index or not valid[j]:
            continue
        # Only near objects can invalidate an online proposal.
        if np.linalg.norm(agent_state[j, :2] - agent_state[sdc_index, :2]) > 65.0:
            continue
        t = np.arange(1, len(traj) + 1, dtype=np.float32)[:, None] * dt
        pred = agent_state[j, :2][None] + agent_state[j, 3:5][None] * t
        radius = ego_radius + max(float(agent_state[j, 7]), float(agent_state[j, 8]), 4.0) * 0.5 + 0.5
        if np.any(np.linalg.norm(traj[:, :2] - pred, axis=-1) < radius):
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
    for old in out:
        if np.linalg.norm(old[-1, :2] - end) < dedup_eps:
            return
    conv = True
    if conventional_check:
        conv = _roadgraph_drivable_mask(traj, roadgraph) and _collision_free_against_constant_velocity(traj, agent_state, sdc_index, cfg)
    out.append(traj.astype(np.float32))
    macros.append(int(macro))
    utils.append(float(utility))
    valids.append(bool(conv))


def _route_lane_aware_candidates(agent_state: np.ndarray, sdc_index: int, roadgraph: dict[str, np.ndarray], cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    # Route/keep-lane timing lattice.
    for acc in acc_bank:
        macro = MacroType.KEEP_LANE if abs(acc) < 1e-6 else (MacroType.ACCELERATE_CROSS if acc > 0 else MacroType.YIELD)
        tr = constant_accel_trajectory(current, H, dt, accel=acc)
        progress = float(np.linalg.norm(tr[-1, :2] - tr[0, :2]))
        # Lower score is better. Prefer progress, penalize aggressive accel.
        util = -0.03 * progress + 0.08 * abs(acc)
        _add_candidate(candidates, macros, utils, conventional, tr, macro, util, agent_state, sdc_index, roadgraph, cfg)

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
            _add_candidate(candidates, macros, utils, conventional, tr, MacroType.STOP_BEFORE_CONFLICT, 0.4 + 0.02 * max(0.0, 20.0 - dist), agent_state, sdc_index, roadgraph, cfg)

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
                    _add_candidate(candidates, macros, utils, conventional, tr, m, 0.05 + max(0.0, offset), agent_state, sdc_index, roadgraph, cfg)

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
                _add_candidate(candidates, macros, utils, conventional, tr, macro, -0.02 * progress + 0.15 + 0.03 * float(delay), agent_state, sdc_index, roadgraph, cfg)

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
            _add_candidate(candidates, macros, utils, conventional, tr, macro, -0.02 * progress + 0.10 * abs(acc), agent_state, sdc_index, roadgraph, cfg)

    # Guaranteed conservative option, even if marked not conventional-safe.
    if len(candidates) < K:
        tr = smooth_stop_trajectory(current, H, dt, decel=float(cfg.get("planning", {}).get("fallback_decel_mps2", -2.0)))
        _add_candidate(candidates, macros, utils, conventional, tr, MacroType.NEUTRAL_EGO, 0.8, agent_state, sdc_index, roadgraph, cfg, conventional_check=False)

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

def build_online_batch(agent_state: np.ndarray, sdc_index: int, cfg: dict, *, history_model_state: np.ndarray | None = None, roadgraph: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
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

    cand_traj, cand_valid, conventional_safe, macro, utility = _route_lane_aware_candidates(agent_state, sdc_index, roadgraph, cfg)
    crit_idx, crit_valid = _critical_interaction_rank(agent_state, sdc_index, cand_traj, cand_valid, cfg)
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
        "cowp/critical/track_index": crit_idx[None],
        "cowp/critical/input_index": crit_idx[None],
        "cowp/critical/valid": crit_valid[None],
        # Targets below are not copied to GPU in online inference; they are kept
        # for schema/debug compatibility and to avoid legacy callers failing.
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
        "map/conflict_regions": conflict[None],
        "map/conflict_region_valid": conflict_valid[None],
    }
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


@dataclass
class COWPWaymaxPolicy:
    checkpoint: str
    cfg: dict
    device: str = "auto"
    witness_threshold: float = 0.5
    action_mode: str = "delta_xy_yaw"
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

    def _trajectory_to_action(self, state: Any, agent_state: np.ndarray, sdc_index: int, traj: np.ndarray) -> Any:
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
        target, accel = _consistent_one_step_target(
            agent_state[sdc_index], desired, self.cfg, self._previous_longitudinal_accel
        )
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
        self._previous_scenario_index = scenario_index
        history, agent_state, sdc_index = extract_agent_history_model_state(state, self.cfg)
        roadgraph = _extract_roadgraph_tokens(state, self.cfg)
        batch_np = build_online_batch(agent_state, sdc_index, self.cfg, history_model_state=history, roadgraph=roadgraph)
        online_keys = (
            "state/history",
            "state/agent_valid",
            "state/is_sdc",
            "cowp/candidates/trajectory",
            "cowp/candidates/valid",
            "cowp/candidates/macro_type",
            "cowp/candidates/ego_utility_prior",
            "cowp/candidates/conventional_safe",
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
            conventional = batch.get("cowp/candidates/conventional_safe", batch["cowp/candidates/valid"])[0].bool()
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
            if isinstance(outcome, dict) and float(self.outcome_risk_penalty) > 0.0:
                col_r = self.torch.sigmoid(outcome.get("collision_logit", self.torch.zeros_like(scores))[0].float())
                off_r = self.torch.sigmoid(outcome.get("offroad_logit", self.torch.zeros_like(scores))[0].float())
                ld_r = outcome.get("logdiv", self.torch.zeros_like(scores))[0].float().clamp_min(0.0) / 10.0
                outcome_risk = self.torch.nan_to_num(col_r + off_r + ld_r, nan=1.0, posinf=10.0, neginf=0.0)
            else:
                outcome_risk = self.torch.zeros_like(scores)
            crit_mask = batch["cowp/critical/valid"][0].bool()
            witness = self.torch.where(crit_mask[None, :], witness, self.torch.zeros_like(witness))
            witness_cert = self.torch.where(crit_mask[None, :], witness_cert, self.torch.zeros_like(witness_cert))
            uncertainty = self.torch.where(crit_mask[None, :], uncertainty, self.torch.zeros_like(uncertainty))
            opr = self.torch.where(crit_mask[None, :], opr, self.torch.ones_like(opr))
            alpha = float(self.cfg.get("planning", {}).get("alpha_opr_infer", self.cfg.get("ncf", {}).get("alpha_opr", 0.35)))
            method = str(getattr(self, "method", "cowp") or "cowp").lower()
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
            method = alias.get(method, method)
            gate_mode = str(self.ncf_gate_mode or "priority").lower()
            if method == "cowp" and gate_mode == "hard":
                gate_mode = "priority"
            if method == "universal_ncf":
                gate_mode = "hard"
            elif method == "soft_burden_cost_only":
                gate_mode = "soft"
            elif method in {"idm_lattice", "conventional_safety", "planner_score_only"}:
                gate_mode = "none"
            adjusted_scores = scores
            priority = self.torch.zeros_like(witness)
            primary_bad = self.torch.zeros_like(cand_valid)
            severe_bad = self.torch.zeros_like(cand_valid)
            option_bad = self.torch.zeros_like(cand_valid)
            utility = batch.get("cowp/candidates/ego_utility_prior", None)
            utility_scores = self.torch.nan_to_num(utility[0].float(), nan=1e6, posinf=1e6, neginf=-1e6) if utility is not None else scores
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
                    priority = 0.5 * heuristic_priority + 0.5 * learned_priority
                else:
                    priority = heuristic_priority
                primary_claim = priority >= float(self.priority_hard_threshold)
                primary_bad = ((witness_cert >= float(self.witness_threshold)) & primary_claim).any(dim=-1)
                option_bad = ((opr < alpha) & primary_claim).any(dim=-1)
                severe_bad = ((witness_cert >= float(self.secondary_witness_threshold)) & (opr <= float(self.secondary_opr_alpha)) & primary_claim).any(dim=-1)
                burden_penalty = (witness * priority).amax(dim=-1)
                option_penalty = (self.torch.relu(alpha - opr) * priority).amax(dim=-1)
                adjusted_scores = scores + float(self.soft_ncf_penalty) * (burden_penalty + option_penalty) + float(self.outcome_risk_penalty) * outcome_risk
                if gate_mode == "soft":
                    accepted = cand_valid & conventional
                elif gate_mode in {"none", "off"}:
                    accepted = cand_valid & conventional
                else:
                    accepted = cand_valid & conventional & ~primary_bad & ~option_bad & ~severe_bad & (outcome_risk <= float(self.outcome_risk_threshold))
            # Scene-adaptive feasibility frontier.  If the absolute witness/OPR
            # calibration is imperfect, restrict COWP to the least-coercive
            # conventional frontier in the current scene.  This makes the online
            # controller consistent with the set-valued NCF certificate used in
            # learned-offline evaluation.
            if method == "cowp" and gate_mode in {"priority", "soft"}:
                frontier_base = cand_valid & conventional & (outcome_risk <= float(self.outcome_risk_threshold))
                if frontier_base.any():
                    if 'priority' in locals() and priority.numel():
                        frontier_risk = (witness * priority).amax(dim=-1) + (self.torch.relu(alpha - opr) * priority).amax(dim=-1)
                    else:
                        frontier_risk = witness.amax(dim=-1) + self.torch.relu(alpha - opr).amax(dim=-1)
                    finite_risk = self.torch.where(frontier_base, frontier_risk, self.torch.full_like(frontier_risk, float("inf")))
                    best_risk = finite_risk.min()
                    frontier = frontier_base & (frontier_risk <= best_risk + float(self.adaptive_frontier_margin))
                    if frontier.any():
                        accepted = (accepted & frontier) if bool(accepted.any().detach().cpu().item()) else frontier
                        adjusted_scores = adjusted_scores + 0.5 * frontier_risk
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
            if accepted.any():
                select_mask = accepted
                fallback_used = False
                fallback_reason = "accepted_ncf" if gate_mode == "hard" else ("accepted_baseline" if gate_mode in {"none", "off"} else "accepted_priority_ncf")
            elif (cand_valid & conventional & stop_like).any():
                select_mask = cand_valid & conventional & stop_like
                fallback_used = True
                fallback_reason = "no_ncf_use_stop_like"
            elif (cand_valid & stop_like).any():
                select_mask = cand_valid & stop_like
                fallback_used = True
                fallback_reason = "no_conventional_use_neutral"
            elif (cand_valid & conventional).any():
                select_mask = cand_valid & conventional
                fallback_used = True
                fallback_reason = "no_stop_like_use_conventional"
            else:
                select_mask = cand_valid
                fallback_used = True
                fallback_reason = "no_conventional_use_valid"
            selected = int(self.torch.argmin(self.torch.where(select_mask, adjusted_scores, self.torch.full_like(adjusted_scores, float("inf")))).item()) if bool(select_mask.any().detach().cpu().item()) else 0
            selected_witness = witness[selected]
            selected_opr = opr[selected]
            diag = {
                "scenario_index": int(scenario_index) if scenario_index is not None else -1,
                "step": int(step) if step is not None else -1,
                "selected_candidate": int(selected),
                "accepted_candidates": int(accepted.sum().detach().cpu().item()),
                "valid_candidates": int(cand_valid.sum().detach().cpu().item()),
                "conventional_candidates": int((cand_valid & conventional).sum().detach().cpu().item()),
                "critical_agents": int(crit_mask.sum().detach().cpu().item()),
                "conflict_tokens": int(batch["map/conflict_region_valid"][0].bool().sum().detach().cpu().item()),
                "fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason,
                "max_witness_prob": float(selected_witness.max().detach().cpu().item()) if selected_witness.numel() else 0.0,
                "mean_witness_prob": float(selected_witness.mean().detach().cpu().item()) if selected_witness.numel() else 0.0,
                "mean_witness_uncertainty": float(uncertainty[selected].mean().detach().cpu().item()) if uncertainty[selected].numel() else 0.0,
                "max_witness_certificate": float(witness_cert[selected].max().detach().cpu().item()) if witness_cert[selected].numel() else 0.0,
                "min_opr": float(selected_opr.min().detach().cpu().item()) if selected_opr.numel() else 1.0,
                "mean_opr": float(selected_opr.mean().detach().cpu().item()) if selected_opr.numel() else 1.0,
                "score": float(scores[selected].detach().cpu().item()),
                "witness_threshold": float(self.witness_threshold),
                "alpha_opr": float(alpha),
                "gate_mode": str(gate_mode),
                    "method": str(method),
                "priority_hard_threshold": float(self.priority_hard_threshold),
                "accepted_primary_bad_candidates": int(primary_bad.sum().detach().cpu().item()) if primary_bad.numel() else 0,
                "severe_bad_candidates": int(severe_bad.sum().detach().cpu().item()) if severe_bad.numel() else 0,
                    "option_bad_candidates": int(option_bad.sum().detach().cpu().item()) if option_bad.numel() else 0,
                "selected_priority_max": float(priority[selected].max().detach().cpu().item()) if priority.numel() else 0.0,
                "selected_priority_mean": float(priority[selected].mean().detach().cpu().item()) if priority.numel() else 0.0,
                "selected_outcome_risk": float(outcome_risk[selected].detach().cpu().item()) if outcome_risk.numel() else 0.0,
                "beta_threshold": float(self.cfg.get("burden", {}).get("beta0_vehicle", 0.65)),
            }
            if burden is not None:
                bsel = self.torch.nan_to_num(burden[0, selected].float(), nan=0.0, posinf=2.0, neginf=0.0)
                diag["max_predicted_burden"] = float(bsel[crit_mask].max().detach().cpu().item()) if bool(crit_mask.any().detach().cpu().item()) else 0.0
            if c_i is not None:
                csel = self.torch.nan_to_num(c_i[0, selected].float(), nan=0.0, posinf=2.0, neginf=0.0)
                diag["max_predicted_c_i"] = float(csel[crit_mask].max().detach().cpu().item()) if bool(crit_mask.any().detach().cpu().item()) else 0.0
        self._last_diagnostics = diag
        self._diagnostics_log.append(diag)
        traj = batch_np["cowp/candidates/trajectory"][0, selected]
        return self._trajectory_to_action(state, agent_state, sdc_index, traj)

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
    action_mode: str = "delta_xy_yaw",
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
