from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from ocrap.data.schema import CandidatePrefix, CounterfactualFuture, RecoveryOption, SceneHistory
from ocrap.simulation.teacher.controllers import rollout_recovery_controller
from ocrap.simulation.teacher.margins import TeacherDiagnostics, teacher_margin, _artifact_margin_override
from ocrap.utils.seed import stable_seed


_JIT_CONTROL_ROLLOUT_CACHE: dict[tuple[int, int, int, int, float], Any] = {}
_WAYMAX_ENV_CACHE: dict[tuple, tuple[Any, str]] = {}
_JIT_CONTROL_ROLLOUT_WARNED = False


def _require_waymax():
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
        from waymax import config as wx_config  # type: ignore
        from waymax import datatypes  # type: ignore
        from waymax import dynamics  # type: ignore
        from waymax import env  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Waymax closed-loop backend requested but waymax/jax is not importable. "
            "Install Waymax and run with a valid WOMD TFExample dataset."
        ) from e
    return jax, jnp, wx_config, datatypes, dynamics, env


def _configure_jax(cfg: dict) -> None:
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    os.environ.setdefault("JAX_PLATFORMS", str(wx.get("jax_platforms", "cuda,cpu")))
    if not bool(wx.get("preallocate_gpu_memory", False)):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _as_np(x: Any) -> np.ndarray:
    try:
        import jax  # type: ignore

        return np.asarray(jax.device_get(x))
    except Exception:
        return np.asarray(x)


def _normalize_agent_time(x: Any, num_agents: int, num_steps: int | None = None, *, name: str = "field") -> np.ndarray:
    """Return an array in agent-time layout ``(A, T)``.

    Waymax can expose static fields such as length/width/height either as
    ``(A,)`` or ``(A, T)``.  Closed-loop rollout conversion needs a consistent
    layout before slicing by object and timestep.
    """
    arr = np.asarray(x)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 0:
        if num_steps is None:
            raise ValueError(f"Cannot infer time dimension for scalar {name}")
        return np.full((num_agents, num_steps), arr, dtype=arr.dtype)

    if arr.ndim == 1:
        if arr.size == num_agents:
            if num_steps is None:
                raise ValueError(f"Cannot infer time dimension for per-agent {name} with shape {arr.shape}")
            return np.broadcast_to(arr[:, None], (num_agents, num_steps))
        if num_steps is not None and arr.size == num_steps:
            return np.broadcast_to(arr[None, :], (num_agents, num_steps))
        if arr.size == 1 and num_steps is not None:
            return np.full((num_agents, num_steps), arr.reshape(()), dtype=arr.dtype)
        raise ValueError(f"Cannot normalize {name} with shape {arr.shape}; expected agent dimension {num_agents}")

    if arr.ndim == 2:
        if arr.shape[0] == num_agents and (num_steps is None or arr.shape[1] == num_steps):
            return arr
        if arr.shape[1] == num_agents and (num_steps is None or arr.shape[0] == num_steps):
            return arr.T
        if num_steps is not None:
            if arr.shape == (num_agents, 1):
                return np.broadcast_to(arr, (num_agents, num_steps))
            if arr.shape == (1, num_agents):
                return np.broadcast_to(arr.reshape(num_agents, 1), (num_agents, num_steps))
            if arr.shape == (1, num_steps):
                return np.broadcast_to(arr, (num_agents, num_steps))
            if arr.shape == (num_steps, 1):
                return np.broadcast_to(arr.T, (num_agents, num_steps))
        raise ValueError(
            f"Cannot normalize {name} with shape {arr.shape}; expected "
            f"({num_agents}, T) or (T, {num_agents})"
        )

    squeezed = np.squeeze(arr)
    if squeezed.shape != arr.shape:
        return _normalize_agent_time(squeezed, num_agents, num_steps, name=name)
    if num_steps is not None and arr.shape[-2:] == (num_agents, num_steps):
        return arr.reshape(-1, num_agents, num_steps)[0]
    if num_steps is not None and arr.shape[-2:] == (num_steps, num_agents):
        return arr.reshape(-1, num_steps, num_agents)[0].T
    raise ValueError(f"Cannot normalize {name} with shape {arr.shape}")


def _waymax_version() -> str:
    try:
        import waymax  # type: ignore

        return str(getattr(waymax, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _env_config(state: Any, cfg: dict, *, allow_new: bool = True, metrics: bool = True):
    _, _, wx_config, _, _, _ = _require_waymax()
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    metrics_names = tuple(wx.get("metrics_to_run", ["log_divergence", "overlap", "offroad", "sdc_wrongway", "sdc_off_route", "sdc_progression", "kinematic_infeasibility"]))
    return wx_config.EnvironmentConfig(
        max_num_objects=int(state.num_objects),
        init_steps=int(cfg.get("_waymax_init_steps_override", 1)),
        controlled_object=wx_config.ObjectType.SDC,
        compute_reward=False,
        allow_new_objects_after_warmup=bool(allow_new),
        metrics=wx_config.MetricsConfig(metrics_to_run=metrics_names if metrics else ("overlap",)),
    )


def _make_env(state: Any, cfg: dict, *, allow_new: bool = True, dynamics_name: str | None = None):
    _, _, _, _, dynamics, env = _require_waymax()
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    name = str(dynamics_name or wx.get("prefix_dynamics", "invertible_bicycle"))
    num_objects = int(getattr(state, "num_objects", 0))
    init_steps = int(cfg.get("_waymax_init_steps_override", 1))
    metrics_names = tuple(wx.get("metrics_to_run", ["log_divergence", "overlap", "offroad", "sdc_wrongway", "sdc_off_route", "sdc_progression", "kinematic_infeasibility"]))
    cache_key = (
        name,
        num_objects,
        init_steps,
        bool(allow_new),
        metrics_names,
        float(cfg.get("sample_rate_hz", 10.0)),
        float(cfg.get("control_limits", {}).get("a_max", 3.0)),
        float((cfg.get("waymax", {}) or {}).get("max_steering_curvature", 0.3)),
    )
    if bool(wx.get("cache_env_objects", False)) and cache_key in _WAYMAX_ENV_CACHE:
        return _WAYMAX_ENV_CACHE[cache_key]
    if name in {"state", "state_dynamics", "StateDynamics"}:
        dyn = dynamics.StateDynamics()
    elif name in {"invertible_bicycle", "bicycle", "InvertibleBicycleModel"}:
        dyn = dynamics.InvertibleBicycleModel(
            dt=1.0 / float(cfg.get("sample_rate_hz", 10.0)),
            max_accel=float(cfg.get("control_limits", {}).get("a_max", 3.0)),
            max_steering=float((cfg.get("waymax", {}) or {}).get("max_steering_curvature", 0.3)),
            normalize_actions=False,
        )
    else:
        raise ValueError(f"Unsupported Waymax dynamics {name}")
    made = (env.BaseEnvironment(dyn, _env_config(state, cfg, allow_new=allow_new, metrics=True)), name)
    if bool(wx.get("cache_env_objects", False)):
        _WAYMAX_ENV_CACHE[cache_key] = made
    return made


def _sdc_index(state: Any) -> int:
    return int(np.argmax(_as_np(state.object_metadata.is_sdc).astype(bool)))


def _global_from_local_xy(xy: np.ndarray, ego_xy: np.ndarray, ego_yaw: float) -> np.ndarray:
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return xy @ R.T + ego_xy[None, :]


def _global_from_local_heading(h: np.ndarray, ego_yaw: float) -> np.ndarray:
    return h + float(ego_yaw)


def _local_from_global_xy(xy: np.ndarray, ego_xy: np.ndarray, ego_yaw: float) -> np.ndarray:
    c = math.cos(-ego_yaw)
    s = math.sin(-ego_yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (xy - ego_xy[None, None, :]) @ R.T


def _set_agent_constant(arr: np.ndarray, slot: int, value: float) -> np.ndarray:
    """Set a per-agent or per-agent/time Waymax field without JAX .at loops."""
    out = np.array(arr, copy=True)
    if out.ndim == 0:
        return out
    if 0 <= slot < out.shape[0]:
        out[slot, ...] = value
    return out


def _traj_to_local_agent_arrays(state: Any, start_t: int, total_steps: int, order: list[int], ego_xy: np.ndarray, ego_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    tr = state.sim_trajectory
    x_full = _as_np(tr.x)
    total_T = int(x_full.shape[-1])
    total_A = int(x_full.shape[0])
    end_t = min(start_t + total_steps, total_T)
    num_objects = int(getattr(state, "num_objects", 0))
    x_all = _normalize_agent_time(_as_np(tr.x), num_objects, name="sim_trajectory.x")
    total_log_steps = int(x_all.shape[1])
    idx = np.asarray(order, dtype=np.int64)
    x = x_full[idx, start_t:end_t]
    y_all = _normalize_agent_time(_as_np(tr.y), num_objects, total_log_steps, name="sim_trajectory.y")
    z_all = _normalize_agent_time(_as_np(tr.z), num_objects, total_log_steps, name="sim_trajectory.z")
    vx_all = _normalize_agent_time(_as_np(tr.vel_x), num_objects, total_log_steps, name="sim_trajectory.vel_x")
    vy_all = _normalize_agent_time(_as_np(tr.vel_y), num_objects, total_log_steps, name="sim_trajectory.vel_y")
    yaw_all = _normalize_agent_time(_as_np(tr.yaw), num_objects, total_log_steps, name="sim_trajectory.yaw")
    valid_all = _normalize_agent_time(_as_np(tr.valid), num_objects, total_log_steps, name="sim_trajectory.valid").astype(bool)
    length_all = _normalize_agent_time(_as_np(tr.length), num_objects, total_log_steps, name="sim_trajectory.length")
    width_all = _normalize_agent_time(_as_np(tr.width), num_objects, total_log_steps, name="sim_trajectory.width")
    height_all = _normalize_agent_time(_as_np(tr.height), num_objects, total_log_steps, name="sim_trajectory.height")
    type_all = _normalize_agent_time(_as_np(state.object_metadata.object_types), num_objects, total_log_steps, name="object_metadata.object_types")
    x = x_all[idx, start_t:end_t]
    y = y_all[idx, start_t:end_t]
    z = z_all[idx, start_t:end_t]
    vx = vx_all[idx, start_t:end_t]
    vy = vy_all[idx, start_t:end_t]
    yaw = yaw_all[idx, start_t:end_t]
    valid = valid_all[idx, start_t:end_t]
    length = length_all[idx, start_t:end_t]
    width = width_all[idx, start_t:end_t]
    height = height_all[idx, start_t:end_t]
    typ = type_all[idx, start_t:end_t]
    T = total_steps
    A = len(idx)
    out = np.zeros((T, A, 16), dtype=np.float32)
    val = np.zeros((T, A), dtype=bool)
    if end_t <= start_t:
        return out, val
    xy = np.stack([x, y], axis=-1).transpose(1, 0, 2)
    xy_l = _local_from_global_xy(xy, ego_xy, ego_yaw)
    vel = np.stack([vx, vy], axis=-1).transpose(1, 0, 2)
    vel_l = _local_from_global_xy(vel, np.zeros(2, dtype=np.float32), ego_yaw)
    n = xy_l.shape[0]
    out[:n, :, 0:2] = xy_l
    out[:n, :, 2] = z.T
    out[:n, :, 3:5] = vel_l
    if n > 1:
        out[:n, :, 5] = np.gradient(out[:n, :, 3], 0.1, axis=0)
        out[:n, :, 6] = np.gradient(out[:n, :, 4], 0.1, axis=0)
    out[:n, :, 7] = yaw.T - float(ego_yaw)
    out[:n, :, 8] = np.sin(out[:n, :, 7])
    out[:n, :, 9] = np.cos(out[:n, :, 7])
    out[:n, :, 10] = length.T
    out[:n, :, 11] = width.T
    out[:n, :, 12] = height.T
    out[:n, :, 13] = typ.T
    out[:n, :, 14] = valid.T.astype(np.float32)
    out[:n, :, 15] = valid.T.astype(np.float32)
    val[:n] = valid.T
    return out, val


def _empty_action(num_objects: int, dim: int):
    _, jnp, _, datatypes, _, _ = _require_waymax()
    return datatypes.Action(data=jnp.zeros((num_objects, dim), dtype=jnp.float32), valid=jnp.zeros((num_objects, 1), dtype=jnp.bool_))


def _bicycle_action(num_objects: int, sdc: int, accel: float, steering_angle: float, wheelbase: float):
    _, jnp, _, datatypes, _, _ = _require_waymax()
    curvature = math.tan(float(steering_angle)) / max(float(wheelbase), 1e-3)
    data = jnp.zeros((num_objects, 2), dtype=jnp.float32)
    valid = jnp.zeros((num_objects, 1), dtype=jnp.bool_)
    data = data.at[sdc, 0].set(float(accel))
    data = data.at[sdc, 1].set(float(curvature))
    valid = valid.at[sdc, 0].set(True)
    return datatypes.Action(data=data, valid=valid)


def _state_action_from_local(prefix_state: np.ndarray, num_objects: int, sdc: int, ego_xy: np.ndarray, ego_yaw: float):
    _, jnp, _, datatypes, _, _ = _require_waymax()
    xy_g = _global_from_local_xy(np.asarray(prefix_state[:2], dtype=np.float32)[None, :], ego_xy, ego_yaw)[0]
    yaw_g = float(prefix_state[4] + ego_yaw)
    vx_l, vy_l = float(prefix_state[2]), float(prefix_state[3])
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    vx_g = c * vx_l - s * vy_l
    vy_g = s * vx_l + c * vy_l
    z = jnp.zeros((num_objects,), dtype=jnp.float32)
    valid = jnp.zeros((num_objects,), dtype=jnp.bool_)
    x = jnp.zeros((num_objects,), dtype=jnp.float32).at[sdc].set(float(xy_g[0]))
    y = jnp.zeros((num_objects,), dtype=jnp.float32).at[sdc].set(float(xy_g[1]))
    yaw = jnp.zeros((num_objects,), dtype=jnp.float32).at[sdc].set(yaw_g)
    vx = jnp.zeros((num_objects,), dtype=jnp.float32).at[sdc].set(float(vx_g))
    vy = jnp.zeros((num_objects,), dtype=jnp.float32).at[sdc].set(float(vy_g))
    valid = valid.at[sdc].set(True)
    return datatypes.TrajectoryUpdate(x=x, y=y, yaw=yaw, vel_x=vx, vel_y=vy, valid=valid).as_action()


def _use_jit_scan_rollouts(cfg: dict) -> bool:
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    return bool(wx.get("use_jit_scan_rollouts", False))


def _rollout_bicycle_controls_loop(st: Any, waymax_env: Any, controls: np.ndarray, cfg: dict):
    sdc = _sdc_index(st)
    wheelbase = float(cfg.get("wheelbase_m", 2.8))
    for k in range(int(controls.shape[0])):
        ctrl = controls[k]
        action = _bicycle_action(int(st.num_objects), sdc, float(ctrl[0]), float(ctrl[1]), wheelbase)
        st = waymax_env.step(st, action)
    return st


def _rollout_bicycle_controls_scan(st: Any, waymax_env: Any, controls: np.ndarray, cfg: dict):
    """Roll out a sequence of SDC bicycle controls with one JAX scan dispatch.

    The old code called ``waymax_env.step`` once from Python for every recovery
    step and for every option/future pair.  On real WOMD snippets this dominates
    wall time even when CUDA is visible, because thousands of tiny JAX dispatches
    keep the GPU under-utilized.  This helper preserves the same controls and
    final SimulatorState but moves the inner time loop into ``jax.lax.scan``.

    It is intentionally optional and falls back to the original loop on any JAX
    tracing/Waymax incompatibility so existing behavior is preserved.
    """
    global _JIT_CONTROL_ROLLOUT_WARNED
    if controls.size == 0:
        return st
    if not _use_jit_scan_rollouts(cfg):
        return _rollout_bicycle_controls_loop(st, waymax_env, controls, cfg)
    try:
        jax, jnp, _, datatypes, _, _ = _require_waymax()
        num_objects = int(st.num_objects)
        sdc = _sdc_index(st)
        steps = int(controls.shape[0])
        wheelbase = float(cfg.get("wheelbase_m", 2.8))
        key = (id(waymax_env), num_objects, sdc, steps, wheelbase)
        fn = _JIT_CONTROL_ROLLOUT_CACHE.get(key)
        if fn is None:
            valid_template = jnp.zeros((num_objects, 1), dtype=jnp.bool_).at[sdc, 0].set(True)

            def body(carry, ctrl):
                accel = ctrl[0]
                steer = ctrl[1]
                curvature = jnp.tan(steer) / max(wheelbase, 1e-3)
                data = jnp.zeros((num_objects, 2), dtype=jnp.float32)
                data = data.at[sdc, 0].set(accel)
                data = data.at[sdc, 1].set(curvature)
                action = datatypes.Action(data=data, valid=valid_template)
                return waymax_env.step(carry, action), None

            def rollout_fn(state, controls_jnp):
                final_state, _ = jax.lax.scan(body, state, controls_jnp, length=steps)
                return final_state

            fn = jax.jit(rollout_fn)
            _JIT_CONTROL_ROLLOUT_CACHE[key] = fn
        controls_jnp = jnp.asarray(np.asarray(controls[:, :2], dtype=np.float32))
        return fn(st, controls_jnp)
    except Exception as e:  # pragma: no cover - depends on optional Waymax/JAX versions
        if not _JIT_CONTROL_ROLLOUT_WARNED:
            print(f"[ocrap-profile] jit_scan_rollout disabled after fallback: {type(e).__name__}: {e}", flush=True)
            _JIT_CONTROL_ROLLOUT_WARNED = True
        return _rollout_bicycle_controls_loop(st, waymax_env, controls, cfg)


def _rollout_prefix(state0: Any, history: SceneHistory, prefix: CandidatePrefix, cfg: dict, *, allow_new: bool, dynamics_name: str | None = None):
    jax, _, _, _, _, _ = _require_waymax()
    t = int(history.metadata.get("waymax_planning_timestep", history.time_index))
    local_cfg = dict(cfg)
    local_cfg["_waymax_init_steps_override"] = t + 1
    waymax_env, dyn_name = _make_env(state0, local_cfg, allow_new=allow_new, dynamics_name=dynamics_name)
    rng = jax.random.PRNGKey(stable_seed("waymax", history.scene_id, history.time_index, prefix.macro_id) & 0x7FFFFFFF)
    if bool(history.metadata.get("_waymax_branch_from_current", False)):
        st = state0
    else:
        st = waymax_env.reset(state0, rng=rng)
    sdc = _sdc_index(state0)
    ego_xy = np.asarray(history.metadata.get("ego_global_xy", [0.0, 0.0]), dtype=np.float32)
    ego_yaw = float(history.metadata.get("ego_global_heading", 0.0))
    wheelbase = float(cfg.get("wheelbase_m", 2.8))
    for k in range(prefix.prefix_states.shape[0] - 1):
        if dyn_name in {"state", "state_dynamics", "StateDynamics"}:
            action = _state_action_from_local(prefix.prefix_states[k + 1], int(state0.num_objects), sdc, ego_xy, ego_yaw)
        else:
            ctrl = prefix.prefix_controls[min(k, prefix.prefix_controls.shape[0] - 1)] if prefix.prefix_controls.size else np.zeros(4, dtype=np.float32)
            action = _bicycle_action(int(state0.num_objects), sdc, float(ctrl[0]), float(ctrl[1]), wheelbase)
        st = waymax_env.step(st, action, rng=rng)
    return st, waymax_env, dyn_name


def _rollout_future_after_prefix(st: Any, waymax_env: Any, steps: int, cfg: dict, *, coast_accel: float = 0.0):
    n = max(0, int(steps))
    if n <= 0:
        return st
    controls = np.zeros((n, 2), dtype=np.float32)
    controls[:, 0] = float(coast_accel)
    return _rollout_bicycle_controls_scan(st, waymax_env, controls, cfg)


def _metric_summary(waymax_env: Any, st: Any, sdc: int) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        metrics = waymax_env.metrics(st)
        for name, res in metrics.items():
            val = _as_np(getattr(res, "value", res))
            if val.ndim > 0 and val.shape[-1] > sdc:
                out[str(name)] = float(val.reshape(-1, val.shape[-1])[-1, sdc])
            else:
                out[str(name)] = float(np.asarray(val).reshape(-1)[-1])
    except Exception as e:
        out["metrics_error"] = float("nan")
        out["metrics_error_present"] = 1.0
    return out


def _base_metadata(history: SceneHistory, prefix: CandidatePrefix, source: str, *, policy: str, scenario_augmented: bool, allow_new: bool, dyn_name: str, seed: int, extra: dict | None = None) -> dict[str, Any]:
    meta = {
        "runtime_backend": "waymax_closed_loop",
        "waymax_runtime": True,
        "waymax_version": _waymax_version(),
        "dynamics_model": dyn_name,
        "prefix_dynamics_model": dyn_name,
        "debug_alignment_dynamics_model": str((history.metadata or {}).get("debug_alignment_dynamics_model", "StateDynamics")),
        "sim_agent_policy": policy,
        "scenario_augmented": bool(scenario_augmented),
        "allow_object_injection": bool(allow_new),
        "controlled_object_ids": ["sdc"],
        "action_dim": 2 if dyn_name not in {"state", "state_dynamics", "StateDynamics"} else 5,
        "rollout_start_timestep": int(history.metadata.get("waymax_planning_timestep", history.time_index)),
        "prefix_steps": int(prefix.prefix_states.shape[0]),
        "recovery_steps": int(round(float((history.metadata or {}).get("recovery_horizon_s", 0.0)))) if False else 0,
        "rng_seed": int(seed),
        "future_source": source,
        "planning_time_not_fabricated": True,
    }
    if extra:
        meta.update(extra)
    return meta


def _is_unknown_spawn(history: SceneHistory, local_xy: np.ndarray, cfg: dict) -> tuple[bool, bool]:
    try:
        mask = history.occ_mask
        radius = float(cfg.get("local_radius_m", 80.0))
        res = float(cfg.get("bev_resolution_m", 2.0))
        ix = int(round((float(local_xy[0]) + radius) / max(res, 1e-6)))
        iy = int(round((float(local_xy[1]) + radius) / max(res, 1e-6)))
        if iy < 0 or ix < 0 or iy >= mask.shape[1] or ix >= mask.shape[2]:
            return False, False
        unknown = bool(mask[2, iy, ix] > 0.5) if mask.shape[0] > 2 else False
        visible_free = bool(mask[0, iy, ix] > 0.5) if mask.shape[0] > 0 else False
        return unknown, visible_free
    except Exception:
        return False, False


def _find_natural_hidden_metadata(st: Any, history: SceneHistory, cfg: dict) -> dict[str, Any]:
    t = int(history.metadata.get("waymax_planning_timestep", history.time_index))
    T_p = int(round(float(cfg.get("prefix_horizon_s", 1.0)) * float(cfg.get("sample_rate_hz", 10.0))))
    delay = int(cfg.get("hidden_emergence_delay_steps", 2))
    valid = _as_np(st.log_trajectory.valid).astype(bool)
    x = _as_np(st.log_trajectory.x)
    y = _as_np(st.log_trajectory.y)
    order = [int(i) for i in history.metadata.get("agent_order", list(range(valid.shape[0])))]
    ego_xy = np.asarray(history.metadata.get("ego_global_xy", [0.0, 0.0]), dtype=np.float32)
    ego_yaw = float(history.metadata.get("ego_global_heading", 0.0))
    candidates = []
    for a in order[1:]:
        if a >= valid.shape[0] or valid[a, t]:
            continue
        future_idx = np.where(valid[a, min(t + T_p + delay, valid.shape[1] - 1) :])[0]
        if future_idx.size == 0:
            continue
        first = int(min(t + T_p + delay, valid.shape[1] - 1) + future_idx[0])
        local = _local_from_global_xy(np.asarray([[[x[a, first], y[a, first]]]], dtype=np.float32), ego_xy, ego_yaw)[0, 0]
        unknown, visible_free = _is_unknown_spawn(history, local, cfg)
        candidates.append((a, first, local, unknown, visible_free))
    if not candidates:
        return {"hidden_emergence": False}
    a, first, local, unknown, visible_free = candidates[0]
    return {
        "hidden_emergence": True,
        "hidden_actor_object_index": int(a),
        "hidden_start_step": int(first - t),
        "hidden_start_ge_prefix_plus_delay": bool(first - t >= T_p + delay),
        "from_unknown_mask": bool(unknown),
        "spawn_in_visible_free": bool(visible_free),
        "hidden_spawn_xy": [float(local[0]), float(local[1])],
        "hidden_intent": "natural_log_playback",
    }


def _sample_unknown_spawn(history: SceneHistory, cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    mask = history.occ_mask
    if mask.size == 0 or mask.shape[0] < 6:
        return None
    wx_cfg = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    unknown_only = bool(wx_cfg.get("augmented_hidden_from_unknown_only", True))
    unknown = mask[2] > 0.5
    drivable = mask[5] > 0.5
    visible_free = mask[0] > 0.5
    occupied = mask[1] > 0.5
    route = mask[4] > 0.5
    legal = unknown & drivable & ~visible_free & ~occupied
    route_legal = legal & route
    cells = np.argwhere(route_legal if route_legal.any() else legal)
    from_unknown = True
    fallback_visible_free = False
    if cells.size == 0 and not unknown_only:
        # Some WOMD/test shards have almost no legal unknown-drivable cells, which
        # silently disables hidden-yield/accelerate artifact mining.  For explicit
        # stress-test builds, allow a deterministic route/drivable fallback while
        # tagging provenance so paper-scale occlusion-only claims can still filter
        # these examples out.
        fallback = drivable & ~occupied
        route_fallback = fallback & route
        cells = np.argwhere(route_fallback if route_fallback.any() else fallback)
        from_unknown = False
        if cells.size == 0:
            return None
    if cells.size == 0:
        return None
    radius = float(cfg.get("local_radius_m", 80.0))
    res = float(cfg.get("bev_resolution_m", 2.0))
    xy = np.stack([(cells[:, 1] + 0.5) * res - radius, (cells[:, 0] + 0.5) * res - radius], axis=-1).astype(np.float32)
    score = np.abs(xy[:, 1]) + 0.02 * np.maximum(-xy[:, 0], 0.0) - 0.01 * xy[:, 0]
    order = np.argsort(score, kind="mergesort")[: max(1, min(50, len(score)))]
    k = int(rng.choice(order))
    iy, ix = cells[k]
    loc = xy[k]
    if not from_unknown:
        fallback_visible_free = bool(visible_free[int(iy), int(ix)])
    return loc, {
        "hidden_spawn_xy": [float(loc[0]), float(loc[1])],
        "hidden_spawn_cell": [int(iy), int(ix)],
        "from_unknown_mask": bool(from_unknown),
        "spawn_in_visible_free": bool(fallback_visible_free),
        "synthetic_hidden_spawn_fallback": bool(not from_unknown),
    }


def can_mine_augmented_hidden_pair(history: SceneHistory, prefix: CandidatePrefix, cfg: dict) -> bool:
    """Cheap preflight for artifact-pair mining before expensive Waymax rollouts.

    The full hidden yield/accelerate branches are still validated after rollout;
    this only rejects prefixes for which the pair is structurally impossible,
    e.g. no legal unknown drivable spawn cell or no remaining log horizon.
    """
    if int(cfg.get("num_targeted_futures", 8)) < 2:
        return False
    state0 = history.metadata.get("_waymax_state")
    if state0 is None:
        return False
    seed = stable_seed("waymax-hidden-preflight", history.scene_id, history.time_index, prefix.macro_id)
    if _sample_unknown_spawn(history, cfg, np.random.default_rng(seed)) is None:
        return False
    try:
        valid = _as_np(state0.log_trajectory.valid).astype(bool)
        if valid.ndim < 2 or valid.shape[0] < 2:
            return False
        t0 = int(history.metadata.get("waymax_planning_timestep", history.time_index))
        T_p = int(prefix.prefix_states.shape[0])
        delay = int(cfg.get("hidden_emergence_delay_steps", 2))
        return bool(t0 + T_p + delay < valid.shape[-1])
    except Exception:
        return True


def _cfg_without_artifact_override(cfg: dict) -> dict:
    """Return a shallow config copy for structural margins without hard-coded artifact overrides."""
    local = dict(cfg)
    art = dict(local.get("artifact", {}) or {})
    art["use_margin_override"] = False
    local["artifact"] = art
    return local


def _artifact_override_adjusted_value(val: float, option: RecoveryOption, future: CounterfactualFuture, cfg: dict) -> tuple[float, bool]:
    """Apply the mined-pair branch label consistently to rolled and screened options."""
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    if not bool(cfg.get("artifact", {}).get("use_margin_override", True)):
        return float(val), False
    if not bool(future.metadata.get("scenario_augmented", False)):
        return float(val), False
    if not bool(wx.get("apply_artifact_override_to_screened_options", True)):
        return float(val), False
    override = _artifact_margin_override(option, future, cfg)
    if override is None:
        return float(val), False
    # Keep the old post-rollout semantics: compatible branches are forced
    # positive, incompatible branches are forced strongly negative.
    if override >= 0.0:
        return max(float(val), float(override)), True
    return min(float(val), float(override)), True


def _augment_hidden_reference(state: Any, history: SceneHistory, prefix: CandidatePrefix, cfg: dict, *, branch: str, seed: int):
    _, jnp, _, _, _, _ = _require_waymax()
    rng = np.random.default_rng(seed)
    spawn = _sample_unknown_spawn(history, cfg, rng)
    if spawn is None:
        return None, {"skip_reason": "no_unknown_drivable_spawn"}
    local_xy, smeta = spawn
    ego_xy = np.asarray(history.metadata.get("ego_global_xy", [0.0, 0.0]), dtype=np.float32)
    ego_yaw = float(history.metadata.get("ego_global_heading", 0.0))
    gxy = _global_from_local_xy(local_xy[None, :], ego_xy, ego_yaw)[0]
    heading = float(prefix.prefix_states[-1, 4] + ego_yaw)
    t0 = int(history.metadata.get("waymax_planning_timestep", history.time_index))
    T_p = int(prefix.prefix_states.shape[0])
    delay = int(cfg.get("hidden_emergence_delay_steps", 2))
    start = min(int(_as_np(state.log_trajectory.x).shape[-1]) - 1, t0 + T_p + delay)
    tr = state.log_trajectory
    valid_np = _as_np(tr.valid).astype(bool)
    sdc = _sdc_index(state)
    candidates = [a for a in range(valid_np.shape[0]) if a != sdc]
    empty = [a for a in candidates if not valid_np[a].any()]
    slot = int(empty[0]) if empty else int(min(candidates, key=lambda a: int(valid_np[a].sum())))
    total_T = int(valid_np.shape[1])
    # Keep the augmentation logic identical to the old scalar update path, but
    # perform all edits on host NumPy arrays and send one batched replacement to
    # JAX.  Repeated ``jnp.ndarray.at[...].set`` calls inside a Python loop are a
    # major source of dispatch/compile overhead during strict/stress generation.
    xs = np.array(_as_np(tr.x), copy=True)
    ys = np.array(_as_np(tr.y), copy=True)
    zs = np.array(_as_np(tr.z), copy=True)
    vxs = np.array(_as_np(tr.vel_x), copy=True)
    vys = np.array(_as_np(tr.vel_y), copy=True)
    yaws = np.array(_as_np(tr.yaw), copy=True)
    valids = np.array(valid_np, copy=True)
    xs[slot, :] = 0.0
    ys[slot, :] = 0.0
    zs[slot, :] = 0.0
    vxs[slot, :] = 0.0
    vys[slot, :] = 0.0
    yaws[slot, :] = heading
    valids[slot, :] = False
    speed0 = max(1.0, float(prefix.prefix_states[-1, 6]))
    accel = -0.5 if branch == "yield" else 1.2
    intent_speed = max(1.0, speed0 - 2.0) if branch == "yield" else max(3.0, speed0 + 2.0)
    if start < total_T:
        taus = np.arange(start, total_T, dtype=np.float32)
        dt = (taus - float(start)) / float(cfg.get("sample_rate_hz", 10.0))
        v = np.maximum(0.0, intent_speed + accel * dt)
        dist = intent_speed * dt + 0.5 * accel * dt * dt
        c = math.cos(heading)
        s = math.sin(heading)
        idx = np.arange(start, total_T)
        xs[slot, idx] = float(gxy[0]) + dist * c
        ys[slot, idx] = float(gxy[1]) + dist * s
        vxs[slot, idx] = v * c
        vys[slot, idx] = v * s
        yaws[slot, idx] = heading
        valids[slot, idx] = True
    length = _set_agent_constant(_as_np(tr.length), slot, 4.8)
    width = _set_agent_constant(_as_np(tr.width), slot, 2.0)
    height = _set_agent_constant(_as_np(tr.height), slot, 1.6)
    new_tr = tr.replace(
        x=jnp.asarray(xs),
        y=jnp.asarray(ys),
        z=jnp.asarray(zs),
        vel_x=jnp.asarray(vxs),
        vel_y=jnp.asarray(vys),
        yaw=jnp.asarray(yaws),
        valid=jnp.asarray(valids),
        length=jnp.asarray(length),
        width=jnp.asarray(width),
        height=jnp.asarray(height),
    )
    md = state.object_metadata
    obj_types = jnp.array(_as_np(md.object_types)).at[slot].set(1)
    is_valid = jnp.array(_as_np(md.is_valid)).at[slot].set(True)
    ids = jnp.array(_as_np(md.ids)).at[slot].set(-100000 - int(seed % 100000))
    is_sdc = jnp.array(_as_np(md.is_sdc)).at[slot].set(False)
    is_modeled = jnp.array(_as_np(md.is_modeled)).at[slot].set(False)
    ooi = jnp.array(_as_np(md.objects_of_interest)).at[slot].set(False)
    is_ctrl = jnp.array(_as_np(md.is_controlled)).at[slot].set(False)
    new_md = md.replace(ids=ids, object_types=obj_types, is_valid=is_valid, is_sdc=is_sdc, is_modeled=is_modeled, objects_of_interest=ooi, is_controlled=is_ctrl)
    meta = dict(smeta)
    meta.update({
        "hidden_emergence": True,
        "hidden_intent": branch,
        "artifact_branch": branch,
        "targeted_type": f"waymax_hidden_vehicle_{branch}",
        "artifact_mined": True,
        "hidden_actor_object_index": int(slot),
        "hidden_start_step": int(start - t0),
        "hidden_start_ge_prefix_plus_delay": bool(start - t0 >= T_p + delay),
        "hidden_invalid_spawn": False,
        "injected_reference_log_playback": True,
        "injected_replaced_logged_agent": bool(not empty),
    })
    return state.replace(log_trajectory=new_tr, object_metadata=new_md), meta


def _augment_visible_reference(state: Any, history: SceneHistory, prefix: CandidatePrefix, cfg: dict, *, branch: str, seed: int):
    """Perturb a currently visible non-SDC actor so observation labels include negatives.

    Hidden yield/accelerate roots create the oracle-artifact alias pairs needed by
    OC-RAP, but they intentionally look identical at the post-prefix observation.
    This augmentation adds a small set of visible counterfactual roots whose actor
    is already observable at the planning instant and whose position/speed changes
    during the executable prefix.  They are useful for training and diagnosing the
    observation-equivalence kernel without leaking hidden branch identity.
    """
    _, jnp, _, _, _, _ = _require_waymax()
    rng = np.random.default_rng(seed)
    t0 = int(history.metadata.get("waymax_planning_timestep", history.time_index))
    T_p = int(prefix.prefix_states.shape[0])
    tr = state.log_trajectory
    x = _normalize_agent_time(_as_np(tr.x), int(state.num_objects), name="log_trajectory.x")
    y = _normalize_agent_time(_as_np(tr.y), int(state.num_objects), x.shape[1], name="log_trajectory.y")
    vx = _normalize_agent_time(_as_np(tr.vel_x), int(state.num_objects), x.shape[1], name="log_trajectory.vel_x")
    vy = _normalize_agent_time(_as_np(tr.vel_y), int(state.num_objects), x.shape[1], name="log_trajectory.vel_y")
    yaw = _normalize_agent_time(_as_np(tr.yaw), int(state.num_objects), x.shape[1], name="log_trajectory.yaw")
    valid = _normalize_agent_time(_as_np(tr.valid), int(state.num_objects), x.shape[1], name="log_trajectory.valid").astype(bool)
    sdc = _sdc_index(state)
    ego_xy = np.asarray(history.metadata.get("ego_global_xy", [0.0, 0.0]), dtype=np.float32)
    ego_yaw = float(history.metadata.get("ego_global_heading", 0.0))
    candidates: list[tuple[float, int]] = []
    if 0 <= t0 < valid.shape[1]:
        for a in range(valid.shape[0]):
            if a == sdc or not valid[a, t0]:
                continue
            local = _local_from_global_xy(np.asarray([[[x[a, t0], y[a, t0]]]], dtype=np.float32), ego_xy, ego_yaw)[0, 0]
            # Prefer actors that are within the ego observation region and not too far.
            dist = float(np.linalg.norm(local))
            if dist <= float(cfg.get("visible_root_max_distance_m", 45.0)) and local[0] > -10.0:
                candidates.append((dist, a))
    if not candidates:
        return None, {"skip_reason": "no_visible_actor_for_perturbation"}
    candidates.sort(key=lambda z: z[0])
    # Pick among a few near actors deterministically but not always the closest.
    _, slot = candidates[int(rng.integers(0, min(len(candidates), 3)))]
    total_T = x.shape[1]
    start = min(max(t0 + 1, 0), total_T - 1)
    end = min(total_T, t0 + max(T_p, 2) + int(cfg.get("visible_root_extra_steps", 8)))
    accel = -1.5 if branch == "visible_brake" else 1.5
    lateral = -0.25 if branch == "visible_left" else (0.25 if branch == "visible_right" else 0.0)
    # Vectorized host-side edit; same trajectory as the old per-timestep JAX
    # updates, with far fewer device dispatches.
    xs = np.array(_as_np(tr.x), copy=True)
    ys = np.array(_as_np(tr.y), copy=True)
    vxs = np.array(_as_np(tr.vel_x), copy=True)
    vys = np.array(_as_np(tr.vel_y), copy=True)
    yaws = np.array(_as_np(tr.yaw), copy=True)
    # Use the actor's current heading as a local tangent and apply a modest speed
    # perturbation over the prefix.  This keeps the branch plausible but visible.
    heading0 = float(yaw[slot, t0])
    speed0 = float(max(0.0, math.hypot(float(vx[slot, t0]), float(vy[slot, t0]))))
    x0 = float(x[slot, t0])
    y0 = float(y[slot, t0])
    nx = -math.sin(heading0)
    ny = math.cos(heading0)
    if start < end:
        taus = np.arange(start, end, dtype=np.float32)
        dt = (taus - float(t0)) / float(cfg.get("sample_rate_hz", 10.0))
        v = np.maximum(0.0, speed0 + accel * dt)
        dist = speed0 * dt + 0.5 * accel * dt * dt
        lat = lateral * dt
        c = math.cos(heading0)
        s = math.sin(heading0)
        idx = np.arange(start, end)
        xs[slot, idx] = x0 + dist * c + lat * nx
        ys[slot, idx] = y0 + dist * s + lat * ny
        vxs[slot, idx] = v * c
        vys[slot, idx] = v * s
        yaws[slot, idx] = heading0
    new_tr = tr.replace(x=jnp.asarray(xs), y=jnp.asarray(ys), vel_x=jnp.asarray(vxs), vel_y=jnp.asarray(vys), yaw=jnp.asarray(yaws))
    meta = {
        "visible_perturbation": True,
        "visible_actor_object_index": int(slot),
        "visible_branch": branch,
        "targeted_type": f"waymax_visible_actor_{branch}",
        "observation_negative_candidate": True,
        "hidden_emergence": False,
        "artifact_mined": False,
    }
    return state.replace(log_trajectory=new_tr), meta

def _valid_hidden_provenance(meta: dict[str, Any]) -> bool:
    if not bool(meta.get("hidden_emergence", False)):
        return False
    return (
        bool(meta.get("from_unknown_mask", False))
        and not bool(meta.get("spawn_in_visible_free", False))
        and not bool(meta.get("hidden_invalid_spawn", False))
    )

def _demote_invalid_hidden_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Keep natural-emergence diagnostics without exposing them as hidden roots."""
    out: dict[str, Any] = {"hidden_emergence": False}
    for k, v in meta.items():
        if k == "hidden_emergence":
            continue
        out[f"natural_{k}"] = v
    out["natural_hidden_candidate"] = True
    out["natural_hidden_rejected_reason"] = "not_from_unknown_mask"
    return out

def _make_future_from_state(
    fid: int,
    source: str,
    prior: float,
    st: Any,
    history: SceneHistory,
    prefix: CandidatePrefix,
    cfg: dict,
    waymax_env: Any,
    dyn_name: str,
    meta_extra: dict[str, Any] | None = None,
    state_after_prefix: Any | None = None,
    materialization_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, float]]] | None = None,
) -> CounterfactualFuture:
    order = [int(i) for i in history.metadata.get("agent_order", list(range(int(st.num_objects))))]
    ego_xy = np.asarray(history.metadata.get("ego_global_xy", [0.0, 0.0]), dtype=np.float32)
    ego_yaw = float(history.metadata.get("ego_global_heading", 0.0))
    t = int(history.metadata.get("waymax_planning_timestep", history.time_index))
    total = int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * float(cfg.get("sample_rate_hz", 10.0))))
    wx_cfg = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    cache_key = (id(st), id(waymax_env))
    cached = materialization_cache.get(cache_key) if materialization_cache is not None else None
    if cached is None:
        arr, val = _traj_to_local_agent_arrays(st, t, total, order, ego_xy, ego_yaw)
        metrics = _metric_summary(waymax_env, st, _sdc_index(st)) if bool(wx_cfg.get("compute_future_metrics", True)) else {}
        if materialization_cache is not None:
            # Futures may intentionally share the same Waymax rollout state but
            # differ in latent metadata (e.g. low-friction vs control-delay
            # teacher branches).  Reuse only trajectory extraction and simulator
            # metrics; metadata remains future-specific.
            materialization_cache[cache_key] = (arr, val, dict(metrics))
    else:
        arr, val, metrics = cached
    seed = stable_seed("waymax-future", history.scene_id, history.time_index, prefix.macro_id, fid)
    base = _base_metadata(history, prefix, source, policy="log_playback", scenario_augmented=bool(meta_extra and meta_extra.get("scenario_augmented", False)), allow_new=bool((cfg.get("waymax", {}) or {}).get("allow_new_objects_after_warmup", True)), dyn_name=dyn_name, seed=seed, extra=meta_extra)
    if bool(wx_cfg.get("detect_natural_hidden_emergence", True)):
        nat = _find_natural_hidden_metadata(st, history, cfg)
        if nat.get("hidden_emergence") and not base.get("hidden_emergence"):
            if _valid_hidden_provenance(nat):
                base.update(nat)
            else:
                base.update(_demote_invalid_hidden_metadata(nat))
    base["waymax_metrics"] = dict(metrics)
    base["recovery_steps"] = int(round(float(cfg.get("recovery_horizon_s", 4.0)) * float(cfg.get("sample_rate_hz", 10.0))))
    fut = CounterfactualFuture(fid, source, prior, arr, val, base)
    setattr(fut, "_waymax_state_after_prefix", state_after_prefix)
    setattr(fut, "_waymax_env", waymax_env)
    setattr(fut, "_waymax_rollout_state", st)
    return fut


def generate_waymax_counterfactual_futures(history: SceneHistory, prefix: CandidatePrefix, cfg: dict) -> list[CounterfactualFuture]:
    _configure_jax(cfg)
    state0 = history.metadata.get("_waymax_state")
    if state0 is None:
        raise ValueError("Waymax backend requires SceneHistory.metadata['_waymax_state']; use data_source=womd with simulation_backend=waymax_closed_loop.")
    total = int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * float(cfg.get("sample_rate_hz", 10.0))))
    T_p = int(prefix.prefix_states.shape[0])
    post_steps = max(0, total - max(1, T_p))
    priors = cfg.get("future_priors", {})
    futures: list[CounterfactualFuture] = []

    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    allow_new = bool(wx.get("allow_new_objects_after_warmup", True))
    st_prefix, wx_env, dyn_name = _rollout_prefix(state0, history, prefix, cfg, allow_new=allow_new)
    postprefix_rollout_cache: dict[tuple[int, int, int, float], Any] = {}
    future_materialization_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}

    def rollout_post_cached(st_after_prefix: Any, waymax_env: Any, accel: float) -> Any:
        if not bool(wx.get("cache_postprefix_rollouts", True)):
            return _rollout_future_after_prefix(st_after_prefix, waymax_env, post_steps, cfg, coast_accel=accel)
        key = (id(st_after_prefix), id(waymax_env), int(post_steps), round(float(accel), 6))
        cached = postprefix_rollout_cache.get(key)
        if cached is not None:
            return cached
        out = _rollout_future_after_prefix(st_after_prefix, waymax_env, post_steps, cfg, coast_accel=accel)
        postprefix_rollout_cache[key] = out
        return out

    st_roll = rollout_post_cached(st_prefix, wx_env, 0.0)
    futures.append(
        _make_future_from_state(
            0,
            "replay",
            float(priors.get("replay", 0.25)),
            st_roll,
            history,
            prefix,
            cfg,
            wx_env,
            dyn_name,
            {"rollout_variant": "natural_log_playback", "scenario_augmented": False, "waymax_prefix_rollout_reused": False},
            state_after_prefix=st_prefix,
            materialization_cache=future_materialization_cache,
        )
    )

    # The prefix execution is identical for all non-augmented futures of this
    # (history, prefix).  Reusing the post-prefix SimulatorState avoids repeating
    # a JAX reset + T_p environment steps for every reactive/targeted branch.
    # Branch-specific diversity is introduced only in the post-prefix rollout or
    # in an augmented reference trajectory, so this reuse is semantically exact.
    n_reactive = int(cfg.get("num_reactive_futures", 4))
    reactive_total = float(priors.get("reactive", 0.35))
    for i in range(n_reactive):
        accel = [-1.0, -0.3, 0.3, 0.8][i % 4]
        str_i = rollout_post_cached(st_prefix, wx_env, accel)
        futures.append(
            _make_future_from_state(
                len(futures),
                "reactive",
                reactive_total / max(n_reactive, 1),
                str_i,
                history,
                prefix,
                cfg,
                wx_env,
                dyn_name,
                {
                    "rollout_variant": "waymax_log_playback_sdc_coast",
                    "ego_after_prefix_accel": float(accel),
                    "scenario_augmented": False,
                    "waymax_prefix_rollout_reused": True,
                    "teacher_base_reuses_replay_prefix_state": True,
                },
                state_after_prefix=st_prefix,
                materialization_cache=future_materialization_cache,
            )
        )

    quality = cfg.get("dataset_quality", {}) if isinstance(cfg.get("dataset_quality", {}), dict) else {}
    require_artifact_pair = bool(quality.get("require_artifact_pairs", False))
    n_targeted = int(cfg.get("num_targeted_futures", 8))
    targeted_total = float(priors.get("targeted", 0.40))
    targeted_added = 0
    hidden_branches_added: set[str] = set()
    artifact_cfg = cfg.get("artifact", {}) if isinstance(cfg.get("artifact", {}), dict) else {}
    mine_prob = float(artifact_cfg.get("mine_probability", 1.0 if artifact_cfg.get("force_mine", True) else 0.0))
    mine_rng = np.random.default_rng(stable_seed("waymax-hidden-mine", history.scene_id, history.time_index, prefix.macro_id))
    stochastic_mine = float(mine_rng.random()) < max(0.0, min(1.0, mine_prob))
    mine_hidden = (bool(wx.get("enable_augmented_hidden_roots", True)) or require_artifact_pair) and (require_artifact_pair or stochastic_mine)
    if mine_hidden:
        for branch in ["yield", "accelerate"]:
            if targeted_added >= n_targeted:
                break
            seed = stable_seed("waymax-hidden", history.scene_id, history.time_index, prefix.macro_id, branch)
            aug_state, ameta = _augment_hidden_reference(state0, history, prefix, cfg, branch=branch, seed=seed)
            if aug_state is None:
                continue
            stp, env_a, dyn_a = _rollout_prefix(aug_state, history, prefix, cfg, allow_new=True)
            str_a = rollout_post_cached(stp, env_a, 0.0)
            ameta.update({"scenario_augmented": True, "artifact_pair_key": f"{history.scene_id}:{history.time_index}:{prefix.macro_id}", "rollout_variant": "augmented_hidden_log_playback"})
            futures.append(_make_future_from_state(len(futures), "targeted", targeted_total / max(n_targeted, 1), str_a, history, prefix, cfg, env_a, dyn_a, ameta, state_after_prefix=stp, materialization_cache=future_materialization_cache))
            hidden_branches_added.add(branch)
            targeted_added += 1
    if require_artifact_pair and not {"yield", "accelerate"}.issubset(hidden_branches_added):
        for f in futures:
            f.metadata["artifact_pair_missing"] = True
            f.metadata["artifact_pair_required"] = True
            f.metadata["artifact_pair_branches_present"] = sorted(hidden_branches_added)
    if bool(wx.get("enable_visible_perturbation_roots", True)):
        for branch in ["visible_brake", "visible_accelerate"]:
            if targeted_added >= n_targeted:
                break
            seed = stable_seed("waymax-visible", history.scene_id, history.time_index, prefix.macro_id, branch)
            aug_state, ameta = _augment_visible_reference(state0, history, prefix, cfg, branch=branch, seed=seed)
            if aug_state is None:
                continue
            stp, env_v, dyn_v = _rollout_prefix(aug_state, history, prefix, cfg, allow_new=allow_new)
            str_v = rollout_post_cached(stp, env_v, 0.0)
            ameta.update({"scenario_augmented": True, "rollout_variant": "augmented_visible_actor_log_playback"})
            futures.append(_make_future_from_state(len(futures), "targeted", targeted_total / max(n_targeted, 1), str_v, history, prefix, cfg, env_v, dyn_v, ameta, state_after_prefix=stp, materialization_cache=future_materialization_cache))
            targeted_added += 1
    # Fill requested stress futures before generic SDC control stress.  The
    # top-level surrogate path uses ``targeted_future_kinds``; older commands put
    # the same list under ``waymax.targeted_future_kinds``.  Honor both so
    # contact/post-contact shards do not silently degrade into ordinary
    # near-contact SDC acceleration/braking futures.
    requested_kinds = cfg.get("targeted_future_kinds", None)
    if requested_kinds is None:
        requested_kinds = wx.get("targeted_future_kinds", None)
    if not isinstance(requested_kinds, (list, tuple)) or not requested_kinds:
        requested_kinds = []
    requested_kinds = [str(k) for k in requested_kinds]
    kind_cursor = 0
    while targeted_added < n_targeted and kind_cursor < max(1, len(requested_kinds)) * 3:
        if not requested_kinds:
            break
        kind = requested_kinds[kind_cursor % len(requested_kinds)]
        kind_cursor += 1
        if kind in {"hidden_vehicle_yields", "hidden_vehicle_accelerates"}:
            # Hidden branches are mined above because they require reference-log
            # augmentation, not a simple post-prefix SDC rollout.
            continue
        meta = {
            "scenario_augmented": False,
            "targeted_type": f"waymax_{kind}",
            "recovery_relevant": True,
            "waymax_prefix_rollout_reused": True,
            "teacher_base_reuses_replay_prefix_state": True,
        }
        accel = -2.0 if targeted_added % 2 == 0 else 1.2
        if kind == "contact_impulse_surrogate":
            # Waymax does not expose an API-level collision impulse perturbation
            # here, but the margin teacher and regime tag need to know that this
            # latent branch is a post-contact recovery branch.  Use a normal
            # Waymax rollout state plus explicit contact-surrogate metadata.
            accel = -0.5
            rng = np.random.default_rng(stable_seed("waymax-contact-surrogate", history.scene_id, history.time_index, prefix.macro_id, targeted_added))
            meta.update({
                "contact_surrogate": True,
                "yaw_rate_impulse": float(rng.choice([-0.55, 0.55])),
                "lateral_velocity_impulse": float(rng.choice([-1.5, 1.5])),
            })
        elif kind == "secondary_collision_approach":
            accel = -1.2
            meta.update({"secondary_collision_approach": True, "contact_surrogate": bool(prefix.diagnostics.get("prefix_contact", False))})
        elif kind == "low_friction_braking":
            accel = -3.0
            meta.update({"low_friction": True})
        elif kind == "control_delay_noise":
            accel = 0.8
            meta.update({"control_delay_noise": True})
        else:
            # visible actor perturbations and generic SDC stress futures are
            # already covered above/below; skip unknown non-Waymax stress kinds.
            continue
        str_t = rollout_post_cached(st_prefix, wx_env, accel)
        meta["ego_after_prefix_accel"] = float(accel)
        futures.append(_make_future_from_state(len(futures), "targeted", targeted_total / max(n_targeted, 1), str_t, history, prefix, cfg, wx_env, dyn_name, meta, state_after_prefix=st_prefix, materialization_cache=future_materialization_cache))
        targeted_added += 1

    # Fill remaining targeted slots with strictly Waymax-generated SDC
    # post-prefix control stress variants.  These do not change the latent
    # background-agent branch, so they deliberately share the same teacher base
    # state; the metadata exposes this to diagnose/papercheck instead of hiding
    # the degeneracy.
    while targeted_added < n_targeted:
        accel = -2.0 if targeted_added % 2 == 0 else 1.2
        str_t = rollout_post_cached(st_prefix, wx_env, accel)
        futures.append(
            _make_future_from_state(
                len(futures),
                "targeted",
                targeted_total / max(n_targeted, 1),
                str_t,
                history,
                prefix,
                cfg,
                wx_env,
                dyn_name,
                {
                    "scenario_augmented": False,
                    "targeted_type": "waymax_sdc_post_prefix_control_stress",
                    "ego_after_prefix_accel": float(accel),
                    "recovery_relevant": True,
                    "waymax_prefix_rollout_reused": True,
                    "teacher_base_reuses_replay_prefix_state": True,
                },
                state_after_prefix=st_prefix,
                materialization_cache=future_materialization_cache,
            )
        )
        targeted_added += 1
    # Normalize priors without importing the surrogate package here.
    s = sum(max(float(f.prior), 0.0) for f in futures)
    if s > 0:
        for f in futures:
            f.prior = float(max(float(f.prior), 0.0) / s)
    return futures


def _action_from_recovery_control(num_objects: int, sdc: int, ctrl: np.ndarray, cfg: dict):
    return _bicycle_action(num_objects, sdc, float(ctrl[0]) if ctrl.size else 0.0, float(ctrl[1]) if ctrl.size > 1 else 0.0, float(cfg.get("wheelbase_m", 2.8)))


def _waymax_margin_from_rollout(metrics_over_time: list[dict[str, float]], cfg: dict) -> tuple[float, dict[str, float], dict[str, bool]]:
    def mx(name: str) -> float:
        vals = [float(m.get(name, 0.0)) for m in metrics_over_time if np.isfinite(float(m.get(name, 0.0)))]
        return max(vals) if vals else 0.0
    overlap = mx("overlap")
    offroad = mx("offroad")
    wrongway = mx("sdc_wrongway")
    offroute = mx("sdc_off_route")
    kin = mx("kinematic_infeasibility")
    logdiv = mx("log_divergence")
    comps = {
        "waymax_overlap": 0.5 - overlap,
        "waymax_offroad": 0.5 - offroad,
        "waymax_wrongway": 0.5 - wrongway,
        "waymax_offroute": 1.0 - offroute / 3.5,
        "waymax_kinematic": 0.5 - kin,
        "waymax_logdiv": 3.0 - logdiv / 2.0,
    }
    active = {k: True for k in comps}
    return float(min(comps.values())), comps, active


def _resolve_waymax_teacher_backend(cfg: dict) -> str:
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    raw = str(wx.get("teacher_backend", "auto")).lower()
    if raw == "auto":
        quality = cfg.get("dataset_quality", {}) if isinstance(cfg.get("dataset_quality", {}), dict) else {}
        artifact = cfg.get("artifact", {}) if isinstance(cfg.get("artifact", {}), dict) else {}
        if bool(quality.get("require_artifact_pairs", False)) and bool(artifact.get("use_margin_override", True)):
            return "structural"
        return "hybrid"
    aliases = {"fast": "structural", "surrogate": "structural", "structural_fast": "structural"}
    return aliases.get(raw, raw)


def _compute_structural_future_option_margins(history: SceneHistory, prefix: CandidatePrefix, futures: list[CounterfactualFuture], options: list[RecoveryOption], cfg: dict, backend_name: str) -> tuple[np.ndarray, list[list[TeacherDiagnostics]]]:
    horizon_steps = max(2, int(round(float(cfg.get("recovery_horizon_s", 4.0)) * float(cfg.get("sample_rate_hz", 10.0)))))
    controllers = [rollout_recovery_controller(prefix, opt, horizon_steps, cfg) for opt in options]
    M = np.zeros((len(futures), len(options)), dtype=np.float32)
    all_diag: list[list[TeacherDiagnostics]] = []
    for j, fut in enumerate(futures):
        row: list[TeacherDiagnostics] = []
        fut.metadata["waymax_teacher_backend"] = backend_name
        fut.metadata["waymax_recovery_rollout_reused"] = False
        for l, opt in enumerate(options):
            rec_states, rec_controls, cdiag = controllers[l]
            val, diag = teacher_margin(history, prefix, fut, opt, rec_states, rec_controls, cfg, cdiag)
            M[j, l] = float(val)
            row.append(
                TeacherDiagnostics(
                    active=diag.active,
                    component_margins=diag.component_margins,
                    controller_diagnostics={**(diag.controller_diagnostics or {}), "waymax_teacher_backend": backend_name, "waymax_recovery_rollout": False},
                )
            )
        all_diag.append(row)
    return M, all_diag


def _should_record_teacher_metric(tt: int, horizon_steps: int, stride: int) -> bool:
    if tt == horizon_steps - 1:
        return True
    if stride <= 0:
        return False
    return (tt % stride) == 0


def compute_waymax_future_option_margins(history: SceneHistory, prefix: CandidatePrefix, futures: list[CounterfactualFuture], options: list[RecoveryOption], cfg: dict) -> tuple[np.ndarray, list[list[TeacherDiagnostics]]]:
    teacher_backend = _resolve_waymax_teacher_backend(cfg)
    if teacher_backend == "structural":
        return _compute_structural_future_option_margins(history, prefix, futures, options, cfg, teacher_backend)
    if teacher_backend not in {"hybrid", "waymax"}:
        raise ValueError(f"Unsupported waymax.teacher_backend={teacher_backend!r}; expected auto, structural, hybrid, or waymax")

    state0 = history.metadata.get("_waymax_state")
    if state0 is None:
        raise ValueError("Waymax teacher requires runtime state from Waymax loader.")
    horizon_steps = max(2, int(round(float(cfg.get("recovery_horizon_s", 4.0)) * float(cfg.get("sample_rate_hz", 10.0)))))
    M = np.zeros((len(futures), len(options)), dtype=np.float32)
    all_diag: list[list[TeacherDiagnostics]] = []
    controllers = [rollout_recovery_controller(prefix, opt, horizon_steps, cfg) for opt in options]
    sdc = _sdc_index(state0)
    margin_cache: dict[tuple[int, int, int], tuple[np.ndarray, list[TeacherDiagnostics]]] = {}
    metric_rollout_cache: dict[tuple[int, int, int, int, int], tuple[float, dict[str, float], dict[str, bool], dict[str, float]]] = {}
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    hybrid_teacher = teacher_backend == "hybrid"
    metric_stride = int(wx.get("teacher_metrics_stride", 1))
    structural_cfg = _cfg_without_artifact_override(cfg)
    artifact_cfg = cfg.get("artifact", {}) if isinstance(cfg.get("artifact", {}), dict) else {}
    artifact_override_enabled = bool(artifact_cfg.get("use_margin_override", True))

    # Screened hybrid mode is an explicit speed/diagnostic knob.  The default
    # top_k=0 and modes=[] preserves the exact old behavior: every option is
    # rolled out in Waymax.  When enabled, the structural teacher scores all
    # options first, and Waymax recovery rollout is executed only for selected
    # options.  This is intended for smoke/debug builds; final experiments should
    # either keep it disabled or report the screened setting.
    rollout_top_k = max(0, int(wx.get("teacher_rollout_top_k_options", 0)))
    raw_modes = wx.get("teacher_rollout_option_modes", [])
    if isinstance(raw_modes, str):
        rollout_modes = {m.strip() for m in raw_modes.split(",") if m.strip()}
    else:
        rollout_modes = {str(m) for m in raw_modes}
    screened_hybrid = bool(hybrid_teacher and (rollout_top_k > 0 or rollout_modes))

    for j, fut in enumerate(futures):
        fut.metadata["waymax_teacher_backend"] = teacher_backend
        base_state = getattr(fut, "_waymax_state_after_prefix", None)
        waymax_env = getattr(fut, "_waymax_env", None)
        if base_state is None or waymax_env is None:
            # Strict mode means no silent surrogate labels.
            raise ValueError("Future is missing Waymax state_after_prefix; cannot compute strict Waymax recovery teacher margin.")
        # With hybrid teacher margins, the label depends on the future branch
        # metadata/trajectory as well as the post-prefix Waymax state.  Caching only
        # by (state, env) incorrectly collapses replay/reactive/targeted roots into
        # identical rows and erases the oracle/deployability gap.
        cache_key = (id(base_state), id(waymax_env), int(j) if hybrid_teacher else -1)
        if bool(wx.get("cache_identical_teacher_rollouts", True)) and cache_key in margin_cache:
            cached_vals, cached_diag = margin_cache[cache_key]
            M[j] = cached_vals
            all_diag.append([
                TeacherDiagnostics(
                    active=dict(d.active),
                    component_margins=dict(d.component_margins),
                    controller_diagnostics={**dict(d.controller_diagnostics), "waymax_recovery_rollout_reused": True},
                )
                for d in cached_diag
            ])
            fut.metadata["waymax_teacher_rollout_reused"] = True
            continue

        structural_vals = np.zeros(len(options), dtype=np.float32)  # no-override values used only for screening/top-k
        structural_label_vals = np.zeros(len(options), dtype=np.float32)  # values used when an option is screened out
        structural_diags: list[TeacherDiagnostics | None] = [None] * len(options)
        structural_label_diags: list[TeacherDiagnostics | None] = [None] * len(options)
        rollout_indices: set[int] | None = None
        if hybrid_teacher:
            for l, opt in enumerate(options):
                rec_states, rec_controls, cdiag = controllers[l]
                sv, sd = teacher_margin(history, prefix, fut, opt, rec_states, rec_controls, structural_cfg, cdiag)
                # The label-side structural teacher differs from ``structural_cfg``
                # only when the explicit artifact margin override can fire.  Avoid
                # recomputing all pairwise structural components for the common
                # non-artifact case (and for the paper-quality builds where
                # artifact.use_margin_override=false).
                if artifact_override_enabled and _artifact_margin_override(opt, fut, cfg) is not None:
                    lv, ld = teacher_margin(history, prefix, fut, opt, rec_states, rec_controls, cfg, cdiag)
                else:
                    lv, ld = sv, sd
                lv, applied = _artifact_override_adjusted_value(float(lv), opt, fut, cfg)
                if applied:
                    fut.metadata["margin_override_applied"] = True
                structural_vals[l] = float(sv)
                structural_label_vals[l] = float(lv)
                structural_diags[l] = sd
                structural_label_diags[l] = ld
            if screened_hybrid:
                rollout_indices = set()
                for l, opt in enumerate(options):
                    if opt.mode in rollout_modes:
                        rollout_indices.add(l)
                if rollout_top_k > 0:
                    valid_idx = [i for i, opt in enumerate(options) if opt.valid]
                    order = sorted(valid_idx, key=lambda i: float(structural_vals[i]), reverse=True)
                    rollout_indices.update(order[:rollout_top_k])
                if not rollout_indices:
                    # Avoid accidentally creating a pure structural row when the
                    # user intended screened hybrid but the selectors are empty.
                    rollout_indices = set(range(len(options)))

        row: list[TeacherDiagnostics] = []
        waymax_rollouts_executed = 0
        waymax_metric_cache_hits = 0
        for l, opt in enumerate(options):
            rec_states, rec_controls, cdiag = controllers[l]
            if screened_hybrid and rollout_indices is not None and l not in rollout_indices:
                sd = structural_label_diags[l] or structural_diags[l]
                assert sd is not None
                val = float(structural_label_vals[l] if hybrid_teacher else structural_vals[l])
                if not opt.valid:
                    val = -1e9
                M[j, l] = float(val)
                row.append(
                    TeacherDiagnostics(
                        active={f"structural_{k}": bool(v) for k, v in sd.active.items()},
                        component_margins={f"structural_{k}": float(v) for k, v in sd.component_margins.items()},
                        controller_diagnostics={
                            **(cdiag or {}),
                            "waymax_recovery_rollout": False,
                            "waymax_recovery_rollout_screened_out": True,
                            "waymax_hybrid_teacher_margin": True,
                            "waymax_teacher_backend": teacher_backend,
                            "waymax_teacher_metrics_stride": int(metric_stride),
                            "waymax_teacher_rollout_top_k_options": int(rollout_top_k),
                            "waymax_teacher_rollout_option_modes": sorted(rollout_modes),
                        },
                    )
                )
                continue

            st = base_state
            metrics_over_time: list[dict[str, float]] = []
            if (
                hybrid_teacher
                and bool(wx.get("skip_waymax_rollout_for_augmented_override", False))
                and bool(fut.metadata.get("scenario_augmented", False))
                and bool(cfg.get("artifact", {}).get("use_margin_override", True))
            ):
                sd = structural_label_diags[l] or structural_diags[l]
                assert sd is not None
                val = float(structural_label_vals[l])
                if not opt.valid:
                    val = -1e9
                M[j, l] = float(val)
                row.append(
                    TeacherDiagnostics(
                        active={f"structural_{k}": bool(v) for k, v in sd.active.items()},
                        component_margins={f"structural_{k}": float(v) for k, v in sd.component_margins.items()},
                        controller_diagnostics={
                            **(cdiag or {}),
                            "waymax_recovery_rollout": False,
                            "waymax_recovery_rollout_skipped_augmented_override": True,
                            "waymax_hybrid_teacher_margin": True,
                            "waymax_teacher_backend": teacher_backend,
                            "waymax_teacher_metrics_stride": int(metric_stride),
                            "waymax_teacher_rollout_top_k_options": int(rollout_top_k),
                            "waymax_teacher_rollout_option_modes": sorted(rollout_modes),
                        },
                    )
                )
                continue
            # When teacher_metrics_stride <= 0, only the final recovery metric is
            # used.  Roll the whole control sequence with one JAX scan dispatch.
            # For stride > 0 we keep the original Python loop because intermediate
            # metrics are semantically required.
            metric_cache_hit = False
            metric_cache_key = (id(base_state), id(waymax_env), int(l), int(metric_stride), int(horizon_steps))
            cached_metric = metric_rollout_cache.get(metric_cache_key) if bool(wx.get("cache_teacher_metric_rollouts", True)) else None
            if cached_metric is not None:
                val, comps, active, metrics_last = cached_metric
                metrics_over_time.append(dict(metrics_last))
                metric_cache_hit = True
                waymax_metric_cache_hits += 1
            else:
                if metric_stride <= 0 and rec_controls.size:
                    controls = np.zeros((horizon_steps, 2), dtype=np.float32)
                    controls[:, 0] = rec_controls[np.minimum(np.arange(horizon_steps), rec_controls.shape[0] - 1), 0]
                    controls[:, 1] = rec_controls[np.minimum(np.arange(horizon_steps), rec_controls.shape[0] - 1), 1]
                    st = _rollout_bicycle_controls_scan(st, waymax_env, controls, cfg)
                    metrics_over_time.append(_metric_summary(waymax_env, st, sdc))
                else:
                    for tt in range(horizon_steps):
                        ctrl = rec_controls[min(tt, rec_controls.shape[0] - 1)] if rec_controls.size else np.zeros(4, dtype=np.float32)
                        action = _action_from_recovery_control(int(st.num_objects), sdc, ctrl, cfg)
                        st = waymax_env.step(st, action)
                        if _should_record_teacher_metric(tt, horizon_steps, metric_stride):
                            metrics_over_time.append(_metric_summary(waymax_env, st, sdc))
                waymax_rollouts_executed += 1
                val, comps, active = _waymax_margin_from_rollout(metrics_over_time, cfg)
                if bool(wx.get("cache_teacher_metric_rollouts", True)):
                    metric_rollout_cache[metric_cache_key] = (float(val), dict(comps), dict(active), metrics_over_time[-1] if metrics_over_time else {})
            if hybrid_teacher:
                if structural_diags[l] is not None:
                    structural_val = float(structural_vals[l])
                    structural_diag = structural_diags[l]
                    assert structural_diag is not None
                else:
                    structural_val, structural_diag = teacher_margin(history, prefix, fut, opt, rec_states, rec_controls, structural_cfg, cdiag)
                # Waymax metrics remain a hard safety cap, while the structural
                # recovery teacher supplies option/root sensitivity when the metric
                # suite is all-zero on benign WOMD snippets.
                val = min(float(val), float(structural_val))
                comps = {**{f"waymax_{k}": float(v) for k, v in comps.items()}, **{f"structural_{k}": float(v) for k, v in structural_diag.component_margins.items()}}
                active = {**{f"waymax_{k}": bool(v) for k, v in active.items()}, **{f"structural_{k}": bool(v) for k, v in structural_diag.active.items()}}
            if not opt.valid:
                val = -1e9
            # Preserve the deliberately augmented hidden pair's incompatibility
            # consistently for both rolled and screened options.
            val, applied = _artifact_override_adjusted_value(float(val), opt, fut, cfg)
            if applied:
                fut.metadata["margin_override_applied"] = True
            M[j, l] = float(val)
            row.append(TeacherDiagnostics(active=active, component_margins=comps, controller_diagnostics={**(cdiag or {}), "waymax_recovery_rollout": True, "waymax_recovery_metric_cache_hit": bool(metric_cache_hit), "waymax_hybrid_teacher_margin": bool(hybrid_teacher), "waymax_teacher_backend": teacher_backend, "waymax_teacher_metrics_stride": int(metric_stride), "waymax_teacher_rollout_top_k_options": int(rollout_top_k), "waymax_teacher_rollout_option_modes": sorted(rollout_modes), "waymax_metrics_last": metrics_over_time[-1] if metrics_over_time else {}}))
        margin_cache[cache_key] = (M[j].copy(), row)
        fut.metadata["waymax_teacher_rollout_reused"] = False
        fut.metadata["waymax_teacher_rollouts_executed"] = int(waymax_rollouts_executed)
        fut.metadata["waymax_teacher_metric_cache_hits"] = int(waymax_metric_cache_hits)
        fut.metadata["waymax_teacher_rollouts_possible"] = int(len(options))
        fut.metadata["waymax_teacher_screened_hybrid"] = bool(screened_hybrid)
        all_diag.append(row)
    return M, all_diag
