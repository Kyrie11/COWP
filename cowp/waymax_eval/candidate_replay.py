from __future__ import annotations

import gc
import json
import os
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cowp.planning.cowp_planner import COWPPlanner
from cowp.utils.progress import tqdm_iter
from cowp.waymax_eval.metrics_standard import WaymaxStandardMetricAccumulator, build_waymax_metric_objects
from cowp.waymax_eval.policy_wrapper import _to_numpy, _wrap_angle, extract_current_agent_state
from cowp.waymax_eval.rollout import _env_step, _make_waymax_environment, _state_done
from cowp.waymax_eval.outcome_attach import attach_rows_to_cache_file


def restore_key(k: str) -> str:
    return k.replace("__", "/")


def store_key(k: str) -> str:
    return k.replace("/", "__")


def load_npz_canonical(path: str | Path, *, keys: set[str] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        if keys is None:
            return {restore_key(k): data[k] for k in data.files}
        wanted_stored = {store_key(k) for k in keys} | set(keys)
        return {restore_key(k): data[k] for k in data.files if k in wanted_stored or restore_key(k) in keys}


_REPLAY_CANDIDATE_KEYS = {
    "cowp/candidates/trajectory",
    "cowp/candidates/valid",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/false_safe",
    "cowp/candidates/noncoercive_feasible",
    "cowp/candidates/ego_utility_prior",
}


# Candidate-selection tensors that are sufficient to reproduce the current
# ``select_candidate_indices`` behavior.  In resume mode this lets us decide
# whether a scene is already fully covered by the existing outcome JSONL before
# paying the expensive cost of loading all cached WOMD tensors and constructing
# a Waymax ``SimulatorState``.  This does not change labels: if any selected
# candidate is missing, the code falls back to the exact same full load + replay
# path as before.
_REPLAY_SELECTION_KEYS = {
    "cowp/candidates/valid",
    "cowp/candidates/certificate_valid",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/false_safe",
    "cowp/candidates/noncoercive_feasible",
    "cowp/candidates/ego_utility_prior",
}


def load_cache_selection_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load only candidate-selection tensors from a tensor-cache NPZ."""
    return load_npz_canonical(path, keys=set(_REPLAY_SELECTION_KEYS))


def load_cache_replay_arrays(path: str | Path, *, candidate_keys: set[str] | None = None) -> dict[str, np.ndarray]:
    """Load only tensors needed by candidate replay from a tensor-cache NPZ.

    A full tensor-cache item can contain large natural-response, map, and
    supervision tensors that are irrelevant to Waymax candidate replay.  Reading
    the whole archive for every scene wastes disk bandwidth and host memory.
    This helper keeps the exact replay semantics by loading all cached WOMD
    features needed to rebuild ``SimulatorState`` plus the candidate-selection
    tensors, while avoiding unrelated label arrays.
    """
    wanted = set(_REPLAY_CANDIDATE_KEYS if candidate_keys is None else candidate_keys)
    with np.load(path, allow_pickle=True) as data:
        out: dict[str, np.ndarray] = {}
        for stored_key in data.files:
            key = restore_key(stored_key)
            if key.startswith("womd/") or key in wanted:
                out[key] = data[stored_key]
        return out




_CANDIDATE_TIMING_KEYS = (
    "timing/env_reset_s",
    "timing/policy_build_s",
    "timing/action_s",
    "timing/env_step_s",
    "timing/metric_update_s",
    "timing/metric_OverlapMetric_s",
    "timing/metric_OffroadMetric_s",
    "timing/done_check_s",
    "timing/metric_finalize_s",
)


def _block_until_ready_tree(value: Any) -> None:
    """Synchronize a JAX pytree without copying it back to host.

    JAX dispatch is asynchronous.  Plain perf_counter measurements around
    env.step/metric.compute therefore measure mostly Python dispatch time and
    can incorrectly charge the real accelerator work to the next NumPy/device
    conversion.  The profile probe uses this helper only for a handful of
    diagnostic candidates so stage timings correspond to completed GPU work.
    It is never enabled for the normal full replay path.
    """
    try:
        import jax  # type: ignore

        leaves = jax.tree_util.tree_leaves(value)
    except Exception:
        leaves = [value]
    for leaf in leaves:
        block = getattr(leaf, "block_until_ready", None)
        if callable(block):
            try:
                block()
            except Exception:
                pass


class _JitCallableWithFallback:
    """Call a jitted function and permanently fall back to eager on trace/runtime setup failure."""

    def __init__(self, jitted: Any, eager: Any):
        self._jitted = jitted
        self._eager = eager
        self.using_jit = True
        self.last_error: str | None = None

    def __call__(self, *args, **kwargs):
        if not self.using_jit:
            return self._eager(*args, **kwargs)
        try:
            return self._jitted(*args, **kwargs)
        except Exception as exc:
            self.using_jit = False
            self.last_error = str(exc)
            return self._eager(*args, **kwargs)


def _make_jitted_env_step(env: Any):
    """Best-effort JIT wrapper for one Waymax env.step call with eager fallback."""
    try:
        import jax  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"jax unavailable for --jit-env-step: {exc}") from exc

    def _step(state, action):
        return _env_step(env, state, action)

    return _JitCallableWithFallback(jax.jit(_step), _step)


def _make_jitted_env_reset(env: Any):
    """Best-effort JIT wrapper for Waymax env.reset with eager fallback."""
    try:
        import jax  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"jax unavailable for --jit-env-reset: {exc}") from exc
    return _JitCallableWithFallback(jax.jit(env.reset), env.reset)

def scenario_id_from_arrays(arrays: dict[str, np.ndarray], path: str | Path | None = None) -> str:
    for key in ("scenario/id", "scenario__id", "womd/scenario/id", "womd__scenario__id"):
        if key in arrays:
            try:
                item = np.asarray(arrays[key]).reshape(-1)[0]
                if isinstance(item, bytes):
                    return item.decode("utf-8")
                return str(item)
            except Exception:
                pass
    if path is not None:
        return Path(path).stem
    raise KeyError("cannot infer scenario id from tensor cache item")


def save_npz_canonical(path: str | Path, arrays: dict[str, np.ndarray], *, compress: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {store_key(k): v for k, v in arrays.items()}
    if compress:
        np.savez_compressed(path, **stored)
    else:
        np.savez(path, **stored)


def _safe_bool(arrays: dict[str, np.ndarray], key: str, like: np.ndarray | None = None) -> np.ndarray:
    if key in arrays:
        return np.asarray(arrays[key], dtype=bool)
    if like is None:
        return np.zeros(0, dtype=bool)
    return np.zeros_like(like, dtype=bool)


def _stable_seed(s: str) -> int:
    # Python's hash() is salted per process, so using it makes candidate sampling
    # non-reproducible across resumed/sharded runs.  This small FNV-1a hash is
    # deterministic and sufficient for per-scene shuffling.
    h = 2166136261
    for b in str(s).encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def select_candidate_indices(
    arrays: dict[str, np.ndarray],
    cfg: dict,
    *,
    selection: str = "balanced",
    max_candidates: int | None = 8,
    seed: int = 0,
) -> list[int]:
    """Return deterministic candidate indices to replay for one scene."""
    valid = np.asarray(arrays.get("cowp/candidates/valid", []), dtype=bool)
    if valid.ndim != 1 or valid.size == 0:
        return []
    K = int(valid.shape[0])
    cert_valid = np.asarray(arrays.get("cowp/candidates/certificate_valid", valid), dtype=bool).reshape(-1)[:K] & valid
    cap = K if max_candidates is None or int(max_candidates) <= 0 else min(K, int(max_candidates))
    if selection == "all":
        return np.where(valid)[0].astype(int).tolist()[:cap]
    if selection == "noncoercive":
        m = cert_valid & _safe_bool(arrays, "cowp/candidates/noncoercive_feasible", valid)
        return np.where(m)[0].astype(int).tolist()[:cap]
    if selection == "false_safe":
        m = cert_valid & _safe_bool(arrays, "cowp/candidates/false_safe", valid)
        return np.where(m)[0].astype(int).tolist()[:cap]
    if selection == "conventional":
        m = valid & np.asarray(arrays.get("cowp/candidates/conventional_safe", valid), dtype=bool)
        return np.where(m)[0].astype(int).tolist()[:cap]
    if selection == "selected":
        try:
            dec = COWPPlanner(cfg).select_from_labels(arrays)
            return [int(dec.candidate_index)] if int(dec.candidate_index) >= 0 and valid[int(dec.candidate_index)] else []
        except Exception:
            idx = np.where(valid)[0]
            return [int(idx[0])] if len(idx) else []
    if selection != "balanced":
        raise ValueError(f"unknown candidate selection {selection!r}")

    buckets: list[np.ndarray] = []
    try:
        dec = COWPPlanner(cfg).select_from_labels(arrays)
        if int(dec.candidate_index) >= 0:
            buckets.append(np.asarray([int(dec.candidate_index)], dtype=np.int64))
    except Exception:
        pass
    ncf = cert_valid & _safe_bool(arrays, "cowp/candidates/noncoercive_feasible", valid)
    fs = cert_valid & _safe_bool(arrays, "cowp/candidates/false_safe", valid)
    conv = valid & np.asarray(arrays.get("cowp/candidates/conventional_safe", valid), dtype=bool)
    # Certificate-unknown candidates remain eligible for physical Waymax replay,
    # but they are kept in an explicit unknown bucket rather than mislabeled as
    # ordinary conventional-safe negatives for NCF/false-safe balancing.
    buckets.extend([
        np.where(ncf)[0],
        np.where(fs)[0],
        np.where(cert_valid & conv & ~ncf & ~fs)[0],
        np.where(valid & ~cert_valid)[0],
        np.where(valid & ~conv)[0],
        np.where(valid)[0],
    ])

    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    seen: set[int] = set()
    bucket_lists = []
    for b in buckets:
        b = np.asarray([int(x) for x in b if 0 <= int(x) < K and valid[int(x)]], dtype=np.int64)
        if b.size:
            b = b.copy()
            rng.shuffle(b)
            bucket_lists.append(b.tolist())
    cursor = [0 for _ in bucket_lists]
    while len(selected) < cap and bucket_lists:
        made_progress = False
        for bi, b in enumerate(bucket_lists):
            while cursor[bi] < len(b) and int(b[cursor[bi]]) in seen:
                cursor[bi] += 1
            if cursor[bi] >= len(b):
                continue
            k = int(b[cursor[bi]])
            cursor[bi] += 1
            selected.append(k)
            seen.add(k)
            made_progress = True
            if len(selected) >= cap:
                break
        if not made_progress:
            break
    return selected


def _state_num_objects(init_state: Any) -> int:
    n = getattr(init_state, "num_objects", None)
    if n is None and hasattr(init_state, "log_trajectory"):
        n = getattr(init_state.log_trajectory, "num_objects", None)
    if n is None:
        try:
            cur, _ = extract_current_agent_state(init_state)
            n = cur.shape[0]
        except Exception:
            n = 128
    try:
        return int(np.asarray(n).reshape(-1)[0])
    except Exception:
        return int(n)


def _initial_sdc_pose(init_state: Any) -> tuple[int, np.ndarray]:
    cur, sdc = extract_current_agent_state(init_state)
    sdc = int(sdc)
    pose = np.zeros(3, dtype=np.float32)
    if 0 <= sdc < cur.shape[0]:
        pose[:] = np.asarray([cur[sdc, 0], cur[sdc, 1], cur[sdc, 6]], dtype=np.float32)
    return sdc, pose


def metric_names_for_set(metric_set: str | None) -> set[str] | None:
    metric_set = str(metric_set or "safety").lower()
    if metric_set in {"none", "off"}:
        return set()
    if metric_set in {"safety", "fast"}:
        return {"OverlapMetric", "OffroadMetric"}
    if metric_set in {"safety_logdiv", "safety+logdiv", "logdiv"}:
        return {"OverlapMetric", "OffroadMetric", "LogDivergenceMetric"}
    if metric_set in {"standard", "all"}:
        return None
    raise ValueError(f"unknown metric_set={metric_set!r}; use safety, safety_logdiv, standard, or none")


@dataclass
class FixedCandidateReplayPolicy:
    trajectory: np.ndarray
    cfg: dict
    action_mode: str = "absolute_xy_yaw"
    sdc_index: int | None = None
    num_objects: int | None = None
    initial_pose: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.trajectory = np.asarray(self.trajectory, dtype=np.float32)
        if self.trajectory.ndim != 2 or self.trajectory.shape[1] < 3:
            raise ValueError(f"candidate trajectory must have shape [T,>=3], got {self.trajectory.shape}")
        if self.initial_pose is not None:
            self.initial_pose = np.asarray(self.initial_pose, dtype=np.float32).reshape(3)
        self._fast_action_data_seq = None
        self._fast_action_valid = None
        # Precompute the fixed candidate actions once per candidate.  The previous
        # implementation rebuilt an [N,D] NumPy array and copied it to JAX at every
        # rollout step, which hides GPU benefit behind Python/host-to-device work.
        # This keeps the exact same action values and only changes where they are
        # allocated.
        if self.sdc_index is not None and self.num_objects is not None:
            try:
                self._precompute_fast_actions()
            except Exception:
                self._fast_action_data_seq = None
                self._fast_action_valid = None

    def _precompute_fast_actions(self) -> None:
        import jax.numpy as jnp  # type: ignore

        N = int(self.num_objects or 128)
        sdc = int(0 if self.sdc_index is None else self.sdc_index)
        if self.action_mode == "absolute_xy_yaw":
            data_dim = max(int(self.cfg.get("waymax", {}).get("action_dim", 5)), 5)
        else:
            data_dim = int(self.cfg.get("waymax", {}).get("action_dim", 3))
        T = int(max(self.trajectory.shape[0], 1))
        data = np.zeros((T, N, data_dim), dtype=np.float32)
        valid = np.zeros((N, 1), dtype=bool)
        if 0 <= sdc < N:
            valid[sdc, 0] = True
            if self.action_mode == "absolute_xy_yaw":
                xy_yaw = self.trajectory[:, :3]
                vel = np.zeros((T, 2), dtype=np.float32)
                if self.trajectory.shape[1] > 3:
                    vel[:, 0] = self.trajectory[:, 3]
                if self.trajectory.shape[1] > 4:
                    vel[:, 1] = self.trajectory[:, 4]
                data[:, sdc, :5] = np.concatenate([xy_yaw, vel], axis=-1).astype(np.float32)
            else:
                refs = np.zeros((T, 3), dtype=np.float32)
                refs[0] = self.initial_pose if self.initial_pose is not None else np.zeros(3, dtype=np.float32)
                if T > 1:
                    refs[1:] = self.trajectory[:-1, :3]
                delta = self.trajectory[:, :3] - refs
                delta[:, 2] = np.asarray([_wrap_angle(float(x)) for x in delta[:, 2]], dtype=np.float32)
                data[:, sdc, : min(data_dim, 3)] = delta[:, : min(data_dim, 3)]
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._fast_action_data_seq = jnp.asarray(data)
        self._fast_action_valid = jnp.asarray(valid)

    def _trajectory_to_action_fast(self, step: int) -> Any:
        try:
            from waymax import datatypes  # type: ignore
            import jax.numpy as jnp  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("waymax.datatypes and jax are required for candidate replay") from exc
        if self._fast_action_data_seq is not None and self._fast_action_valid is not None:
            t = min(max(int(step), 0), int(self._fast_action_data_seq.shape[0]) - 1)
            return datatypes.Action(data=self._fast_action_data_seq[t], valid=self._fast_action_valid)
        N = int(self.num_objects or 128)
        sdc = int(0 if self.sdc_index is None else self.sdc_index)
        if self.action_mode == "absolute_xy_yaw":
            data_dim = max(int(self.cfg.get("waymax", {}).get("action_dim", 5)), 5)
        else:
            data_dim = int(self.cfg.get("waymax", {}).get("action_dim", 3))
        data = np.zeros((N, data_dim), dtype=np.float32)
        valid = np.zeros((N, 1), dtype=bool)
        if 0 <= sdc < N:
            valid[sdc, 0] = True
            t = min(max(int(step), 0), self.trajectory.shape[0] - 1)
            target = self.trajectory[t]
            if self.action_mode == "absolute_xy_yaw":
                vx = float(target[3]) if target.shape[0] > 3 else 0.0
                vy = float(target[4]) if target.shape[0] > 4 else 0.0
                data[sdc, :5] = np.asarray([target[0], target[1], target[2], vx, vy], dtype=np.float32)
            else:
                if t == 0:
                    ref = self.initial_pose if self.initial_pose is not None else np.zeros(3, dtype=np.float32)
                else:
                    ref = self.trajectory[t - 1, :3]
                dx = float(target[0] - ref[0])
                dy = float(target[1] - ref[1])
                dyaw = float(_wrap_angle(float(target[2] - ref[2])))
                data[sdc, : min(data_dim, 3)] = np.asarray([dx, dy, dyaw], dtype=np.float32)[:data_dim]
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return datatypes.Action(data=jnp.asarray(data), valid=jnp.asarray(valid))

    def _trajectory_to_action_slow(self, state: Any, step: int) -> Any:
        try:
            from waymax import datatypes  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("waymax.datatypes is required for candidate replay") from exc
        agent_state, sdc_index = extract_current_agent_state(state)
        N = int(agent_state.shape[0])
        data_dim = int(self.cfg.get("waymax", {}).get("action_dim", 3))
        if self.action_mode == "absolute_xy_yaw":
            data_dim = max(data_dim, 5)
        data = np.zeros((N, data_dim), dtype=np.float32)
        valid = np.zeros((N, 1), dtype=bool)
        if not (0 <= sdc_index < N):
            return datatypes.Action(data=data, valid=valid)
        valid[sdc_index, 0] = True
        target = self.trajectory[min(max(int(step), 0), self.trajectory.shape[0] - 1)]
        dx = float(target[0] - agent_state[sdc_index, 0])
        dy = float(target[1] - agent_state[sdc_index, 1])
        dyaw = float(_wrap_angle(float(target[2] - agent_state[sdc_index, 6])))
        if self.action_mode == "absolute_xy_yaw":
            vx = float(target[3]) if target.shape[0] > 3 else dx / float(self.cfg.get("time", {}).get("dt", 0.1))
            vy = float(target[4]) if target.shape[0] > 4 else dy / float(self.cfg.get("time", {}).get("dt", 0.1))
            data[sdc_index, :5] = np.asarray([target[0], target[1], target[2], vx, vy], dtype=np.float32)
        else:
            data[sdc_index, : min(data_dim, 3)] = np.asarray([dx, dy, dyaw], dtype=np.float32)[:data_dim]
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        try:
            import jax.numpy as jnp  # type: ignore
            return datatypes.Action(data=jnp.asarray(data), valid=jnp.asarray(valid))
        except Exception:
            return datatypes.Action(data=data, valid=valid)

    def __call__(self, state: Any, *, step: int | None = None, scenario_index: int | None = None) -> Any:
        step_i = 0 if step is None else int(step)
        if self.sdc_index is not None and self.num_objects is not None:
            return self._trajectory_to_action_fast(step_i)
        return self._trajectory_to_action_slow(state, step_i)


def _metric_set_is_fast_safety(metric_set: str | None, metric_objects: list[tuple[str, Any]] | None) -> bool:
    if str(metric_set or "").lower() not in {"safety", "fast"}:
        return False
    names = {str(name) for name, _ in (metric_objects or [])}
    return names.issubset({"OverlapMetric", "OffroadMetric"})


def _array_get_any(arrays: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in arrays:
            return np.asarray(arrays[name])
    return None


def _candidate_metric_guard_steps(
    arrays: dict[str, np.ndarray],
    trajectory: np.ndarray,
    *,
    horizon_steps: int,
    radius_m: float = 8.0,
    window_steps: int = 1,
) -> set[int]:
    """Return 1-indexed metric-update steps for risk-guarded sampled replay.

    The expensive part of replay is calling Waymax safety metrics.  Pure sampled
    replay can miss short collisions between sampled instants.  This lightweight
    guard uses cached WOMD logged future centers to add extra metric evaluations
    around timesteps where the replayed SDC candidate is close to any non-SDC
    agent.  It preserves full Waymax metrics at those guarded instants; the
    geometric screen is only used to decide *when* to evaluate the metric.
    """
    try:
        traj = np.asarray(trajectory, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] < 2:
            return set()
        x = _array_get_any(arrays, ("womd/state/future/x", "state/future/x"))
        y = _array_get_any(arrays, ("womd/state/future/y", "state/future/y"))
        valid = _array_get_any(arrays, ("womd/state/future/valid", "state/future/valid"))
        if x is None or y is None or valid is None:
            return set()
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        if x.ndim != 2 or y.shape != x.shape or valid.shape != x.shape:
            return set()
        T = int(min(int(horizon_steps), traj.shape[0], x.shape[1]))
        if T <= 0:
            return set()
        vmask = valid[:, :T].copy()
        is_sdc = _array_get_any(arrays, ("womd/state/is_sdc", "state/is_sdc"))
        if is_sdc is not None:
            is_sdc = np.asarray(is_sdc).reshape(-1).astype(bool)
            n = min(vmask.shape[0], is_sdc.shape[0])
            if n > 0:
                sdc_rows = np.where(is_sdc[:n])[0]
                if sdc_rows.size:
                    vmask[sdc_rows, :] = False
        # Include current validity if present; this removes padded agents that
        # happen to have arbitrary future arrays.
        cur_valid = _array_get_any(arrays, ("womd/state/current/valid", "state/current/valid"))
        if cur_valid is not None:
            cv = np.asarray(cur_valid).reshape(-1).astype(bool)
            n = min(vmask.shape[0], cv.shape[0])
            if n > 0:
                vmask[:n, :] &= cv[:n, None]
        cx = traj[:T, 0].reshape(1, T)
        cy = traj[:T, 1].reshape(1, T)
        r2 = float(max(radius_m, 0.0)) ** 2
        near = vmask & np.isfinite(x[:, :T]) & np.isfinite(y[:, :T]) & (((x[:, :T] - cx) ** 2 + (y[:, :T] - cy) ** 2) <= r2)
        risky_t = np.where(np.any(near, axis=0))[0]
        out: set[int] = set()
        w = max(0, int(window_steps))
        for t in risky_t.astype(int).tolist():
            for tt in range(max(0, t - w), min(T - 1, t + w) + 1):
                out.add(int(tt + 1))  # replay loop uses step+1
        return out
    except Exception:
        return set()


class _FastSafetyMetricAccumulator:
    """Safety-only Waymax metric accumulator with one host sync at finalize.

    The standard accumulator converts every metric result to NumPy at every
    simulator step.  For 80-step × 12-candidate replay this creates hundreds of
    device/host synchronizations per scene and can dominate runtime on GPU.
    Safety labels only need any-over-rollout OverlapMetric and OffroadMetric, so
    we accumulate their SDC values as JAX arrays and transfer them to host once.
    """

    def __init__(self, *, metric_objects: list[tuple[str, Any]], init_errors: dict[str, str] | None = None, sdc_index: int | None = None) -> None:
        self.metric_objects = [(str(n), m) for n, m in (metric_objects or []) if str(n) in {"OverlapMetric", "OffroadMetric"}]
        self.init_errors = dict(init_errors or {})
        self.sdc_index = None if sdc_index is None else int(sdc_index)
        self.max_values: dict[str, Any] = {}
        self.errors: dict[str, str] = {}
        self.step_count = 0

    def _sdc_scalar(self, result: Any):
        import jax.numpy as jnp  # type: ignore

        if hasattr(result, "value"):
            value = getattr(result, "value")
            valid = getattr(result, "valid", None)
        elif isinstance(result, dict) and "value" in result:
            value = result["value"]
            valid = result.get("valid")
        else:
            value = result
            valid = None
        v = jnp.asarray(value, dtype=jnp.float32)
        m = None if valid is None else jnp.asarray(valid, dtype=bool)
        while getattr(v, "ndim", 0) > 1:
            v = v[0]
            if m is not None and getattr(m, "ndim", 0) > 1:
                m = m[0]
        if getattr(v, "ndim", 0) == 1 and self.sdc_index is not None and 0 <= int(self.sdc_index) < int(v.shape[0]):
            s = v[int(self.sdc_index)]
            if m is not None and getattr(m, "ndim", 0) == 1 and int(self.sdc_index) < int(m.shape[0]):
                s = jnp.where(m[int(self.sdc_index)], s, 0.0)
            return jnp.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
        if m is not None and getattr(m, "shape", None) == getattr(v, "shape", None):
            v = jnp.where(m, v, 0.0)
        return jnp.nan_to_num(jnp.max(v), nan=0.0, posinf=1.0, neginf=0.0)

    def update(
        self,
        state: Any,
        *,
        timing_out: dict[str, float] | None = None,
        synchronize_timing: bool = False,
    ) -> None:
        import jax.numpy as jnp  # type: ignore

        self.step_count += 1
        for name, metric in self.metric_objects:
            t0 = time.perf_counter() if timing_out is not None else 0.0
            try:
                scalar = self._sdc_scalar(metric.compute(state))
                prev = self.max_values.get(name)
                value = scalar if prev is None else jnp.maximum(prev, scalar)
                self.max_values[name] = value
                if synchronize_timing:
                    _block_until_ready_tree(value)
            except Exception as exc:
                self.errors.setdefault(name, str(exc))
                continue
            if timing_out is not None:
                key = f"timing/metric_{name}_s"
                timing_out[key] = float(timing_out.get(key, 0.0) + (time.perf_counter() - t0))

    def finalize(self, *, include_errors: bool = True) -> dict[str, float | int | dict[str, str]]:
        try:
            import jax  # type: ignore
        except Exception:  # pragma: no cover
            jax = None
        host: dict[str, float] = {}
        for name, value in self.max_values.items():
            try:
                x = jax.device_get(value) if jax is not None else value
                host[name] = float(np.asarray(x).reshape(-1)[0])
            except Exception as exc:
                self.errors.setdefault(name, str(exc))
        out: dict[str, float | int | dict[str, str]] = {"MetricSteps": int(self.step_count)}
        collision = float(host.get("OverlapMetric", 0.0)) > 0.0
        offroad = float(host.get("OffroadMetric", 0.0)) > 0.0
        if "OverlapMetric" in host:
            out["CollisionRate"] = float(collision)
            out["WaymaxAny/OverlapMetric"] = float(collision)
        if "OffroadMetric" in host:
            out["OffroadRate"] = float(offroad)
            out["WaymaxAny/OffroadMetric"] = float(offroad)
        if "OverlapMetric" in host or "OffroadMetric" in host:
            out["CR"] = float(collision or offroad)
        all_errors = {**self.init_errors, **self.errors}
        if include_errors and all_errors:
            out["metric_errors"] = all_errors
        return out


def _candidate_result_from_metrics(metrics: dict[str, Any], steps: int) -> dict[str, Any]:
    collision = bool(float(metrics.get("CollisionRate", 0.0)) > 0.0 or float(metrics.get("WaymaxAny/OverlapMetric", 0.0)) > 0.0)
    offroad = bool(float(metrics.get("OffroadRate", 0.0)) > 0.0 or float(metrics.get("WaymaxAny/OffroadMetric", 0.0)) > 0.0)
    logdiv = metrics.get("LogDivergence", np.nan)
    try:
        logdiv_f = float(logdiv)
    except Exception:
        logdiv_f = float("nan")
    out: dict[str, Any] = {
        "rollout_valid": True,
        "collision": collision,
        "offroad": offroad,
        "log_divergence": logdiv_f,
        "steps": int(steps),
    }
    for key in ("CR", "EP", "KinematicsInfeasibilityRate", "WrongWayRate", "OffRouteRate"):
        if key in metrics and not isinstance(metrics[key], dict):
            out[key] = float(metrics[key])
    if "metric_errors" in metrics:
        out["metric_errors"] = metrics["metric_errors"]
    return out


def replay_candidate_on_env(
    env: Any,
    init_state: Any,
    trajectory: np.ndarray,
    cfg: dict,
    *,
    horizon_steps: int = 80,
    action_mode: str = "absolute_xy_yaw",
    metric_objects: list[tuple[str, Any]] | None = None,
    metric_init_errors: dict[str, str] | None = None,
    num_objects: int | None = None,
    sdc_index: int | None = None,
    initial_pose: np.ndarray | None = None,
    metric_set: str = "safety",
    collect_timing: bool = False,
    synchronize_timing: bool = False,
    step_fn: Any | None = None,
    reset_fn: Any | None = None,
    done_check_interval: int = 1,
    metric_eval_mode: str = "step",
    metric_eval_interval: int = 1,
    metric_guard_steps: set[int] | None = None,
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    if collect_timing:
        timings["timing/mode"] = "sync" if synchronize_timing else "dispatch"
    t = time.perf_counter()
    state = reset_fn(init_state) if reset_fn is not None else env.reset(init_state)
    if synchronize_timing:
        _block_until_ready_tree(state)
    if collect_timing:
        timings["timing/env_reset_s"] = time.perf_counter() - t

    t = time.perf_counter()
    policy = FixedCandidateReplayPolicy(
        trajectory=np.asarray(trajectory, dtype=np.float32),
        cfg=cfg,
        action_mode=action_mode,
        sdc_index=sdc_index,
        num_objects=num_objects,
        initial_pose=initial_pose,
    )
    if synchronize_timing:
        _block_until_ready_tree((policy._fast_action_data_seq, policy._fast_action_valid))
    if collect_timing:
        timings["timing/policy_build_s"] = time.perf_counter() - t

    metric_list = metric_objects or []
    if _metric_set_is_fast_safety(metric_set, metric_list):
        metric_acc: Any = _FastSafetyMetricAccumulator(metric_objects=metric_list, init_errors=metric_init_errors or {}, sdc_index=sdc_index)
    else:
        metric_acc = WaymaxStandardMetricAccumulator(metric_objects=metric_list, init_errors=metric_init_errors or {})

    metric_eval_mode = str(metric_eval_mode or "step").lower()
    if metric_eval_mode not in {"step", "sampled", "adaptive", "final"}:
        raise ValueError(f"metric_eval_mode must be one of step/sampled/adaptive/final, got {metric_eval_mode!r}")
    metric_eval_interval = max(1, int(metric_eval_interval or 1))
    # Waymax safety metrics are episode/event metrics computed from SimulatorState.
    # Computing OverlapMetric/OffroadMetric after every step is expensive.
    # - step: update metrics at every simulator step (exact reference path).
    # - sampled: update every metric_eval_interval steps and at the final step
    #            (faster, but it can miss short-lived collisions/offroad events
    #             between sampled instants).
    # - adaptive: sampled replay plus extra guarded checks around timesteps where
    #             the candidate is close to logged non-SDC agents.
    # - final: update once after the rollout (fastest, lowest recall for transient
    #          safety events in current Waymax metric semantics).

    steps = 0
    action_s = 0.0
    env_step_s = 0.0
    metric_s = 0.0
    done_s = 0.0
    for step in range(int(horizon_steps)):
        if collect_timing:
            t = time.perf_counter()
        action = policy(state, step=step)
        if synchronize_timing:
            _block_until_ready_tree(action)
        if collect_timing:
            action_s += time.perf_counter() - t
            t = time.perf_counter()
        state = step_fn(state, action) if step_fn is not None else _env_step(env, state, action)
        if synchronize_timing:
            _block_until_ready_tree(state)
        if collect_timing:
            env_step_s += time.perf_counter() - t
            t = time.perf_counter()
        steps += 1
        should_update_metric = False
        if metric_eval_mode == "step":
            should_update_metric = True
        elif metric_eval_mode in {"sampled", "adaptive"}:
            step_no = step + 1
            should_update_metric = ((step_no % metric_eval_interval) == 0) or (step_no >= int(horizon_steps))
            if metric_eval_mode == "adaptive" and metric_guard_steps is not None and step_no in metric_guard_steps:
                should_update_metric = True
        if should_update_metric:
            if isinstance(metric_acc, _FastSafetyMetricAccumulator):
                metric_acc.update(
                    state,
                    timing_out=timings if collect_timing else None,
                    synchronize_timing=bool(synchronize_timing),
                )
            else:
                metric_acc.update(state)
                if synchronize_timing:
                    # The standard accumulator generally transfers results to host
                    # inside update already. Keep this as a defensive barrier for
                    # cross-version Waymax/JAX behavior.
                    _block_until_ready_tree(state)
        if collect_timing:
            metric_s += time.perf_counter() - t
            t = time.perf_counter()
        done = False
        dci = int(done_check_interval)
        if dci > 0 and (((step + 1) % dci == 0) or ((step + 1) >= int(horizon_steps))):
            done = _state_done(state)
        if collect_timing:
            done_s += time.perf_counter() - t
        if done:
            break
    if metric_eval_mode == "final":
        t_metric = time.perf_counter()
        if isinstance(metric_acc, _FastSafetyMetricAccumulator):
            metric_acc.update(
                state,
                timing_out=timings if collect_timing else None,
                synchronize_timing=bool(synchronize_timing),
            )
        else:
            metric_acc.update(state)
            if synchronize_timing:
                _block_until_ready_tree(state)
        # Keep MetricSteps semantically tied to the rollout horizon even though
        # final-mode metrics are evaluated once on the complete SimulatorState.
        try:
            metric_acc.step_count = int(steps)
        except Exception:
            pass
        metric_s += time.perf_counter() - t_metric
    if collect_timing:
        timings["timing/action_s"] = float(action_s)
        timings["timing/env_step_s"] = float(env_step_s)
        timings["timing/metric_update_s"] = float(metric_s)
        timings["timing/done_check_s"] = float(done_s)
    t = time.perf_counter()
    metrics = metric_acc.finalize()
    out = _candidate_result_from_metrics(metrics, steps)
    out["metric_eval_interval"] = int(metric_eval_interval)
    if collect_timing:
        timings["timing/metric_finalize_s"] = time.perf_counter() - t
        out.update(timings)
    return out


def replay_candidate_on_state(
    init_state: Any,
    trajectory: np.ndarray,
    cfg: dict,
    *,
    horizon_steps: int = 80,
    action_mode: str = "absolute_xy_yaw",
    metric_set: str = "safety",
    metric_eval_mode: str = "step",
    metric_eval_interval: int = 1,
    metric_guard_radius_m: float = 8.0,
    metric_guard_window_steps: int = 1,
) -> dict[str, Any]:
    max_objects = _state_num_objects(init_state)
    sdc_index, initial_pose = _initial_sdc_pose(init_state)
    env = _make_waymax_environment(max_num_objects=max_objects, action_mode=action_mode)
    metric_objects, metric_errors = build_waymax_metric_objects(metric_names_for_set(metric_set))
    return replay_candidate_on_env(
        env,
        init_state,
        trajectory,
        cfg,
        horizon_steps=horizon_steps,
        action_mode=action_mode,
        metric_objects=metric_objects,
        metric_init_errors=metric_errors,
        num_objects=max_objects,
        sdc_index=sdc_index,
        initial_pose=initial_pose,
        metric_set=metric_set,
        metric_eval_mode=metric_eval_mode,
        metric_eval_interval=int(metric_eval_interval),
    )


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("scenario_id") or row.get("scenario/id") or ""), int(row.get("candidate_index", row.get("candidate", row.get("k", -1))))


def _jsonl_row_key_or_none(row: dict[str, Any]) -> tuple[str, int] | None:
    try:
        key = _row_key(row)
    except Exception:
        return None
    if key[0] and key[1] >= 0:
        return key
    return None


def _jsonl_row_is_successful(row: dict[str, Any]) -> bool:
    """Return whether an existing replay row should be treated as completed.

    GPU OOM and other transient Waymax/JAX failures are written as JSONL rows
    with ``rollout_valid=false`` and an ``error`` field. Those rows must not be
    considered done in resume mode; otherwise rerunning scripts/13 would skip
    exactly the candidates that need to be repaired.
    """
    for key in ("rollout_valid", "candidate_rollout_valid", "valid"):
        if key in row:
            value = row.get(key)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y"}
            return bool(value)
    # Backward compatibility for very old successful rows that did not store an
    # explicit validity flag. An error/status failure is never complete.
    if row.get("error"):
        return False
    status = str(row.get("status", "")).strip().lower()
    if status and any(tok in status for tok in ("failed", "error", "oom", "invalid")):
        return False
    return True


def repair_existing_outcomes_jsonl_for_resume(
    path: str | Path | None,
    *,
    retry_failed_existing: bool = True,
) -> tuple[set[tuple[str, int]], dict[str, int]]:
    """Read and repair an existing outcome JSONL before resume.

    A killed replay process can leave the final JSONL record partially written.
    Appending to such a file is dangerous: the next JSON object may be glued to
    the broken tail, and scripts/12 can later fail while loading outcomes. This
    helper keeps only complete JSON objects with a usable (scenario_id,
    candidate_index), drops corrupt/incomplete rows, deduplicates by key using
    the latest complete row, and rewrites the JSONL with one newline-terminated
    object per line.

    When retry_failed_existing=True, rows with rollout_valid=false or an error
    are also dropped during resume repair. This makes a rerun skip only already
    successful candidates and recompute failed candidates, which is important
    after transient GPU OOM failures.
    """
    stats = {
        "resume_existing_lines": 0,
        "resume_valid_rows": 0,
        "resume_dropped_empty_lines": 0,
        "resume_dropped_corrupt_lines": 0,
        "resume_dropped_keyless_rows": 0,
        "resume_dropped_failed_rows": 0,
        "resume_deduplicated_rows": 0,
        "resume_repaired_file": 0,
    }
    if path is None:
        return set(), stats
    p = Path(path)
    if not p.exists():
        return set(), stats

    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    key_order: list[tuple[str, int]] = []
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            stats["resume_existing_lines"] += 1
            line = raw.strip()
            if not line:
                stats["resume_dropped_empty_lines"] += 1
                continue
            try:
                row = json.loads(line)
            except Exception:
                stats["resume_dropped_corrupt_lines"] += 1
                continue
            if not isinstance(row, dict):
                stats["resume_dropped_keyless_rows"] += 1
                continue
            key = _jsonl_row_key_or_none(row)
            if key is None:
                stats["resume_dropped_keyless_rows"] += 1
                continue
            if retry_failed_existing and not _jsonl_row_is_successful(row):
                stats["resume_dropped_failed_rows"] += 1
                continue
            if key in rows_by_key:
                stats["resume_deduplicated_rows"] += 1
                try:
                    key_order.remove(key)
                except ValueError:
                    pass
            key_order.append(key)
            rows_by_key[key] = row

    stats["resume_valid_rows"] = len(rows_by_key)
    needs_rewrite = any(
        stats[k] > 0
        for k in (
            "resume_dropped_empty_lines",
            "resume_dropped_corrupt_lines",
            "resume_dropped_keyless_rows",
            "resume_dropped_failed_rows",
            "resume_deduplicated_rows",
        )
    )
    try:
        if p.stat().st_size > 0:
            with p.open("rb") as bf:
                bf.seek(-1, 2)
                needs_rewrite = needs_rewrite or (bf.read(1) != b"\n")
    except OSError:
        needs_rewrite = True

    if needs_rewrite:
        tmp = p.with_name(f"{p.name}.resume_repair.{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for key in key_order:
                row = rows_by_key[key]
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        tmp.replace(p)
        stats["resume_repaired_file"] = 1

    return set(rows_by_key.keys()), stats


def read_existing_outcome_keys(path: str | Path | None) -> set[tuple[str, int]]:
    # Backward-compatible public helper. The replay path uses the repairing
    # variant above so partially written JSONL tails cannot poison resume.
    keys, _ = repair_existing_outcomes_jsonl_for_resume(path, retry_failed_existing=True)
    return keys


def _read_existing_outcome_rows_by_scenario(path: str | Path | None) -> dict[str, dict[int, dict[str, Any]]]:
    """Read a repaired replay JSONL into a compact per-scene candidate map."""
    out: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    if path is None:
        return out
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            key = _jsonl_row_key_or_none(row)
            if key is None:
                continue
            out[key[0]][int(key[1])] = row
    return out


def _append_jsonl_lines(path: str | Path, lines: list[str]) -> None:
    if not lines:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    with p.open("ab+") as f:
        f.seek(0, 2)
        size = f.tell()
        if size > 0:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.seek(0, 2)
                f.write(b"\n")
        f.seek(0, 2)
        f.write(payload)


def _sid_to_cache_paths(cache_dir: Path, *, verify_cache_sid: bool = False, shard_index: int = 0, num_shards: int = 1) -> dict[str, Path]:
    paths = sorted(cache_dir.glob("*.npz"))
    if int(num_shards) > 1:
        paths = [p for i, p in enumerate(paths) if i % int(num_shards) == int(shard_index)]
    sid_to_path: dict[str, Path] = {}
    if not verify_cache_sid:
        return {p.stem: p for p in paths}
    for p in paths:
        try:
            arrays = load_npz_canonical(p, keys={"scenario/id", "womd/scenario/id"})
            sid_to_path[scenario_id_from_arrays(arrays, p)] = p
        except Exception:
            sid_to_path[p.stem] = p
    return sid_to_path




def _cache_path_has_womd_features(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as data:
            return any(str(k).startswith("womd__") or str(k).startswith("womd/") for k in data.files)
    except Exception:
        return False

def replay_cache_candidates_to_jsonl(
    *,
    cache_dir: str | Path,
    data_config: dict,
    outcomes_jsonl: str | Path,
    cfg: dict,
    tfexample_glob: str | None = None,
    split: str | None = None,
    candidate_selection: str = "balanced",
    max_candidates_per_scene: int | None = 8,
    horizon_steps: int = 80,
    action_mode: str = "absolute_xy_yaw",
    metric_set: str = "safety",
    limit_scenes: int | None = None,
    resume: bool = True,
    progress: bool = True,
    matched_only: bool = True,
    verify_cache_sid: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    tfexample_index_jsonl: str | Path | None = None,
    gc_every_scenes: int = 16,
    state_source: str = "auto",
    profile_replay_jsonl: str | Path | None = None,
    profile_detail: str = "candidate",
    profile_probe_candidates: int = 0,
    jit_env_step: bool = False,
    jit_env_reset: bool = False,
    done_check_interval: int = 1,
    metric_eval_mode: str = "step",
    metric_eval_interval: int = 1,
    metric_guard_radius_m: float = 8.0,
    metric_guard_window_steps: int = 1,
    retry_failed_existing: bool = True,
    attach_output_dir: str | Path | None = None,
    attach_compress: bool = False,
    attach_max_pending: int = 2,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    from cowp.waymax_eval.dataloader import simulator_state_from_tensor_cache_arrays, waymax_state_generator_for_sids, waymax_state_generator_with_ids

    cache_dir = Path(cache_dir)
    out_path = Path(outcomes_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sid_to_path = _sid_to_cache_paths(cache_dir, verify_cache_sid=verify_cache_sid, shard_index=shard_index, num_shards=num_shards)
    if limit_scenes is not None:
        keep = set(list(sid_to_path.keys())[: int(limit_scenes)])
        sid_to_path = {sid: p for sid, p in sid_to_path.items() if sid in keep}

    resume_repair_stats: dict[str, int] = {}
    if resume:
        if out_path.exists():
            done, resume_repair_stats = repair_existing_outcomes_jsonl_for_resume(out_path, retry_failed_existing=bool(retry_failed_existing))
        else:
            out_path.write_text("", encoding="utf-8")
            done = set()
            _, resume_repair_stats = repair_existing_outcomes_jsonl_for_resume(out_path, retry_failed_existing=bool(retry_failed_existing))
    else:
        out_path.write_text("", encoding="utf-8")
        done = set()
        _, resume_repair_stats = repair_existing_outcomes_jsonl_for_resume(out_path, retry_failed_existing=bool(retry_failed_existing))

    profile_path = Path(profile_replay_jsonl) if profile_replay_jsonl else None
    if profile_path is not None:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        # Do not truncate an existing profile on resume.  Outcome JSONL continuity
        # is independent of this file, but preserving old profile records makes it
        # possible to compare multiple profiling/optimization runs afterward.
        profile_path.touch(exist_ok=True)
    profile_detail = str(profile_detail or "candidate").strip().lower()
    if profile_detail not in {"scene", "candidate", "probe"}:
        raise ValueError(f"profile_detail must be scene, candidate, or probe, got {profile_detail!r}")
    # candidate: historical dispatch-oriented fine timing for every new candidate.
    # probe: only a small bounded set of new candidates is timed. Half use the
    # normal asynchronous dispatch path; half insert stage barriers so env.step,
    # OverlapMetric and OffroadMetric GPU work can be attributed accurately.
    collect_timing = profile_path is not None and profile_detail == "candidate"
    profile_probe_candidates = max(0, int(profile_probe_candidates or 0))
    probe_dispatch_budget = (profile_probe_candidates + 1) // 2
    probe_sync_budget = profile_probe_candidates // 2
    probe_dispatch_used = 0
    probe_sync_used = 0
    profile_run_id = f"pid{os.getpid()}_{time.time_ns()}"

    # Incremental attachment is independent of replay correctness.  JSONL remains
    # the source of truth; completed scene NPZs are written atomically in a bounded
    # background I/O thread and scripts/12 still performs a final reconciliation.
    attach_dir = Path(attach_output_dir) if attach_output_dir else None
    if attach_dir is not None:
        attach_dir.mkdir(parents=True, exist_ok=True)
        if attach_dir.resolve() == cache_dir.resolve():
            raise ValueError("--attach-output-dir must be different from --cache-dir; incremental attachment must not overwrite the core cache")
    known_rows_by_sid = _read_existing_outcome_rows_by_scenario(out_path) if attach_dir is not None else defaultdict(dict)
    attach_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"cowp-attach-{int(shard_index):03d}") if attach_dir is not None else None
    attach_pending: deque[Future] = deque()
    attach_written = 0
    attach_skipped = 0
    attach_failed = 0
    attach_max_pending = max(1, int(attach_max_pending or 1))

    def _consume_attach_future(fut: Future) -> None:
        nonlocal attach_written, attach_skipped, attach_failed
        try:
            result = fut.result()
        except Exception:
            attach_failed += 1
            raise
        if str(result.get("status", "")) == "written":
            attach_written += 1
        else:
            attach_skipped += 1

    def _drain_attach(*, all_pending: bool = False) -> None:
        if attach_executor is None:
            return
        while attach_pending and (all_pending or len(attach_pending) >= attach_max_pending):
            _consume_attach_future(attach_pending.popleft())

    def _submit_scene_attach(sid: str) -> None:
        if attach_executor is None or attach_dir is None or sid not in sid_to_path:
            return
        rows = list(known_rows_by_sid.get(sid, {}).values())
        if not rows:
            return
        src = sid_to_path[sid]
        dst = attach_dir / src.name
        attach_pending.append(attach_executor.submit(
            attach_rows_to_cache_file, src, dst, rows,
            compress=bool(attach_compress), skip_if_complete=True,
        ))
        _drain_attach(all_pending=False)

    total_written = 0
    total_failed = 0
    scenes_seen = 0
    scenes_matched = 0
    candidate_targets = 0
    candidate_seconds = 0.0
    remaining = set(sid_to_path.keys())

    metric_objects, metric_errors = build_waymax_metric_objects(metric_names_for_set(metric_set))
    env_cache: dict[tuple[int, str], Any] = {}
    env_step_fn_cache: dict[tuple[int, str], Any | None] = {}
    env_reset_fn_cache: dict[tuple[int, str], Any | None] = {}

    state_source = str(state_source or "auto").lower()
    if state_source not in {"auto", "cache", "tfexample"}:
        raise ValueError(f"state_source must be one of auto/cache/tfexample, got {state_source!r}")
    metric_eval_mode = str(metric_eval_mode or "step").lower()
    if metric_eval_mode not in {"step", "sampled", "adaptive", "final"}:
        raise ValueError(f"metric_eval_mode must be one of step/sampled/adaptive/final, got {metric_eval_mode!r}")
    metric_eval_interval = max(1, int(metric_eval_interval or 1))
    use_cache_state = False
    if state_source == "cache":
        use_cache_state = True
    elif state_source == "auto":
        sample_path = next(iter(sid_to_path.values()), None)
        use_cache_state = bool(sample_path is not None and _cache_path_has_womd_features(Path(sample_path)))

    # Fast path: build Waymax SimulatorState directly from the WOMD features
    # already stored inside tensor cache.  This avoids a second TFRecord scan and
    # prevents split/index mismatches from leaving matched=0 for hours.
    if use_cache_state:
        gen = ((sid, path) for sid, path in sid_to_path.items())
        iterator = tqdm_iter(gen, enabled=progress, total=len(sid_to_path) if sid_to_path else None, desc=(progress_desc or "Waymax candidate replay from tensor cache"), unit="scene")
    else:
        iterator_ref = {"iterator": None}

        def scan_progress(**kw):
            it = iterator_ref.get("iterator")
            if hasattr(it, "set_postfix"):
                it.set_postfix(stage="scan_tfexample", scanned=kw.get("scanned"), matched=kw.get("matched"), remaining=kw.get("remaining"), last=str(kw.get("last", ""))[:10], refresh=True)

        if matched_only:
            gen = waymax_state_generator_for_sids(data_config, set(sid_to_path.keys()), tfexample_glob=tfexample_glob, split=split, tfexample_index_jsonl=tfexample_index_jsonl, progress_callback=scan_progress)
        else:
            gen = waymax_state_generator_with_ids(data_config, tfexample_glob=tfexample_glob, split=split)
        iterator = tqdm_iter(gen, enabled=progress, total=len(sid_to_path) if sid_to_path else None, desc=(progress_desc or "Waymax candidate replay"), unit="scene")
        iterator_ref["iterator"] = iterator

    for item in iterator:
        scenes_seen += 1
        arrays = None
        preselected_indices: list[int] | None = None
        scene_t0 = time.perf_counter()
        scene_profile: dict[str, Any] = {
            "scenario_id": None,
            "source": "cache" if use_cache_state else "tfexample",
            "profile_run_id": profile_run_id,
            "profile_detail": profile_detail,
        }
        if use_cache_state:
            sid, cache_path = item
            sid = str(sid)
            scene_profile["scenario_id"] = sid

            # Resume fast-path.  The old code loaded all cached WOMD arrays and
            # rebuilt the Waymax state before discovering that every selected
            # candidate had already been written.  On large tensor-cache NPZs
            # this makes smoke/rerun passes appear extremely slow even when no
            # actual replay is done.  Here we first load only the tiny candidate
            # masks, reproduce the same selected candidate ids, and skip the
            # scene immediately if all selected rows are present in the outcome
            # JSONL.  If any row is missing, execution deliberately falls back
            # to the original full replay path, preserving closed-loop labels.
            if resume and done:
                try:
                    t = time.perf_counter()
                    select_arrays = load_cache_selection_arrays(cache_path)
                    scene_profile["load_select_s"] = time.perf_counter() - t
                    seed = _stable_seed(sid)
                    t = time.perf_counter()
                    preselected_indices = select_candidate_indices(
                        select_arrays,
                        cfg,
                        selection=candidate_selection,
                        max_candidates=max_candidates_per_scene,
                        seed=seed,
                    )
                    scene_profile["select_candidates_s"] = time.perf_counter() - t
                    pending_indices = [int(k) for k in preselected_indices if (sid, int(k)) not in done]
                    if not pending_indices:
                        scenes_matched += 1
                        candidate_targets += len(preselected_indices)
                        remaining.discard(sid)
                        if profile_path is not None:
                            scene_profile.update(
                                {
                                    "status": "ok",
                                    "resume_fast_skip": True,
                                    "candidates_selected": len(preselected_indices),
                                    "new_rows": 0,
                                    "failed_rows": 0,
                                    "resumed_rows": int(len(preselected_indices)),
                                    "load_npz_s": 0.0,
                                    "build_state_s": 0.0,
                                    "env_init_s": 0.0,
                                    "rollout_candidates_s": 0.0,
                                    "write_outcomes_s": 0.0,
                                    "seconds": time.perf_counter() - scene_t0,
                                }
                            )
                            with profile_path.open("a", encoding="utf-8") as pf:
                                pf.write(json.dumps(scene_profile, ensure_ascii=False, allow_nan=True) + "\n")
                        if hasattr(iterator, "set_postfix"):
                            mean_s = candidate_seconds / max(total_written + total_failed, 1)
                            iterator.set_postfix(
                                matched=scenes_matched,
                                rows=total_written,
                                failed=total_failed,
                                remaining=len(remaining),
                                cand_s=f"{mean_s:.3f}",
                                refresh=True,
                            )
                        _submit_scene_attach(sid)
                        if int(gc_every_scenes) > 0 and scenes_matched % int(gc_every_scenes) == 0:
                            gc.collect()
                        if not remaining:
                            break
                        continue
                except Exception as exc:
                    # The fast path is an optimization only.  If an old cache is
                    # missing one of the small selection keys, keep the old full
                    # load behavior rather than changing replay coverage.
                    scene_profile["resume_fast_skip_error"] = str(exc)

            t = time.perf_counter()
            arrays = load_cache_replay_arrays(cache_path)
            scene_profile["load_npz_s"] = time.perf_counter() - t
            try:
                t = time.perf_counter()
                init_state = simulator_state_from_tensor_cache_arrays(arrays, data_config, time_key="all")
                scene_profile["build_state_s"] = time.perf_counter() - t
            except Exception as exc:
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(stage="cache_state_failed", matched=scenes_matched, failed=total_failed, last=sid[:10], refresh=True)
                selected_arrays = {
                    k: arrays[k]
                    for k in (
                        "cowp/candidates/trajectory",
                        "cowp/candidates/valid",
                        "cowp/candidates/conventional_safe",
                        "cowp/candidates/false_safe",
                        "cowp/candidates/noncoercive_feasible",
                        "cowp/candidates/ego_utility_prior",
                    )
                    if k in arrays
                }
                seed = _stable_seed(sid)
                indices = preselected_indices if preselected_indices is not None else select_candidate_indices(selected_arrays, cfg, selection=candidate_selection, max_candidates=max_candidates_per_scene, seed=seed)
                failure_lines: list[str] = []
                for k in indices:
                    if (sid, int(k)) in done:
                        continue
                    row = {"scenario_id": sid, "candidate_index": int(k), "rollout_valid": False, "error": f"cache_state_failed: {exc}"}
                    failure_lines.append(json.dumps(row, ensure_ascii=False, allow_nan=True))
                    known_rows_by_sid[sid][int(k)] = row
                    done.add((sid, int(k)))
                    total_failed += 1
                _append_jsonl_lines(out_path, failure_lines)
                _submit_scene_attach(sid)
                remaining.discard(sid)
                if profile_path is not None:
                    scene_profile.update({"status": "cache_state_failed", "failed": total_failed, "seconds": time.perf_counter() - scene_t0})
                    with profile_path.open("a", encoding="utf-8") as pf:
                        pf.write(json.dumps(scene_profile, ensure_ascii=False, allow_nan=True) + "\n")
                continue
        else:
            sid, init_state = item
            sid = str(sid)
            if sid not in sid_to_path:
                continue
            scene_profile["scenario_id"] = sid
        scenes_matched += 1
        if arrays is None:
            t = time.perf_counter()
            arrays = load_npz_canonical(
                sid_to_path[sid],
                keys={
                    "cowp/candidates/trajectory",
                    "cowp/candidates/valid",
                    "cowp/candidates/conventional_safe",
                    "cowp/candidates/false_safe",
                    "cowp/candidates/noncoercive_feasible",
                    "cowp/candidates/ego_utility_prior",
                },
            )
            scene_profile["load_npz_s"] = time.perf_counter() - t
        seed = _stable_seed(sid)
        if preselected_indices is not None:
            indices = preselected_indices
            scene_profile.setdefault("select_candidates_s", 0.0)
        else:
            t = time.perf_counter()
            indices = select_candidate_indices(arrays, cfg, selection=candidate_selection, max_candidates=max_candidates_per_scene, seed=seed)
            scene_profile["select_candidates_s"] = time.perf_counter() - t
        candidate_targets += len(indices)
        trajs = np.asarray(arrays.get("cowp/candidates/trajectory", []), dtype=np.float32)

        try:
            t = time.perf_counter()
            max_objects = _state_num_objects(init_state)
            sdc_index, initial_pose = _initial_sdc_pose(init_state)
            env_key = (int(max_objects), str(action_mode))
            env = env_cache.get(env_key)
            if env is None:
                env = _make_waymax_environment(max_num_objects=max_objects, action_mode=action_mode)
                env_cache[env_key] = env
            step_fn = None
            reset_fn = None
            if bool(jit_env_step):
                if env_key not in env_step_fn_cache:
                    try:
                        env_step_fn_cache[env_key] = _make_jitted_env_step(env)
                    except Exception as jit_exc:
                        env_step_fn_cache[env_key] = None
                        scene_profile["jit_env_step_error"] = str(jit_exc)
                step_fn = env_step_fn_cache.get(env_key)
            if bool(jit_env_reset):
                if env_key not in env_reset_fn_cache:
                    try:
                        env_reset_fn_cache[env_key] = _make_jitted_env_reset(env)
                    except Exception as jit_exc:
                        env_reset_fn_cache[env_key] = None
                        scene_profile["jit_env_reset_error"] = str(jit_exc)
                reset_fn = env_reset_fn_cache.get(env_key)
            scene_profile["jit_env_step"] = bool(step_fn is not None)
            scene_profile["jit_env_reset"] = bool(reset_fn is not None)
            scene_profile["done_check_interval"] = int(done_check_interval)
            scene_profile["metric_eval_mode"] = str(metric_eval_mode)
            scene_profile["metric_eval_interval"] = int(metric_eval_interval)
            scene_profile["metric_guard_radius_m"] = float(metric_guard_radius_m)
            scene_profile["metric_guard_window_steps"] = int(metric_guard_window_steps)
            scene_profile["env_init_s"] = time.perf_counter() - t
        except Exception as exc:
            # If the scene itself cannot be initialized, record all selected rows
            # as invalid rather than silently leaving them unattached.
            failure_lines: list[str] = []
            for k in indices:
                if (sid, int(k)) in done:
                    continue
                row = {"scenario_id": sid, "candidate_index": int(k), "rollout_valid": False, "error": f"scene_init_failed: {exc}"}
                failure_lines.append(json.dumps(row, ensure_ascii=False, allow_nan=True))
                known_rows_by_sid[sid][int(k)] = row
                done.add((sid, int(k)))
                total_failed += 1
            _append_jsonl_lines(out_path, failure_lines)
            _submit_scene_attach(sid)
            remaining.discard(sid)
            if profile_path is not None:
                scene_profile.update({"status": "scene_init_failed", "failed": total_failed, "seconds": time.perf_counter() - scene_t0})
                with profile_path.open("a", encoding="utf-8") as pf:
                    pf.write(json.dumps(scene_profile, ensure_ascii=False, allow_nan=True) + "\n")
            continue

        rows_to_write: list[str] = []
        new_rows_scene = 0
        failed_rows_scene = 0
        skipped_rows_scene = 0
        rollout_s_scene = 0.0
        timing_sums = {k: 0.0 for k in _CANDIDATE_TIMING_KEYS}
        timing_count = 0
        probe_records_scene: list[dict[str, Any]] = []
        for k in indices:
            if (sid, int(k)) in done:
                skipped_rows_scene += 1
                continue
            row: dict[str, Any] = {"scenario_id": sid, "candidate_index": int(k), "rollout_valid": False}
            t0 = time.perf_counter()
            try:
                if trajs.ndim != 3 or not (0 <= int(k) < trajs.shape[0]):
                    raise ValueError(f"missing candidate trajectory for k={k}")
                metric_guard_steps = None
                if str(metric_eval_mode).lower() == "adaptive":
                    metric_guard_steps = _candidate_metric_guard_steps(
                        arrays,
                        trajs[int(k)],
                        horizon_steps=int(horizon_steps),
                        radius_m=float(metric_guard_radius_m),
                        window_steps=int(metric_guard_window_steps),
                    )
                candidate_collect_timing = bool(collect_timing)
                candidate_sync_timing = False
                candidate_probe_mode: str | None = None
                if profile_path is not None and profile_detail == "probe":
                    if probe_dispatch_used < probe_dispatch_budget:
                        candidate_collect_timing = True
                        candidate_probe_mode = "dispatch"
                        probe_dispatch_used += 1
                    elif probe_sync_used < probe_sync_budget:
                        candidate_collect_timing = True
                        candidate_sync_timing = True
                        candidate_probe_mode = "sync"
                        probe_sync_used += 1
                outcome = replay_candidate_on_env(
                    env,
                    init_state,
                    trajs[int(k)],
                    cfg,
                    horizon_steps=int(horizon_steps),
                    action_mode=action_mode,
                    metric_objects=metric_objects,
                    metric_init_errors=metric_errors,
                    num_objects=max_objects,
                    sdc_index=sdc_index,
                    initial_pose=initial_pose,
                    metric_set=metric_set,
                    collect_timing=candidate_collect_timing,
                    synchronize_timing=candidate_sync_timing,
                    step_fn=step_fn,
                    reset_fn=reset_fn,
                    done_check_interval=int(done_check_interval),
                    metric_eval_mode=str(metric_eval_mode),
                    metric_eval_interval=int(metric_eval_interval),
                    metric_guard_steps=metric_guard_steps,
                )
                row.update(outcome)
                if candidate_collect_timing:
                    timing_count += 1
                    for _tk in _CANDIDATE_TIMING_KEYS:
                        if _tk in row:
                            try:
                                timing_sums[_tk] += float(row[_tk])
                            except Exception:
                                pass
                    if candidate_probe_mode is not None:
                        probe_record: dict[str, Any] = {
                            "candidate_index": int(k),
                            "mode": str(candidate_probe_mode),
                        }
                        for _tk, _tv in row.items():
                            if str(_tk).startswith("timing/") or _tk in {"steps", "rollout_seconds"}:
                                try:
                                    probe_record[str(_tk)] = float(_tv)
                                except Exception:
                                    probe_record[str(_tk)] = _tv
                        probe_records_scene.append(probe_record)
                total_written += 1
                new_rows_scene += 1
            except Exception as exc:
                row.update({"rollout_valid": False, "error": str(exc)})
                total_failed += 1
                failed_rows_scene += 1
            sec = time.perf_counter() - t0
            rollout_s_scene += sec
            candidate_seconds += sec
            row["rollout_seconds"] = float(sec)
            row["action_mode"] = str(action_mode)
            row["metric_set"] = str(metric_set)
            row["metric_eval_mode"] = str(metric_eval_mode)
            row["metric_eval_interval"] = int(metric_eval_interval)
            known_rows_by_sid[sid][int(k)] = dict(row)
            rows_to_write.append(json.dumps(row, ensure_ascii=False, allow_nan=True))
            done.add((sid, int(k)))
        t = time.perf_counter()
        if rows_to_write:
            _append_jsonl_lines(out_path, rows_to_write)
        write_outcomes_s = time.perf_counter() - t
        _submit_scene_attach(sid)
        remaining.discard(sid)
        if profile_path is not None:
            scene_profile.update(
                {
                    "status": "ok",
                    "candidates_selected": len(indices),
                    "new_rows": int(new_rows_scene),
                    "failed_rows": int(failed_rows_scene),
                    "resumed_rows": int(skipped_rows_scene),
                    "rollout_candidates_s": float(rollout_s_scene),
                    "mean_rollout_candidate_s": float(rollout_s_scene / max(new_rows_scene + failed_rows_scene, 1)),
                    "write_outcomes_s": float(write_outcomes_s),
                    "seconds": time.perf_counter() - scene_t0,
                }
            )
            if timing_count > 0:
                for _tk, _val in timing_sums.items():
                    suffix = _tk.split("/", 1)[1] if "/" in _tk else _tk
                    scene_profile[f"timing_sum/{suffix}"] = float(_val)
                    scene_profile[f"timing_mean/{suffix}"] = float(_val / max(timing_count, 1))
            if probe_records_scene:
                scene_profile["timing_probe_candidates"] = probe_records_scene
                scene_profile["timing_probe_dispatch_used_total"] = int(probe_dispatch_used)
                scene_profile["timing_probe_sync_used_total"] = int(probe_sync_used)
            with profile_path.open("a", encoding="utf-8") as pf:
                pf.write(json.dumps(scene_profile, ensure_ascii=False, allow_nan=True) + "\n")
        if hasattr(iterator, "set_postfix"):
            mean_s = candidate_seconds / max(total_written + total_failed, 1)
            iterator.set_postfix(matched=scenes_matched, rows=total_written, failed=total_failed, npz=attach_written, pending_npz=len(attach_pending), remaining=len(remaining), cand_s=f"{mean_s:.3f}", refresh=True)
        del arrays
        if int(gc_every_scenes) > 0 and scenes_matched % int(gc_every_scenes) == 0:
            gc.collect()
        if not remaining:
            break
    _drain_attach(all_pending=True)
    if attach_executor is not None:
        attach_executor.shutdown(wait=True)
    return {
        "cache_dir": str(cache_dir),
        "outcomes_jsonl": str(out_path),
        "cache_scenes": len(sid_to_path),
        "waymax_scenes_seen": scenes_seen,
        "scenes_matched": scenes_matched,
        "candidate_targets": candidate_targets,
        "rows_written_or_resumed": len(done),
        **resume_repair_stats,
        "new_success_rows": total_written,
        "new_failed_rows": total_failed,
        "unmatched_cache_scenes": len(remaining),
        "mean_seconds_per_new_candidate": candidate_seconds / max(total_written + total_failed, 1),
        "action_mode": action_mode,
        "metric_set": metric_set,
        "matched_only_generator": bool(matched_only),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "tfexample_index_jsonl": str(tfexample_index_jsonl) if tfexample_index_jsonl else None,
        "gc_every_scenes": int(gc_every_scenes),
        "state_source": "cache" if use_cache_state else "tfexample",
        "profile_replay_jsonl": str(profile_path) if profile_path is not None else None,
        "profile_detail": str(profile_detail),
        "profile_probe_candidates": int(profile_probe_candidates),
        "profile_probe_dispatch_used": int(probe_dispatch_used),
        "profile_probe_sync_used": int(probe_sync_used),
        "profile_run_id": profile_run_id,
        "jit_env_step": bool(jit_env_step),
        "jit_env_reset": bool(jit_env_reset),
        "attach_output_dir": str(attach_dir) if attach_dir is not None else None,
        "incremental_npz_written": int(attach_written),
        "incremental_npz_skipped_complete": int(attach_skipped),
        "incremental_npz_failed": int(attach_failed),
        "done_check_interval": int(done_check_interval),
        "metric_eval_mode": str(metric_eval_mode),
        "metric_eval_interval": int(metric_eval_interval),
        "metric_guard_radius_m": float(metric_guard_radius_m),
        "metric_guard_window_steps": int(metric_guard_window_steps),
        "retry_failed_existing": bool(retry_failed_existing),
    }
