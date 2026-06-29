from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cowp.planning.cowp_planner import COWPPlanner
from cowp.utils.progress import tqdm_iter
from cowp.waymax_eval.metrics_standard import WaymaxStandardMetricAccumulator, build_waymax_metric_objects
from cowp.waymax_eval.policy_wrapper import _to_numpy, _wrap_angle, extract_current_agent_state
from cowp.waymax_eval.rollout import _env_step, _make_waymax_environment, _state_done


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
    cap = K if max_candidates is None or int(max_candidates) <= 0 else min(K, int(max_candidates))
    if selection == "all":
        return np.where(valid)[0].astype(int).tolist()[:cap]
    if selection == "noncoercive":
        m = valid & _safe_bool(arrays, "cowp/candidates/noncoercive_feasible", valid)
        return np.where(m)[0].astype(int).tolist()[:cap]
    if selection == "false_safe":
        m = valid & _safe_bool(arrays, "cowp/candidates/false_safe", valid)
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
    ncf = valid & _safe_bool(arrays, "cowp/candidates/noncoercive_feasible", valid)
    fs = valid & _safe_bool(arrays, "cowp/candidates/false_safe", valid)
    conv = valid & np.asarray(arrays.get("cowp/candidates/conventional_safe", valid), dtype=bool)
    buckets.extend([np.where(ncf)[0], np.where(fs)[0], np.where(conv & ~ncf & ~fs)[0], np.where(valid & ~conv)[0], np.where(valid)[0]])

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

    def _trajectory_to_action_fast(self, step: int) -> Any:
        try:
            from waymax import datatypes  # type: ignore
            import jax.numpy as jnp  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("waymax.datatypes and jax are required for candidate replay") from exc
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
) -> dict[str, Any]:
    state = env.reset(init_state)
    policy = FixedCandidateReplayPolicy(
        trajectory=np.asarray(trajectory, dtype=np.float32),
        cfg=cfg,
        action_mode=action_mode,
        sdc_index=sdc_index,
        num_objects=num_objects,
        initial_pose=initial_pose,
    )
    metric_acc = WaymaxStandardMetricAccumulator(metric_objects=metric_objects or [], init_errors=metric_init_errors or {})
    steps = 0
    for step in range(int(horizon_steps)):
        action = policy(state, step=step)
        state = _env_step(env, state, action)
        steps += 1
        metric_acc.update(state)
        if _state_done(state):
            break
    metrics = metric_acc.finalize()
    return _candidate_result_from_metrics(metrics, steps)


def replay_candidate_on_state(
    init_state: Any,
    trajectory: np.ndarray,
    cfg: dict,
    *,
    horizon_steps: int = 80,
    action_mode: str = "absolute_xy_yaw",
    metric_set: str = "safety",
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
    )


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("scenario_id") or row.get("scenario/id") or ""), int(row.get("candidate_index", row.get("candidate", row.get("k", -1))))


def read_existing_outcome_keys(path: str | Path | None) -> set[tuple[str, int]]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = _row_key(row)
                if key[0] and key[1] >= 0:
                    keys.add(key)
            except Exception:
                continue
    return keys


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
) -> dict[str, Any]:
    from cowp.waymax_eval.dataloader import waymax_state_generator_for_sids, waymax_state_generator_with_ids

    cache_dir = Path(cache_dir)
    out_path = Path(outcomes_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sid_to_path = _sid_to_cache_paths(cache_dir, verify_cache_sid=verify_cache_sid, shard_index=shard_index, num_shards=num_shards)
    if limit_scenes is not None:
        keep = set(list(sid_to_path.keys())[: int(limit_scenes)])
        sid_to_path = {sid: p for sid, p in sid_to_path.items() if sid in keep}

    done = read_existing_outcome_keys(out_path) if resume else set()
    if not resume:
        out_path.write_text("", encoding="utf-8")
    elif not out_path.exists():
        out_path.write_text("", encoding="utf-8")

    total_written = 0
    total_failed = 0
    scenes_seen = 0
    scenes_matched = 0
    candidate_targets = 0
    candidate_seconds = 0.0
    remaining = set(sid_to_path.keys())

    metric_objects, metric_errors = build_waymax_metric_objects(metric_names_for_set(metric_set))

    # Prefer the cache-matched generator: it scans tf.Examples cheaply and only
    # materializes Waymax SimulatorState for sids in the tensor cache.
    iterator_ref = {"iterator": None}

    def scan_progress(**kw):
        it = iterator_ref.get("iterator")
        if hasattr(it, "set_postfix"):
            it.set_postfix(stage="scan_tfexample", scanned=kw.get("scanned"), matched=kw.get("matched"), remaining=kw.get("remaining"), last=str(kw.get("last", ""))[:10], refresh=True)

    if matched_only:
        gen = waymax_state_generator_for_sids(data_config, set(sid_to_path.keys()), tfexample_glob=tfexample_glob, split=split, progress_callback=scan_progress)
    else:
        gen = waymax_state_generator_with_ids(data_config, tfexample_glob=tfexample_glob, split=split)
    iterator = tqdm_iter(gen, enabled=progress, total=len(sid_to_path) if sid_to_path else None, desc="Waymax candidate replay", unit="scene")
    iterator_ref["iterator"] = iterator

    for sid, init_state in iterator:
        scenes_seen += 1
        sid = str(sid)
        if sid not in sid_to_path:
            continue
        scenes_matched += 1
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
        seed = _stable_seed(sid)
        indices = select_candidate_indices(arrays, cfg, selection=candidate_selection, max_candidates=max_candidates_per_scene, seed=seed)
        candidate_targets += len(indices)
        trajs = np.asarray(arrays.get("cowp/candidates/trajectory", []), dtype=np.float32)

        try:
            max_objects = _state_num_objects(init_state)
            sdc_index, initial_pose = _initial_sdc_pose(init_state)
            env = _make_waymax_environment(max_num_objects=max_objects, action_mode=action_mode)
        except Exception as exc:
            # If the scene itself cannot be initialized, record all selected rows
            # as invalid rather than silently leaving them unattached.
            with out_path.open("a", encoding="utf-8") as f:
                for k in indices:
                    if (sid, int(k)) in done:
                        continue
                    row = {"scenario_id": sid, "candidate_index": int(k), "rollout_valid": False, "error": f"scene_init_failed: {exc}"}
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
                    done.add((sid, int(k)))
                    total_failed += 1
            remaining.discard(sid)
            continue

        with out_path.open("a", encoding="utf-8") as f:
            for k in indices:
                if (sid, int(k)) in done:
                    continue
                row: dict[str, Any] = {"scenario_id": sid, "candidate_index": int(k), "rollout_valid": False}
                t0 = time.perf_counter()
                try:
                    if trajs.ndim != 3 or not (0 <= int(k) < trajs.shape[0]):
                        raise ValueError(f"missing candidate trajectory for k={k}")
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
                    )
                    row.update(outcome)
                    total_written += 1
                except Exception as exc:
                    row.update({"rollout_valid": False, "error": str(exc)})
                    total_failed += 1
                sec = time.perf_counter() - t0
                candidate_seconds += sec
                row["rollout_seconds"] = float(sec)
                row["action_mode"] = str(action_mode)
                row["metric_set"] = str(metric_set)
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
                f.flush()
                done.add((sid, int(k)))
        remaining.discard(sid)
        if hasattr(iterator, "set_postfix"):
            mean_s = candidate_seconds / max(total_written + total_failed, 1)
            iterator.set_postfix(matched=scenes_matched, rows=total_written, failed=total_failed, remaining=len(remaining), cand_s=f"{mean_s:.3f}", refresh=True)
        del arrays
        gc.collect()
        if not remaining:
            break
    return {
        "cache_dir": str(cache_dir),
        "outcomes_jsonl": str(out_path),
        "cache_scenes": len(sid_to_path),
        "waymax_scenes_seen": scenes_seen,
        "scenes_matched": scenes_matched,
        "candidate_targets": candidate_targets,
        "rows_written_or_resumed": len(done),
        "new_success_rows": total_written,
        "new_failed_rows": total_failed,
        "unmatched_cache_scenes": len(remaining),
        "mean_seconds_per_new_candidate": candidate_seconds / max(total_written + total_failed, 1),
        "action_mode": action_mode,
        "metric_set": metric_set,
        "matched_only_generator": bool(matched_only),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
    }
