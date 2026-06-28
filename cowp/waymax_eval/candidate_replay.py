from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cowp.planning.cowp_planner import COWPPlanner
from cowp.utils.progress import tqdm_iter
from cowp.waymax_eval.metrics_standard import WaymaxStandardMetricAccumulator
from cowp.waymax_eval.policy_wrapper import _to_numpy, _wrap_angle, extract_current_agent_state
from cowp.waymax_eval.rollout import _env_step, _make_waymax_environment, _state_done


def restore_key(k: str) -> str:
    return k.replace("__", "/")


def store_key(k: str) -> str:
    return k.replace("/", "__")


def load_npz_canonical(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {restore_key(k): data[k] for k in data.files}


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


def select_candidate_indices(
    arrays: dict[str, np.ndarray],
    cfg: dict,
    *,
    selection: str = "balanced",
    max_candidates: int | None = 8,
    seed: int = 0,
) -> list[int]:
    """Return deterministic candidate indices to replay for one scene.

    ``all`` can be very expensive for large train sets.  ``balanced`` is the
    default because it covers the candidates that matter for COWP training:
    planner-selected, NCF, false-safe, conventional-safe and remaining valid
    candidates.  The returned order is stable so the generated JSONL can be
    resumed and compared across runs.
    """
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
    # 1) Planner-selected candidate.
    try:
        dec = COWPPlanner(cfg).select_from_labels(arrays)
        if int(dec.candidate_index) >= 0:
            buckets.append(np.asarray([int(dec.candidate_index)], dtype=np.int64))
    except Exception:
        pass
    # 2) Positive/negative COWP training categories.
    ncf = valid & _safe_bool(arrays, "cowp/candidates/noncoercive_feasible", valid)
    fs = valid & _safe_bool(arrays, "cowp/candidates/false_safe", valid)
    conv = valid & np.asarray(arrays.get("cowp/candidates/conventional_safe", valid), dtype=bool)
    buckets.extend([np.where(ncf)[0], np.where(fs)[0], np.where(conv & ~ncf & ~fs)[0], np.where(valid & ~conv)[0], np.where(valid)[0]])

    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    seen: set[int] = set()
    # Round-robin through buckets so rare NCF/false-safe examples are not crowded
    # out by conventional candidates.
    bucket_lists = []
    for b in buckets:
        b = np.asarray([int(x) for x in b if 0 <= int(x) < K and valid[int(x)]], dtype=np.int64)
        if b.size:
            # Stable pseudo-random order within a scene, controlled by scenario hash.
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


@dataclass
class FixedCandidateReplayPolicy:
    trajectory: np.ndarray
    cfg: dict
    action_mode: str = "delta_xy_yaw"

    def __post_init__(self) -> None:
        self.trajectory = np.asarray(self.trajectory, dtype=np.float32)
        if self.trajectory.ndim != 2 or self.trajectory.shape[1] < 3:
            raise ValueError(f"candidate trajectory must have shape [T,>=3], got {self.trajectory.shape}")

    def _trajectory_to_action(self, state: Any, step: int) -> Any:
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
        return self._trajectory_to_action(state, 0 if step is None else int(step))


def replay_candidate_on_state(
    init_state: Any,
    trajectory: np.ndarray,
    cfg: dict,
    *,
    horizon_steps: int = 80,
    action_mode: str = "delta_xy_yaw",
) -> dict[str, Any]:
    """Replay one fixed ego candidate in Waymax and return candidate-level outcome."""
    max_objects = getattr(init_state, "num_objects", None)
    if max_objects is None and hasattr(init_state, "log_trajectory"):
        max_objects = getattr(init_state.log_trajectory, "num_objects", None)
    env = _make_waymax_environment(max_num_objects=max_objects, action_mode=action_mode)
    state = env.reset(init_state)
    policy = FixedCandidateReplayPolicy(trajectory=np.asarray(trajectory, dtype=np.float32), cfg=cfg, action_mode=action_mode)
    metric_acc = WaymaxStandardMetricAccumulator()
    steps = 0
    for step in range(int(horizon_steps)):
        action = policy(state, step=step)
        state = _env_step(env, state, action)
        steps += 1
        metric_acc.update(state)
        if _state_done(state):
            break
    metrics = metric_acc.finalize()
    collision = bool(float(metrics.get("CollisionRate", 0.0)) > 0.0 or float(metrics.get("WaymaxAny/OverlapMetric", 0.0)) > 0.0)
    offroad = bool(float(metrics.get("OffroadRate", 0.0)) > 0.0 or float(metrics.get("WaymaxAny/OffroadMetric", 0.0)) > 0.0)
    logdiv = metrics.get("LogDivergence", np.nan)
    try:
        logdiv_f = float(logdiv)
    except Exception:
        logdiv_f = float("nan")
    out = {
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
    action_mode: str = "delta_xy_yaw",
    limit_scenes: int | None = None,
    resume: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    from cowp.waymax_eval.dataloader import waymax_state_generator_with_ids

    cache_dir = Path(cache_dir)
    out_path = Path(outcomes_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_paths = sorted(cache_dir.glob("*.npz"))
    sid_to_path: dict[str, Path] = {}
    for p in cache_paths:
        try:
            arrays = load_npz_canonical(p)
            sid_to_path[scenario_id_from_arrays(arrays, p)] = p
        except Exception:
            continue
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
    gen = waymax_state_generator_with_ids(data_config, tfexample_glob=tfexample_glob, split=split)
    iterator = tqdm_iter(gen, enabled=progress, total=len(sid_to_path) if sid_to_path else None, desc="Waymax candidate replay", unit="scene")
    remaining = set(sid_to_path.keys())
    for sid, init_state in iterator:
        scenes_seen += 1
        sid = str(sid)
        if sid not in sid_to_path:
            continue
        scenes_matched += 1
        arrays = load_npz_canonical(sid_to_path[sid])
        seed = abs(hash(sid)) % (2**32)
        indices = select_candidate_indices(arrays, cfg, selection=candidate_selection, max_candidates=max_candidates_per_scene, seed=seed)
        candidate_targets += len(indices)
        trajs = np.asarray(arrays.get("cowp/candidates/trajectory", []), dtype=np.float32)
        with out_path.open("a", encoding="utf-8") as f:
            for k in indices:
                if (sid, int(k)) in done:
                    continue
                row: dict[str, Any] = {"scenario_id": sid, "candidate_index": int(k), "rollout_valid": False}
                try:
                    if trajs.ndim != 3 or not (0 <= int(k) < trajs.shape[0]):
                        raise ValueError(f"missing candidate trajectory for k={k}")
                    outcome = replay_candidate_on_state(
                        init_state,
                        trajs[int(k)],
                        cfg,
                        horizon_steps=int(horizon_steps),
                        action_mode=action_mode,
                    )
                    row.update(outcome)
                    total_written += 1
                except Exception as exc:
                    row.update({"rollout_valid": False, "error": str(exc)})
                    total_failed += 1
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
                done.add((sid, int(k)))
        remaining.discard(sid)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(matched=scenes_matched, rows=total_written, failed=total_failed, remaining=len(remaining), refresh=True)
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
    }
