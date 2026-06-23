from __future__ import annotations

import numpy as np

from cowp.geometry.collision import unsafe_between


def closed_loop_standard_metrics(ego_trajs: list[np.ndarray], other_trajs: list[list[np.ndarray]], cfg: dict, route_lengths: list[float] | None = None) -> dict[str, float]:
    n = len(ego_trajs)
    collisions = 0
    offroad = 0
    progress = []
    for i, ego in enumerate(ego_trajs):
        collided = False
        for other in other_trajs[i] if i < len(other_trajs) else []:
            if unsafe_between(ego, other, cfg).collision:
                collided = True
                break
        collisions += int(collided)
        prog = float(np.linalg.norm(ego[-1, :2] - ego[0, :2])) if len(ego) else 0.0
        denom = route_lengths[i] if route_lengths and i < len(route_lengths) else max(prog, 1.0)
        progress.append(prog / max(float(denom), 1.0))
    return {"CR": collisions / max(n, 1), "Offroad": offroad / max(n, 1), "EP": float(np.mean(progress)) if progress else 0.0}


def _device_get(x):
    """Move JAX/PyTorch metric outputs to host when possible."""
    try:
        import jax  # type: ignore

        return jax.device_get(x)
    except Exception:
        return x


def waymax_metric_dict(rollout_state):
    try:
        from waymax import metrics  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("waymax.metrics is required to compute official Waymax standard metrics.") from exc
    metric_objects = [metrics.OverlapMetric(), metrics.OffroadMetric(), metrics.WrongWayMetric(), metrics.RouteProgressMetric(), metrics.KinematicsInfeasibilityMetric(), metrics.LogDivergenceMetric()]
    out = {}
    for m in metric_objects:
        result = m.compute(rollout_state)
        out[m.__class__.__name__] = _device_get(result)
    return out


def _numeric_metric_value(x) -> float | None:
    """Best-effort scalar extraction from Waymax metric outputs."""
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        arr = arr.astype(float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        return float(np.mean(finite))
    except Exception:
        pass
    for attr in ("value", "result", "metric_value"):
        if hasattr(x, attr):
            val = _numeric_metric_value(getattr(x, attr))
            if val is not None:
                return val
    if isinstance(x, dict):
        vals = [_numeric_metric_value(v) for v in x.values()]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    return None


def aggregate_waymax_standard_metrics(rollouts: list[dict]) -> dict[str, float]:
    """Aggregate official Waymax metric objects into JSON-ready scalars."""
    buckets: dict[str, list[float]] = {}
    for item in rollouts:
        metrics = item.get("standard_metrics", {}) or {}
        for name, value in metrics.items():
            scalar = _numeric_metric_value(value)
            if scalar is not None:
                buckets.setdefault(name, []).append(scalar)
    return {name: float(np.mean(vals)) for name, vals in sorted(buckets.items()) if vals}
