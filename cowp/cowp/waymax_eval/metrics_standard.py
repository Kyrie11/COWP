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


def waymax_metric_dict(rollout_state):
    try:
        from waymax import metrics  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("waymax.metrics is required to compute official Waymax standard metrics.") from exc
    metric_objects = [metrics.OverlapMetric(), metrics.OffroadMetric(), metrics.WrongWayMetric(), metrics.RouteProgressMetric(), metrics.KinematicsInfeasibilityMetric(), metrics.LogDivergenceMetric()]
    out = {}
    for m in metric_objects:
        result = m.compute(rollout_state)
        out[m.__class__.__name__] = result
    return out
