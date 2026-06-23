from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

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


# Canonical metric names used by this project, with aliases across Waymax releases.
# Current public Waymax exposes ProgressionMetric/OffRouteMetric for route metrics,
# not RouteProgressMetric.  Some local builds also keep classes inside submodules
# without re-exporting them from waymax.metrics.
_WAYMAX_METRIC_SPECS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("OverlapMetric", (("waymax.metrics", "OverlapMetric"), ("waymax.metrics.overlap", "OverlapMetric"))),
    ("OffroadMetric", (("waymax.metrics", "OffroadMetric"), ("waymax.metrics.roadgraph", "OffroadMetric"))),
    ("WrongWayMetric", (("waymax.metrics", "WrongWayMetric"), ("waymax.metrics.roadgraph", "WrongWayMetric"))),
    # RouteProgressMetric is an older/non-public name.  ProgressionMetric is the
    # Waymax API name for SDC route progress.
    ("ProgressionMetric", (("waymax.metrics", "ProgressionMetric"), ("waymax.metrics.route", "ProgressionMetric"), ("waymax.metrics", "RouteProgressMetric"))),
    ("OffRouteMetric", (("waymax.metrics", "OffRouteMetric"), ("waymax.metrics.route", "OffRouteMetric"))),
    ("KinematicsInfeasibilityMetric", (("waymax.metrics", "KinematicsInfeasibilityMetric"), ("waymax.metrics.comfort", "KinematicsInfeasibilityMetric"))),
    ("LogDivergenceMetric", (("waymax.metrics", "LogDivergenceMetric"), ("waymax.metrics.imitation", "LogDivergenceMetric"))),
)

_EVENT_METRICS = {
    "OverlapMetric",
    "OffroadMetric",
    "WrongWayMetric",
    "OffRouteMetric",
    "KinematicsInfeasibilityMetric",
}


def _device_get(x: Any) -> Any:
    """Move JAX metric outputs to host when possible."""
    try:
        import jax  # type: ignore

        return jax.device_get(x)
    except Exception:
        return x


def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(_device_get(x))


def _resolve_class(candidates: tuple[tuple[str, str], ...]) -> type | None:
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        cls = getattr(module, class_name, None)
        if cls is not None:
            return cls
    return None


def _available_waymax_metric_classes() -> tuple[list[tuple[str, type]], dict[str, str]]:
    available: list[tuple[str, type]] = []
    missing: dict[str, str] = {}
    for canonical_name, candidates in _WAYMAX_METRIC_SPECS:
        cls = _resolve_class(candidates)
        if cls is None:
            aliases = ", ".join(f"{m}.{c}" for m, c in candidates)
            missing[canonical_name] = f"unavailable; tried {aliases}"
        else:
            available.append((canonical_name, cls))
    return available, missing


def build_waymax_metric_objects() -> tuple[list[tuple[str, Any]], dict[str, str]]:
    """Instantiate all Waymax metrics available in the installed API.

    The function is deliberately tolerant: unavailable or renamed metrics are
    skipped and reported in the returned diagnostics instead of crashing the
    entire closed-loop rollout.  This keeps --waymax-standard-metrics usable
    across Waymax releases.
    """
    available, missing = _available_waymax_metric_classes()
    objects: list[tuple[str, Any]] = []
    errors = dict(missing)
    for canonical_name, cls in available:
        try:
            objects.append((canonical_name, cls()))
        except Exception as exc:
            errors[canonical_name] = f"failed to instantiate {cls}: {exc}"
    return objects, errors


def _sdc_index_from_state(state: Any) -> int | None:
    meta = getattr(state, "object_metadata", None)
    is_sdc = getattr(meta, "is_sdc", None) if meta is not None else getattr(state, "is_sdc", None)
    if is_sdc is None:
        return None
    try:
        arr = _to_numpy(is_sdc)
    except Exception:
        return None
    while arr.ndim > 1:
        arr = arr[0]
    if arr.size == 0:
        return None
    return int(np.argmax(arr.astype(float)))


def _metric_value_and_valid(result: Any) -> tuple[np.ndarray, np.ndarray | None]:
    if hasattr(result, "value"):
        value = _to_numpy(getattr(result, "value"))
        valid = _to_numpy(getattr(result, "valid")) if hasattr(result, "valid") else None
        return value, valid
    if isinstance(result, dict) and "value" in result:
        value = _to_numpy(result["value"])
        valid = _to_numpy(result["valid"]) if "valid" in result else None
        return value, valid
    return _to_numpy(result), None


def metric_result_to_sdc_scalar(result: Any, state: Any | None = None) -> float | None:
    """Extract a JSON-safe scalar from a Waymax MetricResult.

    Waymax metrics usually return a MetricResult with shape (..., num_objects).
    Paper-level closed-loop metrics are ego/SDC metrics, so when an SDC mask is
    available this function selects the SDC entry.  If no SDC mask is available,
    it falls back to the mean finite valid value.
    """
    try:
        value, valid = _metric_value_and_valid(result)
        value = np.asarray(value, dtype=np.float32)
        if valid is not None:
            valid = np.asarray(valid, dtype=bool)
    except Exception:
        return _numeric_metric_value(result)

    # Remove leading singleton/batch dimensions in the common unbatched case.
    sdc_index = _sdc_index_from_state(state) if state is not None else None
    v = value
    m = valid
    while v.ndim > 1:
        v = v[0]
        if m is not None:
            m = m[0]
    if v.ndim == 1 and sdc_index is not None and 0 <= sdc_index < v.shape[0]:
        if m is not None and m.ndim == 1 and not bool(m[sdc_index]):
            return None
        scalar = float(v[sdc_index])
        return scalar if np.isfinite(scalar) else None

    if m is not None and m.shape == v.shape:
        v = v[m]
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def waymax_metric_dict(rollout_state: Any, *, include_errors: bool = True) -> dict[str, float | dict[str, str]]:
    """Compute installed Waymax metrics for one SimulatorState.

    This function is kept for compatibility.  For full closed-loop evaluation,
    prefer WaymaxStandardMetricAccumulator, which aggregates event metrics over
    every simulated step rather than only the final timestep.
    """
    metric_objects, errors = build_waymax_metric_objects()
    out: dict[str, float | dict[str, str]] = {}
    for name, metric in metric_objects:
        try:
            result = metric.compute(rollout_state)
            scalar = metric_result_to_sdc_scalar(result, rollout_state)
            if scalar is not None:
                out[name] = scalar
        except Exception as exc:
            errors[name] = str(exc)
    if include_errors and errors:
        out["metric_errors"] = errors
    return out


@dataclass
class WaymaxStandardMetricAccumulator:
    """Episode-level accumulator for Waymax closed-loop standard metrics.

    Collision/offroad/wrong-way/off-route/kinematic metrics are event metrics:
    the episode is positive if the SDC is positive at any simulated step.  Route
    progress is read from the final available ProgressionMetric value.  Log
    divergence is averaged over simulated steps.
    """

    metric_objects: list[tuple[str, Any]] = field(default_factory=list)
    init_errors: dict[str, str] = field(default_factory=dict)
    max_values: dict[str, float] = field(default_factory=dict)
    final_values: dict[str, float] = field(default_factory=dict)
    mean_values: dict[str, list[float]] = field(default_factory=dict)
    step_count: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metric_objects:
            self.metric_objects, self.init_errors = build_waymax_metric_objects()

    def update(self, state: Any) -> None:
        self.step_count += 1
        for name, metric in self.metric_objects:
            try:
                result = metric.compute(state)
                scalar = metric_result_to_sdc_scalar(result, state)
            except Exception as exc:
                # Keep the first error per metric and continue.  Route metrics can
                # legitimately fail if sdc_paths are missing from the dataset.
                self.errors.setdefault(name, str(exc))
                continue
            if scalar is None:
                continue
            self.final_values[name] = float(scalar)
            self.max_values[name] = max(self.max_values.get(name, float("-inf")), float(scalar))
            self.mean_values.setdefault(name, []).append(float(scalar))

    def finalize(self, *, include_errors: bool = True) -> dict[str, float | int | dict[str, str]]:
        out: dict[str, float | int | dict[str, str]] = {"MetricSteps": int(self.step_count)}

        collision = float(self.max_values.get("OverlapMetric", 0.0)) > 0.0
        offroad = float(self.max_values.get("OffroadMetric", 0.0)) > 0.0
        if "OverlapMetric" in self.max_values:
            out["CollisionRate"] = float(collision)
        if "OffroadMetric" in self.max_values:
            out["OffroadRate"] = float(offroad)
        if "OverlapMetric" in self.max_values or "OffroadMetric" in self.max_values:
            # Paper CR: collision OR road departure.
            out["CR"] = float(collision or offroad)

        if "ProgressionMetric" in self.final_values:
            # Waymax route progression is already route-normalized toward the final
            # logged SDC position when sdc_paths are available.
            out["EP"] = float(self.final_values["ProgressionMetric"])
        if "WrongWayMetric" in self.max_values:
            out["WrongWayRate"] = float(self.max_values["WrongWayMetric"] > 0.0)
        if "OffRouteMetric" in self.max_values:
            out["OffRouteRate"] = float(self.max_values["OffRouteMetric"] > 0.0)
        if "KinematicsInfeasibilityMetric" in self.max_values:
            out["KinematicsInfeasibilityRate"] = float(self.max_values["KinematicsInfeasibilityMetric"] > 0.0)
        if "LogDivergenceMetric" in self.mean_values and self.mean_values["LogDivergenceMetric"]:
            out["LogDivergence"] = float(np.mean(self.mean_values["LogDivergenceMetric"]))

        # Also expose canonical Waymax class-name aggregates for debugging.
        for name in sorted(self.final_values):
            out[f"WaymaxFinal/{name}"] = float(self.final_values[name])
        for name in sorted(self.max_values):
            if name in _EVENT_METRICS:
                out[f"WaymaxAny/{name}"] = float(self.max_values[name] > 0.0)
        for name, vals in sorted(self.mean_values.items()):
            if vals and name not in _EVENT_METRICS and name != "ProgressionMetric":
                out[f"WaymaxMean/{name}"] = float(np.mean(vals))

        all_errors = {**self.init_errors, **self.errors}
        if include_errors and all_errors:
            out["metric_errors"] = all_errors
        return out


def _numeric_metric_value(x: Any) -> float | None:
    """Best-effort scalar extraction from older/raw metric outputs."""
    for attr in ("value", "result", "metric_value"):
        if hasattr(x, attr):
            val = _numeric_metric_value(getattr(x, attr))
            if val is not None:
                return val
    if isinstance(x, dict):
        vals = [_numeric_metric_value(v) for v in x.values() if not isinstance(v, str)]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    try:
        arr = np.asarray(_device_get(x))
        if arr.size == 0:
            return None
        arr = arr.astype(float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        return float(np.mean(finite))
    except Exception:
        return None


def aggregate_waymax_standard_metrics(rollouts: list[dict]) -> dict[str, float]:
    """Aggregate per-episode Waymax metric summaries into JSON-ready scalars."""
    buckets: dict[str, list[float]] = {}
    for item in rollouts:
        metrics = item.get("standard_metrics", {}) or {}
        for name, value in metrics.items():
            if name == "metric_errors" or isinstance(value, dict):
                continue
            scalar = _numeric_metric_value(value)
            if scalar is not None:
                buckets.setdefault(name, []).append(scalar)
    return {name: float(np.mean(vals)) for name, vals in sorted(buckets.items()) if vals}
