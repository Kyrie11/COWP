"""V48.74 OC-SVBW: observation-consistent signed-viability barrier witness.

This module is intentionally independent of the training stack.  It computes two
nested, parameter-free diagnostics from the signed clearance trace of an
actuator-projected executable recovery against observation-only agent motion:

B1(t) = h(t) + tau(t) * h_dot(t)
B2(t) = h(t) + tau(t) * h_dot(t) + 0.5 * tau(t)^2 * h_ddot(t)

where tau is the remaining executable-recovery time.  A negative B1/B2 is a
finite-time viability debt.  The same signed expression covers Safe (h>0),
Near-Contact (h≈0), and Contact (h<0), without a regime identifier.

The feature adapter is strict and fail-closed.  It is used only by the V48.74
experimental overlay and leaves V48.73 behavior bitwise unchanged when the
V48.74 environment switch is absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence
import math
import os

import numpy as np

try:  # Torch is optional for unit-level NumPy use.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


V48_74_ENV = "OCRAP_V48_74_SIGNED_VIABILITY"
V48_74_SOURCE = "signed_finite_time_viability_projected_recovery_witness"
V48_74_ENGINEERING_VERSION = "v48.74.2-OC-SVBW-ENGFIX"
V48_74_SCHEMA = 10
V48_74_FEATURE_DIM = 22

_EPS = 1.0e-8


@dataclass(frozen=True)
class SignedViabilityDiagnostics:
    first_order_debt: np.ndarray
    second_order_debt: np.ndarray
    first_order_factor: np.ndarray
    second_order_factor: np.ndarray
    min_first_order_barrier: np.ndarray
    min_second_order_barrier: np.ndarray
    resolved_clearance_key: str | None = None
    resolved_time_key: str | None = None


def enabled() -> bool:
    value = os.getenv(V48_74_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _as_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _restore_like(array: np.ndarray, reference: Any) -> Any:
    if torch is not None and isinstance(reference, torch.Tensor):
        return torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    ref = np.asarray(reference)
    return np.asarray(array, dtype=ref.dtype)


def _finite_derivatives(values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized first/second derivatives on the final axis.

    ``times`` must be one-dimensional and strictly increasing.  Edge values use
    one-sided differences; interior values use the non-uniform three-point
    derivative supplied by NumPy.  This is deterministic and has no trainable or
    swept hyperparameter.
    """
    if values.shape[-1] != times.size:
        raise ValueError(
            f"time length mismatch: values.shape[-1]={values.shape[-1]} vs {times.size}"
        )
    if times.size < 3:
        raise ValueError("signed-viability witness requires at least three time points")
    dt = np.diff(times)
    if not np.all(np.isfinite(dt)) or not np.all(dt > 0.0):
        raise ValueError("time grid must be finite and strictly increasing")
    first = np.gradient(values, times, axis=-1, edge_order=2)
    second = np.gradient(first, times, axis=-1, edge_order=2)
    return first, second


def _default_scale(clearance: np.ndarray, scale: Any | None) -> np.ndarray:
    """Choose a physical, data-derived normalization without a tuned constant."""
    lead = clearance.shape[:-1]
    if scale is not None:
        arr = np.asarray(scale, dtype=np.float64)
        while arr.ndim < len(lead):
            arr = np.expand_dims(arr, axis=-1)
        try:
            arr = np.broadcast_to(arr, lead)
        except ValueError as exc:
            raise ValueError(f"cannot broadcast clearance scale {arr.shape} to {lead}") from exc
        return np.maximum(np.abs(arr), _EPS)

    # The trajectory's own robust magnitude is used only for unit normalization;
    # it cannot alter the sign/set of the certificate.  A median is less sensitive
    # to isolated far-away agents than a max and contains no learned parameter.
    robust = np.nanmedian(np.abs(clearance), axis=-1)
    finite = np.isfinite(robust)
    fallback = np.ones_like(robust, dtype=np.float64)
    robust = np.where(finite, robust, fallback)
    return np.maximum(robust, _EPS)


def signed_viability_diagnostics(
    signed_clearance: Any,
    times_s: Any,
    *,
    valid_mask: Any | None = None,
    clearance_scale: Any | None = None,
    reduce_axes: Sequence[int] | None = None,
) -> SignedViabilityDiagnostics:
    """Compute nested first/second-order finite-time viability diagnostics.

    Parameters
    ----------
    signed_clearance:
        Array with time on the last axis.  Positive means separated, zero is the
        contact boundary, and negative means overlap/contact.
    times_s:
        Strictly increasing one-dimensional time grid.
    valid_mask:
        Optional boolean array broadcastable to ``signed_clearance``.
    clearance_scale:
        Optional physical length scale.  If omitted, a deterministic robust
        trajectory magnitude is used.
    reduce_axes:
        Axes (excluding time) over which a common worst witness is required.
        If omitted, no non-time axis is reduced.
    """
    h = np.asarray(_as_numpy(signed_clearance), dtype=np.float64)
    t = np.asarray(_as_numpy(times_s), dtype=np.float64).reshape(-1)
    if h.ndim < 1 or h.shape[-1] != t.size:
        raise ValueError(f"invalid signed-clearance shape {h.shape} for time grid {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("time grid contains non-finite values")

    h_dot, h_ddot = _finite_derivatives(h, t)
    dt_floor = float(np.min(np.diff(t)))
    tau = np.maximum(float(t[-1]) - t, dt_floor)
    tau_shape = (1,) * (h.ndim - 1) + (t.size,)
    tau_b = tau.reshape(tau_shape)

    b1 = h + tau_b * h_dot
    b2 = b1 + 0.5 * np.square(tau_b) * h_ddot

    valid = np.isfinite(h) & np.isfinite(b1) & np.isfinite(b2)
    if valid_mask is not None:
        valid &= np.broadcast_to(np.asarray(_as_numpy(valid_mask), dtype=bool), h.shape)

    # Invalid observations are neutral for a min-barrier reduction.  A trajectory
    # with no valid time point fails closed below.
    inf = np.array(np.inf, dtype=np.float64)
    b1_masked = np.where(valid, b1, inf)
    b2_masked = np.where(valid, b2, inf)
    min_b1 = np.min(b1_masked, axis=-1)
    min_b2 = np.min(b2_masked, axis=-1)
    any_valid = np.any(valid, axis=-1)
    min_b1 = np.where(any_valid, min_b1, -np.inf)
    min_b2 = np.where(any_valid, min_b2, -np.inf)

    scale = _default_scale(h, clearance_scale)
    debt1 = np.maximum(-min_b1, 0.0) / scale
    debt2_raw = np.maximum(-min_b2, 0.0) / scale
    # The second-order arm is a non-compensatory high-order barrier: acceleration
    # evidence may not erase a failed first-order finite-time condition.
    debt2 = np.maximum(debt1, debt2_raw)

    if reduce_axes is not None:
        axes = tuple(sorted({int(a) for a in reduce_axes}))
        if any(a < 0 or a >= debt1.ndim for a in axes):
            raise ValueError(f"invalid reduce_axes={axes} for debt shape {debt1.shape}")
        if axes:
            debt1 = np.max(debt1, axis=axes)
            debt2 = np.max(debt2, axis=axes)
            min_b1 = np.min(min_b1, axis=axes)
            min_b2 = np.min(min_b2, axis=axes)

    factor1 = 1.0 / (1.0 + debt1)
    factor2 = 1.0 / (1.0 + debt2)
    return SignedViabilityDiagnostics(
        first_order_debt=debt1,
        second_order_debt=debt2,
        first_order_factor=factor1,
        second_order_factor=factor2,
        min_first_order_barrier=min_b1,
        min_second_order_barrier=min_b2,
    )


_CLEARANCE_KEYS = (
    "projected_recovery_signed_clearance",
    "signed_projected_recovery_clearance",
    "projected_signed_clearance",
    "signed_clearance_alt",
    "signed_clearance_cv",
    "pairwise_signed_clearance",
    "clearance_alt",
    "clearance_cv",
    "signed_clearance",
    "clearance",
)
_TIME_KEYS = (
    "future_times_s",
    "future_time_s",
    "recovery_times_s",
    "prefix_times_s",
    "times_s",
    "time_s",
    "t_s",
)
_VALID_KEYS = (
    "projected_recovery_clearance_valid",
    "clearance_valid_mask",
    "future_valid_mask",
    "valid_mask",
)
_SCALE_KEYS = (
    "combined_radius",
    "pair_radius",
    "clearance_scale",
    "safety_radius",
)


def _candidate_mapping(context: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in context.items():
        if key.startswith("__"):
            continue
        out[str(key)] = value
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                out.setdefault(str(child_key), child_value)
    return out


def _pick_named_array(
    values: Mapping[str, Any], keys: Sequence[str], *, min_ndim: int = 1
) -> tuple[str | None, np.ndarray | None]:
    lower = {k.lower(): k for k in values}
    for alias in keys:
        original = lower.get(alias.lower())
        if original is None:
            continue
        try:
            arr = _as_numpy(values[original])
        except Exception:
            continue
        if arr.ndim >= min_ndim and arr.size:
            return original, arr
    # Strict semantic fallback: name must explicitly contain both clearance and
    # trajectory/path/future semantics, so a scalar aggregate is never selected.
    ranked: list[tuple[int, str, np.ndarray]] = []
    for name, value in values.items():
        lname = name.lower()
        if "clearance" not in lname:
            continue
        try:
            arr = _as_numpy(value)
        except Exception:
            continue
        if arr.ndim < min_ndim or arr.size < 3:
            continue
        score = sum(token in lname for token in ("projected", "recovery", "pair", "future", "signed", "alt"))
        ranked.append((score, name, arr))
    if ranked:
        ranked.sort(key=lambda x: (-x[0], x[1]))
        _, name, arr = ranked[0]
        return name, arr
    return None, None


def _pick_time(values: Mapping[str, Any], length: int) -> tuple[str | None, np.ndarray | None]:
    lower = {k.lower(): k for k in values}
    for alias in _TIME_KEYS:
        original = lower.get(alias.lower())
        if original is None:
            continue
        try:
            arr = _as_numpy(values[original]).astype(np.float64, copy=False).reshape(-1)
        except Exception:
            continue
        if arr.size == length:
            return original, arr
    for name, value in values.items():
        lname = name.lower()
        if "time" not in lname and lname not in {"t", "ts"}:
            continue
        try:
            arr = _as_numpy(value).astype(np.float64, copy=False).reshape(-1)
        except Exception:
            continue
        if arr.size == length:
            return name, arr
    # Last-resort dt is accepted only if explicitly named and finite.
    for name, value in values.items():
        lname = name.lower()
        if lname not in {"dt", "dt_s", "time_step_s", "future_dt_s"}:
            continue
        try:
            dt = float(np.asarray(value).reshape(()))
        except Exception:
            continue
        if math.isfinite(dt) and dt > 0.0:
            return name, np.arange(length, dtype=np.float64) * dt
    return None, None


def _find_feature_leaf(value: Any, path: tuple[Any, ...] = ()) -> tuple[tuple[Any, ...], Any] | None:
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim >= 1 and value.shape[-1] == V48_74_FEATURE_DIM:
            return path, value
        return None
    if isinstance(value, np.ndarray):
        if value.ndim >= 1 and value.shape[-1] == V48_74_FEATURE_DIM:
            return path, value
        return None
    if isinstance(value, tuple):
        for i, child in enumerate(value):
            hit = _find_feature_leaf(child, path + (i,))
            if hit is not None:
                return hit
    elif isinstance(value, list):
        for i, child in enumerate(value):
            hit = _find_feature_leaf(child, path + (i,))
            if hit is not None:
                return hit
    elif isinstance(value, Mapping):
        for key, child in value.items():
            hit = _find_feature_leaf(child, path + (key,))
            if hit is not None:
                return hit
    return None


def _replace_path(value: Any, path: tuple[Any, ...], replacement: Any) -> Any:
    if not path:
        return replacement
    head, *tail = path
    tail_t = tuple(tail)
    if isinstance(value, tuple):
        items = list(value)
        items[int(head)] = _replace_path(items[int(head)], tail_t, replacement)
        return tuple(items)
    if isinstance(value, list):
        items = list(value)
        items[int(head)] = _replace_path(items[int(head)], tail_t, replacement)
        return items
    if isinstance(value, Mapping):
        items: MutableMapping[Any, Any] = dict(value)
        items[head] = _replace_path(items[head], tail_t, replacement)
        try:
            return type(value)(items)
        except Exception:
            return items
    raise TypeError(f"cannot replace nested path through {type(value)!r}")


def _align_to_feature_leading(array: np.ndarray, leading: tuple[int, ...]) -> np.ndarray:
    """Reduce only non-feature axes by a common worst-case maximum."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.shape == leading:
        return arr
    if not leading:
        return np.asarray(np.max(arr), dtype=np.float64)
    # Prefer exact leading prefix/suffix alignment.
    if arr.ndim >= len(leading) and tuple(arr.shape[: len(leading)]) == leading:
        axes = tuple(range(len(leading), arr.ndim))
        return np.max(arr, axis=axes) if axes else arr
    if arr.ndim >= len(leading) and tuple(arr.shape[-len(leading) :]) == leading:
        axes = tuple(range(0, arr.ndim - len(leading)))
        return np.max(arr, axis=axes) if axes else arr
    # Broadcast scalar; otherwise fail closed instead of silently mixing axes.
    if arr.size == 1:
        return np.broadcast_to(arr.reshape(()), leading)
    raise ValueError(f"cannot align viability debt shape {arr.shape} to feature leading shape {leading}")


def apply_v48_74_feature_overlay(result: Any, context: Mapping[str, Any]) -> Any:
    """Replace feature coordinates 20/21 in a V48.73-compatible result.

    The function is a no-op unless ``OCRAP_V48_74_SIGNED_VIABILITY=1``.  It is
    deliberately fail-closed when enabled: missing trajectory clearance or time
    information raises an actionable error rather than falling back to the old
    N/O diagnostics.
    """
    if not enabled():
        return result
    hit = _find_feature_leaf(result)
    if hit is None:
        return result
    path, feature_ref = hit
    feature_np = _as_numpy(feature_ref).copy()
    leading = tuple(int(x) for x in feature_np.shape[:-1])

    values = _candidate_mapping(context)
    clearance_key, clearance = _pick_named_array(values, _CLEARANCE_KEYS, min_ndim=1)
    if clearance is None or clearance_key is None:
        raise RuntimeError(
            "V48.74 could not resolve a time-resolved signed-clearance tensor; "
            "the overlay refuses to use an aggregate or stale V48.73 feature"
        )
    clearance = np.asarray(clearance, dtype=np.float64)
    if clearance.shape[-1] < 3:
        raise RuntimeError(f"resolved clearance '{clearance_key}' is not time-resolved: {clearance.shape}")
    time_key, times = _pick_time(values, clearance.shape[-1])
    if times is None:
        raise RuntimeError(
            f"V48.74 resolved clearance '{clearance_key}' but no matching time grid of "
            f"length {clearance.shape[-1]}"
        )

    valid = None
    for key in _VALID_KEYS:
        if key in values:
            valid = values[key]
            break
    scale = None
    for key in _SCALE_KEYS:
        if key in values:
            scale = values[key]
            break

    diag = signed_viability_diagnostics(
        clearance,
        times,
        valid_mask=valid,
        clearance_scale=scale,
    )
    d1 = _align_to_feature_leading(diag.first_order_debt, leading)
    d2 = _align_to_feature_leading(diag.second_order_debt, leading)
    feature_np[..., 20] = d1
    feature_np[..., 21] = d2
    replacement = _restore_like(feature_np, feature_ref)
    return _replace_path(result, path, replacement)


def runtime_contract_fragment() -> dict[str, Any]:
    return {
        "enabled": enabled(),
        "engineering_version": V48_74_ENGINEERING_VERSION,
        "schema": V48_74_SCHEMA,
        "feature_dim": V48_74_FEATURE_DIM,
        "source": V48_74_SOURCE,
        "coordinates": {
            "20": "first_order_signed_finite_time_viability_debt",
            "21": "second_order_signed_finite_time_viability_debt",
        },
        "regime_conditioned": False,
        "uses_privileged_future": False,
        "uses_test_roots": False,
    }
