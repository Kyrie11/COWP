from __future__ import annotations

"""V48.81 switch-aware inverse structural truth contract.

V48.80 propagated valid but deliberately loose one-sided bounds whenever a
root-option cell *could* be touched by a structural floor/cap.  This module
uses the exact monotone operator order implemented by ``teacher_margin`` and
inverts the observed stored cell value.  If a floor/cap is inactive, the
pre-structural physical value is point-identified; if it is active, only the
mathematically valid one-sided preimage is retained.  Hidden/artifact hard
replacements remain unidentifiable.

No teacher metadata is exposed to the model.  The result is a training-only
interval sidecar and does not modify the dataset or teacher labels.
"""

from dataclasses import asdict, dataclass
from typing import Any
import numpy as np

from ocrap.algorithms.ocmero import oc_mero
from ocrap.v48_79_truth_contract import (
    STRUCT_HIDDEN_BRANCH,
    STRUCT_RECOVERY_FLOOR,
    STRUCT_ROUTE_OVERRIDE,
    STRUCT_SECONDARY_FLOOR,
    nested_tail_truth_contract,
    structural_root_option_reason_bits,
)

_BOUND = 1.0e6
_RECOVERY_FLOOR = 0.6
_ROUTE_CAP = -0.8
_SECONDARY_FLOOR = 0.9
_TOL = 1.0e-6


@dataclass(frozen=True)
class SwitchInverseTruthContractRecord:
    valid: bool
    physical_lower: float
    physical_upper: float
    exact_physical: bool
    informative: bool
    interval_width: float | None
    lower_finite: bool
    upper_finite: bool
    structural_exposure_mass: float
    r_dep_stored: float
    r_dep_recomputed: float
    r_dep_abs_error: float
    alpha: float
    beta: float
    top_m: int
    exact_cell_fraction: float
    inactive_structural_cell_fraction: float
    mixed_structural_cell_fraction: float
    incomplete_profile_cell_fraction: float
    inverse_contradiction_cell_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)




def structural_root_option_reason_profile(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(any_bits, all_bits, complete)`` over futures in each root.

    ``m_star`` is an intra-root aggregate.  Switch inversion is valid only when
    the structural operator profile is known for *every* future contributing to
    a root-option cell.  Older/compact dataset sidecars can contain fewer
    metadata entries than ``root_assignments``; treating the observed subset as
    the complete root was the remaining V48.81.1 engineering bug.

    ``complete[k,l]`` is therefore true only when every assigned future has a
    dictionary metadata record and the option mode itself is available.  An
    incomplete profile is fail-closed as physically unidentifiable.
    """
    m = np.asarray(sample.get("m_star", np.zeros((0, 0))), dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("m_star must be [K,L]")
    K, L = m.shape
    assignments = np.asarray(sample.get("root_assignments", []), dtype=np.int64).reshape(-1)
    from ocrap.v48_79_truth_contract import _json_scalar, _string_list, _HIDDEN_VALUES, _FLOORED_MODES
    metas = _json_scalar(sample.get("future_metadata"), [])
    if not isinstance(metas, list):
        metas = []
    modes_raw = _string_list(sample.get("recovery_modes", []))
    modes = modes_raw[:L] + [""] * max(0, L - len(modes_raw))

    any_bits = np.zeros((K, L), dtype=np.int64)
    all_bits = np.zeros((K, L), dtype=np.int64)
    complete = np.zeros((K, L), dtype=bool)

    for k in range(K):
        fis = [fi for fi in range(assignments.size) if int(assignments[fi]) == k]
        if not fis:
            continue
        root_meta_complete = all(fi < len(metas) and isinstance(metas[fi], dict) for fi in fis)
        for l in range(L):
            option_mode_complete = l < len(modes_raw) and bool(str(modes_raw[l]).strip())
            complete[k, l] = bool(root_meta_complete and option_mode_complete)
            vals: list[int] = []
            mode = modes[l]
            # Use all assigned futures when the profile is complete.  If it is
            # incomplete, compute ANY from the available subset only for audit
            # compatibility with V48.79, but never authorize inversion.
            for fi in fis:
                if fi >= len(metas) or not isinstance(metas[fi], dict):
                    continue
                meta = metas[fi]
                hidden = meta.get("artifact_branch") in _HIDDEN_VALUES or meta.get("hidden_intent") in _HIDDEN_VALUES
                b = 0
                if hidden:
                    b |= STRUCT_HIDDEN_BRANCH
                if (not hidden) and mode in _FLOORED_MODES:
                    b |= STRUCT_RECOVERY_FLOOR
                if bool(meta.get("route_blocked", False)) and mode == "yield_rejoin":
                    b |= STRUCT_ROUTE_OVERRIDE
                if bool(meta.get("secondary_threat", False)) and mode == "avoid_secondary":
                    b |= STRUCT_SECONDARY_FLOOR
                vals.append(int(b))
            if not vals:
                continue
            a = 0
            for b in vals:
                a |= b
            c = vals[0]
            for b in vals[1:]:
                c &= b
            any_bits[k, l] = a
            all_bits[k, l] = c
    return any_bits, all_bits, complete

def _inverse_floor(lo: float, hi: float, c: float) -> tuple[float, float]:
    """Preimage of [lo,hi] through ``z=max(x,c)``.

    Empty preimages are semantic contradictions, not process-level exceptions.
    ``_cell_preimage`` catches them and fail-closes the affected cell.
    """
    if hi < c - _TOL:
        raise ArithmeticError(f"floor inverse empty: [{lo},{hi}] below {c}")
    if lo > c + _TOL:
        return lo, hi
    return -_BOUND, hi


def _inverse_cap(lo: float, hi: float, c: float) -> tuple[float, float]:
    """Preimage of [lo,hi] through ``z=min(x,c)``."""
    if lo > c + _TOL:
        raise ArithmeticError(f"cap inverse empty: [{lo},{hi}] above {c}")
    if hi < c - _TOL:
        return lo, hi
    return lo, _BOUND


def _cell_preimage(
    y: float, any_bits: int, all_bits: int, profile_complete: bool
) -> tuple[float, float, bool, bool, bool, bool]:
    """Invert one root-option cell conservatively.

    Returns ``(lo, hi, exact, mixed, incomplete, contradiction)``.  The last
    three states are all fail-closed to an unbounded physical interval.  This
    is essential because root-level ``m_star`` is an aggregate: a structural
    bit derived from partial metadata is not proof that a single monotone
    operator acted uniformly on every contributing future.
    """
    mixed = bool(any_bits != all_bits)
    incomplete = not bool(profile_complete)
    if (any_bits & STRUCT_HIDDEN_BRANCH) or mixed or incomplete:
        return -_BOUND, _BOUND, False, mixed, incomplete, False
    bits = int(all_bits)
    lo = hi = float(y)
    try:
        # Reverse teacher_margin's exact forward order:
        # recovery floor -> route cap -> secondary floor.
        if bits & STRUCT_SECONDARY_FLOOR:
            lo, hi = _inverse_floor(lo, hi, _SECONDARY_FLOOR)
        if bits & STRUCT_ROUTE_OVERRIDE:
            lo, hi = _inverse_cap(lo, hi, _ROUTE_CAP)
        if bits & STRUCT_RECOVERY_FLOOR:
            lo, hi = _inverse_floor(lo, hi, _RECOVERY_FLOOR)
    except ArithmeticError:
        # A contradiction such as uniform floor exposure with aggregated
        # m_star=0.5 proves that the stored root aggregate plus sidecar metadata
        # are insufficient for switch inversion.  Do not abort the whole index
        # build and do not fabricate a bound: mark this cell unidentifiable.
        return -_BOUND, _BOUND, False, False, False, True
    exact = (lo > -0.5 * _BOUND) and (hi < 0.5 * _BOUND) and abs(hi - lo) <= 1e-10
    return float(lo), float(hi), bool(exact), False, False, False

def nested_tail_switch_inverse_interval(
    sample: dict[str, Any], *, alpha: float = 0.2, beta: float = 0.2,
    top_m: int = 8, recompute_tolerance: float = 1.0e-5,
) -> SwitchInverseTruthContractRecord:
    base = nested_tail_truth_contract(
        sample, alpha=alpha, beta=beta, top_m=top_m,
        recompute_tolerance=recompute_tolerance,
    )
    M = np.asarray(sample.get("m_star"), dtype=np.float64)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("m_star must be a non-empty [K,L] matrix")
    K, L = M.shape
    bits = structural_root_option_reason_bits(sample)
    any_bits, all_bits, profile_complete = structural_root_option_reason_profile(sample)
    if bits.shape != M.shape or any_bits.shape != M.shape or all_bits.shape != M.shape or profile_complete.shape != M.shape:
        raise ValueError("structural bitmask shape mismatch")
    if not np.array_equal(bits, any_bits):
        raise ValueError("V48.79 ANY structural exposure profile mismatch")

    lower = np.empty_like(M)
    upper = np.empty_like(M)
    exact_cells = 0
    structural_cells = 0
    inactive_structural_cells = 0
    mixed_structural_cells = 0
    incomplete_profile_cells = 0
    inverse_contradiction_cells = 0
    for k in range(K):
        for l in range(L):
            b = int(bits[k, l])
            lo, hi, exact, mixed, incomplete, contradiction = _cell_preimage(
                float(M[k, l]), int(any_bits[k,l]), int(all_bits[k,l]), bool(profile_complete[k,l])
            )
            lower[k, l], upper[k, l] = lo, hi
            exact_cells += int(exact)
            if b:
                structural_cells += 1
                inactive_structural_cells += int(exact)
                mixed_structural_cells += int(mixed)
            incomplete_profile_cells += int(incomplete)
            inverse_contradiction_cells += int(contradiction)

    p = np.asarray(sample.get("root_probs", np.zeros(K)), dtype=np.float64).reshape(-1)[:K]
    rv = np.asarray(sample.get("root_valid", np.ones(K)), dtype=bool).reshape(-1)[:K]
    C = np.asarray(sample.get("c_star", np.eye(K)), dtype=np.float64)
    ov = np.asarray(sample.get("option_valid", np.ones(L)), dtype=bool).reshape(-1)[:L]
    kwargs = dict(alpha=float(alpha), beta=float(beta), option_valid=ov,
                  root_valid=rv, use_lcvar=True, use_obs_kernel=True, top_m=int(top_m))
    lo = float(oc_mero(lower, p, C, **kwargs).r_dep)
    hi = float(oc_mero(upper, p, C, **kwargs).r_dep)
    if lo > hi + 1e-6:
        raise ValueError(f"invalid propagated interval: lower={lo} upper={hi}")
    lower_finite = lo > -0.5 * _BOUND
    upper_finite = hi < +0.5 * _BOUND
    exact = bool(lower_finite and upper_finite and abs(hi - lo) <= 1e-10)
    informative = bool(lower_finite or upper_finite)
    width = float(hi - lo) if lower_finite and upper_finite else None
    return SwitchInverseTruthContractRecord(
        valid=bool(base.valid), physical_lower=lo, physical_upper=hi,
        exact_physical=exact, informative=informative, interval_width=width,
        lower_finite=lower_finite, upper_finite=upper_finite,
        structural_exposure_mass=float(base.structural_exposure_mass),
        r_dep_stored=float(base.r_dep_stored), r_dep_recomputed=float(base.r_dep_recomputed),
        r_dep_abs_error=float(base.r_dep_abs_error), alpha=float(alpha), beta=float(beta), top_m=int(top_m),
        exact_cell_fraction=float(exact_cells / max(1, K * L)),
        inactive_structural_cell_fraction=float(inactive_structural_cells / max(1, structural_cells)),
        mixed_structural_cell_fraction=float(mixed_structural_cells / max(1, structural_cells)),
        incomplete_profile_cell_fraction=float(incomplete_profile_cells / max(1, K * L)),
        inverse_contradiction_cell_fraction=float(inverse_contradiction_cells / max(1, K * L)),
    )
