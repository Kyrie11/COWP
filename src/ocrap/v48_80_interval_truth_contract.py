from __future__ import annotations

"""V48.80 partial-identification truth contract.

The V48.79 hard physical-identifiable mask is deliberately conservative and
leaves only a few percent of candidates supervised.  V48.80 does not rewrite
teacher labels and does not expose structural metadata to the model.  Instead,
it converts known structural teacher operations into conservative cell-wise
bounds on the latent *physical* root-option margin, then propagates those bounds
through the monotone OC-MERO operator.

For a structural floor ``y=max(x,c)``, the pre-floor physical value satisfies
``x<=y``.  For a structural cap/negative override ``y=min(x,c)``, ``x>=y``.
Hidden/artifact branch overrides are not invertible from the stored sample and
therefore give no finite physical bound.  Because root aggregation and OC-MERO
are monotone, propagating these cellwise bounds yields a valid candidate-level
interval for physical R_dep without reconstructing the dataset.
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


@dataclass(frozen=True)
class IntervalTruthContractRecord:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def nested_tail_physical_interval(
    sample: dict[str, Any], *, alpha: float = 0.2, beta: float = 0.2,
    top_m: int = 8, recompute_tolerance: float = 1.0e-5,
) -> IntervalTruthContractRecord:
    base = nested_tail_truth_contract(
        sample, alpha=alpha, beta=beta, top_m=top_m,
        recompute_tolerance=recompute_tolerance,
    )
    M = np.asarray(sample.get("m_star"), dtype=np.float64)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("m_star must be a non-empty [K,L] matrix")
    K, L = M.shape
    bits = structural_root_option_reason_bits(sample)
    if bits.shape != M.shape:
        raise ValueError("structural bitmask shape mismatch")

    lower = M.copy()
    upper = M.copy()
    for k in range(K):
        for l in range(L):
            b = int(bits[k, l])
            if b == 0:
                continue
            # Hidden/artifact semantics are branch-specific hard replacements;
            # stored root-level margins do not identify the pre-override value.
            if b & STRUCT_HIDDEN_BRANCH:
                lower[k, l] = -_BOUND
                upper[k, l] = +_BOUND
                continue
            has_floor = bool(b & (STRUCT_RECOVERY_FLOOR | STRUCT_SECONDARY_FLOOR))
            has_cap = bool(b & STRUCT_ROUTE_OVERRIDE)
            if has_floor and has_cap:
                # Composition order can destroy a one-sided inverse bound.
                lower[k, l] = -_BOUND
                upper[k, l] = +_BOUND
            elif has_floor:
                # max(x,c) >= x
                lower[k, l] = -_BOUND
                upper[k, l] = M[k, l]
            elif has_cap:
                # min(x,c) <= x
                lower[k, l] = M[k, l]
                upper[k, l] = +_BOUND

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
    return IntervalTruthContractRecord(
        valid=bool(base.valid), physical_lower=lo, physical_upper=hi,
        exact_physical=exact, informative=informative, interval_width=width,
        lower_finite=lower_finite, upper_finite=upper_finite,
        structural_exposure_mass=float(base.structural_exposure_mass),
        r_dep_stored=float(base.r_dep_stored), r_dep_recomputed=float(base.r_dep_recomputed),
        r_dep_abs_error=float(base.r_dep_abs_error), alpha=float(alpha), beta=float(beta), top_m=int(top_m),
    )
