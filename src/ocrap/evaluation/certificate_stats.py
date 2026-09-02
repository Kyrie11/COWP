"""Statistical support checks shared by certificate tools and tests.

The Natural gate uses a lower confidence bound (LCB) for precision and upper
confidence bounds (UCBs) for harmful exposure.  These are directional claims,
so the default is a *one-sided* 90% Wilson bound.  Historically v48.18 used
``z=1.64485`` while labelling the fields LCB90/UCB90; that value is appropriate
for a central two-sided 90% interval (or a one-sided 95% bound), and therefore
made the gate materially more conservative than its declared protocol.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Literal

BoundType = Literal["one_sided", "two_sided"]


def wilson_z(*, confidence_level: float = 0.90, bound_type: BoundType = "one_sided") -> float:
    """Return the normal critical value for the declared Wilson bound.

    ``one_sided`` means each reported lower/upper bound has the requested
    directional coverage. ``two_sided`` means a central interval with the
    requested simultaneous coverage.
    """
    confidence = float(confidence_level)
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence_level must be in (0.5, 1.0), got {confidence!r}")
    if bound_type == "one_sided":
        quantile = confidence
    elif bound_type == "two_sided":
        quantile = 0.5 + confidence / 2.0
    else:  # pragma: no cover - protected by type/CLI validation
        raise ValueError(f"unsupported bound_type: {bound_type!r}")
    return float(NormalDist().inv_cdf(quantile))


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float = 0.90,
    bound_type: BoundType = "one_sided",
    z: float | None = None,
) -> tuple[float, float]:
    """Return Wilson lower/upper bounds for a binomial proportion.

    Passing ``z`` is retained for audits of historical protocols. New code
    should declare ``confidence_level`` and ``bound_type`` explicitly.
    """
    n = int(total)
    if n <= 0:
        return 0.0, 1.0
    x = min(max(int(successes), 0), n)
    p = x / n
    critical = float(z) if z is not None else wilson_z(
        confidence_level=confidence_level, bound_type=bound_type,
    )
    z2 = critical**2
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = critical * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def certificate_support_feasibility(
    *,
    num_groups: int,
    num_opportunities: int,
    min_selected: int,
    min_precision_lcb: float,
    max_harmful_selected_ucb: float,
    max_harmful_group_ucb: float,
    confidence_level: float = 0.90,
    bound_type: BoundType = "one_sided",
    z: float | None = None,
) -> dict[str, Any]:
    """Audit whether a certificate can pass even under an ideal ranking.

    Every available opportunity is assumed to be selected before any
    non-opportunity, and no selected/group harm is assumed. Failure under this
    optimistic bound means the specification is unsupported by the observed
    split, independent of model scores or thresholds.
    """
    n_groups = max(0, int(num_groups))
    n_opp = max(0, min(int(num_opportunities), n_groups))
    min_sel = max(0, int(min_selected))
    critical = float(z) if z is not None else wilson_z(
        confidence_level=confidence_level, bound_type=bound_type,
    )
    candidates: list[dict[str, float | int | bool]] = []
    for selected in range(min_sel, n_groups + 1):
        positives = min(n_opp, selected)
        precision_lcb, _ = wilson_interval(positives, selected, z=critical)
        _, harmful_selected_ucb = wilson_interval(0, selected, z=critical)
        _, harmful_group_ucb = wilson_interval(0, n_groups, z=critical)
        feasible = bool(
            precision_lcb + 1e-12 >= float(min_precision_lcb)
            and harmful_selected_ucb <= float(max_harmful_selected_ucb) + 1e-12
            and harmful_group_ucb <= float(max_harmful_group_ucb) + 1e-12
        )
        candidates.append({
            "selected": selected,
            "optimistic_positive": positives,
            "optimistic_precision_lcb": precision_lcb,
            "zero_harm_selected_ucb": harmful_selected_ucb,
            "zero_harm_group_ucb": harmful_group_ucb,
            "feasible": feasible,
        })
    feasible_rows = [row for row in candidates if bool(row["feasible"])]
    best_precision = max((float(row["optimistic_precision_lcb"]) for row in candidates), default=0.0)
    min_harm_selected = min((float(row["zero_harm_selected_ucb"]) for row in candidates), default=1.0)
    group_harm_ucb = wilson_interval(0, n_groups, z=critical)[1]
    return {
        "feasible": bool(feasible_rows),
        "num_groups": n_groups,
        "num_opportunities": n_opp,
        "min_selected": min_sel,
        "min_precision_lcb": float(min_precision_lcb),
        "max_harmful_selected_ucb": float(max_harmful_selected_ucb),
        "max_harmful_group_ucb": float(max_harmful_group_ucb),
        "confidence_level": float(confidence_level),
        "bound_type": str(bound_type),
        "wilson_z": critical,
        "optimistic_best_precision_lcb": best_precision,
        "optimistic_min_harmful_selected_ucb": min_harm_selected,
        "optimistic_zero_harmful_group_ucb": group_harm_ucb,
        "first_feasible_selected": int(feasible_rows[0]["selected"]) if feasible_rows else None,
        "reason": None if feasible_rows else "certificate constraints are infeasible under optimistic labels",
    }
