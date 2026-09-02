from __future__ import annotations

"""V48.79 physical-vs-structural truth-contract utilities.

The stored teacher root-option margin ``m_star`` mixes a physical signed margin
with a small number of structural teacher interventions (recovery-mode floors,
route overrides, secondary-recovery floors and deliberately mined hidden-branch
semantics).  V48.79 does *not* reconstruct the dataset and does not rewrite any
teacher label.  Instead, it identifies a conservative subset of candidates for
which the *nested teacher OC-MERO lower tail* is guaranteed not to traverse any
root-option cell that could have been modified by those structural rules.

That subset is used only as a supervision/adjudication mask.  The deployed model
never receives this mask or any future/teacher metadata.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any

import numpy as np

from ocrap.algorithms.lcv import normalize_weights, weighted_lcvar
from ocrap.algorithms.ocmero import sparsify_compatibility


# Bit flags describing *potential* structural intervention on a future-option
# cell.  They intentionally match the current teacher implementation rather
# than paper prose.
STRUCT_RECOVERY_FLOOR = 1 << 0
STRUCT_ROUTE_OVERRIDE = 1 << 1
STRUCT_SECONDARY_FLOOR = 1 << 2
STRUCT_HIDDEN_BRANCH = 1 << 3

REASON_NAMES = {
    STRUCT_RECOVERY_FLOOR: "recovery_mode_floor_0p6",
    STRUCT_ROUTE_OVERRIDE: "route_override_neg_0p8",
    STRUCT_SECONDARY_FLOOR: "secondary_floor_0p9",
    STRUCT_HIDDEN_BRANCH: "hidden_or_artifact_branch_semantics",
}

_FLOORED_MODES = frozenset({"post_contact_stabilize", "yield_rejoin", "pull_over"})
_HIDDEN_VALUES = frozenset({"yield", "accelerate"})


@dataclass(frozen=True)
class TailTruthContractRecord:
    valid: bool
    physical_identifiable: bool
    structural_exposure_mass: float
    structural_reason_mass: dict[str, float]
    active_nested_tail_cells: int
    structurally_exposed_tail_cells: int
    r_dep_stored: float
    r_dep_recomputed: float
    r_dep_abs_error: float
    alpha: float
    beta: float
    top_m: int
    conservative_root_aggregation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_scalar(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        arr = np.asarray(value)
        if arr.ndim == 0:
            value = arr.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value


def _string_list(value: Any) -> list[str]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    out: list[str] = []
    for item in arr.reshape(-1):
        if isinstance(item, bytes):
            item = item.decode("utf-8", errors="ignore")
        out.append(str(item))
    return out


def weighted_lcvar_influence_np(scores: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    """Exact stable-sort/fractional-tail LCVAR subgradient for one vector."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise ValueError("LCVAR influence requires non-empty scores")
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in (0,1], got {alpha}")
    w = normalize_weights(np.asarray(weights, dtype=np.float64).reshape(-1))
    if w.size != s.size:
        raise ValueError("scores/weights mismatch")
    order = np.argsort(s, kind="mergesort")
    remaining = float(alpha)
    out = np.zeros_like(s, dtype=np.float64)
    for idx in order:
        take = min(float(w[idx]), remaining)
        if take > 0.0:
            out[idx] = take / float(alpha)
            remaining -= take
        if remaining <= 1e-12:
            break
    if remaining > 1e-8:
        # Mirrors weighted_lcvar's degenerate fallback.  With normalized weights
        # and alpha<=1 this should not normally be reached.
        out[order[-1]] += remaining / float(alpha)
    return out


def structural_root_option_reason_bits(sample: dict[str, Any]) -> np.ndarray:
    """Return conservative [K,L] bitmasks for teacher-structural exposure.

    We only have stored root-level margins, not the pre-override future-option
    margins.  Therefore a root-option cell is marked exposed if *any* future
    assigned to that root could invoke a structural teacher rule for that
    option.  This can exclude extra rows but cannot falsely call an exposed cell
    physically identifiable, which is the required fail-closed direction.
    """
    m = np.asarray(sample.get("m_star", np.zeros((0, 0))), dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("m_star must be [K,L]")
    K, L = m.shape
    assignments = np.asarray(sample.get("root_assignments", []), dtype=np.int64).reshape(-1)
    metas = _json_scalar(sample.get("future_metadata"), [])
    if not isinstance(metas, list):
        metas = []
    modes = _string_list(sample.get("recovery_modes", []))
    if len(modes) < L:
        modes = modes + [""] * (L - len(modes))
    bits = np.zeros((K, L), dtype=np.int64)
    n = min(len(metas), assignments.size)
    for fi in range(n):
        k = int(assignments[fi])
        if k < 0 or k >= K:
            continue
        meta = metas[fi] if isinstance(metas[fi], dict) else {}
        branch = meta.get("artifact_branch")
        hidden_intent = meta.get("hidden_intent")
        hidden = branch in _HIDDEN_VALUES or hidden_intent in _HIDDEN_VALUES
        route_blocked = bool(meta.get("route_blocked", False))
        secondary = bool(meta.get("secondary_threat", False))
        for l in range(L):
            mode = modes[l]
            b = 0
            # Conservatively treat mined/hidden branch semantics as structural
            # whether implemented by hard override or branch-intent component.
            if hidden:
                b |= STRUCT_HIDDEN_BRANCH
            if (not hidden) and mode in _FLOORED_MODES:
                b |= STRUCT_RECOVERY_FLOOR
            if route_blocked and mode == "yield_rejoin":
                b |= STRUCT_ROUTE_OVERRIDE
            if secondary and mode == "avoid_secondary":
                b |= STRUCT_SECONDARY_FLOOR
            bits[k, l] |= int(b)
    return bits


def nested_tail_truth_contract(
    sample: dict[str, Any],
    *,
    alpha: float = 0.2,
    beta: float = 0.2,
    top_m: int = 8,
    recompute_tolerance: float = 1.0e-5,
) -> TailTruthContractRecord:
    """Conservatively identify whether the teacher R_dep tail is physical-only.

    The active path is computed with the same nested OC-MERO order as the
    teacher target: inner observation-compatible beta-LCVAR -> best option per
    observation anchor -> outer alpha-LCVAR.  Structural exposure is the exact
    nested tail influence mass landing on a root-option cell that *could* have
    been structurally modified by the current teacher implementation.
    """
    M = np.asarray(sample.get("m_star"), dtype=np.float64)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("m_star must be a non-empty [K,L] matrix")
    K, L = M.shape
    p = np.asarray(sample.get("root_probs", np.zeros(K)), dtype=np.float64).reshape(-1)[:K]
    rv = np.asarray(sample.get("root_valid", np.ones(K)), dtype=bool).reshape(-1)[:K]
    if rv.size < K:
        rv = np.pad(rv, (0, K-rv.size), constant_values=False)
    p = np.where(rv, p, 0.0)
    p = normalize_weights(p)
    C = np.asarray(sample.get("c_star", np.eye(K)), dtype=np.float64)
    if C.shape != (K, K):
        raise ValueError(f"c_star shape mismatch: {C.shape} vs {(K,K)}")
    C_eff = sparsify_compatibility(C, int(top_m))
    ov = np.asarray(sample.get("option_valid", np.ones(L)), dtype=bool).reshape(-1)[:L]
    if ov.size < L:
        ov = np.pad(ov, (0, L-ov.size), constant_values=False)

    q = np.full((K, L), -1.0e9, dtype=np.float64)
    inner_inf: list[list[np.ndarray | None]] = [[None for _ in range(L)] for _ in range(K)]
    for i in range(K):
        w = normalize_weights(C_eff[i] * p)
        for l in range(L):
            if not ov[l]:
                continue
            q[i, l] = weighted_lcvar(M[:, l], w, float(beta))
            inner_inf[i][l] = weighted_lcvar_influence_np(M[:, l], w, float(beta))
    best_l = np.argmax(q, axis=1)
    r = q[np.arange(K), best_l]
    outer_inf = weighted_lcvar_influence_np(r, p, float(alpha))
    r_dep = weighted_lcvar(r, p, float(alpha))

    reason_bits = structural_root_option_reason_bits(sample)
    reason_mass = {name: 0.0 for name in REASON_NAMES.values()}
    structural_mass = 0.0
    active_cells = 0
    structural_cells = 0
    for i in range(K):
        oi = float(outer_inf[i])
        if oi <= 0.0:
            continue
        l = int(best_l[i])
        ii = inner_inf[i][l]
        if ii is None:
            continue
        for k in range(K):
            ik = float(ii[k])
            if ik <= 0.0:
                continue
            mass = oi * ik
            active_cells += 1
            b = int(reason_bits[k, l])
            if b:
                structural_cells += 1
                structural_mass += mass
                for flag, name in REASON_NAMES.items():
                    if b & flag:
                        reason_mass[name] += mass

    stored = float(np.asarray(sample.get("r_dep_star", np.nan)).item())
    err = abs(float(r_dep) - stored) if np.isfinite(stored) else float("inf")
    valid = bool(np.isfinite(r_dep) and np.isfinite(stored) and err <= float(recompute_tolerance))
    physical = bool(valid and structural_mass <= 1.0e-12)
    return TailTruthContractRecord(
        valid=valid,
        physical_identifiable=physical,
        structural_exposure_mass=float(structural_mass),
        structural_reason_mass={k: float(v) for k, v in reason_mass.items()},
        active_nested_tail_cells=int(active_cells),
        structurally_exposed_tail_cells=int(structural_cells),
        r_dep_stored=stored,
        r_dep_recomputed=float(r_dep),
        r_dep_abs_error=float(err),
        alpha=float(alpha),
        beta=float(beta),
        top_m=int(top_m),
    )
