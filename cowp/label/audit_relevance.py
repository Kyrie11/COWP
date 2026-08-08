from __future__ import annotations

import numpy as np

from cowp.core.constants import PriorityRelation
from cowp.core.types import ScenarioData
from cowp.geometry.collision import unsafe_between
from cowp.label.burden import compute_burden


def canonical_root_weights(natural: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    """Return the floor-smoothed canonical natural-root distribution [A,M].

    This is the single probability measure used by candidate-conditioned audit,
    witness certification, and RootTransport.  Keeping it in one helper avoids
    small but damaging semantic drift between data-construction stages.
    """
    valid = np.asarray(natural["valid"], dtype=bool)
    raw = np.asarray(natural["weight"], dtype=np.float32)
    A, M = valid.shape
    out = np.zeros((A, M), dtype=np.float32)
    eps_p = float(np.clip(cfg.get("ncf", {}).get("root_probability_floor", 0.02), 0.0, 0.25))
    for a in range(A):
        idx = np.where(valid[a])[0]
        if not len(idx):
            continue
        w_all = np.maximum(raw[a, idx], 0.0)
        # Match SetTransport and paper_aligned_supervision_batch exactly: very
        # low-probability roots are removed from the canonical support before
        # floor smoothing.  Earlier v16.8.9 draft audit weights retained them,
        # creating another subtle train/label probability-measure mismatch.
        p_min = max(float(cfg.get("ncf", {}).get(
            "min_alt_weight", cfg.get("planning", {}).get("set_transport_min_alt_weight", 0.03)
        )), 0.0)
        keep = w_all >= p_min
        if not np.any(keep):
            keep = np.ones_like(w_all, dtype=bool)
        idx_keep = idx[keep]
        w = w_all[keep]
        w = w / max(float(np.sum(w)), 1.0e-8)
        out[a, idx_keep] = (1.0 - eps_p) * w + eps_p / len(idx_keep)
    return out


def compute_candidate_agent_audit(
    scene: ScenarioData,
    candidates: dict[str, np.ndarray],
    critical: dict[str, np.ndarray],
    natural: dict[str, np.ndarray],
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Compute candidate-conditioned *causal audit relevance*.

    A global critical-agent universe is useful for stable scene semantics, but a
    candidate should only be universally certified against agents whose factual
    low-burden natural options it actually perturbs.  Earlier versions audited
    every candidate against every global critical agent.  This created silent NCF
    blockers: an unrelated pair could have low OPR, no positive witness, yet still
    invalidate the whole candidate.

    A natural root is affected when it is low-burden under the neutral reference
    and the ego candidate either makes it geometrically unsafe or directly pushes
    its burden above the same adaptive beta.  The pair is auditable when the
    floor-smoothed affected root mass exceeds ``ncf.audit_min_relevance_mass``.
    This is a causal support mask, not a relaxation of burden/OPR thresholds.
    """
    limits = cfg.get("limits", {})
    K = int(limits.get("max_candidates", 64))
    A = int(limits.get("max_critical_agents", 8))
    M = int(limits.get("max_natural_alternatives", natural["valid"].shape[-1]))

    pair_relevant = np.zeros((K, A), dtype=bool)
    relevance_mass = np.zeros((K, A), dtype=np.float32)
    root_affected = np.zeros((K, A, M), dtype=bool)
    root_unsafe = np.zeros((K, A, M), dtype=bool)
    root_direct_burden = np.zeros((K, A, M), dtype=np.float32)
    # Keep the budget-crossing subset explicit.  v16.8.9 inferred
    # ``affected & ~unsafe`` downstream; storing it directly makes the data
    # contract auditable without forcing a minimum prevalence in a small probe.
    root_budget_crossed = np.zeros((K, A, M), dtype=bool)
    root_burden_only_affected = np.zeros((K, A, M), dtype=bool)
    root_weight = canonical_root_weights(natural, cfg)

    min_mass = float(cfg.get("ncf", {}).get(
        "audit_min_relevance_mass",
        cfg.get("ncf", {}).get("positive_min_natural_conflict_mass", 0.10),
    ))
    direct_margin = float(cfg.get("ncf", {}).get("audit_direct_burden_margin", 0.0))

    for k in range(K):
        if not bool(candidates["valid"][k]):
            continue
        ego = np.asarray(candidates["trajectory"][k], dtype=np.float32)
        for a in range(A):
            if not bool(critical["valid"][a]):
                continue
            idx = int(critical["track_index"][a])
            object_type = int(scene.object_type[idx])
            rho = PriorityRelation(int(critical.get("base_priority", np.zeros(A, dtype=np.int32))[a]))
            beta = float(natural.get("beta", np.full(A, 0.65, dtype=np.float32))[a])
            valid_roots = np.where(np.asarray(natural["valid"][a], dtype=bool))[0]
            for m in valid_roots:
                if float(natural["burden_neutral"][a, m]) > beta:
                    # This option is already outside the neutral low-burden set;
                    # ego cannot be blamed for its exclusion from the certificate.
                    continue
                nat = np.asarray(natural["traj"][a, m], dtype=np.float32)
                unsafe = bool(unsafe_between(ego, nat, cfg, agent_type=object_type).unsafe)
                b_under, _ = compute_burden(
                    nat, ego, cfg, object_type, natural_ref=nat, rho=rho,
                )
                root_unsafe[k, a, m] = unsafe
                root_direct_burden[k, a, m] = float(b_under)
                budget_crossed = bool(float(b_under) > beta + direct_margin)
                burden_only = bool(budget_crossed and not unsafe)
                root_budget_crossed[k, a, m] = budget_crossed
                root_burden_only_affected[k, a, m] = burden_only
                affected = bool(unsafe or budget_crossed)
                root_affected[k, a, m] = affected
            mass = float(np.sum(root_weight[a] * root_affected[k, a].astype(np.float32)))
            relevance_mass[k, a] = mass
            pair_relevant[k, a] = bool(mass > min_mass - 1.0e-8)

    return {
        "pair_relevant": pair_relevant,
        "relevance_mass": relevance_mass,
        "root_affected": root_affected,
        "root_unsafe": root_unsafe,
        "root_direct_burden": root_direct_burden,
        "root_budget_crossed": root_budget_crossed,
        "root_burden_only_affected": root_burden_only_affected,
        "canonical_root_weight": root_weight,
    }
