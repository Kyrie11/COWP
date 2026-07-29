from __future__ import annotations

import numpy as np


def false_recoverability_admission(admitted: np.ndarray, teacher_r_dep: np.ndarray) -> float:
    admitted = np.asarray(admitted, dtype=bool)
    r = np.asarray(teacher_r_dep, dtype=float)
    if admitted.sum() == 0:
        return 0.0
    return float(np.mean(r[admitted] < 0.0))


def executed_false_admission(selected_idx: int, teacher_r_dep: np.ndarray) -> float:
    r = np.asarray(teacher_r_dep, dtype=float)
    return float(r[int(selected_idx)] < 0.0)


def deployable_recovery_success(
    m_star: np.ndarray,
    root_probs: np.ndarray,
    selected_options: np.ndarray | int,
    root_valid: np.ndarray | None = None,
) -> float:
    """Probability mass of valid roots whose selected shared option succeeds.

    Padded/invalid roots must be removed before normalization; otherwise mixed
    datasets with different root counts make DRS artificially low.  If a model
    outputs an option index outside the sample's option range, that root is
    counted as failed rather than raising an indexing error.
    """
    M = np.asarray(m_star, dtype=float)
    if M.ndim != 2 or M.shape[0] == 0:
        return 0.0
    K, L = M.shape
    p = np.asarray(root_probs, dtype=float).reshape(-1)[:K]
    if p.size < K:
        p = np.pad(p, (0, K - p.size))
    valid = np.ones(K, dtype=bool) if root_valid is None else np.asarray(root_valid, dtype=bool).reshape(-1)[:K]
    if valid.size < K:
        valid = np.pad(valid, (0, K - valid.size), constant_values=False)
    p = np.where(valid, np.clip(p, 0.0, None), 0.0)
    denom = float(p.sum())
    if denom <= 1e-8:
        return 0.0
    p = p / denom
    if isinstance(selected_options, (int, np.integer)):
        opt = np.full(K, int(selected_options), dtype=int)
    else:
        opt = np.asarray(selected_options, dtype=int).reshape(-1)
        if opt.size < K:
            opt = np.pad(opt, (0, K - opt.size), mode="edge" if opt.size else "constant")
        opt = opt[:K]
    vals = np.zeros(K, dtype=float)
    for k in range(K):
        if not valid[k]:
            continue
        l = int(opt[k])
        vals[k] = float(0 <= l < L and np.isfinite(M[k, l]) and M[k, l] >= 0.0)
    return float(np.sum(p * vals))


def _valid_root_weights(root_probs: np.ndarray, k: int, root_valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(root_probs, dtype=float).reshape(-1)[:k]
    if p.size < k:
        p = np.pad(p, (0, k - p.size))
    valid = np.ones(k, dtype=bool) if root_valid is None else np.asarray(root_valid, dtype=bool).reshape(-1)[:k]
    if valid.size < k:
        valid = np.pad(valid, (0, k - valid.size), constant_values=False)
    p = np.where(valid, np.clip(p, 0.0, None), 0.0)
    denom = float(p.sum())
    if denom <= 1e-8:
        return np.zeros(k, dtype=float), valid
    return p / denom, valid


def best_shared_option_index(
    q: np.ndarray,
    root_probs: np.ndarray,
    gamma: float = 0.0,
    root_valid: np.ndarray | None = None,
    option_valid: np.ndarray | None = None,
) -> int:
    """Return one globally shared recovery-option index.

    DRS is supposed to evaluate a deployable *shared* action.  The previous
    implementation used ``argmax`` independently for each root, which inflated
    diagnostics and did not match the executed recovery-action claim.  This
    helper picks one option that maximizes the root-probability mass whose
    option value clears ``gamma``; mean value is only a small tie-breaker.
    """
    Q = np.asarray(q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] == 0 or Q.shape[1] == 0:
        return 0
    K, L = Q.shape
    w, valid = _valid_root_weights(root_probs, K, root_valid)
    if float(w.sum()) <= 1e-8:
        return 0
    opt_valid = np.ones(L, dtype=bool) if option_valid is None else np.asarray(option_valid, dtype=bool).reshape(-1)[:L]
    if opt_valid.size < L:
        opt_valid = np.pad(opt_valid, (0, L - opt_valid.size), constant_values=False)
    finite = np.isfinite(Q) & valid[:, None] & opt_valid[None, :]
    success = ((Q >= float(gamma)) & finite).astype(float)
    success_mass = (success * w[:, None]).sum(axis=0)
    value = np.where(finite, np.clip(Q, -5.0, 5.0), 0.0)
    value_score = (value * w[:, None]).sum(axis=0)
    score = success_mass + 0.01 * value_score
    score = np.where(opt_valid, score, -1.0e9)
    return int(np.argmax(score))


def predicted_shared_option_success(
    q: np.ndarray,
    root_probs: np.ndarray,
    gamma: float = 0.0,
    root_valid: np.ndarray | None = None,
    option_valid: np.ndarray | None = None,
) -> float:
    """Predicted DRS proxy for one globally shared option."""
    Q = np.asarray(q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] == 0 or Q.shape[1] == 0:
        return 0.0
    K, _L = Q.shape
    w, valid = _valid_root_weights(root_probs, K, root_valid)
    if float(w.sum()) <= 1e-8:
        return 0.0
    opt = best_shared_option_index(Q, root_probs, gamma=gamma, root_valid=root_valid, option_valid=option_valid)
    col = Q[:, int(opt)]
    return float(np.sum(w * (valid & np.isfinite(col) & (col >= float(gamma)))))


def post_contact_deployability_score(drs: float, r_dep: float, odg: float) -> float:
    """Compact post-contact recovery score for unavoidable-contact regimes.

    It rewards actual shared-option deployability, then discounts large
    oracle--deployable gaps and negative deployable margin.  This is not a
    replacement for reporting DRS/FRA/ODG separately; it is a contact-regime
    summary for ranking operating points.
    """
    try:
        drs_f = float(drs)
        r_f = float(r_dep)
        gap_f = max(0.0, float(odg))
    except Exception:
        return 0.0
    margin_gate = 1.0 / (1.0 + np.exp(-r_f))
    gap_discount = np.exp(-gap_f)
    return float(np.clip(drs_f, 0.0, 1.0) * margin_gate * gap_discount)


def nominal_utility_preservation(nominal_u: float, selected_u: float, sigma_u: float = 1.0) -> dict[str, float]:
    regret = float(nominal_u - selected_u)
    return {"nominal_regret": regret, "bounded_NUP": float(np.exp(-max(0.0, regret) / max(float(sigma_u), 1e-6)))}


def summarize_selection_metrics(records: list[dict], sigma_u: float = 1.0) -> dict:
    if not records:
        return {}
    fra_cand = np.mean([r["fra_cand"] for r in records])
    fra_exec = np.mean([r["fra_exec"] for r in records])
    drs = np.mean([r["drs"] for r in records])
    odg = np.mean([r["odg"] for r in records])
    odg_pos = np.mean([max(0.0, r["odg"]) for r in records])
    nup = np.mean([r["nup"] for r in records])
    art = [r for r in records if r.get("artifact", False)]
    out = {"FRA_cand": float(fra_cand), "FRA_exec": float(fra_exec), "DRS": float(drs), "ODG": float(odg), "ODG_pos": float(odg_pos), "bounded_NUP": float(nup), "artifact_selection_rate": float(np.mean([r.get("selected_artifact", False) for r in records]))}
    if any("post_contact_deployability" in r for r in records):
        vals = [float(r["post_contact_deployability"]) for r in records if "post_contact_deployability" in r and np.isfinite(float(r["post_contact_deployability"]))]
        out["post_contact_deployability"] = float(np.mean(vals)) if vals else None
    if art:
        out.update({"ODG_artifact": float(np.mean([r["odg"] for r in art])), "FRA_exec_artifact": float(np.mean([r["fra_exec"] for r in art])), "DRS_artifact": float(np.mean([r["drs"] for r in art]))})
    else:
        out.update({"ODG_artifact": None, "FRA_exec_artifact": None, "DRS_artifact": None})
    return out
