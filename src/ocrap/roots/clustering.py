from __future__ import annotations

import numpy as np

from ocrap.algorithms.lcv import normalize_weights
from ocrap.data.schema import CounterfactualFuture, RootClusteringResult

from .signatures import future_recovery_signature


def _pairwise_l1(X: np.ndarray, scale: np.ndarray) -> np.ndarray:
    Z = X / np.maximum(scale, 1e-6)
    return np.abs(Z[:, None, :] - Z[None, :, :]).sum(axis=-1)


def _initial_groups_by_metadata(futures: list[CounterfactualFuture]) -> list[list[int]]:
    groups: dict[tuple, list[int]] = {}
    for j, f in enumerate(futures):
        key = (
            f.source,
            f.metadata.get("artifact_branch", f.metadata.get("targeted_type", f.metadata.get("reactive_variant", ""))),
            bool(f.metadata.get("hidden_emergence", False)),
            bool(f.metadata.get("secondary_threat", False)),
            bool(f.metadata.get("contact_surrogate", False)),
        )
        groups.setdefault(key, []).append(j)
    return list(groups.values())


def cluster_roots(M_future: np.ndarray, probs: np.ndarray, futures: list[CounterfactualFuture], cfg: dict) -> RootClusteringResult:
    K = int(cfg.get("num_roots", 8))
    probs = normalize_weights(probs)
    sig = future_recovery_signature(M_future, futures, cfg)
    J = len(futures)
    groups = _initial_groups_by_metadata(futures)
    # If too many metadata groups, merge smallest-probability groups to nearest larger group by signature distance.
    scale = np.median(np.abs(sig - np.median(sig, axis=0, keepdims=True)), axis=0) * 1.4826
    if not np.all(scale > 1e-6):
        scale = np.where(scale > 1e-6, scale, 1.0)
    metadata = {"scale_source": "sample_local_mad_fallback"}
    def is_protected(g: list[int]) -> bool:
        return any(futures[j].metadata.get("artifact_branch") in {"yield", "accelerate"} for j in g)

    while len(groups) > K:
        weights = [float(probs[g].sum()) for g in groups]
        candidates = [i for i, g in enumerate(groups) if not is_protected(g)] or list(range(len(groups)))
        small = min(candidates, key=lambda i: weights[i])
        center_small = np.average(sig[groups[small]], axis=0, weights=normalize_weights(probs[groups[small]]))
        best = None
        best_d = float("inf")
        for i, g in enumerate(groups):
            if i == small or (is_protected(g) and is_protected(groups[small])):
                continue
            center = np.average(sig[g], axis=0, weights=normalize_weights(probs[g]))
            d = float(np.sum(np.abs((center_small - center) / scale)))
            if d < best_d:
                best_d, best = d, i
        if best is None:
            best = 0 if small != 0 else 1
        groups[best].extend(groups[small])  # type: ignore[index]
        groups.pop(small)
    assignments = np.full(J, -1, dtype=np.int64)
    root_probs = np.zeros(K, dtype=np.float32)
    root_sig = np.zeros((K, sig.shape[1]), dtype=np.float32)
    root_valid = np.zeros(K, dtype=bool)
    reps = np.full(K, -1, dtype=np.int64)
    f2r = np.zeros((J, K), dtype=np.float32)
    for k, g in enumerate(groups[:K]):
        g = sorted(g)
        assignments[g] = k
        root_valid[k] = True
        w = normalize_weights(probs[g])
        root_probs[k] = float(probs[g].sum())
        root_sig[k] = np.average(sig[g], axis=0, weights=w)
        D = _pairwise_l1(sig[g], scale)
        medoid = g[int(np.argmin(D.sum(axis=1)))]
        reps[k] = int(medoid)
        for jj, ww in zip(g, w):
            f2r[jj, k] = float(ww)
    # Assign any unassigned future to root 0 as a safe fallback.
    for j in np.where(assignments < 0)[0]:
        assignments[j] = 0
        f2r[j, 0] = 1.0
    root_probs = normalize_weights(root_probs).astype(np.float32)
    # Placeholder dispersion is filled after observations are rendered; keep field present.
    dispersion = np.zeros(K, dtype=np.float32)
    return RootClusteringResult(assignments, root_probs, root_sig, root_valid, reps, f2r, dispersion, metadata)
