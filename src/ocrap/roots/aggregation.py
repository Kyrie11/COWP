from __future__ import annotations

import numpy as np

from ocrap.algorithms.lcv import normalize_weights, weighted_lcvar, weighted_mean


def aggregate_root_margins(M_future: np.ndarray, assignments: np.ndarray, probs: np.ndarray, K: int, cfg: dict) -> np.ndarray:
    M_future = np.asarray(M_future, dtype=np.float32)
    assignments = np.asarray(assignments, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    L = M_future.shape[1]
    out = np.zeros((K, L), dtype=np.float32)
    mode = str(cfg.get("root_margin_aggregation", "lcvar"))
    alpha = float(cfg.get("intra_root_lcvar_alpha", 0.2))
    for k in range(K):
        idx = np.where(assignments == k)[0]
        if len(idx) == 0:
            out[k] = -1e9
            continue
        w = normalize_weights(probs[idx])
        for l in range(L):
            vals = M_future[idx, l]
            if mode == "mean":
                out[k, l] = weighted_mean(vals, w)
            elif mode == "min":
                out[k, l] = float(np.min(vals))
            else:
                out[k, l] = weighted_lcvar(vals, w, alpha)
    return out
