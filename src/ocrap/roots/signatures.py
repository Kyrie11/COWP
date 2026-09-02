from __future__ import annotations

import numpy as np

from ocrap.data.schema import CounterfactualFuture


def future_recovery_signature(M_future: np.ndarray, futures: list[CounterfactualFuture], cfg: dict) -> np.ndarray:
    clip = float(cfg.get("margin_clip", 5.0))
    rows = []
    for j, fut in enumerate(futures):
        m = np.clip(M_future[j], -clip, clip)
        meta = fut.metadata
        c = np.array([
            float(meta.get("contact_surrogate", False)),
            float(meta.get("friction_factor", 1.0) < 0.9),
            float(abs(float(meta.get("yaw_rate_impulse", 0.0))) > 0),
            float(meta.get("secondary_threat", False)),
            float(meta.get("control_envelope_uncertain", False)),
            float(meta.get("hidden_emergence", False)),
        ], dtype=np.float32)
        kappa = np.array([
            float(meta.get("route_blocked", False)),
            float(not meta.get("lateral_escape_blocked", False)),
            float(meta.get("rejoin_corridor_available", True)),
            float(meta.get("from_unknown_mask", False)),
        ], dtype=np.float32)
        rows.append(np.concatenate([m.astype(np.float32), c, kappa], axis=0))
    return np.asarray(rows, dtype=np.float32)


def future_trajectory_signature(futures: list[CounterfactualFuture], assignments: np.ndarray, probs: np.ndarray, K: int, width: int = 32) -> np.ndarray:
    out = np.zeros((K, width), dtype=np.float32)
    for k in range(K):
        idx = np.where(assignments == k)[0]
        if len(idx) == 0:
            continue
        w = probs[idx].astype(np.float64)
        w = w / max(float(w.sum()), 1e-8)
        features = []
        for j in idx:
            f = futures[int(j)]
            valid = f.agent_valid.astype(bool)
            ego = f.agent_states[:, 0, :2]
            mean_speed = np.linalg.norm(f.agent_states[..., 3:5], axis=-1)[valid].mean() if valid.any() else 0.0
            features.append(np.array([ego[-1, 0], ego[-1, 1], mean_speed, float(f.metadata.get("hidden_emergence", False)), float(f.metadata.get("secondary_threat", False)), float(f.metadata.get("contact_surrogate", False))], dtype=np.float32))
        arr = np.asarray(features, dtype=np.float32)
        sig = np.average(arr, axis=0, weights=w)
        out[k, : min(width, len(sig))] = sig[:width]
    return out
