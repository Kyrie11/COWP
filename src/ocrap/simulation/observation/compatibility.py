from __future__ import annotations

import math

import numpy as np

from ocrap.data.schema import Observation
from ocrap.utils.geometry import greedy_assignment_cost, wrap_angle


def observation_distance(a: Observation, b: Observation, cfg: dict) -> float:
    ocfg = cfg.get("obs_distance", {})
    p_unmatch = float(ocfg.get("p_unmatch", 5.0))
    d_box = greedy_assignment_cost(a.boxes[a.box_valid], b.boxes[b.box_valid], p_unmatch)
    denom = max(len(a.boxes[a.box_valid]), len(b.boxes[b.box_valid]), 1)
    d_box /= denom
    ua = a.occ_mask[2] > 0.5 if a.occ_mask.shape[0] > 2 else np.zeros(a.occ_mask.shape[-2:], dtype=bool)
    ub = b.occ_mask[2] > 0.5 if b.occ_mask.shape[0] > 2 else np.zeros(b.occ_mask.shape[-2:], dtype=bool)
    union = np.logical_or(ua, ub).sum()
    d_occ = 0.0 if union == 0 else 1.0 - float(np.logical_and(ua, ub).sum() / union)
    da = np.asarray(a.ego_state)
    db = np.asarray(b.ego_state)
    d_ego = float(np.linalg.norm(da[:2] - db[:2]) / float(ocfg.get("s_p", 2.0)))
    if da.size > 6 and db.size > 6:
        d_ego += float(abs(da[6] - db[6]) / float(ocfg.get("s_v", 2.0)))
    if da.size > 4 and db.size > 4:
        d_ego += 0.5 * float(abs(wrap_angle(float(da[4] - db[4]))) / max(float(ocfg.get("s_yaw", 0.2)), 1e-6))
    d_map = 0.0
    if a.route_visible is not None and b.route_visible is not None and len(a.route_visible) and len(b.route_visible):
        d_map = abs(len(a.route_visible) - len(b.route_visible)) / max(len(a.route_visible), len(b.route_visible), 1)
    return float(d_box + float(ocfg.get("lambda_occ", 1.0)) * d_occ + float(ocfg.get("lambda_ego", 0.5)) * d_ego + float(ocfg.get("lambda_map", 0.5)) * d_map)


def compatibility_labels(observations: list[Observation], cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = len(observations)
    D = np.zeros((K, K), dtype=np.float32)
    eps = float(cfg.get("epsilon_obs", 1.0))
    tau = float(cfg.get("tau_obs", 1.0))
    for i in range(K):
        for j in range(i + 1, K):
            d = observation_distance(observations[i], observations[j], cfg)
            D[i, j] = D[j, i] = d
    Y = (D <= eps).astype(np.float32)
    C = np.exp(-(D.astype(np.float64) ** 2) / max(tau, 1e-6)).astype(np.float32)
    np.fill_diagonal(Y, 1.0)
    np.fill_diagonal(C, 1.0)
    return Y, C, D
