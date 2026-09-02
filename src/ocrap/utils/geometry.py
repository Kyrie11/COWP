from __future__ import annotations

import math
from itertools import permutations
from typing import Iterable

import numpy as np


EPS = 1e-8


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def transform_points_to_ego(points: np.ndarray, ego_xy: np.ndarray, ego_heading: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    R = rotation_matrix(-float(ego_heading))
    return (pts[..., :2] - np.asarray(ego_xy, dtype=np.float32)) @ R.T


def transform_vectors_to_ego(vecs: np.ndarray, ego_heading: float) -> np.ndarray:
    R = rotation_matrix(-float(ego_heading))
    return np.asarray(vecs, dtype=np.float32) @ R.T


def transform_states_to_ego(states: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
    out = np.asarray(states, dtype=np.float32).copy()
    if out.size == 0:
        return out
    ego_xy = np.asarray(ego_state[:2], dtype=np.float32)
    ego_heading = float(ego_state[7] if len(ego_state) > 7 else ego_state[5])
    xy = transform_points_to_ego(out[..., :2], ego_xy, ego_heading)
    vxy = transform_vectors_to_ego(out[..., 3:5], ego_heading)
    out[..., :2] = xy
    out[..., 3:5] = vxy
    if out.shape[-1] > 7:
        out[..., 7] = wrap_angle(out[..., 7] - ego_heading)
        out[..., 8] = np.sin(out[..., 7])
        out[..., 9] = np.cos(out[..., 7])
    elif out.shape[-1] > 5:
        out[..., 5] = wrap_angle(out[..., 5] - ego_heading)
    return out


def speed_from_state(s: np.ndarray) -> float:
    return float(math.hypot(float(s[3]), float(s[4])))


def heading_from_state(s: np.ndarray) -> float:
    # Agent state uses heading at index 7 in the new schema and index 5 in the legacy schema.
    if len(s) >= 10:
        return float(s[7])
    return float(s[5])


def agent_state_to_box(s: np.ndarray) -> np.ndarray:
    """Return [x,y,vx,vy,heading,length,width,height,type]."""
    s = np.asarray(s, dtype=np.float32)
    if s.shape[-1] >= 16:
        return np.array([s[0], s[1], s[3], s[4], s[7], s[10], s[11], s[12], s[13]], dtype=np.float32)
    return np.array([s[0], s[1], s[3], s[4], s[5], s[6], s[7], s[8], s[9]], dtype=np.float32)


def ego_state_to_box(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float32)
    return np.array([s[0], s[1], s[2], s[3], s[4], s[7], s[8], 1.5, 1.0], dtype=np.float32)


def oriented_box_corners(box: np.ndarray) -> np.ndarray:
    x, y, _, _, h, length, width = np.asarray(box, dtype=np.float32)[:7]
    local = np.array([[ length/2,  width/2], [ length/2, -width/2], [-length/2, -width/2], [-length/2,  width/2]], dtype=np.float32)
    return local @ rotation_matrix(float(h)).T + np.array([x, y], dtype=np.float32)


def approximate_box_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    a = np.asarray(box_a, dtype=np.float32)
    b = np.asarray(box_b, dtype=np.float32)
    center = float(np.linalg.norm(a[:2] - b[:2]))
    radius_a = 0.5 * math.hypot(float(a[5]), float(a[6]))
    radius_b = 0.5 * math.hypot(float(b[5]), float(b[6]))
    return max(0.0, center - radius_a - radius_b)


def min_box_clearance(ego_box: np.ndarray, boxes: np.ndarray, valid: np.ndarray) -> float:
    vals = [approximate_box_distance(ego_box, b) for b, ok in zip(boxes, valid.astype(bool)) if ok]
    return float(min(vals)) if vals else 99.0


def compute_ttc(ego_state: np.ndarray, boxes: np.ndarray, valid: np.ndarray) -> float:
    ego = np.asarray(ego_state, dtype=np.float32)
    ego_xy = ego[:2]
    ego_v = ego[2:4] if ego.shape[0] <= 9 else ego[3:5]
    best = 99.0
    for b, ok in zip(boxes, valid.astype(bool)):
        if not ok:
            continue
        rel = np.asarray(b[:2], dtype=np.float32) - ego_xy
        rv = ego_v - np.asarray(b[2:4], dtype=np.float32)
        closing = float(np.dot(rv, rel) / (np.linalg.norm(rel) + EPS))
        if closing > 0:
            best = min(best, max(0.0, float(np.linalg.norm(rel)) / closing))
    return float(best)


def greedy_assignment_cost(A: np.ndarray, B: np.ndarray, unmatch_penalty: float) -> float:
    """Small dependency-free visible-box matching cost."""
    n, m = len(A), len(B)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return float(max(n, m) * unmatch_penalty)
    if min(n, m) <= 7:
        if n <= m:
            best = float("inf")
            for perm in permutations(range(m), n):
                cost = sum(float(np.linalg.norm(A[i, :2] - B[j, :2])) for i, j in enumerate(perm))
                cost += (m - n) * unmatch_penalty
                best = min(best, cost)
            return float(best)
        best = float("inf")
        for perm in permutations(range(n), m):
            cost = sum(float(np.linalg.norm(A[i, :2] - B[j, :2])) for j, i in enumerate(perm))
            cost += (n - m) * unmatch_penalty
            best = min(best, cost)
        return float(best)
    # Greedy fallback for many boxes.
    remaining = set(range(m))
    cost = 0.0
    for i in range(n):
        if not remaining:
            cost += unmatch_penalty
            continue
        j = min(remaining, key=lambda jj: float(np.linalg.norm(A[i, :2] - B[jj, :2])))
        cost += float(np.linalg.norm(A[i, :2] - B[j, :2]))
        remaining.remove(j)
    cost += len(remaining) * unmatch_penalty
    return float(cost)
