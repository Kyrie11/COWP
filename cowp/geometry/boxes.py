from __future__ import annotations

import numpy as np


def normalize_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def box_corners(x: float, y: float, heading: float, length: float, width: float, inflation: float = 0.0) -> np.ndarray:
    l2 = 0.5 * max(float(length) + 2 * inflation, 1e-3)
    w2 = 0.5 * max(float(width) + 2 * inflation, 1e-3)
    local = np.array([[l2, w2], [l2, -w2], [-l2, -w2], [-l2, w2]], dtype=np.float32)
    c, s = np.cos(float(heading)), np.sin(float(heading))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def _poly_axes(poly: np.ndarray) -> np.ndarray:
    edges = np.roll(poly, -1, axis=0) - poly
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(norms, 1e-8)


def _project(poly: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    vals = poly @ axis
    return float(vals.min()), float(vals.max())


def polygons_overlap(poly_a: np.ndarray, poly_b: np.ndarray, eps: float = 0.0) -> bool:
    for axis in np.concatenate([_poly_axes(poly_a), _poly_axes(poly_b)], axis=0):
        a0, a1 = _project(poly_a, axis)
        b0, b1 = _project(poly_b, axis)
        if a1 + eps < b0 or b1 + eps < a0:
            return False
    return True


def obb_overlap(box_a: np.ndarray, box_b: np.ndarray, inflation: float = 0.0) -> bool:
    a = box_corners(box_a[0], box_a[1], box_a[2], box_a[5], box_a[6], inflation)
    b = box_corners(box_b[0], box_b[1], box_b[2], box_b[5], box_b[6], inflation)
    return polygons_overlap(a, b)


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def polygon_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    if polygons_overlap(poly_a, poly_b):
        return 0.0
    d = np.inf
    for p in poly_a:
        for j in range(len(poly_b)):
            d = min(d, _point_segment_distance(p, poly_b[j], poly_b[(j + 1) % len(poly_b)]))
    for p in poly_b:
        for j in range(len(poly_a)):
            d = min(d, _point_segment_distance(p, poly_a[j], poly_a[(j + 1) % len(poly_a)]))
    return float(d)


def obb_distance(box_a: np.ndarray, box_b: np.ndarray, inflation: float = 0.0) -> float:
    a = box_corners(box_a[0], box_a[1], box_a[2], box_a[5], box_a[6], inflation)
    b = box_corners(box_b[0], box_b[1], box_b[2], box_b[5], box_b[6], inflation)
    return polygon_distance(a, b)


def trajectory_min_obb_distance(traj_a: np.ndarray, traj_b: np.ndarray, valid_a: np.ndarray | None = None, valid_b: np.ndarray | None = None) -> float:
    t = min(len(traj_a), len(traj_b))
    if t == 0:
        return float("inf")
    va = np.ones(t, dtype=bool) if valid_a is None else np.asarray(valid_a[:t], dtype=bool)
    vb = np.ones(t, dtype=bool) if valid_b is None else np.asarray(valid_b[:t], dtype=bool)
    dmin = float("inf")
    for k in range(t):
        if va[k] and vb[k]:
            dmin = min(dmin, obb_distance(traj_a[k], traj_b[k]))
    return float(dmin)
