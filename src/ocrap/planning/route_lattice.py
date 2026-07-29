from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RouteProjection:
    s: float
    d: float
    heading: float
    index: int
    distance: float


def route_xy(route: np.ndarray) -> np.ndarray:
    r = np.asarray(route, dtype=np.float32)
    if r.size == 0:
        xs = np.linspace(0, 100, 80, dtype=np.float32)
        return np.stack([xs, np.zeros_like(xs)], axis=-1)
    return r[:, :2].astype(np.float32)


def cumulative_s(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros(0, dtype=np.float32)
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)


def project_to_route(point: np.ndarray, route: np.ndarray) -> RouteProjection:
    xy = route_xy(route)
    svals = cumulative_s(xy)
    if len(xy) < 2:
        return RouteProjection(0.0, float(point[1]), 0.0, 0, float(np.linalg.norm(point[:2] - xy[0])))
    best = (float("inf"), 0, 0.0, np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    p = np.asarray(point[:2], dtype=np.float32)
    for i in range(len(xy) - 1):
        a, b = xy[i], xy[i + 1]
        v = b - a
        denom = float(np.dot(v, v)) + 1e-8
        u = min(1.0, max(0.0, float(np.dot(p - a, v) / denom)))
        q = a + u * v
        dist = float(np.linalg.norm(p - q))
        if dist < best[0]:
            best = (dist, i, u, q, v)
    dist, i, u, q, v = best
    heading = float(np.arctan2(v[1], v[0]))
    normal = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
    d = float(np.dot(p - q, normal))
    s = float(svals[i] + u * np.linalg.norm(v))
    return RouteProjection(s=s, d=d, heading=heading, index=i, distance=dist)


def interpolate_route(route: np.ndarray, s_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = route_xy(route)
    svals = cumulative_s(xy)
    if len(xy) < 2:
        pts = np.repeat(xy[:1], len(s_query), axis=0)
        hd = np.zeros(len(s_query), dtype=np.float32)
        return pts, hd
    s_query = np.asarray(s_query, dtype=np.float32)
    pts = np.zeros((len(s_query), 2), dtype=np.float32)
    hd = np.zeros(len(s_query), dtype=np.float32)
    for qi, sq in enumerate(s_query):
        sq = float(np.clip(sq, svals[0], svals[-1]))
        idx = int(np.searchsorted(svals, sq, side="right") - 1)
        idx = min(max(idx, 0), len(xy) - 2)
        denom = float(svals[idx + 1] - svals[idx]) + 1e-8
        u = (sq - float(svals[idx])) / denom
        v = xy[idx + 1] - xy[idx]
        pts[qi] = xy[idx] + u * v
        hd[qi] = np.arctan2(v[1], v[0])
    return pts, hd


def offset_route_points(route: np.ndarray, s_query: np.ndarray, lateral_offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts, hd = interpolate_route(route, s_query)
    n = np.stack([-np.sin(hd), np.cos(hd)], axis=-1)
    return pts + n * lateral_offsets[:, None], hd
