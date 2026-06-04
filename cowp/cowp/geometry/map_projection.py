from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cowp.core.types import Lane, MapData
from cowp.geometry.boxes import normalize_angle


@dataclass
class Projection:
    lane_id: int
    s: float
    l: float
    lane_heading: float
    heading_error: float
    distance: float
    on_lane_prob: float


def polyline_arc_length(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    if len(xy) == 0:
        return np.zeros(0, dtype=np.float32)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    return np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)


def project_point_to_polyline(point: np.ndarray, polyline_xy: np.ndarray) -> tuple[float, float, float, float]:
    point = np.asarray(point[:2], dtype=np.float32)
    xy = np.asarray(polyline_xy[:, :2], dtype=np.float32)
    if len(xy) < 2:
        if len(xy) == 1:
            d = float(np.linalg.norm(point - xy[0]))
            return 0.0, 0.0, 0.0, d
        return 0.0, 0.0, 0.0, float("inf")
    arcs = polyline_arc_length(xy)
    best = (0.0, 0.0, 0.0, float("inf"))
    for j in range(len(xy) - 1):
        a, b = xy[j], xy[j + 1]
        ab = b - a
        seg_len2 = float(ab @ ab)
        if seg_len2 <= 1e-9:
            continue
        u = float(np.clip(((point - a) @ ab) / seg_len2, 0.0, 1.0))
        proj = a + u * ab
        diff = point - proj
        d = float(np.linalg.norm(diff))
        heading = float(np.arctan2(ab[1], ab[0]))
        cross = ab[0] * diff[1] - ab[1] * diff[0]
        l = float(np.sign(cross) * d)
        s = float(arcs[j] + u * np.sqrt(seg_len2))
        if d < best[3]:
            best = (s, l, heading, d)
    return best


def project_state_to_lane(state: np.ndarray, map_data: MapData, search_radius: float = 6.0) -> Projection:
    best_score = float("inf")
    best_proj = Projection(-1, 0.0, 0.0, 0.0, 0.0, float("inf"), 0.0)
    heading = float(state[6] if len(state) >= 7 else state[2])
    for lane_id, lane in map_data.lanes.items():
        if len(lane.xy) < 2:
            continue
        s, l, lane_heading, dist = project_point_to_polyline(state[:2], lane.xy)
        heading_error = float(abs(normalize_angle(heading - lane_heading)))
        score = dist + 2.0 * heading_error
        if score < best_score:
            prob = float(np.exp(-0.5 * (dist / max(search_radius, 1e-3)) ** 2)) if dist <= search_radius else 0.0
            best_score = score
            best_proj = Projection(int(lane_id), s, l, lane_heading, heading_error, dist, prob)
    if best_proj.distance > search_radius:
        return Projection(-1, best_proj.s, best_proj.l, best_proj.lane_heading, best_proj.heading_error, best_proj.distance, 0.0)
    return best_proj


def project_trajectory_to_lane(traj: np.ndarray, map_data: MapData, search_radius: float = 6.0) -> list[Projection]:
    return [project_state_to_lane(t, map_data, search_radius) for t in traj]
