from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ocrap.utils.geometry import agent_state_to_box, wrap_angle


@lru_cache(maxsize=32)
def grid_coords(radius: float, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    # Use an ego-centred lattice that contains (0, 0).  Cache read-only arrays
    # because BEV construction repeatedly requests the same geometry.
    radius = float(radius); resolution = float(resolution)
    n = max(3, int(round(2 * radius / resolution)) + 1)
    xs = np.linspace(-radius, radius, n, dtype=np.float32)
    ys = np.linspace(-radius, radius, n, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    X.setflags(write=False); Y.setflags(write=False)
    return X, Y


@lru_cache(maxsize=32)
def ego_centered_grid_geometry(radius: float, resolution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cached XY, polar radius/angle and in-range mask for ego-centred BEV."""
    X, Y = grid_coords(float(radius), float(resolution))
    R = np.sqrt(X * X + Y * Y).astype(np.float32)
    T = np.arctan2(Y, X).astype(np.float32)
    M = (R <= float(radius))
    R.setflags(write=False); T.setflags(write=False); M.setflags(write=False)
    return X, Y, R, T, M


def angular_interval_for_box(box: np.ndarray, ego_xy: np.ndarray) -> tuple[float, float, float]:
    box = np.asarray(box, dtype=np.float32)
    dx, dy = float(box[0] - ego_xy[0]), float(box[1] - ego_xy[1])
    dist = math.hypot(dx, dy)
    center = math.atan2(dy, dx)
    half = math.atan2(0.5 * max(float(box[5]), float(box[6])), max(dist, 1e-3))
    return center - half, center + half, dist


def project_occlusion_shadow(ego_xy: np.ndarray, occluder_box: np.ndarray, grid: tuple[np.ndarray, np.ndarray], max_range: float) -> np.ndarray:
    X, Y = grid
    r = np.sqrt((X - ego_xy[0]) ** 2 + (Y - ego_xy[1]) ** 2)
    theta = np.arctan2(Y - ego_xy[1], X - ego_xy[0])
    a0, a1, d = angular_interval_for_box(occluder_box, ego_xy)
    center = 0.5 * (a0 + a1)
    width = max(abs(a1 - a0), math.atan2(max(float(occluder_box[5]), float(occluder_box[6])), max(d, 1e-3)))
    diff = np.abs((theta - center + math.pi) % (2.0 * math.pi) - math.pi)
    return (r > d + 0.5 * max(float(occluder_box[5]), float(occluder_box[6]))) & (r <= max_range) & (diff <= 0.5 * width)


def box_grid_mask(box: np.ndarray, grid: tuple[np.ndarray, np.ndarray], pad: float = 0.0) -> np.ndarray:
    X, Y = grid
    x, y, _, _, heading, length, width = np.asarray(box, dtype=np.float32)[:7]
    c, s = math.cos(-float(heading)), math.sin(-float(heading))
    dx, dy = X - float(x), Y - float(y)
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    return (np.abs(lx) <= 0.5 * float(length) + pad) & (np.abs(ly) <= 0.5 * float(width) + pad)


def is_occluded_by_dynamic(box: np.ndarray, occluders: list[np.ndarray], ego_xy: np.ndarray) -> bool:
    a0, a1, d = angular_interval_for_box(box, ego_xy)
    center = 0.5 * (a0 + a1)
    for occ in occluders:
        b0, b1, od = angular_interval_for_box(occ, ego_xy)
        if od >= d - 0.5:
            continue
        width = abs(b1 - b0)
        if abs(float(wrap_angle(center - 0.5 * (b0 + b1)))) <= 0.5 * width:
            return True
    return False


def visible_agent_boxes(agent_states: np.ndarray, agent_valid: np.ndarray, ego_state: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    radius = float(cfg.get("local_radius_m", 80.0))
    ego_xy = np.asarray(ego_state[:2], dtype=np.float32)
    boxes = []
    idxs = []
    occluders: list[np.ndarray] = []
    for i, (s, ok) in enumerate(zip(agent_states, agent_valid.astype(bool))):
        if i == 0 or not ok:
            continue
        b = agent_state_to_box(s)
        if np.linalg.norm(b[:2] - ego_xy) <= radius and (b[-1] in (1, 2, 3) or b[5] > 4.5):
            occluders.append(b)
    for i, (s, ok) in enumerate(zip(agent_states, agent_valid.astype(bool))):
        if i == 0 or not ok:
            continue
        b = agent_state_to_box(s)
        if np.linalg.norm(b[:2] - ego_xy) <= radius and not is_occluded_by_dynamic(b, occluders, ego_xy):
            boxes.append(b)
            idxs.append(i)
    if not boxes:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0,), dtype=bool), []
    return np.asarray(boxes, dtype=np.float32), np.ones((len(boxes),), dtype=bool), idxs
