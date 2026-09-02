from __future__ import annotations

import numpy as np

from ocrap.data.schema import SceneHistory
from ocrap.utils.geometry import agent_state_to_box

from .visibility import box_grid_mask, grid_coords, project_occlusion_shadow

# Channel convention: 0 visible_free, 1 occupied_visible, 2 unknown, 3 occluder, 4 route, 5 drivable, 6 confidence.


def paint_route_and_drivable(mask: np.ndarray, history: SceneHistory, grid: tuple[np.ndarray, np.ndarray], cfg: dict) -> None:
    X, Y = grid
    route_width = float(cfg.get("route_width", 3.5))
    if history.route.size:
        route_xy = history.route[:, :2]
    else:
        route_xy = np.stack([np.linspace(-10, 80, 50), np.zeros(50)], axis=-1)
    route_mask = np.zeros_like(X, dtype=bool)
    drivable = np.zeros_like(X, dtype=bool)
    for pt in route_xy[:: max(1, len(route_xy) // 80)]:
        d2 = (X - pt[0]) ** 2 + (Y - pt[1]) ** 2
        route_mask |= d2 <= (route_width * 0.75) ** 2
        drivable |= d2 <= (route_width * 2.2) ** 2
    # Include map polyline lanes/crosswalks when available.
    if history.map_polylines.size:
        valid_pts = history.map_polylines[..., :2][history.map_valid.astype(bool)]
        for pt in valid_pts[:: max(1, len(valid_pts) // 250)]:
            d2 = (X - pt[0]) ** 2 + (Y - pt[1]) ** 2
            drivable |= d2 <= (route_width * 1.5) ** 2
    mask[4, route_mask] = 1.0
    mask[5, drivable] = 1.0


def render_base_occ_mask(history: SceneHistory, cfg: dict) -> np.ndarray:
    radius = float(cfg.get("local_radius_m", 80.0))
    res = float(cfg.get("bev_resolution_m", 1.0))
    C = int(cfg.get("bev_channels", 7))
    X, Y = grid_coords(radius, res)
    H, W = X.shape
    mask = np.zeros((C, H, W), dtype=np.float32)
    grid = (X, Y)
    r = np.sqrt(X**2 + Y**2)
    in_range = r <= radius
    paint_route_and_drivable(mask, history, grid, cfg)
    # If map is sparse, keep a broad ego-centered drivable ribbon so unknown ratios are meaningful.
    if mask[5].sum() < 10:
        mask[5, np.abs(Y) <= float(cfg.get("route_width", 3.5)) * 2.2] = 1.0
    mask[0, in_range & (mask[5] > 0.5)] = 1.0
    mask[2, (~in_range) | ((mask[5] < 0.5) & in_range)] = 1.0

    current = history.agent_history[-1] if history.agent_history.size else np.zeros((0, 16), dtype=np.float32)
    valid = history.agent_valid[-1].astype(bool) if history.agent_valid.size else np.zeros((0,), dtype=bool)
    ego_xy = np.zeros(2, dtype=np.float32)
    for i, (s, ok) in enumerate(zip(current, valid)):
        if i == 0 or not ok:
            continue
        box = agent_state_to_box(s)
        if np.linalg.norm(box[:2] - ego_xy) > radius:
            continue
        occ_cells = box_grid_mask(box, grid, pad=0.3)
        mask[1, occ_cells] = 1.0
        mask[0, occ_cells] = 0.0
        mask[2, occ_cells] = 0.0
        vehicle_like = bool(box[-1] in (1, 2, 3) or box[5] > 4.5)
        if vehicle_like:
            mask[3, occ_cells] = 1.0
            shadow = project_occlusion_shadow(ego_xy, box, grid, radius)
            shadow &= mask[5] > 0.5
            mask[2, shadow] = 1.0
            mask[0, shadow] = 0.0
    # Visible free and unknown should not overlap in drivable crop.
    mask[0, mask[2] > 0.5] = 0.0
    mask[6] = np.where(mask[2] > 0.5, 0.15, np.maximum(mask[0], mask[1]))
    return mask


def unknown_ratio_in_corridor(occ_mask: np.ndarray) -> float:
    if occ_mask.size == 0 or occ_mask.shape[0] < 6:
        return 0.0
    valid = (occ_mask[5] > 0.5) & ((occ_mask[4] > 0.5) | (occ_mask[0] > 0.0) | (occ_mask[2] > 0.0))
    denom = int(valid.sum())
    if denom <= 0:
        return 0.0
    return float(((occ_mask[2] > 0.5) & valid).sum() / denom)
