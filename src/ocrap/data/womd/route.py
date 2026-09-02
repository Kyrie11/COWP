from __future__ import annotations

import numpy as np


def _dedupe_and_resample(points: list[tuple[float, float]] | np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2) if len(points) else np.zeros((0, 2), dtype=np.float32)
    if pts.size == 0:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if float(np.linalg.norm(pts[i] - pts[keep[-1]])) > 0.25:
            keep.append(i)
    pts = pts[keep]
    if len(pts) <= 1:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    if float(s[-1]) < 1e-3:
        return pts[:1]
    qs = np.linspace(0.0, float(s[-1]), max_points, dtype=np.float32)
    out = np.zeros((max_points, 2), dtype=np.float32)
    for i, q in enumerate(qs):
        j = int(np.searchsorted(s, q, side="right") - 1)
        j = min(max(j, 0), len(pts) - 2)
        u = float((q - s[j]) / max(s[j + 1] - s[j], 1e-6))
        out[i] = pts[j] * (1.0 - u) + pts[j + 1] * u
    return out


def _sdc_logged_path(scenario, max_points: int) -> np.ndarray:
    sdc = int(getattr(scenario, "sdc_track_index", 0))
    tracks = list(getattr(scenario, "tracks", []))
    pts: list[tuple[float, float]] = []
    if 0 <= sdc < len(tracks):
        for st in getattr(tracks[sdc], "states", []):
            if bool(getattr(st, "valid", False)):
                pts.append((float(getattr(st, "center_x", 0.0)), float(getattr(st, "center_y", 0.0))))
    return _dedupe_and_resample(pts, max_points)


def _nearest_lane_centerline(map_polylines: np.ndarray, map_valid: np.ndarray, anchor_xy: np.ndarray, max_points: int) -> np.ndarray:
    if not map_polylines.size:
        return np.zeros((0, 2), dtype=np.float32)
    best_p = -1
    best_d = float("inf")
    for p in range(map_polylines.shape[0]):
        valid = map_valid[p].astype(bool)
        if not valid.any():
            continue
        # Feature kind 1 is lane centerline in map_features.py.
        kind = map_polylines[p, valid, 5]
        if not np.any(np.isclose(kind, 1.0)):
            continue
        xy = map_polylines[p, valid, :2]
        d = float(np.min(np.linalg.norm(xy - anchor_xy[None, :], axis=-1)))
        if d < best_d:
            best_d = d
            best_p = p
    if best_p < 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = map_polylines[best_p, map_valid[best_p].astype(bool), :2]
    return _dedupe_and_resample(pts, max_points)


def _sdc_path_anchor(scenario) -> np.ndarray:
    sdc = int(getattr(scenario, "sdc_track_index", 0))
    tracks = list(getattr(scenario, "tracks", []))
    if 0 <= sdc < len(tracks):
        for st in getattr(tracks[sdc], "states", []):
            if bool(getattr(st, "valid", False)):
                return np.asarray([float(getattr(st, "center_x", 0.0)), float(getattr(st, "center_y", 0.0))], dtype=np.float32)
    return np.zeros(2, dtype=np.float32)


def parse_route(scenario, map_polylines: np.ndarray, map_valid: np.ndarray, max_points: int = 80) -> np.ndarray:
    """Build an SDC-relevant route polyline.

    WOMD protos available in different local installations do not always expose
    an explicit SDC route in the same field.  The previous implementation used
    the first lane polyline when sdc_paths was missing, which can be far from the
    SDC and caused every prefix to be marked off-route/wrong-way.  This function
    now prefers the official SDC path if present, otherwise uses the logged SDC
    trajectory as the planner route proxy, and only then falls back to the lane
    centerline nearest the SDC.
    """
    pts: list[tuple[float, float]] = []
    paths = list(getattr(scenario, "sdc_paths", [])) if hasattr(scenario, "sdc_paths") else []
    if paths:
        for p in getattr(paths[0], "points", []):
            pts.append((float(getattr(p, "x", 0.0)), float(getattr(p, "y", 0.0))))
    pts_arr = _dedupe_and_resample(pts, max_points) if pts else np.zeros((0, 2), dtype=np.float32)
    if len(pts_arr) < 2:
        pts_arr = _sdc_logged_path(scenario, max_points)
    if len(pts_arr) < 2:
        pts_arr = _nearest_lane_centerline(map_polylines, map_valid, _sdc_path_anchor(scenario), max_points)
    if len(pts_arr) < 2:
        pts_arr = np.stack([np.linspace(0, max_points - 1, max_points, dtype=np.float32), np.zeros(max_points, dtype=np.float32)], axis=-1)
    if len(pts_arr) < max_points:
        pad = np.repeat(pts_arr[-1:, :], max_points - len(pts_arr), axis=0)
        pts_arr = np.concatenate([pts_arr, pad], axis=0)
    else:
        pts_arr = pts_arr[:max_points]
    route = np.zeros((max_points, 6), dtype=np.float32)
    route[:, :2] = pts_arr[:, :2]
    if max_points > 1:
        d = np.diff(route[:, :2], axis=0, append=route[-1:, :2])
        route[:, 2] = np.arctan2(d[:, 1], d[:, 0])
    route[:, 3] = 13.4
    route[:, 5] = 1.0
    return route
