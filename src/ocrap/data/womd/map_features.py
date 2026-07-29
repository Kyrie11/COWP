from __future__ import annotations

import numpy as np


def _polyline_from_points(points) -> list[tuple[float, float, float]]:
    out = []
    for p in points:
        out.append((float(getattr(p, "x", getattr(p, "center_x", 0.0))), float(getattr(p, "y", getattr(p, "center_y", 0.0))), float(getattr(p, "z", 0.0))))
    return out


def _feature_kind_and_points(feature) -> tuple[int, list[tuple[float, float, float]], dict]:
    # Kind ids: 1 lane, 2 road_line, 3 road_edge, 4 crosswalk, 5 speed_bump, 6 stop_sign, 7 driveway.
    for name, kind in [("lane", 1), ("road_line", 2), ("road_edge", 3), ("crosswalk", 4), ("speed_bump", 5), ("stop_sign", 6), ("driveway", 7)]:
        if hasattr(feature, name):
            obj = getattr(feature, name)
            pts = []
            if hasattr(obj, "polyline"):
                pts = _polyline_from_points(getattr(obj, "polyline"))
            elif hasattr(obj, "polygon"):
                pts = _polyline_from_points(getattr(obj, "polygon"))
            elif hasattr(obj, "position"):
                pts = _polyline_from_points([getattr(obj, "position")])
            meta = {
                "speed_limit": float(getattr(obj, "speed_limit_mph", 0.0)) * 0.44704 if hasattr(obj, "speed_limit_mph") else 0.0,
                "interpolating": float(getattr(obj, "interpolating", False)),
                "entry_lanes": list(getattr(obj, "entry_lanes", [])) if hasattr(obj, "entry_lanes") else [],
                "exit_lanes": list(getattr(obj, "exit_lanes", [])) if hasattr(obj, "exit_lanes") else [],
            }
            return kind, pts, meta
    return 0, [], {}


def parse_map_features(scenario, max_polylines: int = 256, max_points: int = 64) -> tuple[np.ndarray, np.ndarray]:
    feats = list(getattr(scenario, "map_features", []))
    F = 10  # x,y,z,dx,dy,kind,speed_limit,route_flag,traffic_control,valid
    arr = np.zeros((max_polylines, max_points, F), dtype=np.float32)
    valid = np.zeros((max_polylines, max_points), dtype=bool)
    for p, feat in enumerate(feats[:max_polylines]):
        kind, pts, meta = _feature_kind_and_points(feat)
        if not pts:
            continue
        pts = pts[:max_points]
        xy = np.asarray([[x, y, z] for x, y, z in pts], dtype=np.float32)
        arr[p, : len(pts), :3] = xy
        if len(pts) > 1:
            d = np.diff(xy[:, :2], axis=0, append=xy[-1:, :2])
            arr[p, : len(pts), 3:5] = d
        arr[p, : len(pts), 5] = float(kind)
        arr[p, : len(pts), 6] = float(meta.get("speed_limit", 0.0))
        arr[p, : len(pts), 8] = 1.0 if kind == 1 and (meta.get("entry_lanes") or meta.get("exit_lanes")) else 0.0
        arr[p, : len(pts), 9] = 1.0
        valid[p, : len(pts)] = True
    return arr, valid
