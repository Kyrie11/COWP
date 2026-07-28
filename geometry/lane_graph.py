from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cowp.core.types import Lane, MapData
from cowp.geometry.boxes import normalize_angle
from cowp.geometry.map_projection import polyline_arc_length


@dataclass
class ConflictRegion:
    conflict_id: int
    conflict_type: str
    center_xy: np.ndarray
    radius: float
    involved_lane_ids: tuple[int, int]
    priority_rule_hint: str = "UNKNOWN"


def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return bool((c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]))


def segment_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray | None:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    if _ccw(a, c, d) == _ccw(b, c, d) or _ccw(a, b, c) == _ccw(a, b, d):
        return None
    r = b - a
    s = d - c
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(float(denom)) < 1e-8:
        return None
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denom
    return a + t * r




def _bbox_xy(xy: np.ndarray) -> tuple[float, float, float, float]:
    return (float(np.min(xy[:, 0])), float(np.min(xy[:, 1])), float(np.max(xy[:, 0])), float(np.max(xy[:, 1])))


def _bbox_might_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], margin: float) -> bool:
    return not (a[2] + margin < b[0] or b[2] + margin < a[0] or a[3] + margin < b[1] or b[3] + margin < a[1])


def _segment_bboxes(xy: np.ndarray) -> list[tuple[float, float, float, float]]:
    return [
        (
            float(min(xy[i, 0], xy[i + 1, 0])),
            float(min(xy[i, 1], xy[i + 1, 1])),
            float(max(xy[i, 0], xy[i + 1, 0])),
            float(max(xy[i, 1], xy[i + 1, 1])),
        )
        for i in range(max(0, len(xy) - 1))
    ]


def lane_heading_at(lane: Lane, s_query: float | None = None) -> float:
    xy = lane.xy
    if len(xy) < 2:
        return 0.0
    if s_query is None:
        j = 0
    else:
        arcs = polyline_arc_length(xy)
        j = int(np.clip(np.searchsorted(arcs, s_query) - 1, 0, len(xy) - 2))
    d = xy[j + 1] - xy[j]
    return float(np.arctan2(d[1], d[0]))


def build_conflict_regions(map_data: MapData, cfg: dict) -> list[ConflictRegion]:
    conflict_cfg = cfg.get("conflict", cfg)
    heading_thresh = np.deg2rad(float(conflict_cfg.get("lane_intersection_heading_threshold_deg", 30.0)))
    intersection_radius = float(conflict_cfg.get("intersection_radius_m", 5.0))
    merge_radius = float(conflict_cfg.get("merge_radius_m", 8.0))
    regions: list[ConflictRegion] = []
    lanes = list(map_data.lanes.values())
    lane_bboxes = {lane.lane_id: _bbox_xy(lane.xy) for lane in lanes if len(lane.xy) >= 2}
    lane_segment_bboxes = {lane.lane_id: _segment_bboxes(lane.xy) for lane in lanes if len(lane.xy) >= 2}
    seen: set[tuple[int, int, int, int]] = set()
    max_regions = int(cfg.get("limits", {}).get("max_conflict_regions", 64))
    bbox_margin = max(intersection_radius, merge_radius) + 2.0
    for i, lane_a in enumerate(lanes):
        xy_a = lane_a.xy
        if len(xy_a) < 2:
            continue
        for lane_b in lanes[i + 1 :]:
            xy_b = lane_b.xy
            if len(xy_b) < 2:
                continue
            # Topological merges: two lanes exit into the same successor or endpoints are close.
            common_exit = set(lane_a.exit_lanes).intersection(lane_b.exit_lanes)
            endpoint_dist = float(np.linalg.norm(xy_a[-1] - xy_b[-1]))
            if not common_exit and endpoint_dist >= 4.0:
                if not _bbox_might_overlap(lane_bboxes[lane_a.lane_id], lane_bboxes[lane_b.lane_id], bbox_margin):
                    continue
            if common_exit or endpoint_dist < 4.0:
                hdiff = abs(float(normalize_angle(lane_heading_at(lane_a, None) - lane_heading_at(lane_b, None))))
                center = 0.5 * (xy_a[-1] + xy_b[-1])
                regions.append(
                    ConflictRegion(
                        len(regions),
                        "MERGE",
                        center.astype(np.float32),
                        merge_radius,
                        (lane_a.lane_id, lane_b.lane_id),
                        "MAINLINE_OR_ARRIVAL",
                    )
                )
                if len(regions) >= max_regions:
                    return regions[:max_regions]
                continue
            seg_boxes_a = lane_segment_bboxes.get(lane_a.lane_id, [])
            seg_boxes_b = lane_segment_bboxes.get(lane_b.lane_id, [])
            for ia in range(len(xy_a) - 1):
                for ib in range(len(xy_b) - 1):
                    # Exact negative test: if segment bounding boxes do not
                    # overlap, two straight segments cannot intersect. This
                    # preserves the original conflict-region set while avoiding
                    # most segment_intersection calls on dense HD maps.
                    if seg_boxes_a and seg_boxes_b and not _bbox_might_overlap(seg_boxes_a[ia], seg_boxes_b[ib], 0.0):
                        continue
                    key = (lane_a.lane_id, lane_b.lane_id, ia, ib)
                    if key in seen:
                        continue
                    p = segment_intersection(xy_a[ia], xy_a[ia + 1], xy_b[ib], xy_b[ib + 1])
                    if p is None:
                        continue
                    ha = float(np.arctan2(*(xy_a[ia + 1] - xy_a[ia])[::-1]))
                    hb = float(np.arctan2(*(xy_b[ib + 1] - xy_b[ib])[::-1]))
                    hdiff = abs(float(normalize_angle(ha - hb)))
                    ctype = "LANE_INTERSECTION" if hdiff > heading_thresh else "SWEPT_OVERLAP"
                    regions.append(
                        ConflictRegion(
                            len(regions),
                            ctype,
                            p.astype(np.float32),
                            intersection_radius if ctype == "LANE_INTERSECTION" else merge_radius,
                            (lane_a.lane_id, lane_b.lane_id),
                            "TOPOLOGY_OR_ARRIVAL",
                        )
                    )
                    seen.add(key)
                    if len(regions) >= max_regions:
                        return regions[:max_regions]
    return regions[:max_regions]


def tta_to_region(traj: np.ndarray, region: ConflictRegion, radius: float | None = None, dt: float = 0.1) -> float:
    if len(traj) == 0:
        return float("inf")
    r = float(radius if radius is not None else region.radius)
    half_diag = 0.5 * np.sqrt(np.maximum(traj[:, 5], 0.1) ** 2 + np.maximum(traj[:, 6], 0.1) ** 2)
    dist = np.linalg.norm(traj[:, :2] - region.center_xy[None, :], axis=-1)
    hit = np.where(dist < r + half_diag)[0]
    if len(hit) == 0:
        return float("inf")
    return float(hit[0] * dt)


def closest_conflict_for_pair(traj_a: np.ndarray, traj_b: np.ndarray, regions: Iterable[ConflictRegion], dt: float = 0.1) -> tuple[ConflictRegion | None, float, float, float]:
    best = (None, float("inf"), float("inf"), float("inf"))
    for region in regions:
        ta = tta_to_region(traj_a, region, dt=dt)
        tb = tta_to_region(traj_b, region, dt=dt)
        if np.isinf(ta) and np.isinf(tb):
            continue
        delta = abs(ta - tb)
        score = delta + 0.01 * min(ta, tb)
        if score < best[3]:
            best = (region, ta, tb, score)
    return best
